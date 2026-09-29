// Host-staged two-GPU all-reduce for tensor parallelism without P2P.
//
// Port of the chunked-kernel path of llama.cpp ggml/src/ggml-cuda/allreduce.cu
// (PR #22299) to two processes and to CUDA-graph replay.  Each rank stages its
// partial through a pinned host buffer that both processes map (/dev/shm file,
// cudaHostRegister Portable|Mapped), signals arrival with a per-block flag in
// the same mapping, spins on the peer's flag inside the kernel, then sums.
//
// Contracts the caller must uphold (see REPORT.md for the arguments):
//   * Both ranks issue the same sequence of hsar_allreduce calls (same count,
//     dtype, nblocks, slot_bytes), and each rank's calls are totally ordered on
//     its GPU (one stream, or graph edges equivalent to one stream).  This is
//     what makes the per-block device counters agree without host involvement,
//     so a captured graph replays with fresh tokens.
//   * variant and nblocks are fixed for the life of the counters: they fix which
//     host bytes each block owns, which the two-slot reuse argument relies on.
//   * in and out need only element alignment.  They may alias in the classic
//     variant; in the pipelined variant they must not overlap.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <type_traits>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

constexpr int    kMaxBlocks  = 32;   // grid size, both variants
constexpr int    kMaxChunks  = 256;  // pipelined variant: flags per (slot, rank)
constexpr int    kSlots      = 2;
// One flag per cache line: the peer polls it over PCIe, the owner writes it.
constexpr size_t kFlagStride = 64;
constexpr size_t kBlockFlagBytes = (size_t)kSlots * 2 * kMaxBlocks * kFlagStride;  // 8 KiB
constexpr size_t kChunkFlagBytes = (size_t)kSlots * 2 * kMaxChunks * kFlagStride;  // 64 KiB
constexpr size_t kFlagBytes      = kBlockFlagBytes + kChunkFlagBytes;              // data starts here

#ifndef HSAR_PER_THREAD_FENCE
#define HSAR_PER_THREAD_FENCE 1
#endif

#ifndef HSAR_POLL_SLEEP_NS
#define HSAR_POLL_SLEEP_NS 100
#endif

__device__ __forceinline__ uint32_t ld_relaxed_sys(const uint32_t * p) {
    uint32_t v;
    asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ uint64_t ld_relaxed_sys(const uint64_t * p) {
    uint64_t v;
    asm volatile("ld.relaxed.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void st_release_sys(uint32_t * p, uint32_t v) {
    asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ void st_release_sys(uint64_t * p, uint64_t v) {
    asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}

__device__ __forceinline__ void fence_acq_rel_sys() {
    asm volatile("fence.acq_rel.sys;" ::: "memory");
}

__device__ __forceinline__ uint64_t globaltimer_ns() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

template <typename T> __device__ __forceinline__ float to_f(T x);
template <> __device__ __forceinline__ float to_f<float>(float x) { return x; }
template <> __device__ __forceinline__ float to_f<__half>(__half x) { return __half2float(x); }
template <> __device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }

template <typename T> __device__ __forceinline__ T from_f(float x);
template <> __device__ __forceinline__ float from_f<float>(float x) { return x; }
template <> __device__ __forceinline__ __half from_f<__half>(float x) { return __float2half_rn(x); }
template <> __device__ __forceinline__ __nv_bfloat16 from_f<__nv_bfloat16>(float x) { return __float2bfloat16_rn(x); }

__device__ __forceinline__ uint32_t * block_flag(uint8_t * shm, int slot, int rank, int block) {
    return reinterpret_cast<uint32_t *>(
        shm + (((size_t)slot * 2 + rank) * kMaxBlocks + block) * kFlagStride);
}

__device__ __forceinline__ uint64_t * chunk_flag(uint8_t * shm, int slot, int rank, int chunk) {
    return reinterpret_cast<uint64_t *>(
        shm + kBlockFlagBytes + (((size_t)slot * 2 + rank) * kMaxChunks + chunk) * kFlagStride);
}

// Per-block counter, same history on both ranks; every block of a launch draws the same
// value because every launch increments every block's counter once.
__device__ __forceinline__ uint64_t next_token(uint64_t * counters) {
    __shared__ uint64_t s_token;
    if (threadIdx.x == 0) {
        const uint64_t t = counters[blockIdx.x] + 1;
        counters[blockIdx.x] = t;
        s_token = t;
    }
    __syncthreads();
    return s_token;
}

// Stages vector i (E elements, zero-padded past count) of my partial.
template <typename T>
__device__ __forceinline__ void stage_vec(const T * in, uint4 * mine, int i, int count) {
    constexpr int E = 16 / sizeof(T);
    const int base = i * E;
    uint4 v;
    T * e = reinterpret_cast<T *>(&v);
    if (base + E <= count) {
#pragma unroll
        for (int k = 0; k < E; ++k) e[k] = in[base + k];
    } else {
#pragma unroll
        for (int k = 0; k < E; ++k) e[k] = base + k < count ? in[base + k] : from_f<T>(0.0f);
    }
    mine[i] = v;
}

// out = round(float(in) + float(peer)); a+b == b+a, so both ranks produce identical bits.
// .cv: never serve the peer's bytes from a line cached on an earlier token.
template <typename T>
__device__ __forceinline__ void sum_vec(const T * in, T * out, const uint4 * peer, int i, int count) {
    constexpr int E = 16 / sizeof(T);
    const int base = i * E;
    const uint4 v = __ldcv(&peer[i]);
    const T * e = reinterpret_cast<const T *>(&v);
    if (base + E <= count) {
#pragma unroll
        for (int k = 0; k < E; ++k) out[base + k] = from_f<T>(to_f(in[base + k]) + to_f(e[k]));
    } else {
#pragma unroll
        for (int k = 0; k < E; ++k) {
            if (base + k < count) out[base + k] = from_f<T>(to_f(in[base + k]) + to_f(e[k]));
        }
    }
}

// Thread 0 only: spin until *pf reaches token, then acquire.  The peer's flag holds an
// older token (not yet arrived) or token; it cannot pass token before we finish, because
// its next write of this flag needs our arrival at a later token.
template <typename F>
__device__ __forceinline__ void wait_flag(const F * pf, F token, int rank, uint64_t timeout_ns) {
    using S = typename std::conditional<sizeof(F) == 4, int32_t, int64_t>::type;
    const uint64_t t0 = timeout_ns ? globaltimer_ns() : 0;
    while ((S)(ld_relaxed_sys(pf) - token) < 0) {
#if HSAR_POLL_SLEEP_NS > 0
        __nanosleep(HSAR_POLL_SLEEP_NS);
#endif
        if (timeout_ns && globaltimer_ns() - t0 > timeout_ns) {
            printf("hsar: rank %d block %d token %llu: peer flag %llu after %llu ns; trapping\n",
                   rank, (int)blockIdx.x, (unsigned long long)token,
                   (unsigned long long)ld_relaxed_sys(pf), (unsigned long long)timeout_ns);
            __trap();
        }
    }
    fence_acq_rel_sys();  // acquire: orders the flag read before the data reads that follow
}

__device__ __forceinline__ uint8_t * slot_ptr(uint8_t * shm, size_t slot_bytes, int rank, int slot) {
    return shm + kFlagBytes + ((size_t)rank * kSlots + slot) * slot_bytes;
}

// Classic (llama.cpp) variant: every block stages, handshakes, then sums.
// Block b owns vectors i with (i mod gridDim*blockDim) in [b*blockDim, (b+1)*blockDim),
// on both ranks and for every count, so its handshake covers exactly the host bytes
// it and its peer block touch.  Every block signals every launch, so a 32-bit flag
// never lags its token by more than 2 and the signed compare survives wrap.
template <typename T>
__global__ void __launch_bounds__(1024) hsar_kernel(
        const T * in, T * out, int count, uint8_t * shm, size_t slot_bytes,
        uint64_t * counters, int rank, uint64_t timeout_ns) {
    constexpr int E = 16 / sizeof(T);
    const int tid  = threadIdx.x;
    const int bid  = blockIdx.x;
    const int gtid = bid * blockDim.x + tid;
    const int gnt  = gridDim.x * blockDim.x;

    const uint32_t token = (uint32_t)next_token(counters);
    const int      slot  = token & 1;
    uint4 * mine = reinterpret_cast<uint4 *>(slot_ptr(shm, slot_bytes, rank, slot));
    const uint4 * peer = reinterpret_cast<const uint4 *>(slot_ptr(shm, slot_bytes, 1 - rank, slot));
    const int nvec = (count + E - 1) / E;

    for (int i = gtid; i < nvec; i += gnt) stage_vec(in, mine, i, count);

    // Release: thread 0's st.release.sys is cumulative over the CTA's stores via bar.sync
    // (PTX memory model).  The per-thread fence is what llama.cpp ships and is proven on
    // this pair; HSAR_PER_THREAD_FENCE=0 drops it for an A/B.
#if HSAR_PER_THREAD_FENCE
    __threadfence_system();
#endif
    __syncthreads();
    if (tid == 0) {
        st_release_sys(block_flag(shm, slot, rank, bid), token);
        wait_flag(block_flag(shm, slot, 1 - rank, bid), token, rank, timeout_ns);
    }
    __syncthreads();

    for (int i = gtid; i < nvec; i += gnt) sum_vec(in, out, peer, i, count);
}

// Pipelined variant: blocks [0, nw) stage chunk after chunk and release a flag per chunk;
// blocks [nw, grid) wait for the peer's flag of a chunk and sum it.  The upstream stores
// of later chunks overlap the downstream loads of earlier ones (PCIe is full duplex).
// Chunk j is always bytes [j*chunk_bytes, (j+1)*chunk_bytes) of the slot, for every count.
// A small all-reduce leaves high chunk flags untouched for many launches, so flags and
// tokens are 64-bit: no wrap.  out must not overlap in: a summing block may write out[x]
// before the local staging block has read in[x].
template <typename T>
__global__ void __launch_bounds__(1024) hsar_pipe_kernel(
        const T * in, T * out, int count, uint8_t * shm, size_t slot_bytes,
        uint64_t * counters, int rank, uint64_t timeout_ns, int nw, int chunk_bytes) {
    constexpr int E = 16 / sizeof(T);
    const int tid = threadIdx.x;
    const int nt  = blockDim.x;

    const uint64_t token = next_token(counters);
    const int      slot  = token & 1;
    const int vpc  = chunk_bytes / 16;
    const int nvec = (count + E - 1) / E;
    const int nch  = (nvec + vpc - 1) / vpc;

    if ((int)blockIdx.x < nw) {
        uint4 * mine = reinterpret_cast<uint4 *>(slot_ptr(shm, slot_bytes, rank, slot));
        for (int j = blockIdx.x; j < nch; j += nw) {
            const int end = min(nvec, (j + 1) * vpc);
            for (int i = j * vpc + tid; i < end; i += nt) stage_vec(in, mine, i, count);
#if HSAR_PER_THREAD_FENCE
            __threadfence_system();
#endif
            __syncthreads();
            if (tid == 0) st_release_sys(chunk_flag(shm, slot, rank, j), token);
        }
    } else {
        const uint4 * peer =
            reinterpret_cast<const uint4 *>(slot_ptr(shm, slot_bytes, 1 - rank, slot));
        const int nr = gridDim.x - nw;
        for (int j = blockIdx.x - nw; j < nch; j += nr) {
            if (tid == 0) wait_flag(chunk_flag(shm, slot, 1 - rank, j), token, rank, timeout_ns);
            __syncthreads();
            const int end = min(nvec, (j + 1) * vpc);
            for (int i = j * vpc + tid; i < end; i += nt) sum_vec(in, out, peer, i, count);
        }
    }
}

template <typename T>
int launch(int variant, const void * in, void * out, long long count, int rank, void * shm,
           size_t slot_bytes, void * counters, int nblocks, int nthreads, uint64_t timeout_ns,
           int chunk_bytes, cudaStream_t stream) {
    const long long per_launch = (long long)(slot_bytes / sizeof(T));
    for (long long off = 0; off < count; off += per_launch) {
        const long long n = count - off < per_launch ? count - off : per_launch;
        const T * i = static_cast<const T *>(in) + off;
        T *       o = static_cast<T *>(out) + off;
        uint8_t * s = static_cast<uint8_t *>(shm);
        uint64_t * c = static_cast<uint64_t *>(counters);
        if (variant == 0) {
            hsar_kernel<T><<<nblocks, nthreads, 0, stream>>>(i, o, (int)n, s, slot_bytes, c, rank,
                                                             timeout_ns);
        } else {
            hsar_pipe_kernel<T><<<2 * nblocks, nthreads, 0, stream>>>(
                i, o, (int)n, s, slot_bytes, c, rank, timeout_ns, nblocks, chunk_bytes);
        }
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) return (int)e;
    }
    return 0;
}

}  // namespace

extern "C" {

int hsar_abi_version() { return 2; }

size_t hsar_flag_bytes() { return kFlagBytes; }

int hsar_max_blocks() { return kMaxBlocks; }

int hsar_max_chunks() { return kMaxChunks; }

const char * hsar_error_string(int err) {
    return err > 0 ? cudaGetErrorString((cudaError_t)err) : "os error (negative errno)";
}

// Returns 0 or -errno.  create: O_CREAT|O_EXCL + ftruncate (zero-filled).
int hsar_map_shm(const char * path, size_t bytes, int create, void ** host_ptr) {
    const int fd = create ? open(path, O_RDWR | O_CREAT | O_EXCL, 0600) : open(path, O_RDWR);
    if (fd < 0) return -errno;
    if (create && ftruncate(fd, (off_t)bytes) != 0) {
        const int e = errno;
        close(fd);
        return -e;
    }
    void * p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, 0);
    const int e = errno;
    close(fd);
    if (p == MAP_FAILED) return -e;
    *host_ptr = p;
    return 0;
}

int hsar_unmap_shm(void * host_ptr, size_t bytes) {
    return munmap(host_ptr, bytes) == 0 ? 0 : -errno;
}

int hsar_host_register(int device, void * host_ptr, size_t bytes, void ** dev_ptr) {
    cudaError_t e = cudaSetDevice(device);
    if (e != cudaSuccess) return (int)e;
    int can_map = 0;
    e = cudaDeviceGetAttribute(&can_map, cudaDevAttrCanMapHostMemory, device);
    if (e != cudaSuccess) return (int)e;
    if (!can_map) return (int)cudaErrorNotSupported;
    e = cudaHostRegister(host_ptr, bytes, cudaHostRegisterPortable | cudaHostRegisterMapped);
    if (e != cudaSuccess) return (int)e;
    e = cudaHostGetDevicePointer(dev_ptr, host_ptr, 0);
    if (e != cudaSuccess) {
        cudaHostUnregister(host_ptr);
        return (int)e;
    }
    return 0;
}

// Loads the kernels now: under lazy module loading a first launch inside a stream
// capture would load the module mid-capture.
int hsar_preload(int device) {
    cudaError_t e = cudaSetDevice(device);
    if (e != cudaSuccess) return (int)e;
    cudaFuncAttributes a;
    const void * fns[] = {
        (const void *)hsar_kernel<float>, (const void *)hsar_kernel<__half>,
        (const void *)hsar_kernel<__nv_bfloat16>, (const void *)hsar_pipe_kernel<float>,
        (const void *)hsar_pipe_kernel<__half>, (const void *)hsar_pipe_kernel<__nv_bfloat16>,
    };
    for (const void * f : fns) {
        if ((e = cudaFuncGetAttributes(&a, f)) != cudaSuccess) return (int)e;
    }
    return 0;
}

int hsar_host_unregister(int device, void * host_ptr) {
    cudaError_t e = cudaSetDevice(device);
    if (e != cudaSuccess) return (int)e;
    return (int)cudaHostUnregister(host_ptr);
}

// dtype: 0 = float32, 1 = float16, 2 = bfloat16.  variant: 0 = classic (nblocks blocks),
// 1 = pipelined (nblocks staging + nblocks summing blocks, chunk_bytes per flag).
int hsar_allreduce(int variant, int dtype, const void * in, void * out, long long count,
                   int device, int rank, void * shm_dev, size_t slot_bytes, void * counters,
                   int nblocks, int nthreads, int chunk_bytes, unsigned long long timeout_ns,
                   void * stream) {
    const int grid = variant == 0 ? nblocks : 2 * nblocks;
    if (rank < 0 || rank > 1 || variant < 0 || variant > 1 || nblocks < 1 || grid > kMaxBlocks ||
        nthreads < 32 || nthreads > 1024 || slot_bytes == 0 || slot_bytes % 16 != 0 || count < 0) {
        return (int)cudaErrorInvalidValue;
    }
    if (variant == 1 && (chunk_bytes < 16 || chunk_bytes % 16 != 0 ||
                         (slot_bytes + chunk_bytes - 1) / chunk_bytes > (size_t)kMaxChunks)) {
        return (int)cudaErrorInvalidValue;
    }
    if (count == 0) return 0;
    int cur = -1;
    cudaError_t e = cudaGetDevice(&cur);
    if (e != cudaSuccess) return (int)e;
    if (cur != device && (e = cudaSetDevice(device)) != cudaSuccess) return (int)e;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    switch (dtype) {
        case 0: return launch<float>(variant, in, out, count, rank, shm_dev, slot_bytes, counters, nblocks, nthreads, timeout_ns, chunk_bytes, s);
        case 1: return launch<__half>(variant, in, out, count, rank, shm_dev, slot_bytes, counters, nblocks, nthreads, timeout_ns, chunk_bytes, s);
        case 2: return launch<__nv_bfloat16>(variant, in, out, count, rank, shm_dev, slot_bytes, counters, nblocks, nthreads, timeout_ns, chunk_bytes, s);
        default: return (int)cudaErrorInvalidValue;
    }
}

}  // extern "C"
