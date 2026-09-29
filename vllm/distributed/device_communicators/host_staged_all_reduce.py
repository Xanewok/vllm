# SPDX-License-Identifier: Apache-2.0
"""Host-staged two-GPU all-reduce for TP2 without P2P (VLLM_HOST_STAGED_AR=1).

Both ranks map one /dev/shm file, register it with cudaHostRegister(Portable|
Mapped), and reduce through it inside one kernel that polls the peer's flag.
The kernel lives in host_staged_all_reduce.cu beside this file and is built on
first use with nvcc into a per-source-hash cache (or loaded from
VLLM_HOST_STAGED_AR_LIB).

Contract with the caller: both ranks call all_reduce() with the same sequence
of (numel, dtype) and each rank's calls are ordered on one stream.  should_use()
depends only on numel and dtype, so both ranks always take the same branch.
"""

from __future__ import annotations

import ctypes
import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

try:
    from vllm.logger import init_logger

    logger = init_logger(__name__)
except ImportError:  # standalone test
    logger = logging.getLogger(__name__)

_SRC = Path(__file__).with_suffix(".cu")
_ABI_VERSION = 2
_VARIANTS = {"classic": 0, "pipe": 1}
_DTYPES = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}
_lib: ctypes.CDLL | None = None


def _env(name: str, default: str) -> str:
    try:
        import vllm.envs as envs

        if hasattr(envs, name):
            return str(getattr(envs, name))
    except ImportError:
        pass
    return os.environ.get(name, default)


def _nvcc() -> str:
    for cand in (
        os.environ.get("VLLM_HOST_STAGED_AR_NVCC"),
        os.path.join(os.environ["CUDA_HOME"], "bin", "nvcc")
        if os.environ.get("CUDA_HOME")
        else None,
        shutil.which("nvcc"),
        "/usr/local/cuda/bin/nvcc",
    ):
        if cand and os.access(cand, os.X_OK):
            return cand
    raise RuntimeError("host-staged all-reduce: no nvcc (set CUDA_HOME)")


def _cache_dir() -> Path:
    root = os.environ.get("VLLM_CACHE_ROOT") or os.path.join(
        os.path.expanduser("~"), ".cache", "vllm"
    )
    return Path(root) / "host_staged_ar"


def build_library(device: torch.device, out_dir: Path | None = None) -> Path:
    """Compile the kernel for `device`'s arch; reuses a build of the same source."""
    major, minor = torch.cuda.get_device_capability(device)
    arch = f"{major}{minor}"
    src = _SRC.read_bytes()
    tag = hashlib.sha256(src).hexdigest()[:16]
    out_dir = out_dir or _cache_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    so = out_dir / f"libhsar_{tag}_sm{arch}.so"
    if so.exists():
        return so
    # Both TP workers may race here; each builds to its own temp name and the
    # rename is atomic, so the loser just replaces an identical file.
    fd, tmp = tempfile.mkstemp(suffix=".so", dir=out_dir)
    os.close(fd)
    nvcc = _nvcc()
    cmd = [
        nvcc, "-O3", "-std=c++17", "-shared", "-Xcompiler", "-fPIC",
        f"-gencode=arch=compute_{arch},code=sm_{arch}",
        str(_SRC), "-o", tmp,
    ]  # fmt: skip
    # pip's nvidia/cu13 toolchain keeps libcudart_static.a in lib/, which its nvcc.profile
    # does not search.
    root = Path(nvcc).resolve().parent.parent
    cmd += [f"-L{d}" for d in (root / "lib", root / "lib64") if d.is_dir()]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        os.unlink(tmp)
        raise RuntimeError(f"host-staged all-reduce build failed: {e.stderr}") from e
    os.replace(tmp, so)
    return so


def _load(device: torch.device) -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib
    path = os.environ.get("VLLM_HOST_STAGED_AR_LIB") or str(build_library(device))
    lib = ctypes.CDLL(path)
    lib.hsar_abi_version.restype = ctypes.c_int
    if lib.hsar_abi_version() != _ABI_VERSION:
        raise RuntimeError(f"{path}: ABI {lib.hsar_abi_version()} != {_ABI_VERSION}")
    lib.hsar_flag_bytes.restype = ctypes.c_size_t
    lib.hsar_max_blocks.restype = ctypes.c_int
    lib.hsar_max_chunks.restype = ctypes.c_int
    lib.hsar_error_string.restype = ctypes.c_char_p
    lib.hsar_error_string.argtypes = [ctypes.c_int]
    lib.hsar_map_shm.argtypes = [
        ctypes.c_char_p, ctypes.c_size_t, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p),
    ]  # fmt: skip
    lib.hsar_unmap_shm.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.hsar_host_register.argtypes = [
        ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p),
    ]  # fmt: skip
    lib.hsar_host_unregister.argtypes = [ctypes.c_int, ctypes.c_void_p]
    lib.hsar_preload.argtypes = [ctypes.c_int]
    lib.hsar_allreduce.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong,
        ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_ulonglong, ctypes.c_void_p,
    ]  # fmt: skip
    for f in (
        lib.hsar_map_shm, lib.hsar_unmap_shm, lib.hsar_host_register,
        lib.hsar_host_unregister, lib.hsar_allreduce, lib.hsar_preload,
    ):  # fmt: skip
        f.restype = ctypes.c_int
    _lib = lib
    return lib


def _check(lib: ctypes.CDLL, rc: int, what: str) -> None:
    if rc == 0:
        return
    if rc < 0:
        raise OSError(-rc, f"{what}: {os.strerror(-rc)}")
    raise RuntimeError(f"{what}: {lib.hsar_error_string(rc).decode()}")


def _default_stream() -> torch.cuda.Stream:
    try:
        from vllm.utils.torch_utils import current_stream

        return current_stream()
    except ImportError:
        return torch.cuda.current_stream()


class HostStagedAllReduce:
    """Two-rank all-reduce staged through shared pinned host memory."""

    def __init__(
        self,
        group: ProcessGroup,
        device: int | str | torch.device,
        max_bytes: int | None = None,
        blocks: int | None = None,
        threads: int = 256,
        timeout_s: float | None = None,
        check_same_node: bool = True,
        variant: str | None = None,
        chunk_bytes: int | None = None,
    ) -> None:
        self.disabled = True
        self.group = group
        self.device = torch.device(device)
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        self._host = ctypes.c_void_p()
        self._registered = False
        self.lib: ctypes.CDLL | None = None
        if self.world_size != 2:
            logger.warning(
                "host-staged all-reduce needs world_size 2, got %d", self.world_size
            )
            return
        if check_same_node:
            try:
                from vllm.distributed.parallel_state import in_the_same_node_as

                if not all(in_the_same_node_as(group, source_rank=0)):
                    logger.warning("host-staged all-reduce needs both ranks on one node")
                    return
            except ImportError:
                pass

        self.max_bytes = int(
            max_bytes or _env("VLLM_HOST_STAGED_AR_MAX_BYTES", str(256 * 1024))
        )
        self.blocks = int(blocks or _env("VLLM_HOST_STAGED_AR_BLOCKS", "8"))
        self.variant = variant or _env("VLLM_HOST_STAGED_AR_VARIANT", "classic")
        chunk = int(chunk_bytes or _env("VLLM_HOST_STAGED_AR_CHUNK_BYTES", "4096"))
        self.threads = threads
        timeout = (
            timeout_s
            if timeout_s is not None
            else float(_env("VLLM_HOST_STAGED_AR_TIMEOUT_S", "0"))
        )
        self.timeout_ns = int(timeout * 1e9)

        # Every rank must reach every collective below even when its own setup fails, or
        # the peer hangs; so errors are collected and the backend is disabled on both.
        err = ""
        lib = None
        try:
            lib = _load(self.device)
            if self.variant not in _VARIANTS:
                raise ValueError(f"variant must be one of {list(_VARIANTS)}")
            grid = self.blocks * (2 if self.variant == "pipe" else 1)
            if not 1 <= grid <= lib.hsar_max_blocks():
                raise ValueError(f"grid of {grid} blocks outside 1..{lib.hsar_max_blocks()}")
        except Exception as e:  # noqa: BLE001
            err = repr(e)
        self.lib = lib = None if err else lib
        # Slots are whole pages; the launcher chunks anything larger than one slot.
        self.slot_bytes = max(4096, (self.max_bytes + 4095) // 4096 * 4096)
        # The pipelined variant has a fixed number of chunk flags per slot.
        max_chunks = lib.hsar_max_chunks() if lib else 256
        self.chunk_bytes = max(chunk, -(-self.slot_bytes // max_chunks))
        self.chunk_bytes = (self.chunk_bytes + 15) // 16 * 16
        page = os.sysconf("SC_PAGE_SIZE")
        flag_bytes = lib.hsar_flag_bytes() if lib else 0
        self.shm_bytes = (flag_bytes + 4 * self.slot_bytes + page - 1) // page * page

        name = [f"/dev/shm/vllm_hsar_{os.getpid()}_{uuid.uuid4().hex[:12]}"]
        dist.broadcast_object_list(name, src=dist.get_global_rank(group, 0), group=group)
        self.shm_path = name[0]
        for creator in (True, False):
            if not err and (self.rank == 0) == creator:
                try:
                    what = "create" if creator else "open"
                    rc = lib.hsar_map_shm(self.shm_path.encode(), self.shm_bytes,
                                          int(creator), ctypes.byref(self._host))
                    _check(lib, rc, f"{what} {self.shm_path}")
                except Exception as e:  # noqa: BLE001
                    err = repr(e)
            dist.barrier(group=group)
        if self.rank == 0 and os.path.exists(self.shm_path):
            os.unlink(self.shm_path)  # both ranks mapped it or failed; nothing to leak
        shm_dev = ctypes.c_void_p()
        if not err:
            try:
                rc = lib.hsar_host_register(self.device.index, self._host,
                                            self.shm_bytes, ctypes.byref(shm_dev))
                _check(lib, rc, "cudaHostRegister")
                self._registered = True
                _check(lib, lib.hsar_preload(self.device.index), "load kernels")
            except Exception as e:  # noqa: BLE001
                err = repr(e)
        self.shm_dev = shm_dev.value
        # Device-resident so a replayed graph draws fresh tokens; one per block.
        self.counters = torch.zeros(32, dtype=torch.int64, device=self.device)
        torch.cuda.synchronize(self.device)
        errs: list[str | None] = [None, None]
        dist.all_gather_object(errs, err, group=group)
        if any(errs):
            logger.warning("host-staged all-reduce disabled: %s", errs)
            self.close()
            return
        self.disabled = False
        logger.info(
            "host-staged all-reduce: rank %d, %s, <= %d bytes, %d x %d threads, "
            "chunk %d bytes, shm %d bytes",
            self.rank, self.variant, self.max_bytes, self.blocks, self.threads,
            self.chunk_bytes, self.shm_bytes,
        )

    def should_use(self, inp: torch.Tensor) -> bool:
        # Must depend only on values identical on both ranks (not on data_ptr).
        return (
            not self.disabled
            and inp.dtype in _DTYPES
            and inp.is_cuda
            and inp.is_contiguous()
            and 0 < inp.numel() * inp.element_size() <= self.max_bytes
        )

    def all_reduce(
        self,
        inp: torch.Tensor,
        out: torch.Tensor | None = None,
        stream: torch.cuda.Stream | None = None,
    ) -> torch.Tensor:
        if out is None:
            out = torch.empty_like(inp)
        # The pipelined kernel may write out[x] before it reads in[x].
        dst = torch.empty_like(inp) if self.variant == "pipe" and out.data_ptr() == inp.data_ptr() else out
        stream = stream if stream is not None else _default_stream()
        rc = self.lib.hsar_allreduce(
            _VARIANTS[self.variant], _DTYPES[inp.dtype], inp.data_ptr(), dst.data_ptr(),
            inp.numel(), self.device.index, self.rank, self.shm_dev, self.slot_bytes,
            self.counters.data_ptr(), self.blocks, self.threads, self.chunk_bytes,
            self.timeout_ns, stream.cuda_stream,
        )  # fmt: skip
        _check(self.lib, rc, "hsar_allreduce")
        if dst is not out:
            out.copy_(dst)
        return out

    def close(self) -> None:
        self.disabled = True
        if self.lib is None:
            return
        if self._registered:
            torch.cuda.synchronize(self.device)
            self.lib.hsar_host_unregister(self.device.index, self._host)
            self._registered = False
        if self._host.value:
            self.lib.hsar_unmap_shm(self._host, self.shm_bytes)
            self._host = ctypes.c_void_p()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass
