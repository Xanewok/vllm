# SPDX-License-Identifier: Apache-2.0
"""syv patch: exact min_p + top-k + top-p for batches whose top_k is small.

The kept set is the one the full-vocab kernels define, computed on candidates:
order tokens by (logit desc, index asc); drop logit < max + log(min_p); keep the
first k; keep token j while the renormalised mass ranked before it is < p. The
index tie-break is the one _topk_topp_kernel applies to duplicates, so the output
matches it token for token up to fp32 rounding of the top-p mass (the same slack
the Triton and PyTorch paths already have between each other).

Every token outside a chunk's top-KK (KK >= k_max) is outside the global top-k,
so selecting per chunk and merging is exact. Output is written in place: kept
logits keep their value, everything else becomes -inf, the form the rejection
sampler and gumbel sampler read.
"""

import torch

from vllm.triton_utils import HAS_TRITON, tl, triton

FAST_TOPK_MAX_K = 256
_TILE = 4096
_MAX_CAND = 4096
_KEY_PAD = tl.constexpr(-(2**63)) if HAS_TRITON else -(2**63)


@triton.jit
def _encode(x, idx):
    # int64 key whose signed order is (value desc, index asc) under max-first topk.
    # -0.0 is folded to +0.0 so float-equal values tie on index only.
    x = tl.where(x == 0.0, 0.0, x)
    b = x.to(tl.int32, bitcast=True)
    b = b ^ ((b >> 31) & 0x7FFFFFFF)
    return (b.to(tl.int64) << 32) | (0x7FFFFFFF - idx).to(tl.int64)


@triton.jit
def _candidates_kernel(
    logits_ptr,
    logits_stride,
    cand_ptr,
    vocab_size,
    NUM_CHUNKS: tl.constexpr,
    TILES_PER_CHUNK: tl.constexpr,
    TILE: tl.constexpr,
    KK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1)
    row_ptr = logits_ptr + row * logits_stride
    best = tl.full([KK], _KEY_PAD, tl.int64)
    for t in tl.static_range(TILES_PER_CHUNK):
        offs = (chunk * TILES_PER_CHUNK + t) * TILE + tl.arange(0, TILE)
        mask = offs < vocab_size
        x = tl.load(row_ptr + offs, mask=mask, other=float("-inf"))
        key = tl.where(mask, _encode(x, offs), _KEY_PAD)
        # Safe: the select kernel reads values back from the keys, never from logits.
        tl.store(row_ptr + offs, tl.full([TILE], float("-inf"), tl.float32), mask=mask)
        top = tl.topk(key, KK)
        best = tl.topk(tl.reshape(tl.join(best, top), [2 * KK]), KK)
    offs_k = tl.arange(0, KK)
    tl.store(cand_ptr + (row * NUM_CHUNKS + chunk) * KK + offs_k, best)


@triton.jit
def _select_kernel(
    logits_ptr,
    logits_stride,
    cand_ptr,
    expanded_idx_mapping_ptr,
    top_k_ptr,
    top_p_ptr,
    min_p_ptr,
    NUM_CAND: tl.constexpr,
    CAND: tl.constexpr,
    KK: tl.constexpr,
    HAS_TOP_P: tl.constexpr,
    HAS_MIN_P: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, CAND)
    keys = tl.load(
        cand_ptr + row * NUM_CAND + offs, mask=offs < NUM_CAND, other=_KEY_PAD
    )
    top = tl.topk(keys, KK)
    hi = (top >> 32).to(tl.int32)
    bits = hi ^ ((hi >> 31) & 0x7FFFFFFF)
    val = bits.to(tl.float32, bitcast=True)
    idx = 0x7FFFFFFF - (top & 0x7FFFFFFF).to(tl.int32)
    rank = tl.arange(0, KK)

    state = tl.load(expanded_idx_mapping_ptr + row)
    k = tl.load(top_k_ptr + state)
    keep = (top != _KEY_PAD) & (rank < k) & (val > float("-inf"))

    if HAS_MIN_P:
        # Same expression as _min_p_kernel, so the threshold is bit-identical;
        # the row max is rank 0.
        min_p = tl.load(min_p_ptr + state).to(tl.float32)
        if min_p != 0.0:
            max_val = tl.max(tl.where(top != _KEY_PAD, val, float("-inf")))
            threshold = max_val + tl.log(min_p)
            keep = keep & ~(val < threshold)

    if HAS_TOP_P:
        p = tl.load(top_p_ptr + state)
        if p < 1.0:
            max_kept = tl.max(tl.where(keep, val, float("-inf")))
            e = tl.where(keep, tl.exp(val - max_kept), 0.0)
            prob = e / tl.sum(e)
            mass_before = tl.cumsum(prob) - prob
            keep = keep & ((mass_before < p) | (rank == 0))

    tl.store(logits_ptr + row * logits_stride + idx, val, mask=keep)


def _plan(vocab_size: int, k_max: int) -> tuple[int, int, int, int]:
    kk = max(16, triton.next_power_of_2(k_max))
    tiles = triton.cdiv(vocab_size, _TILE)
    tiles_per_chunk = triton.cdiv(tiles, _MAX_CAND // kk)
    num_chunks = triton.cdiv(tiles, tiles_per_chunk)
    return kk, tiles_per_chunk, num_chunks, triton.next_power_of_2(num_chunks * kk)


def fast_top_k_top_p(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor | None,
    min_p: torch.Tensor | None,
    k_max: int,
) -> torch.Tensor:
    """In place. top_k/top_p/min_p are per request state, indexed through
    expanded_idx_mapping. Caller guarantees 1 <= every row's top_k <= k_max <=
    FAST_TOPK_MAX_K."""
    assert logits.dtype == torch.float32 and logits.stride(-1) == 1
    assert 0 < k_max <= FAST_TOPK_MAX_K
    num_rows, vocab_size = logits.shape
    kk, tiles_per_chunk, num_chunks, cand = _plan(vocab_size, k_max)
    cand_buf = torch.empty(
        num_rows, num_chunks * kk, dtype=torch.int64, device=logits.device
    )
    _candidates_kernel[(num_rows, num_chunks)](
        logits,
        logits.stride(0),
        cand_buf,
        vocab_size,
        NUM_CHUNKS=num_chunks,
        TILES_PER_CHUNK=tiles_per_chunk,
        TILE=_TILE,
        KK=kk,
        num_warps=8,
    )
    _select_kernel[(num_rows,)](
        logits,
        logits.stride(0),
        cand_buf,
        expanded_idx_mapping,
        top_k,
        top_p if top_p is not None else top_k,
        min_p if min_p is not None else top_k,
        NUM_CAND=num_chunks * kk,
        CAND=cand,
        KK=kk,
        HAS_TOP_P=top_p is not None,
        HAS_MIN_P=min_p is not None,
        num_warps=4,
    )
    return logits
