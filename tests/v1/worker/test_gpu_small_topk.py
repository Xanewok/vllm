# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exactness of the Model Runner V2 small-k min_p/top-k/top-p path.

The kept set must equal `reference()` token for token, including on bf16-valued
logits (many exact ties) and when fewer than k logits are finite. Runs on CUDA,
or on CPU under TRITON_INTERPRET=1.
"""

import os

import pytest
import torch

pytest.importorskip("triton")
if not (torch.cuda.is_available() or os.environ.get("TRITON_INTERPRET") == "1"):
    pytest.skip("CUDA or TRITON_INTERPRET=1 required", allow_module_level=True)

from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch
from vllm.v1.worker.gpu.sample.small_topk import small_top_k_top_p

DEVICE = "cpu" if os.environ.get("TRITON_INTERPRET") == "1" else "cuda"


def reference(logits, k, p, min_p):
    """(value desc, index asc) order; min_p; first k; keep while mass before < p."""
    logits = logits.cpu()
    out = torch.full_like(logits, float("-inf"))
    for r in range(logits.shape[0]):
        v, order = torch.sort(logits[r], descending=True, stable=True)
        rank = torch.arange(v.numel())
        keep = v > float("-inf")
        if min_p is not None and float(min_p[r]) != 0.0:
            keep &= ~(v < v[0] + torch.log(min_p[r].cpu().to(torch.float32)))
        keep &= rank < int(k[r])
        if p is not None and float(p[r]) < 1.0:
            e = torch.where(keep, torch.exp(v - v[keep].max()), torch.zeros_like(v))
            prob = e / e.sum()
            before = torch.cumsum(prob, 0) - prob
            keep &= (before < float(p[r])) | (rank == 0)
        out[r, order[keep]] = v[keep]
    return out


def make_logits(num_rows, vocab_size, *, bf16, seed, num_neg_inf=0):
    g = torch.Generator().manual_seed(seed)
    # A broad body plus a peaked head, so min_p and top-p cut inside the top-k.
    x = torch.randn(num_rows, vocab_size, generator=g) * 2.0
    head = torch.randint(0, vocab_size, (num_rows, 40), generator=g)
    x.scatter_(1, head, 8.0 + 3.0 * torch.rand(num_rows, 40, generator=g))
    if num_neg_inf:
        x[:, torch.randint(0, vocab_size, (num_neg_inf,), generator=g)] = float("-inf")
    if bf16:
        x = x.to(torch.bfloat16).to(torch.float32)
    return x


def run(x, k, p, min_p):
    num_rows = x.shape[0]
    to = lambda t: None if t is None else t.to(DEVICE)  # noqa: E731
    out = small_top_k_top_p(
        x.clone().to(DEVICE),
        torch.arange(num_rows, dtype=torch.int64, device=DEVICE),
        k.to(torch.int32).to(DEVICE),
        to(p),
        to(min_p),
        int(k.max()),
    )
    return out.cpu()


@pytest.mark.parametrize(
    "num_rows,vocab_size,ks,p,min_p,bf16",
    [
        (4, 40000, [20, 20, 20, 20], 0.95, 0.05, True),
        (4, 40000, [20, 20, 20, 20], 0.95, None, True),
        (3, 40000, [20, 5, 1], None, None, True),
        (3, 40000, [64, 40, 33], 0.8, 0.02, True),
        (2, 70000, [256, 200], 0.9, None, True),
        (2, 70000, [128, 100], 0.99, 0.1, False),
        (4, 12345, [20, 20, 16, 7], None, 0.05, True),
    ],
)
@pytest.mark.parametrize("seed", [0, 1])
def test_matches_reference(num_rows, vocab_size, ks, p, min_p, bf16, seed):
    x = make_logits(num_rows, vocab_size, bf16=bf16, seed=seed, num_neg_inf=50)
    k = torch.tensor(ks)
    pp = None if p is None else torch.full((num_rows,), p)
    mp = None if min_p is None else torch.full((num_rows,), min_p)
    assert torch.equal(run(x, k, pp, mp), reference(x, k, pp, mp))


def test_ties_lowest_index_wins():
    x = torch.zeros(2, 9000)
    x[:, ::7] = 1.0  # ~1286 tied maxima per row
    got = run(x, torch.tensor([20, 3]), None, None)
    for r, k in enumerate([20, 3]):
        kept = torch.nonzero(got[r] > float("-inf")).flatten()
        assert kept.tolist() == list(range(0, 7 * k, 7))


def test_fewer_finite_than_k():
    x = torch.full((1, 5000), float("-inf"))
    x[0, [3, 77, 4000]] = torch.tensor([1.0, 2.0, 0.5])
    k, p, mp = torch.tensor([20]), torch.tensor([0.95]), torch.tensor([0.05])
    assert torch.equal(run(x, k, p, mp), reference(x, k, p, mp))


def test_forced_huge_logit():
    # Thinking-budget forcing writes 1e9 into one logit.
    x = make_logits(1, 30000, bf16=True, seed=3)
    x[0, 1234] = 1e9
    got = run(x, torch.tensor([20]), torch.tensor([0.95]), None)
    assert torch.nonzero(got[0] > float("-inf")).flatten().tolist() == [1234]


def test_reference_matches_pytorch_on_tie_free_logits():
    for seed in range(10):
        x = make_logits(4, 30000, bf16=False, seed=seed)
        k = torch.tensor([20, 20, 7, 50])
        p = torch.tensor([0.95, 0.5, 0.9, 0.99])
        mp = torch.tensor([0.05, 0.0, 0.2, 0.01])
        thr = x.max(dim=-1, keepdim=True).values + torch.log(mp).unsqueeze(1)
        y = x.masked_fill(x < thr, float("-inf"))
        want = apply_top_k_top_p_pytorch(y, k, p)
        assert torch.equal(reference(x, k, p, mp), want), seed
