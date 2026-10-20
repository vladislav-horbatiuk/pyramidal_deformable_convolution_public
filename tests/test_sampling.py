"""Correctness tests for pyramid sampling."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyrdc.sampling import trilinear_pyramid_sampling, attention_pyramid_sampling


def make_pyramid(B=2, C=3, sizes=((4, 8), (8, 16), (16, 32)), seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(B, C, h, w, generator=g) for h, w in sizes]


def naive_reference(pyramid, grid):
    """Straightforward loop implementation of trilinear pyramid sampling
    (align_corners=True convention, in-bounds coordinates only)."""
    L = len(pyramid)
    B, M, _ = grid.shape
    C = pyramid[0].shape[1]
    out = torch.zeros(B, C, M)

    def bilinear(t, x, y):
        H, W = t.shape[-2:]
        px, py = x * (W - 1), y * (H - 1)
        x0, y0 = int(px), int(py)
        x1, y1 = min(x0 + 1, W - 1), min(y0 + 1, H - 1)
        dx, dy = px - x0, py - y0
        top = t[:, y0, x0] * (1 - dx) + t[:, y0, x1] * dx
        bot = t[:, y1, x0] * (1 - dx) + t[:, y1, x1] * dx
        return top * (1 - dy) + bot * dy

    for b in range(B):
        for m in range(M):
            x, y, z = grid[b, m].tolist()
            zc = z * (L - 1)
            z0 = min(int(zc), L - 1)
            z1 = min(z0 + 1, L - 1)
            zd = zc - z0
            v0 = bilinear(pyramid[z0][b], x, y)
            v1 = bilinear(pyramid[z1][b], x, y)
            out[b, :, m] = v0 * (1 - zd) + v1 * zd
    return out


def test_matches_naive_reference():
    pyramid = make_pyramid()
    B, M = 2, 37
    g = torch.Generator().manual_seed(1)
    grid = torch.rand(B, M, 3, generator=g)
    # keep x, y strictly inside [0, 1] so padding semantics don't matter
    grid[..., :2] = grid[..., :2] * 0.9 + 0.05
    out = trilinear_pyramid_sampling(pyramid, grid)
    ref = naive_reference(pyramid, grid)
    assert torch.allclose(out, ref, atol=1e-5), (out - ref).abs().max()


@pytest.mark.parametrize("level", [0, 1, 2])
def test_integer_z_selects_single_level(level):
    """z exactly at a level with the identity xy grid recovers that level."""
    pyramid = make_pyramid()
    L = len(pyramid)
    t = pyramid[level]
    B, C, H, W = t.shape
    ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    grid = torch.zeros(B, H * W, 3)
    grid[..., 0] = (xs.flatten() / (W - 1))
    grid[..., 1] = (ys.flatten() / (H - 1))
    grid[..., 2] = level / (L - 1)
    out = trilinear_pyramid_sampling(pyramid, grid)
    assert torch.allclose(out, t.reshape(B, C, -1), atol=1e-5)


def test_attention_uniform_is_mean_peaked_is_single_level():
    B, C = 2, 3
    # equal sizes so the same xy hits the same content location on every level
    pyramid = make_pyramid(B, C, sizes=((8, 8), (8, 8), (8, 8)))
    M = 11
    g = torch.Generator().manual_seed(2)
    xy = torch.rand(B, M, 2, generator=g) * 0.9 + 0.05

    uniform = attention_pyramid_sampling(pyramid, xy, torch.zeros(B, M, 3))
    mean_ref = sum(
        attention_pyramid_sampling(
            pyramid, xy, torch.full((B, M, 3), -100.0).index_fill(-1, torch.tensor([l]), 100.0)
        )
        for l in range(3)
    ) / 3
    assert torch.allclose(uniform, mean_ref, atol=1e-4)

    peaked_logits = torch.full((B, M, 3), -100.0)
    peaked_logits[..., 1] = 100.0
    peaked = attention_pyramid_sampling(pyramid, xy, peaked_logits)
    only_l1 = trilinear_pyramid_sampling(
        [pyramid[1], pyramid[1]],
        torch.cat([xy, torch.zeros(B, M, 1)], dim=-1),
    )
    assert torch.allclose(peaked, only_l1, atol=1e-4)


def test_gradients_flow_to_grid_and_features():
    pyramid = [t.requires_grad_(True) for t in make_pyramid()]
    grid = (torch.rand(2, 20, 3) * 0.8 + 0.1).requires_grad_(True)
    out = trilinear_pyramid_sampling(pyramid, grid)
    out.sum().backward()
    assert grid.grad is not None and torch.isfinite(grid.grad).all()
    for t in pyramid:
        assert t.grad is not None and torch.isfinite(t.grad).all()

    xy = (torch.rand(2, 20, 2) * 0.8 + 0.1).requires_grad_(True)
    logits = torch.randn(2, 20, 3, requires_grad=True)
    pyramid2 = [t.detach().requires_grad_(True) for t in make_pyramid()]
    out = attention_pyramid_sampling(pyramid2, xy, logits)
    out.sum().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    # softmax attention: every level gets gradient, even far-from-z ones
    for t in pyramid2:
        assert t.grad is not None and t.grad.abs().sum() > 0
