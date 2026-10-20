"""Continuous sampling from a feature pyramid.

Coordinate convention: x, y are normalized to [0, 1], where 0 maps to the center
of the first pixel and 1 to the center of the last pixel of every pyramid level
(i.e. pixel coordinate = x * (W - 1), the align_corners=True convention).
z is normalized to [0, 1], where 0 is pyramid[0] and 1 is pyramid[-1].
"""
import torch
import torch.nn.functional as F


def _sample_all_levels(pyramid, xy, padding_mode):
    """Bilinearly sample every pyramid level at the same normalized (x, y) points.

    pyramid: list of L tensors (B, C, Hl, Wl)
    xy: (B, M, 2) in [0, 1]
    returns: (L, B, C, M)
    """
    B, M, _ = xy.shape
    grid = (xy * 2.0 - 1.0).view(B, 1, M, 2)
    sampled = [
        F.grid_sample(
            t, grid, mode="bilinear", padding_mode=padding_mode, align_corners=True
        ).squeeze(2)
        for t in pyramid
    ]
    return torch.stack(sampled, dim=0)


def sample_one_level(level, xy, padding_mode="zeros"):
    """Bilinearly sample a single feature map at normalized (x, y) points.

    The residual cross-scale branch pins z to the reference level, so routing it
    through `trilinear_pyramid_sampling` would grid_sample all L levels to read
    one of them -- an L-fold waste that measured ~1.9x per training step.

    level: (B, C, H, W); xy: (B, M, 2) in [0, 1]; returns (B, C, M)
    """
    B, M, _ = xy.shape
    grid = (xy * 2.0 - 1.0).view(B, 1, M, 2)
    return F.grid_sample(
        level, grid, mode="bilinear", padding_mode=padding_mode, align_corners=True
    ).squeeze(2)


def trilinear_pyramid_sampling(pyramid, grid, padding_mode="zeros"):
    """Trilinear sampling: bilinear in (x, y) at the two z-adjacent levels,
    then linear interpolation along z.

    pyramid: list of L tensors (B, C, Hl, Wl), all with the same B and C
    grid: (B, M, 3) with x, y in [0, 1] (values outside are handled by
        `padding_mode`) and z in [0, 1]
    returns: (B, C, M)
    """
    L = len(pyramid)
    B, M, _ = grid.shape
    C = pyramid[0].shape[1]

    z = grid[..., 2].clamp(0.0, 1.0) * (L - 1)  # (B, M)
    z0 = z.floor().long().clamp(0, L - 1)
    z1 = (z0 + 1).clamp(max=L - 1)

    samples = _sample_all_levels(pyramid, grid[..., :2], padding_mode)  # (L, B, C, M)

    idx0 = z0.view(1, B, 1, M).expand(1, B, C, M)
    idx1 = z1.view(1, B, 1, M).expand(1, B, C, M)
    s0 = torch.gather(samples, 0, idx0).squeeze(0)  # (B, C, M)
    s1 = torch.gather(samples, 0, idx1).squeeze(0)  # (B, C, M)

    zd = (z - z0.float()).unsqueeze(1)  # (B, 1, M)
    return s0 * (1.0 - zd) + s1 * zd


def attention_pyramid_sampling(pyramid, xy, level_logits, padding_mode="zeros"):
    """Soft-attention sampling: bilinear in (x, y) at every level, aggregated
    with softmax weights over levels. Unlike trilinear z-interpolation, every
    level receives gradient at every sample point.

    pyramid: list of L tensors (B, C, Hl, Wl)
    xy: (B, M, 2) in [0, 1]
    level_logits: (B, M, L) unnormalized level-attention scores
    returns: (B, C, M)
    """
    samples = _sample_all_levels(pyramid, xy, padding_mode)  # (L, B, C, M)
    weights = torch.softmax(level_logits, dim=-1)  # (B, M, L)
    weights = weights.permute(2, 0, 1).unsqueeze(2)  # (L, B, 1, M)
    return (samples * weights).sum(dim=0)
