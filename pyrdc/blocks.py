"""Pyramidal deformable convolution blocks and matched baselines.

All blocks share the same interface: forward(pyramid) where pyramid is a list of
L tensors (B, C, Hl, Wl) ordered coarse -> fine, and the output is computed at
the resolution of pyramid[-1]. Baselines that only use one level take pyramid[-1].
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

from .sampling import (
    trilinear_pyramid_sampling, attention_pyramid_sampling, sample_one_level,
)


def _kernel_offsets(kernel_size):
    """Relative (x, y) pixel offsets of the K*K kernel sample points, shape (2, K2)."""
    r = torch.arange(kernel_size) - (kernel_size - 1) // 2
    oy, ox = torch.meshgrid(r, r, indexing="ij")
    return torch.stack((ox, oy), dim=0).reshape(2, -1).float()


def _pixel_grid(h, w):
    """Absolute (x, y) pixel coordinates of every output location, shape (2, H, W)."""
    ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    return torch.stack((xs, ys), dim=0).float()


def _stride_at_z(pyramid, z, ref_h):
    """Reference-pixels-per-level-pixel factor at continuous depth z.

    z: (B, zch, H, W) in [0, 1], 0 = pyramid[0], 1 = pyramid[-1] (reference).
    Returns the per-sample stride, linearly interpolated between the two
    z-adjacent levels. Uses heights only: every pyramid in this project scales
    H and W by the same factor per level.
    """
    strides = torch.tensor(
        [(ref_h - 1) / max(t.shape[-2] - 1, 1) for t in pyramid],
        device=z.device, dtype=z.dtype,
    )
    zc = z.clamp(0.0, 1.0) * (len(pyramid) - 1)
    lo = zc.floor().long().clamp(max=max(len(pyramid) - 2, 0))
    frac = zc - lo
    return strides[lo] * (1.0 - frac) + strides[lo + 1] * frac


def _group_norm_groups(channels, preferred=4):
    """Largest group count <= `preferred` that divides `channels`.

    GroupNorm requires channels % groups == 0, so a plain min(4, C) crashes on
    any width that is not a multiple of 4 — which is exactly what the
    capacity-matching sweeps use (ch 17, 19, 66, ...). Widths that already
    worked (1, 2, 3, 4, 8, 16, 64, ...) keep the group count they had."""
    return next(g for g in range(min(preferred, channels), 0, -1)
                if channels % g == 0)


class _GridCache:
    """Caches sampling-grid constants per (H, W, device), so blocks work with
    changing input sizes and batch sizes (unlike the original notebook code,
    which froze them on the first forward pass)."""

    def __init__(self):
        self._store = {}

    def get(self, kernel_size, h, w, device):
        key = (kernel_size, h, w, str(device))
        if key not in self._store:
            p_n = _kernel_offsets(kernel_size).view(1, 2, -1, 1, 1)
            p_0 = _pixel_grid(h, w).view(1, 2, 1, h, w)
            base = (p_0 + p_n).to(device)  # (1, 2, K2, H, W), pixel units
            scale = torch.tensor(
                [1.0 / max(w - 1, 1), 1.0 / max(h - 1, 1)], device=device
            ).view(1, 2, 1, 1, 1)
            norm_p0 = (p_0.squeeze(2).to(device) * scale.squeeze(2))  # (1, 2, H, W)
            self._store[key] = (base, scale, norm_p0)
        return self._store[key]


class PyramidalDeformConv(nn.Module):
    """Deformable convolution that samples its K*K points from a feature pyramid.

    For every output location and kernel position the block predicts a spatial
    offset (dx, dy) plus a depth coordinate. Depth is parameterized either as
    a continuous z in [0, 1] resolved with trilinear interpolation
    (z_mode='trilinear') or as softmax attention weights over all L levels
    (z_mode='attention'). Optionally a DCNv2-style modulation mask is predicted
    per sample point.

    Args:
        in_channels / out_channels: channels of every pyramid level / output.
        kernel_size: convolution kernel size.
        num_levels: pyramid length L. Required for z_mode='attention'
            (the number of predicted logits depends on it); optional otherwise.
        z_mode: 'trilinear' or 'attention'.
        z_init: initial depth preference: 'first', 'middle', 'last' or 'uniform'.
        use_mask: predict a sigmoid modulation mask per sample point (DCNv2).
        tanh_spatial_offsets: bound spatial offsets to +-1 px via tanh.
        coords_as_extra_input: append normalized (x, y) coordinate maps to the
            offset-branch input (CoordConv-style).
        padding_mode: grid_sample padding for out-of-bounds samples. 'zeros'
            matches standard conv zero-padding semantics.
        freeze_z: detach the z / attention-logit branch from the graph, so the
            depth stays at its z_init value (a fixed scale mixture) while
            spatial offsets keep learning. Ablation that isolates *learned*
            scale selection from mere fixed multi-scale access.
        z_per_kernel_point: if False, one z (or one attention distribution) is
            predicted per output location and shared by all K*K kernel points
            ("cheap z": ~K2x fewer depth parameters, keeps pyramid access
            nearly parameter-free).
        use_level_norm: apply a per-level GroupNorm (with affine) to every
            pyramid level before sampling. Levels of a feature pyramid have
            different activation statistics; normalizing them lets the shared
            kernel see a consistent distribution regardless of which level a
            sample came from (the affine doubles as a learned per-level
            scale/bias, i.e. a level embedding). ~2*C params per level.
        z_stretch: eps >= 0. Map the depth as clamp((1+2*eps)*sigmoid(z) - eps,
            0, 1) instead of sigmoid(z) (stretch-and-clip, as in hard-concrete
            gates). With eps > 0 the first and last pyramid levels are reached
            exactly at a finite z, where the sigmoid still has a usable
            gradient; eps = 0 is the plain sigmoid. Trilinear z only.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, num_levels=None,
                 z_mode="trilinear", z_init="last", use_mask=False,
                 tanh_spatial_offsets=False, coords_as_extra_input=False,
                 padding_mode="zeros", bias=True, freeze_z=False,
                 z_per_kernel_point=True, use_level_norm=False,
                 residual_cross_scale=False, z_weight_init_std=0.0,
                 offset_mode="free", scale_offsets_by_level=False, z_stretch=0.0):
        super().__init__()
        if z_mode not in ("trilinear", "attention"):
            raise ValueError(f"unknown z_mode: {z_mode}")
        if offset_mode not in ("free", "affine", "similarity"):
            raise ValueError(f"unknown offset_mode: {offset_mode}")
        if z_mode == "attention" and num_levels is None:
            raise ValueError("z_mode='attention' requires num_levels")
        if scale_offsets_by_level and (z_mode != "trilinear" or num_levels == 1):
            # Attention mixes all L levels at one (x, y), so per-level offset
            # scaling would need L separate grids -- not implemented.
            raise ValueError(
                "scale_offsets_by_level requires z_mode='trilinear' and a real pyramid")
        if z_stretch < 0 or (z_stretch > 0 and z_mode != "trilinear"):
            raise ValueError("z_stretch must be >= 0 and needs z_mode='trilinear'")
        self.kernel_size = kernel_size
        self.K2 = kernel_size ** 2
        self.num_levels = num_levels
        self.z_mode = z_mode
        self.z_stretch = z_stretch
        self.use_mask = use_mask
        self.tanh_spatial_offsets = tanh_spatial_offsets
        self.coords_as_extra_input = coords_as_extra_input
        self.padding_mode = padding_mode
        self.freeze_z = freeze_z
        self.z_per_kernel_point = z_per_kernel_point
        self.residual_cross_scale = residual_cross_scale
        self.scale_offsets_by_level = scale_offsets_by_level

        if num_levels == 1:
            # single-level ablation: depth has no meaning, predict no z at all
            self.num_z_channels = 0
        elif z_per_kernel_point:
            self.num_z_channels = self.K2 if z_mode == "trilinear" else self.K2 * num_levels
        else:
            self.num_z_channels = 1 if z_mode == "trilinear" else num_levels
        # How many numbers describe the K^2 spatial offsets at one location.
        #   free       2*K^2  -- every kernel point moves independently
        #   affine     6      -- Delta_k = t + A p_k  (translation + 2x2 map)
        #   similarity 4      -- A restricted to scale*rotation
        # A decomposition of trained 'free' checkpoints
        # showed translation alone explains only 15-27% of what the block does,
        # while translation+affine explains 54-70%, so these are the rungs
        # between the reduced block (translation only) and free offsets.
        self.offset_mode = offset_mode
        self.n_xy = {"free": 2 * self.K2, "affine": 6, "similarity": 4}[offset_mode]
        n_off = self.n_xy + self.num_z_channels + (self.K2 if use_mask else 0)
        in_ch = in_channels + (2 if coords_as_extra_input else 0)
        self.offset_conv = nn.Conv2d(in_ch, n_off, kernel_size=3, padding=1)
        self._init_offset_conv(z_init)
        if z_weight_init_std > 0 and self.num_z_channels > 0:
            # The default zero-init makes the depth branch a constant function
            # at step 0, so it has to break symmetry through gradients alone --
            # a candidate explanation for the 3.4x seed variance on CIFAR-10.
            nn.init.normal_(
                self.offset_conv.weight.data[
                    self.n_xy: self.n_xy + self.num_z_channels],
                std=z_weight_init_std,
            )

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, self.K2))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        if residual_cross_scale:
            if num_levels is None or num_levels < 2:
                raise ValueError("residual_cross_scale requires num_levels >= 2")
            # out = W * (reference + g (*) cross_scale), g initialised to 0, one
            # gate per *input* channel. The block therefore starts as an exact
            # DCN and can only add to it, instead of having to trade the
            # reference level away to buy context -- which is what the level-mass
            # analysis shows it doing (trilinear+LN puts
            # ~0.01 of its sampling weight on the reference level).
            # Gating the inputs rather than the outputs keeps this to a single
            # einsum; an output-channel gate would need two, which measured ~2x
            # slower per step on CPU for no expressiveness we need here.
            self.cross_gate = nn.Parameter(torch.zeros(in_channels))
        else:
            self.cross_gate = None

        if use_level_norm:
            if num_levels is None:
                raise ValueError("use_level_norm requires num_levels")
            groups = _group_norm_groups(in_channels)
            self.level_norms = nn.ModuleList(
                nn.GroupNorm(groups, in_channels) for _ in range(num_levels)
            )
        else:
            self.level_norms = None

        self._grids = _GridCache()

    def _init_offset_conv(self, z_init):
        # DCN-style: zero offsets at init, depth bias set by z_init.
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        if self.num_z_channels == 0:
            if self.use_mask:
                self.offset_conv.bias.data[-self.K2:] += 2.0
            return
        z_bias = self.offset_conv.bias.data[self.n_xy: self.n_xy + self.num_z_channels]
        if self.z_mode == "trilinear":
            # sigmoid(+-6.9) ~ 0.999 / 0.001; 0.0 -> middle
            if z_init == "last":
                z_bias += 6.9
            elif z_init == "first":
                z_bias -= 6.9
            elif z_init == "uniform":
                # spread initial z across (0, 1) by giving the z branch
                # non-negligible random weights
                nn.init.normal_(
                    self.offset_conv.weight.data[self.n_xy: self.n_xy + self.num_z_channels],
                    std=1.0,
                )
        else:
            L = self.num_levels
            per_point = z_bias.view(-1, L)
            if z_init == "last":
                per_point[:, -1] += 4.0
            elif z_init == "first":
                per_point[:, 0] += 4.0
            elif z_init == "middle":
                per_point[:, (L - 1) // 2] += 4.0
            # 'uniform': zero logits == uniform attention
        if self.use_mask:
            # sigmoid(2.0) ~ 0.88 so the block starts close to an unmasked conv
            self.offset_conv.bias.data[-self.K2:] += 2.0

    def _spatial_offsets(self, raw, B, H, W):
        """Turn the predicted spatial numbers into (B, 2, K2, H, W) offsets.

        'free' is the identity reshape. The structured modes generate all K^2
        offsets from a per-location linear map applied to the base kernel grid:
        Delta_k = t + A p_k, so the sampled pattern stays a linearly warped
        version of the kernel (dilation, rotation, shear, anisotropic scale)
        rather than an arbitrary point cloud. All modes are zero at
        initialisation, so the block still reduces exactly to a plain
        convolution before training.
        """
        if self.offset_mode == "free":
            return raw.view(B, 2, self.K2, H, W)

        p_n = _kernel_offsets(self.kernel_size).to(raw.device)  # (2, K2)
        t = raw[:, :2].unsqueeze(2)                             # (B, 2, 1, H, W)
        if self.offset_mode == "affine":
            A = raw[:, 2:6].view(B, 2, 2, H, W)
        else:  # similarity: A = s*R(theta) - I, so zero raw gives the identity
            log_s, theta = raw[:, 2], raw[:, 3]
            s, cos, sin = log_s.exp(), theta.cos(), theta.sin()
            A = torch.stack((s * cos, -s * sin, s * sin, s * cos), dim=1)
            A = A.view(B, 2, 2, H, W) - torch.eye(2, device=raw.device).view(1, 2, 2, 1, 1)
        return t + torch.einsum("bijhw,jk->bikhw", A, p_n)

    def depth(self, z_raw):
        """Raw depth prediction -> normalized depth in [0, 1] (1 = reference)."""
        z = torch.sigmoid(z_raw)
        if self.z_stretch:
            eps = self.z_stretch
            z = ((1 + 2 * eps) * z - eps).clamp(0.0, 1.0)
        return z

    def forward(self, pyramid):
        x = pyramid[-1]
        B, C, H, W = x.shape
        L = len(pyramid)
        if self.num_levels is not None and L != self.num_levels:
            raise ValueError(f"expected pyramid of {self.num_levels} levels, got {L}")
        base, scale, norm_p0 = self._grids.get(self.kernel_size, H, W, x.device)

        raw_pyramid = pyramid
        if self.level_norms is not None:
            # offsets are predicted from the raw reference level (x, above);
            # sampling sees the normalized pyramid
            pyramid = [n(t) for n, t in zip(self.level_norms, pyramid)]

        inp = x
        if self.coords_as_extra_input:
            inp = torch.cat((x, norm_p0.expand(B, -1, -1, -1)), dim=1)
        raw = self.offset_conv(inp)  # (B, n_off, H, W)

        z_raw = raw[:, self.n_xy: self.n_xy + self.num_z_channels]
        if self.freeze_z:
            z_raw = z_raw.detach()

        xy_off = self._spatial_offsets(raw[:, : self.n_xy], B, H, W)
        if self.tanh_spatial_offsets:
            xy_off = torch.tanh(xy_off)
        if self.scale_offsets_by_level and self.num_z_channels > 0:
            # Offsets are predicted in reference-level pixels; a sample taken
            # at a 4x coarser level should get 4x the reach for the same
            # predicted number (constant reach in *sampled-level* pixels).
            # stride(z) interpolates linearly between adjacent levels, matching
            # the trilinear sampling weights; at z = reference it is exactly 1,
            # so DCN-equivalent behaviour is unchanged there.
            stride = _stride_at_z(pyramid, self.depth(z_raw), H)
            xy_off = xy_off * stride.unsqueeze(1)  # (B,1,{1|K2},H,W) bcast
        xy = (base + xy_off) * scale  # normalized to [0, 1] on pyramid[-1]
        xy = xy.reshape(B, 2, -1).permute(0, 2, 1)  # (B, M, 2), M = K2*H*W
        if self.num_z_channels == 0:
            grid = torch.cat((xy, torch.zeros_like(xy[..., :1])), dim=-1)
            sampled = trilinear_pyramid_sampling(pyramid, grid, self.padding_mode)
        elif self.z_mode == "trilinear":
            if not self.z_per_kernel_point:
                z_raw = z_raw.expand(-1, self.K2, -1, -1)
            z = self.depth(z_raw).reshape(B, 1, -1).permute(0, 2, 1)  # (B, M, 1)
            grid = torch.cat((xy, z), dim=-1)
            sampled = trilinear_pyramid_sampling(pyramid, grid, self.padding_mode)
        else:
            if self.z_per_kernel_point:
                logits = z_raw.view(B, self.K2, L, H, W)
            else:
                logits = z_raw.view(B, 1, L, H, W).expand(B, self.K2, L, H, W)
            logits = logits.permute(0, 1, 3, 4, 2).reshape(B, -1, L)  # (B, M, L)
            sampled = attention_pyramid_sampling(pyramid, xy, logits, self.padding_mode)

        sampled = sampled.view(B, C, self.K2, H, W)
        if self.use_mask:
            mask = torch.sigmoid(raw[:, -self.K2:])  # (B, K2, H, W)
            sampled = sampled * mask.unsqueeze(1)

        if self.cross_gate is not None:
            # Reference-level branch: same spatial offsets, depth pinned to the
            # reference level, i.e. exactly what the DCN control computes.
            # Deliberately reads the *un-normalized* pyramid: level-norm exists
            # to make levels commensurable for cross-scale mixing, but the
            # residual branch should reproduce the DCN baseline we know works,
            # not a normalized variant of it. z==1 means exactly pyramid[-1],
            # so this is one grid_sample, not a full L-level trilinear pass.
            ref = sample_one_level(raw_pyramid[-1], xy, self.padding_mode)
            ref = ref.view(B, C, self.K2, H, W)
            if self.use_mask:
                ref = ref * mask.unsqueeze(1)
            sampled = ref + self.cross_gate.view(1, -1, 1, 1, 1) * sampled
        out = torch.einsum("bikhw,oik->bohw", sampled, self.weight)
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out


class ReducedPyramidalDeformConv(nn.Module):
    """Cheaper variant: one deformed pyramid sample per output location (instead
    of per kernel position), followed by a regular 3x3 convolution."""

    def __init__(self, in_channels, out_channels, num_levels=None,
                 z_mode="trilinear", z_init="middle", use_mask=False,
                 tanh_spatial_offsets=True, coords_as_extra_input=False,
                 padding_mode="zeros", freeze_z=False, use_level_norm=False):
        super().__init__()
        if z_mode == "attention" and num_levels is None:
            raise ValueError("z_mode='attention' requires num_levels")
        self.num_levels = num_levels
        self.z_mode = z_mode
        self.use_mask = use_mask
        self.tanh_spatial_offsets = tanh_spatial_offsets
        self.coords_as_extra_input = coords_as_extra_input
        self.padding_mode = padding_mode
        self.freeze_z = freeze_z

        # Same fix as PyramidalDeformConv: without it the level with the largest
        # activations dominates the weighted sum and the depth branch collapses
        # back onto the reference level. The reduced block was only ever
        # evaluated without this, i.e. in the regime that invalidated the full
        # block's iteration-2/3 results.
        if use_level_norm:
            if num_levels is None:
                raise ValueError("use_level_norm requires num_levels")
            groups = _group_norm_groups(in_channels)
            self.level_norms = nn.ModuleList(
                nn.GroupNorm(groups, in_channels) for _ in range(num_levels)
            )
        else:
            self.level_norms = None

        if num_levels == 1:
            self.num_z_channels = 0
        else:
            self.num_z_channels = 1 if z_mode == "trilinear" else num_levels
        n_off = 2 + self.num_z_channels + (1 if use_mask else 0)
        in_ch = in_channels + (2 if coords_as_extra_input else 0)
        self.offset_conv = nn.Conv2d(in_ch, n_off, kernel_size=3, padding=1)
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        if self.num_z_channels > 0 and z_mode == "trilinear" and z_init == "last":
            self.offset_conv.bias.data[2:2 + self.num_z_channels] += 6.9
        elif self.num_z_channels > 0 and z_mode == "trilinear" and z_init == "first":
            self.offset_conv.bias.data[2:2 + self.num_z_channels] -= 6.9
        if use_mask:
            self.offset_conv.bias.data[-1:] += 2.0

        self.out_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self._grids = _GridCache()

    def forward(self, pyramid):
        x = pyramid[-1]
        B, C, H, W = x.shape
        L = len(pyramid)
        _, scale, norm_p0 = self._grids.get(1, H, W, x.device)

        if self.level_norms is not None:
            # offsets are predicted from the raw reference level (x, above);
            # sampling sees the normalized pyramid
            pyramid = [n(t) for n, t in zip(self.level_norms, pyramid)]

        inp = x
        if self.coords_as_extra_input:
            inp = torch.cat((x, norm_p0.expand(B, -1, -1, -1)), dim=1)
        raw = self.offset_conv(inp)  # (B, n_off, H, W)

        xy_off = raw[:, :2]
        if self.tanh_spatial_offsets:
            xy_off = torch.tanh(xy_off)
        xy = norm_p0 + xy_off * scale.squeeze(2)  # offsets in pixels of pyramid[-1]
        xy = xy.reshape(B, 2, -1).permute(0, 2, 1)  # (B, M, 2), M = H*W

        z_raw = raw[:, 2:2 + self.num_z_channels]
        if self.freeze_z:
            z_raw = z_raw.detach()
        if self.num_z_channels == 0:
            grid = torch.cat((xy, torch.zeros_like(xy[..., :1])), dim=-1)
            sampled = trilinear_pyramid_sampling(pyramid, grid, self.padding_mode)
        elif self.z_mode == "trilinear":
            z = torch.sigmoid(z_raw).reshape(B, 1, -1).permute(0, 2, 1)
            grid = torch.cat((xy, z), dim=-1)
            sampled = trilinear_pyramid_sampling(pyramid, grid, self.padding_mode)
        else:
            logits = z_raw.reshape(B, L, -1).permute(0, 2, 1)  # (B, M, L)
            sampled = attention_pyramid_sampling(pyramid, xy, logits, self.padding_mode)

        sampled = sampled.view(B, C, H, W)
        if self.use_mask:
            sampled = sampled * torch.sigmoid(raw[:, -1:])
        return self.out_conv(sampled)


class SingleLevelReducedDeformConv(nn.Module):
    """Reference-level-only control for ReducedPyramidalDeformConv: one
    spatially-deformed sample per location + 3x3 conv, no pyramid access.
    Isolates whether the reduced block's win comes from its cheap spatial
    warp or from cross-scale sampling."""

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        kwargs.pop("z_init", None)
        self.block = ReducedPyramidalDeformConv(
            in_channels, out_channels, num_levels=1, **kwargs
        )

    def forward(self, pyramid):
        x = pyramid[-1] if isinstance(pyramid, (list, tuple)) else pyramid
        return self.block([x])


class SingleLevelPyramidalDeformConv(nn.Module):
    """DCNv2-style baseline built from the exact same machinery as
    PyramidalDeformConv, restricted to the reference level (pyramid[-1]).

    This is the cleanest control for the pyramid: identical sampling code,
    identical init, identical options — the only difference is access to the
    other levels. It is also ~10x faster on CPU than torchvision's
    deform_conv2d kernel (see DeformConvBaseline), which makes it the practical
    deformable baseline for laptop experiments.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, **kwargs):
        super().__init__()
        kwargs.pop("z_init", None)  # meaningless with one level
        self.block = PyramidalDeformConv(
            in_channels, out_channels, kernel_size=kernel_size, num_levels=1,
            **kwargs,
        )

    def forward(self, pyramid):
        x = pyramid[-1] if isinstance(pyramid, (list, tuple)) else pyramid
        return self.block([x])


class DeformConvBaseline(nn.Module):
    """Standard (single-level) deformable convolution baseline (DCNv1/v2),
    built on torchvision's optimized DeformConv2d. Takes pyramid[-1]."""

    def __init__(self, in_channels, out_channels, kernel_size=3, use_mask=False):
        super().__init__()
        self.use_mask = use_mask
        K2 = kernel_size ** 2
        n_off = 2 * K2 + (K2 if use_mask else 0)
        self.offset_conv = nn.Conv2d(in_channels, n_off, kernel_size=3, padding=1)
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        if use_mask:
            self.offset_conv.bias.data[2 * K2:] += 2.0
        self.K2 = K2
        self.deform_conv = DeformConv2d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2
        )

    def forward(self, pyramid):
        x = pyramid[-1] if isinstance(pyramid, (list, tuple)) else pyramid
        raw = self.offset_conv(x)
        if self.use_mask:
            offset, mask = raw[:, : 2 * self.K2], torch.sigmoid(raw[:, 2 * self.K2:])
        else:
            offset, mask = raw, None
        return self.deform_conv(x, offset, mask)


class MultiScaleFusionBaseline(nn.Module):
    """Control baseline: non-deformable access to the same pyramid. All levels
    are bilinearly upsampled to the finest resolution, concatenated, reduced by
    a 1x1 conv and passed through a 3x3 conv. Distinguishes 'any multi-scale
    access helps' from 'learned deformable scale selection helps'."""

    def __init__(self, in_channels, out_channels, num_levels, kernel_size=3):
        super().__init__()
        self.reduce = nn.Conv2d(in_channels * num_levels, in_channels, kernel_size=1)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2
        )

    def forward(self, pyramid):
        target = pyramid[-1].shape[-2:]
        ups = [
            t if t.shape[-2:] == target
            else F.interpolate(t, size=target, mode="bilinear", align_corners=False)
            for t in pyramid
        ]
        return self.conv(F.relu(self.reduce(torch.cat(ups, dim=1))))


class PlainConvBaseline(nn.Module):
    """Plain 3x3 convolution on pyramid[-1]."""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2
        )

    def forward(self, pyramid):
        x = pyramid[-1] if isinstance(pyramid, (list, tuple)) else pyramid
        return self.conv(x)


HEAD_TYPES = ("plain_conv", "deform_conv", "deform_conv_v2", "multiscale_fusion",
              "pyramidal", "pyramidal_attention", "pyramidal_reduced",
              "pyramidal_single_level", "pyramidal_reduced_single_level")


def make_head(head_type, in_channels, out_channels, num_levels, **kwargs):
    """Factory used by models and experiment scripts."""
    if head_type == "plain_conv":
        return PlainConvBaseline(in_channels, out_channels)
    if head_type == "deform_conv":
        return DeformConvBaseline(in_channels, out_channels, use_mask=False)
    if head_type == "deform_conv_v2":
        return DeformConvBaseline(in_channels, out_channels, use_mask=True)
    if head_type == "multiscale_fusion":
        return MultiScaleFusionBaseline(in_channels, out_channels, num_levels)
    if head_type == "pyramidal":
        return PyramidalDeformConv(
            in_channels, out_channels, num_levels=num_levels, z_mode="trilinear", **kwargs
        )
    if head_type == "pyramidal_attention":
        return PyramidalDeformConv(
            in_channels, out_channels, num_levels=num_levels, z_mode="attention", **kwargs
        )
    if head_type == "pyramidal_reduced":
        return ReducedPyramidalDeformConv(
            in_channels, out_channels, num_levels=num_levels, **kwargs
        )
    if head_type == "pyramidal_single_level":
        return SingleLevelPyramidalDeformConv(in_channels, out_channels, **kwargs)
    if head_type == "pyramidal_reduced_single_level":
        return SingleLevelReducedDeformConv(in_channels, out_channels, **kwargs)
    raise ValueError(f"unknown head_type: {head_type} (choose from {HEAD_TYPES})")
