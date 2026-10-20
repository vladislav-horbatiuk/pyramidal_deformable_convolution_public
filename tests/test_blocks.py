"""Tests for the blocks and models."""
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyrdc.blocks import (
    HEAD_TYPES,
    PyramidalDeformConv,
    ReducedPyramidalDeformConv,
    make_head,
)
from pyrdc.models import SmallClassifier, SmallUNet


def make_pyramid(B=2, C=8, sizes=((8, 8), (16, 16), (32, 32), (32, 32)), seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(B, C, h, w, generator=g) for h, w in sizes]


@pytest.mark.parametrize("head_type", HEAD_TYPES)
def test_head_forward_shapes(head_type):
    pyramid = make_pyramid()
    head = make_head(head_type, in_channels=8, out_channels=5, num_levels=4)
    out = head(pyramid)
    assert out.shape == (2, 5, 32, 32)


@pytest.mark.parametrize("z_mode", ["trilinear", "attention"])
def test_block_at_init_equals_plain_conv(z_mode):
    """Offsets are zero-initialized, so on a pyramid whose levels are all the
    same tensor the block must reduce exactly to a plain 3x3 convolution
    (regardless of the initial z, which becomes irrelevant)."""
    torch.manual_seed(0)
    x = torch.randn(2, 8, 16, 16)
    pyramid = [x, x, x, x]
    block = PyramidalDeformConv(8, 5, num_levels=4, z_mode=z_mode)
    out = block(pyramid)
    ref = F.conv2d(
        x, block.weight.view(5, 8, 3, 3), bias=block.bias, padding=1
    )
    assert torch.allclose(out, ref, atol=1e-5), (out - ref).abs().max()


def test_z_init_last_ignores_coarse_levels_at_init():
    torch.manual_seed(0)
    fine = torch.randn(2, 8, 16, 16)
    pyr_a = [torch.randn(2, 8, 4, 4), torch.randn(2, 8, 8, 8), fine]
    pyr_b = [torch.zeros(2, 8, 4, 4), torch.zeros(2, 8, 8, 8), fine]
    block = PyramidalDeformConv(8, 5, num_levels=3, z_init="last")
    out_a, out_b = block(pyr_a), block(pyr_b)
    # sigmoid(6.9) ~ 0.999 -> coarse levels contribute ~0.1%
    assert torch.allclose(out_a, out_b, atol=1e-2)
    block_attn = PyramidalDeformConv(
        8, 5, num_levels=3, z_mode="attention", z_init="last"
    )
    assert torch.allclose(block_attn(pyr_a), block_attn(pyr_b), atol=1e-1)


def test_changing_spatial_size_and_batch_size():
    """Regression test for the notebook bug where grids were cached on the
    first forward pass and silently reused for other input sizes."""
    block = PyramidalDeformConv(4, 4, num_levels=2)
    out1 = block([torch.randn(3, 4, 8, 8), torch.randn(3, 4, 16, 16)])
    out2 = block([torch.randn(5, 4, 4, 4), torch.randn(5, 4, 8, 8)])
    assert out1.shape == (3, 4, 16, 16)
    assert out2.shape == (5, 4, 8, 8)

    reduced = ReducedPyramidalDeformConv(4, 4, num_levels=2)
    assert reduced([torch.randn(3, 4, 8, 8), torch.randn(3, 4, 16, 16)]).shape \
        == (3, 4, 16, 16)
    assert reduced([torch.randn(1, 4, 4, 4), torch.randn(1, 4, 8, 8)]).shape \
        == (1, 4, 8, 8)


@pytest.mark.parametrize("z_mode", ["trilinear", "attention"])
@pytest.mark.parametrize("use_mask", [False, True])
def test_all_parameters_receive_gradients(z_mode, use_mask):
    torch.manual_seed(0)
    block = PyramidalDeformConv(
        4, 4, num_levels=3, z_mode=z_mode, use_mask=use_mask, z_init="middle"
    )
    pyramid = [torch.randn(2, 4, s, s) for s in (4, 8, 16)]
    block(pyramid).sum().backward()
    for name, p in block.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"


def test_single_level_block_equals_plain_conv_at_init_and_matches_dcn_shape():
    """The single-level baseline must behave exactly like PyramidalDeformConv
    restricted to the reference level: plain conv at init, no z parameters."""
    from pyrdc.blocks import SingleLevelPyramidalDeformConv

    torch.manual_seed(0)
    x = torch.randn(2, 8, 16, 16)
    block = SingleLevelPyramidalDeformConv(8, 5)
    out = block([torch.randn(2, 8, 8, 8), x])  # coarse level must be ignored
    ref = F.conv2d(
        x, block.block.weight.view(5, 8, 3, 3), bias=block.block.bias, padding=1
    )
    assert torch.allclose(out, ref, atol=1e-5)
    assert block.block.num_z_channels == 0
    # after a step with nonzero offsets, all params must receive gradients
    block.block.offset_conv.weight.data.normal_(std=0.01)
    block([x]).sum().backward()
    for name, p in block.named_parameters():
        assert p.grad is not None, f"no grad for {name}"


@pytest.mark.parametrize("head_type", ["plain_conv", "deform_conv_v2",
                                       "multiscale_fusion", "pyramidal",
                                       "pyramidal_attention", "pyramidal_reduced",
                                       "pyramidal_single_level"])
def test_models_forward_and_train_step(head_type):
    torch.manual_seed(0)
    clf = SmallClassifier(head_type, num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    logits = clf(x)
    assert logits.shape == (2, 10)
    logits.sum().backward()

    unet = SmallUNet(head_type, out_channels=1)
    x = torch.randn(2, 3, 32, 32)
    out = unet(x)
    assert out.shape == (2, 1, 32, 32)
    out.sum().backward()
    for name, p in unet.named_parameters():
        assert p.grad is not None, f"no grad for {name}"


@pytest.mark.parametrize("z_mode", ["trilinear", "attention"])
def test_freeze_z_stops_gradient_to_z_branch_only(z_mode):
    torch.manual_seed(0)
    block = PyramidalDeformConv(4, 4, num_levels=3, z_mode=z_mode,
                                z_init="middle", freeze_z=True)
    # give offsets nonzero weights so xy gradients are nontrivial
    block.offset_conv.weight.data.normal_(std=0.01)
    pyramid = [torch.randn(2, 4, s, s) for s in (4, 8, 16)]
    block(pyramid).sum().backward()
    K2 = block.K2
    w_grad = block.offset_conv.weight.grad
    b_grad = block.offset_conv.bias.grad
    z_sl = slice(2 * K2, 2 * K2 + block.num_z_channels)
    assert torch.all(w_grad[z_sl] == 0), "z weights must get no gradient"
    assert torch.all(b_grad[z_sl] == 0), "z bias must get no gradient"
    assert w_grad[: 2 * K2].abs().sum() > 0, "xy offsets must still learn"
    assert block.weight.grad is not None and block.weight.grad.abs().sum() > 0


def test_per_image_dice_matches_dataset_meter():
    from pyrdc.metrics import per_image_dice, SegmentationMeter

    torch.manual_seed(0)
    logits = torch.randn(4, 1, 8, 8)
    targets = (torch.rand(4, 8, 8) > 0.5).long()
    d = per_image_dice(logits, targets)
    assert d.shape == (4,)
    assert ((d >= 0) & (d <= 1)).all()
    # perfect prediction -> dice 1
    perfect = torch.where(targets.unsqueeze(1) > 0, 100.0, -100.0)
    assert torch.allclose(per_image_dice(perfect, targets), torch.ones(4), atol=1e-4)


@pytest.mark.parametrize("z_mode", ["trilinear", "attention"])
def test_cheap_z_shared_across_kernel_points(z_mode):
    torch.manual_seed(0)
    full = PyramidalDeformConv(8, 5, num_levels=4, z_mode=z_mode)
    cheap = PyramidalDeformConv(8, 5, num_levels=4, z_mode=z_mode,
                                z_per_kernel_point=False)
    n_full = sum(p.numel() for p in full.parameters())
    n_cheap = sum(p.numel() for p in cheap.parameters())
    assert n_cheap < n_full
    assert cheap.num_z_channels == (1 if z_mode == "trilinear" else 4)
    # still exactly a plain conv at init on an identical-level pyramid
    x = torch.randn(2, 8, 16, 16)
    out = cheap([x, x, x, x])
    ref = F.conv2d(x, cheap.weight.view(5, 8, 3, 3), bias=cheap.bias, padding=1)
    assert torch.allclose(out, ref, atol=1e-5)
    # gradients reach every parameter
    cheap([torch.randn(2, 8, s, s) for s in (4, 8, 8, 16)]).sum().backward()
    for name, p in cheap.named_parameters():
        assert p.grad is not None, f"no grad for {name}"


def test_reduced_single_level_and_freeze_z():
    from pyrdc.blocks import SingleLevelReducedDeformConv

    torch.manual_seed(0)
    block = SingleLevelReducedDeformConv(8, 5)
    assert block.block.num_z_channels == 0
    x = torch.randn(2, 8, 16, 16)
    out = block([torch.randn(2, 8, 8, 8), x])  # coarse level ignored
    assert out.shape == (2, 5, 16, 16)
    out.sum().backward()

    frozen = ReducedPyramidalDeformConv(8, 5, num_levels=3, freeze_z=True)
    frozen.offset_conv.weight.data.normal_(std=0.01)
    pyr = [torch.randn(2, 8, s, s) for s in (4, 8, 16)]
    frozen(pyr).sum().backward()
    z_sl = slice(2, 2 + frozen.num_z_channels)
    assert torch.all(frozen.offset_conv.weight.grad[z_sl] == 0)
    assert frozen.offset_conv.weight.grad[:2].abs().sum() > 0


def test_kvasir_dataset_loads():
    import os
    if not os.path.isdir("kvasir_seg_raw/Kvasir-SEG"):
        pytest.skip("kvasir dataset not present")
    from pyrdc.data import build_segmentation_datasets

    train_ds, test_ds = build_segmentation_datasets("kvasir", size=(64, 64))
    assert len(train_ds) == 800 and len(test_ds) == 200
    img, mask = train_ds[0]
    assert img.shape == (3, 64, 64) and mask.shape == (64, 64)
    assert set(mask.unique().tolist()) <= {0, 1}
    img2, mask2 = test_ds[len(test_ds) - 1]
    assert mask2.dtype == torch.long


@pytest.mark.parametrize("z_mode", ["trilinear", "attention"])
def test_level_norm(z_mode):
    torch.manual_seed(0)
    block = PyramidalDeformConv(8, 5, num_levels=3, z_mode=z_mode,
                                use_level_norm=True)
    assert len(block.level_norms) == 3
    pyr = [torch.randn(2, 8, s, s) for s in (4, 8, 16)]
    out = block(pyr)
    assert out.shape == (2, 5, 16, 16)
    out.sum().backward()
    for name, p in block.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
    # extra params = 2*C per level
    plain = PyramidalDeformConv(8, 5, num_levels=3, z_mode=z_mode)
    n_extra = sum(p.numel() for p in block.parameters()) - \
        sum(p.numel() for p in plain.parameters())
    assert n_extra == 3 * 2 * 8


def test_level_norm_supports_widths_not_divisible_by_four():
    """Regression: level-norm used GroupNorm(min(4, C), C), which crashes for
    every width that is not a multiple of 4 — exactly the widths the
    capacity-matched sweeps use (ch 17, 19, 66). Iteration 6's
    `kv_pyr_ln_ch17` run died on this."""
    from pyrdc.blocks import _group_norm_groups

    for channels in (17, 19, 21, 66):
        block = PyramidalDeformConv(channels, 4, num_levels=3, use_level_norm=True)
        pyramid = make_pyramid(B=2, C=channels, sizes=((4, 4), (8, 8), (8, 8)))
        assert block(pyramid).shape == (2, 4, 8, 8)

    # widths that already worked keep their previous group count
    for channels in (1, 2, 3, 4, 8, 16, 20, 64):
        assert _group_norm_groups(channels) == min(4, channels)


def test_residual_cross_scale_starts_as_exact_dcn():
    """The gate is initialised to zero, so at step 0 the residual variant must
    produce bit-identical output to the same block with depth pinned to the
    reference level -- the property that makes it unable to lose to DCN."""
    torch.manual_seed(0)
    pyramid = make_pyramid(B=2, C=8, sizes=((4, 4), (8, 8), (16, 16), (16, 16)))

    res = PyramidalDeformConv(8, 5, num_levels=4, z_init="middle",
                              residual_cross_scale=True)
    ref = PyramidalDeformConv(8, 5, num_levels=4, z_init="middle")
    ref.load_state_dict({k: v for k, v in res.state_dict().items()
                         if k != "cross_gate"}, strict=False)
    # Pin ref's depth to the reference level exactly: sigmoid(30) == 1.0 in
    # float32, whereas z_init='last' only reaches 0.999 and would still bleed a
    # little weight into level L-2.
    K2, n_z = ref.K2, ref.num_z_channels
    with torch.no_grad():
        ref.offset_conv.bias[2 * K2: 2 * K2 + n_z] = 30.0
    assert torch.allclose(res(pyramid), ref(pyramid), atol=1e-5)

    # and the gate is a real parameter that training can open
    assert res.cross_gate.requires_grad and res.cross_gate.numel() == 8  # in_channels
    with torch.no_grad():
        res.cross_gate.fill_(1.0)
    assert not torch.allclose(res(pyramid), ref(pyramid), atol=1e-5)


def test_residual_branch_reads_unnormalised_pyramid():
    """With level-norm on, the residual branch must still reproduce plain DCN
    on the raw features -- normalisation exists to make levels commensurable
    for mixing, not to alter the baseline the block falls back to."""
    torch.manual_seed(0)
    pyramid = make_pyramid(B=2, C=8, sizes=((4, 4), (8, 8), (16, 16), (16, 16)))

    res = PyramidalDeformConv(8, 5, num_levels=4, z_init="middle",
                              use_level_norm=True, residual_cross_scale=True)
    ref = PyramidalDeformConv(8, 5, num_levels=4, z_init="middle")
    ref.load_state_dict({k: v for k, v in res.state_dict().items()
                         if not k.startswith("level_norms") and k != "cross_gate"},
                        strict=False)
    K2, n_z = ref.K2, ref.num_z_channels
    with torch.no_grad():
        ref.offset_conv.bias[2 * K2: 2 * K2 + n_z] = 30.0   # pin to reference level
    # gate is 0, so res == plain DCN on the *raw* reference level
    assert torch.allclose(res(pyramid), ref(pyramid), atol=1e-5)


def test_reduced_block_accepts_level_norm():
    """Regression: the reduced variant had no use_level_norm, so it was only
    ever evaluated in the unnormalised regime that invalidated the full block's
    early results."""
    pyramid = make_pyramid(B=2, C=8, sizes=((4, 4), (8, 8), (16, 16), (16, 16)))
    block = ReducedPyramidalDeformConv(8, 5, num_levels=4, use_level_norm=True)
    assert block(pyramid).shape == (2, 5, 16, 16)
    assert len(block.level_norms) == 4
    # +2*C affine params per level, and nothing else
    plain = ReducedPyramidalDeformConv(8, 5, num_levels=4)
    n_extra = sum(p.numel() for p in block.parameters()) - \
        sum(p.numel() for p in plain.parameters())
    assert n_extra == 4 * 2 * 8


def test_z_weight_init_std_only_touches_depth_rows():
    torch.manual_seed(0)
    block = PyramidalDeformConv(8, 5, num_levels=4, z_init="middle",
                                z_weight_init_std=1e-2)
    w = block.offset_conv.weight.data
    K2, n_z = block.K2, block.num_z_channels
    assert (w[: 2 * K2] == 0).all()                      # spatial rows untouched
    assert w[2 * K2: 2 * K2 + n_z].abs().sum() > 0       # depth rows randomised
    # default stays exactly zero-init
    assert (PyramidalDeformConv(8, 5, num_levels=4).offset_conv.weight.data == 0).all()


@pytest.mark.parametrize("mode", ["free", "affine", "similarity"])
def test_offset_mode_is_plain_conv_at_init(mode):
    """All three parameterisations must be zero at init, so the block still
    reduces exactly to a 3x3 convolution on a pyramid of identical levels."""
    torch.manual_seed(0)
    x = torch.randn(2, 8, 16, 16)
    block = PyramidalDeformConv(8, 5, num_levels=4, z_init="middle",
                                offset_mode=mode)
    out = block([x, x, x, x])
    ref = F.conv2d(x, block.weight.view(5, 8, 3, 3), bias=block.bias, padding=1)
    assert torch.allclose(out, ref, atol=1e-5), (out - ref).abs().max()


def test_affine_offsets_generate_a_linear_warp_of_the_kernel():
    """Delta_k = t + A p_k: setting the offset conv's bias to a known (t, A)
    must reproduce exactly that warp at every location, including a pure
    dilation -- the mode the decomposition says dominates."""
    from pyrdc.blocks import _kernel_offsets

    block = PyramidalDeformConv(8, 5, num_levels=4, z_init="middle",
                                offset_mode="affine")
    # t = (2, -1); A = diag(0.5, 0.5) -> kernel dilated by 1.5x about its centre
    with torch.no_grad():
        block.offset_conv.bias[:6] = torch.tensor([2.0, -1.0, 0.5, 0.0, 0.0, 0.5])
    raw = block.offset_conv.bias[: block.n_xy].view(1, 6, 1, 1).expand(1, 6, 4, 4)
    delta = block._spatial_offsets(raw, 1, 4, 4)          # (1, 2, K2, 4, 4)
    p_n = _kernel_offsets(3)                               # (2, K2)
    expect = torch.tensor([2.0, -1.0]).view(1, 2, 1) + 0.5 * p_n.unsqueeze(0)
    assert torch.allclose(delta[..., 0, 0], expect, atol=1e-6)
    # the warped kernel spans 1.5x the base grid in each axis
    warped = p_n + delta[0, :, :, 0, 0]
    assert torch.allclose(warped.max(1).values - warped.min(1).values,
                          torch.tensor([3.0, 3.0]), atol=1e-6)


def test_similarity_offsets_are_scale_times_rotation():
    from pyrdc.blocks import _kernel_offsets

    block = PyramidalDeformConv(8, 5, num_levels=4, z_init="middle",
                                offset_mode="similarity")
    import math
    with torch.no_grad():  # log s = 0 -> s = 1; theta = 90 degrees
        block.offset_conv.bias[:4] = torch.tensor([0.0, 0.0, 0.0, math.pi / 2])
    raw = block.offset_conv.bias[: block.n_xy].view(1, 4, 1, 1).expand(1, 4, 4, 4)
    delta = block._spatial_offsets(raw, 1, 4, 4)
    p_n = _kernel_offsets(3)
    warped = p_n + delta[0, :, :, 0, 0]
    # a 90-degree rotation maps (x, y) -> (-y, x); the set of kernel points is
    # symmetric under it, so the warped set equals the original set
    a = {tuple(round(v, 5) for v in c) for c in warped.t().tolist()}
    b = {tuple(round(v, 5) for v in c) for c in p_n.t().tolist()}
    assert a == b, (sorted(a), sorted(b))
    # and it is a rotation, not the identity map, on individual points
    assert not torch.allclose(warped, p_n, atol=1e-4)


def test_offset_mode_channel_budget():
    for mode, n_xy in (("free", 18), ("affine", 6), ("similarity", 4)):
        b = PyramidalDeformConv(8, 5, num_levels=4, z_init="middle",
                                z_per_kernel_point=False, offset_mode=mode)
        assert b.n_xy == n_xy
        assert b.offset_conv.out_channels == n_xy + 1   # + one shared z


def test_confusion_meter_miou():
    from pyrdc.metrics import ConfusionMeter

    meter = ConfusionMeter(num_classes=3)
    targets = torch.tensor([[[0, 1], [2, 255]]])  # 255 ignored
    logits = torch.zeros(1, 3, 2, 2)
    logits[0, 0, 0, 0] = 5  # correct (0)
    logits[0, 2, 0, 1] = 5  # wrong (pred 2, true 1)
    logits[0, 2, 1, 0] = 5  # correct (2)
    logits[0, 0, 1, 1] = 5  # ignored pixel
    meter.update(logits, targets)
    assert meter.mat.sum().item() == 3  # ignored pixel excluded
    # class0: 1/1, class1: 0/1, class2: 1/2 -> mIoU = (1 + 0 + 0.5)/3
    assert abs(meter.miou - (1 + 0 + 0.5) / 3) < 1e-6
    assert abs(meter.pixel_acc - 2 / 3) < 1e-6
    iou = meter.per_class_iou
    assert torch.allclose(iou, torch.tensor([1.0, 0.0, 0.5]))
    # each class owns one labelled GT pixel out of three
    assert torch.allclose(meter.class_pixel_freq, torch.full((3,), 1 / 3))


def test_per_class_iou_marks_absent_classes_nan():
    """A class with no GT and no prediction must not drag mIoU down as a zero."""
    from pyrdc.metrics import ConfusionMeter

    meter = ConfusionMeter(num_classes=4)
    meter.update(torch.zeros(1, 4, 1, 1), torch.zeros(1, 1, 1, dtype=torch.long))
    iou = meter.per_class_iou
    assert iou[0].item() == 1.0
    assert torch.isnan(iou[1:]).all()
    assert abs(meter.miou - 1.0) < 1e-6


def test_per_image_miou_matches_confusion_meter_on_one_image():
    from pyrdc.metrics import ConfusionMeter, per_image_miou

    torch.manual_seed(0)
    logits = torch.randn(3, 5, 6, 6)
    targets = torch.randint(0, 5, (3, 6, 6))
    targets[0, 0, 0] = 255  # ignored pixels must be dropped, not counted
    per_img = per_image_miou(logits, targets, num_classes=5)
    assert per_img.shape == (3,)
    for i in range(3):
        meter = ConfusionMeter(num_classes=5)
        meter.update(logits[i : i + 1], targets[i : i + 1])
        assert abs(per_img[i].item() - meter.miou) < 1e-6


def test_cityscapes_synthetic_end_to_end(tmp_path):
    """Build a tiny synthetic dataset in the official Cityscapes layout and
    run one multiclass train epoch through the real pipeline."""
    import numpy as np
    from PIL import Image
    from pyrdc.data import (
        CityscapesSeg, CityscapesTrainTransform, CityscapesValTransform,
    )
    from pyrdc.engine import train_segmentation_multiclass
    from pyrdc.models import SmallUNet

    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        img_dir = tmp_path / "leftImg8bit" / split / "cityA"
        gt_dir = tmp_path / "gtFine" / split / "cityA"
        img_dir.mkdir(parents=True)
        gt_dir.mkdir(parents=True)
        for i in range(2):
            stem = f"cityA_{i:06d}_000019"
            Image.fromarray(
                rng.integers(0, 255, (128, 256, 3), dtype=np.uint8)
            ).save(img_dir / f"{stem}_leftImg8bit.png")
            Image.fromarray(
                rng.integers(0, 34, (128, 256), dtype=np.uint8)
            ).save(gt_dir / f"{stem}_gtFine_labelIds.png")
            # torchvision's Cityscapes also indexes these companion files
            Image.fromarray(
                np.zeros((128, 256), dtype=np.uint8)
            ).save(gt_dir / f"{stem}_gtFine_instanceIds.png")
            (gt_dir / f"{stem}_gtFine_polygons.json").write_text("{}")
            Image.fromarray(
                np.zeros((128, 256, 3), dtype=np.uint8)
            ).save(gt_dir / f"{stem}_gtFine_color.png")

    train_ds = CityscapesSeg(
        str(tmp_path), "train", CityscapesTrainTransform((128, 64), (32, 64))
    )
    val_ds = CityscapesSeg(str(tmp_path), "val", CityscapesValTransform((128, 64)))
    img, mask = train_ds[0]
    assert img.shape == (3, 32, 64)
    assert mask.max() <= 255 and ((mask < 19) | (mask == 255)).all()

    tl = torch.utils.data.DataLoader(train_ds, batch_size=2)
    vl = torch.utils.data.DataLoader(val_ds, batch_size=2)
    model = SmallUNet("pyramidal_attention", out_channels=19, channels=8)
    res = train_segmentation_multiclass(
        model, tl, vl, "cpu", 19, writer=None, epochs=1, eval_every=1
    )
    assert 0.0 <= res["final"]["miou"] <= 1.0

    # the per-image / per-class breakdown that feeds the Cityscapes tables
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
    from run_cityscapes import dump_val_breakdown

    dump_val_breakdown(model, vl, "cpu", "unit_test_tag", str(tmp_path / "out"))
    per_image = (tmp_path / "out" / "per_image" / "unit_test_tag.csv").read_text()
    per_class = (tmp_path / "out" / "per_image" / "unit_test_tag_per_class.csv").read_text()
    assert len(per_image.strip().splitlines()) == 1 + len(val_ds)
    assert len(per_class.strip().splitlines()) == 1 + 19
    assert "road" in per_class


def test_adapters_are_registered_parameters():
    """Regression test: notebook models kept adapter convs in a plain Python
    list, so their parameters were never trained, moved or checkpointed."""
    unet = SmallUNet("pyramidal", out_channels=1)
    adapter_params = [n for n, _ in unet.named_parameters() if "adapters" in n]
    assert len(adapter_params) == 6  # 3 adapters x (weight, bias)
    assert any("adapters" in k for k in unet.state_dict())


# ---------------------------------------------------------------------------
# Level-aware offset scaling (scale_offsets_by_level)
# ---------------------------------------------------------------------------

def _twin_blocks(**kwargs):
    """Two blocks with identical weights, the first with offset scaling on."""
    torch.manual_seed(0)
    a = PyramidalDeformConv(8, 5, num_levels=4, scale_offsets_by_level=True, **kwargs)
    b = PyramidalDeformConv(8, 5, num_levels=4, **kwargs)
    b.load_state_dict({k: v for k, v in a.state_dict().items()})
    return a, b


def test_offset_scale_at_init_equals_plain_conv():
    torch.manual_seed(0)
    x = torch.randn(2, 8, 16, 16)
    block = PyramidalDeformConv(8, 5, num_levels=4, scale_offsets_by_level=True)
    out = block([x, x, x, x])
    ref = F.conv2d(x, block.weight.view(5, 8, 3, 3), bias=block.bias, padding=1)
    assert torch.allclose(out, ref, atol=1e-5)


def test_offset_scale_identity_at_reference_level():
    """With z pinned exactly to the reference level, stride(z) == 1 and the
    scaled block must match the unscaled block bit-for-bit."""
    a, b = _twin_blocks()
    for blk in (a, b):
        with torch.no_grad():
            # random spatial offsets, z pinned to the reference (sigmoid(50)=1.)
            torch.manual_seed(1)
            blk.offset_conv.weight[: blk.n_xy].normal_(std=0.3)
            blk.offset_conv.bias[blk.n_xy: blk.n_xy + blk.num_z_channels] = 50.0
    pyramid = make_pyramid()
    assert torch.equal(a(pyramid), b(pyramid))


def test_offset_scale_multiplies_by_stride_at_coarsest_level():
    """With z pinned exactly to the coarsest level, the scaled block with
    offset bias d must equal the unscaled block with offset bias d*stride."""
    pyramid = make_pyramid()  # coarsest 8x8, reference 32x32
    stride = (32 - 1) / (8 - 1)
    a, b = _twin_blocks()
    d = 0.7
    with torch.no_grad():
        for blk, bias in ((a, d), (b, d * stride)):
            blk.offset_conv.bias[: blk.n_xy] = bias
            blk.offset_conv.bias[blk.n_xy: blk.n_xy + blk.num_z_channels] = -50.0
    out_a, out_b = a(pyramid), b(pyramid)
    assert torch.allclose(out_a, out_b, atol=1e-6), (out_a - out_b).abs().max()


def test_offset_scale_rejects_attention_mode():
    with pytest.raises(ValueError):
        PyramidalDeformConv(8, 5, num_levels=4, z_mode="attention",
                            scale_offsets_by_level=True)


def test_offset_scale_works_with_shared_z():
    a, _ = _twin_blocks(z_per_kernel_point=False)
    out = a(make_pyramid())
    assert out.shape == (2, 5, 32, 32)


def test_ade20k_synthetic_end_to_end(tmp_path):
    """Tiny synthetic dataset in the ADEChallengeData2016 layout through the
    real multiclass pipeline (150 classes, label 0 -> ignore=255)."""
    import numpy as np
    from PIL import Image
    from pyrdc.data import (
        ADE20K_LABEL_MAP, Ade20kSeg, CityscapesTrainTransform,
        CityscapesValTransform,
    )
    from pyrdc.engine import train_segmentation_multiclass
    from pyrdc.models import SmallUNet

    rng = np.random.default_rng(0)
    for split in ("training", "validation"):
        img_dir = tmp_path / "images" / split
        ann_dir = tmp_path / "annotations" / split
        img_dir.mkdir(parents=True)
        ann_dir.mkdir(parents=True)
        for i in range(2):
            Image.fromarray(
                rng.integers(0, 255, (96, 128, 3), dtype=np.uint8)
            ).save(img_dir / f"ADE_{split}_{i:08d}.jpg")
            Image.fromarray(
                rng.integers(0, 151, (96, 128), dtype=np.uint8)
            ).save(ann_dir / f"ADE_{split}_{i:08d}.png")

    train_ds = Ade20kSeg(
        str(tmp_path), "train",
        CityscapesTrainTransform((64, 64), (32, 32), label_map=ADE20K_LABEL_MAP),
    )
    val_ds = Ade20kSeg(
        str(tmp_path), "val",
        CityscapesValTransform((64, 64), label_map=ADE20K_LABEL_MAP),
    )
    img, mask = train_ds[0]
    assert img.shape == (3, 32, 32)
    assert ((mask < 150) | (mask == 255)).all()

    tl = torch.utils.data.DataLoader(train_ds, batch_size=2)
    vl = torch.utils.data.DataLoader(val_ds, batch_size=2)
    model = SmallUNet("pyramidal", out_channels=150, channels=8,
                      z_per_kernel_point=False, use_level_norm=True)
    res = train_segmentation_multiclass(
        model, tl, vl, "cpu", 150, writer=None, epochs=1, eval_every=1
    )
    assert 0.0 <= res["final"]["miou"] <= 1.0


def test_z_stretch_zero_is_plain_sigmoid():
    block = PyramidalDeformConv(8, 5, num_levels=4)
    z_raw = torch.linspace(-8, 8, 101)
    assert torch.equal(block.depth(z_raw), torch.sigmoid(z_raw))


def test_z_stretch_reaches_both_ends_at_finite_z():
    eps = 0.1
    block = PyramidalDeformConv(8, 5, num_levels=4, z_stretch=eps)
    # the end levels are hit exactly once sigmoid(z) = (1 + eps) / (1 + 2 eps)
    z_end = torch.logit(torch.tensor((1 + eps) / (1 + 2 * eps)))
    assert block.depth(z_end + 1e-3).item() == 1.0
    assert block.depth(-z_end - 1e-3).item() == 0.0
    assert abs(block.depth(torch.tensor(0.0)).item() - 0.5) < 1e-7
    # strictly inside, the map still has a gradient
    z = torch.tensor(1.0, requires_grad=True)
    block.depth(z).backward()
    assert z.grad.item() > 0


def test_z_stretch_reads_only_reference_level_at_finite_bias():
    """Stretch-and-clip makes 'exactly DCN' reachable with a finite depth bias:
    the other levels must then have no influence on the output."""
    torch.manual_seed(0)
    x = torch.randn(2, 8, 16, 16)
    block = PyramidalDeformConv(8, 5, num_levels=4, z_stretch=0.1)
    with torch.no_grad():
        block.offset_conv.bias[block.n_xy: block.n_xy + block.num_z_channels] = 3.0
    noise = [torch.randn(2, 8, s, s) for s in (32, 8, 16)]
    out = block(noise + [x])
    ref = F.conv2d(x, block.weight.view(5, 8, 3, 3), bias=block.bias, padding=1)
    assert torch.allclose(out, ref, atol=1e-5)


def test_z_stretch_rejects_attention_mode():
    with pytest.raises(ValueError):
        PyramidalDeformConv(8, 5, num_levels=4, z_mode="attention", z_stretch=0.1)
