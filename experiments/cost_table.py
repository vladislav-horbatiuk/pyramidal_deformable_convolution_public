#!/usr/bin/env python3
"""Cost table for the paper: params, dense-op FLOPs, measured CPU latency.

    micromamba run -n ml_env python3 experiments/cost_table.py

FLOPs come from torch.profiler (with_flops=True), which covers conv / matmul /
einsum-style dense ops but NOT grid_sample; the resampling work of the pyramid
heads is therefore under-counted there. Measured wall-clock latency is the
honest cost number and is reported alongside. Run on an otherwise idle machine;
threads are pinned for stability.
"""
import statistics
import sys
import time

import torch

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(
    __import__("os").path.abspath(__file__))))
from pyrdc.engine import count_parameters  # noqa: E402
from pyrdc.models import SmallUNet  # noqa: E402

torch.set_num_threads(4)

# (label, head_type, channels, head_kwargs) — the headline table rows.
SMALL = [  # Oxford/Kvasir bracket, input 3x64x64
    ("plain conv ch16", "plain_conv", 16, {}),
    ("spatial-only DCN ch17", "pyramidal_single_level", 17, {"z_init": "middle"}),
    ("spatial-only DCN+LN ch17", "pyramidal_single_level", 17,
     {"z_init": "middle", "use_level_norm": True}),
    ("cheap-z (zshared+LN) ch16", "pyramidal", 16,
     {"z_init": "middle", "z_per_kernel_point": False, "use_level_norm": True}),
    ("full-z+LN ch16", "pyramidal", 16,
     {"z_init": "middle", "use_level_norm": True}),
    ("attention+LN ch16", "pyramidal_attention", 16,
     {"z_init": "middle", "use_level_norm": True}),
    ("spatial-only DCN ch19", "pyramidal_single_level", 19, {"z_init": "middle"}),
]
WIDE = [  # Cityscapes bracket, input 3x256x256 (19-class head)
    ("plain conv ch64", "plain_conv", 64, {}),
    ("spatial-only DCN ch66", "pyramidal_single_level", 66, {"z_init": "middle"}),
    ("cheap-z ch64", "pyramidal", 64,
     {"z_init": "middle", "z_per_kernel_point": False, "use_level_norm": True}),
    ("attention+zshared+LN ch64", "pyramidal_attention", 64,
     {"z_init": "middle", "z_per_kernel_point": False, "use_level_norm": True}),
]


def profiler_flops(model, x):
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU], with_flops=True) as prof:
        with torch.no_grad():
            model(x)
    return sum(e.flops for e in prof.key_averages() if e.flops)


def latency_ms(model, x, warmup=3, runs=15):
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def table(rows, size, out_channels):
    x = torch.randn(1, 3, size, size)
    print(f"\n--- input 1x3x{size}x{size}, {out_channels}-ch head ---")
    print(f"  {'config':28s} {'params':>9s} {'GFLOPs*':>9s} {'ms/img':>8s} {'rel':>6s}")
    base_ms = None
    for label, head, ch, kw in rows:
        torch.manual_seed(0)
        model = SmallUNet(head, out_channels=out_channels, channels=ch, **kw).eval()
        p = count_parameters(model)
        gf = profiler_flops(model, x) / 1e9
        ms = latency_ms(model, x)
        base_ms = base_ms or ms
        print(f"  {label:28s} {p:9,d} {gf:9.3f} {ms:8.1f} {ms / base_ms:5.1f}x")


if __name__ == "__main__":
    table(SMALL, 64, 1)
    table(WIDE, 256, 19)
    print("\n* profiler dense-op FLOPs; grid_sample resampling is not counted, "
          "so pyramid-head FLOPs are lower bounds. Latency is the honest cost.")
