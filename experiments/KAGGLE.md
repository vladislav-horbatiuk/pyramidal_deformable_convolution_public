# Kaggle GPU experiments (Cityscapes / ADE20K, T4×2, 16 GB)

## STATUS: complete — no sessions needed

The ADE20K normalisation-matched control finished 2026-10-05 (seeds 0–2,
sessions 1–2). Outcome under the rule below: **downgrade** — cheap-z wins
per-image mIoU but not pixel accuracy (RESULTS §5b).
Everything below is kept as reference for reproducing or extending the study.

### ADE20K LN control (done; cell kept as reference)

**Pre-registered decision rule (fixed before launch; do not edit after).**
The LN baseline is an *additional* arm: the existing un-normalised comparison
stays as run, and both are reported. Compare cheap-z ch64 (seeds 0–2, already
in `results/ade20k.csv`) against single ch66 +LN (seeds 0–2):
- cheap-z wins **both** per-image mIoU and pixel accuracy at seed level
  (Welch p < 0.05) → the ADE20K claim stands as written; the paper's
  normalisation-asymmetry caveat becomes "the normalisation-matched control
  was run".
- otherwise → the ADE20K claim is downgraded to "beats the un-normalised
  baseline", and the introduction's finding 2 / the §ade text change
  accordingly.

**Checklist.**
1. Notebook with **Accelerator = GPU T4 x2**.
2. Attach the ADE20K dataset and the `pyrdc-code` dataset (the 2026-08-16 zip
   is current: no file in `pyrdc/` or `run_cityscapes.py` is newer).
3. Unzip code: `!cp /kaggle/input/pyrdc-code/pyrdc_code.zip . && unzip -qo pyrdc_code.zip`
4. Session 1: run the cell below; **verify both logs print `params=337758`**
   before walking away. Session 2: same cell with only `p0`, `--seeds 2`
   (drop `p1` and its poll/tail lines).
5. Afterwards, download per "After every session" below (results CSV:
   `results/ade20k.csv`).

```python
import os, subprocess, shlex, time

ROOT   = "/kaggle/input/<your-ade20k-dataset>/<dir containing images/ and annotations/>"
COMMON = (f"--dataset ade20k --root {ROOT} --base-size 512 512 --crop 256 256 "
          "--batch-size 4 --val-batch-size 1 --epochs 10 --eval-every 2 "
          "--num-workers 2 --out-dir /kaggle/working")

def launch(gpu, head, extra):
    cmd = (f"python experiments/run_cityscapes.py --head {head} {extra} {COMMON} "
           f"--results-csv /kaggle/working/ade_gpu{gpu}.csv")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    return subprocess.Popen(shlex.split(cmd), env=env,
                            stdout=open(f"/kaggle/working/gpu{gpu}.log", "w"),
                            stderr=subprocess.STDOUT)

p0 = launch(0, "pyramidal_single_level", "--channels 66 --level-norm --seeds 0")
p1 = launch(1, "pyramidal_single_level", "--channels 66 --level-norm --seeds 1")

def tail(path, n=2):
    try:
        with open(path) as f:
            return "".join(f.readlines()[-n:]).strip()
    except FileNotFoundError:
        return "(no log yet)"

while p0.poll() is None or p1.poll() is None:
    time.sleep(300)
    print("-" * 60, flush=True)
    print("[gpu0]", tail("/kaggle/working/gpu0.log"), flush=True)
    print("[gpu1]", tail("/kaggle/working/gpu1.log"), flush=True)
print("exit codes:", p0.returncode, p1.returncode)
```

### Reference cell (the Cityscapes LN-control runs, as executed on Kaggle protocol)

**Checklist.**
1. Notebook with **Accelerator = GPU T4 x2** (never P100 — see Troubleshooting).
2. Attach the `cityscapes-fine` dataset and the `pyrdc-code` dataset.
3. Unzip code: `!cp /kaggle/input/pyrdc-code/pyrdc_code.zip . && unzip -qo pyrdc_code.zip`
4. Run the cell below; **verify both logs print `params=259813`** before
   walking away (any other number = wrong flags, kill it).
5. Afterwards, download per the checklist in "After every session" below.

```python
import os, subprocess, shlex, time

ROOT   = "/kaggle/input/<your-cityscapes-dataset>"
COMMON = (f"--root {ROOT} --base-size 1024 512 --crop 256 256 "
          "--batch-size 4 --val-batch-size 1 --epochs 60 --eval-every 5 "
          "--num-workers 2 --out-dir /kaggle/working")

def launch(gpu, head, extra):
    cmd = (f"python experiments/run_cityscapes.py --head {head} {extra} {COMMON} "
           f"--results-csv /kaggle/working/cs_gpu{gpu}.csv")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    return subprocess.Popen(shlex.split(cmd), env=env,
                            stdout=open(f"/kaggle/working/gpu{gpu}.log", "w"),
                            stderr=subprocess.STDOUT)

p0 = launch(0, "pyramidal_single_level", "--channels 66 --level-norm --seeds 1")
p1 = launch(1, "pyramidal_single_level", "--channels 66 --level-norm --seeds 2")

def tail(path, n=2):
    try:
        with open(path) as f:
            return "".join(f.readlines()[-n:]).strip()
    except FileNotFoundError:
        return "(no log yet)"

while p0.poll() is None or p1.poll() is None:
    time.sleep(300)
    print("-" * 60, flush=True)
    print("[gpu0]", tail("/kaggle/working/gpu0.log"), flush=True)
    print("[gpu1]", tail("/kaggle/working/gpu1.log"), flush=True)
print("exit codes:", p0.returncode, p1.returncode)
```

**Analysis after merge:** `analyze.py cityscapes` picks the new dumps up with
no edits (the LN control appears as its own arm vs the canonical
un-normalised control). The decision rule: if the LN control's per-class
redistribution replicates at ρ≈−0.5 across three seeds, the paper's caveat
hardens into "normalisation accounts for roughly half"; if seeds 1–2 regress
to ρ≈0, the caveat softens to a one-seed anomaly and the signature belongs to
cross-scale sampling.

---

## Reference

### One-time setup

1. **Cityscapes** (license requires your own account — cannot be a public
   Kaggle dataset): download `leftImg8bit_trainvaltest.zip` (11 GB) and
   `gtFine_trainvaltest.zip` (241 MB) from cityscapes-dataset.com, extract so
   `leftImg8bit/{train,val}/...` and `gtFine/{train,val}/...` sit side by
   side, upload as a private Kaggle dataset (e.g. `cityscapes-fine`).
2. **ADE20K**: the SceneParse150 subset (`ADEChallengeData2016`, ~1.2 GB) —
   public Kaggle copies exist (e.g. awsaf's "ADE20K Dataset"). `--root` must
   be the directory *containing* `images/` and `annotations/`; public copies
   usually nest it one level down.
3. **Code**: from the project root, `zip -r pyrdc_code.zip pyrdc experiments
   -x "*__pycache__*"`, upload as a private dataset (e.g. `pyrdc-code`).
   Re-upload whenever `pyrdc/` or the runners change — a stale zip fails
   loudly on new flags (e.g. `--dataset`) but silently on behaviour changes.
   Current Kaggle-side version: 2026-08-16 (has `--dataset ade20k`, offset
   scaling, all fixes).

### Per-session workflow

Kaggle bills **session wall time**, not per-GPU time, so always pair two runs
on T4×2 (`CUDA_VISIBLE_DEVICES=0/1`) — two configs for one session's quota.
Kaggle's IPython rejects `!cmd &`, so drive both GPUs from a Python cell with
`subprocess.Popen` (the cell above is the template; swap `launch(...)` lines
and, for ADE20K, the `COMMON` line per the locked protocol below). The poll
loop doubles as a keep-alive for interactive runs; for detached commit
("Save & Run All") mode replace it with `p0.wait(); p1.wait()`.

### Locked protocols (do not change mid-study)

| | Cityscapes | ADE20K |
|---|---|---|
| flags | `--base-size 1024 512 --crop 256 256 --batch-size 4 --val-batch-size 1 --epochs 60 --eval-every 5` | `--dataset ade20k --base-size 512 512 --crop 256 256 --batch-size 4 --val-batch-size 1 --epochs 10 --eval-every 2` |
| steps | ~44.6k | ~50.5k (budget-matched to Cityscapes, not epoch-matched) |
| ~h per run (T4) | 5–6.5 | 6–8 |
| results CSV | `results/cityscapes.csv` | `results/ade20k.csv` (runner default) |

`pyramidal` / `pyramidal_attention` rows always take `--z-shared
--level-norm`; `pyramidal_single_level` takes neither unless the run *is* the
LN control.

**Params printed on the first log line must match** (kill the run otherwise):

| config | Cityscapes (19-ch head) | ADE20K (150-ch head) |
|---|---|---|
| plain ch64 / ch68 | 234 515 / 263 859 | — |
| single ch66 | 259 681 | 337 626 |
| single ch66 +LN | 259 813 | 337 758 |
| cheap-z ch64 (`--z-shared --level-norm`) | 258 470 | 334 057 |
| attention ch64 +zshared+LN | 260 201 | — |
| reduced ch64 +LN | 249 238 | — |

### After every session, download from `/kaggle/working`

1. `*_gpu*.csv` → append data rows to `results/{cityscapes,ade20k}.csv`
2. `per_image/*.csv` **and** `per_image/*_per_class.csv` → `kaggle_artifacts/per_image/`
3. `tb/*` → `kaggle_artifacts/tb_logs/` (**load-bearing** — the train-loss
   analyses read these, they are not archival)
4. `*.pth` → `kaggle_artifacts/`

Then `micromamba run -n ml_env python3 experiments/analyze.py cityscapes`
picks everything up.

### Troubleshooting

- **P100 does not work** with Kaggle's stock PyTorch (sm_60 kernels missing;
  every CUDA op fails with `no kernel image is available`). Use T4×2. If P100
  is unavoidable: `pip install torch==2.4.1 torchvision==0.19.1 --index-url
  https://download.pytorch.org/whl/cu118`, restart kernel.
- **`CUBLAS_STATUS_ALLOC_FAILED` on pyramid heads** at val resolution: the
  sampler reserves several GB through PyTorch's caching allocator before the
  first matmul, leaving cuBLAS nothing. `warmup_cuda_handles()` in
  `run_cityscapes.py` prevents it; if it ever recurs, add
  `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"` to the Popen env.
- **Session cap**: if an ADE run dies at the wall-time limit, drop to
  `--epochs 8` and relock the protocol.

### Running locally instead (24-core CPU box, for calibration)

plain_conv ch68 11.7 h (measured); reduced+LN ch70 14.1 h; single ch66
42.5 h (+LN 43.8 h); cheap-z ch64 69.4 h; attention ch64 82.2 h. Only
plain_conv is practical locally — the pyramid heads are 5–10× slower than a
T4 because the sampler is bandwidth-bound.

---

## Completed work (numbers live in RESULTS.md)

### Cityscapes (sessions 1–3 + plan-sessions 1–4; final seed counts)

| head | params | seeds | mIoU | outcome |
|---|---|---|---|---|
| plain_conv ch64 | 234 515 | 0 | .2811 | floor |
| plain_conv ch68 | 263 859 | 0 (local) | .2820 | width worthless → deformability gain is real (RESULTS §5) |
| single ch66 | 259 681 | 0–3 | .3693–.3769 | canonical control |
| single ch66 +LN | 259 813 | 0 | .3748 | **reproduces ~half the redistribution → NEXT SESSION** |
| cheap-z ch64 | 258 470 | 0–3 | .3709–.3772 | dataset-mIoU tie (p=.47); **fits better across seeds (p=.029)**; loses per-image mIoU (p=.007) |
| attention ch64 +zshared+LN | 260 201 | 0–2 | .3667–.3823 | aggregate tie; redistribution ρ=−.61…−.66 all seeds |
| reduced ch64 +LN | 249 238 | 0 | .2911 | does not scale to the wide regime |

Redistribution replicated on all 7 pyramid seeds (cross-seed agreement
ρ=+.818/+.872); details in RESULTS §5, figure `fig2`.

### ADE20K (plan-sessions A1–A3 — the regime test, confirmed)

| head | params | seeds | mIoU | pixel-acc |
|---|---|---|---|---|
| cheap-z ch64 | 334 057 | 0–2 | **.0549 ± .0007** | **.5035** |
| single ch66 | 337 626 | 0–2 | .0507 ± .0007 | .4796 |

Cheap-z won every metric on every seed (pixel-acc +2.4 pts p<1e-4; per-image
mIoU +.030, 86 % of images; train −.181 p=.006; val CE −.110) and the
redistribution signature disappeared (ρ≈0). RESULTS §5b; paper
§"ADE20K: the data-rich regime confirms the conversion". The only further
ADE run at this model scale is the LN control (see STATUS).
