# DSD-Restore V2-lite C1 Validation Screen

C1 tests one structural change only: identity-initialized dense/sparse
conditioning at encoder scales `f2`, `f3` and `f4`.

The control remains A1:

- V1 NAFNet-width-32 backbone and decoder;
- legacy complementary bottleneck fusion;
- mean token pooling and softmax prototypes;
- classification, prototype and sparse-mask auxiliary losses disabled;
- the same frozen V1 Stage 2 `best.pth`, seed 42, data and 15k schedule.

The adapters split each feature into parameter-free low- and high-frequency
components. Dense and sparse prompts generate channel gates, followed by a
small bottleneck residual. Its final projection is zero-initialized, so C1 is
exactly equivalent to V1 before the first optimizer update.

## 1. Update and Verify

```bash
cd /root/autodl-tmp/DSD-Restore
python scripts/smoke_test_v1.py
```

The smoke test must report:

```text
V2-lite C1 identity initialization and gradients: passed
```

## 2. Run C1

```bash
python train.py --config configs/screen_v2lite_c1_multiscale.yaml
```

Do not add `--resume`. Startup must report compatible Stage 2 initialization
above 98%. Missing parameters must be restricted to
`multiscale_conditioner.*`; any other missing parameter is an error.

C1 writes only to:

```text
experiments/v2lite_screen_c1_multiscale
/root/autodl-tmp/DSD-Restor-checkpoints/v2lite_screen_c1_multiscale
/root/tf-logs/v2lite_screen_c1_multiscale
```

## 3. Compare on Validation Only

```bash
python scripts/summarize_v11_screens.py
```

Compare C1 against A1 (`26.5218` validation PSNR in the frozen seed-42 screen)
and inspect:

```text
val/psnr
val/ssim
val/lpips
val/dists
val_by_task/*/psnr
val_diagnostics/multiscale/f2/residual_rms
val_diagnostics/multiscale/f3/residual_rms
val_diagnostics/multiscale/f4/residual_rms
val_diagnostics/multiscale/*/dense_gate_*
val_diagnostics/multiscale/*/sparse_gate_*
```

Do not run `test.py`. Continue only if C1 improves the fixed validation set,
especially low-light and triple-degradation groups.
