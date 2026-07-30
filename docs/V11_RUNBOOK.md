# DSD-Restore V1.1 Validation-First Runbook

V1.1 is a controlled repair of the failure modes observed in V1:

- independent dense and sparse gates instead of a complementary scalar;
- presence-weighted token pooling;
- independent multi-label prototype activation;
- top-k sparse-mask presence supervision that rejects one-pixel collapse;
- positive-weighted degradation classification;
- late Stage 3 auxiliary-loss decay;
- dataset-level routing variance and active-label tokenizer metrics.

The original V1 configs retain their legacy defaults and remain reproducible.

## 1. Verify Code Before Training

From the repository root:

```bash
python scripts/smoke_test_v1.py
python scripts/smoke_test_naf_baselines.py
python train.py --config configs/smoke_v11.yaml
```

The two code checks must pass, and the 20-iteration data-pipeline run must
create `best.pth` and `latest.pth`. The V1.1 code smoke test reports compatible
loading of more than 95% of the old V1 state by element count.

## 2. Freeze the Test Set

Do not run `test.py` while selecting V1.1 components. Use only the validation
metrics produced by `train.py`. Keep the existing V1, NAF32 and NAF52 test
CSVs unchanged.

## 3. Run Six Short Screens

Run one command at a time:

```bash
python train.py --config configs/screen_v11_control.yaml
python train.py --config configs/screen_v11_aux_off.yaml
python train.py --config configs/screen_v11_dual_gate.yaml
python train.py --config configs/screen_v11_weighted_tokens.yaml
python train.py --config configs/screen_v11_multilabel_proto.yaml
python train.py --config configs/screen_v11_full.yaml
```

All six screens:

- start from the same frozen V1 Stage 2 `best.pth`;
- use seed 42 and the same Stage 3 data;
- run for 15,000 iterations;
- write to separate checkpoint and TensorBoard directories;
- never initialize from another screen.

The architecture-changing screens print compatible checkpoint coverage. The
legacy scalar gate is migrated to independent logits `[z, -z]`, which preserves
the old `dense=sigmoid(z), sparse=1-dense` function exactly at initialization.
Stop if coverage is below 95%, if the reported source stage is not `stage2`, or
if the migration message is missing.

## 3.1 Run the Corrected B1/B2 Screens

The original A0-A5 outputs are frozen evidence and must not be overwritten.
Run only these two follow-up screens after updating the code:

```bash
python train.py --config configs/screen_v11b_dual_gate_migrated.yaml
python train.py --config configs/screen_v11b_full_migrated.yaml
```

Both runs still start from the same frozen V1 Stage 2 `best.pth`. B1 repeats A2
with function-preserving gate migration; B2 repeats A5 with the same migration.
Their checkpoints, experiment outputs and TensorBoard runs use new directories.

At startup both commands must report:

```text
Migrated the legacy complementary fusion gate to equivalent independent logits [z, -z].
Initialized ... (source stage: stage2); optimizer and iteration reset.
```

Do not use `--resume`. Compare B1/B2 against the existing A0/A1/A2/A5 results
using validation metrics only.

## 4. Summarize Validation Checkpoints

```bash
python scripts/summarize_v11_screens.py
```

This writes:

```text
results/v11_screen_summary.csv
```

Use TensorBoard to additionally compare:

```text
val/psnr
val/ssim
val/lpips
val/dists
val_diagnostics/routing/dense_gate_mean
val_diagnostics/routing/dense_gate_std
val_diagnostics/routing/sparse_gate_mean
val_diagnostics/routing/sparse_gate_std
val_diagnostics/routing/*_gate_by_task/*
val_diagnostics/tokenizer/active_macro_recall
val_diagnostics/tokenizer/dense/*/recall
val_diagnostics/tokenizer/sparse/*/recall
val_diagnostics/mask/coverage_by_task/*
val_diagnostics/prototype/*/active_count
validation_images/fixed_panel
```

Select a component only when it improves the fixed validation set. In
particular, check low-light combinations, triple degradations, rain/snow mask
coverage and task-dependent gates.

## 5. Train the Selected Full V1.1

The supplied formal configs represent the complete A5 candidate. Run them only
after the short screen supports that choice:

```bash
python train.py --config configs/train_v11_stage1.yaml
python train.py --config configs/train_v11_stage2.yaml
python train.py --config configs/train_v11_stage3.yaml
```

Verify each checkpoint before starting the next stage:

```bash
ls -lh /root/autodl-tmp/DSD-Restor-checkpoints/v11_stage1_single/best.pth
ls -lh /root/autodl-tmp/DSD-Restor-checkpoints/v11_stage2_joint/best.pth
ls -lh /root/autodl-tmp/DSD-Restor-checkpoints/v11_stage3_composite/best.pth
```

Formal Stage 2 and Stage 3 use strict checkpoint loading. Do not initialize
them from V1 checkpoints.

## 6. Evaluate the Frozen Test Set Once

Only after the full three-stage run:

```bash
python test.py \
  --config configs/train_v11_stage3.yaml \
  --checkpoint /root/autodl-tmp/DSD-Restor-checkpoints/v11_stage3_composite/best.pth \
  --test-input-dir datasets/Synthetic-Mixed-Test-1K/input \
  --test-gt-dir datasets/Synthetic-Mixed-Test-1K/gt \
  --test-metadata datasets/Synthetic-Mixed-Test-1K/metadata.json \
  --output-csv results/baselines/dsd_v11_stage3_test.csv \
  2>&1 | tee results/baselines/dsd_v11_stage3_test.log
```

Compare V1.1 against the frozen NAF52 Stage 3 and V1 results. Do not tune V1.1
after reading this test result.
