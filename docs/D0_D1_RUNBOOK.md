# DSD-Restore D0/D1 Controlled Screens

D0 and D1 are independent 15k validation screens. Both start from the same
frozen V1 Stage 2 `best.pth`, use seed 42, keep the A1 auxiliary losses off,
and must be selected on validation only. Do not run `test.py` for either screen.

## D0: Composite Sampling Balance

D0 changes no model or loss code. It replaces the Stage 3 conditional mixture
of 50% dense+dense, 40% dense+sparse and 10% triple with:

- dense+dense: 40%
- dense+sparse: 35%
- triple: 25%

The three logical sources reuse the same frozen `SyntheticCompositeTrain`
files and filter them by metadata group.

Run:

```bash
python train.py --config configs/screen_d0_triple_balance.yaml
```

D0 passes only if, relative to A1:

- triple PSNR improves by at least 0.15 dB;
- macro PSNR drops by no more than 0.02 dB;
- no two-degradation task drops by more than 0.15 dB.

## D1: Shared Capacity Reallocation

D1 keeps the original 50%/40%/10% Stage 3 mixture. It reduces each bottleneck
expert from four blocks to two and adds four shared NAFBlocks before
tokenization and expert routing. NAFBlock residual scales start at zero, so the
new shared path is an identity mapping at checkpoint initialization. The two
removed blocks from each expert are intentionally not loaded.

The default model changes from about 10.056M to 9.792M parameters, a 2.62%
reduction, so D1 remains within the prescribed 3% parameter budget.

Run:

```bash
python train.py --config configs/screen_d1_shared_capacity.yaml
```

Startup should report compatible Stage 2 checkpoint coverage of about 81%.
Missing parameters must be restricted to `shared_bottleneck.*`.

D1 passes only if, relative to A1:

- macro PSNR improves by at least 0.08 dB;
- triple PSNR does not decrease;
- LPIPS and DISTS do not materially worsen;
- at least eight degradation combinations do not decrease.

## Required Order

1. Run `python scripts/smoke_test_v1.py` after updating the repository.
2. Audit both configurations against the server datasets:

```bash
python scripts/audit_training_data.py --config configs/screen_d0_triple_balance.yaml
python scripts/audit_training_data.py --config configs/screen_d1_shared_capacity.yaml
python scripts/smoke_test_data_pipeline.py \
  --config configs/screen_d0_triple_balance.yaml --num-workers 8
python scripts/smoke_test_data_pipeline.py \
  --config configs/screen_d1_shared_capacity.yaml --num-workers 8
```

3. Run D0 by itself.
4. Run D1 by itself; do not add `--resume`.
5. Summarize the best validation checkpoints:

```bash
python scripts/summarize_v11_screens.py \
  | tee results/d0_d1_summary.log
```

The new checkpoints and TensorBoard runs are isolated at:

```text
/root/autodl-tmp/DSD-Restor-checkpoints/screen_d0_triple_balance
/root/autodl-tmp/DSD-Restor-checkpoints/screen_d1_shared_capacity
/root/tf-logs/screen_d0_triple_balance
/root/tf-logs/screen_d1_shared_capacity
```

If both pass, test a combined D0+D1 screen next. If both fail, stop extending
the current bottleneck-expert design and move to factor-level spatial routing.
