# Internal NAFNet-Style Baseline Runbook

This runbook evaluates two controlled baselines built from the NAFNet-style
encoder and decoder already present in this repository:

- `NAF32`: 3.928355M parameters, the exact shared-backbone-width ablation.
- `NAF52`: 10.278375M parameters, capacity-matched to DSD-Restore V1
  (10.056040M parameters).

These are controlled internal baselines, not the unmodified official NAFNet
implementation. Use the label `NAFNet-style` in tables and prose.

## 1. Verify the Code Without Loading a Dataset

From the repository root:

```bash
python scripts/smoke_test_naf_baselines.py
python scripts/smoke_test_v1.py
```

Expected baseline output includes:

```text
NAF32: params=3.928355M
NAF52: params=10.278375M
NAF baseline smoke test passed
```

The V1 smoke test must also continue to pass.

## 2. Run Short Data-Pipeline Training Tests

Run both 20-iteration jobs before any formal training:

```bash
python train.py --config configs/smoke_naf32.yaml
python train.py --config configs/smoke_naf52.yaml
```

Confirm that each run:

- reaches iteration 20;
- completes validation at iterations 10 and 20;
- creates `best.pth` and `latest.pth`;
- logs reconstruction, SSIM, frequency, color and system diagnostics;
- does not raise missing tokenizer, prototype or sparse-mask key errors.

The smoke checkpoints are disposable and must not be used as initializers for
formal training.

## 3. Train NAF32 Through the Same Three Stages

```bash
python train.py --config configs/train_naf32_stage1.yaml
python train.py --config configs/train_naf32_stage2.yaml
python train.py --config configs/train_naf32_stage3.yaml
```

Before starting the next stage, verify the previous best checkpoint:

```bash
ls -lh /root/autodl-tmp/DSD-Restor-checkpoints/naf32_stage1/best.pth
ls -lh /root/autodl-tmp/DSD-Restor-checkpoints/naf32_stage2/best.pth
ls -lh /root/autodl-tmp/DSD-Restor-checkpoints/naf32_stage3/best.pth
```

Stage 2 must report that it initialized from `naf32_stage1`; Stage 3 must
report that it initialized from `naf32_stage2`.

## 4. Train NAF52 Through the Same Three Stages

```bash
python train.py --config configs/train_naf52_stage1.yaml
python train.py --config configs/train_naf52_stage2.yaml
python train.py --config configs/train_naf52_stage3.yaml
```

Verify:

```bash
ls -lh /root/autodl-tmp/DSD-Restor-checkpoints/naf52_stage1/best.pth
ls -lh /root/autodl-tmp/DSD-Restor-checkpoints/naf52_stage2/best.pth
ls -lh /root/autodl-tmp/DSD-Restor-checkpoints/naf52_stage3/best.pth
```

Do not initialize an NAF52 stage from an NAF32 checkpoint. Their channel
dimensions are different.

## 5. Evaluate Stage 2 and Stage 3 With One Metric Implementation

Use the same frozen test set and `test.py` for every model. Do not change
training settings after inspecting these test results.

### NAF32 Stage 2

```bash
python test.py \
  --config configs/train_naf32_stage2.yaml \
  --checkpoint /root/autodl-tmp/DSD-Restor-checkpoints/naf32_stage2/best.pth \
  --test-input-dir datasets/Synthetic-Mixed-Test-1K/input \
  --test-gt-dir datasets/Synthetic-Mixed-Test-1K/gt \
  --test-metadata datasets/Synthetic-Mixed-Test-1K/metadata.json \
  --output-csv results/baselines/naf32_stage2_test.csv
```

### NAF32 Stage 3

```bash
python test.py \
  --config configs/train_naf32_stage3.yaml \
  --checkpoint /root/autodl-tmp/DSD-Restor-checkpoints/naf32_stage3/best.pth \
  --test-input-dir datasets/Synthetic-Mixed-Test-1K/input \
  --test-gt-dir datasets/Synthetic-Mixed-Test-1K/gt \
  --test-metadata datasets/Synthetic-Mixed-Test-1K/metadata.json \
  --output-csv results/baselines/naf32_stage3_test.csv
```

### NAF52 Stage 2

```bash
python test.py \
  --config configs/train_naf52_stage2.yaml \
  --checkpoint /root/autodl-tmp/DSD-Restor-checkpoints/naf52_stage2/best.pth \
  --test-input-dir datasets/Synthetic-Mixed-Test-1K/input \
  --test-gt-dir datasets/Synthetic-Mixed-Test-1K/gt \
  --test-metadata datasets/Synthetic-Mixed-Test-1K/metadata.json \
  --output-csv results/baselines/naf52_stage2_test.csv
```

### NAF52 Stage 3

```bash
python test.py \
  --config configs/train_naf52_stage3.yaml \
  --checkpoint /root/autodl-tmp/DSD-Restor-checkpoints/naf52_stage3/best.pth \
  --test-input-dir datasets/Synthetic-Mixed-Test-1K/input \
  --test-gt-dir datasets/Synthetic-Mixed-Test-1K/gt \
  --test-metadata datasets/Synthetic-Mixed-Test-1K/metadata.json \
  --output-csv results/baselines/naf52_stage3_test.csv
```

### DSD-Restore V1 Stage 3

```bash
python test.py \
  --config configs/train_v1_stage3.yaml \
  --checkpoint /root/autodl-tmp/DSD-Restor-checkpoints/v1_stage3_composite/best.pth \
  --test-input-dir datasets/Synthetic-Mixed-Test-1K/input \
  --test-gt-dir datasets/Synthetic-Mixed-Test-1K/gt \
  --test-metadata datasets/Synthetic-Mixed-Test-1K/metadata.json \
  --output-csv results/baselines/dsd_v1_stage3_test.csv
```

## 6. Fill the Controlled Comparison Table

| Model | Params | Macro PSNR | Macro SSIM | Macro LPIPS | Macro DISTS |
| --- | ---: | ---: | ---: | ---: | ---: |
| NAF32 Stage 3 | 3.928M | | | | |
| NAF52 Stage 3 | 10.278M | | | | |
| DSD-Restore V1 Stage 3 | 10.056M | 26.4879 | 0.8815 | 0.0919 | 0.1007 |

Also compare every low-light combination, every triple-degradation
combination, and the Stage 2-to-Stage 3 change.

Interpret the first-seed experiment as follows:

- DSD only beats NAF32: the complete system helps, but capacity is confounded.
- DSD beats NAF52 by about 0.3 dB or more: strong architectural evidence.
- Similar PSNR with materially better LPIPS/DISTS: a perceptual-quality result.
- DSD and NAF52 differ by less than about 0.1 dB: fix V1 before starting V2.
- DSD loses to NAF52: investigate routing, mask collapse and low-light recovery.

Run the first comparison with seed 42. Only after the direction is confirmed,
repeat DSD and NAF52 with at least two additional predetermined seeds and report
mean plus standard deviation.

## 7. Reproduce Official NAFNet Only After the Controlled Study

Keep the official repository outside this Git working tree:

```bash
cd /root/autodl-tmp
git clone https://github.com/megvii-research/NAFNet.git NAFNet-official
cd NAFNet-official
git rev-parse HEAD
```

Record the exact commit. For the final paper comparison:

- train the official architecture on the same training pairs;
- select checkpoints only on validation data;
- do not tune on `Synthetic-Mixed-Test-1K`;
- evaluate through the same PSNR/SSIM/LPIPS/DISTS implementation;
- report parameter count, FLOPs, latency and memory;
- do not add another outer residual if the official implementation already
  applies its own residual connection.

Use two distinct table labels:

- `NAFNet-style (controlled)` for the internal NAF32/NAF52 models;
- `NAFNet (official)` for the external unmodified architecture.
