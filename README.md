# DSD-Restore V1

This repository contains the V1 implementation of **DSD-Restore: Dense-Sparse Degradation Disentangled Restoration**.

V1 follows the project documents:

- Shared feature encoder
- Degradation tokenizer
- Dense degradation expert for dust / sand / haze / low-light / color-cast
- Sparse occlusion expert for rain / raindrop / snow / local occlusion
- Simple dense-sparse fusion
- Restoration decoder
- V1 losses: reconstruction, SSIM, frequency, color, and degradation classification

V2 modules such as the compositional prototype bank and physics-frequency fusion are intentionally not included here.

## Quick Smoke Test

```bash
python scripts/smoke_test_v1.py
```

## Train

Prepare paired data as:

```text
data/mixed/train/input/*.png
data/mixed/train/gt/*.png
data/mixed/val/input/*.png
data/mixed/val/gt/*.png
```

Optional metadata labels can be stored in `metadata.json`:

```json
{
  "0001.png": {"dust": 0.6, "haze": 0.3, "lowlight": 0.5, "rain": 0.2, "snow": 0.0}
}
```

Run:

```bash
python train.py --config configs/train_v1_minimal.yaml
```

## Test

```bash
python test.py --config configs/train_v1_minimal.yaml --checkpoint experiments/v1_minimal/checkpoints/best.pt
```

## Inference

```bash
python infer_real.py --checkpoint experiments/v1_minimal/checkpoints/best.pt --input data/real --output results/real_v1
```
