# DSD-Restore V2 Factor-Spatial Routing Screen

V2 is a controlled 15k validation screen motivated by the D0 and D1 failures.
D0 redistributed quality between tasks without improving triple degradation,
while D1 showed that generic shared capacity cannot replace expert-specific
processing. V2 therefore changes routing granularity instead of data balance or
expert depth.

## Method

V1 makes one global dense/sparse fusion decision for an entire image. V2 keeps
that stable fusion path and inserts a factor-spatial router immediately before
it:

1. The shared bottleneck is projected to a 32-channel spatial query.
2. Every tokenizer output supplies a factor key and a channel value: five dense
   factors (dust, sand, haze, lowlight, colorcast) and four sparse factors
   (rain, raindrop, snow, occlusion).
3. Query-key correlation produces one bottleneck-resolution spatial gate per
   factor. Token presence logits suppress absent factors.
4. The factor maps and token values form separate dense and sparse spatial
   modulation fields, which reweight the corresponding expert feature at each
   location.
5. Channel modulation scales start at zero, so a Stage 2 checkpoint produces
   exactly the legacy output before V2 training begins.

The router adds about 28.1k parameters (roughly 0.28% of V1), well inside the
3% screen budget. The original encoder, tokenizer, prototypes, 4/4 expert
depths, fusion and decoder remain unchanged.

## Frozen Controls

- Stage 2 initialization: `v1_stage2_joint/best.pth`
- seed: 42
- iterations: 15,000
- warmup: 500 iterations
- Stage 3 conditional composite mixture: 50% dense+dense, 40% dense+sparse,
  10% triple
- A1 auxiliary losses: classification, prototype and sparse-mask losses off
- validation-only model selection; do not run `test.py` during the screen

## Preflight

```bash
cd /root/autodl-tmp/DSD-Restore
python scripts/smoke_test_v1.py
python scripts/audit_training_data.py \
  --config configs/screen_v2_factor_spatial_routing.yaml
python scripts/smoke_test_data_pipeline.py \
  --config configs/screen_v2_factor_spatial_routing.yaml --num-workers 8
```

Checkpoint initialization should report more than 99% compatible coverage and
missing keys only under `factor_router.*`.

## Train

```bash
python train.py --config configs/screen_v2_factor_spatial_routing.yaml
```

Do not add `--resume` for the first V2 screen. Checkpoints and TensorBoard data
are isolated at:

```text
/root/autodl-tmp/DSD-Restor-checkpoints/screen_v2_factor_spatial_routing
/root/tf-logs/screen_v2_factor_spatial_routing
```

TensorBoard exposes `factor_routing/*` diagnostics for factor-map mean, spread,
coverage, task-conditional activation and modulation RMS.

After training, add the V2 best checkpoint to the existing comparison table:

```bash
python scripts/summarize_v11_screens.py \
  --output-csv results/v2_factor_screen_summary.csv \
  | tee results/v2_factor_screen_summary.txt
```

## Go/No-Go Gate

Compare the best validation checkpoint with `v11_screen_a1_aux_off`:

- triple PSNR improves by at least 0.15 dB;
- macro PSNR improves by at least 0.08 dB, or at minimum does not drop by more
  than 0.02 dB while satisfying every other gate;
- no two-degradation task drops by more than 0.15 dB;
- at least eight of the thirteen combinations do not decrease;
- LPIPS and DISTS do not materially worsen;
- factor maps show non-collapsed spatial variation rather than uniform 0/1
  activation.

Only a passing V2 checkpoint should be evaluated on the frozen 1k test set.
