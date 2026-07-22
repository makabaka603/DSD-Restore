# V1 Implementation Map

This file maps the V1 code to the Word documents and model figures.

## V1 Scope

V1 validates the core hypothesis:

> Dense degradations and sparse occlusion degradations should be disentangled before restoration.

Included:

- Shared encoder
- Degradation tokenizer
- Dense degradation expert
- Sparse occlusion expert
- Compositional prototype bank
- Simple dense-sparse fusion
- Restoration decoder
- V1 loss set

Excluded until V2/V3:

- Physics-frequency guided fusion
- Real no-GT re-degradation consistency
- Downstream task loss

## Module Mapping

| Document / Figure module | Code |
| --- | --- |
| Input degraded image `I` | `train.py`, `datasets/multi_source_restoration_dataset.py` |
| Online composite degradation | `synthesize_composite_degradation`, dense first then sparse |
| V1 category-balanced sampling | `configs/train_v1_minimal.yaml`, `build_balanced_sampler` |
| Shared Feature Encoder `E` | `models/backbone/nafnet_backbone.py::NAFNetSharedEncoder` |
| Multi-scale features `F1...F4` | `NAFNetSharedEncoder.forward` |
| Shallow image statistics `phi(I)` | `models/tokenizer/degradation_tokenizer.py::_image_statistics` |
| Degradation Tokenizer `T` | `models/tokenizer/degradation_tokenizer.py::DegradationTokenizer` |
| Dense token `z_d` | `outputs["dense_tokens"]`, shape `B x 5 x D` |
| Sparse token `z_s` | `outputs["sparse_tokens"]`, shape `B x 4 x D` |
| Dense Expert | `models/experts/dense_expert.py::DenseDegradationExpert` |
| Sparse Expert | `models/experts/sparse_expert.py::SparseOcclusionExpert` |
| Rain/snow sparse mask | `outputs["sparse_mask"]` |
| Compositional Prototype Bank | `models/prototype/compositional_prototype_bank.py` |
| Simple Fusion | `models/fusion/simple_fusion.py::SimpleDenseSparseFusion` |
| Restoration Decoder `D` | `models/backbone/nafnet_backbone.py::NAFNetRestorationDecoder` |
| Restored image `J_hat` | `outputs["restored"]` |

## Loss Mapping

| V1 loss | Code |
| --- | --- |
| `L_rec` | `losses/v1_losses.py`, L1 reconstruction |
| `L_ssim` | `losses/v1_losses.py::ssim_loss` |
| `L_freq` | `losses/v1_losses.py::frequency_loss` |
| `L_color` | `losses/v1_losses.py::color_loss` |
| `L_cls` | BCE over dense/sparse degradation labels |
| `L_proto` | Prototype diversity + multi-label composition alignment |
| `L_sparse` | Sparse-mask presence, coverage, and TV regularization |

## Labels

Dense labels:

```text
dust, sand, haze, lowlight, colorcast
```

Sparse labels:

```text
rain, raindrop, snow, occlusion
```

Synthetic composite samples use continuous labels sampled with the degradation
parameters. Paired single-task sources retain binary source labels.

Metadata example:

```json
{
  "0001.png": {
    "dust": 0.60,
    "sand": 0.00,
    "haze": 0.30,
    "lowlight": 0.50,
    "colorcast": 0.40,
    "rain": 0.20,
    "raindrop": 0.00,
    "snow": 0.00,
    "occlusion": 0.00
  }
}
```
