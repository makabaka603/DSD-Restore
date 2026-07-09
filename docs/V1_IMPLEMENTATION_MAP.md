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
- Simple dense-sparse fusion
- Restoration decoder
- V1 loss set

Excluded until V2/V3:

- Compositional prototype bank
- Physics-frequency guided fusion
- Real no-GT re-degradation consistency
- Downstream task loss

## Module Mapping

| Document / Figure module | Code |
| --- | --- |
| Input degraded image `I` | `train.py`, `datasets/paired_restoration_dataset.py` |
| Shared Feature Encoder `E` | `models/backbone/simple_restoration_backbone.py::SharedEncoder` |
| Multi-scale features `F1...F4` | `SharedEncoder.forward` |
| Shallow image statistics `phi(I)` | `models/tokenizer/degradation_tokenizer.py::_image_statistics` |
| Degradation Tokenizer `T` | `models/tokenizer/degradation_tokenizer.py::DegradationTokenizer` |
| Dense token `z_d` | `outputs["dense_tokens"]`, shape `B x 5 x D` |
| Sparse token `z_s` | `outputs["sparse_tokens"]`, shape `B x 4 x D` |
| Dense Expert | `models/experts/dense_expert.py::DenseDegradationExpert` |
| Transmission / illumination / color-cast heads | `DenseDegradationExpert` auxiliary heads |
| Sparse Expert | `models/experts/sparse_expert.py::SparseOcclusionExpert` |
| Rain/snow sparse mask | `outputs["sparse_mask"]` |
| Simple Fusion | `models/fusion/simple_fusion.py::SimpleDenseSparseFusion` |
| Restoration Decoder `D` | `models/backbone/simple_restoration_backbone.py::RestorationDecoder` |
| Restored image `J_hat` | `outputs["restored"]` |

## Loss Mapping

| V1 loss | Code |
| --- | --- |
| `L_rec` | `losses/v1_losses.py`, L1 reconstruction |
| `L_ssim` | `losses/v1_losses.py::ssim_loss` |
| `L_freq` | `losses/v1_losses.py::frequency_loss` |
| `L_color` | `losses/v1_losses.py::color_loss` |
| `L_cls` | BCE over dense/sparse degradation labels |

## Labels

Dense labels:

```text
dust, sand, haze, lowlight, colorcast
```

Sparse labels:

```text
rain, raindrop, snow, occlusion
```

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
