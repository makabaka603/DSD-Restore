from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


DENSE_NAMES = ("dust", "sand", "haze", "lowlight", "colorcast")
SPARSE_NAMES = ("rain", "raindrop", "snow", "occlusion")


def _safe_div(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return numerator / denominator.clamp_min(1.0)


def _multilabel_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    names: Sequence[str],
    prefix: str,
) -> dict[str, torch.Tensor]:
    predicted = torch.sigmoid(logits) >= 0.5
    target = labels > 0
    tp = (predicted & target).sum(dim=0).float()
    fp = (predicted & ~target).sum(dim=0).float()
    fn = (~predicted & target).sum(dim=0).float()
    return multilabel_metrics_from_counts(tp, fp, fn, names, prefix)


def multilabel_metrics_from_counts(
    tp: torch.Tensor,
    fp: torch.Tensor,
    fn: torch.Tensor,
    names: Sequence[str],
    prefix: str,
) -> dict[str, torch.Tensor]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)
    active = (tp + fn) > 0
    if bool(active.any()):
        active_precision = precision[active].mean()
        active_recall = recall[active].mean()
        active_f1 = f1[active].mean()
    else:
        active_precision = precision.new_zeros(())
        active_recall = recall.new_zeros(())
        active_f1 = f1.new_zeros(())
    result = {
        f"{prefix}/macro_precision": precision.mean(),
        f"{prefix}/macro_recall": recall.mean(),
        f"{prefix}/macro_f1": f1.mean(),
        f"{prefix}/active_macro_precision": active_precision,
        f"{prefix}/active_macro_recall": active_recall,
        f"{prefix}/active_macro_f1": active_f1,
        f"{prefix}/active_class_count": active.sum().float(),
    }
    for index, name in enumerate(names):
        result[f"{prefix}/{name}/precision"] = precision[index]
        result[f"{prefix}/{name}/recall"] = recall[index]
        result[f"{prefix}/{name}/f1"] = f1[index]
    return result


class MultilabelAccumulator:
    """Accumulate dataset-level tokenizer counts without retaining predictions."""

    def __init__(self) -> None:
        self.dense_tp = torch.zeros(len(DENSE_NAMES), dtype=torch.float64)
        self.dense_fp = torch.zeros(len(DENSE_NAMES), dtype=torch.float64)
        self.dense_fn = torch.zeros(len(DENSE_NAMES), dtype=torch.float64)
        self.sparse_tp = torch.zeros(len(SPARSE_NAMES), dtype=torch.float64)
        self.sparse_fp = torch.zeros(len(SPARSE_NAMES), dtype=torch.float64)
        self.sparse_fn = torch.zeros(len(SPARSE_NAMES), dtype=torch.float64)

    @staticmethod
    def _counts(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, ...]:
        predicted = (torch.sigmoid(logits) >= 0.5).detach().cpu()
        target = (labels > 0).detach().cpu()
        tp = (predicted & target).sum(dim=0).to(torch.float64)
        fp = (predicted & ~target).sum(dim=0).to(torch.float64)
        fn = (~predicted & target).sum(dim=0).to(torch.float64)
        return tp, fp, fn

    def update(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> None:
        dense = self._counts(outputs["dense_logits"], batch["dense_label"])
        sparse = self._counts(outputs["sparse_logits"], batch["sparse_label"])
        self.dense_tp += dense[0]
        self.dense_fp += dense[1]
        self.dense_fn += dense[2]
        self.sparse_tp += sparse[0]
        self.sparse_fp += sparse[1]
        self.sparse_fn += sparse[2]

    def compute(self) -> dict[str, torch.Tensor]:
        result = multilabel_metrics_from_counts(
            self.dense_tp, self.dense_fp, self.dense_fn, DENSE_NAMES, "tokenizer/dense"
        )
        result.update(
            multilabel_metrics_from_counts(
                self.sparse_tp,
                self.sparse_fp,
                self.sparse_fn,
                SPARSE_NAMES,
                "tokenizer/sparse",
            )
        )
        dense_f1 = result["tokenizer/dense/macro_f1"]
        sparse_f1 = result["tokenizer/sparse/macro_f1"]
        result["tokenizer/macro_precision"] = 0.5 * (
            result["tokenizer/dense/macro_precision"]
            + result["tokenizer/sparse/macro_precision"]
        )
        result["tokenizer/macro_recall"] = 0.5 * (
            result["tokenizer/dense/macro_recall"]
            + result["tokenizer/sparse/macro_recall"]
        )
        result["tokenizer/macro_f1"] = 0.5 * (dense_f1 + sparse_f1)
        result["tokenizer/active_macro_precision"] = 0.5 * (
            result["tokenizer/dense/active_macro_precision"]
            + result["tokenizer/sparse/active_macro_precision"]
        )
        result["tokenizer/active_macro_recall"] = 0.5 * (
            result["tokenizer/dense/active_macro_recall"]
            + result["tokenizer/sparse/active_macro_recall"]
        )
        result["tokenizer/active_macro_f1"] = 0.5 * (
            result["tokenizer/dense/active_macro_f1"]
            + result["tokenizer/sparse/active_macro_f1"]
        )
        return result


def _prototype_stats(
    weights: torch.Tensor,
    prototypes: torch.Tensor,
    names: Sequence[str],
    family: str,
) -> dict[str, torch.Tensor]:
    probabilities = weights.float().clamp_min(1e-8)
    entropy = -(probabilities * probabilities.log()).sum(dim=1)
    entropy = entropy / torch.log(
        torch.tensor(float(probabilities.shape[1]), device=probabilities.device)
    )
    normalized = F.normalize(prototypes.float(), dim=-1)
    similarity = normalized @ normalized.transpose(0, 1)
    if similarity.shape[0] > 1:
        offdiag = similarity[~torch.eye(
            similarity.shape[0], device=similarity.device, dtype=torch.bool
        )]
        offdiag_mean = offdiag.mean()
        offdiag_abs_mean = offdiag.abs().mean()
    else:
        offdiag_mean = similarity.new_zeros(())
        offdiag_abs_mean = similarity.new_zeros(())
    result = {
        f"prototype/{family}/entropy": entropy.mean(),
        f"prototype/{family}/max_weight": probabilities.amax(dim=1).mean(),
        f"prototype/{family}/offdiag_cosine": offdiag_mean,
        f"prototype/{family}/offdiag_abs_cosine": offdiag_abs_mean,
    }
    for index, name in enumerate(names):
        result[f"prototype/{family}/usage/{name}"] = probabilities[:, index].mean()
    return result


def _feature_high_frequency_energy(feature: torch.Tensor) -> torch.Tensor:
    feature = feature.float()
    low = F.avg_pool2d(feature, kernel_size=3, stride=1, padding=1)
    return (feature - low).square().mean().sqrt()


def _prototype_activation_stats(
    activations: torch.Tensor,
    names: Sequence[str],
    family: str,
) -> dict[str, torch.Tensor]:
    activations = activations.float()
    result = {
        f"prototype/{family}/activation_mean": activations.mean(),
        f"prototype/{family}/active_count": (
            activations >= 0.5
        ).sum(dim=1).float().mean(),
    }
    for index, name in enumerate(names):
        result[
            f"prototype/{family}/activation/{name}"
        ] = activations[:, index].mean()
    return result


def _task_names(batch: dict, batch_size: int) -> list[str]:
    raw = batch.get("task", ["mixed"] * batch_size)
    if isinstance(raw, str):
        return [raw] * batch_size
    names = list(raw)
    if len(names) != batch_size:
        return ["mixed"] * batch_size
    return [str(name).replace(" ", "_") for name in names]


def model_diagnostics(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    include_tokenizer: bool = True,
) -> dict[str, torch.Tensor]:
    """Return cheap, differentiable-free scalar diagnostics for TensorBoard."""
    dense_feature = outputs["dense_feature"].float()
    sparse_feature = outputs["sparse_feature"].float()
    mask = outputs["sparse_mask"].float()
    dense_gate = outputs.get(
        "fusion_dense_gate", outputs["fusion_alpha"]
    ).float().flatten()
    sparse_gate = outputs.get(
        "fusion_sparse_gate", 1.0 - outputs["fusion_alpha"]
    ).float().flatten()
    result = {
        "routing/fusion_alpha_mean": dense_gate.mean(),
        "routing/fusion_alpha_std": dense_gate.std(unbiased=False),
        "routing/dense_gate_mean": dense_gate.mean(),
        "routing/dense_gate_std": dense_gate.std(unbiased=False),
        "routing/sparse_gate_mean": sparse_gate.mean(),
        "routing/sparse_gate_std": sparse_gate.std(unbiased=False),
        "routing/gate_sum_mean": (dense_gate + sparse_gate).mean(),
        "expert/dense_feature_norm": dense_feature.square().mean().sqrt(),
        "expert/sparse_feature_norm": sparse_feature.square().mean().sqrt(),
        "expert/dense_high_frequency_energy": _feature_high_frequency_energy(dense_feature),
        "expert/sparse_high_frequency_energy": _feature_high_frequency_energy(sparse_feature),
        "mask/mean_coverage": mask.mean(),
        "mask/max_response": mask.flatten(1).amax(dim=1).mean(),
        "mask/tv": (
            torch.abs(mask[:, :, 1:, :] - mask[:, :, :-1, :]).mean()
            + torch.abs(mask[:, :, :, 1:] - mask[:, :, :, :-1]).mean()
        ),
    }
    dense_factor_gates = outputs.get("factor_dense_spatial_gates")
    sparse_factor_gates = outputs.get("factor_sparse_spatial_gates")
    if dense_factor_gates is not None and sparse_factor_gates is not None:
        dense_factor_gates = dense_factor_gates.float()
        sparse_factor_gates = sparse_factor_gates.float()
        result["factor_routing/dense/mean"] = dense_factor_gates.mean()
        result["factor_routing/dense/std"] = dense_factor_gates.std(
            unbiased=False
        )
        result["factor_routing/sparse/mean"] = sparse_factor_gates.mean()
        result["factor_routing/sparse/std"] = sparse_factor_gates.std(
            unbiased=False
        )
        for family, gates, configured_names in (
            ("dense", dense_factor_gates, DENSE_NAMES),
            ("sparse", sparse_factor_gates, SPARSE_NAMES),
        ):
            names = list(configured_names[: gates.shape[1]])
            names.extend(
                f"factor_{index}"
                for index in range(len(names), gates.shape[1])
            )
            for index, name in enumerate(names):
                factor_gate = gates[:, index]
                result[f"factor_routing/{family}/{name}/mean"] = (
                    factor_gate.mean()
                )
                result[f"factor_routing/{family}/{name}/std"] = (
                    factor_gate.std(unbiased=False)
                )
                result[f"factor_routing/{family}/{name}/coverage"] = (
                    (factor_gate >= 0.5).float().mean()
                )
        for family in ("dense", "sparse"):
            modulation = outputs[f"factor_{family}_modulation"].float()
            result[f"factor_routing/{family}/modulation_rms"] = (
                modulation.square().mean().sqrt()
            )
            pair_modulation = outputs.get(f"factor_{family}_pair_modulation")
            pair_scale = outputs.get(f"factor_{family}_pair_scale")
            if pair_modulation is not None and pair_scale is not None:
                result[f"factor_routing/{family}/pair_modulation_rms"] = (
                    pair_modulation.float().square().mean().sqrt()
                )
                result[f"factor_routing/{family}/pair_scale_rms"] = (
                    pair_scale.float().square().mean().sqrt()
                )
    for level_name in ("f1", "f2", "f3", "f4"):
        residual_key = f"multiscale_residual_{level_name}"
        if residual_key not in outputs:
            continue
        residual = outputs[residual_key].float()
        dense_scale_gate = outputs[
            f"multiscale_dense_gate_{level_name}"
        ].float()
        sparse_scale_gate = outputs[
            f"multiscale_sparse_gate_{level_name}"
        ].float()
        result[f"multiscale/{level_name}/residual_rms"] = (
            residual.square().mean().sqrt()
        )
        result[f"multiscale/{level_name}/dense_gate_mean"] = (
            dense_scale_gate.mean()
        )
        result[f"multiscale/{level_name}/dense_gate_std"] = (
            dense_scale_gate.std(unbiased=False)
        )
        result[f"multiscale/{level_name}/sparse_gate_mean"] = (
            sparse_scale_gate.mean()
        )
        result[f"multiscale/{level_name}/sparse_gate_std"] = (
            sparse_scale_gate.std(unbiased=False)
        )
    tasks = _task_names(batch, dense_gate.numel())
    for task in sorted(set(tasks)):
        indices = torch.tensor(
            [index for index, name in enumerate(tasks) if name == task],
            device=dense_gate.device,
        )
        result[
            f"routing/fusion_alpha_by_task/{task}"
        ] = dense_gate[indices].mean()
        result[
            f"routing/dense_gate_by_task/{task}"
        ] = dense_gate[indices].mean()
        result[
            f"routing/sparse_gate_by_task/{task}"
        ] = sparse_gate[indices].mean()
        result[f"mask/coverage_by_task/{task}"] = mask[indices].mean()
        if dense_factor_gates is not None and sparse_factor_gates is not None:
            result[f"factor_routing/dense_by_task/{task}"] = (
                dense_factor_gates[indices].mean()
            )
            result[f"factor_routing/sparse_by_task/{task}"] = (
                sparse_factor_gates[indices].mean()
            )

    result.update(
        _prototype_stats(
            outputs["dense_prototype_weights"],
            outputs["dense_prototypes"],
            DENSE_NAMES,
            "dense",
        )
    )
    if "dense_prototype_activations" in outputs:
        result.update(
            _prototype_activation_stats(
                outputs["dense_prototype_activations"],
                DENSE_NAMES,
                "dense",
            )
        )
    if "sparse_prototype_activations" in outputs:
        result.update(
            _prototype_activation_stats(
                outputs["sparse_prototype_activations"],
                SPARSE_NAMES,
                "sparse",
            )
        )
    result.update(
        _prototype_stats(
            outputs["sparse_prototype_weights"],
            outputs["sparse_prototypes"],
            SPARSE_NAMES,
            "sparse",
        )
    )
    if include_tokenizer:
        result.update(
            _multilabel_metrics(
                outputs["dense_logits"], batch["dense_label"], DENSE_NAMES, "tokenizer/dense"
            )
        )
        result.update(
            _multilabel_metrics(
                outputs["sparse_logits"],
                batch["sparse_label"],
                SPARSE_NAMES,
                "tokenizer/sparse",
            )
        )
        result["tokenizer/macro_precision"] = 0.5 * (
            result["tokenizer/dense/macro_precision"]
            + result["tokenizer/sparse/macro_precision"]
        )
        result["tokenizer/macro_recall"] = 0.5 * (
            result["tokenizer/dense/macro_recall"]
            + result["tokenizer/sparse/macro_recall"]
        )
        result["tokenizer/macro_f1"] = 0.5 * (
            result["tokenizer/dense/macro_f1"]
            + result["tokenizer/sparse/macro_f1"]
        )
        result["tokenizer/active_macro_precision"] = 0.5 * (
            result["tokenizer/dense/active_macro_precision"]
            + result["tokenizer/sparse/active_macro_precision"]
        )
        result["tokenizer/active_macro_recall"] = 0.5 * (
            result["tokenizer/dense/active_macro_recall"]
            + result["tokenizer/sparse/active_macro_recall"]
        )
        result["tokenizer/active_macro_f1"] = 0.5 * (
            result["tokenizer/dense/active_macro_f1"]
            + result["tokenizer/sparse/active_macro_f1"]
        )
    return {name: value.detach() for name, value in result.items()}


def gradient_global_norm(model: torch.nn.Module) -> torch.Tensor:
    squared_norms = [
        parameter.grad.detach().float().square().sum()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not squared_norms:
        return torch.zeros(())
    return torch.stack(squared_norms).sum().sqrt()


def restoration_panel(
    batch: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    max_samples: int = 6,
) -> torch.Tensor:
    """Build rows of Input | Restored | GT | Error | Mask | Mask overlay."""
    image = batch["input"].detach().float().clamp(0, 1)
    restored = outputs["restored"].detach().float().clamp(0, 1)
    target = batch["gt"].detach().float().clamp(0, 1)
    mask = outputs["sparse_mask"].detach().float().clamp(0, 1)
    count = min(max_samples, image.shape[0])
    rows = []
    for index in range(count):
        error = torch.abs(restored[index] - target[index])
        mask_rgb = mask[index].expand(3, -1, -1)
        red = torch.zeros_like(restored[index])
        red[0] = 1.0
        overlay = restored[index] * (1.0 - 0.45 * mask_rgb) + red * (0.45 * mask_rgb)
        rows.append(
            torch.cat(
                [image[index], restored[index], target[index], error, mask_rgb, overlay],
                dim=2,
            )
        )
    if not rows:
        raise ValueError("Cannot create a restoration panel from an empty batch")
    return torch.cat(rows, dim=1).cpu()
