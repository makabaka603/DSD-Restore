import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from losses import DSDRestoreV1Loss
from losses.v1_losses import sparse_mask_loss
from metrics import gradient_global_norm, model_diagnostics, restoration_panel
from metrics.training_diagnostics import multilabel_metrics_from_counts
from models import DSDRestoreV1, DSDRestoreV2
from models.experts.dense_expert import pool_tokens
from train import load_initial_model_weights
from utils.config import load_config


def check_sparse_mask_amp() -> None:
    """Regression test for probability-space BCE under automatic mixed precision."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = (
        torch.bfloat16
        if device.type == "cpu" or torch.cuda.is_bf16_supported()
        else torch.float16
    )
    logits = torch.randn(2, 1, 16, 16, device=device, requires_grad=True)
    sparse_label = torch.tensor(
        [[1, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.float32, device=device
    )
    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
        loss = sparse_mask_loss(logits.sigmoid(), sparse_label)
    loss.backward()
    if loss.dtype != torch.float32:
        raise RuntimeError(f"Sparse-mask AMP loss must be FP32, got {loss.dtype}")
    if not torch.isfinite(loss) or logits.grad is None or not torch.isfinite(logits.grad).all():
        raise RuntimeError("Sparse-mask AMP loss or gradients are not finite")


def check_v11_mask_loss() -> None:
    sparse_label = torch.tensor([[1, 0, 0, 0]], dtype=torch.float32)
    collapsed = torch.full((1, 1, 10, 10), 0.001, requires_grad=True)
    collapsed.data[:, :, 0, 0] = 0.99
    distributed = torch.full((1, 1, 10, 10), 0.001, requires_grad=True)
    distributed.data[:, :, 0, :5] = 0.90
    collapsed_loss = sparse_mask_loss(
        collapsed,
        sparse_label,
        mode="topk",
        topk_fraction=0.05,
        min_positive_coverage=0.02,
    )
    distributed_loss = sparse_mask_loss(
        distributed,
        sparse_label,
        mode="topk",
        topk_fraction=0.05,
        min_positive_coverage=0.02,
    )
    if not distributed_loss < collapsed_loss:
        raise RuntimeError(
            "V1.1 mask loss must penalize a one-hot positive mask more than "
            "a distributed top-k response"
        )
    (collapsed_loss + distributed_loss).backward()


def check_active_macro_metrics() -> None:
    metrics = multilabel_metrics_from_counts(
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.0, 0.0]),
        ("present", "absent"),
        "check",
    )
    if not torch.isclose(metrics["check/macro_f1"], torch.tensor(0.5)):
        raise RuntimeError("Legacy macro F1 changed unexpectedly")
    if not torch.isclose(metrics["check/active_macro_f1"], torch.tensor(1.0)):
        raise RuntimeError("Active-label macro F1 must exclude absent classes")


def check_v11_checkpoint_compatibility() -> None:
    torch.manual_seed(17)
    legacy = DSDRestoreV1(
        base_channels=8,
        token_dim=16,
        backbone_type="simple",
    )
    independent = DSDRestoreV1(
        base_channels=8,
        token_dim=16,
        backbone_type="simple",
        fusion_mode="independent",
    )
    legacy.eval()
    independent.eval()
    image = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        legacy_outputs = legacy(image)
    with tempfile.TemporaryDirectory() as directory:
        checkpoint_path = Path(directory) / "legacy_stage2.pth"
        torch.save(
            {
                "model": legacy.state_dict(),
                "config": {"stage": {"name": "stage2"}},
            },
            checkpoint_path,
        )
        load_initial_model_weights(
            independent,
            checkpoint_path,
            {
                "stage": {
                    "name": "v11_screen",
                    "init_from_stage": "stage2",
                },
                "runtime": {
                    "init_strict": False,
                    "init_min_coverage": 0.95,
                },
            },
        )
    with torch.no_grad():
        independent_outputs = independent(image)
    for output_name in (
        "restored",
        "fusion_dense_gate",
        "fusion_sparse_gate",
    ):
        torch.testing.assert_close(
            independent_outputs[output_name],
            legacy_outputs[output_name],
            rtol=1e-5,
            atol=1e-6,
            msg=(
                "Legacy-to-independent gate migration changed "
                f"{output_name} at initialization"
            ),
        )
    gate_sum = (
        independent_outputs["fusion_dense_gate"]
        + independent_outputs["fusion_sparse_gate"]
    )
    torch.testing.assert_close(
        gate_sum,
        torch.ones_like(gate_sum),
        rtol=0.0,
        atol=1e-6,
    )


def check_v2lite_multiscale_identity_and_gradients() -> None:
    torch.manual_seed(23)
    legacy = DSDRestoreV1(
        base_channels=8,
        token_dim=16,
        backbone_type="nafnet",
    )
    c1 = DSDRestoreV1(
        base_channels=8,
        token_dim=16,
        backbone_type="nafnet",
        multiscale_conditioning=True,
        multiscale_levels=(1, 2, 3),
        multiscale_reduction=4,
    )
    legacy.eval()
    c1.eval()
    image = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        legacy_outputs = legacy(image)
    with tempfile.TemporaryDirectory() as directory:
        checkpoint_path = Path(directory) / "legacy_stage2.pth"
        torch.save(
            {
                "model": legacy.state_dict(),
                "config": {"stage": {"name": "stage2"}},
            },
            checkpoint_path,
        )
        load_initial_model_weights(
            c1,
            checkpoint_path,
            {
                "stage": {
                    "name": "v2lite_screen_c1_multiscale",
                    "init_from_stage": "stage2",
                },
                "runtime": {
                    "init_strict": False,
                    "init_min_coverage": 0.95,
                    "init_allowed_missing_prefixes": [
                        "multiscale_conditioner.",
                    ],
                },
            },
        )
    with torch.no_grad():
        c1_outputs = c1(image)
    torch.testing.assert_close(
        c1_outputs["restored"],
        legacy_outputs["restored"],
        rtol=1e-5,
        atol=1e-6,
        msg="Zero-initialized multiscale adapters changed the V1 output",
    )
    for level_name in ("f2", "f3", "f4"):
        residual = c1_outputs[f"multiscale_residual_{level_name}"]
        torch.testing.assert_close(
            residual,
            torch.zeros_like(residual),
            rtol=0.0,
            atol=0.0,
        )

    c1.train()
    c1.zero_grad(set_to_none=True)
    train_outputs = c1(image)
    train_outputs["restored"].square().mean().backward()
    adapter_parameters = {
        name: parameter
        for name, parameter in c1.named_parameters()
        if name.startswith("multiscale_conditioner.")
    }
    missing_gradients = [
        name
        for name, parameter in adapter_parameters.items()
        if parameter.grad is None
    ]
    if missing_gradients:
        raise RuntimeError(
            "V2-lite multiscale parameters without gradients: "
            f"{missing_gradients}"
        )
    expand_gradient = sum(
        float(parameter.grad.detach().abs().sum())
        for name, parameter in adapter_parameters.items()
        if name.endswith("expand.weight")
    )
    if expand_gradient <= 0:
        raise RuntimeError(
            "V2-lite zero-initialized output projections received no gradient"
        )
    diagnostics = model_diagnostics(train_outputs, {
        "dense_label": torch.zeros(1, 5),
        "sparse_label": torch.zeros(1, 4),
        "task": ["smoke"],
    })
    required = {
        f"multiscale/{level_name}/residual_rms"
        for level_name in ("f2", "f3", "f4")
    }
    missing = required - diagnostics.keys()
    if missing:
        raise RuntimeError(
            f"Missing V2-lite multiscale diagnostics: {sorted(missing)}"
        )

    full_legacy = DSDRestoreV1()
    full_c1 = DSDRestoreV1(multiscale_conditioning=True)
    legacy_parameters = sum(
        parameter.numel()
        for parameter in full_legacy.parameters()
    )
    c1_parameters = sum(
        parameter.numel()
        for parameter in full_c1.parameters()
    )
    overhead = c1_parameters - legacy_parameters
    if overhead > 250_000:
        raise RuntimeError(
            f"V2-lite C1 overhead is too large: {overhead:,} parameters"
        )
    coverage = legacy_parameters / c1_parameters
    if coverage < 0.98:
        raise RuntimeError(
            "V2-lite C1 old-parameter coverage is below 98%: "
            f"{coverage:.2%}"
        )


def check_d1_shared_capacity_reallocation() -> None:
    legacy = DSDRestoreV1()
    d1 = DSDRestoreV1(
        dense_expert_blocks=2,
        sparse_expert_blocks=2,
        shared_bottleneck_blocks=4,
    )
    legacy_parameters = sum(parameter.numel() for parameter in legacy.parameters())
    d1_parameters = sum(parameter.numel() for parameter in d1.parameters())
    parameter_change = (d1_parameters - legacy_parameters) / legacy_parameters
    if not (-0.03 <= parameter_change < 0.0):
        raise RuntimeError(
            "D1 must reduce V1 parameters by less than 3%, found "
            f"{parameter_change:+.2%} ({legacy_parameters:,} -> {d1_parameters:,})"
        )
    if len(d1.dense_expert.blocks) != 2 or len(d1.sparse_expert.blocks) != 2:
        raise RuntimeError("D1 expert depths are not 2/2")
    if len(d1.shared_bottleneck) != 4:
        raise RuntimeError("D1 shared bottleneck depth is not 4")
    residual_scales = [
        parameter
        for name, parameter in d1.shared_bottleneck.named_parameters()
        if name.endswith("beta") or name.endswith("gamma")
    ]
    if len(residual_scales) != 8:
        raise RuntimeError("D1 shared bottleneck must expose 8 zero residual scales")
    if any(torch.count_nonzero(parameter).item() for parameter in residual_scales):
        raise RuntimeError("D1 shared NAFBlocks are not identity-initialized")

    legacy_reference = legacy.state_dict()[
        "dense_expert.blocks.0.large_kernel.weight"
    ].clone()
    with tempfile.TemporaryDirectory() as directory:
        checkpoint_path = Path(directory) / "legacy_stage2.pth"
        torch.save(
            {
                "model": legacy.state_dict(),
                "config": {"stage": {"name": "stage2"}},
            },
            checkpoint_path,
        )
        load_initial_model_weights(
            d1,
            checkpoint_path,
            {
                "stage": {
                    "name": "screen_d1_shared_capacity",
                    "init_from_stage": "stage2",
                },
                "runtime": {
                    "init_strict": False,
                    "init_min_coverage": 0.80,
                    "init_allowed_missing_prefixes": [
                        "shared_bottleneck.",
                    ],
                },
            },
        )
    torch.testing.assert_close(
        d1.state_dict()["dense_expert.blocks.0.large_kernel.weight"],
        legacy_reference,
        msg="D1 did not inherit retained expert blocks from Stage 2",
    )
    if any(torch.count_nonzero(parameter).item() for parameter in residual_scales):
        raise RuntimeError("D1 checkpoint loading changed zero residual scales")

    small_d1 = DSDRestoreV1(
        base_channels=8,
        token_dim=16,
        dense_expert_blocks=2,
        sparse_expert_blocks=2,
        shared_bottleneck_blocks=4,
    )
    image = torch.rand(1, 3, 64, 64)
    outputs = small_d1(image)
    if not torch.isfinite(outputs["restored"]).all():
        raise RuntimeError("D1 forward produced non-finite restored pixels")
    outputs["restored"].square().mean().backward()
    scale_gradient = sum(
        float(parameter.grad.detach().abs().sum())
        for name, parameter in small_d1.named_parameters()
        if name.startswith("shared_bottleneck.")
        and (name.endswith("beta") or name.endswith("gamma"))
        and parameter.grad is not None
    )
    if scale_gradient <= 0:
        raise RuntimeError("D1 shared zero-residual scales received no gradient")


def check_v2_factor_spatial_routing() -> None:
    legacy = DSDRestoreV1()
    v2 = DSDRestoreV2()
    legacy_parameters = sum(parameter.numel() for parameter in legacy.parameters())
    v2_parameters = sum(parameter.numel() for parameter in v2.parameters())
    parameter_change = (v2_parameters - legacy_parameters) / legacy_parameters
    if not (0.0 < parameter_change <= 0.03):
        raise RuntimeError(
            "V2 factor router must add at most 3% parameters, found "
            f"{parameter_change:+.2%} ({legacy_parameters:,} -> {v2_parameters:,})"
        )
    if v2.factor_router is None:
        raise RuntimeError("V2 factor-spatial router is disabled")
    if torch.count_nonzero(v2.factor_router.dense_scale).item():
        raise RuntimeError("V2 dense modulation scale is not zero-initialized")
    if torch.count_nonzero(v2.factor_router.sparse_scale).item():
        raise RuntimeError("V2 sparse modulation scale is not zero-initialized")

    torch.manual_seed(31)
    small_legacy = DSDRestoreV1(
        base_channels=8,
        token_dim=16,
        backbone_type="simple",
    )
    small_v2 = DSDRestoreV2(
        base_channels=8,
        token_dim=16,
        backbone_type="simple",
        factor_router_dim=8,
    )
    small_legacy.eval()
    small_v2.eval()
    image = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        legacy_outputs = small_legacy(image)
    with tempfile.TemporaryDirectory() as directory:
        checkpoint_path = Path(directory) / "legacy_stage2.pth"
        torch.save(
            {
                "model": small_legacy.state_dict(),
                "config": {"stage": {"name": "stage2"}},
            },
            checkpoint_path,
        )
        load_initial_model_weights(
            small_v2,
            checkpoint_path,
            {
                "stage": {
                    "name": "screen_v2_factor_spatial_routing",
                    "init_from_stage": "stage2",
                },
                "runtime": {
                    "init_strict": False,
                    "init_min_coverage": 0.99,
                    "init_allowed_missing_prefixes": ["factor_router."],
                },
            },
        )
    with torch.no_grad():
        v2_outputs = small_v2(image)
    torch.testing.assert_close(
        v2_outputs["restored"],
        legacy_outputs["restored"],
        rtol=1e-5,
        atol=1e-6,
        msg="Zero-initialized V2 factor routing changed the Stage 2 output",
    )
    if v2_outputs["factor_dense_spatial_gates"].shape[1] != 5:
        raise RuntimeError("V2 must emit five dense factor maps")
    if v2_outputs["factor_sparse_spatial_gates"].shape[1] != 4:
        raise RuntimeError("V2 must emit four sparse factor maps")
    for name in ("factor_dense_spatial_gates", "factor_sparse_spatial_gates"):
        gates = v2_outputs[name]
        if float(gates.min()) < 0.0 or float(gates.max()) > 1.0:
            raise RuntimeError(f"V2 spatial gates are outside [0, 1]: {name}")

    small_v2.train()
    small_v2.zero_grad(set_to_none=True)
    train_outputs = small_v2(image)
    train_outputs["restored"].square().mean().backward()
    scale_gradient = sum(
        float(parameter.grad.detach().abs().sum())
        for name, parameter in small_v2.named_parameters()
        if name in {
            "factor_router.dense_scale",
            "factor_router.sparse_scale",
        }
        and parameter.grad is not None
    )
    if scale_gradient <= 0:
        raise RuntimeError("V2 zero modulation scales received no gradient")
    diagnostics = model_diagnostics(
        train_outputs,
        {
            "dense_label": torch.zeros(2, 5),
            "sparse_label": torch.zeros(2, 4),
            "task": ["smoke", "smoke"],
        },
    )
    required = {
        "factor_routing/dense/dust/mean",
        "factor_routing/dense/haze/coverage",
        "factor_routing/sparse/rain/mean",
        "factor_routing/sparse/snow/coverage",
        "factor_routing/dense/modulation_rms",
        "factor_routing/sparse/modulation_rms",
    }
    missing = required - diagnostics.keys()
    if missing:
        raise RuntimeError(
            f"Missing V2 factor-routing diagnostics: {sorted(missing)}"
        )


def check_v11_forward_backward() -> None:
    model = DSDRestoreV1(
        base_channels=16,
        token_dim=32,
        fusion_mode="independent",
        token_pooling="presence_weighted",
        prototype_weighting="sigmoid",
    )
    batch = {
        "input": torch.rand(2, 3, 64, 64),
        "gt": torch.rand(2, 3, 64, 64),
        "dense_label": torch.tensor(
            [[1, 0, 1, 0, 0], [0, 1, 0, 1, 0]],
            dtype=torch.float32,
        ),
        "sparse_label": torch.tensor(
            [[1, 0, 0, 0], [0, 0, 1, 0]],
            dtype=torch.float32,
        ),
    }
    outputs = model(batch["input"])
    criterion = DSDRestoreV1Loss(
        classification_pos_weight=2.0,
        prototype_loss_mode="multilabel",
        mask_loss_mode="topk",
        auxiliary_decay_start=0.25,
        auxiliary_final_scale=0.25,
    )
    criterion.set_training_progress(1.0)
    if abs(criterion.auxiliary_scale - 0.25) > 1e-8:
        raise RuntimeError("V1.1 auxiliary-loss decay did not reach its final scale")
    losses = criterion(outputs, batch)
    losses["total"].backward()
    diagnostics = model_diagnostics(outputs, batch)
    missing_gradients = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    if missing_gradients:
        raise RuntimeError(
            f"V1.1 trainable parameters without gradients: {missing_gradients}"
        )
    if outputs["fusion_dense_gate"].shape != (2, 1, 1, 1):
        raise RuntimeError("Unexpected V1.1 dense-gate shape")
    if outputs["fusion_sparse_gate"].shape != (2, 1, 1, 1):
        raise RuntimeError("Unexpected V1.1 sparse-gate shape")
    if outputs["dense_prototype_activations"].shape != (2, 5):
        raise RuntimeError("Unexpected V1.1 dense prototype activation shape")
    required_diagnostics = {
        "routing/dense_gate_mean",
        "routing/sparse_gate_mean",
        "routing/gate_sum_mean",
        "prototype/dense/active_count",
        "tokenizer/active_macro_recall",
    }
    missing_diagnostics = required_diagnostics - diagnostics.keys()
    if missing_diagnostics:
        raise RuntimeError(
            f"Missing V1.1 diagnostics: {sorted(missing_diagnostics)}"
        )
    absent_logits = torch.full((2, 5), -20.0)
    absent_pool = pool_tokens(
        torch.ones(2, 5, 4),
        absent_logits,
        "presence_weighted",
    )
    if float(absent_pool.abs().max()) >= 1e-6:
        raise RuntimeError("Absent V1.1 token family must approach a zero prompt")


def check_v11_configs() -> None:
    config_paths = [
        "configs/screen_v11_control.yaml",
        "configs/screen_v11_aux_off.yaml",
        "configs/screen_v11_dual_gate.yaml",
        "configs/screen_v11_weighted_tokens.yaml",
        "configs/screen_v11_multilabel_proto.yaml",
        "configs/screen_v11_full.yaml",
        "configs/screen_v11b_dual_gate_migrated.yaml",
        "configs/screen_v11b_full_migrated.yaml",
        "configs/screen_v2lite_c1_multiscale.yaml",
        "configs/screen_d0_triple_balance.yaml",
        "configs/screen_d1_shared_capacity.yaml",
        "configs/screen_v2_factor_spatial_routing.yaml",
        "configs/smoke_v11.yaml",
        "configs/train_v11_stage1.yaml",
        "configs/train_v11_stage2.yaml",
        "configs/train_v11_stage3.yaml",
    ]
    for config_path in config_paths:
        config = load_config(config_path)
        if "model" not in config or "loss" not in config:
            raise RuntimeError(f"Incomplete V1.1 config: {config_path}")

    d0 = load_config("configs/screen_d0_triple_balance.yaml")
    d0_sources = d0["data"]["train_sources"]
    d0_probabilities = [float(source["probability"]) for source in d0_sources]
    d0_groups = [tuple(source["include_groups"]) for source in d0_sources]
    if d0_groups != [("dense_dense",), ("dense_sparse",), ("triple",)]:
        raise RuntimeError(f"Unexpected D0 metadata groups: {d0_groups}")
    if any(
        abs(actual - expected) > 1e-8
        for actual, expected in zip(d0_probabilities, (0.40, 0.35, 0.25))
    ):
        raise RuntimeError(f"Unexpected D0 probabilities: {d0_probabilities}")

    d1 = load_config("configs/screen_d1_shared_capacity.yaml")
    d1_model = d1["model"]
    if (
        int(d1_model["dense_expert_blocks"]),
        int(d1_model["sparse_expert_blocks"]),
        int(d1_model["shared_bottleneck_blocks"]),
    ) != (2, 2, 4):
        raise RuntimeError(f"Unexpected D1 capacity allocation: {d1_model}")
    for config, name in ((d0, "D0"), (d1, "D1")):
        auxiliary_weights = tuple(
            float(config["loss"][key])
            for key in ("lambda_cls", "lambda_proto", "lambda_sparse")
        )
        if auxiliary_weights != (0.0, 0.0, 0.0):
            raise RuntimeError(f"{name} must retain the A1 auxiliary-loss control")
        if int(config["optimization"]["max_iters"]) != 15000:
            raise RuntimeError(f"{name} must remain a 15k validation screen")

    v2 = load_config("configs/screen_v2_factor_spatial_routing.yaml")
    if v2["model"].get("type") != "dsd_restore_v2":
        raise RuntimeError("V2 screen must build dsd_restore_v2")
    if int(v2["model"].get("factor_router_dim", 0)) != 32:
        raise RuntimeError("V2 screen must use a 32-D factor router")
    if float(v2["model"].get("factor_router_temperature", 0.0)) != 1.0:
        raise RuntimeError("V2 screen must use router temperature 1.0")
    if v2["data"]["train_sources"] != load_config(
        "configs/screen_v11_aux_off.yaml"
    )["data"]["train_sources"]:
        raise RuntimeError("V2 must retain the frozen A1/Stage 3 sampling")
    if tuple(
        float(v2["loss"][key])
        for key in ("lambda_cls", "lambda_proto", "lambda_sparse")
    ) != (0.0, 0.0, 0.0):
        raise RuntimeError("V2 must retain the A1 auxiliary-loss control")
    if int(v2["optimization"]["max_iters"]) != 15000:
        raise RuntimeError("V2 must remain a 15k validation screen")
    if tuple(v2["runtime"].get("init_allowed_missing_prefixes", ())) != (
        "factor_router.",
    ):
        raise RuntimeError("V2 may initialize only factor_router.* from scratch")


def main() -> None:
    check_sparse_mask_amp()
    check_v11_mask_loss()
    check_active_macro_metrics()
    check_v11_checkpoint_compatibility()
    check_v2lite_multiscale_identity_and_gradients()
    check_d1_shared_capacity_reallocation()
    check_v2_factor_spatial_routing()
    check_v11_forward_backward()
    check_v11_configs()
    model = DSDRestoreV1(base_channels=16, token_dim=32)
    batch = {
        "input": torch.rand(2, 3, 64, 64),
        "gt": torch.rand(2, 3, 64, 64),
        "dense_label": torch.tensor([[1, 0, 1, 0, 0], [0, 1, 0, 1, 0]], dtype=torch.float32),
        "sparse_label": torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=torch.float32),
    }
    outputs = model(batch["input"])
    loss = DSDRestoreV1Loss()(outputs, batch)
    loss["total"].backward()
    diagnostics = model_diagnostics(outputs, batch)
    diagnostics["system/gradient_global_norm"] = gradient_global_norm(model)
    panel = restoration_panel(batch, outputs, max_samples=2)
    missing_gradients = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    if missing_gradients:
        raise RuntimeError(f"Trainable parameters without gradients: {missing_gradients}")
    if outputs["dense_prototype_weights"].shape != (2, 5):
        raise ValueError("Unexpected dense prototype weight shape")
    if outputs["sparse_prototype_weights"].shape != (2, 4):
        raise ValueError("Unexpected sparse prototype weight shape")
    required_diagnostics = {
        "prototype/dense/entropy",
        "prototype/sparse/offdiag_cosine",
        "routing/fusion_alpha_mean",
        "expert/dense_high_frequency_energy",
        "mask/mean_coverage",
        "tokenizer/macro_f1",
        "system/gradient_global_norm",
    }
    missing_diagnostics = required_diagnostics - diagnostics.keys()
    if missing_diagnostics:
        raise RuntimeError(f"Missing training diagnostics: {sorted(missing_diagnostics)}")
    if panel.shape != (3, 128, 384):
        raise ValueError(f"Unexpected TensorBoard panel shape: {tuple(panel.shape)}")
    print("DSD-Restore V1 smoke test passed")
    print("DSD-Restore V1.1 smoke test passed")
    print("V1.1 equivalent legacy-gate checkpoint migration: passed")
    print("V2-lite C1 identity initialization and gradients: passed")
    print("D1 shared-capacity checkpoint migration and gradients: passed")
    print("D0/D1 controlled-screen configs: passed")
    print("V2 factor-spatial checkpoint migration and gradients: passed")
    print("V2 factor-spatial controlled-screen config: passed")
    print("V1.1 configs: passed")
    print("sparse-mask AMP forward/backward: passed")
    print(f"restored: {tuple(outputs['restored'].shape)}")
    print(f"dense_tokens: {tuple(outputs['dense_tokens'].shape)}")
    print(f"sparse_tokens: {tuple(outputs['sparse_tokens'].shape)}")
    print(f"prototype loss: {float(loss['proto']):.4f}")
    print(f"sparse-mask loss: {float(loss['sparse']):.4f}")
    print(f"diagnostic scalars: {len(diagnostics)}")
    print(f"visual panel: {tuple(panel.shape)}")
    print(f"loss: {float(loss['total'].detach()):.4f}")


if __name__ == "__main__":
    main()
