from .dsd_restore import DSDRestoreV1, DSDRestoreV2
from .nafnet_baseline import NAFNetBaseline


def build_model(config: dict):
    model_config = dict(config)
    model_type = str(model_config.pop("type", "dsd_restore_v1")).lower()
    if model_type in {"dsd_restore_v1", "dsd"}:
        return DSDRestoreV1(**model_config)
    if model_type in {"dsd_restore_v2", "dsd_v2"}:
        return DSDRestoreV2(**model_config)
    if model_type in {"nafnet_baseline", "naf_baseline"}:
        return NAFNetBaseline(**model_config)
    raise ValueError(f"Unknown model type: {model_type}")


__all__ = ["DSDRestoreV1", "DSDRestoreV2", "NAFNetBaseline", "build_model"]
