from copy import deepcopy
from pathlib import Path

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            # Lists intentionally replace the base list. Stage-specific source
            # distributions must never be appended to the joint-training list.
            merged[key] = deepcopy(value)
    return merged


def _load_config(path: Path, loading: tuple[Path, ...]) -> dict:
    resolved = path.resolve()
    if resolved in loading:
        chain = " -> ".join(str(item) for item in (*loading, resolved))
        raise ValueError(f"Circular base_config chain: {chain}")
    with resolved.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise TypeError(f"Config root must be a mapping: {resolved}")

    base_refs = config.pop("base_config", None)
    if not base_refs:
        return config
    if isinstance(base_refs, (str, Path)):
        base_refs = [base_refs]
    if not isinstance(base_refs, list):
        raise TypeError(f"base_config must be a path or list of paths: {resolved}")

    merged: dict = {}
    for base_ref in base_refs:
        base_path = Path(base_ref)
        if not base_path.is_absolute():
            base_path = resolved.parent / base_path
        merged = _deep_merge(merged, _load_config(base_path, (*loading, resolved)))
    return _deep_merge(merged, config)


def load_config(path: str | Path) -> dict:
    """Load YAML with optional recursive ``base_config`` inheritance."""
    return _load_config(Path(path), ())
