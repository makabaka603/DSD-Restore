import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
LABEL_KEYS = (
    "dust",
    "sand",
    "haze",
    "lowlight",
    "colorcast",
    "rain",
    "raindrop",
    "snow",
    "occlusion",
)
EXPECTED_TEST_COUNTS = {
    "dust_haze": 150,
    "dust_lowlight": 150,
    "haze_lowlight": 150,
    "dust_rain": 100,
    "haze_rain": 100,
    "lowlight_rain": 100,
    "haze_snow": 75,
    "lowlight_snow": 75,
    "triple": 100,
}


def audit_dataset(path: Path, check_sizes: bool) -> dict:
    metadata_path = path / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    samples = metadata.get("samples", {})
    input_names = {
        file.name for file in (path / "input").iterdir() if file.suffix.lower() in IMAGE_SUFFIXES
    }
    gt_names = {
        file.name for file in (path / "gt").iterdir() if file.suffix.lower() in IMAGE_SUFFIXES
    }
    metadata_names = set(samples)
    if input_names != gt_names or input_names != metadata_names:
        raise ValueError(
            f"{path}: input/GT/metadata names differ: "
            f"input={len(input_names)} gt={len(gt_names)} metadata={len(metadata_names)}"
        )

    categories = Counter()
    groups = Counter()
    clean_sources = set()
    for filename, record in samples.items():
        strengths = record.get("strengths", {})
        tasks = record.get("tasks", [])
        active = {key for key in LABEL_KEYS if float(strengths.get(key, 0.0)) > 0}
        if active != set(tasks):
            raise ValueError(
                f"{path / 'metadata.json'}: task/strength mismatch for {filename}: "
                f"tasks={tasks}, active={sorted(active)}"
            )
        if any(not 0.0 <= float(strengths.get(key, 0.0)) <= 1.0 for key in LABEL_KEYS):
            raise ValueError(f"Strength outside [0, 1] for {filename}")
        categories[record.get("category", "+".join(tasks))] += 1
        if record.get("group"):
            groups[record["group"]] += 1
        clean_sources.add(record.get("source"))

    declared_groups = metadata.get("group_counts")
    if declared_groups and dict(groups) != {key: int(value) for key, value in declared_groups.items()}:
        raise ValueError(
            f"{path}: group counts differ: actual={dict(groups)} declared={declared_groups}"
        )

    if metadata.get("name") == "Synthetic-Mixed-Test-1K":
        if len(samples) != 1000:
            raise ValueError(f"{path}: expected 1000 test samples, found {len(samples)}")
        test_counts = Counter(record["category"] for record in samples.values())
        if dict(test_counts) != EXPECTED_TEST_COUNTS:
            raise ValueError(
                f"{path}: Mixed-Test category counts differ: {dict(test_counts)}"
            )

    if check_sizes:
        for filename in sorted(input_names):
            with Image.open(path / "input" / filename) as image, Image.open(
                path / "gt" / filename
            ) as target:
                if image.size != target.size:
                    raise ValueError(
                        f"{path}: size mismatch for {filename}: {image.size} != {target.size}"
                    )

    print(
        f"{path}: samples={len(samples)} clean_sources={len(clean_sources)} "
        f"groups={dict(groups) or 'n/a'}"
    )
    print(f"  categories={dict(categories)}")
    return {"clean_sources": clean_sources, "sample_count": len(samples)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit materialized composite datasets")
    parser.add_argument("--train", default="datasets/SyntheticCompositeTrain")
    parser.add_argument("--val", default="datasets/SyntheticCompositeVal")
    parser.add_argument("--test", default="datasets/Synthetic-Mixed-Test-1K")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--check-sizes", action="store_true")
    args = parser.parse_args()

    audited = {
        "train": audit_dataset(Path(args.train), args.check_sizes),
        "val": audit_dataset(Path(args.val), args.check_sizes),
    }
    if not args.skip_test:
        audited["test"] = audit_dataset(Path(args.test), args.check_sizes)

    names = list(audited)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = audited[left]["clean_sources"] & audited[right]["clean_sources"]
            if overlap:
                preview = ", ".join(sorted(overlap)[:10])
                raise ValueError(
                    f"Clean-source leakage between {left} and {right}: {preview}"
                )
    print("Offline composite dataset audit passed")


if __name__ == "__main__":
    main()
