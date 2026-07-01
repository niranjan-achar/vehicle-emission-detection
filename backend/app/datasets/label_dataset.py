"""Create a labeled image dataset from a manifest file.

Usage:
    python -m app.datasets.label_dataset --manifest labels.csv --output dataset_root

Manifest format:
    path,label
    C:\data\img001.jpg,smoky
    C:\data\img002.jpg,non_smoky

The script copies files into:
    output_root/smoky/
    output_root/non_smoky/

This is useful after manual annotation or when you already have a list of
cropped key-region samples to train the paper-based SVM models.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_ALIASES = {
    "smoke": "smoky",
    "smoky": "smoky",
    "polluting": "smoky",
    "non_smoke": "non_smoky",
    "non_smoky": "non_smoky",
    "clean": "non_smoky",
}


def normalize_label(label: str) -> str:
    normalized = label.strip().lower()
    if normalized not in LABEL_ALIASES:
        raise ValueError(f"Unsupported label: {label}")
    return LABEL_ALIASES[normalized]


def label_dataset(manifest_path: Path, output_root: Path, copy_files: bool = True) -> int:
    """Copy files into label-based folders from a CSV manifest."""
    output_root.mkdir(parents=True, exist_ok=True)
    created = 0

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "path" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError("Manifest must include 'path' and 'label' columns")

        for row in reader:
            source_path = Path(row["path"])
            label = normalize_label(row["label"])
            if source_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if not source_path.exists():
                continue

            label_dir = output_root / label
            label_dir.mkdir(parents=True, exist_ok=True)

            destination = label_dir / source_path.name
            if copy_files:
                shutil.copy2(source_path, destination)
            else:
                shutil.move(str(source_path), str(destination))
            created += 1

    return created


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a labeled dataset from a CSV manifest")
    parser.add_argument("--manifest", required=True, help="CSV file with path,label columns")
    parser.add_argument("--output", required=True, help="Destination dataset root")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest_path = Path(args.manifest)
    output_root = Path(args.output)
    count = label_dataset(manifest_path, output_root, copy_files=not args.move)
    print(f"Labeled {count} samples into {output_root}")


if __name__ == "__main__":
    main()
