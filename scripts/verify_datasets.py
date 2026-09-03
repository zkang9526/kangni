#!/usr/bin/env python3
"""Check that the versioned datasets are self-contained on any machine."""

from pathlib import Path
import csv
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "datasets/v3.0"
errors = []

def check_images(relative: str, expected: int) -> None:
    folder = DATASET_ROOT / relative
    count = len(list(folder.rglob("*.jpg"))) + len(list(folder.rglob("*.png")))
    if count != expected:
        errors.append(f"{relative}: expected {expected} images, got {count}")

check_images("fcw/images/bdd100k/train", 70000)
check_images("fcw/images/bdd100k/val", 10000)
check_images("fcw/images/nuscenes_mini/train", 323)
check_images("fcw/images/nuscenes_mini/val", 81)
check_images("dms/eye_state/images", 84898)
check_images("dms/drowsiness_binary/images", 18000)
check_images("dms/rear_occupancy_sviro/images", 5000)

for manifest in (DATASET_ROOT / "manifests").glob("*.csv"):
    with manifest.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    for index, row in enumerate(rows, 2):
        for key in ("image_rel", "image_path"):
            value = row.get(key, "")
            if value:
                path = Path(value)
                if path.is_absolute() or ":" in path.parts[0]:
                    errors.append(f"{manifest.name}:{index}: absolute path {value}")
                elif not (DATASET_ROOT / path).is_file():
                    errors.append(f"{manifest.name}:{index}: missing {value}")

for config in (PROJECT_ROOT / "configs").glob("*.yaml"):
    if "E:/" in config.read_text(encoding="utf-8") or "C:/" in config.read_text(encoding="utf-8"):
        errors.append(f"{config.name}: contains an absolute drive path")

if errors:
    print("FAIL")
    print("\n".join(errors[:50]))
    sys.exit(1)
print("PASS: dataset files, counts, relative manifests, and configuration paths are valid")
