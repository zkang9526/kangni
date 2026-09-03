#!/usr/bin/env python3
"""Check that the portable package is self-contained on any machine."""

from pathlib import Path
import csv
import sys

ROOT = Path(__file__).resolve().parents[1]
errors = []

def check_images(relative: str, expected: int) -> None:
    folder = ROOT / relative
    count = len(list(folder.rglob("*.jpg"))) + len(list(folder.rglob("*.png")))
    if count != expected:
        errors.append(f"{relative}: expected {expected} images, got {count}")

check_images("data/fcw/images/bdd100k/train", 70000)
check_images("data/fcw/images/bdd100k/val", 10000)
check_images("data/fcw/images/nuscenes_mini/train", 323)
check_images("data/fcw/images/nuscenes_mini/val", 81)
check_images("data/dms/eye_state/images", 84898)
check_images("data/dms/drowsiness_binary/images", 18000)
check_images("data/dms/rear_occupancy_sviro/images", 5000)

for manifest in (ROOT / "manifests").glob("*.csv"):
    with manifest.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    for index, row in enumerate(rows, 2):
        for key in ("image_rel", "image_path"):
            value = row.get(key, "")
            if value:
                path = Path(value)
                if path.is_absolute() or ":" in path.parts[0]:
                    errors.append(f"{manifest.name}:{index}: absolute path {value}")
                elif not (ROOT / path).is_file():
                    errors.append(f"{manifest.name}:{index}: missing {value}")

for config in (ROOT / "configs").glob("*.yaml"):
    if "E:/" in config.read_text(encoding="utf-8") or "C:/" in config.read_text(encoding="utf-8"):
        errors.append(f"{config.name}: contains an absolute drive path")

if errors:
    print("FAIL")
    print("\n".join(errors[:50]))
    sys.exit(1)
print("PASS: package files, counts, relative manifests, and configuration paths are valid")
