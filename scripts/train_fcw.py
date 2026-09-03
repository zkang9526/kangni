#!/usr/bin/env python3
"""Portable Ultralytics FCW training entry point."""
import argparse
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="yolo11n.pt")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--batch", type=int, default=8)
parser.add_argument("--workers", type=int, default=0)
parser.add_argument("--device", default="0")
parser.add_argument("--fraction", type=float, default=1.0)
parser.add_argument("--name", default="fcw_baseline")
args = parser.parse_args()

model = YOLO(args.model)
model.train(
    data=str((ROOT / "configs/fcw_unified.yaml").resolve()),
    epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
    workers=args.workers, device=args.device, fraction=args.fraction,
    project=str((ROOT / "runs/fcw").resolve()), name=args.name,
    cache=False, plots=True,
)
