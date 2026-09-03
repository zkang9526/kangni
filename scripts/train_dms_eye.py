#!/usr/bin/env python3
"""Train the initial DMS eye-state baseline on the portable V2 manifest."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "manifests/dms_eye_state.csv"
DEFAULT_OUTPUT = ROOT / "runs/dms_eye_mobilenetv2"
CLASS_TO_ID = {"closed": 0, "open": 1}


class ManifestDataset(Dataset):
    def __init__(self, manifest: Path, split: str, transform, limit: int = 0) -> None:
        self.transform = transform
        with manifest.open("r", encoding="utf-8-sig", newline="") as file:
            rows = [row for row in csv.DictReader(file) if row.get("split") == split]
        rows.sort(key=lambda row: row.get("image_rel", row.get("image_path", "")))
        if limit > 0:
            # Keep smoke tests representative instead of truncating into one
            # class directory after the deterministic path sort.
            grouped = {
                class_name: [row for row in rows if row.get("class_name") == class_name]
                for class_name in CLASS_TO_ID
            }
            selected = []
            while len(selected) < limit and any(grouped.values()):
                for class_name in CLASS_TO_ID:
                    if grouped[class_name] and len(selected) < limit:
                        selected.append(grouped[class_name].pop(0))
            rows = selected
        self.samples: List[Tuple[Path, int]] = []
        for row in rows:
            value = row.get("image_rel") or row.get("image_path") or ""
            path = Path(value)
            if not path.is_absolute():
                path = ROOT / path
            class_name = row.get("class_name", "")
            if class_name not in CLASS_TO_ID:
                raise ValueError("Unsupported DMS class: {}".format(class_name))
            if not path.is_file():
                raise FileNotFoundError(path)
            self.samples.append((path, CLASS_TO_ID[class_name]))
        if not self.samples:
            raise ValueError("No samples found for split={!r}".format(split))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            return self.transform(image), label


def make_transforms(image_size: int):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val = transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor(), normalize]
    )
    return train, val


def build_model(pretrained: bool) -> nn.Module:
    weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = mobilenet_v2(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASS_TO_ID))
    return model


def metrics_from_confusion(confusion: List[List[int]]) -> Dict[str, object]:
    total = sum(sum(row) for row in confusion)
    accuracy = sum(confusion[index][index] for index in range(2)) / max(total, 1)
    precision: List[float] = []
    recall: List[float] = []
    f1: List[float] = []
    for class_id in range(2):
        true_positive = confusion[class_id][class_id]
        false_positive = sum(confusion[row][class_id] for row in range(2) if row != class_id)
        false_negative = sum(confusion[class_id][column] for column in range(2) if column != class_id)
        p = true_positive / max(true_positive + false_positive, 1)
        r = true_positive / max(true_positive + false_negative, 1)
        precision.append(p)
        recall.append(r)
        f1.append(2 * p * r / max(p + r, 1e-12))
    return {
        "accuracy": accuracy,
        "closed_recall": recall[CLASS_TO_ID["closed"]],
        "macro_precision": sum(precision) / 2,
        "macro_recall": sum(recall) / 2,
        "macro_f1": sum(f1) / 2,
        "confusion_matrix": confusion,
    }


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, object]:
    model.eval()
    confusion = [[0, 0], [0, 0]]
    elapsed = 0.0
    sample_count = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            predictions = model(images).argmax(dim=1).cpu()
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed += time.perf_counter() - started
            for expected, predicted in zip(labels.tolist(), predictions.tolist()):
                confusion[expected][predicted] += 1
            sample_count += len(labels)
    result = metrics_from_confusion(confusion)
    result["mean_inference_ms_per_sample_debug"] = 1000.0 * elapsed / max(sample_count, 1)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--resume", type=Path, help="Resume from a checkpoint produced by this script.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device
    )
    device = torch.device(device_name)
    train_transform, val_transform = make_transforms(args.image_size)
    train_data = ManifestDataset(args.manifest.resolve(), "train", train_transform, args.limit_train)
    val_data = ManifestDataset(args.manifest.resolve(), "val", val_transform, args.limit_val)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda"
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda"
    )
    model = build_model(args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    args.output.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(args.resume.resolve(), map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_f1 = float(checkpoint.get("best_f1", checkpoint.get("metrics", {}).get("macro_f1", -1.0)))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
    history: List[Dict[str, object]] = []

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(labels)
            seen += len(labels)
        validation = evaluate(model, val_loader, device)
        record = {"epoch": epoch, "train_loss": running_loss / max(seen, 1), **validation}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        checkpoint = {
            "model_name": "mobilenet_v2",
            "task": "dms_eye_state",
            "class_to_id": CLASS_TO_ID,
            "image_size": args.image_size,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": validation,
            "best_f1": max(best_f1, float(validation["macro_f1"])),
        }
        torch.save(checkpoint, args.output / "last.pt")
        if float(validation["macro_f1"]) > best_f1:
            best_f1 = float(validation["macro_f1"])
            torch.save(checkpoint, args.output / "best.pt")

    summary = {
        "model": "mobilenet_v2",
        "device": str(device),
        "pretrained": args.pretrained,
        "resumed_from": str(args.resume.resolve()) if args.resume else None,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "history": history,
    }
    (args.output / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
