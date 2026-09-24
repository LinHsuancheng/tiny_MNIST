"""Command-line entry point for training the compact MNIST model."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent


def checkpoint_paths(output_dir: Path) -> dict[str, Path]:
    checkpoint_dir = output_dir / "checkpoints"
    return {
        "best": checkpoint_dir / "best.pt",
        "last": checkpoint_dir / "last.pt",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=14596)
    parser.add_argument("--max-lr", type=float, default=1e-3)
    parser.add_argument("--initial-div", type=float, default=25.0)
    parser.add_argument("--final-div", type=float, default=175.0)
    parser.add_argument("--warmup-pct", type=float, default=0.5)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def choose_device(torch: Any, requested: str):
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if requested == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return torch.device("mps")

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(torch: Any, numpy: Any, seed: int) -> None:
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, scheduler, device, torch, functional, tqdm):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for data, target in tqdm(loader, desc="train", leave=False):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()

        batch_size = target.size(0)
        total_loss += loss.item() * batch_size
        total_correct += output.argmax(dim=1).eq(target).sum().item()
        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "accuracy": 100.0 * total_correct / total_samples,
    }


def evaluate(model, loader, device, torch, functional):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            batch_size = target.size(0)
            total_loss += functional.cross_entropy(
                output, target, reduction="sum"
            ).item()
            total_correct += output.argmax(dim=1).eq(target).sum().item()
            total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "accuracy": 100.0 * total_correct / total_samples,
    }


def save_checkpoint(
    path: Path,
    epoch: int,
    model,
    optimizer,
    scheduler,
    best_accuracy: float,
    config: dict[str, Any],
    torch,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_accuracy": best_accuracy,
        "config": config,
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def load_checkpoint(path: Path, model, optimizer, scheduler, device, torch):
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    return payload["epoch"] + 1, payload["best_accuracy"]


def run_training(args) -> None:
    import numpy as np
    import torch
    import torch.nn.functional as functional
    from tqdm import tqdm

    from main_mnist import (
        MNISTAlbumentations,
        Net,
        test_transforms,
        train_transforms,
    )

    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")

    args.data_dir = Path(args.data_dir)
    args.output_dir = Path(args.output_dir)
    device = choose_device(torch, args.device)
    seed_everything(torch, np, args.seed)

    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = checkpoint_paths(args.output_dir)
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key != "dry_run"
    }

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type in {"cuda", "mps"},
    }
    train_loader = torch.utils.data.DataLoader(
        MNISTAlbumentations(
            args.data_dir,
            train=True,
            download=True,
            transform=train_transforms,
        ),
        shuffle=True,
        **loader_kwargs,
    )
    test_loader = torch.utils.data.DataLoader(
        MNISTAlbumentations(
            args.data_dir,
            train=False,
            download=True,
            transform=test_transforms,
        ),
        shuffle=False,
        **loader_kwargs,
    )

    model = Net().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.max_lr / args.initial_div,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.max_lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=args.warmup_pct,
        div_factor=args.initial_div,
        final_div_factor=args.final_div,
    )

    start_epoch = 1
    best_accuracy = -1.0
    if args.resume is not None:
        start_epoch, best_accuracy = load_checkpoint(
            Path(args.resume), model, optimizer, scheduler, device, torch
        )
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    metrics_path = args.output_dir / "metrics.jsonl"
    if start_epoch == 1:
        metrics_path.write_text("", encoding="utf-8")

    print(f"device={device}")
    print(f"data_dir={args.data_dir}")
    print(f"output_dir={args.output_dir}")
    print(f"parameters={sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, torch, functional, tqdm
        )
        test_metrics = evaluate(model, test_loader, device, torch, functional)
        current_accuracy = test_metrics["accuracy"]
        is_best = current_accuracy > best_accuracy
        best_accuracy = max(best_accuracy, current_accuracy)

        metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "test_loss": test_metrics["loss"],
            "test_accuracy": current_accuracy,
            "best_accuracy": best_accuracy,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics) + "\n")

        save_checkpoint(
            paths["last"],
            epoch,
            model,
            optimizer,
            scheduler,
            best_accuracy,
            config,
            torch,
        )
        if is_best:
            save_checkpoint(
                paths["best"],
                epoch,
                model,
                optimizer,
                scheduler,
                best_accuracy,
                config,
                torch,
            )

        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_metrics['loss']:.5f} "
            f"train_accuracy={train_metrics['accuracy']:.2f}% "
            f"test_loss={test_metrics['loss']:.5f} "
            f"test_accuracy={current_accuracy:.2f}%"
        )

    print(f"best_checkpoint={paths['best']}")
    print(f"last_checkpoint={paths['last']}")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        print(f"data_dir={args.data_dir}")
        print(f"output_dir={args.output_dir}")
        print(f"checkpoints={checkpoint_paths(args.output_dir)}")
        return
    run_training(args)


if __name__ == "__main__":
    main()
