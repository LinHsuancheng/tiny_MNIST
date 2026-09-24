"""Post-training static INT8 accuracy test for the compact MNIST model."""

from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs/checkpoints/best.pt",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT_ROOT / "data"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs/quantized/model_int8.pt",
    )
    parser.add_argument("--calibration-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--backend", choices=("fbgemm", "qnnpack"), default="fbgemm")
    return parser


def fuse_conv_bn_relu(model, torch):
    for block_name in ("conv1", "conv2", "conv3", "conv4", "conv5"):
        torch.ao.quantization.fuse_modules(
            getattr(model, block_name), [["0", "1", "2"]], inplace=True
        )
    return model


def build_quantized_model(float_model, torch, backend):
    class QuantizedWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.quant = torch.ao.quantization.QuantStub()
            self.model = model
            self.dequant = torch.ao.quantization.DeQuantStub()

        def forward(self, inputs):
            return self.dequant(self.model(self.quant(inputs)))

    torch.backends.quantized.engine = backend
    float_model.eval()
    fuse_conv_bn_relu(float_model, torch)
    quantized_model = QuantizedWrapper(float_model).eval()
    quantized_model.qconfig = torch.ao.quantization.get_default_qconfig(backend)
    prepared_model = torch.ao.quantization.prepare(quantized_model, inplace=False)
    return prepared_model


def make_loader(torch, dataset_cls, transform, root, train, batch_size, num_workers):
    dataset = dataset_cls(
        root,
        train=train,
        download=False,
        transform=transform,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=False,
    )


def calibrate(model, loader, device, torch, calibration_batches):
    model.eval()
    with torch.no_grad():
        for batch_index, (data, _) in enumerate(loader):
            model(data.to(device))
            if batch_index + 1 >= calibration_batches:
                break


def accuracy(model, loader, device, torch):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            output = model(data.to(device))
            correct += output.argmax(dim=1).eq(target.to(device)).sum().item()
            total += target.size(0)
    return 100.0 * correct / total


def load_float_model(torch, checkpoint_path):
    from main_mnist import Net

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = Net()
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def run(args) -> None:
    import torch

    from main_mnist import MNISTAlbumentations, test_transforms

    if args.calibration_batches < 1:
        raise ValueError("--calibration-batches must be at least 1")

    device = torch.device("cpu")
    float_model = load_float_model(torch, args.checkpoint).to(device).eval()
    test_loader = make_loader(
        torch,
        MNISTAlbumentations,
        test_transforms,
        args.data_dir,
        train=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    calibration_loader = make_loader(
        torch,
        MNISTAlbumentations,
        test_transforms,
        args.data_dir,
        train=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    fp32_accuracy = accuracy(float_model, test_loader, device, torch)
    prepared_model = build_quantized_model(float_model, torch, args.backend)
    calibrate(
        prepared_model,
        calibration_loader,
        device,
        torch,
        args.calibration_batches,
    )
    int8_model = torch.ao.quantization.convert(prepared_model, inplace=False)
    int8_accuracy = accuracy(int8_model, test_loader, device, torch)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": int8_model.state_dict(),
            "backend": args.backend,
            "calibration_batches": args.calibration_batches,
            "fp32_accuracy": fp32_accuracy,
            "int8_accuracy": int8_accuracy,
            "accuracy_drop": fp32_accuracy - int8_accuracy,
        },
        args.output,
    )

    print(f"fp32_accuracy={fp32_accuracy:.2f}%")
    print(f"int8_accuracy={int8_accuracy:.2f}%")
    print(f"accuracy_drop={fp32_accuracy - int8_accuracy:.2f} percentage points")
    print(f"int8_checkpoint={args.output}")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
