"""Run the saved INT8 MNIST model on a test image or the full test set."""

import argparse
import os
import warnings
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import torch

from main_mnist import MNISTAlbumentations, Net, test_transforms
from quantize import accuracy, build_quantized_model


PROJECT_ROOT = Path(__file__).resolve().parent


def load_int8_model(checkpoint_path):
    with warnings.catch_warnings():
        # PyTorch warns about the legacy quantization API used by this checkpoint.
        warnings.filterwarnings("ignore", message=r"TypedStorage is deprecated.*")
        warnings.filterwarnings("ignore", message=r"torch\.quantize_per_tensor, torch\.quantize_per_channel.*")
        warnings.filterwarnings("ignore", message=r"Please use quant_min and quant_max.*")
        warnings.filterwarnings("ignore", message=r"torch\.ao\.quantization is deprecated.*")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        prepared = build_quantized_model(Net(), torch, checkpoint["backend"])
        with torch.no_grad():
            prepared(torch.zeros(1, 1, 28, 28))
        model = torch.ao.quantization.convert(prepared, inplace=False).eval()
        model.load_state_dict(checkpoint["state_dict"])
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs/pool-stride2-fp32-int8/model_int8.pt",
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--accuracy", action="store_true", help="also evaluate all test images")
    args = parser.parse_args()

    model = load_int8_model(args.checkpoint)
    dataset = MNISTAlbumentations(
        args.data_dir, train=False, download=True, transform=test_transforms
    )
    if not 0 <= args.index < len(dataset):
        parser.error(f"--index must be between 0 and {len(dataset) - 1}")

    image, label = dataset[args.index]
    with torch.no_grad():
        prediction = model(image.unsqueeze(0)).argmax(dim=1).item()
    print(f"index={args.index} label={int(label)} prediction={prediction}")

    if args.accuracy:
        loader = torch.utils.data.DataLoader(dataset, batch_size=512)
        print(f"test_accuracy={accuracy(model, loader, torch.device('cpu'), torch):.2f}%")


if __name__ == "__main__":
    main()
