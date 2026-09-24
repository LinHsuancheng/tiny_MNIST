import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "train.py"
QUANTIZE_SCRIPT = PROJECT_ROOT / "quantize.py"
sys.path.insert(0, str(PROJECT_ROOT))


def test_train_help_exposes_one_shot_training_options():
    result = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--epochs" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--resume" in result.stdout


def test_quantize_help_exposes_accuracy_test_options():
    result = subprocess.run(
        [sys.executable, str(QUANTIZE_SCRIPT), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--checkpoint" in result.stdout
    assert "--calibration-batches" in result.stdout
    assert "--backend" in result.stdout


def test_default_paths_are_project_local():
    train = pytest.importorskip("train")

    parser = train.build_parser()
    args = parser.parse_args([])

    assert args.data_dir == PROJECT_ROOT / "data"
    assert args.output_dir == PROJECT_ROOT / "outputs"
    assert args.max_lr == pytest.approx(1e-3)


def test_checkpoint_paths_are_explicit_and_stable():
    train = pytest.importorskip("train")

    paths = train.checkpoint_paths(PROJECT_ROOT / "outputs")

    assert paths["best"] == PROJECT_ROOT / "outputs" / "checkpoints" / "best.pt"
    assert paths["last"] == PROJECT_ROOT / "outputs" / "checkpoints" / "last.pt"


def test_checkpoint_round_trip_restores_training_state(tmp_path):
    torch = pytest.importorskip("torch")
    train = pytest.importorskip("train")

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    checkpoint = tmp_path / "checkpoints" / "last.pt"

    train.save_checkpoint(
        checkpoint,
        epoch=3,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        best_accuracy=98.5,
        config={"seed": 14596},
        torch=torch,
    )

    restored_model = torch.nn.Linear(2, 2)
    restored_optimizer = torch.optim.SGD(
        restored_model.parameters(), lr=0.1, momentum=0.9
    )
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(
        restored_optimizer, lambda _: 1.0
    )
    next_epoch, best_accuracy = train.load_checkpoint(
        checkpoint,
        restored_model,
        restored_optimizer,
        restored_scheduler,
        torch.device("cpu"),
        torch,
    )

    assert next_epoch == 4
    assert best_accuracy == 98.5
    for expected, actual in zip(model.parameters(), restored_model.parameters()):
        assert torch.equal(expected, actual)
