import sys
import os
import torch
import pytest

# Add the parent directory to the path so we can import the main file
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main_mnist import Net

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_parameter_count():
    model = Net()
    total_params = count_parameters(model)
    assert total_params == 3632, f"Expected 3632 parameters, but got {total_params}"

def test_batch_norm_usage():
    model = Net()
    has_batch_norm = any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())
    assert has_batch_norm, "Model should use BatchNormalization"

def test_same_padding_usage():
    model = Net()
    convs = [module for module in model.modules() if isinstance(module, torch.nn.Conv2d)]
    assert len(convs) == 5
    assert all(conv.padding == (1, 1) for conv in convs)

def test_fully_connected_layer():
    model = Net()
    has_fc = any(isinstance(m, torch.nn.Linear) for m in model.modules())
    assert has_fc, "Model should use at least one Fully Connected layer"
    
    # Check the specific FC layer dimensions
    fc_layer = next(m for m in model.modules() if isinstance(m, torch.nn.Linear))
    assert fc_layer.in_features == 144, f"Expected input features to be 144 (3*3*16), but got {fc_layer.in_features}"
    assert fc_layer.out_features == 10, f"Expected output features to be 10, but got {fc_layer.out_features}"


def test_convolutions_are_bn_friendly():
    model = Net()
    convs = [module for module in model.modules() if isinstance(module, torch.nn.Conv2d)]

    assert len(convs) == 5
    assert [(conv.in_channels, conv.out_channels) for conv in convs] == [
        (1, 2),
        (2, 4),
        (4, 8),
        (8, 8),
        (8, 16),
    ]
    assert all(conv.bias is None for conv in convs)


def test_downsampling_uses_strided_convolutions_without_pooling():
    model = Net()
    assert not any(isinstance(layer, torch.nn.MaxPool2d) for layer in model.modules())
    assert [getattr(model, f"conv{i}")[0].stride for i in range(1, 6)] == [
        (2, 2), (1, 1), (2, 2), (2, 2), (1, 1)
    ]


def test_downsampling_matches_top_left_sampling_in_eval_mode():
    model = Net().eval()
    for block_name, channels, size in (
        ("conv1", 1, 28),
        ("conv3", 4, 14),
        ("conv4", 8, 7),
    ):
        x = torch.randn(2, channels, size, size)
        block = getattr(model, block_name)
        conv = block[0]
        reference = torch.nn.functional.conv2d(
            x,
            conv.weight,
            bias=None,
            stride=1,
            padding=1,
        )
        sampled = block[2](block[1](reference))[:, :, ::2, ::2]
        actual = block(x)
        if block_name == "conv4":
            sampled = sampled[:, :, :3, :3]
            actual = actual[:, :, :3, :3]
        torch.testing.assert_close(actual, sampled)

def test_model_forward_pass():
    model = Net()
    batch_size = 1
    input_tensor = torch.randn(batch_size, 1, 28, 28)
    output = model(input_tensor)
    
    assert output.shape == (batch_size, 10), f"Expected output shape (1, 10), but got {output.shape}"


def test_model_forward_handles_noncontiguous_feature_maps():
    model = Net()
    model.conv5.register_forward_hook(
        lambda _module, _inputs, output: output.transpose(2, 3)
    )

    output = model(torch.randn(1, 1, 28, 28))

    assert output.shape == (1, 10)
