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
    # The expected total from the comments in the code is:
    # 414 + 1056 + 750 + 910 = 3130 parameters
    assert total_params == 3130, f"Expected 3130 parameters, but got {total_params}"

def test_batch_norm_usage():
    model = Net()
    has_batch_norm = any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())
    assert has_batch_norm, "Model should use BatchNormalization"

def test_dropout_usage():
    model = Net()
    has_dropout = any(isinstance(m, torch.nn.Dropout) for m in model.modules())
    assert has_dropout, "Model should use Dropout"

def test_fully_connected_layer():
    model = Net()
    has_fc = any(isinstance(m, torch.nn.Linear) for m in model.modules())
    assert has_fc, "Model should use at least one Fully Connected layer"
    
    # Check the specific FC layer dimensions
    fc_layer = next(m for m in model.modules() if isinstance(m, torch.nn.Linear))
    assert fc_layer.in_features == 90, f"Expected input features to be 90 (10*3*3), but got {fc_layer.in_features}"
    assert fc_layer.out_features == 10, f"Expected output features to be 10, but got {fc_layer.out_features}"

def test_model_forward_pass():
    model = Net()
    batch_size = 1
    input_tensor = torch.randn(batch_size, 1, 28, 28)
    output = model(input_tensor)
    
    assert output.shape == (batch_size, 10), f"Expected output shape (1, 10), but got {output.shape}" 