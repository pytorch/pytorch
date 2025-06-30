import torch
import pytest

class ExtremeValueModel(torch.nn.Module):
    def forward(self, x):
        x = x + 1e308  # extremely large number to trigger internal error
        return x

@pytest.mark.xfail(reason="torch.compile raises UnicodeDecodeError due to extreme tensor values")
def test_extreme_values_compile():
    model = ExtremeValueModel()
    try:
        compiled_model = torch.compile(model)
        output = compiled_model(torch.tensor([1.0]))
    except UnicodeDecodeError as e:
        pytest.fail(f"torch.compile raised UnicodeDecodeError: {e}")
