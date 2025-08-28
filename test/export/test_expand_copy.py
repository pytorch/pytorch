# test/test_expand_copy.py

import torch
import torch.nn as nn
import pytest

class ExpandModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, implicit):
        return torch.expand_copy(x, [3, 3], implicit=implicit)

def test_expand_copy_export_handles_implicit_true():
    model = ExpandModel()
    x = torch.ones([3])

    # These should succeed
    model(x, False)
    model(x, True)
    torch.export.export(model, (x, False))

    # This used to fail with TypeError; now should succeed
    try:
        torch.export.export(model, (x, True))
    except TypeError as e:
        pytest.fail(f"expand_copy export with implicit=True raised TypeError: {e}")
