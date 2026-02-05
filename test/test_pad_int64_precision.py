import pytest
import torch
import torch.nn.functional as F


def test_pad_int64_large_value_raises():
    x = torch.tensor([1], dtype=torch.int64)
    v = 2**53 + 1
    with pytest.raises(RuntimeError, match="2\\^53"):
        F.pad(x, (0, 1), value=v)


def test_pad_int64_max_value_raises():
    x = torch.tensor([1], dtype=torch.int64)
    v = torch.iinfo(torch.int64).max
    with pytest.raises(RuntimeError):
        F.pad(x, (0, 1), value=v)


def test_pad_int64_safe_value_works():
    x = torch.tensor([1], dtype=torch.int64)
    v = 2**53 - 1
    out = F.pad(x, (0, 1), value=v)
    assert out[1].item() == v


def test_pad_int64_negative_large_value_raises():
    x = torch.tensor([1], dtype=torch.int64)
    v = -(2**53 + 1)
    with pytest.raises(RuntimeError, match="2\\^53"):
        F.pad(x, (0, 1), value=v)
