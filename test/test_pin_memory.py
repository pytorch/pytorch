import pytest
import torch
import torch.utils.data._utils.pin_memory as pm
from torch.utils.data import DataLoader, TensorDataset


def test_pin_memory_on_tensor():
    x = torch.randn(2, 3)
    y = pm.pin_memory(x)
    assert isinstance(y, torch.Tensor)
    assert y.is_pinned()
    assert torch.allclose(x, y)


def test_pin_memory_on_nested_structure():
    nested = {
        "a": torch.zeros(5),
        "b": [torch.ones(2), ("x", torch.arange(3))],
        "c": "string",
        "d": (b"bytes", None),
    }
    out = pm.pin_memory(nested)

    assert out["a"].is_pinned()
    assert out["b"][0].is_pinned()
    assert out["b"][1][1].is_pinned()
    # non-tensor objects unchanged
    assert out["c"] == "string"
    assert out["d"][0] == b"bytes"
    assert out["d"][1] is None


def test_pin_memory_with_empty_containers():
    assert pm.pin_memory([]) == []
    assert pm.pin_memory({}) == {}
    # Note: implementation normalizes empty tuple -> empty list
    assert pm.pin_memory(()) == []


def test_dataloader_pin_memory_enabled():
    ds = TensorDataset(torch.randn(8, 4))
    dl = DataLoader(ds, batch_size=2, pin_memory=True, num_workers=0)

    for batch, in dl:
        assert batch.is_pinned()


def test_dataloader_pin_memory_disabled():
    ds = TensorDataset(torch.randn(8, 4))
    dl = DataLoader(ds, batch_size=2, pin_memory=False, num_workers=0)

    for batch, in dl:
        assert not batch.is_pinned()


def test_pin_memory_does_not_accept_device_arg():
    # The deprecated device arg should now raise
    x = torch.randn(2, 2)
    with pytest.raises(TypeError):
        pm.pin_memory(x, device="cuda:0")

