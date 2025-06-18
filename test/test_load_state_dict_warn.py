import warnings
import torch
import pytest

def test_warn_on_size_mismatch():
    m = torch.nn.Linear(4, 3)
    sd = m.state_dict()
    sd["weight"] = torch.randn(5, 4)  # force size mismatch

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # should not raise:
        m.load_state_dict(sd, strict=True, warn_on_mismatch=True)
        assert any("size mismatch" in str(wi.message).lower() for wi in w)
    # and it should still have kept the original shape
    assert m.weight.shape == torch.Size([4, 3])
