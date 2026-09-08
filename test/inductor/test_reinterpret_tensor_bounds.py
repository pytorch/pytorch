import pytest
import torch


def test_cpu_reinterpret_tensor_rejects_out_of_bounds_view():
    base = torch.arange(8, dtype=torch.int64)
    with pytest.raises(RuntimeError, match="out of bounds"):
        torch.ops.inductor._reinterpret_tensor(base, [1], [1], 8)


def test_cpu_reinterpret_tensor_rejects_negative_offset():
    base = torch.arange(8, dtype=torch.int64)
    with pytest.raises(RuntimeError, match="invalid storage offset"):
        torch.ops.inductor._reinterpret_tensor(base, [1], [1], -1)


def test_cpu_reinterpret_tensor_accepts_in_bounds_view():
    base = torch.arange(8, dtype=torch.int64)
    view = torch.ops.inductor._reinterpret_tensor(base, [1], [1], 7)
    assert view.item() == 7


if __name__ == "__main__":
    torch.testing._internal.common_utils.run_tests()
