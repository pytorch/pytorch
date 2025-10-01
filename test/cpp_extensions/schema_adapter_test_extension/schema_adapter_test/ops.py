import torch
from torch import Tensor


def test_schema_upgrader_v1(input: Tensor) -> Tensor:
    """
    Test _test_schema_upgrader with V1 schema (PyTorch 2.6.0).
    Schema: _test_schema_upgrader(Tensor self) -> Tensor

    Uses schema adapter to simulate calling from PyTorch 2.6.0 where only
    the tensor argument was available. The adapter fills missing arguments.

    Args:
        input: Input tensor

    Returns:
        Tensor filled with value 2 (V1 behavior)
    """
    return torch.ops.schema_adapter_test.test_schema_upgrader_v1.default(input)


def test_schema_upgrader_v2(input: Tensor) -> Tensor:
    """
    Test _test_schema_upgrader with V2 schema (PyTorch 2.7.0).
    Schema: _test_schema_upgrader(Tensor self, *, bool a = True) -> Tensor

    Uses schema adapter to simulate calling from PyTorch 2.7.0 where the
    'a' parameter was added with default value True. The function uses the
    default value internally (a=True).

    Args:
        input: Input tensor

    Returns:
        Tensor filled with value 2 (fills with 2 when a=True, or -2 if a is False)
    """
    return torch.ops.schema_adapter_test.test_schema_upgrader_v2.default(input)


def test_schema_upgrader_v3(input: Tensor) -> Tensor:
    """
    Test _test_schema_upgrader with V3 schema (PyTorch 2.8.0).
    Schema: _test_schema_upgrader(Tensor self, *, bool a = True, int b = 2) -> Tensor

    Uses schema adapter to simulate calling from PyTorch 2.8.0 where the
    'b' parameter was added with default value 2. The function uses the
    default values internally (a=True, b=2).

    Args:
        input: Input tensor

    Returns:
        Tensor filled with value 2 (fills Tensor with b, default b=2)
    """
    return torch.ops.schema_adapter_test.test_schema_upgrader_v3.default(input)


def test_schema_upgrader_v4(input: Tensor) -> Tensor:
    """
    Test _test_schema_upgrader with V4 schema (PyTorch 2.9.0).
    Schema: _test_schema_upgrader(Tensor self, *, bool a = True, int b = 3) -> Tensor

    Uses schema adapter to simulate calling from PyTorch 2.9.0 where the
    default value of 'b' was changed from 2 to 3 (BC-breaking change).
    The function uses the default values internally (a=True, b=3).

    Args:
        input: Input tensor

    Returns:
        Tensor filled with value 3 (fills Tensor with b, default b=3 - BC-breaking change)
    """
    return torch.ops.schema_adapter_test.test_schema_upgrader_v4.default(input)
