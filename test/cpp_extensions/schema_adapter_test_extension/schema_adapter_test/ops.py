import torch
from torch import Tensor


def dummy_op(input, a=2) -> Tensor:
    """
    Dummy operation that fills tensor with the value of 'a'

    Args:
        input: Input tensor
        a: Integer value to fill with (default: 2)

    Returns:
        Tensor filled with a
    """
    return torch.ops.schema_adapter_test.dummy_op.default(input, a)


def register_adapter() -> None:
    """
    Register the schema adapter for dummy_op.
    This must be called before using test_dummy_op_v1.
    """
    # Import the C++ extension module
    from schema_adapter_test import _C

    _C.register_adapter()


def test_dummy_op_v1(input: Tensor) -> Tensor:
    """
    Test dummy_op with v1 schema (1 argument).
    This will use the schema adapter to convert to v2 format.

    Args:
        input: Input tensor

    Returns:
        Tensor filled with value 2 (adapter sets a=2 as default)
    """
    return torch.ops.schema_adapter_test.test_dummy_op_v1.default(input)


def test_dummy_op_v2(input: Tensor, a: int = 2) -> Tensor:
    """
    Test dummy_op with v2 schema (2 arguments).
    This calls the operation directly without adaptation.

    Args:
        input: Input tensor
        a: Integer value to fill with

    Returns:
        Tensor filled with a
    """
    return torch.ops.schema_adapter_test.test_dummy_op_v2.default(input, a)
