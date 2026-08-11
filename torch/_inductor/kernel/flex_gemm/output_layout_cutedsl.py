# mypy: allow-untyped-defs
"""CuTeDSL views for PyTorch-owned FlexGEMM output layouts."""

import cutlass
import cutlass.cute as cute


def blocked_128x4_fake_shape(batch, _rows, _cols):
    """Return the symbolic physical carrier shape used during compilation."""
    return (batch, cute.sym_int(), cute.sym_int(), 512)


@cute.jit
def blocked_128x4_output_tensor(
    tensor: cute.Tensor, transposed: cutlass.Constexpr[bool]
) -> cute.Tensor:
    """Expose a blocked carrier as a logical matrix in candidate orientation."""
    layout = cute.make_layout(
        (
            tensor.shape[0],
            ((32, 4), tensor.shape[1]),
            (4, tensor.shape[2]),
        ),
        stride=(
            tensor.stride[0],
            ((16, 4), tensor.stride[1]),
            (1, tensor.stride[2]),
        ),
    )
    if cutlass.const_expr(transposed):
        layout = cute.select(layout, mode=[0, 2, 1])
    return cute.make_tensor(tensor.iterator, layout)


def transposed_fake_shape(batch, rows, cols):
    """Return contiguous transposed carrier geometry for async compilation."""
    return (batch, cols, rows)


@cute.jit
def transposed_output_tensor(
    tensor: cute.Tensor, transposed: cutlass.Constexpr[bool]
) -> cute.Tensor:
    """Expose contiguous transposed storage in logical reduction coordinates."""
    if cutlass.const_expr(transposed):
        raise ValueError("transposed output layouts do not support swap_ab")
    return cute.make_tensor(
        tensor.iterator,
        cute.select(tensor.layout, mode=[0, 2, 1]),
    )
