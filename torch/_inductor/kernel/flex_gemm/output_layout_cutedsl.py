# mypy: allow-untyped-defs
"""CuTeDSL implementations of PyTorch-owned FlexGEMM output layouts."""

import functools
import hashlib
import inspect

import cutlass
import cutlass.cute as cute


def blocked_128x4_output_shape(batch, _m, _n, _axis, _ndim):
    """Return the symbolic physical carrier shape for blocked 128x4 storage."""
    return (batch, cute.sym_int(), cute.sym_int(), 512)


@cute.jit
def blocked_128x4_output_tensor_impl(
    tensor: cute.Tensor, transposed: cutlass.Constexpr[bool]
) -> cute.Tensor:
    """Expose blocked storage in the requested GEMM output orientation."""
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


@cute.jit
def blocked_128x4_output_tensor(tensor: cute.Tensor) -> cute.Tensor:
    """Expose blocked storage in its ordinary logical orientation."""
    return blocked_128x4_output_tensor_impl(tensor, False)


@cute.jit
def blocked_128x4_transposed_output_tensor(tensor: cute.Tensor) -> cute.Tensor:
    """Expose blocked storage in the swapped GEMM output orientation."""
    return blocked_128x4_output_tensor_impl(tensor, True)


@functools.cache
def blocked_128x4_output_layout_key(transposed: bool) -> str:
    """Include PyTorch-owned callback source in QuACK's compiled-kernel key."""
    source = "".join(
        inspect.getsource(callback)
        for callback in (
            blocked_128x4_output_shape,
            blocked_128x4_output_tensor_impl,
            blocked_128x4_output_tensor,
            blocked_128x4_transposed_output_tensor,
        )
    )
    orientation = "transposed" if transposed else "ordinary"
    digest = hashlib.sha256(source.encode()).hexdigest()
    return f"blocked_128x4:{orientation}:{digest}"


def transposed_output_shape(batch, m, n, axis, ndim):
    """Return the contiguous carrier shape for a transposed local reduction."""
    if axis == 0:
        return (batch, n, cute.sym_int()) if ndim == 3 else (n, cute.sym_int())
    return (batch, cute.sym_int(), m) if ndim == 3 else (cute.sym_int(), m)


@cute.jit
def transposed_output_tensor(tensor: cute.Tensor) -> cute.Tensor:
    """Expose contiguous transposed storage in logical reduction coordinates."""
    return cute.make_tensor(
        tensor.iterator,
        cute.select(tensor.layout, mode=[0, 2, 1]),
    )


@functools.cache
def transposed_output_layout_key() -> str:
    """Include the transposed-layout callback source in QuACK's cache key."""
    source = "".join(
        inspect.getsource(callback)
        for callback in (transposed_output_shape, transposed_output_tensor)
    )
    return f"transposed:{hashlib.sha256(source.encode()).hexdigest()}"
