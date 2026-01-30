"""Shared utilities for vendored CuTeDSL kernel wrappers."""

from cutlass_api.utils import TensorWrapper


def ensure_3d_tensor_wrapper(tensor_wrapper: TensorWrapper) -> TensorWrapper:
    """Ensure tensor wrapper contains a 3D tensor for compilation.

    Vendored CuTeDSL templates are written for batched GEMM and expect 3D tensors.
    For 2D matmul, we add a batch dimension of 1 so the same kernel code works.
    """
    if (
        hasattr(tensor_wrapper, "_runtime_tensor")
        and tensor_wrapper._runtime_tensor is not None
    ):
        rt = tensor_wrapper._runtime_tensor
        if hasattr(rt, "dim") and rt.dim() == 2:
            m, k = rt.shape
            # Add batch dim of 1. Use m*k as batch stride to avoid aliasing.
            rt_3d = rt.as_strided((m, k, 1), (rt.stride(0), rt.stride(1), m * k))
            return TensorWrapper(rt_3d, alignment_bytes=16)
    return tensor_wrapper


def get_3d_runtime_tensor(tensor_wrapper: TensorWrapper):
    """Get runtime tensor as 3D for kernel execution.

    Same 2D->3D conversion as ensure_3d_tensor_wrapper, but returns the raw
    tensor directly for the fast path during kernel execution.
    """
    rt = tensor_wrapper.runtime_tensor
    if hasattr(rt, "dim") and rt.dim() == 2:
        m, k = rt.shape
        # Add batch dim of 1. Use m*k as batch stride to avoid aliasing.
        return rt.as_strided((m, k, 1), (rt.stride(0), rt.stride(1), m * k))
    return rt
