"""
Python override for aten::_foreach_mm.

Dispatch:

  nvmath available (nvmath-python installed, cublasLt >= 13.2 runtime)?
  |
  |-- YES --> nvmath cublasLt grouped GEMM
  |           Cached descriptors, pinned async H2D, no data copy.
  |           Best path: matches or beats C++ CUTLASS at all sizes.
  |
  |-- NO  --> min(M, N, K) >= 2048?
              |
              |-- YES --> [torch.mm(a, b) for a, b in zip(self, mat2)]
              |           At large dims each cuBLAS mm saturates the GPU.
              |           torch.stack copy would cost more than batching saves.
              |
              |-- NO  --> torch.stack -> _grouped_mm (CUTLASS 3D path)
                          Grouped GEMM gives 2-6x over loop at small dims.
"""

import torch

from ... import registry


# nvmath cublasLt grouped GEMM availability (requires cuBLAS >= 13.2 runtime).
# Checked lazily on first use — the ForeachMMCublasLt constructor calls
# grouped_matrix_layout_create which raises FunctionNotFoundError if the
# runtime library lacks the symbol. We cache the result after first attempt.
_nvmath_available: "bool | None" = None


def _check_nvmath_cublaslt() -> bool:
    global _nvmath_available
    if _nvmath_available is not None:
        return _nvmath_available
    try:
        from nvmath.bindings import cublasLt  # noqa: F401

        _nvmath_available = True
    except ImportError:
        _nvmath_available = False
    return _nvmath_available


def _foreach_mm_cond(
    self: list[torch.Tensor],
    mat2: list[torch.Tensor],
) -> bool:
    if len(self) < 2 or len(self) != len(mat2):
        return False

    first_a = self[0]
    first_b = mat2[0]

    if not first_a.is_cuda:
        return False
    if first_a.dim() != 2 or first_b.dim() != 2:
        return False
    if first_a.dtype != torch.bfloat16 or first_b.dtype != torch.bfloat16:
        return False
    # Must be either row-major (stride(-1)==1) or column-major (stride(0)==1)
    if not (first_a.stride(0) == 1 or first_a.stride(1) == 1):
        return False
    if not (first_b.stride(0) == 1 or first_b.stride(1) == 1):
        return False

    props = torch.cuda.get_device_properties(first_a.device)
    if props.major < 9:
        return False

    # All tensors must have the same shape, stride, dtype, and device
    M, K = first_a.shape
    if first_b.size(0) != K:
        return False
    for i in range(1, len(self)):
        a, b = self[i], mat2[i]
        if a.shape != first_a.shape or b.shape != first_b.shape:
            return False
        if a.stride() != first_a.stride() or b.stride() != first_b.stride():
            return False
        if a.dtype != first_a.dtype or b.dtype != first_b.dtype:
            return False
        if a.device != first_a.device or b.device != first_b.device:
            return False

    return True


def _foreach_mm_impl_stack(
    self: list[torch.Tensor],
    mat2: list[torch.Tensor],
) -> list[torch.Tensor]:
    out_3d = torch._grouped_mm(torch.stack(self), torch.stack(mat2))
    return list(out_3d.unbind(0))


_nvmath_cache: dict[tuple, object] = {}
_ForeachMMCublasLt = None


def _get_nvmath_cls():
    global _ForeachMMCublasLt
    if _ForeachMMCublasLt is None:
        from torch._native.ops.foreach_mm.nvmath_impl import ForeachMMCublasLt

        _ForeachMMCublasLt = ForeachMMCublasLt
    return _ForeachMMCublasLt


def _foreach_mm_impl_nvmath(
    self: list[torch.Tensor],
    mat2: list[torch.Tensor],
) -> list[torch.Tensor]:
    G = len(self)
    first_a, first_b = self[0], mat2[0]
    M, K = first_a.shape
    N = first_b.size(1)
    a_row_major = first_a.stride(-1) == 1
    b_row_major = first_b.stride(-1) == 1
    key = (M, N, K, G, a_row_major, b_row_major, first_a.device)
    if key not in _nvmath_cache:
        _nvmath_cache[key] = _get_nvmath_cls()(
            M, N, K, G, a_row_major=a_row_major, b_row_major=b_row_major
        )
    return _nvmath_cache[key](self, mat2)


def _foreach_mm_impl(
    self: list[torch.Tensor],
    mat2: list[torch.Tensor],
) -> list[torch.Tensor]:
    global _nvmath_available
    if _check_nvmath_cublaslt():
        try:
            return _foreach_mm_impl_nvmath(self, mat2)
        except Exception:
            _nvmath_available = False

    # At dim >= 2048, individual cuBLAS mm calls already saturate the GPU
    # and the torch.stack data copy in the _grouped_mm path hurts more
    # than the kernel launch savings help.
    M = self[0].size(0)
    K = self[0].size(1)
    N = mat2[0].size(1)
    if min(M, N, K) >= 2048:
        return [torch.mm(a, b) for a, b in zip(self, mat2)]

    return _foreach_mm_impl_stack(self, mat2)


def register_to_dispatch() -> None:
    registry.register_op_override(
        "native",
        "aten",
        "_foreach_mm",
        "CUDA",
        cond=_foreach_mm_cond,
        impl=_foreach_mm_impl,
    )
