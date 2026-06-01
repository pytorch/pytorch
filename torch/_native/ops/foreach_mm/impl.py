"""
Python override for aten::_foreach_mm. Supports bf16 and fp32.

Dispatch (CUDA, SM90+):

  bf16 + nvmath + N,K 16-byte aligned?
  |-- YES --> nvmath cublasLt grouped GEMM (uniform and mixed shapes)
  |
  bf16/fp32 + uniform K,N + aligned + min(M,N,K) < 2048?
  |-- YES --> torch.cat + _grouped_mm with offs (CUTLASS, uniform or variable-M)
  |
  |-- else -> C++ fallback (loop of at::mm)

Notes:
- cublasLt grouped GEMM is bf16-only (fp32 returns CUBLAS_STATUS_NOT_SUPPORTED).
- cublasLt has a bug producing wrong results for non-16-byte-aligned N or K.
- At dim >= 2048 each mm saturates the GPU; loop fallback is equivalent.
"""

import warnings

import torch
from torch._prims_common import is_non_overlapping_and_dense_or_false

from ... import registry
from ...common_utils import _unavailable_reason


# nvmath cublasLt grouped GEMM availability (requires cuBLAS >= 13.2 runtime).
# Checked lazily on first use — the ForeachMMCublasLt constructor calls
# grouped_matrix_layout_create which raises FunctionNotFoundError if the
# runtime library lacks the symbol. We cache the result after first attempt.
_nvmath_available: "bool | None" = None
_nvmath_warned = False

_NVMATH_DEPS = [
    ("nvmath-python", "nvmath.bindings"),
]


def _check_nvmath_cublaslt() -> bool:
    global _nvmath_available
    if _nvmath_available is not None:
        return _nvmath_available
    _nvmath_available = _unavailable_reason(_NVMATH_DEPS) is None
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
    _SUPPORTED_DTYPES = {torch.bfloat16, torch.float32}
    if first_a.dtype not in _SUPPORTED_DTYPES or first_b.dtype not in _SUPPORTED_DTYPES:
        return False
    props = torch.cuda.get_device_properties(first_a.device)
    if props.major < 9:
        return False

    first_a_rm = first_a.stride(-1) == 1
    first_b_rm = first_b.stride(-1) == 1

    for i in range(len(self)):
        a, b = self[i], mat2[i]
        if a.size(1) != b.size(0):
            return False
        # cuBLAS errors on self-overlapping memory
        if not is_non_overlapping_and_dense_or_false(a):
            return False
        if not is_non_overlapping_and_dense_or_false(b):
            return False
        if i > 0:
            if a.dtype != first_a.dtype or b.dtype != first_b.dtype:
                return False
            if a.device != first_a.device or b.device != first_b.device:
                return False
            if (a.stride(-1) == 1) != first_a_rm or (b.stride(-1) == 1) != first_b_rm:
                return False

    M, K = first_a.shape
    N = first_b.size(1)
    elem_size = first_a.element_size()

    # nvmath cublasLt grouped GEMM: bf16-only, requires 16-byte aligned N and K
    use_nvmath = (
        first_a.dtype == torch.bfloat16
        and _check_nvmath_cublaslt()
        and N % (16 // elem_size) == 0
        and K % (16 // elem_size) == 0
    )

    if not use_nvmath:
        # _grouped_mm (CUTLASS) requires 16-byte aligned strides
        if (K * elem_size) % 16 != 0 or (N * elem_size) % 16 != 0:
            return False
        # At dim >= 2048, individual mm calls saturate the GPU
        if min(M, N, K) >= 2048:
            return False

    return True


def _foreach_mm_impl_stack(
    self: list[torch.Tensor],
    mat2: list[torch.Tensor],
) -> list[torch.Tensor]:
    # _grouped_mm with offs handles both uniform and variable-M shapes
    # (K and N must be uniform across groups)
    K = self[0].size(1)
    N = mat2[0].size(1)
    if all(a.size(1) == K for a in self) and all(
        b.size(0) == K and b.size(1) == N for b in mat2
    ):
        M_sizes = [a.size(0) for a in self]
        A_cat = torch.cat(self, dim=0)  # (sum_M, K)
        B_3d = torch.stack(mat2)  # (G, K, N)
        offs = torch.tensor(M_sizes, dtype=torch.int32, device=self[0].device).cumsum(
            0, dtype=torch.int32
        )
        out_cat = torch._grouped_mm(A_cat, B_3d, offs=offs)  # (sum_M, N)
        return list(out_cat.split(M_sizes, dim=0))

    return [torch.mm(a, b) for a, b in zip(self, mat2)]


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
    first_a = self[0]
    a_row_major = first_a.stride(-1) == 1
    b_row_major = mat2[0].stride(-1) == 1
    shapes = tuple((a.size(0), b.size(1), a.size(1)) for a, b in zip(self, mat2))
    key = (shapes, a_row_major, b_row_major, first_a.device)
    if key not in _nvmath_cache:
        _nvmath_cache[key] = _get_nvmath_cls()(
            shapes, G, a_row_major=a_row_major, b_row_major=b_row_major
        )
    return _nvmath_cache[key](self, mat2)  # pyrefly: ignore[not-callable]


def _foreach_mm_impl(
    self: list[torch.Tensor],
    mat2: list[torch.Tensor],
) -> list[torch.Tensor]:
    if self[0].dtype == torch.bfloat16:
        global _nvmath_available
        if _check_nvmath_cublaslt():
            try:
                return _foreach_mm_impl_nvmath(self, mat2)
            except Exception:
                _nvmath_available = False
        else:
            global _nvmath_warned
            if not _nvmath_warned:
                _nvmath_warned = True
                reason = _unavailable_reason(_NVMATH_DEPS) or (
                    "cublasLt >= 13.2 runtime not found"
                )
                warnings.warn(
                    f"_foreach_mm: nvmath cublasLt grouped GEMM unavailable ({reason}), "
                    f"using slower fallback.",
                    stacklevel=3,
                )

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
