"""
Python override for aten::_foreach_mm. Supports bf16 and fp32.

Dispatch prerequisites:
  CUDA SM90+, not HIP
  equal-length lists with >= 2 tensors
  same CUDA device, same bf16/fp32 dtype
  2D dense/non-overlapping tensors
  same row/column-major orientation across groups
  matching K, K,N > 0, K,N 16-byte aligned

Dispatch:
  nvmath cublasLt grouped GEMM:
  - bf16
  - nvmath installed
  |-- YES --> nvmath cublasLt grouped GEMM (uniform and mixed shapes, any G)
  |
  torch._grouped_mm with offs:
  - non-nvmath path
  - max(M,N,K) < 2048
  - uniform K,N
  - G < 1024
  |-- YES --> torch.cat + torch._grouped_mm with offs (uniform or variable-M)
  |
  fallback loop:
  - non-nvmath path
  - max(M,N,K) < 2048
  |-- YES --> fallback loop: torch.mm per input pair
  |
  |-- else -> native _foreach_mm fallback (loop of at::mm)

Notes:
- cublasLt grouped GEMM is bf16-only (fp32 returns CUBLAS_STATUS_NOT_SUPPORTED).
- At dim >= 2048 each mm saturates the GPU; native fallback is equivalent.
"""

import warnings
from functools import cache
from typing import Literal

import torch

from ... import registry
from ...common_utils import _unavailable_reason


_nvmath_warned = False

_NVMATH_DEPS = [
    ("nvmath-python", "nvmath.bindings"),
]

# cublasLtGroupedMatrixLayoutCreate, used by the nvmath grouped GEMM path, was
# introduced in cuBLAS 13.2.0 (CUDA 13.1) and is experimental as of that release.
# cublasLtGetVersion() encodes the version as 10000*major + 100*minor + patch.
# This is the cuBLAS library version, not the CUDA toolkit version.
_MIN_CUBLASLT_GROUPED_GEMM_VERSION = 130200


# Cached: availability is a process-global property of the loaded cuBLASLt.
@cache
def _check_nvmath_cublaslt() -> bool:
    if _unavailable_reason(_NVMATH_DEPS) is not None:
        return False
    try:
        from nvmath.bindings import cublasLt  # pyrefly: ignore[missing-import]

        return cublasLt.get_version() >= _MIN_CUBLASLT_GROUPED_GEMM_VERSION
    except Exception:
        return False


def _k_n_16_byte_aligned(a: torch.Tensor, b: torch.Tensor, elem_size: int) -> bool:
    return (a.size(1) * elem_size) % 16 == 0 and (b.size(1) * elem_size) % 16 == 0


def _foreach_mm_cond(
    self: list[torch.Tensor],
    mat2: list[torch.Tensor],
) -> bool:
    if len(self) < 2 or len(self) != len(mat2):
        return False

    first_a = self[0]
    first_b = mat2[0]

    if not first_a.is_cuda or not first_b.is_cuda:
        return False
    if torch.version.hip:
        return False
    if first_a.dtype != first_b.dtype:
        return False
    if first_a.dtype not in {torch.bfloat16, torch.float32}:
        return False
    props = torch.cuda.get_device_properties(first_a.device)
    if props.major < 9:
        return False

    first_a_rm = first_a.stride(-1) == 1
    first_b_rm = first_b.stride(-1) == 1

    elem_size = first_a.element_size()
    nvmath_ok = first_a.dtype == torch.bfloat16 and _check_nvmath_cublaslt()
    max_dim = 0

    for i in range(len(self)):
        a, b = self[i], mat2[i]
        if a.dim() != 2 or b.dim() != 2:
            return False
        if a.size(1) != b.size(0):
            return False
        # K or N == 0 not supported by cutlass_grouped_mm
        if a.size(1) == 0 or b.size(1) == 0:
            return False
        if not torch.ops.aten.is_non_overlapping_and_dense.default(a):
            return False
        if not torch.ops.aten.is_non_overlapping_and_dense.default(b):
            return False
        if not a.is_cuda or not b.is_cuda:
            return False
        if a.device != first_a.device or b.device != first_a.device:
            return False
        if a.dtype != first_a.dtype or b.dtype != first_a.dtype:
            return False
        K_i, N_i = a.size(1), b.size(1)
        aligned = _k_n_16_byte_aligned(a, b, elem_size)
        if nvmath_ok and not aligned:
            nvmath_ok = False
        if not nvmath_ok and not aligned:
            return False
        max_dim = max(max_dim, a.size(0), K_i, N_i)
        if i > 0:
            if (a.stride(-1) == 1) != first_a_rm or (b.stride(-1) == 1) != first_b_rm:
                return False

    if not nvmath_ok and max_dim >= 2048:
        return False

    return True


def _foreach_mm_impl_stack(
    self: list[torch.Tensor],
    mat2: list[torch.Tensor],
) -> list[torch.Tensor]:
    # Router guarantees uniform K,N and group count < _CUTLASS_GROUPED_MM_GROUP_LIMIT.
    # _grouped_mm with offs handles uniform and variable-M shapes in one kernel.
    M_sizes = [a.size(0) for a in self]
    A_cat = torch.cat(self, dim=0)  # (sum_M, K)
    B_3d = torch.stack(mat2)  # (G, K, N)
    offs = torch.tensor(M_sizes, dtype=torch.int32, device=self[0].device).cumsum(
        0, dtype=torch.int32
    )
    out_cat = torch._grouped_mm(A_cat, B_3d, offs=offs)  # (sum_M, N)
    return list(out_cat.split(M_sizes, dim=0))


_ForeachMMCublasLt = None


def _get_nvmath_cls():
    global _ForeachMMCublasLt
    if _ForeachMMCublasLt is None:
        from torch._native.ops.foreach_mm.nvmath_impl import ForeachMMCublasLt

        _ForeachMMCublasLt = ForeachMMCublasLt
    return _ForeachMMCublasLt


_nvmath_cache: dict[tuple, object] = {}


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


# torch._grouped_mm's bf16 kernel requires group_count < 1024; nvmath has no cap.
_CUTLASS_GROUPED_MM_GROUP_LIMIT = 1024

_Route = Literal["nvmath_cublaslt_grouped_mm", "cutlass_grouped_mm", "mm_loop"]


def _can_use_nvmath_cublaslt_grouped_mm(
    self: list[torch.Tensor],
    mat2: list[torch.Tensor],
) -> bool:
    # Metadata check only (bf16 + 16-byte-aligned N,K); does not check nvmath is installed
    # (_check_nvmath_cublaslt). dtype on [0]; _foreach_mm_cond ensures uniform.
    if self[0].dtype != torch.bfloat16 or mat2[0].dtype != torch.bfloat16:
        return False
    elem_size = self[0].element_size()
    return all(_k_n_16_byte_aligned(a, b, elem_size) for a, b in zip(self, mat2))


def _can_use_cutlass_grouped_mm(
    self: list[torch.Tensor],
    mat2: list[torch.Tensor],
) -> bool:
    # torch._grouped_mm needs uniform K,N; its bf16 kernel needs < cap groups
    if len(self) >= _CUTLASS_GROUPED_MM_GROUP_LIMIT:
        return False
    K, N = self[0].size(1), mat2[0].size(1)
    return all(a.size(1) == K for a in self) and all(
        b.size(0) == K and b.size(1) == N for b in mat2
    )


def _foreach_mm_route(self: list[torch.Tensor], mat2: list[torch.Tensor]) -> _Route:
    """Choose a backend route; assumes _foreach_mm_cond already validated inputs."""
    if _check_nvmath_cublaslt() and _can_use_nvmath_cublaslt_grouped_mm(self, mat2):
        return "nvmath_cublaslt_grouped_mm"
    if _can_use_cutlass_grouped_mm(self, mat2):
        return "cutlass_grouped_mm"
    return "mm_loop"


def _warn_nvmath_unavailable_once() -> None:
    global _nvmath_warned
    if _nvmath_warned:
        return
    _nvmath_warned = True
    reason = (
        _unavailable_reason(_NVMATH_DEPS)
        or "cuBLASLt lacks cublasLtGroupedMatrixLayoutCreate"
    )
    warnings.warn(
        f"_foreach_mm: nvmath cublasLt grouped GEMM unavailable ({reason}), "
        f"using slower fallback.",
        stacklevel=3,
    )


def _foreach_mm_impl(
    self: list[torch.Tensor],
    mat2: list[torch.Tensor],
) -> list[torch.Tensor]:
    route = _foreach_mm_route(self, mat2)
    if route == "nvmath_cublaslt_grouped_mm":
        return _foreach_mm_impl_nvmath(self, mat2)
    # warn once when nvmath-eligible inputs fall back only because nvmath is absent.
    if (
        not _nvmath_warned
        and _can_use_nvmath_cublaslt_grouped_mm(self, mat2)
        and not _check_nvmath_cublaslt()
    ):
        _warn_nvmath_unavailable_once()
    if route == "cutlass_grouped_mm":
        return _foreach_mm_impl_stack(self, mat2)
    return [torch.mm(a, b) for a, b in zip(self, mat2)]  # mm_loop


def register_to_dispatch() -> None:
    registry.register_op_override(
        "native",
        "aten",
        "_foreach_mm",
        "CUDA",
        cond=_foreach_mm_cond,
        impl=_foreach_mm_impl,
    )
