import functools
import logging
from dataclasses import dataclass
from typing import Any, List, Optional

import sympy

# Import cutlass python scripts.
from . import HAS_CUTLASS

if HAS_CUTLASS:
    import cutlass_generator  # type: ignore[import]
    import cutlass_library  # type: ignore[import]
    import cutlass_manifest  # type: ignore[import]

import torch

from ...ir import Layout
from .cuda_env import get_cuda_arch, get_cuda_version

log = logging.getLogger(__name__)


@dataclass
class CUTLASSArgs:
    """
    CUTLASS args used to initialize a CUTLASS Manifest.
    """

    architectures: Optional[str] = None
    cuda_version: Optional[str] = None

    operations = "all"
    build_dir = ""
    curr_build_dir = ""
    generator_target = ""
    kernels = "all"
    ignore_kernels = ""
    kernel_filter_file = None
    selected_kernel_list = None
    interface_dir = None
    filter_by_cc = True
    disable_full_archs_compilation = False

    def __post_init__(self):
        if self.architectures is None or self.cuda_version is None:
            raise RuntimeError(
                f"{self.architectures=} or {self.cuda_version=} is None!"
            )
        self._normalize_cuda_arch()

    def _normalize_cuda_arch(self) -> None:
        assert self.architectures is not None
        if int(self.architectures) >= 90:
            self.architectures = "90"
        elif int(self.architectures) >= 80:
            self.architectures = "80"
        elif int(self.architectures) >= 75:
            self.architectures = "75"
        elif int(self.architectures) >= 70:
            self.architectures = "70"
        else:
            raise NotImplementedError(f"Unsupported cuda arch: {self.architectures}")


@functools.lru_cache(maxsize=1)
def gen_ops() -> List[Any]:
    """
    Generates all supported CUTLASS operations.
    """

    arch = get_cuda_arch()
    version = get_cuda_version()
    if arch is None or version is None:
        log.error(
            "Cannot detect cuda arch %s or cuda version %s. "
            "Will discard all cutlass ops. "
            "Please consider setting _inductor.cuda.arch and _inductor.cuda.version configs.",
            arch,
            version,
        )
        return list()
    args = CUTLASSArgs(architectures=arch, cuda_version=version)
    manifest = cutlass_manifest.Manifest(args)

    if arch == "90":
        cutlass_generator.GenerateSM90(manifest, args.cuda_version)
        cutlass_generator.GenerateSM80(manifest, args.cuda_version)
    else:
        try:
            func = getattr(cutlass_generator, "GenerateSM" + arch)
            func(manifest, args.cuda_version)
        except AttributeError as e:
            raise NotImplementedError(
                "Arch " + arch + " is not supported by current cutlass lib."
            ) from e

    return manifest.operations


def dtype_match(
    torch_dtype: torch.dtype, cutlass_dtype: cutlass_library.DataType
) -> bool:
    if torch_dtype == torch.float:
        return (
            cutlass_dtype == cutlass_library.DataType.f32
            or cutlass_dtype == cutlass_library.DataType.tf32
        )
    elif torch_dtype == torch.half:
        return cutlass_dtype == cutlass_library.DataType.f16
    elif torch_dtype == torch.bfloat16:
        return cutlass_dtype == cutlass_library.DataType.bf16
    else:
        return False


def get_accumulator_dtype(input_torch_dtypes: List[torch.dtype]) -> torch.dtype:
    """
    Given a list of input torch dtypes, returns the inferred accumulator torch dtype.
    """

    if len(input_torch_dtypes) == 0:
        return None
    torch_dtype = input_torch_dtypes[0]
    for dtype in input_torch_dtypes[1:]:
        if torch_dtype != dtype:
            raise RuntimeError(f"Unmatched input dtypes: {torch_dtype=}, {dtype=}")
    if torch_dtype in {torch.float16, torch.half}:
        if torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction:
            return torch_dtype
        else:
            return torch.float
    if torch_dtype in {torch.bfloat16, torch.float, torch.float32}:
        return torch.float
    raise NotImplementedError(f"Unsupported data type: {input_torch_dtypes=}")


def get_alignments(torch_dtype: torch.dtype) -> List[int]:
    """
    Returns all possible valid CUTLASS alignments in terms of the number of elements for a given dtype.
    CUTLASS gemm / conv SM80 APIs support 16 bytes max alignment, and 2 bytes min alignment.
    """

    if torch_dtype in (torch.float16, torch.half, torch.bfloat16):
        return [8, 4, 2, 1]
    elif torch_dtype in (torch.float, torch.float32):
        return [4, 2, 1]
    else:
        raise NotImplementedError(f"unsupported {torch_dtype=} for alignments")


def get_max_alignment(inductor_layout: Layout) -> int:
    """
    Returns the max alignment (in terms of number of elements) for a given Inductor Layout.
    """

    dtype = inductor_layout.dtype
    size = inductor_layout.size
    offset = inductor_layout.offset

    def is_static_int(number):
        return isinstance(number, (int, sympy.Integer))

    if is_static_int(size[-1]) and is_static_int(offset):
        alignments = get_alignments(dtype)
        for alignment in alignments:
            if int(size[-1]) % alignment == 0 and int(offset) % alignment == 0:
                return alignment

    return 1
