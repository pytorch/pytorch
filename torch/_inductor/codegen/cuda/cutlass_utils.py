import functools
import logging
from typing import Any, List

# Import cutlass python scripts.
import generator as cutlass_generator
import library as cutlass_lib
import manifest as cutlass_manifest

import sympy

import torch

from ...ir import Layout
from .cuda_env import get_cuda_arch, get_cuda_version


class Args:
    """
    CUTLASS args used to initialize a CUTLASS Manifest.
    """

    def __init__(self, cuda_arch, cuda_version):
        self.operations = "all"
        self.build_dir = ""
        self.curr_build_dir = ""
        self.generator_target = ""
        self.architectures = cuda_arch
        self._normalize_cuda_arch()
        self.kernels = "all"
        self.ignore_kernels = ""
        self.cuda_version = cuda_version
        self.kernel_filter_file = None
        self.selected_kernel_list = None
        self.interface_dir = None
        self.filter_by_cc = True
        self.disable_full_archs_compilation = False

    def _normalize_cuda_arch(self) -> None:
        if self.architectures >= 90:
            self.architectures = "90"
        elif self.architectures >= 80:
            self.architectures = "80"
        elif self.architectures >= 75:
            self.architectures = "75"
        elif self.architectures >= 70:
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
    args = Args(arch, version)
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


def torch_dtype_match(torch_dtype0: torch.dtype, torch_dtype1: torch.dtype) -> bool:
    _MATCHED_DTYPES = [
        {torch.float, torch.float32},
        {torch.float16, torch.half},
    ]

    if torch_dtype0 == torch_dtype1:
        return True
    for matched_dtypes in _MATHED_DTYPES:
        if torch_dtype0 in matched_dtypes and torch_dtype1 in matched_dtypes:
            return True
    return False


def dtype_match(torch_dtype: torch.dtype, cutlass_dtype: cutlass_lib.DataType) -> bool:
    if torch_dtype == torch.float or torch_dtype == torch.float32:
        return (
            cutlass_dtype == cutlass_lib.DataType.f32
            or cutlass_dtype == cutlass_lib.DataType.tf32
        )
    elif torch_dtype == torch.float16 or torch_dtype == torch.half:
        return cutlass_dtype == cutlass_lib.DataType.f16
    elif torch_dtype == torch.bfloat16:
        return cutlass_dtype == cutlass_lib.DataType.bf16
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
        if not torch_dtype_match(torch_dtype, dtype):
            raise RuntimeError(f"Unmatched input dtypes: {torch_dtype=}, {dtype=}")
    if torch_dtype in {torch.float16, torch.half}:
        if torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction:
            return torch_dtype
        else:
            return torch.float
    if torch_dtype in {torch.bfloat16, torch.float, torch.float32}:
        return torch.float
    raise NotimplementedError(f"Unsupported data type: {input_torch_dtypes=}")


def get_alignments(torch_dtype: torch.dtype) -> List[int]:
    """
    Returns all possible alignments for a given torch dtype.
    """

    if torch_dtype in (torch.float16, torch.half, torch.bfloat16):
        return [8, 4, 2, 1]
    elif torch_dtype in (torch.float, torch.float32):
        return [4, 2, 1]
    else:
        raise NotImplementedError(f"unsupported {torch_dtype=} for alignments")


def get_alignment(inductor_layout: Layout) -> int:
    """
    Returns the max alignment for a given Inductor Layout.
    """

    dtype = inductor_layout.dtype
    size = inductor_layout.size
    offset = inductor_layout.offset

    def is_static_int(number):
        return isinstance(number, int) or isinstance(number, sympy.Integer)

    if is_static_int(size[-1]) and is_static_int(offset):
        alignments = get_alignments(dtype)
        for alignment in alignments:
            if int(size[-1]) % alignment == 0 and int(offset) % alignment == 0:
                return alignment

    return 1
