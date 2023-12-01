import functools
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, List, Optional

import sympy

import torch

from ...codecache import cache_dir
from ...config import cuda as inductor_cuda_config
from ...ir import Layout
from .cuda_env import get_cuda_arch, get_cuda_version

log = logging.getLogger(__name__)


def _rename_cutlass_import(content: str, cutlass_modules: List[str]) -> str:
    for cutlass_module in cutlass_modules:
        content = content.replace(
            f"from {cutlass_module} import ",
            f"from cutlass_library.{cutlass_module} import ",
        )
    return content


def _gen_cutlass_file(
    file_name: str, cutlass_modules: List[str], src_dir: str, dst_dir: str
) -> None:
    orig_full_path = os.path.abspath(os.path.join(src_dir, file_name))
    text = ""
    with open(orig_full_path) as f:
        text = f.read()
    text = _rename_cutlass_import(text, cutlass_modules)
    dst_full_path = os.path.abspath(
        os.path.join(
            dst_dir,
            file_name,
        )
    )
    with open(dst_full_path, "w") as f:
        f.write(text)


@functools.lru_cache(None)
def try_import_cutlass() -> bool:
    # Copy CUTLASS python scripts to a temp dir and add the temp dir to Python search path.
    # This is a temporary hack to avoid CUTLASS module naming conflicts.
    # TODO(ipiszy): remove this hack when CUTLASS solves Python scripts packaging structure issues.

    cutlass_py_full_path = os.path.abspath(
        os.path.join(inductor_cuda_config.cutlass_dir, "python/cutlass_library")
    )
    tmp_cutlass_py_full_path = os.path.abspath(
        os.path.join(cache_dir(), "torch_cutlass_library")
    )
    dst_link = os.path.join(tmp_cutlass_py_full_path, "cutlass_library")

    if os.path.isdir(cutlass_py_full_path):
        if tmp_cutlass_py_full_path not in sys.path:
            if os.path.exists(dst_link):
                assert os.path.islink(
                    dst_link
                ), f"{dst_link} is not a symlink. Try to remove {dst_link} manually and try again."
                assert os.path.realpath(os.readlink(dst_link)) == os.path.realpath(
                    cutlass_py_full_path
                ), f"Symlink at {dst_link} does not point to {cutlass_py_full_path}"
            else:
                os.makedirs(tmp_cutlass_py_full_path, exist_ok=True)
                os.symlink(cutlass_py_full_path, dst_link)
            sys.path.append(tmp_cutlass_py_full_path)
        try:
            import cutlass_library.generator  # noqa: F401
            import cutlass_library.library  # noqa: F401
            import cutlass_library.manifest  # noqa: F401

            return True

        except ImportError as e:
            log.debug(
                "Failed to import CUTLASS packages: %s, ignoring the CUTLASS backend.",
                str(e),
            )
    else:
        log.debug(
            "Failed to import CUTLASS packages: CUTLASS repo does not exist: %s",
            cutlass_py_full_path,
        )
    return False


def _normalize_cuda_arch(arch: str) -> str:
    if int(arch) >= 90:
        return "90"
    elif int(arch) >= 80:
        return "80"
    elif int(arch) >= 75:
        return "75"
    elif int(arch) >= 70:
        return "70"
    else:
        raise NotImplementedError(f"Unsupported cuda arch: {arch}")


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
        self.architectures = _normalize_cuda_arch(self.architectures)


@functools.lru_cache(None)
def _gen_ops_cached(arch, version) -> List[Any]:
    # Note: Cache needs to be specific for cuda architecture and version

    # Import cutlass python scripts.
    assert try_import_cutlass()
    import cutlass_library.generator as cutlass_generator
    import cutlass_library.manifest as cutlass_manifest

    if arch is None or version is None:
        log.error(
            "Cannot detect cuda arch %s or cuda version %s. "
            "Will discard all cutlass ops. "
            "Please consider setting _inductor.cuda.arch and _inductor.cuda.version configs.",
            arch,
            version,
        )
        return list()
    arch = _normalize_cuda_arch(arch)
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


def gen_ops() -> List[Any]:
    """
    Generates all supported CUTLASS operations.
    """
    arch = get_cuda_arch()
    version = get_cuda_version()
    return _gen_ops_cached(arch, version)


def dtype_match(
    torch_dtype: Optional[torch.dtype],
    cutlass_dtype: "cutlass_library.library.DataType",  # type: ignore[name-defined]
) -> bool:
    # Import cutlass python scripts.
    assert try_import_cutlass()
    import cutlass_library

    if torch_dtype == torch.float:
        return (
            cutlass_dtype == cutlass_library.library.DataType.f32
            or cutlass_dtype == cutlass_library.library.DataType.tf32
        )
    elif torch_dtype == torch.half:
        return cutlass_dtype == cutlass_library.library.DataType.f16
    elif torch_dtype == torch.bfloat16:
        return cutlass_dtype == cutlass_library.library.DataType.bf16
    else:
        return False


def get_accumulator_dtype(
    input_torch_dtypes: List[torch.dtype],
) -> Optional[torch.dtype]:
    """
    Given a list of input torch dtypes, returns the inferred accumulator torch dtype.
    """

    if len(input_torch_dtypes) == 0:
        return None
    torch_dtype = input_torch_dtypes[0]
    for dtype in input_torch_dtypes[1:]:
        if torch_dtype != dtype:
            raise RuntimeError(f"Unmatched input dtypes: {torch_dtype=}, {dtype=}")
    if torch_dtype == torch.half:
        if torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction:
            return torch_dtype
        else:
            return torch.float
    if torch_dtype in {torch.bfloat16, torch.float}:
        return torch.float
    raise NotImplementedError(f"Unsupported data type: {input_torch_dtypes=}")


def get_alignments(torch_dtype: torch.dtype) -> List[int]:
    """
    Returns all possible valid CUTLASS alignments in terms of the number of elements for a given dtype.
    CUTLASS gemm / conv SM80 APIs support 16 bytes max alignment, and 2 bytes min alignment.
    """

    if torch_dtype in (torch.half, torch.bfloat16):
        return [8, 4, 2, 1]
    elif torch_dtype == torch.float:
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
