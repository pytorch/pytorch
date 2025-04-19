# mypy: allow-untyped-defs
import functools
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import sympy

import torch
from torch._inductor.utils import clear_on_fresh_inductor_cache

from ... import config
from ...ir import Layout
from ...runtime.runtime_utils import cache_dir
from ...virtualized import V
from .cuda_env import get_cuda_arch, get_cuda_version


log = logging.getLogger(__name__)


def _rename_cutlass_import(content: str, cutlass_modules: list[str]) -> str:
    for cutlass_module in cutlass_modules:
        content = content.replace(
            f"from {cutlass_module} import ",
            f"from cutlass_library.{cutlass_module} import ",
        )
    return content


@functools.lru_cache(None)
def try_import_cutlass() -> bool:
    """
    We want to support three ways of passing in CUTLASS:
    1. fbcode, handled by the internal build system.
    2. pip install nvidia-cutlass, which provides the cutlass_library package
       and the header files in the cutlass_library/source directory.
    3. User specifies cutlass_dir. The default is ../third_party/cutlass/,
       which is the directory when developers build from source.
    """
    if config.is_fbcode():
        return True

    try:
        import cutlass  # type: ignore[import-not-found]
        import cutlass_library  # type: ignore[import-not-found]

        cutlass_minor_vesion = int(cutlass.__version__.split(".")[1])
        if cutlass_minor_vesion < 7:
            log.warning("CUTLASS version < 3.7 is not recommended.")

        log.debug(
            "Found cutlass_library in python search path, overriding config.cuda.cutlass_dir"
        )
        cutlass_library_dir = os.path.dirname(cutlass_library.__file__)
        assert os.path.isdir(cutlass_library_dir), (
            f"{cutlass_library_dir} is not a directory"
        )
        config.cuda.cutlass_dir = os.path.abspath(
            os.path.join(
                cutlass_library_dir,
                "source",
            )
        )
        return True
    except ModuleNotFoundError:
        log.debug(
            "cutlass_library not found in sys.path, trying to import from config.cuda.cutlass_dir"
        )

    # Copy CUTLASS python scripts to a temp dir and add the temp dir to Python search path.
    # This is a temporary hack to avoid CUTLASS module naming conflicts.
    # TODO(ipiszy): remove this hack when CUTLASS solves Python scripts packaging structure issues.

    cutlass_py_full_path = os.path.abspath(
        os.path.join(config.cuda.cutlass_dir, "python/cutlass_library")
    )
    tmp_cutlass_py_full_path = os.path.abspath(
        os.path.join(cache_dir(), "torch_cutlass_library")
    )
    dst_link = os.path.join(tmp_cutlass_py_full_path, "cutlass_library")

    if os.path.isdir(cutlass_py_full_path):
        if tmp_cutlass_py_full_path not in sys.path:
            if os.path.exists(dst_link):
                assert os.path.islink(dst_link), (
                    f"{dst_link} is not a symlink. Try to remove {dst_link} manually and try again."
                )
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


@functools.lru_cache(8)
def _normalize_cuda_arch(arch: str) -> str:
    if int(arch) >= 100:
        log.warning(
            "Detected CUDA architecture >= 100: %s. We will generate operations with "
            "GenerateSM100 (if available) and GenerateSM90. Please file an "
            "issue for any problems and feedback. ",
            arch,
        )

    if int(arch) >= 100:
        return "100"
    elif int(arch) >= 90:
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
    instantiation_level: Optional[str] = None

    operations = "all"
    build_dir = ""
    curr_build_dir = ""
    generator_target = ""
    kernels = "all"
    ignore_kernels = ""
    exclude_kernels = ""
    # TODO: these three look dead?
    kernel_filter_file: None = None
    selected_kernel_list: None = None
    interface_dir: None = None
    filter_by_cc = True
    disable_full_archs_compilation = False

    def __post_init__(self):
        if self.architectures is None or self.cuda_version is None:
            raise RuntimeError(
                f"{self.architectures=} or {self.cuda_version=} is None!"
            )
        self.architectures = _normalize_cuda_arch(self.architectures)


@clear_on_fresh_inductor_cache
@functools.lru_cache(None)
def _gen_ops_cached(arch, version) -> list[Any]:
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
        return []
    arch = _normalize_cuda_arch(arch)
    instantiation_level: str = config.cuda.cutlass_instantiation_level
    args = CUTLASSArgs(
        architectures=arch,
        cuda_version=version,
        instantiation_level=instantiation_level,
    )
    manifest = cutlass_manifest.Manifest(args)

    if arch == "100":
        try:
            from cutlass_generator import GenerateSM100  # type: ignore[import]

            GenerateSM100(manifest, args.cuda_version)
        except ImportError:
            log.warning("Cannot find GenerateSM100. Only GenerateSM90 will be used. ")
        cutlass_generator.GenerateSM90(manifest, args.cuda_version)
    elif arch == "90":
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


def gen_ops() -> list[Any]:
    """
    Generates all supported CUTLASS operations.
    """
    arch = get_cuda_arch()
    version = get_cuda_version()
    return _gen_ops_cached(arch, version)


def torch_dtype_to_cutlass_type(
    torch_dtype: torch.dtype,
) -> "cutlass_library.library.DataType":  # type: ignore[name-defined] # noqa: F821
    # Import cutlass python scripts.
    assert try_import_cutlass()
    import cutlass_library  # type: ignore[import]

    if torch_dtype == torch.float:
        return cutlass_library.library.DataType.f32
    elif torch_dtype == torch.half:
        return cutlass_library.library.DataType.f16
    elif torch_dtype == torch.bfloat16:
        return cutlass_library.library.DataType.bf16
    else:
        raise NotImplementedError(f"Unsupported data type: {torch_dtype=}")


def dtype_match(
    torch_dtype: Optional[torch.dtype],
    cutlass_dtype: "cutlass_library.library.DataType",  # type: ignore[name-defined]  # noqa: F821
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
    elif torch_dtype == torch.int8:
        return cutlass_dtype == cutlass_library.library.DataType.s8
    elif torch_dtype == torch.uint8:
        return cutlass_dtype == cutlass_library.library.DataType.u8
    elif torch_dtype == torch.int32:
        return cutlass_dtype == cutlass_library.library.DataType.s32
    else:
        return False


def get_accumulator_dtype(
    input_torch_dtypes: list[torch.dtype],
) -> Optional[torch.dtype]:
    """
    Given a pair of input torch dtypes, returns the inferred accumulator torch dtype.
    """

    if len(input_torch_dtypes) != 2:
        return None

    torch_dtype = None
    if input_torch_dtypes[0] == input_torch_dtypes[1]:
        torch_dtype = input_torch_dtypes[0]
    else:
        size0 = torch.tensor([], dtype=input_torch_dtypes[0]).element_size()
        size1 = torch.tensor([], dtype=input_torch_dtypes[1]).element_size()
        if size0 > size1:
            dtype0, dtype1 = input_torch_dtypes
        else:
            dtype1, dtype0 = input_torch_dtypes
        if dtype0 in [torch.half, torch.bfloat16] and dtype1 in [
            torch.int8,
            torch.uint8,
        ]:
            torch_dtype = dtype0

    if torch_dtype in (torch.float16, torch.bfloat16, torch.float):
        return torch.float
    if torch_dtype == torch.int8:
        return torch.int32
    raise NotImplementedError(f"Unsupported data types: {input_torch_dtypes=}")


def get_alignments(torch_dtype: torch.dtype) -> list[int]:
    """
    Returns all possible valid CUTLASS alignments in terms of the number of elements for a given dtype.
    CUTLASS gemm / conv SM80 APIs support 16 bytes max alignment, and 2 bytes min alignment.
    """

    if torch_dtype in (torch.half, torch.bfloat16):
        return [8, 4, 2, 1]
    elif torch_dtype == torch.float:
        return [4, 2, 1]
    elif torch_dtype in (torch.uint8, torch.int8):
        return [16, 8, 4, 2]
    elif torch_dtype == torch.int32:
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

    def a_factor_of(x, alignment):
        if is_static_int(x) and is_static_int(alignment):
            return x % alignment == 0
        rem = sympy.Mod(x, alignment)
        return V.graph.sizevars.evaluate_expr(sympy.Eq(rem, 0))

    try:
        contiguous_dim = inductor_layout.stride.index(1)
    except ValueError:
        # No dim with stride 1 found, return 1
        return 1
    alignments = get_alignments(dtype)
    for alignment in alignments:
        if not a_factor_of(size[contiguous_dim], alignment) or not a_factor_of(
            offset, alignment
        ):
            continue
        if all(
            (dim == contiguous_dim)
            or a_factor_of(inductor_layout.stride[dim], alignment)
            for dim in range(len(size))
        ):
            return alignment
    return 1


class CUDACompileSourceCapturingContext:
    # Helper class for Benchmarking and Testing CUTLASS Kernels in isolation.
    # Can be used to capture the sourcecode passed to CUDACodeCache.compile

    def __init__(self):
        self.sources = []
        self._compile_patch = None

    def __enter__(self, *args, **kwargs):
        import unittest.mock as mock

        import torch._inductor.codecache

        _compile_method_orig = torch._inductor.codecache.CUDACodeCache.compile

        def my_compile(source_code, dst_file_ext):
            self.sources.append(source_code)
            return _compile_method_orig(source_code, dst_file_ext)

        self._compile_patch = mock.patch(
            "torch._inductor.codecache.CUDACodeCache.compile", my_compile
        )
        self._compile_patch.__enter__(*args, **kwargs)  # type: ignore[union-attr]
        return self

    def __exit__(self, *args, **kwargs):
        self._compile_patch.__exit__(*args, **kwargs)  # type: ignore[union-attr]


def cuda_standalone_runner_compile_command(srcpath: Path, exepath: Path):
    # returns command string to compile a (captured) CUDA GEMM Kernel source to a standalone executable that's ready to run
    # Passes the correct preprocessor define to nvcc to ensure the standalone runner is enabled.
    from torch._inductor.codecache import cuda_compile_command

    extra_args = ["-DGENERATE_STANDALONE_RUNNER=1", "-DCUTLASS_DEBUG_TRACE_LEVEL=1"]
    compile_command = cuda_compile_command(
        [str(srcpath)], str(exepath), "exe", extra_args=extra_args
    )
    return compile_command
