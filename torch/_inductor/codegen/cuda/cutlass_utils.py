# mypy: allow-untyped-defs
import atexit
import functools
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from typing_extensions import TypeIs

import sympy

import torch
from torch._inductor.runtime.runtime_utils import dynamo_timed
from torch._inductor.utils import clear_on_fresh_cache
from torch.utils._ordered_set import OrderedSet

from ... import config
from ...ir import Layout
from ...runtime.runtime_utils import cache_dir
from ...virtualized import V
from ..cpp_utils import DTYPE_TO_CPP
from .cuda_env import get_cuda_arch, get_cuda_version


log = logging.getLogger(__name__)

CUTLASS_OPERATION_KIND: str = "gemm"
ACCUMULATOR_DTYPES: OrderedSet[torch.dtype] = OrderedSet([torch.float, torch.int32])
XW_DTYPES: OrderedSet[torch.dtype] = OrderedSet(
    [torch.half, torch.bfloat16, torch.float8_e4m3fn, torch.int8, torch.float8_e5m2]
)


@atexit.register
def move_cutlass_compiled_cache() -> None:
    """Move CUTLASS compiled cache file to the cache directory if it exists."""
    if not try_import_cutlass.cache_info().currsize > 0:
        return

    import cutlass_cppgen  # type: ignore[import-not-found]

    # Check if the CACHE_FILE attribute exists in cutlass_cppgen and if the file exists
    if not hasattr(cutlass_cppgen, "CACHE_FILE") or not os.path.exists(
        cutlass_cppgen.CACHE_FILE
    ):
        return

    try:
        filename = os.path.basename(cutlass_cppgen.CACHE_FILE)
        shutil.move(cutlass_cppgen.CACHE_FILE, os.path.join(cache_dir(), filename))
        log.debug("Moved CUTLASS compiled cache file to %s", cache_dir())
    except OSError:
        log.warning("Failed to move CUTLASS compiled cache file", exc_info=True)


def _rename_cutlass_import(content: str, cutlass_modules: list[str]) -> str:
    for cutlass_module in cutlass_modules:
        content = content.replace(
            f"from {cutlass_module} import ",
            f"from cutlass_library.{cutlass_module} import ",
        )
    return content


@functools.cache
def try_import_cutlass() -> bool:
    """
    We want to support three ways of passing in CUTLASS:
    1. fbcode, handled by the internal build system.
    2. User specifies cutlass_dir. The default is ../third_party/cutlass/,
       which is the directory when developers build from source.
    """
    if config.is_fbcode():
        try:
            import cutlass_cppgen  # type: ignore[import-not-found]  # noqa: F401
            import cutlass_library  # type: ignore[import-not-found]
        except ImportError as e:
            log.warning(  # noqa: G200
                "Failed to import CUTLASS packages in fbcode: %s, ignoring the CUTLASS backend.",
                str(e),
            )
            return False

        return True

    # Copy CUTLASS python scripts to a temp dir and add the temp dir to Python search path.
    # This is a temporary hack to avoid CUTLASS module naming conflicts.
    # TODO(ipiszy): remove this hack when CUTLASS solves Python scripts packaging structure issues.

    # TODO(mlazos): epilogue visitor tree currently lives in python/cutlass,
    # but will be moved to python/cutlass_library in the future (later 2025)
    def path_join(path0, path1):
        return os.path.abspath(os.path.join(path0, path1))

    # contains both cutlass and cutlass_library
    # we need cutlass for eVT
    cutlass_python_path = path_join(config.cuda.cutlass_dir, "python")
    torch_root = os.path.abspath(os.path.dirname(torch.__file__))
    mock_src_path = os.path.join(
        torch_root,
        "_inductor",
        "codegen",
        "cuda",
        "cutlass_lib_extensions",
        "cutlass_mock_imports",
    )

    cutlass_library_src_path = path_join(cutlass_python_path, "cutlass_library")
    cutlass_cppgen_src_path = path_join(cutlass_python_path, "cutlass_cppgen")
    pycute_src_path = path_join(cutlass_python_path, "pycute")

    tmp_cutlass_full_path = os.path.abspath(os.path.join(cache_dir(), "torch_cutlass"))

    dst_link_library = path_join(tmp_cutlass_full_path, "cutlass_library")
    dst_link_cutlass_cppgen = path_join(tmp_cutlass_full_path, "cutlass_cppgen")
    dst_link_pycute = path_join(tmp_cutlass_full_path, "pycute")

    # mock modules to import cutlass
    mock_modules = ["cuda", "scipy", "pydot"]

    if os.path.isdir(cutlass_python_path):
        if tmp_cutlass_full_path not in sys.path:

            def link_and_append(dst_link, src_path, parent_dir):
                if os.path.lexists(dst_link):
                    assert os.path.islink(dst_link), (
                        f"{dst_link} is not a symlink. Try to remove {dst_link} manually and try again."
                    )
                    assert os.path.realpath(os.readlink(dst_link)) == os.path.realpath(
                        src_path,
                    ), f"Symlink at {dst_link} does not point to {src_path}"
                else:
                    os.makedirs(parent_dir, exist_ok=True)
                    os.symlink(src_path, dst_link)

                if parent_dir not in sys.path:
                    sys.path.append(parent_dir)

            link_and_append(
                dst_link_library, cutlass_library_src_path, tmp_cutlass_full_path
            )
            link_and_append(
                dst_link_cutlass_cppgen, cutlass_cppgen_src_path, tmp_cutlass_full_path
            )
            link_and_append(dst_link_pycute, pycute_src_path, tmp_cutlass_full_path)

            for module in mock_modules:
                link_and_append(
                    path_join(tmp_cutlass_full_path, module),  # dst_link
                    path_join(mock_src_path, module),  # src_path
                    tmp_cutlass_full_path,  # parent
                )

        try:
            import cutlass_cppgen  # type: ignore[import-not-found]  # noqa: F401, F811
            import cutlass_library.generator  # noqa: F401
            import cutlass_library.library  # noqa: F401
            import cutlass_library.manifest  # noqa: F401
            import pycute  # type: ignore[import-not-found]  # noqa: F401

            return True
        except ImportError as e:
            log.debug(  # noqa: G200
                "Failed to import CUTLASS packages: %s, ignoring the CUTLASS backend.",
                str(e),
            )
    else:
        log.debug(
            "Failed to import CUTLASS packages: CUTLASS repo does not exist: %s",
            cutlass_python_path,
        )
    return False


@functools.lru_cache(8)
def _normalize_cuda_arch(arch: str) -> str:
    arch_num = arch
    if isinstance(arch, str):
        digits = "".join(ch for ch in arch if ch.isdigit())
        if not digits:
            raise ValueError(f"Unrecognized cuda arch: {arch}")
        arch_num = int(digits)
    else:
        arch_num = int(arch)

    if arch_num > 103:
        log.warning("Detected CUDA architecture > 103: %s. Please file an issue.", arch)
        return str(arch_num)
    if arch_num >= 103:
        return "103"
    if arch_num >= 100:
        return "100"
    if arch_num >= 90:
        return "90"
    if arch_num >= 80:
        return "80"
    if arch_num >= 75:
        return "75"
    if arch_num >= 70:
        return "70"
    raise NotImplementedError(f"Unsupported cuda arch: {arch}")


@dataclass
class CUTLASSArgs:
    """
    CUTLASS args used to initialize a CUTLASS Manifest.
    """

    architectures: Optional[str] = None
    cuda_version: Optional[str] = None
    instantiation_level: Optional[str] = None
    operations: Optional[str] = None

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


@clear_on_fresh_cache
@functools.cache
def _gen_ops_cached(arch, version) -> dict[Any, Any]:
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
        return {}
    arch = _normalize_cuda_arch(arch)
    gen_arch = (
        "100" if arch == "103" else arch
    )  # CUTLASS SM103 generator only covers NVFB4; fallback to SM100 set
    instantiation_level: str = config.cuda.cutlass_instantiation_level
    args = CUTLASSArgs(
        architectures=gen_arch,
        cuda_version=version,
        instantiation_level=instantiation_level,
        operations=CUTLASS_OPERATION_KIND,
    )
    manifest = cutlass_manifest.Manifest(args)

    start_time = time.time()
    if gen_arch == "100":
        if hasattr(cutlass_generator, "GenerateSM100"):
            cutlass_generator.GenerateSM100(manifest, args.cuda_version)
        cutlass_generator.GenerateSM90(manifest, args.cuda_version)
    else:
        try:
            func = getattr(cutlass_generator, "GenerateSM" + gen_arch)
            func(manifest, args.cuda_version)
        except AttributeError as e:
            raise NotImplementedError(
                "Arch " + gen_arch + " is not supported by current cutlass lib."
            ) from e

    log.info(
        "CUTLASS library generated a dict of %d operation kinds in %.2f seconds",
        len(manifest.operations),
        time.time() - start_time,
    )
    return manifest.operations


def gen_ops() -> dict[Any, Any]:
    """
    Generates all supported CUTLASS operations.
    """
    with dynamo_timed("cutlass_utils.gen_ops"):
        arch = get_cuda_arch()
        version = get_cuda_version()
        return _gen_ops_cached(arch, version)


DTYPE_TO_CUTLASS_TYPE = {
    **DTYPE_TO_CPP,
    torch.float16: "__half",
    torch.bfloat16: "__nv_bfloat16",
    torch.float8_e4m3fn: "__nv_fp8_e4m3",
    torch.float8_e5m2: "__nv_fp8_e5m2",
}


@functools.lru_cache(32)
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


@functools.lru_cache(32)
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
    elif torch_dtype == torch.float8_e4m3fn:
        return cutlass_dtype == cutlass_library.library.DataType.e4m3
    elif torch_dtype == torch.float8_e5m2:
        return cutlass_dtype == cutlass_library.library.DataType.e5m2
    else:
        return False


def get_accumulator_dtype(
    input_torch_dtypes: list[torch.dtype],
) -> Optional[torch.dtype]:
    """
    Given a pair of input torch dtypes, returns the inferred accumulator torch dtype.
    """

    assert OrderedSet(input_torch_dtypes) <= XW_DTYPES, (
        f"{input_torch_dtypes=} is not supported"
    )

    if len(input_torch_dtypes) != 2:
        return None

    if OrderedSet(input_torch_dtypes) == OrderedSet(
        [torch.float8_e5m2, torch.float8_e4m3fn]
    ):
        return torch.float

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

    if torch_dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
        accumulator_dtype = torch.float
    elif torch_dtype == torch.int8:
        accumulator_dtype = torch.int32
    else:
        raise NotImplementedError(f"Unsupported data types: {input_torch_dtypes=}")

    assert accumulator_dtype in ACCUMULATOR_DTYPES, (
        f"{accumulator_dtype=} is not supported"
    )
    return accumulator_dtype


@functools.lru_cache(32)
def get_alignments(torch_dtype: torch.dtype) -> list[int]:
    """
    Returns all possible valid CUTLASS alignments in terms of the number of elements for a given dtype.
    CUTLASS gemm / conv SM80 APIs support 16 bytes max alignment, and 2 bytes min alignment.
    """

    if torch_dtype in (torch.half, torch.bfloat16):
        return [8, 4, 2, 1]
    elif torch_dtype == torch.float:
        return [4, 2, 1]
    elif torch_dtype in (
        torch.uint8,
        torch.int8,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
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

    def is_static_int(number: object) -> TypeIs[int | sympy.Integer]:
        return isinstance(number, (int | sympy.Integer))

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

        def my_compile(
            source_code, dst_file_ext, extra_args: Optional[list[str]] = None
        ):
            self.sources.append(source_code)
            return _compile_method_orig(source_code, dst_file_ext)

        # pyrefly: ignore [bad-assignment]
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
