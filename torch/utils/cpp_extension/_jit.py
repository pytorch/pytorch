# mypy: allow-untyped-defs
"""JIT compile path for :mod:`torch.utils.cpp_extension`.

Implements :func:`load` and :func:`load_inline` plus the ninja-driven compile
pipeline. No ``setuptools`` imports anywhere -- the Windows-only
``_get_vc_env`` helper shells out to ``vswhere.exe`` and ``vcvarsall.bat``
directly.
"""

import collections
import errno
import importlib
import importlib.abc
import importlib.util
import logging
import os
import re
import shlex
import subprocess
import sys
import sysconfig
import types
from pathlib import Path

import torch

from ..file_baton import FileBaton
from ._discovery import (
    _append_sycl_std_if_no_std_present,
    _append_sycl_targets_if_missing,
    _COMMON_SYCL_FLAGS,
    _get_icpx_version,
    _get_sycl_device_flags,
    _is_cuda_file,
    _is_sycl_file,
    _join_cuda_home,
    _join_rocm_home,
    _join_sycl_home,
    _maybe_write,
    _nt_quote_args,
    _SYCL_DLINK_FLAGS,
    _TORCH_PATH,
    _wrap_sycl_host_flags,
    check_compiler_is_gcc,
    COMMON_HIP_FLAGS,
    COMMON_HIPCC_FLAGS,
    COMMON_MSVC_FLAGS,
    COMMON_NVCC_FLAGS,
    CUDNN_HOME,
    EXEC_EXT,
    get_compiler_abi_compatibility_and_version,
    get_cxx_compiler,
    get_default_build_root,
    include_paths,
    IS_HIP_EXTENSION,
    IS_LINUX,
    IS_MACOS,
    IS_WINDOWS,
    JIT_EXTENSION_VERSIONER,
    LIB_EXT,
    MSVC_IGNORE_CUDAFE_WARNINGS,
    PLAT_TO_VCVARS,
    ROCM_VERSION,
    SHARED_FLAG,
    SUBPROCESS_DECODE_ARGS,
    TORCH_LIB_PATH,
    verify_ninja_availability,
)


logger = logging.getLogger(__name__)


def load(
    name,
    sources: str | list[str],
    extra_cflags=None,
    extra_cuda_cflags=None,
    extra_sycl_cflags=None,
    extra_ldflags=None,
    extra_include_paths=None,
    build_directory=None,
    verbose=False,
    with_cuda: bool | None = None,
    with_sycl: bool | None = None,
    is_python_module=True,
    is_standalone=False,
    keep_intermediates=True,
):
    """
    Load a PyTorch C++ extension just-in-time (JIT).

    To load an extension, a Ninja build file is emitted, which is used to
    compile the given sources into a dynamic library. This library is
    subsequently loaded into the current Python process as a module and
    returned from this function, ready for use.

    By default, the directory to which the build file is emitted and the
    resulting library compiled to is ``<tmp>/torch_extensions/<name>``, where
    ``<tmp>`` is the temporary folder on the current platform and ``<name>``
    the name of the extension. This location can be overridden in two ways.
    First, if the ``TORCH_EXTENSIONS_DIR`` environment variable is set, it
    replaces ``<tmp>/torch_extensions`` and all extensions will be compiled
    into subfolders of this directory. Second, if the ``build_directory``
    argument to this function is supplied, it overrides the entire path, i.e.
    the library will be compiled into that folder directly.

    To compile the sources, the default system compiler (``c++``) is used,
    which can be overridden by setting the ``CXX`` environment variable. To pass
    additional arguments to the compilation process, ``extra_cflags`` or
    ``extra_ldflags`` can be provided. For example, to compile your extension
    with optimizations, pass ``extra_cflags=['-O3']``. You can also use
    ``extra_cflags`` to pass further include directories.

    CUDA support with mixed compilation is provided. Simply pass CUDA source
    files (``.cu`` or ``.cuh``) along with other sources. Such files will be
    detected and compiled with nvcc rather than the C++ compiler. This includes
    passing the CUDA lib64 directory as a library directory, and linking
    ``cudart``. You can pass additional flags to nvcc via
    ``extra_cuda_cflags``, just like with ``extra_cflags`` for C++. Various
    heuristics for finding the CUDA install directory are used, which usually
    work fine. If not, setting the ``CUDA_HOME`` environment variable is the
    safest option.

    SYCL support with mixed compilation is provided. Simply pass SYCL source
    files (``.sycl``) along with other sources. Such files will be detected
    and compiled with SYCL compiler (such as Intel DPC++ Compiler) rather
    than the C++ compiler. You can pass additional flags to SYCL compiler
    via ``extra_sycl_cflags``, just like with ``extra_cflags`` for C++.
    SYCL compiler is expected to be found via system PATH environment
    variable.

    Args:
        name: The name of the extension to build. This MUST be the same as the
            name of the pybind11 module!
        sources: A list of relative or absolute paths to C++ source files.
        extra_cflags: optional list of compiler flags to forward to the build.
        extra_cuda_cflags: optional list of compiler flags to forward to nvcc
            when building CUDA sources.
        extra_sycl_cflags: optional list of compiler flags to forward to SYCL
            compiler when building SYCL sources.
        extra_ldflags: optional list of linker flags to forward to the build.
        extra_include_paths: optional list of include directories to forward
            to the build.
        build_directory: optional path to use as build workspace.
        verbose: If ``True``, turns on verbose logging of load steps.
        with_cuda: Determines whether CUDA headers and libraries are added to
            the build. If set to ``None`` (default), this value is
            automatically determined based on the existence of ``.cu`` or
            ``.cuh`` in ``sources``. Set it to `True`` to force CUDA headers
            and libraries to be included.
        with_sycl: Determines whether SYCL headers and libraries are added to
            the build. If set to ``None`` (default), this value is
            automatically determined based on the existence of ``.sycl`` in
            ``sources``. Set it to `True`` to force SYCL headers and
            libraries to be included.
        is_python_module: If ``True`` (default), imports the produced shared
            library as a Python module. If ``False``, behavior depends on
            ``is_standalone``.
        is_standalone: If ``False`` (default) loads the constructed extension
            into the process as a plain dynamic library. If ``True``, build a
            standalone executable.

    Returns:
        If ``is_python_module`` is ``True``:
            Returns the loaded PyTorch extension as a Python module.

        If ``is_python_module`` is ``False`` and ``is_standalone`` is ``False``:
            Returns nothing. (The shared library is loaded into the process as
            a side effect.)

        If ``is_standalone`` is ``True``.
            Return the path to the executable. (On Windows, TORCH_LIB_PATH is
            added to the PATH environment variable as a side effect.)

    Example:
        >>> # xdoctest: +SKIP
        >>> from torch.utils.cpp_extension import load
        >>> module = load(
        ...     name="extension",
        ...     sources=["extension.cpp", "extension_kernel.cu"],
        ...     extra_cflags=["-O2"],
        ...     verbose=True,
        ... )
    """
    return _jit_compile(
        name,
        [sources] if isinstance(sources, str) else sources,
        extra_cflags,
        extra_cuda_cflags,
        extra_sycl_cflags,
        extra_ldflags,
        extra_include_paths,
        build_directory or _get_build_directory(name, verbose),
        verbose,
        with_cuda,
        with_sycl,
        is_python_module,
        is_standalone,
        keep_intermediates=keep_intermediates,
    )


def _check_and_build_extension_h_precompiler_headers(
    extra_cflags, extra_include_paths, is_standalone=False
) -> None:
    r"""
    Precompiled Headers(PCH) can pre-build the same headers and reduce build time for pytorch load_inline modules.
    GCC official manual: https://gcc.gnu.org/onlinedocs/gcc-4.0.4/gcc/Precompiled-Headers.html
    PCH only works when built pch file(header.h.gch) and build target have the same build parameters. So, We need
    add a signature file to record PCH file parameters. If the build parameters(signature) changed, it should rebuild
    PCH file.

    Note:
    1. Windows and MacOS have different PCH mechanism. We only support Linux currently.
    2. It only works on GCC/G++.
    """
    if not IS_LINUX:
        return

    compiler = get_cxx_compiler()

    b_is_gcc = check_compiler_is_gcc(compiler)
    if b_is_gcc is False:
        return

    head_file = os.path.join(_TORCH_PATH, "include", "torch", "extension.h")
    head_file_pch = os.path.join(_TORCH_PATH, "include", "torch", "extension.h.gch")
    head_file_signature = os.path.join(
        _TORCH_PATH, "include", "torch", "extension.h.sign"
    )

    def listToString(s):
        # initialize an empty string
        string = ""
        if s is None:
            return string

        # traverse in the string
        for element in s:
            string += element + " "
        # return string
        return string

    def format_precompiler_header_cmd(
        compiler,
        head_file,
        head_file_pch,
        common_cflags,
        torch_include_dirs,
        extra_cflags,
        extra_include_paths,
    ):
        return re.sub(
            r"[ \n]+",
            " ",
            f"""
                {compiler} -x c++-header {head_file} -o {head_file_pch} {torch_include_dirs} {extra_include_paths} {extra_cflags} {common_cflags}
            """,
        ).strip()

    def command_to_signature(cmd):
        signature = cmd.replace(" ", "_")
        return signature

    def check_pch_signature_in_file(file_path, signature):
        b_exist = os.path.isfile(file_path)
        if b_exist is False:
            return False

        with open(file_path) as file:
            # read all content of a file
            content = file.read()
            # check if string present in a file
            return signature == content

    def _create_if_not_exist(path_dir) -> None:
        if not os.path.exists(path_dir):
            try:
                Path(path_dir).mkdir(parents=True, exist_ok=True)
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise RuntimeError(f"Fail to create path {path_dir}") from exc

    def write_pch_signature_to_file(file_path, pch_sign) -> None:
        _create_if_not_exist(os.path.dirname(file_path))
        with open(file_path, "w") as f:
            f.write(pch_sign)
            f.close()

    def build_precompile_header(pch_cmd) -> None:
        try:
            subprocess.check_output(shlex.split(pch_cmd), stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Compile PreCompile Header fail, command: {pch_cmd}"
            ) from e

    extra_cflags_str = listToString(extra_cflags)
    extra_include_paths_str = " ".join(
        [f"-I{include}" for include in extra_include_paths]
        if extra_include_paths
        else []
    )

    lib_include = os.path.join(_TORCH_PATH, "include")
    torch_include_dirs = [
        f"-I {lib_include}",
        # Python.h
        "-I {}".format(sysconfig.get_path("include")),
        # torch/all.h
        "-I {}".format(os.path.join(lib_include, "torch", "csrc", "api", "include")),
    ]

    torch_include_dirs_str = listToString(torch_include_dirs)

    common_cflags = []
    if not is_standalone:
        common_cflags += ["-DTORCH_API_INCLUDE_EXTENSION_H"]

    common_cflags += ["-std=c++20", "-fPIC"]
    common_cflags_str = listToString(common_cflags)

    pch_cmd = format_precompiler_header_cmd(
        compiler,
        head_file,
        head_file_pch,
        common_cflags_str,
        torch_include_dirs_str,
        extra_cflags_str,
        extra_include_paths_str,
    )
    pch_sign = command_to_signature(pch_cmd)

    if os.path.isfile(head_file_pch) is not True:
        build_precompile_header(pch_cmd)
        write_pch_signature_to_file(head_file_signature, pch_sign)
    else:
        b_same_sign = check_pch_signature_in_file(head_file_signature, pch_sign)
        if b_same_sign is False:
            build_precompile_header(pch_cmd)
            write_pch_signature_to_file(head_file_signature, pch_sign)


def remove_extension_h_precompiler_headers() -> None:
    def _remove_if_file_exists(path_file) -> None:
        if os.path.exists(path_file):
            os.remove(path_file)

    head_file_pch = os.path.join(_TORCH_PATH, "include", "torch", "extension.h.gch")
    head_file_signature = os.path.join(
        _TORCH_PATH, "include", "torch", "extension.h.sign"
    )

    _remove_if_file_exists(head_file_pch)
    _remove_if_file_exists(head_file_signature)


def load_inline(
    name,
    cpp_sources,
    cuda_sources=None,
    sycl_sources=None,
    functions=None,
    extra_cflags=None,
    extra_cuda_cflags=None,
    extra_sycl_cflags=None,
    extra_ldflags=None,
    extra_include_paths=None,
    build_directory=None,
    verbose=False,
    with_cuda=None,
    with_sycl=None,
    is_python_module=True,
    with_pytorch_error_handling=True,
    keep_intermediates=True,
    use_pch=False,
    no_implicit_headers=False,
):
    r'''
    Load a PyTorch C++ extension just-in-time (JIT) from string sources.

    This function behaves exactly like :func:`load`, but takes its sources as
    strings rather than filenames. These strings are stored to files in the
    build directory, after which the behavior of :func:`load_inline` is
    identical to :func:`load`.

    See `the
    tests <https://github.com/pytorch/pytorch/blob/master/test/test_cpp_extensions_jit.py>`_
    for good examples of using this function.

    Sources may omit two required parts of a typical non-inline C++ extension:
    the necessary header includes, as well as the (pybind11) binding code. More
    precisely, strings passed to ``cpp_sources`` are first concatenated into a
    single ``.cpp`` file. This file is then prepended with ``#include
    <torch/extension.h>``

    Furthermore, if the ``functions`` argument is supplied, bindings will be
    automatically generated for each function specified. ``functions`` can
    either be a list of function names, or a dictionary mapping from function
    names to docstrings. If a list is given, the name of each function is used
    as its docstring.

    The sources in ``cuda_sources`` are concatenated into a separate ``.cu``
    file and  prepended with ``torch/types.h``, ``cuda.h`` and
    ``cuda_runtime.h`` includes. The ``.cpp`` and ``.cu`` files are compiled
    separately, but ultimately linked into a single library. Note that no
    bindings are generated for functions in ``cuda_sources`` per se. To bind
    to a CUDA kernel, you must create a C++ function that calls it, and either
    declare or define this C++ function in one of the ``cpp_sources`` (and
    include its name in ``functions``).

    The sources in ``sycl_sources`` are concatenated into a separate ``.sycl``
    file and  prepended with ``torch/types.h``, ``sycl/sycl.hpp`` includes.
    The ``.cpp`` and ``.sycl`` files are compiled separately, but ultimately
    linked into a single library. Note that no bindings are generated for
    functions in ``sycl_sources`` per se. To bind to a SYCL kernel, you must
    create a C++ function that calls it, and either declare or define this
    C++ function in one of the ``cpp_sources`` (and include its name
    in ``functions``).



    See :func:`load` for a description of arguments omitted below.

    Args:
        cpp_sources: A string, or list of strings, containing C++ source code.
        cuda_sources: A string, or list of strings, containing CUDA source code.
        sycl_sources: A string, or list of strings, containing SYCL source code.
        functions: A list of function names for which to generate function
            bindings. If a dictionary is given, it should map function names to
            docstrings (which are otherwise just the function names).
        with_cuda: Determines whether CUDA headers and libraries are added to
            the build. If set to ``None`` (default), this value is
            automatically determined based on whether ``cuda_sources`` is
            provided. Set it to ``True`` to force CUDA headers
            and libraries to be included.
        with_sycl: Determines whether SYCL headers and libraries are added to
            the build. If set to ``None`` (default), this value is
            automatically determined based on whether ``sycl_sources`` is
            provided. Set it to ``True`` to force SYCL headers
            and libraries to be included.
        with_pytorch_error_handling: Determines whether pytorch error and
            warning macros are handled by pytorch instead of pybind. To do
            this, each function ``foo`` is called via an intermediary ``_safe_foo``
            function. This redirection might cause issues in obscure cases
            of cpp. This flag should be set to ``False`` when this redirect
            causes issues.
        no_implicit_headers: If ``True``, skips automatically adding headers, most notably
            ``#include <torch/extension.h>`` and ``#include <torch/types.h>`` lines.
            Use this option to improve cold start times when you
            already include the necessary headers in your source code. Default: ``False``.

    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CPP_EXT)
        >>> from torch.utils.cpp_extension import load_inline
        >>> source = """
        at::Tensor sin_add(at::Tensor x, at::Tensor y) {
          return x.sin() + y.sin();
        }
        """
        >>> module = load_inline(
        ...     name="inline_extension", cpp_sources=[source], functions=["sin_add"]
        ... )

    .. note::
        Since load_inline will just-in-time compile the source code, please ensure
        that you have the right toolchains installed in the runtime. For example,
        when loading C++, make sure a C++ compiler is available. If you're loading
        a CUDA extension, you will need to additionally install the corresponding CUDA
        toolkit (nvcc and any other dependencies your code has). Compiling toolchains
        are not included when you install torch and must be additionally installed.

        During compiling, by default, the Ninja backend uses #CPUS + 2 workers to build
        the extension. This may use up too many resources on some systems. One
        can control the number of workers by setting the `MAX_JOBS` environment
        variable to a non-negative number.
    '''
    build_directory = build_directory or _get_build_directory(name, verbose)

    if isinstance(cpp_sources, str):
        cpp_sources = [cpp_sources]
    cuda_sources = cuda_sources or []
    if isinstance(cuda_sources, str):
        cuda_sources = [cuda_sources]
    sycl_sources = sycl_sources or []
    if isinstance(sycl_sources, str):
        sycl_sources = [sycl_sources]

    if not no_implicit_headers:
        cpp_sources.insert(0, "#include <torch/extension.h>")

    if use_pch is True:
        # Using PreCompile Header('torch/extension.h') to reduce compile time.
        _check_and_build_extension_h_precompiler_headers(
            extra_cflags, extra_include_paths
        )
    else:
        remove_extension_h_precompiler_headers()

    # If `functions` is supplied, we create the pybind11 bindings for the user.
    # Here, `functions` is (or becomes, after some processing) a map from
    # function names to function docstrings.
    if functions is not None:
        module_def = []
        module_def.append("PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {")
        if isinstance(functions, str):
            functions = [functions]
        if isinstance(functions, list):
            # Make the function docstring the same as the function name.
            functions = {f: f for f in functions}
        elif not isinstance(functions, dict):
            raise ValueError(
                f"Expected 'functions' to be a list or dict, but was {type(functions)}"
            )
        for function_name, docstring in functions.items():
            if with_pytorch_error_handling:
                module_def.append(
                    f'm.def("{function_name}", torch::wrap_pybind_function({function_name}), "{docstring}");'
                )
            else:
                module_def.append(
                    f'm.def("{function_name}", {function_name}, "{docstring}");'
                )
        module_def.append("}")
        cpp_sources += module_def

    cpp_source_path = os.path.join(build_directory, "main.cpp")
    _maybe_write(cpp_source_path, "\n".join(cpp_sources))

    sources = [cpp_source_path]

    if cuda_sources:
        if not no_implicit_headers:
            cuda_sources.insert(0, "#include <torch/types.h>")
            cuda_sources.insert(1, "#include <cuda.h>")
            cuda_sources.insert(2, "#include <cuda_runtime.h>")

        cuda_source_path = os.path.join(build_directory, "cuda.cu")
        _maybe_write(cuda_source_path, "\n".join(cuda_sources))

        sources.append(cuda_source_path)

    if sycl_sources:
        if not no_implicit_headers:
            sycl_sources.insert(0, "#include <torch/types.h>")
            sycl_sources.insert(1, "#include <sycl/sycl.hpp>")

        sycl_source_path = os.path.join(build_directory, "sycl.sycl")
        _maybe_write(sycl_source_path, "\n".join(sycl_sources))

        sources.append(sycl_source_path)

    return _jit_compile(
        name,
        sources,
        extra_cflags,
        extra_cuda_cflags,
        extra_sycl_cflags,
        extra_ldflags,
        extra_include_paths,
        build_directory,
        verbose,
        with_cuda,
        with_sycl,
        is_python_module,
        is_standalone=False,
        keep_intermediates=keep_intermediates,
    )


def _jit_compile(
    name,
    sources,
    extra_cflags,
    extra_cuda_cflags,
    extra_sycl_cflags,
    extra_ldflags,
    extra_include_paths,
    build_directory: str,
    verbose: bool,
    with_cuda: bool | None,
    with_sycl: bool | None,
    is_python_module,
    is_standalone,
    keep_intermediates=True,
) -> types.ModuleType | str:
    if is_python_module and is_standalone:
        raise ValueError(
            "`is_python_module` and `is_standalone` are mutually exclusive."
        )

    if with_cuda is None:
        with_cuda = any(map(_is_cuda_file, sources))
    with_cudnn = any("cudnn" in f for f in extra_ldflags or [])
    if with_sycl is None:
        with_sycl = any(map(_is_sycl_file, sources))
    if with_sycl and with_cuda:
        raise AssertionError(
            "cannot have both SYCL and CUDA files in the same extension"
        )
    old_version = JIT_EXTENSION_VERSIONER.get_version(name)
    version = JIT_EXTENSION_VERSIONER.bump_version_if_changed(
        name,
        sources,
        build_arguments=[
            extra_cflags,
            extra_cuda_cflags,
            extra_ldflags,
            extra_include_paths,
        ],
        build_directory=build_directory,
        with_cuda=with_cuda,
        with_sycl=with_sycl,
        is_python_module=is_python_module,
        is_standalone=is_standalone,
    )
    if version > 0:
        if version != old_version and verbose:
            logger.info(
                "The input conditions for extension module %s have changed.", name
            )
            logger.info(
                "Bumping to version %s and re-building as %s_v%s...",
                version,
                name,
                version,
            )
        name = f"{name}_v{version}"

    baton = FileBaton(os.path.join(build_directory, "lock"))
    if baton.try_acquire():
        try:
            if version != old_version:
                if IS_HIP_EXTENSION and (with_cuda or with_cudnn):
                    from ..hipify import hipify_python
                    from ..hipify.hipify_python import GeneratedFileCleaner

                    clean_ctx_mgr = GeneratedFileCleaner(
                        keep_intermediates=keep_intermediates
                    )
                else:
                    import contextlib

                    hipify_python = None  # type: ignore[assignment]
                    clean_ctx_mgr = contextlib.nullcontext()
                with clean_ctx_mgr as clean_ctx:
                    if IS_HIP_EXTENSION and (with_cuda or with_cudnn):
                        assert hipify_python is not None  # noqa: S101
                        hipify_result = hipify_python.hipify(
                            project_directory=build_directory,
                            output_directory=build_directory,
                            header_include_dirs=(
                                extra_include_paths
                                if extra_include_paths is not None
                                else []
                            ),
                            extra_files=[os.path.abspath(s) for s in sources],
                            ignores=[
                                _join_rocm_home("*"),
                                os.path.join(_TORCH_PATH, "*"),
                            ],  # no need to hipify ROCm or PyTorch headers
                            show_detailed=verbose,
                            show_progress=verbose,
                            is_pytorch_extension=True,
                            clean_ctx=clean_ctx,
                        )

                        hipified_sources = set()
                        for source in sources:
                            s_abs = os.path.abspath(source)
                            if (
                                s_abs in hipify_result
                                and hipify_result[s_abs].hipified_path is not None
                            ):
                                hipified_s_abs = hipify_result[s_abs].hipified_path
                            else:
                                hipified_s_abs = s_abs
                            hipified_sources.add(hipified_s_abs)
                        sources = list(hipified_sources)

                    _write_ninja_file_and_build_library(
                        name=name,
                        sources=sources,
                        extra_cflags=extra_cflags or [],
                        extra_cuda_cflags=extra_cuda_cflags or [],
                        extra_sycl_cflags=extra_sycl_cflags or [],
                        extra_ldflags=extra_ldflags or [],
                        extra_include_paths=extra_include_paths or [],
                        build_directory=build_directory,
                        verbose=verbose,
                        with_cuda=with_cuda,
                        with_sycl=with_sycl,
                        is_standalone=is_standalone,
                    )
            elif verbose:
                logger.debug(
                    "No modifications detected for re-loaded extension module %s, skipping build step...",
                    name,
                )
        finally:
            baton.release()
    else:
        baton.wait()

    if verbose:
        logger.info("Loading extension module %s...", name)

    if is_standalone:
        return _get_exec_path(name, build_directory)

    return _import_module_from_library(name, build_directory, is_python_module)


def _get_hipcc_path():
    if IS_WINDOWS:
        # mypy thinks ROCM_VERSION is None but it will never be None here
        hipcc_exe = "hipcc.exe" if ROCM_VERSION >= (6, 4) else "hipcc.bat"  # type: ignore[operator]
        return _join_rocm_home("bin", hipcc_exe)
    else:
        return _join_rocm_home("bin", "hipcc")


def _write_ninja_file_and_compile_objects(
    sources: list[str],
    objects,
    cflags,
    post_cflags,
    cuda_cflags,
    cuda_post_cflags,
    cuda_dlink_post_cflags,
    sycl_cflags,
    sycl_post_cflags,
    sycl_dlink_post_cflags,
    build_directory: str,
    verbose: bool,
    with_cuda: bool | None,
    with_sycl: bool | None,
) -> None:
    verify_ninja_availability()

    compiler = get_cxx_compiler()

    get_compiler_abi_compatibility_and_version(compiler)
    if with_cuda is None:
        with_cuda = any(map(_is_cuda_file, sources))
    if with_sycl is None:
        with_sycl = any(map(_is_sycl_file, sources))
    if with_sycl and with_cuda:
        raise AssertionError(
            "cannot have both SYCL and CUDA files in the same extension"
        )
    build_file_path = os.path.join(build_directory, "build.ninja")
    if verbose:
        logger.debug("Emitting ninja build file %s...", build_file_path)

    # Create build_directory if it does not exist
    if not os.path.exists(build_directory):
        if verbose:
            logger.debug("Creating directory %s...", build_directory)
        # This is like mkdir -p, i.e. will also create parent directories.
        os.makedirs(build_directory, exist_ok=True)

    _write_ninja_file(
        path=build_file_path,
        cflags=cflags,
        post_cflags=post_cflags,
        cuda_cflags=cuda_cflags,
        cuda_post_cflags=cuda_post_cflags,
        cuda_dlink_post_cflags=cuda_dlink_post_cflags,
        sycl_cflags=sycl_cflags,
        sycl_post_cflags=sycl_post_cflags,
        sycl_dlink_post_cflags=sycl_dlink_post_cflags,
        sources=sources,
        objects=objects,
        ldflags=None,
        library_target=None,
        with_cuda=with_cuda,
        with_sycl=with_sycl,
    )
    if verbose:
        logger.info("Compiling objects...")
    _run_ninja_build(
        build_directory,
        verbose,
        # It would be better if we could tell users the name of the extension
        # that failed to build but there isn't a good way to get it here.
        error_prefix="Error compiling objects for extension",
    )


def _write_ninja_file_and_build_library(
    name,
    sources: list[str],
    extra_cflags,
    extra_cuda_cflags,
    extra_sycl_cflags,
    extra_ldflags,
    extra_include_paths,
    build_directory: str,
    verbose: bool,
    with_cuda: bool | None,
    with_sycl: bool | None,
    is_standalone: bool = False,
) -> None:
    verify_ninja_availability()

    compiler = get_cxx_compiler()

    get_compiler_abi_compatibility_and_version(compiler)
    if with_cuda is None:
        with_cuda = any(map(_is_cuda_file, sources))
    if with_sycl is None:
        with_sycl = any(map(_is_sycl_file, sources))
    if with_sycl and with_cuda:
        raise AssertionError(
            "cannot have both SYCL and CUDA files in the same extension"
        )
    extra_ldflags = _prepare_ldflags(
        extra_ldflags or [], with_cuda, with_sycl, verbose, is_standalone
    )
    build_file_path = os.path.join(build_directory, "build.ninja")
    if verbose:
        logger.debug("Emitting ninja build file %s...", build_file_path)

    # Create build_directory if it does not exist
    if not os.path.exists(build_directory):
        if verbose:
            logger.debug("Creating directory %s...", build_directory)
        # This is like mkdir -p, i.e. will also create parent directories.
        os.makedirs(build_directory, exist_ok=True)

    # NOTE: Emitting a new ninja build file does not cause re-compilation if
    # the sources did not change, so it's ok to re-emit (and it's fast).
    _write_ninja_file_to_build_library(
        path=build_file_path,
        name=name,
        sources=sources,
        extra_cflags=extra_cflags or [],
        extra_cuda_cflags=extra_cuda_cflags or [],
        extra_sycl_cflags=extra_sycl_cflags or [],
        extra_ldflags=extra_ldflags or [],
        extra_include_paths=extra_include_paths or [],
        with_cuda=with_cuda,
        with_sycl=with_sycl,
        is_standalone=is_standalone,
    )

    if verbose:
        logger.info("Building extension module %s...", name)
    _run_ninja_build(
        build_directory, verbose, error_prefix=f"Error building extension '{name}'"
    )


def _prepare_ldflags(extra_ldflags, with_cuda, with_sycl, verbose, is_standalone):
    if IS_WINDOWS:
        python_lib_path = os.path.join(sys.base_exec_prefix, "libs")

        extra_ldflags.append("c10.lib")
        if with_cuda:
            extra_ldflags.append("c10_hip.lib" if IS_HIP_EXTENSION else "c10_cuda.lib")
        if with_sycl:
            extra_ldflags.append("c10_xpu.lib")
        extra_ldflags.append("torch_cpu.lib")
        if with_cuda:
            extra_ldflags.append(
                "torch_hip.lib" if IS_HIP_EXTENSION else "torch_cuda.lib"
            )
            # /INCLUDE is used to ensure torch_cuda is linked against in a project that relies on it.
            # Related issue: https://github.com/pytorch/pytorch/issues/31611
            extra_ldflags.append("-INCLUDE:?warp_size@cuda@at@@YAHXZ")
        if with_sycl:
            extra_ldflags.append("torch_xpu.lib")
        extra_ldflags.append("torch.lib")
        extra_ldflags.append(f"/LIBPATH:{TORCH_LIB_PATH}")
        if not is_standalone:
            extra_ldflags.append("torch_python.lib")
            extra_ldflags.append(f"/LIBPATH:{python_lib_path}")

    else:
        extra_ldflags.append(f"-L{TORCH_LIB_PATH}")
        extra_ldflags.append("-lc10")
        if with_cuda:
            extra_ldflags.append("-lc10_hip" if IS_HIP_EXTENSION else "-lc10_cuda")
        if with_sycl:
            extra_ldflags.append("-lc10_xpu")
        extra_ldflags.append("-ltorch_cpu")
        if with_cuda:
            extra_ldflags.append("-ltorch_hip" if IS_HIP_EXTENSION else "-ltorch_cuda")
        if with_sycl:
            extra_ldflags.append("-ltorch_xpu")
        extra_ldflags.append("-ltorch")
        if not is_standalone:
            extra_ldflags.append("-ltorch_python")

        if is_standalone:
            extra_ldflags.append(f"-Wl,-rpath,{TORCH_LIB_PATH}")

    if with_cuda:
        if verbose:
            logger.info("Detected CUDA files, patching ldflags")
        if IS_WINDOWS and not IS_HIP_EXTENSION:
            extra_ldflags.append(f"/LIBPATH:{_join_cuda_home('lib', 'x64')}")
            extra_ldflags.append("cudart.lib")
            if CUDNN_HOME is not None:
                extra_ldflags.append(
                    f"/LIBPATH:{os.path.join(CUDNN_HOME, 'lib', 'x64')}"
                )
        elif not IS_HIP_EXTENSION:
            extra_lib_dir = "lib64"
            if not os.path.exists(_join_cuda_home(extra_lib_dir)) and os.path.exists(
                _join_cuda_home("lib")
            ):
                # 64-bit CUDA may be installed in "lib"
                # Note that it's also possible both don't exist (see _find_cuda_home) - in that case we stay with "lib64"
                extra_lib_dir = "lib"
            extra_ldflags.append(f"-L{_join_cuda_home(extra_lib_dir)}")
            extra_ldflags.append("-lcudart")
            if CUDNN_HOME is not None:
                extra_ldflags.append(f"-L{os.path.join(CUDNN_HOME, 'lib64')}")
        elif IS_HIP_EXTENSION:
            if IS_WINDOWS:
                extra_ldflags.append(f"/LIBPATH:{_join_rocm_home('lib')}")
                extra_ldflags.append("amdhip64.lib")
            else:
                extra_ldflags.append(f"-L{_join_rocm_home('lib')}")
                extra_ldflags.append("-lamdhip64")
    if with_sycl:
        if IS_WINDOWS:
            extra_ldflags.append(f"/LIBPATH:{_join_sycl_home('lib')}")
            extra_ldflags.append("sycl.lib")
        else:
            extra_ldflags.append(f"-L{_join_sycl_home('lib')}")
            extra_ldflags.append("-lsycl")
    return extra_ldflags


def _get_cuda_arch_flags(cflags: list[str] | None = None) -> list[str]:
    """
    Determine CUDA arch flags to use.

    For an arch, say "6.1", the added compile flag will be
    ``-gencode=arch=compute_61,code=sm_61``.
    For an added "+PTX", an additional
    ``-gencode=arch=compute_xx,code=compute_xx`` is added.

    See select_compute_arch.cmake for corresponding named and supported arches
    when building with CMake.
    """
    # If cflags is given, there may already be user-provided arch flags in it
    # (from `extra_compile_args`)
    if cflags is not None:
        for flag in cflags:
            if "TORCH_EXTENSION_NAME" in flag:
                continue
            if "arch" in flag:
                return []

    # Note: keep combined names ("arch1+arch2") above single names, otherwise
    # string replacement may not do the right thing
    named_arches = collections.OrderedDict(
        [
            ("Kepler+Tesla", "3.7"),
            ("Kepler", "3.5+PTX"),
            ("Maxwell+Tegra", "5.3"),
            ("Maxwell", "5.0;5.2+PTX"),
            ("Pascal", "6.0;6.1+PTX"),
            ("Volta+Tegra", "7.2"),
            ("Volta", "7.0+PTX"),
            ("Turing", "7.5+PTX"),
            ("Ampere+Tegra", "8.7"),
            ("Ampere", "8.0;8.6+PTX"),
            ("Ada", "8.9+PTX"),
            ("Hopper", "9.0+PTX"),
            ("Blackwell+Tegra", "11.0"),
            ("Blackwell", "10.0;10.3;12.0;12.1+PTX"),
        ]
    )

    supported_arches = [
        "3.5",
        "3.7",
        "5.0",
        "5.2",
        "5.3",
        "6.0",
        "6.1",
        "6.2",
        "7.0",
        "7.2",
        "7.5",
        "8.0",
        "8.6",
        "8.7",
        "8.9",
        "9.0",
        "9.0a",
        "10.0",
        "10.0a",
        "11.0",
        "11.0a",
        "10.3",
        "10.3a",
        "12.0",
        "12.0a",
        "12.1",
        "12.1a",
    ]
    valid_arch_strings = supported_arches + [s + "+PTX" for s in supported_arches]

    # The default is sm_30 for CUDA 9.x and 10.x
    # First check for an env var (same as used by the main setup.py)
    # Can be one or more architectures, e.g. "6.1" or "3.5;5.2;6.0;6.1;7.0+PTX"
    # See cmake/Modules_CUDA_fix/upstream/FindCUDA/select_compute_arch.cmake
    _arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)

    # If not given or set as native, determine what's best for the GPU / CUDA version that can be found
    if not _arch_list or _arch_list == "native":
        arch_list = []
        # the assumption is that the extension should run on any of the currently visible cards,
        # which could be of different types - therefore all archs for visible cards should be included
        for i in range(torch.cuda.device_count()):
            capability = torch.cuda.get_device_capability(i)
            supported_sm = [
                int("".join(re.findall(r"\d+", arch.split("_")[1])))
                for arch in torch.cuda.get_arch_list()
                if "sm_" in arch
            ]
            max_supported_sm = max((sm // 10, sm % 10) for sm in supported_sm)
            # Capability of the device may be higher than what's supported by the user's
            # NVCC, causing compilation error. User's NVCC is expected to match the one
            # used to build pytorch, so we use the maximum supported capability of pytorch
            # to clamp the capability.
            capability = min(max_supported_sm, capability)
            arch = f"{capability[0]}.{capability[1]}"
            if arch not in arch_list:
                arch_list.append(arch)
        arch_list = sorted(arch_list)
        arch_list[-1] += "+PTX"

        if not _arch_list:
            # Only log on rank 0 in distributed settings to avoid spam
            if (
                not torch.distributed.is_available()
                or not torch.distributed.is_initialized()
                or torch.distributed.get_rank() == 0
            ):
                arch_list_str = ";".join(arch_list)
                logger.debug(
                    "TORCH_CUDA_ARCH_LIST is not set, using TORCH_CUDA_ARCH_LIST='%s' "
                    "for visible GPU architectures. Set os.environ['TORCH_CUDA_ARCH_LIST'] to override.",
                    arch_list_str,
                )
    else:
        # Deal with lists that are ' ' separated (only deal with ';' after)
        _arch_list = _arch_list.replace(" ", ";")
        # Expand named arches
        for named_arch, archival in named_arches.items():
            _arch_list = _arch_list.replace(named_arch, archival)

        arch_list = _arch_list.split(";")

    flags = []
    for arch in arch_list:
        if arch not in valid_arch_strings:
            raise ValueError(f"Unknown CUDA arch ({arch}) or GPU not supported")
        else:
            # Handle both single and double-digit architecture versions
            version = arch.split("+")[0]  # Remove "+PTX" if present
            major, minor = version.split(".")
            num = f"{major}{minor}"
            flags.append(f"-gencode=arch=compute_{num},code=sm_{num}")
            if arch.endswith("+PTX"):
                flags.append(f"-gencode=arch=compute_{num},code=compute_{num}")

    return sorted(set(flags))


def _get_rocm_arch_flags(cflags: list[str] | None = None) -> list[str]:
    # If cflags is given, there may already be user-provided arch flags in it
    # (from `extra_compile_args`). If user also specified -fgpu-rdc or -fno-gpu-rdc, we
    # assume they know what they're doing. Otherwise, we force -fno-gpu-rdc default.
    has_gpu_rdc_flag = False
    if cflags is not None:
        has_custom_flags = False
        for flag in cflags:
            if "amdgpu-target" in flag or "offload-arch" in flag:
                has_custom_flags = True
            elif "gpu-rdc" in flag:
                has_gpu_rdc_flag = True
        if has_custom_flags:
            return [] if has_gpu_rdc_flag else ["-fno-gpu-rdc"]
    # Use same defaults as used for building PyTorch
    # Allow env var to override, just like during initial cmake build.
    _archs = os.environ.get("PYTORCH_ROCM_ARCH", None)
    if not _archs:
        arch_set = set()
        # the assumption is that the extension should run on any of the currently visible cards,
        # which could be of different types - therefore all archs for visible cards should be included
        for i in range(torch.cuda.device_count()):
            device_properties = torch.cuda.get_device_properties(i)
            if hasattr(device_properties, "gcnArchName"):
                device_arch = (device_properties.gcnArchName).split(":", 1)[0]
                arch_set.add(device_arch)

        archs = ";".join(arch_set)

        logger.warning(
            "The environment variable `PYTORCH_ROCM_ARCH` is not set, all archs for visible cards are included for compilation (%s).\n"
            "If this is not desired, please set the environment variable `PYTORCH_ROCM_ARCH` to specific architectures.",
            archs,
        )
    else:
        archs = _archs.replace(" ", ";")

    archs = archs.split(";")
    flags = [f"--offload-arch={arch}" for arch in archs]
    flags += [] if has_gpu_rdc_flag else ["-fno-gpu-rdc"]
    return flags


def _get_build_directory(name: str, verbose: bool) -> str:
    """
    Get the build directory for an extension.

    Args:
        name: The name of the extension
        verbose: Whether to print verbose information

    Returns:
        The path to the build directory
    """
    root_extensions_directory = os.environ.get("TORCH_EXTENSIONS_DIR")
    if root_extensions_directory is None:
        root_extensions_directory = get_default_build_root()
        # Determine GPU accelerator prefix based on available accelerators. Fallback to CPU.
        # Priority: ROCm/HIP > CUDA > CPU
        # Note: torch.backends.cuda.is_built() returns True for both CUDA and ROCm,
        # so we need to check torch.version.hip to distinguish them
        if torch.version.hip is not None:
            accelerator_str = f"rocm{torch.version.hip.replace('.', '')}"
        elif torch.version.cuda is not None:
            accelerator_str = f"cu{torch.version.cuda.replace('.', '')}"
        else:
            accelerator_str = "cpu"
        python_version = f"py{sys.version_info.major}{sys.version_info.minor}{getattr(sys, 'abiflags', '')}"
        build_folder = f"{python_version}_{accelerator_str}"

        root_extensions_directory = os.path.join(
            root_extensions_directory, build_folder
        )

    if verbose:
        logger.info("Using %s as PyTorch extensions root...", root_extensions_directory)

    build_directory = os.path.join(root_extensions_directory, name)
    if not os.path.exists(build_directory):
        if verbose:
            logger.debug("Creating extension directory %s...", build_directory)
        # This is like mkdir -p, i.e. will also create parent directories.
        os.makedirs(build_directory, exist_ok=True)

    return build_directory


def _get_num_workers(verbose: bool) -> int | None:
    max_jobs = os.environ.get("MAX_JOBS")
    if max_jobs is not None and max_jobs.isdigit():
        if verbose:
            logger.debug(
                "Using envvar MAX_JOBS (%s) as the number of workers...", max_jobs
            )
        return int(max_jobs)
    if verbose:
        logger.info(
            "Allowing ninja to set a default number of workers... "
            "(overridable by setting the environment variable MAX_JOBS=N)"
        )
    return None


def _get_vc_env(vc_arch: str) -> dict[str, str]:
    """Return the Visual C++ build environment for ``vc_arch``.

    Locates the latest Visual Studio install via ``vswhere.exe`` and captures
    what ``vcvarsall.bat <vc_arch>`` sets. Keys are lowercased; callers that
    merge this into ``os.environ`` should uppercase them.
    """
    if os.environ.get("DISTUTILS_USE_SDK"):
        return {k.lower(): v for k, v in os.environ.items()}

    program_files = os.environ.get(
        "ProgramFiles(x86)",
        os.environ.get("ProgramFiles", r"C:\Program Files (x86)"),
    )
    vswhere = os.path.join(
        program_files, "Microsoft Visual Studio", "Installer", "vswhere.exe"
    )
    if not os.path.exists(vswhere):
        raise RuntimeError(
            f"vswhere.exe not found at {vswhere}; install Visual Studio or the "
            "Microsoft C++ Build Tools with the 'Desktop development with C++' "
            "workload."
        )

    install_path = (
        subprocess.check_output(
            [
                vswhere,
                "-latest",
                "-prerelease",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
                "-products",
                "*",
            ]
        )
        .decode("mbcs", errors="strict")
        .strip()
    )
    if not install_path:
        raise RuntimeError(
            "No Visual Studio installation with the MSVC build tools "
            "(Microsoft.VisualStudio.Component.VC.Tools.x86.x64) was found."
        )

    vcvarsall = os.path.join(install_path, "VC", "Auxiliary", "Build", "vcvarsall.bat")
    if not os.path.exists(vcvarsall):
        raise RuntimeError(f"vcvarsall.bat not found at {vcvarsall}")

    # /u forces cmd.exe to emit output as UTF-16LE so non-ASCII paths in
    # localized Windows installs round-trip intact.
    try:
        out = subprocess.check_output(
            f'cmd /u /c "{vcvarsall}" {vc_arch} && set',
            stderr=subprocess.STDOUT,
        ).decode("utf-16le", errors="replace")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"vcvarsall.bat {vc_arch} failed:\n{exc.output}") from exc

    return {
        key.lower(): value
        for key, _, value in (line.partition("=") for line in out.splitlines())
        if key and value
    }


def _run_ninja_build(build_directory: str, verbose: bool, error_prefix: str) -> None:
    command = ["ninja", "-v"]
    num_workers = _get_num_workers(verbose)
    if num_workers is not None:
        command.extend(["-j", str(num_workers)])
    env = os.environ.copy()
    # Try to activate the vc env for the users
    if IS_WINDOWS and "VSCMD_ARG_TGT_ARCH" not in env:
        plat_name = sysconfig.get_platform()
        plat_spec = PLAT_TO_VCVARS[plat_name]
        vc_env = {k.upper(): v for k, v in _get_vc_env(plat_spec).items()}
        for k, v in env.items():
            uk = k.upper()
            if uk not in vc_env:
                vc_env[uk] = v
        env = vc_env
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        # Warning: don't pass stdout=None to subprocess.run to get output.
        # subprocess.run assumes that sys.__stdout__ has not been modified and
        # attempts to write to it by default.  However, when we call _run_ninja_build
        # from ahead-of-time cpp extensions, the following happens:
        # 1) If the stdout encoding is not utf-8, setuptools detaches __stdout__.
        #    https://github.com/pypa/setuptools/blob/7e97def47723303fafabe48b22168bbc11bb4821/setuptools/dist.py#L1110
        #    (it probably shouldn't do this)
        # 2) subprocess.run (on POSIX, with no stdout override) relies on
        #    __stdout__ not being detached:
        #    https://github.com/python/cpython/blob/c352e6c7446c894b13643f538db312092b351789/Lib/subprocess.py#L1214
        # To work around this, we pass in the fileno directly and hope that
        # it is valid.
        stdout_fileno = 1
        subprocess.run(
            command,
            shell=IS_WINDOWS and IS_HIP_EXTENSION,
            stdout=stdout_fileno if verbose else subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=build_directory,
            check=True,
            env=env,
        )
    except subprocess.CalledProcessError as e:
        # Python 2 and 3 compatible way of getting the error object.
        _, error, _ = sys.exc_info()
        # error.output contains the stdout and stderr of the build attempt.
        message = error_prefix
        # `error` is a CalledProcessError (which has an `output`) attribute, but
        # mypy thinks it's Optional[BaseException] and doesn't narrow
        if hasattr(error, "output") and error.output:  # type: ignore[union-attr]
            message += f": {error.output.decode(*SUBPROCESS_DECODE_ARGS)}"  # type: ignore[union-attr]
        raise RuntimeError(message) from e


def _get_exec_path(module_name, path):
    if IS_WINDOWS and TORCH_LIB_PATH not in os.getenv("PATH", "").split(";"):
        torch_lib_in_path = any(
            os.path.exists(p) and os.path.samefile(p, TORCH_LIB_PATH)
            for p in os.getenv("PATH", "").split(";")
        )
        if not torch_lib_in_path:
            os.environ["PATH"] = f"{TORCH_LIB_PATH};{os.getenv('PATH', '')}"
    return os.path.join(path, f"{module_name}{EXEC_EXT}")


def _import_module_from_library(module_name, path, is_python_module):
    filepath = os.path.join(path, f"{module_name}{LIB_EXT}")
    if is_python_module:
        # https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        if spec is None:
            raise AssertionError(
                f"Failed to create spec for module {module_name} at {filepath}"
            )
        module = importlib.util.module_from_spec(spec)
        if not isinstance(spec.loader, importlib.abc.Loader):
            raise AssertionError("spec.loader is not a valid importlib Loader")
        spec.loader.exec_module(module)
        return module
    else:
        torch.ops.load_library(filepath)
        return filepath


def _write_ninja_file_to_build_library(
    path,
    name,
    sources,
    extra_cflags,
    extra_cuda_cflags,
    extra_sycl_cflags,
    extra_ldflags,
    extra_include_paths,
    with_cuda,
    with_sycl,
    is_standalone,
) -> None:
    extra_cflags = [flag.strip() for flag in extra_cflags]
    extra_cuda_cflags = [flag.strip() for flag in extra_cuda_cflags]
    extra_sycl_cflags = [flag.strip() for flag in extra_sycl_cflags]
    extra_ldflags = [flag.strip() for flag in extra_ldflags]
    extra_include_paths = [flag.strip() for flag in extra_include_paths]

    # Turn into absolute paths so we can emit them into the ninja build
    # file wherever it is.
    user_includes = [os.path.abspath(file) for file in extra_include_paths]

    # include_paths() gives us the location of torch/extension.h
    # TODO generalize with_cuda as specific device type.
    if with_cuda:
        system_includes = include_paths("cuda")
    elif with_sycl:
        system_includes = include_paths("xpu")
    else:
        system_includes = include_paths("cpu")
    # sysconfig.get_path('include') gives us the location of Python.h
    # Explicitly specify 'posix_prefix' scheme on non-Windows platforms to workaround error on some MacOS
    # installations where default `get_path` points to non-existing `/Library/Python/M.m/include` folder
    python_include_path = sysconfig.get_path(
        "include", scheme="nt" if IS_WINDOWS else "posix_prefix"
    )
    if python_include_path is not None:
        system_includes.append(python_include_path)

    common_cflags = []
    if not is_standalone:
        common_cflags.append(f"-DTORCH_EXTENSION_NAME={name}")
        common_cflags.append("-DTORCH_API_INCLUDE_EXTENSION_H")

    # Windows does not understand `-isystem` and quotes flags later.
    if IS_WINDOWS:
        common_cflags += [f"-I{include}" for include in user_includes + system_includes]
    else:
        common_cflags += [f"-I{shlex.quote(include)}" for include in user_includes]
        common_cflags += [
            f"-isystem {shlex.quote(include)}" for include in system_includes
        ]

    if IS_WINDOWS:
        COMMON_HIP_FLAGS.extend(["-fms-runtime-lib=dll"])
        cflags = common_cflags + ["/std:c++20"] + extra_cflags
        cflags += COMMON_MSVC_FLAGS + (COMMON_HIP_FLAGS if IS_HIP_EXTENSION else [])
        cflags = _nt_quote_args(cflags)
    else:
        cflags = common_cflags + ["-fPIC", "-std=c++20"] + extra_cflags

    if with_cuda and IS_HIP_EXTENSION:
        cuda_flags = (
            ["-DWITH_HIP"]
            + common_cflags
            + extra_cflags
            + COMMON_HIP_FLAGS
            + COMMON_HIPCC_FLAGS
        )
        cuda_flags = cuda_flags + ["-std=c++20"]
        cuda_flags += _get_rocm_arch_flags(cuda_flags)
        cuda_flags += extra_cuda_cflags
        if IS_WINDOWS:
            cuda_flags = _nt_quote_args(cuda_flags)
    elif with_cuda:
        cuda_flags = (
            common_cflags + COMMON_NVCC_FLAGS + _get_cuda_arch_flags(extra_cuda_cflags)
        )
        if IS_WINDOWS:
            for flag in COMMON_MSVC_FLAGS:
                cuda_flags = ["-Xcompiler", flag] + cuda_flags
            for ignore_warning in MSVC_IGNORE_CUDAFE_WARNINGS:
                cuda_flags = [
                    "-Xcudafe",
                    "--diag_suppress=" + ignore_warning,
                ] + cuda_flags
            cuda_flags = cuda_flags + ["-std=c++20"]
            cuda_flags = _nt_quote_args(cuda_flags)
            cuda_flags += _nt_quote_args(extra_cuda_cflags)
        else:
            cuda_flags += ["--compiler-options", "'-fPIC'"]
            cuda_flags += extra_cuda_cflags
            if not any(flag.startswith("-std=") for flag in cuda_flags):
                cuda_flags.append("-std=c++20")
            cc_env = os.getenv("CC")
            if cc_env is not None:
                cuda_flags = ["-ccbin", cc_env] + cuda_flags
    else:
        cuda_flags = None

    if with_sycl:
        sycl_cflags = cflags + _COMMON_SYCL_FLAGS
        sycl_cflags += extra_sycl_cflags
        _append_sycl_targets_if_missing(sycl_cflags)
        _append_sycl_std_if_no_std_present(sycl_cflags)
        host_cflags = cflags
        # escaping quoted arguments to pass them thru SYCL compiler
        icpx_version = _get_icpx_version()
        if int(icpx_version) < 20250200:
            host_cflags = [item.replace('\\"', '\\\\"') for item in host_cflags]

        sycl_cflags += _wrap_sycl_host_flags(host_cflags)
        sycl_dlink_post_cflags = _SYCL_DLINK_FLAGS.copy()
        sycl_dlink_post_cflags += _get_sycl_device_flags(sycl_cflags)
    else:
        sycl_cflags = None
        sycl_dlink_post_cflags = None

    def object_file_path(source_file: str) -> str:
        # '/path/to/file.cpp' -> 'file'
        file_name = os.path.splitext(os.path.basename(source_file))[0]
        if _is_cuda_file(source_file) and with_cuda:
            # Use a different object filename in case a C++ and CUDA file have
            # the same filename but different extension (.cpp vs. .cu).
            target = f"{file_name}.cuda.o"
        elif _is_sycl_file(source_file) and with_sycl:
            target = f"{file_name}.sycl.o"
        else:
            target = f"{file_name}.o"
        return target

    objects = [object_file_path(src) for src in sources]
    ldflags = ([] if is_standalone else [SHARED_FLAG]) + extra_ldflags

    # The darwin linker needs explicit consent to ignore unresolved symbols.
    if IS_MACOS:
        ldflags.append("-undefined dynamic_lookup")
    elif IS_WINDOWS:
        ldflags = _nt_quote_args(ldflags)

    ext = EXEC_EXT if is_standalone else LIB_EXT
    library_target = f"{name}{ext}"

    _write_ninja_file(
        path=path,
        cflags=cflags,
        post_cflags=None,
        cuda_cflags=cuda_flags,
        cuda_post_cflags=None,
        cuda_dlink_post_cflags=None,
        sycl_cflags=sycl_cflags,
        sycl_post_cflags=[],
        sycl_dlink_post_cflags=sycl_dlink_post_cflags,
        sources=sources,
        objects=objects,
        ldflags=ldflags,
        library_target=library_target,
        with_cuda=with_cuda,
        with_sycl=with_sycl,
    )


def _write_ninja_file(
    path,
    cflags,
    post_cflags,
    cuda_cflags,
    cuda_post_cflags,
    cuda_dlink_post_cflags,
    sycl_cflags,
    sycl_post_cflags,
    sycl_dlink_post_cflags,
    sources,
    objects,
    ldflags,
    library_target,
    with_cuda,
    with_sycl,
) -> None:
    r"""Write a ninja file that does the desired compiling and linking.

        `path`: Where to write this file
        `cflags`: list of flags to pass to $cxx. Can be None.
        `post_cflags`: list of flags to append to the $cxx invocation. Can be None.
        `cuda_cflags`: list of flags to pass to $nvcc. Can be None.
        `cuda_post_cflags`: list of flags to append to the $nvcc invocation. Can be None.
        `cuda_dlink_post_cflags`: list of flags to append to the $nvcc device code link invocation. Can be None.
        `sycl_cflags`: list of flags to pass to SYCL compiler. Can be None.
        `sycl_post_cflags`: list of flags to append to the SYCL compiler invocation. Can be None.
        `sycl_dlink_post_cflags`: list of flags to append to the SYCL compiler device code link invocation. Can be None.
    e.
        `sources`: list of paths to source files
        `objects`: list of desired paths to objects, one per source.
        `ldflags`: list of flags to pass to linker. Can be None.
        `library_target`: Name of the output library. Can be None; in that case,
                          we do no linking.
        `with_cuda`: If we should be compiling with CUDA.
    """

    def sanitize_flags(flags):
        if flags is None:
            return []
        else:
            return [flag.strip() for flag in flags]

    cflags = sanitize_flags(cflags)
    post_cflags = sanitize_flags(post_cflags)
    cuda_cflags = sanitize_flags(cuda_cflags)
    cuda_post_cflags = sanitize_flags(cuda_post_cflags)
    cuda_dlink_post_cflags = sanitize_flags(cuda_dlink_post_cflags)
    sycl_cflags = sanitize_flags(sycl_cflags)
    sycl_post_cflags = sanitize_flags(sycl_post_cflags)
    sycl_dlink_post_cflags = sanitize_flags(sycl_dlink_post_cflags)
    ldflags = sanitize_flags(ldflags)

    # Sanity checks...
    if len(sources) != len(objects):
        raise AssertionError("sources and objects lists must be the same length")
    if len(sources) == 0:
        raise AssertionError("At least one source is required to build a library")

    compiler = get_cxx_compiler()

    # Version 1.3 is required for the `deps` directive.
    config = ["ninja_required_version = 1.3"]
    config.append(f"cxx = {compiler}")
    if with_cuda or cuda_dlink_post_cflags:
        if "PYTORCH_NVCC" in os.environ:
            nvcc = os.getenv(
                "PYTORCH_NVCC"
            )  # user can set nvcc compiler with ccache using the environment variable here
        else:
            if IS_HIP_EXTENSION:
                nvcc = _get_hipcc_path()
            else:
                nvcc = _join_cuda_home("bin", "nvcc")
        config.append(f"nvcc = {nvcc}")
    if with_sycl or sycl_dlink_post_cflags:
        sycl = "icx" if IS_WINDOWS else "icpx"
        config.append(f"sycl = {sycl}")

    if IS_HIP_EXTENSION:
        post_cflags = COMMON_HIP_FLAGS + post_cflags
    flags = [f"cflags = {' '.join(cflags)}"]
    flags.append(f"post_cflags = {' '.join(post_cflags)}")
    if with_cuda:
        flags.append(f"cuda_cflags = {' '.join(cuda_cflags)}")
        flags.append(f"cuda_post_cflags = {' '.join(cuda_post_cflags)}")
    flags.append(f"cuda_dlink_post_cflags = {' '.join(cuda_dlink_post_cflags)}")
    if with_sycl:
        flags.append(f"sycl_cflags = {' '.join(sycl_cflags)}")
        flags.append(f"sycl_post_cflags = {' '.join(sycl_post_cflags)}")
    flags.append(f"sycl_dlink_post_cflags = {' '.join(sycl_dlink_post_cflags)}")
    flags.append(f"ldflags = {' '.join(ldflags)}")

    # Turn into absolute paths so we can emit them into the ninja build
    # file wherever it is.
    sources = [os.path.abspath(file) for file in sources]

    # See https://ninja-build.org/build.ninja.html for reference.
    compile_rule = ["rule compile"]
    if IS_WINDOWS:
        compiler_name = "$cxx" if IS_HIP_EXTENSION else "cl"
        compile_rule.append(
            f"  command = {compiler_name} "
            "/showIncludes $cflags -c $in /Fo$out $post_cflags"  # codespell:ignore
        )
        if not IS_HIP_EXTENSION:
            compile_rule.append("  deps = msvc")
    else:
        compile_rule.append(
            "  command = $cxx -MMD -MF $out.d $cflags -c $in -o $out $post_cflags"
        )
        compile_rule.append("  depfile = $out.d")
        compile_rule.append("  deps = gcc")

    if with_cuda:
        cuda_compile_rule = ["rule cuda_compile"]
        nvcc_gendeps = ""
        # -MD is not supported by ROCm
        # Nvcc flag `-MD` is not supported by sccache, which may increase build time.
        if (
            torch.version.cuda is not None
            and os.getenv("TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES", "0") != "1"
        ):
            cuda_compile_rule.append("  depfile = $out.d")
            cuda_compile_rule.append("  deps = gcc")
            # Note: non-system deps with nvcc are only supported
            # on Linux so use -MD to make this work on Windows too.
            nvcc_gendeps = "-MD -MF $out.d"
        cuda_compile_rule.append(
            f"  command = $nvcc {nvcc_gendeps} $cuda_cflags -c $in -o $out $cuda_post_cflags"
        )

    if with_sycl:
        sycl_compile_rule = ["rule sycl_compile"]
        # SYCL compiler does not recognize .sycl extension automatically,
        # so we pass '-x c++' explicitly notifying compiler of file format
        sycl_compile_rule.append(
            "  command = $sycl $sycl_cflags -c -x c++ $in -o $out $sycl_post_cflags"
        )

    # Emit one build rule per source to enable incremental build.
    build = []
    for source_file, object_file in zip(sources, objects, strict=True):
        is_cuda_source = _is_cuda_file(source_file) and with_cuda
        is_sycl_source = _is_sycl_file(source_file) and with_sycl
        if is_cuda_source:
            rule = "cuda_compile"
        elif is_sycl_source:
            rule = "sycl_compile"
        else:
            rule = "compile"
        if IS_WINDOWS:
            source_file = source_file.replace(":", "$:")
            object_file = object_file.replace(":", "$:")
        source_file = source_file.replace(" ", "$ ")
        object_file = object_file.replace(" ", "$ ")
        build.append(f"build {object_file}: {rule} {source_file}")

    if cuda_dlink_post_cflags:
        cuda_devlink_out = os.path.join(os.path.dirname(objects[0]), "dlink.o")
        cuda_devlink_rule = ["rule cuda_devlink"]
        cuda_devlink_rule.append(
            "  command = $nvcc $in -o $out $cuda_dlink_post_cflags"
        )
        cuda_devlink = [f"build {cuda_devlink_out}: cuda_devlink {' '.join(objects)}"]
        objects += [cuda_devlink_out]
    else:
        cuda_devlink_rule, cuda_devlink = [], []

    if sycl_dlink_post_cflags:
        sycl_devlink_out = os.path.join(os.path.dirname(objects[0]), "sycl_dlink.o")
        if IS_WINDOWS:
            sycl_devlink_objects = [obj.replace(":", "$:") for obj in objects]
            objects += [sycl_devlink_out]
            sycl_devlink_out = sycl_devlink_out.replace(":", "$:")
        else:
            sycl_devlink_objects = list(objects)
            objects += [sycl_devlink_out]
        sycl_devlink_rule = ["rule sycl_devlink"]
        sycl_devlink_rule.append(
            "  command = $sycl $in -o $out $sycl_dlink_post_cflags"
        )
        sycl_devlink = [
            f"build {sycl_devlink_out}: sycl_devlink {' '.join(sycl_devlink_objects)}"
        ]
    else:
        sycl_devlink_rule, sycl_devlink = [], []

    if library_target is not None:
        link_rule = ["rule link"]
        if IS_WINDOWS:
            cl_paths = (
                subprocess.check_output(["where", "cl"])
                .decode(*SUBPROCESS_DECODE_ARGS)
                .split("\r\n")
            )
            if len(cl_paths) >= 1:
                cl_path = os.path.dirname(cl_paths[0]).replace(":", "$:")
            else:
                raise RuntimeError("MSVC is required to load C++ extensions")
            link_rule.append(
                f'  command = "{cl_path}/link.exe" $in /nologo $ldflags /out:$out'
            )
        else:
            link_rule.append("  command = $cxx $in $ldflags -o $out")

        link = [f"build {library_target}: link {' '.join(objects)}"]

        default = [f"default {library_target}"]
    else:
        link_rule, link, default = [], [], []

    # 'Blocks' should be separated by newlines, for visual benefit.
    blocks = [config, flags, compile_rule]
    if with_cuda:
        blocks.append(cuda_compile_rule)  # type: ignore[possibly-undefined]
    if with_sycl:
        blocks.append(sycl_compile_rule)  # type: ignore[possibly-undefined]
    blocks += [
        cuda_devlink_rule,
        sycl_devlink_rule,
        link_rule,
        build,
        cuda_devlink,
        sycl_devlink,
        link,
        default,
    ]
    content = "\n\n".join("\n".join(b) for b in blocks)
    # Ninja requires a new lines at the end of the .ninja file
    content += "\n"
    _maybe_write(path, content)
