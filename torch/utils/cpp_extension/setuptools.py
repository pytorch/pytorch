# mypy: allow-untyped-defs
"""Setuptools adapter for :mod:`torch.utils.cpp_extension`.

Provides :class:`BuildExtension` and the :func:`CppExtension` /
:func:`CUDAExtension` / :func:`SyclExtension` factories. This module is the
only place in the package that requires ``setuptools`` at import time; the
JIT path in :mod:`._jit` is setuptools-independent.
"""

import copy
import logging
import os
import re
import shlex

import setuptools
from setuptools.command.build_ext import build_ext

import torch
from torch.torch_version import TorchVersion

from ._discovery import (
    _append_sycl_std_if_no_std_present,
    _append_sycl_targets_if_missing,
    _check_cuda_version,
    _COMMON_SYCL_FLAGS,
    _get_icpx_version,
    _get_sycl_device_flags,
    _is_cuda_file,
    _is_sycl_file,
    _join_cuda_home,
    _join_rocm_home,
    _nt_quote_args,
    _set_hipcc_runtime_lib,
    _SYCL_DLINK_FLAGS,
    _wrap_sycl_host_flags,
    COMMON_HIP_FLAGS,
    COMMON_HIPCC_FLAGS,
    COMMON_MSVC_FLAGS,
    COMMON_NVCC_FLAGS,
    get_compiler_abi_compatibility_and_version,
    get_cxx_compiler,
    include_paths,
    IS_HIP_EXTENSION,
    is_ninja_available,
    IS_WINDOWS,
    library_paths,
    min_supported_cpython,
    MSVC_IGNORE_CUDAFE_WARNINGS,
)
from ._jit import (
    _get_cuda_arch_flags,
    _get_hipcc_path,
    _get_rocm_arch_flags,
    _write_ninja_file_and_compile_objects,
)


logger = logging.getLogger(__name__)


__all__ = [
    "BuildExtension",
    "CppExtension",
    "CUDAExtension",
    "SyclExtension",
]


class BuildExtension(build_ext):
    """
    A custom :mod:`setuptools` build extension .

    This :class:`setuptools.build_ext` subclass takes care of passing the
    minimum required compiler flags (e.g. ``-std=c++20``) as well as mixed
    C++/CUDA/SYCL compilation (and support for CUDA/SYCL files in general).

    When using :class:`BuildExtension`, it is allowed to supply a dictionary
    for ``extra_compile_args`` (rather than the usual list) that maps from
    languages/compilers (the only expected values are ``cxx``, ``nvcc`` or
    ``sycl``) to a list of additional compiler flags to supply to the compiler.
    This makes it possible to supply different flags to the C++, CUDA and SYCL
    compiler during mixed compilation.

    ``use_ninja`` (bool): If ``use_ninja`` is ``True`` (default), then we
    attempt to build using the Ninja backend. Ninja greatly speeds up
    compilation compared to the standard ``setuptools.build_ext``.
    Fallbacks to the standard distutils backend if Ninja is not available.

    .. note::
        By default, the Ninja backend uses #CPUS + 2 workers to build the
        extension. This may use up too many resources on some systems. One
        can control the number of workers by setting the `MAX_JOBS` environment
        variable to a non-negative number.
    """

    @classmethod
    def with_options(cls, **options):
        """Return a subclass with alternative constructor that extends any original keyword arguments to the original constructor with the given options."""

        class cls_with_options(cls):  # type: ignore[misc, valid-type]
            def __init__(self, *args, **kwargs) -> None:
                kwargs.update(options)
                super().__init__(*args, **kwargs)

        return cls_with_options

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.no_python_abi_suffix = kwargs.get("no_python_abi_suffix", False)

        self.use_ninja = kwargs.get("use_ninja", True)
        if self.use_ninja:
            # Test if we can use ninja. Fallback otherwise.
            msg = (
                "Attempted to use ninja as the BuildExtension backend but "
                "%s. Falling back to using the slow distutils backend."
            )
            if not is_ninja_available():
                logger.warning(msg, "we could not find ninja.")
                self.use_ninja = False

    def finalize_options(self) -> None:
        super().finalize_options()
        if self.use_ninja:
            self.force = True

    def build_extensions(self) -> None:
        compiler_name, compiler_version = self._check_abi()

        cuda_ext = False
        sycl_ext = False
        extension_iter = iter(self.extensions)
        extension = next(extension_iter, None)
        while not (cuda_ext and sycl_ext) and extension:
            for source in extension.sources:
                _, ext = os.path.splitext(source)
                if ext == ".cu":
                    cuda_ext = True
                elif ext == ".sycl":
                    sycl_ext = True

                # This check accounts on a case when cuda and sycl sources
                # are mixed in the same extension. We can stop checking
                # sources if both are found or there is no more sources.
                if cuda_ext and sycl_ext:
                    break

            extension = next(extension_iter, None)

        if sycl_ext:
            if not self.use_ninja:
                raise AssertionError("ninja is required to build sycl extensions.")

        if cuda_ext and not IS_HIP_EXTENSION:
            _check_cuda_version(compiler_name, compiler_version)

        for extension in self.extensions:
            # Ensure at least an empty list of flags for 'cxx', 'nvcc' and 'sycl' when
            # extra_compile_args is a dict. Otherwise, default torch flags do
            # not get passed. Necessary when only one of 'cxx', 'nvcc' or 'sycl' is
            # passed to extra_compile_args in CUDAExtension or SyclExtension, i.e.
            #   CUDAExtension(..., extra_compile_args={'cxx': [...]})
            # or
            #   CUDAExtension(..., extra_compile_args={'nvcc': [...]})
            if isinstance(extension.extra_compile_args, dict):
                for ext in ["cxx", "nvcc", "sycl"]:
                    if ext not in extension.extra_compile_args:
                        extension.extra_compile_args[ext] = []

            self._add_compile_flag(extension, "-DTORCH_API_INCLUDE_EXTENSION_H")

            if IS_HIP_EXTENSION:
                self._hipify_compile_flags(extension)

            if extension.py_limited_api:
                # compile any extension that has passed in py_limited_api to the
                # Extension constructor with the Py_LIMITED_API flag set to our
                # min supported CPython version.
                # See https://docs.python.org/3/c-api/stable.html#c.Py_LIMITED_API
                self._add_compile_flag(
                    extension, f"-DPy_LIMITED_API={min_supported_cpython}"
                )
            self._define_torch_extension_name(extension)

            if "nvcc_dlink" in extension.extra_compile_args:
                if not self.use_ninja:
                    raise AssertionError(
                        f"With dlink=True, ninja is required to build cuda extension {extension.name}."
                    )

        # Register .cu, .cuh, .hip, .mm and .sycl as valid source extensions.
        # NOTE: At the moment .sycl is not a standard extension for SYCL supported
        # by compiler. Here we introduce a torch level convention that SYCL sources
        # should have .sycl file extension.
        self.compiler.src_extensions += [".cu", ".cuh", ".hip", ".sycl"]
        if torch.backends.mps.is_built():
            self.compiler.src_extensions += [".mm"]
        # Save the original _compile method for later.
        if self.compiler.compiler_type == "msvc":
            self.compiler._cpp_extensions += [".cu", ".cuh"]
            original_compile = self.compiler.compile
            original_spawn = self.compiler.spawn
        else:
            original_compile = self.compiler._compile

        def append_std17_if_no_std_present(cflags) -> None:
            # NVCC does not allow multiple -std to be passed, so we avoid
            # overriding the option if the user explicitly passed it.
            cpp_format_prefix = (
                "/{}:" if self.compiler.compiler_type == "msvc" else "-{}="
            )
            cpp_flag_prefix = cpp_format_prefix.format("std")
            cpp_flag = cpp_flag_prefix + "c++20"
            if not any(flag.startswith(cpp_flag_prefix) for flag in cflags):
                cflags.append(cpp_flag)

        def unix_cuda_flags(cflags):
            cflags = (
                COMMON_NVCC_FLAGS
                + ["--compiler-options", "'-fPIC'"]
                + cflags
                + _get_cuda_arch_flags(cflags)
            )

            # NVCC does not allow multiple -ccbin/--compiler-bindir to be passed, so we avoid
            # overriding the option if the user explicitly passed it.
            _ccbin = os.getenv("CC")
            if _ccbin is not None and not any(
                flag.startswith(("-ccbin", "--compiler-bindir")) for flag in cflags
            ):
                cflags.extend(["-ccbin", _ccbin])

            return cflags

        def convert_to_absolute_paths_inplace(paths) -> None:
            # Helper function. See Note [Absolute include_dirs]
            if paths is not None:
                for i in range(len(paths)):
                    if not os.path.isabs(paths[i]):
                        paths[i] = os.path.abspath(paths[i])

        def unix_wrap_single_compile(
            obj, src, ext, cc_args, extra_postargs, pp_opts
        ) -> None:
            # Copy before we make any modifications.
            cflags = copy.deepcopy(extra_postargs)
            try:
                original_compiler = self.compiler.compiler_so
                if _is_cuda_file(src):
                    nvcc = [
                        _join_rocm_home("bin", "hipcc")
                        if IS_HIP_EXTENSION
                        else _join_cuda_home("bin", "nvcc")
                    ]
                    self.compiler.set_executable("compiler_so", nvcc)
                    if isinstance(cflags, dict):
                        cflags = cflags["nvcc"]
                    if IS_HIP_EXTENSION:
                        cflags = (
                            COMMON_HIPCC_FLAGS + cflags + _get_rocm_arch_flags(cflags)
                        )
                    else:
                        cflags = unix_cuda_flags(cflags)
                elif isinstance(cflags, dict):
                    cflags = cflags["cxx"]
                if IS_HIP_EXTENSION:
                    cflags = COMMON_HIP_FLAGS + cflags
                append_std17_if_no_std_present(cflags)

                original_compile(obj, src, ext, cc_args, cflags, pp_opts)
            finally:
                # Put the original compiler back in place.
                self.compiler.set_executable("compiler_so", original_compiler)

        def unix_wrap_ninja_compile(
            sources,
            output_dir=None,
            macros=None,
            include_dirs=None,
            debug=0,
            extra_preargs=None,
            extra_postargs=None,
            depends=None,
        ):
            r"""Compiles sources by outputting a ninja file and running it."""
            # NB: I copied some lines from self.compiler (which is an instance
            # of distutils.UnixCCompiler). See the following link.
            # https://github.com/python/cpython/blob/f03a8f8d5001963ad5b5b28dbd95497e9cc15596/Lib/distutils/ccompiler.py#L564-L567  # codespell:ignore
            # This can be fragile, but a lot of other repos also do this
            # (see https://github.com/search?q=_setup_compile&type=Code)
            # so it is probably OK; we'll also get CI signal if/when
            # we update our python version (which is when distutils can be
            # upgraded)

            # Use absolute path for output_dir so that the object file paths
            # (`objects`) get generated with absolute paths.
            # pyrefly: ignore [no-matching-overload]
            output_dir = os.path.abspath(output_dir)

            # See Note [Absolute include_dirs]
            convert_to_absolute_paths_inplace(self.compiler.include_dirs)

            _, objects, extra_postargs, pp_opts, _ = self.compiler._setup_compile(
                output_dir, macros, include_dirs, sources, depends, extra_postargs
            )
            common_cflags = self.compiler._get_cc_args(pp_opts, debug, extra_preargs)
            extra_cc_cflags = self.compiler.compiler_so[1:]
            with_cuda = any(map(_is_cuda_file, sources))
            with_sycl = any(map(_is_sycl_file, sources))
            if with_sycl and with_cuda:
                raise AssertionError(
                    "cannot have both SYCL and CUDA files in the same extension"
                )

            # extra_postargs can be either:
            # - a dict mapping cxx/nvcc/sycl to extra flags
            # - a list of extra flags.
            if isinstance(extra_postargs, dict):
                post_cflags = extra_postargs["cxx"]
            else:
                post_cflags = list(extra_postargs)
            if IS_HIP_EXTENSION:
                post_cflags = COMMON_HIP_FLAGS + post_cflags
            append_std17_if_no_std_present(post_cflags)

            cuda_post_cflags = None
            cuda_cflags = None
            if with_cuda:
                cuda_cflags = common_cflags
                if isinstance(extra_postargs, dict):
                    cuda_post_cflags = extra_postargs["nvcc"]
                else:
                    cuda_post_cflags = list(extra_postargs)
                if IS_HIP_EXTENSION:
                    cuda_post_cflags = cuda_post_cflags + _get_rocm_arch_flags(
                        cuda_post_cflags
                    )
                    cuda_post_cflags = (
                        COMMON_HIP_FLAGS + COMMON_HIPCC_FLAGS + cuda_post_cflags
                    )
                else:
                    cuda_post_cflags = unix_cuda_flags(cuda_post_cflags)
                append_std17_if_no_std_present(cuda_post_cflags)
                cuda_cflags = [shlex.quote(f) for f in cuda_cflags]
                cuda_post_cflags = [shlex.quote(f) for f in cuda_post_cflags]

            if isinstance(extra_postargs, dict) and "nvcc_dlink" in extra_postargs:
                cuda_dlink_post_cflags = unix_cuda_flags(extra_postargs["nvcc_dlink"])
                cuda_dlink_post_cflags = [
                    shlex.quote(f) for f in cuda_dlink_post_cflags
                ]
            else:
                cuda_dlink_post_cflags = None

            sycl_post_cflags = None
            sycl_cflags = None
            sycl_dlink_post_cflags = None
            if with_sycl:
                sycl_cflags = extra_cc_cflags + common_cflags + _COMMON_SYCL_FLAGS
                if isinstance(extra_postargs, dict):
                    sycl_post_cflags = extra_postargs["sycl"]
                else:
                    sycl_post_cflags = list(extra_postargs)
                _append_sycl_targets_if_missing(sycl_post_cflags)
                append_std17_if_no_std_present(sycl_cflags)
                _append_sycl_std_if_no_std_present(sycl_cflags)
                host_cflags = extra_cc_cflags + common_cflags + post_cflags
                append_std17_if_no_std_present(host_cflags)
                # escaping quoted arguments to pass them thru SYCL compiler
                icpx_version = _get_icpx_version()
                if int(icpx_version) >= 20250200:
                    host_cflags = [item.replace('"', '\\"') for item in host_cflags]
                else:
                    host_cflags = [item.replace('"', '\\\\"') for item in host_cflags]
                # Note the order: shlex.quote sycl_flags first, _wrap_sycl_host_flags
                # second. Reason is that sycl host flags are quoted, space containing
                # strings passed to SYCL compiler.
                sycl_cflags = [shlex.quote(f) for f in sycl_cflags]
                sycl_cflags += _wrap_sycl_host_flags(host_cflags)
                sycl_dlink_post_cflags = _SYCL_DLINK_FLAGS.copy()
                sycl_dlink_post_cflags += _get_sycl_device_flags(sycl_post_cflags)
                sycl_post_cflags = [shlex.quote(f) for f in sycl_post_cflags]

            _write_ninja_file_and_compile_objects(
                sources=sources,
                objects=objects,
                cflags=[shlex.quote(f) for f in extra_cc_cflags + common_cflags],
                post_cflags=[shlex.quote(f) for f in post_cflags],
                cuda_cflags=cuda_cflags,
                cuda_post_cflags=cuda_post_cflags,
                cuda_dlink_post_cflags=cuda_dlink_post_cflags,
                sycl_cflags=sycl_cflags,
                sycl_post_cflags=sycl_post_cflags,
                sycl_dlink_post_cflags=sycl_dlink_post_cflags,
                build_directory=output_dir,
                verbose=True,
                with_cuda=with_cuda,
                with_sycl=with_sycl,
            )

            # Return *all* object filenames, not just the ones we just built.
            return objects

        def win_cuda_flags(cflags):
            return COMMON_NVCC_FLAGS + cflags + _get_cuda_arch_flags(cflags)

        def win_hip_flags(cflags):
            return (
                COMMON_HIPCC_FLAGS
                + COMMON_HIP_FLAGS
                + cflags
                + _get_rocm_arch_flags(cflags)
            )

        def win_filter_msvc_include_dirs(pp_opts) -> list[str]:
            """Filter out MSVC include dirs from pp_opts for oneAPI 2025.3+."""
            # oneAPI 2025.3+ changed include path ordering to match MSVC behavior.
            # Filter out MSVC headers to avoid conflicting declarations with oneAPI's std headers.
            icpx_version = int(_get_icpx_version())
            if icpx_version >= 20250300:
                vc_tools_dir = os.path.normcase(os.environ.get("VCToolsInstallDir", ""))
                if vc_tools_dir:
                    pp_opts = [
                        path
                        for path in pp_opts
                        if vc_tools_dir not in os.path.normcase(path)
                    ]
            return pp_opts

        def win_wrap_single_compile(
            sources,
            output_dir=None,
            macros=None,
            include_dirs=None,
            debug=0,
            extra_preargs=None,
            extra_postargs=None,
            depends=None,
        ):
            self.cflags = copy.deepcopy(extra_postargs)
            extra_postargs = None

            def spawn(cmd):
                # Using regex to match src, obj and include files
                src_regex = re.compile("/T(p|c)(.*)")
                src_list = [
                    m.group(2) for m in (src_regex.match(elem) for elem in cmd) if m
                ]

                obj_regex = re.compile("/Fo(.*)")  # codespell:ignore
                obj_list = [
                    m.group(1) for m in (obj_regex.match(elem) for elem in cmd) if m
                ]

                include_regex = re.compile(r"((\-|\/)I.*)")
                include_list = [
                    m.group(1) for m in (include_regex.match(elem) for elem in cmd) if m
                ]

                if len(src_list) >= 1 and len(obj_list) >= 1:
                    src = src_list[0]
                    obj = obj_list[0]
                    if _is_cuda_file(src):
                        if IS_HIP_EXTENSION:
                            nvcc = _get_hipcc_path()
                        else:
                            nvcc = _join_cuda_home("bin", "nvcc")
                        if isinstance(self.cflags, dict):
                            cflags = self.cflags["nvcc"]
                        elif isinstance(self.cflags, list):
                            cflags = self.cflags
                        else:
                            cflags = []

                        if IS_HIP_EXTENSION:
                            cflags = win_hip_flags(cflags)
                        else:
                            cflags = win_cuda_flags(cflags) + [
                                "-std=c++20",
                                "--use-local-env",
                            ]
                            for ignore_warning in MSVC_IGNORE_CUDAFE_WARNINGS:
                                cflags = [
                                    "-Xcudafe",
                                    "--diag_suppress=" + ignore_warning,
                                ] + cflags
                        for flag in COMMON_MSVC_FLAGS:
                            cflags = ["-Xcompiler", flag] + cflags
                        cmd = [nvcc, "-c", src, "-o", obj] + include_list + cflags
                    elif isinstance(self.cflags, dict):
                        cflags = COMMON_MSVC_FLAGS + self.cflags["cxx"]
                        append_std17_if_no_std_present(cflags)
                        cmd += cflags
                    elif isinstance(self.cflags, list):
                        cflags = COMMON_MSVC_FLAGS + self.cflags
                        append_std17_if_no_std_present(cflags)
                        cmd += cflags

                return original_spawn(cmd)

            try:
                self.compiler.spawn = spawn
                return original_compile(
                    sources,
                    output_dir,
                    macros,
                    include_dirs,
                    debug,
                    extra_preargs,
                    extra_postargs,
                    depends,
                )
            finally:
                self.compiler.spawn = original_spawn

        def win_wrap_ninja_compile(
            sources,
            output_dir=None,
            macros=None,
            include_dirs=None,
            debug=0,
            extra_preargs=None,
            extra_postargs=None,
            depends=None,
            is_standalone=False,
        ):
            if not self.compiler.initialized:
                self.compiler.initialize()
            # pyrefly: ignore [no-matching-overload]
            output_dir = os.path.abspath(output_dir)

            # Note [Absolute include_dirs]
            # Convert relative path in self.compiler.include_dirs to absolute path if any.
            # For ninja build, the build location is not local, but instead, the build happens
            # in a script-created build folder. Thus, relative paths lose their correctness.
            # To be consistent with jit extension, we allow user to enter relative include_dirs
            # in setuptools.setup, and we convert the relative path to absolute path here.
            convert_to_absolute_paths_inplace(self.compiler.include_dirs)

            _, objects, extra_postargs, pp_opts, _ = self.compiler._setup_compile(
                output_dir, macros, include_dirs, sources, depends, extra_postargs
            )
            # Replace space with \ when using hipcc (hipcc passes includes to clang without ""s so clang sees space in include paths as new argument)
            if IS_HIP_EXTENSION:
                pp_opts = [
                    "-I{}".format(s[2:].replace(" ", "\\")) if s.startswith("-I") else s
                    for s in pp_opts
                ]
            common_cflags = extra_preargs or []
            cflags = []
            if debug:
                cflags.extend(self.compiler.compile_options_debug)
            else:
                cflags.extend(self.compiler.compile_options)
            cflags = cflags + common_cflags + pp_opts + COMMON_MSVC_FLAGS
            if IS_HIP_EXTENSION:
                _set_hipcc_runtime_lib(is_standalone, debug)
                common_cflags.extend(COMMON_HIP_FLAGS)
            else:
                common_cflags.extend(COMMON_MSVC_FLAGS)
            with_cuda = any(map(_is_cuda_file, sources))
            with_sycl = any(map(_is_sycl_file, sources))
            if with_sycl and with_cuda:
                raise AssertionError(
                    "cannot have both SYCL and CUDA files in the same extension"
                )

            # extra_postargs can be either:
            # - a dict mapping cxx/nvcc to extra flags
            # - a list of extra flags.
            if isinstance(extra_postargs, dict):
                post_cflags = extra_postargs["cxx"]
            else:
                post_cflags = list(extra_postargs)
            if IS_HIP_EXTENSION:
                post_cflags = COMMON_HIP_FLAGS + post_cflags
            append_std17_if_no_std_present(post_cflags)

            cuda_post_cflags = None
            cuda_cflags = None
            if with_cuda:
                cuda_cflags = ["-std=c++20"]
                for common_cflag in common_cflags:
                    cuda_cflags.append("-Xcompiler")
                    cuda_cflags.append(common_cflag)
                if not IS_HIP_EXTENSION:
                    cuda_cflags.append("--use-local-env")
                    for ignore_warning in MSVC_IGNORE_CUDAFE_WARNINGS:
                        cuda_cflags.append("-Xcudafe")
                        cuda_cflags.append("--diag_suppress=" + ignore_warning)
                cuda_cflags.extend(pp_opts)
                if isinstance(extra_postargs, dict):
                    cuda_post_cflags = extra_postargs["nvcc"]
                else:
                    cuda_post_cflags = list(extra_postargs)
                if IS_HIP_EXTENSION:
                    cuda_post_cflags = win_hip_flags(cuda_post_cflags)
                else:
                    cuda_post_cflags = win_cuda_flags(cuda_post_cflags)
            cflags = _nt_quote_args(cflags)
            post_cflags = _nt_quote_args(post_cflags)
            if with_cuda:
                cuda_cflags = _nt_quote_args(cuda_cflags)
                cuda_post_cflags = _nt_quote_args(cuda_post_cflags)
            if isinstance(extra_postargs, dict) and "nvcc_dlink" in extra_postargs:
                cuda_dlink_post_cflags = win_cuda_flags(extra_postargs["nvcc_dlink"])
            else:
                cuda_dlink_post_cflags = None

            sycl_cflags = None
            sycl_post_cflags = None
            sycl_dlink_post_cflags = None
            if with_sycl:
                sycl_cflags = (
                    common_cflags
                    + win_filter_msvc_include_dirs(pp_opts)
                    + _COMMON_SYCL_FLAGS
                )
                if isinstance(extra_postargs, dict):
                    sycl_post_cflags = extra_postargs["sycl"]
                else:
                    sycl_post_cflags = list(extra_postargs)
                _append_sycl_targets_if_missing(sycl_post_cflags)
                append_std17_if_no_std_present(sycl_cflags)
                _append_sycl_std_if_no_std_present(sycl_cflags)
                host_cflags = common_cflags + pp_opts + post_cflags
                append_std17_if_no_std_present(host_cflags)

                sycl_cflags = _nt_quote_args(sycl_cflags)
                host_cflags = _nt_quote_args(host_cflags)

                sycl_cflags += _wrap_sycl_host_flags(host_cflags)
                sycl_dlink_post_cflags = _SYCL_DLINK_FLAGS.copy()
                sycl_dlink_post_cflags += _get_sycl_device_flags(sycl_post_cflags)
                sycl_post_cflags = _nt_quote_args(sycl_post_cflags)

            _write_ninja_file_and_compile_objects(
                sources=sources,
                objects=objects,
                cflags=cflags,
                post_cflags=post_cflags,
                cuda_cflags=cuda_cflags,
                cuda_post_cflags=cuda_post_cflags,
                cuda_dlink_post_cflags=cuda_dlink_post_cflags,
                sycl_cflags=sycl_cflags,
                sycl_post_cflags=sycl_post_cflags,
                sycl_dlink_post_cflags=sycl_dlink_post_cflags,
                build_directory=output_dir,
                verbose=True,
                with_cuda=with_cuda,
                with_sycl=with_sycl,
            )

            # Return *all* object filenames, not just the ones we just built.
            return objects

        # Monkey-patch the _compile or compile method.
        # https://github.com/python/cpython/blob/dc0284ee8f7a270b6005467f26d8e5773d76e959/Lib/distutils/ccompiler.py#L511  # codespell:ignore
        if self.compiler.compiler_type == "msvc":
            if self.use_ninja:
                self.compiler.compile = win_wrap_ninja_compile
            else:
                self.compiler.compile = win_wrap_single_compile
        else:
            if self.use_ninja:
                self.compiler.compile = unix_wrap_ninja_compile
            else:
                self.compiler._compile = unix_wrap_single_compile

        build_ext.build_extensions(self)

    def get_ext_filename(self, ext_name):
        # Get the original shared library name. For Python 3, this name will be
        # suffixed with "<SOABI>.so", where <SOABI> will be something like
        # cpython-37m-x86_64-linux-gnu.
        ext_filename = super().get_ext_filename(ext_name)
        # If `no_python_abi_suffix` is `True`, we omit the Python 3 ABI
        # component. This makes building shared libraries with setuptools that
        # aren't Python modules nicer.
        if self.no_python_abi_suffix:
            # The parts will be e.g. ["my_extension", "cpython-37m-x86_64-linux-gnu", "so"].
            ext_filename_parts = ext_filename.split(".")
            # Remove ABI component only if it actually exists in a file name, see gh-170542.
            if len(ext_filename_parts) > 2:
                # Omit the second to last element.
                without_abi = ext_filename_parts[:-2] + ext_filename_parts[-1:]
                ext_filename = ".".join(without_abi)
        return ext_filename

    def get_export_symbols(self, ext):
        if IS_WINDOWS:
            # Skips exporting the module "PyInit_" function that the
            # distutils Extension.get_export_symbols would add to
            # ext.export_symbols. Only relevant for Windows builds.
            return ext.export_symbols
        return super().get_export_symbols(ext)

    def _check_abi(self) -> tuple[str, TorchVersion]:
        # On some platforms, like Windows, compiler_cxx is not available.
        if hasattr(self.compiler, "compiler_cxx"):
            compiler = self.compiler.compiler_cxx[0]
        else:
            compiler = get_cxx_compiler()
        _, version = get_compiler_abi_compatibility_and_version(compiler)
        # Warn user if VC env is activated but `DISTUILS_USE_SDK` is not set.
        if (
            IS_WINDOWS
            and "VSCMD_ARG_TGT_ARCH" in os.environ
            and "DISTUTILS_USE_SDK" not in os.environ
        ):
            msg = (
                "It seems that the VC environment is activated but DISTUTILS_USE_SDK is not set."
                "This may lead to multiple activations of the VC env."
                "Please set `DISTUTILS_USE_SDK=1` and try again."
            )
            raise UserWarning(msg)
        return compiler, version

    def _add_compile_flag(self, extension, flag) -> None:
        extension.extra_compile_args = copy.deepcopy(extension.extra_compile_args)
        if isinstance(extension.extra_compile_args, dict):
            for args in extension.extra_compile_args.values():
                args.append(flag)
        else:
            extension.extra_compile_args.append(flag)

    # Simple hipify, replace the first occurrence of CUDA with HIP
    # in flags starting with "-" and containing "CUDA", but exclude -I flags
    def _hipify_compile_flags(self, extension) -> None:
        if (
            isinstance(extension.extra_compile_args, dict)
            and "nvcc" in extension.extra_compile_args
        ):
            modified_flags = []
            for flag in extension.extra_compile_args["nvcc"]:
                if (
                    flag.startswith("-")
                    and "CUDA" in flag
                    and not flag.startswith("-I")
                ):
                    # check/split flag into flag and value
                    parts = flag.split("=", 1)
                    if len(parts) == 2:
                        flag_part, value_part = parts
                        # replace fist instance of "CUDA" with "HIP" only in the flag and not flag value
                        modified_flag_part = flag_part.replace("CUDA", "HIP", 1)
                        modified_flag = f"{modified_flag_part}={value_part}"
                    else:
                        # replace fist instance of "CUDA" with "HIP" in flag
                        modified_flag = flag.replace("CUDA", "HIP", 1)
                    modified_flags.append(modified_flag)
                    logger.info("Modified flag: %s -> %s", flag, modified_flag)
                else:
                    modified_flags.append(flag)
            extension.extra_compile_args["nvcc"] = modified_flags

    def _define_torch_extension_name(self, extension) -> None:
        # pybind11 doesn't support dots in the names
        # so in order to support extensions in the packages
        # like torch._C, we take the last part of the string
        # as the library name
        names = extension.name.split(".")
        name = names[-1]
        define = f"-DTORCH_EXTENSION_NAME={name}"
        self._add_compile_flag(extension, define)


def CppExtension(name, sources, *args, **kwargs):
    """
    Create a :class:`setuptools.Extension` for C++.

    Convenience method that creates a :class:`setuptools.Extension` with the
    bare minimum (but often sufficient) arguments to build a C++ extension.

    All arguments are forwarded to the :class:`setuptools.Extension`
    constructor. Full list arguments can be found at
    https://setuptools.pypa.io/en/latest/userguide/ext_modules.html#extension-api-reference

    .. warning::
        The PyTorch python API (as provided in libtorch_python) cannot be built
        with the flag ``py_limited_api=True``.  When this flag is passed, it is
        the user's responsibility in their library to not use APIs from
        libtorch_python (in particular pytorch/python bindings) and to only use
        APIs from libtorch (aten objects, operators and the dispatcher). For
        example, to give access to custom ops from python, the library should
        register the ops through the dispatcher.

        Contrary to CPython setuptools, who does not define -DPy_LIMITED_API
        as a compile flag when py_limited_api is specified as an option for
        the "bdist_wheel" command in ``setup``, PyTorch does! We will specify
        -DPy_LIMITED_API=min_supported_cpython to best enforce consistency,
        safety, and sanity in order to encourage best practices. To target a
        different version, set min_supported_cpython to the hexcode of the
        CPython version of choice.

    Example:
        >>> # xdoctest: +SKIP
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CPP_EXT)
        >>> from setuptools import setup
        >>> from torch.utils.cpp_extension import BuildExtension, CppExtension
        >>> setup(
        ...     name="extension",
        ...     ext_modules=[
        ...         CppExtension(
        ...             name="extension",
        ...             sources=["extension.cpp"],
        ...             extra_compile_args=["-g"],
        ...             extra_link_args=["-Wl,--no-as-needed", "-lm"],
        ...         )
        ...     ],
        ...     cmdclass={"build_ext": BuildExtension},
        ... )
    """
    include_dirs = kwargs.get("include_dirs", [])
    include_dirs += include_paths()
    kwargs["include_dirs"] = include_dirs

    library_dirs = kwargs.get("library_dirs", [])
    library_dirs += library_paths()
    kwargs["library_dirs"] = library_dirs

    libraries = kwargs.get("libraries", [])
    libraries.append("c10")
    libraries.append("torch")
    libraries.append("torch_cpu")
    if not kwargs.get("py_limited_api", False):
        # torch_python uses more than the python limited api
        libraries.append("torch_python")
    if IS_WINDOWS:
        libraries.append("sleef")

    kwargs["libraries"] = libraries

    kwargs["language"] = "c++"
    return setuptools.Extension(name, sources, *args, **kwargs)


def CUDAExtension(name, sources, *args, **kwargs):
    """
    Create a :class:`setuptools.Extension` for CUDA/C++.

    Convenience method that creates a :class:`setuptools.Extension` with the
    bare minimum (but often sufficient) arguments to build a CUDA/C++
    extension. This includes the CUDA include path, library path and runtime
    library.

    All arguments are forwarded to the :class:`setuptools.Extension`
    constructor. Full list arguments can be found at
    https://setuptools.pypa.io/en/latest/userguide/ext_modules.html#extension-api-reference

    .. warning::
        The PyTorch python API (as provided in libtorch_python) cannot be built
        with the flag ``py_limited_api=True``.  When this flag is passed, it is
        the user's responsibility in their library to not use APIs from
        libtorch_python (in particular pytorch/python bindings) and to only use
        APIs from libtorch (aten objects, operators and the dispatcher). For
        example, to give access to custom ops from python, the library should
        register the ops through the dispatcher.

        Contrary to CPython setuptools, who does not define -DPy_LIMITED_API
        as a compile flag when py_limited_api is specified as an option for
        the "bdist_wheel" command in ``setup``, PyTorch does! We will specify
        -DPy_LIMITED_API=min_supported_cpython to best enforce consistency,
        safety, and sanity in order to encourage best practices. To target a
        different version, set min_supported_cpython to the hexcode of the
        CPython version of choice.

    Example:
        >>> # xdoctest: +SKIP
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CPP_EXT)
        >>> from setuptools import setup
        >>> from torch.utils.cpp_extension import BuildExtension, CUDAExtension
        >>> setup(
        ...     name="cuda_extension",
        ...     ext_modules=[
        ...         CUDAExtension(
        ...             name="cuda_extension",
        ...             sources=["extension.cpp", "extension_kernel.cu"],
        ...             extra_compile_args={"cxx": ["-g"], "nvcc": ["-O2"]},
        ...             extra_link_args=["-Wl,--no-as-needed", "-lcuda"],
        ...         )
        ...     ],
        ...     cmdclass={"build_ext": BuildExtension},
        ... )

    Compute capabilities:

    By default the extension will be compiled to run on all archs of the cards visible during the
    building process of the extension, plus PTX. If down the road a new card is installed the
    extension may need to be recompiled. If a visible card has a compute capability (CC) that's
    newer than the newest version for which your nvcc can build fully-compiled binaries, PyTorch
    will make nvcc fall back to building kernels with the newest version of PTX your nvcc does
    support (see below for details on PTX).

    You can override the default behavior using `TORCH_CUDA_ARCH_LIST` to explicitly specify which
    CCs you want the extension to support:

    ``TORCH_CUDA_ARCH_LIST="6.1 8.6" python build_my_extension.py``
    ``TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0 8.6+PTX" python build_my_extension.py``

    The +PTX option causes extension kernel binaries to include PTX instructions for the specified
    CC. PTX is an intermediate representation that allows kernels to runtime-compile for any CC >=
    the specified CC (for example, 8.6+PTX generates PTX that can runtime-compile for any GPU with
    CC >= 8.6). This improves your binary's forward compatibility. However, relying on older PTX to
    provide forward compat by runtime-compiling for newer CCs can modestly reduce performance on
    those newer CCs. If you know exact CC(s) of the GPUs you want to target, you're always better
    off specifying them individually. For example, if you want your extension to run on 8.0 and 8.6,
    "8.0+PTX" would work functionally because it includes PTX that can runtime-compile for 8.6, but
    "8.0 8.6" would be better.

    Note that while it's possible to include all supported archs, the more archs get included the
    slower the building process will be, as it will build a separate kernel image for each arch.

    Note that CUDA-11.5 nvcc will hit internal compiler error while parsing torch/extension.h on Windows.
    To workaround the issue, move python binding logic to pure C++ file.

    Example use:
        #include <ATen/ATen.h>
        at::Tensor SigmoidAlphaBlendForwardCuda(....)

    Instead of:
        #include <torch/extension.h>
        torch::Tensor SigmoidAlphaBlendForwardCuda(...)

    Currently open issue for nvcc bug: https://github.com/pytorch/pytorch/issues/69460
    Complete workaround code example: https://github.com/facebookresearch/pytorch3d/commit/cb170ac024a949f1f9614ffe6af1c38d972f7d48

    Relocatable device code linking:

    If you want to reference device symbols across compilation units (across object files),
    the object files need to be built with `relocatable device code` (-rdc=true or -dc).
    An exception to this rule is "dynamic parallelism" (nested kernel launches)  which is not used a lot anymore.
    `Relocatable device code` is less optimized so it needs to be used only on object files that need it.
    Using `-dlto` (Device Link Time Optimization) at the device code compilation step and `dlink` step
    helps reduce the protentional perf degradation of `-rdc`.
    Note that it needs to be used at both steps to be useful.

    If you have `rdc` objects you need to have an extra `-dlink` (device linking) step before the CPU symbol linking step.
    There is also a case where `-dlink` is used without `-rdc`:
    when an extension is linked against a static lib containing rdc-compiled objects
    like the [NVSHMEM library](https://developer.nvidia.com/nvshmem).

    Note: Ninja is required to build a CUDA Extension with RDC linking.

    Example:
        >>> # xdoctest: +SKIP
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CPP_EXT)
        >>> CUDAExtension(
        ...     name="cuda_extension",
        ...     sources=["extension.cpp", "extension_kernel.cu"],
        ...     dlink=True,
        ...     dlink_libraries=["dlink_lib"],
        ...     extra_compile_args={"cxx": ["-g"], "nvcc": ["-O2", "-rdc=true"]},
        ... )
    """
    library_dirs = kwargs.get("library_dirs", [])
    library_dirs += library_paths(device_type="cuda")
    kwargs["library_dirs"] = library_dirs

    libraries = kwargs.get("libraries", [])
    libraries.append("c10")
    libraries.append("torch")
    libraries.append("torch_cpu")
    if not kwargs.get("py_limited_api", False):
        # torch_python uses more than the python limited api
        libraries.append("torch_python")
    if IS_HIP_EXTENSION:
        libraries.append("amdhip64")
        libraries.append("c10_hip")
        libraries.append("torch_hip")
    else:
        libraries.append("cudart")
        libraries.append("c10_cuda")
        libraries.append("torch_cuda")
    kwargs["libraries"] = libraries

    include_dirs = kwargs.get("include_dirs", [])

    if IS_HIP_EXTENSION:
        from ..hipify import hipify_python

        build_dir = os.getcwd()
        hipify_result = hipify_python.hipify(
            project_directory=build_dir,
            output_directory=build_dir,
            header_include_dirs=include_dirs,
            includes=[os.path.join(build_dir, "*")],  # limit scope to build_dir only
            extra_files=[os.path.abspath(s) for s in sources],
            show_detailed=True,
            is_pytorch_extension=True,
            hipify_extra_files_only=True,  # don't hipify everything in includes path
        )

        hipified_sources = set()
        for source in sources:
            s_abs = os.path.abspath(source)
            hipified_s_abs = (
                hipify_result[s_abs].hipified_path
                if (
                    s_abs in hipify_result
                    and hipify_result[s_abs].hipified_path is not None
                )
                else s_abs
            )
            # setup() arguments must *always* be /-separated paths relative to the setup.py directory,
            # *never* absolute paths
            hipified_sources.add(os.path.relpath(hipified_s_abs, build_dir))

        sources = list(hipified_sources)

    include_dirs += include_paths(device_type="cuda")
    kwargs["include_dirs"] = include_dirs

    kwargs["language"] = "c++"

    dlink_libraries = kwargs.get("dlink_libraries", [])
    dlink = kwargs.get("dlink", False) or dlink_libraries
    if dlink:
        extra_compile_args = kwargs.get("extra_compile_args", {})

        extra_compile_args_dlink = extra_compile_args.get("nvcc_dlink", [])
        extra_compile_args_dlink += ["-dlink"]
        extra_compile_args_dlink += [f"-L{x}" for x in library_dirs]
        extra_compile_args_dlink += [f"-l{x}" for x in dlink_libraries]

        if (torch.version.cuda is not None) and TorchVersion(
            torch.version.cuda
        ) >= "11.2":
            extra_compile_args_dlink += [
                "-dlto"
            ]  # Device Link Time Optimization started from cuda 11.2

        extra_compile_args["nvcc_dlink"] = extra_compile_args_dlink

        kwargs["extra_compile_args"] = extra_compile_args

    return setuptools.Extension(name, sources, *args, **kwargs)


def SyclExtension(name, sources, *args, **kwargs):
    r"""
    Creates a :class:`setuptools.Extension` for SYCL/C++.

    Convenience method that creates a :class:`setuptools.Extension` with the
    bare minimum (but often sufficient) arguments to build a SYCL/C++
    extension.

    All arguments are forwarded to the :class:`setuptools.Extension`
    constructor.

    .. warning::
        The PyTorch python API (as provided in libtorch_python) cannot be built
        with the flag ``py_limited_api=True``.  When this flag is passed, it is
        the user's responsibility in their library to not use APIs from
        libtorch_python (in particular pytorch/python bindings) and to only use
        APIs from libtorch (aten objects, operators and the dispatcher). For
        example, to give access to custom ops from python, the library should
        register the ops through the dispatcher.

        Contrary to CPython setuptools, who does not define -DPy_LIMITED_API
        as a compile flag when py_limited_api is specified as an option for
        the "bdist_wheel" command in ``setup``, PyTorch does! We will specify
        -DPy_LIMITED_API=min_supported_cpython to best enforce consistency,
        safety, and sanity in order to encourage best practices. To target a
        different version, set min_supported_cpython to the hexcode of the
        CPython version of choice.

    Example:
        >>> # xdoctest: +SKIP
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CPP_EXT)
        >>> from torch.utils.cpp_extension import BuildExtension, SyclExtension
        >>> setup(
        ...     name="xpu_extension",
        ...     ext_modules=[
        ...         SyclExtension(
        ...             name="xpu_extension",
        ...             sources=["extension.cpp", "extension_kernel.cpp"],
        ...             extra_compile_args={"cxx": ["-g", "-std=c++20", "-fPIC"]},
        ...         )
        ...     ],
        ...     cmdclass={"build_ext": BuildExtension},
        ... )

    By default the extension will be compiled to run on all archs of the cards visible during the
    building process of the extension. If down the road a new card is installed the
    extension may need to be recompiled. You can override the default behavior using
    `TORCH_XPU_ARCH_LIST` to explicitly specify which device architectures you want the extension
    to support:

    ``TORCH_XPU_ARCH_LIST="pvc,xe-lpg" python build_my_extension.py``

    Note that while it's possible to include all supported archs, the more archs get included the
    slower the building process will be, as it will build a separate kernel image for each arch.

    Note: Ninja is required to build SyclExtension.
    """
    library_dirs = kwargs.get("library_dirs", [])
    library_dirs += library_paths()
    kwargs["library_dirs"] = library_dirs

    libraries = kwargs.get("libraries", [])
    libraries.append("c10")
    libraries.append("c10_xpu")
    libraries.append("torch")
    libraries.append("torch_cpu")
    libraries.append("sycl")
    if not kwargs.get("py_limited_api", False):
        # torch_python uses more than the python limited api
        libraries.append("torch_python")
    libraries.append("torch_xpu")
    kwargs["libraries"] = libraries

    include_dirs = kwargs.get("include_dirs", [])
    include_dirs += include_paths(device_type="xpu")
    kwargs["include_dirs"] = include_dirs

    kwargs["language"] = "c++"

    return setuptools.Extension(name, sources, *args, **kwargs)
