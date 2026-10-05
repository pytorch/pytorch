from __future__ import annotations

import inspect
from typing import Any

import torch


try:
    import triton
except ImportError:
    triton = None


if triton is not None:
    import triton.language as tl
    from triton import Config
    from triton.compiler import CompiledKernel
    from triton.runtime.autotuner import OutOfResources
    from triton.runtime.jit import JITFunction, KernelInterface

    try:
        from triton.runtime.autotuner import PTXASError
    except ImportError:

        class PTXASError(Exception):  # type: ignore[no-redef]
            pass

    try:
        from triton.compiler.compiler import ASTSource
    except ImportError:
        ASTSource = None

    try:
        from triton.backends.compiler import GPUTarget
    except ImportError:

        def GPUTarget(
            backend: str,
            arch: int | str,
            warp_size: int,
        ) -> Any:
            if torch.version.hip:
                return [backend, arch, warp_size]
            return (backend, arch)

    # In the latest triton, math functions were shuffled around into different modules:
    # https://github.com/triton-lang/triton/pull/3172
    try:
        from triton.language.extra import libdevice

        libdevice = tl.extra.libdevice  # noqa: F811
        math = tl.math
    except ImportError:
        if hasattr(tl.extra, "cuda") and hasattr(tl.extra.cuda, "libdevice"):
            libdevice = tl.extra.cuda.libdevice
            math = tl.math
        elif hasattr(tl.extra, "intel") and hasattr(tl.extra.intel, "libdevice"):
            libdevice = tl.extra.intel.libdevice
            math = tl.math
        else:
            libdevice = tl.math
            math = tl

    try:
        from triton.language.standard import _log2
    except ImportError:

        def _log2(x: Any) -> Any:
            raise NotImplementedError

    def _triton_config_has(param_name: str) -> bool:
        if not hasattr(triton, "Config"):
            return False
        if not hasattr(triton.Config, "__init__"):
            return False
        return param_name in inspect.signature(triton.Config.__init__).parameters

    # Drop the legacy support of autoWS
    HAS_WARP_SPEC = False

    try:
        from triton import knobs
    except ImportError:
        knobs = None

    try:
        from triton.runtime.cache import triton_key  # type: ignore[attr-defined]
    except ImportError:
        from triton.compiler.compiler import (
            triton_key,  # type: ignore[attr-defined,no-redef]
        )

    try:
        from triton.runtime.errors import IntelGPUError
    except ImportError:

        class IntelGPUError(Exception):  # type: ignore[no-redef]
            pass

    def _validate_launcher_source(
        src: str, marker: str, scratch_binding_line: str
    ) -> None:
        """Validate Intel Triton launcher source has expected format.

        Args:
            src: Launcher source code to validate
            marker: Expected marker string in the source
            scratch_binding_line: Expected binding line (should not already be present)

        Raises:
            RuntimeError: If launcher format is unrecognized
        """
        if scratch_binding_line in src:
            # Binding already injected, nothing to do
            return
        if src and marker not in src:
            # Source is non-empty but lacks expected marker
            raise RuntimeError(
                "Intel Triton launcher source layout is not recognized; "
                "expected the 'if (shared_memory)' block for the "
                "global_scratch compatibility fix."
            )

    def _patch_triton_intel_launcher(intel_driver: Any | None = None) -> None:
        if intel_driver is None:
            try:
                import triton.backends.intel.driver as intel_driver  # type: ignore[import-not-found]
            except ImportError:
                # Intel Triton is not available in this environment.
                return

        make_launcher = getattr(intel_driver, "make_launcher", None)
        if make_launcher is None:
            # Older or unexpected Intel Triton builds may not expose this hook.
            return

        if getattr(make_launcher, "_torch_patched_global_scratch", False):
            # Avoid wrapping the launcher multiple times.
            return

        try:
            make_launcher_source = inspect.getsource(make_launcher)
        except (OSError, TypeError):
            # Some launchers may not have inspectable Python source.
            make_launcher_source = ""

        scratch_binding = (
            "    set_scalar_arg<void*>(cgh, num_params - 1, &global_scratch);\n"
        )
        scratch_binding_line = scratch_binding.strip()
        marker = "    if (shared_memory) {"

        # Validate launcher source and check if patch is needed
        _validate_launcher_source(make_launcher_source, marker, scratch_binding_line)
        if scratch_binding_line in make_launcher_source:
            # Newer launchers may already bind global_scratch correctly.
            return

        def patched_make_launcher(constants: Any, signature: Any) -> str:
            src = make_launcher(constants, signature)
            _validate_launcher_source(src, marker, scratch_binding_line)
            if scratch_binding_line in src:
                return src
            return src.replace(marker, scratch_binding + marker, 1)

        patched_make_launcher._torch_patched_global_scratch = True  # type: ignore[attr-defined]
        intel_driver.make_launcher = patched_make_launcher

    _patch_triton_intel_launcher()

    builtins_use_semantic_kwarg = (
        "_semantic" in inspect.signature(triton.language.core.view).parameters
    )
    HAS_TRITON = True
else:

    def _raise_error(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("triton package is not installed")

    class OutOfResources(Exception):  # type: ignore[no-redef]
        pass

    class PTXASError(Exception):  # type: ignore[no-redef]
        pass

    class IntelGPUError(Exception):  # type: ignore[no-redef]
        pass

    Config = object
    CompiledKernel = object
    KernelInterface = object
    ASTSource = None
    GPUTarget = None
    _log2 = _raise_error
    libdevice = None
    math = None
    knobs = None
    builtins_use_semantic_kwarg = False

    class triton:  # type: ignore[no-redef]
        @staticmethod
        def jit(*args: Any, **kwargs: Any) -> Any:
            return _raise_error

    class tl:  # type: ignore[no-redef]
        @staticmethod
        def constexpr(val: Any) -> Any:
            return val

        tensor = Any
        dtype = Any

    class JITFunction:  # type: ignore[no-redef]
        pass

    HAS_WARP_SPEC = False
    triton_key = _raise_error
    HAS_TRITON = False


try:
    autograd_profiler = torch.autograd.profiler
except AttributeError:  # Compile workers only have a mock version of torch

    class autograd_profiler:  # type: ignore[no-redef]
        _is_profiler_enabled = False


__all__ = [
    "Config",
    "CompiledKernel",
    "OutOfResources",
    "KernelInterface",
    "PTXASError",
    "IntelGPUError",
    "ASTSource",
    "GPUTarget",
    "tl",
    "_log2",
    "libdevice",
    "math",
    "triton",
    "knobs",
    "triton_key",
]
