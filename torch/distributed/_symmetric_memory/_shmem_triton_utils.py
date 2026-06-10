"""
Shared utilities for SHMEM Triton backends (_nvshmem_triton, _rocshmem_triton).

This is a leaf module — it does NOT import any backend or dispatch module,
so backend modules can safely import from here without circular dependencies.
"""

from collections.abc import Callable
from typing import Any

from torch.utils._triton import has_triton


class ShmemKernelRegistry:
    """Base registry tracking Triton kernels that need SHMEM module init."""

    _to_init: dict[str, Any] = {}

    @classmethod
    def register(cls, name: str) -> None:
        cls._to_init.setdefault(name)

    @classmethod
    def deregister(cls, name: str) -> None:
        cls._to_init.pop(name, None)

    @classmethod
    def has(cls, name: str) -> bool:
        return name in cls._to_init


def run_shmem_init_hook(
    *,
    kwargs: dict[str, Any],
    registry: type[ShmemKernelRegistry],
    module_init: Callable[[Any], None],
    logger: Any,
) -> None:
    """Common post-compile hook logic shared by NVSHMEM and rocSHMEM backends."""
    jit_function = kwargs["fn"].jit_function
    fn_name = jit_function.fn.__name__

    if not registry.has(fn_name):
        return

    key = kwargs["key"]
    device = kwargs["compile"]["device"]
    kernel_cache = jit_function.device_caches[device][0]
    kernel = kernel_cache.get(key, None)
    if kernel is not None:
        kernel.run
        module_init(kernel.module)
    else:
        logger.warning(
            "It seems Triton hasn't created a kernel for function %s. "
            "Please report this issue to Triton.",
            fn_name,
        )


if has_triton():
    from triton.runtime.jit import JITFunction, KernelInterface

    class GridCallableWithExtern(KernelInterface):
        """
        KernelInterface invokes self.run in __getitem__, i.e. [].
        We direct the call to JITFunction.run with extern_libs injected,
        so that users don't have to pass it.
        """

        def __init__(self, jit_func: JITFunction, extern_libs: dict[str, str]) -> None:
            self.jit_func = jit_func
            self.extern_libs = extern_libs

        def run(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            return self.jit_func.run(*args, **kwargs, extern_libs=self.extern_libs)

    def build_requires_shmem_decorator(  # type: ignore[no-untyped-def]
        *,
        jit_func,
        find_device_library: Callable[[], str],
        extern_libs_key: str,
        registry: type[ShmemKernelRegistry],
        init_hook: Callable[..., None],
        error_prefix: str,
    ):
        """Build the @requires_<backend> decorator for a Triton JIT function."""
        import triton
        from triton.runtime.jit import JITFunction

        if not isinstance(jit_func, JITFunction):
            raise TypeError(
                f"{error_prefix} must be applied to a @triton.jit function, "
                f"got {type(jit_func)}"
            )

        lib_path = find_device_library()
        extern_libs = {extern_libs_key: lib_path}
        registry.register(jit_func.fn.__name__)
        triton.knobs.runtime.jit_post_compile_hook = init_hook
        return GridCallableWithExtern(jit_func, extern_libs)
