import os
import sysconfig
from typing import Optional

from torch.utils._triton import has_triton


def enable_triton(lib_dir: Optional[str] = None) -> dict[str, str]:
    """
    Enable NVSHMEM device functions for Triton. It performs a NVSHMEM
    device-side initialization on the kernel module created by Triton.

    Args:
        lib_dir (Optional[str]): The directory where the NVSHMEM device library
        is located. If not provided, it will use the default path where NVSHMEM
        wheel is installed.

    Returns:
        dict[str, str]: A dictionary containing the NVSHMEM device library name
        and path.
    """
    from triton.runtime.jit import JITFunction

    from torch._C._distributed_c10d import _nvshmemx_cumodule_init

    # Detect NVSHMEM device library path from python library path
    if lib_dir is None:
        py_lib_path = sysconfig.get_path("purelib")
        lib_dir = py_lib_path + "/nvidia/nvshmem/lib"

    lib_path = os.path.join(lib_dir, "libnvshmem_device.bc")
    if not os.path.exists(lib_path):
        raise RuntimeError("NVSHMEM device library not found")

    extern_libs = {"libnvshmem_device": lib_path}

    # A hook function to initialize NVSHMEM in Triton
    def nvshmem_init_hook(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        key = kwargs["key"]
        device = kwargs["compile"]["device"]
        jit_function = kwargs["fn"].jit_function
        kernel_cache, _, _, _ = jit_function.device_caches[device]
        kernel = kernel_cache.get(key, None)
        kernel.run
        _nvshmemx_cumodule_init(kernel.module)

    # Register the function as a post-compile hook
    JITFunction.compiled_hook = nvshmem_init_hook

    # Return to user so that they can use it in Triton kernel invocation
    return extern_libs


if has_triton():
    from triton.language import core

    @core.extern
    def putmem_block(dst, src, nelems, pe, _builder=None):  # type: ignore[no-untyped-def]
        return core.extern_elementwise(
            "",
            "",
            [dst, src, nelems, pe],
            {
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("nvshmemx_putmem_block", core.dtype("int32"))
            },
            is_pure=False,
            _builder=_builder,
        )

    @core.extern
    def getmem_block(dst, src, nelems, pe, _builder=None):  # type: ignore[no-untyped-def]
        return core.extern_elementwise(
            "",
            "",
            [dst, src, nelems, pe],
            {
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("nvshmemx_getmem_block", core.dtype("int32"))
            },
            is_pure=False,
            _builder=_builder,
        )

    @core.extern
    def putmem_signal_block(  # type: ignore[no-untyped-def]
        dst,
        src,
        nelems,
        sig_addr,
        signal,
        sig_op,
        pe,
        _builder=None,
    ):  # type: ignore[no-untyped-def]
        return core.extern_elementwise(
            "",
            "",
            [dst, src, nelems, sig_addr, signal, sig_op, pe],
            {
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                    core.dtype("int64"),
                    core.dtype("int64"),
                    core.dtype("int64"),
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("nvshmemx_putmem_signal_block", core.dtype("int32"))
            },
            is_pure=False,
            _builder=_builder,
        )

    @core.extern
    def wait_until(ivar, cmp, cmp_val, _builder=None):  # type: ignore[no-untyped-def]
        return core.extern_elementwise(
            "",
            "",
            [ivar, cmp, cmp_val],
            {
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("nvshmem_longlong_wait_until", core.dtype("int32"))
            },
            is_pure=False,
            _builder=_builder,
        )

    @core.extern
    def signal_wait_until(sig_addr, cmp, cmp_val, _builder=None):  # type: ignore[no-untyped-def]
        return core.extern_elementwise(
            "",
            "",
            [sig_addr, cmp, cmp_val],
            {
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("nvshmem_signal_wait_until", core.dtype("int32"))
            },
            is_pure=False,
            _builder=_builder,
        )

    @core.extern
    def signal_op(sig_addr, signal, sig_op, pe, _builder=None):  # type: ignore[no-untyped-def]
        return core.extern_elementwise(
            "",
            "",
            [sig_addr, signal, sig_op, pe],
            {
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("nvshmemx_signal_op", core.dtype("int32"))
            },
            is_pure=False,
            _builder=_builder,
        )

    @core.extern
    def fence(_builder=None):  # type: ignore[no-untyped-def]
        return core.extern_elementwise(
            "",
            "",
            [],
            {
                (): ("nvshmem_fence", core.dtype("int32")),
            },
            is_pure=False,
            _builder=_builder,
        )

    @core.extern
    def quiet(_builder=None):  # type: ignore[no-untyped-def]
        return core.extern_elementwise(
            "",
            "",
            [],
            {
                (): ("nvshmem_quiet", core.dtype("int32")),
            },
            is_pure=False,
            _builder=_builder,
        )

    @core.extern
    def my_pe(_builder=None):  # type: ignore[no-untyped-def]
        return core.extern_elementwise(
            "",
            "",
            [],
            {(): ("nvshmem_my_pe", core.dtype("int32"))},
            is_pure=True,
            _builder=_builder,
        )

    @core.extern
    def n_pes(_builder=None):  # type: ignore[no-untyped-def]
        return core.extern_elementwise(
            "",
            "",
            [],
            {(): ("nvshmem_n_pes", core.dtype("int32"))},
            is_pure=True,
            _builder=_builder,
        )

    @core.extern
    def barrier_all(_builder=None):  # type: ignore[no-untyped-def]
        return core.extern_elementwise(
            "",
            "",
            [],
            {(): ("nvshmem_barrier_all", core.dtype("int32"))},
            is_pure=False,
            _builder=_builder,
        )

    @core.extern
    def sync_all(_builder=None):  # type: ignore[no-untyped-def]
        return core.extern_elementwise(
            "",
            "",
            [],
            {(): ("nvshmem_sync_all", core.dtype("int32"))},
            is_pure=False,
            _builder=_builder,
        )
