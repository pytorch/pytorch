import os
import subprocess
import sysconfig
from typing import Optional

from torch.utils._triton import has_triton


def _find_nvshmem_device_library() -> str:
    paths = [os.path.join(sysconfig.get_path("purelib"), "nvidia", "nvshmem", "lib")]

    # Add common system installation paths
    common_paths = [
        "/usr/local/lib",
        "/usr/lib",
        "/opt/nvidia/nvshmem/lib",
    ]
    paths.extend(common_paths)

    try:
        import torch

        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        so_path = os.path.join(torch_lib, "libtorch_nvshmem.so")

        if os.path.exists(so_path):
            try:
                result = subprocess.run(
                    ["readelf", "-d", so_path],
                    capture_output=True,
                    text=True,
                    check=True,
                )

                for line in result.stdout.splitlines():
                    if ("RPATH" in line or "RUNPATH" in line) and "[" in line:
                        rpath = line.split("[", 1)[1].split("]", 1)[0]
                        for p in rpath.split(":"):
                            p = p.strip().replace("$ORIGIN", torch_lib)
                            if p and p not in paths:
                                paths.append(p)
            except subprocess.CalledProcessError:
                pass

    except ImportError:
        pass

    for path in paths:
        device_lib = os.path.join(path, "libnvshmem_device.bc")
        if os.path.exists(device_lib):
            return device_lib

    raise RuntimeError(f"NVSHMEM device library not found. Searched: {paths}")


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
    import triton

    from torch._C._distributed_c10d import _nvshmemx_cumodule_init

    if lib_dir is not None:
        lib_path = os.path.join(lib_dir, "libnvshmem_device.bc")
        if not os.path.exists(lib_path):
            raise RuntimeError(
                f"NVSHMEM device library not found at specified path: {lib_path}"
            )
    else:
        # Otherwise, search for the library automatically.
        lib_path = _find_nvshmem_device_library()

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
    triton.knobs.runtime.jit_post_compile_hook = nvshmem_init_hook

    # Return to user so that they can use it in Triton kernel invocation
    return extern_libs


if has_triton():
    from triton.language import core

    # RMA Operations (mem-based APIs - sizes in bytes)
    @core.extern
    def putmem_block(dst, src, size_bytes, pe, _semantic=None):  # type: ignore[no-untyped-def]
        """Put data to remote PE. size_bytes specifies the size in bytes."""
        return core.extern_elementwise(
            "",
            "",
            [dst, src, size_bytes, pe],
            {
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("nvshmemx_putmem_block", core.dtype("int32"))
            },
            is_pure=False,
            _semantic=_semantic,
        )

    @core.extern
    def getmem_block(dst, src, size_bytes, pe, _semantic=None):  # type: ignore[no-untyped-def]
        """Get data from remote PE. size_bytes specifies the size in bytes."""
        return core.extern_elementwise(
            "",
            "",
            [dst, src, size_bytes, pe],
            {
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("nvshmemx_getmem_block", core.dtype("int32"))
            },
            is_pure=False,
            _semantic=_semantic,
        )

    @core.extern
    def putmem_signal_block(  # type: ignore[no-untyped-def]
        dst,
        src,
        size_bytes,
        sig_addr,
        signal,
        sig_op,
        pe,
        _semantic=None,
    ):  # type: ignore[no-untyped-def]
        """Put data to remote PE with signal. size_bytes specifies the size in bytes."""
        return core.extern_elementwise(
            "",
            "",
            [dst, src, size_bytes, sig_addr, signal, sig_op, pe],
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
            _semantic=_semantic,
        )

    # Wait and Signal Operations
    @core.extern
    def wait_until(ivar, cmp, cmp_val, _semantic=None):  # type: ignore[no-untyped-def]
        """Wait until a condition is met on a symmetric variable."""
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
            _semantic=_semantic,
        )

    @core.extern
    def signal_wait_until(sig_addr, cmp, cmp_val, _semantic=None):  # type: ignore[no-untyped-def]
        """Wait until a signal variable meets a condition."""
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
            _semantic=_semantic,
        )

    @core.extern
    def signal_op(sig_addr, signal, sig_op, pe, _semantic=None):  # type: ignore[no-untyped-def]
        """Perform a signal operation on a remote PE."""
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
            _semantic=_semantic,
        )

    # Memory Ordering Operations
    @core.extern
    def fence(_semantic=None):  # type: ignore[no-untyped-def]
        """Ensure ordering of put operations."""
        return core.extern_elementwise(
            "",
            "",
            [],
            {
                (): ("nvshmem_fence", core.dtype("int32")),
            },
            is_pure=False,
            _semantic=_semantic,
        )

    @core.extern
    def quiet(_semantic=None):  # type: ignore[no-untyped-def]
        """Wait for completion of all outstanding put operations."""
        return core.extern_elementwise(
            "",
            "",
            [],
            {
                (): ("nvshmem_quiet", core.dtype("int32")),
            },
            is_pure=False,
            _semantic=_semantic,
        )

    # PE Information Operations
    @core.extern
    def my_pe(_semantic=None):  # type: ignore[no-untyped-def]
        """Get the PE number of the calling PE."""
        return core.extern_elementwise(
            "",
            "",
            [],
            {(): ("nvshmem_my_pe", core.dtype("int32"))},
            is_pure=True,
            _semantic=_semantic,
        )

    @core.extern
    def n_pes(_semantic=None):  # type: ignore[no-untyped-def]
        """Get the total number of PEs."""
        return core.extern_elementwise(
            "",
            "",
            [],
            {(): ("nvshmem_n_pes", core.dtype("int32"))},
            is_pure=True,
            _semantic=_semantic,
        )

    # Synchronization Operations
    @core.extern
    def barrier_all(_semantic=None):  # type: ignore[no-untyped-def]
        """Synchronize all PEs."""
        return core.extern_elementwise(
            "",
            "",
            [],
            {(): ("nvshmem_barrier_all", core.dtype("int32"))},
            is_pure=False,
            _semantic=_semantic,
        )

    @core.extern
    def sync_all(_semantic=None):  # type: ignore[no-untyped-def]
        """Synchronize all PEs (lightweight version, does not ensure completion of remote memory updates)."""
        return core.extern_elementwise(
            "",
            "",
            [],
            {(): ("nvshmem_sync_all", core.dtype("int32"))},
            is_pure=False,
            _semantic=_semantic,
        )

    # Collective Operations (mem-based APIs - sizes in bytes)
    @core.extern
    def alltoallmem_block(team, dest, source, size_bytes, _semantic=None):  # type: ignore[no-untyped-def]
        """Perform alltoall operation on symmetric memory. size_bytes specifies the number of bytes to exchange per PE."""
        return core.extern_elementwise(
            "",
            "",
            [team, dest, source, size_bytes],
            {
                (
                    core.dtype("int64"),  # team handle
                    core.dtype("int64"),  # dest ptr
                    core.dtype("int64"),  # source ptr
                    core.dtype("int64"),  # size in bytes
                ): ("nvshmemx_alltoallmem_block", core.dtype("int32"))
            },
            is_pure=False,
            _semantic=_semantic,
        )

    @core.extern
    def broadcastmem_block(team, dest, source, size_bytes, pe_root, _semantic=None):  # type: ignore[no-untyped-def]
        """Broadcast data from a root PE to all other PEs in a team. size_bytes specifies the size in bytes."""
        return core.extern_elementwise(
            "",
            "",
            [team, dest, source, size_bytes, pe_root],
            {
                (
                    core.dtype("int64"),  # team handle
                    core.dtype("int64"),  # dest ptr
                    core.dtype("int64"),  # source ptr
                    core.dtype("int64"),  # size in bytes
                    core.dtype("int64"),  # pe_root
                ): ("nvshmemx_broadcastmem_block", core.dtype("int32"))
            },
            is_pure=False,
            _semantic=_semantic,
        )

    # Reduction Operations
    @core.extern
    def sum_reduce(team, dest, source, nreduce, _semantic=None):  # type: ignore[no-untyped-def]
        """Sum reduction for int64. nreduce is number of elements in the dest and source arrays."""
        return core.extern_elementwise(
            "",
            "",
            [team, dest, source, nreduce],
            {
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("nvshmem_int64_sum_reduce", core.dtype("int32"))
            },
            is_pure=False,
            _semantic=_semantic,
        )

    @core.extern
    def max_reduce(team, dest, source, nreduce, _semantic=None):  # type: ignore[no-untyped-def]
        """Max reduction for int64. nreduce is number of elements in the dest and source arrays."""
        return core.extern_elementwise(
            "",
            "",
            [team, dest, source, nreduce],
            {
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("nvshmem_int64_max_reduce", core.dtype("int32"))
            },
            is_pure=False,
            _semantic=_semantic,
        )

    @core.extern
    def min_reduce(team, dest, source, nreduce, _semantic=None):  # type: ignore[no-untyped-def]
        """Min reduction for int64. nreduce is number of elements in the dest and source arrays."""
        return core.extern_elementwise(
            "",
            "",
            [team, dest, source, nreduce],
            {
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                    core.dtype("int64"),
                    core.dtype("int64"),
                ): ("nvshmem_int64_min_reduce", core.dtype("int32"))
            },
            is_pure=False,
            _semantic=_semantic,
        )
