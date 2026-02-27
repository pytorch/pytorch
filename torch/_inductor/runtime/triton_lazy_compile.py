"""
Lazy Triton kernel compilation for C++ wrapper.

This module provides functionality to compile and autotune Triton kernels at runtime
when the C++ wrapper is used without autotune_at_compile_time.

The workflow is:
1. At model initialization: Call start_kernel_compile() for all kernels to start
   parallel compilation using multi-process async_compile
2. At kernel execution time: Call run_triton_kernel_with_autotune() which waits
   for the specific kernel to be ready, then runs it with autotuning
"""

from __future__ import annotations

import dataclasses
import logging
import re
from typing import Any

from .triton_heuristics import CachingAutotuner


log = logging.getLogger(__name__)


@dataclasses.dataclass
class TritonKernelCompileResult:
    cubin_path: str
    mangled_name: str
    num_warps: int
    shared_mem: int
    xblock: int
    yblock: int
    zblock: int
    r0block: int
    rsplit: int
    rsplit_size: int
    config_index: int | None
    global_scratch: int | None
    profile_scratch: int | None


_pending_kernels: dict[str, Any] = {}

_async_compile: Any = None


def _get_async_compile() -> Any:
    """Get or create the shared AsyncCompile instance."""
    global _async_compile
    if _async_compile is None:
        from torch._inductor.async_compile import AsyncCompile

        _async_compile = AsyncCompile()
    return _async_compile


def _wrap_tma_args(args: list[Any], kernel_fn: CachingAutotuner) -> list[Any]:
    """Wrap tensor args with TMA descriptors where the signature requires them."""
    signature = kernel_fn.triton_meta.get("signature", {})
    sig_items = list(signature.items())

    # Track args index separately from sig_items index since the signature
    # may include constexpr entries that are not present in args.
    tma_indices = []
    arg_idx = 0
    for name, sig_type in sig_items:
        if isinstance(sig_type, str) and sig_type == "constexpr":
            continue
        if isinstance(sig_type, str) and (
            sig_type == "nvTmaDesc" or sig_type.startswith("tensordesc<")
        ):
            tma_indices.append((arg_idx, name, sig_type))
        arg_idx += 1

    if not tma_indices:
        return args

    from triton.tools.tensor_descriptor import TensorDescriptor

    wrapped = list(args)
    for arg_idx, name, sig_type in tma_indices:
        if arg_idx >= len(wrapped):
            break
        tensor = wrapped[arg_idx]
        # Parse block_shape from tensordesc<dtype[dim0, dim1, ...]>
        match = re.match(r"tensordesc<[^[]*\[([^\]]*)\]", sig_type)
        if match:
            block_shape = [int(x.strip()) for x in match.group(1).split(",")]
            wrapped[arg_idx] = TensorDescriptor.from_tensor(tensor, block_shape)

    return wrapped


def start_kernel_compile(kernel_name: str, kernel_source: str) -> None:
    """
    This function is called from C++ at model initialization time for each kernel.
    It starts the compilation in a background process but does NOT wait for it.
    The actual kernel execution happens later in run_triton_kernel_with_autotune().
    """
    if kernel_name in _pending_kernels:
        return

    async_compile = _get_async_compile()  # noqa: F841 (used by eval below)

    # Evaluate the kernel source to get the Future or CachingAutotuner
    # The kernel_source is like: async_compile.triton('name', '''...''', ...)
    kernel_obj = eval(kernel_source.strip())  # noqa: S307

    _pending_kernels[kernel_name] = kernel_obj


def run_triton_kernel_with_autotune(
    kernel_name: str,
    stream: Any,
    args: list[Any],
) -> TritonKernelCompileResult:
    """
    Run a Triton kernel with full autotuning using actual tensor arguments.
    """
    from torch._inductor.codecache import CodeCacheFuture, CudaKernelParamCache
    from torch._inductor.runtime.triton_heuristics import config_to_dict

    if kernel_name not in _pending_kernels:
        raise RuntimeError(f"Kernel {kernel_name} not found in pending kernels. ")
    kernel_obj = _pending_kernels.pop(kernel_name)

    if isinstance(kernel_obj, CodeCacheFuture):
        kernel_fn = kernel_obj.result()
    elif isinstance(kernel_obj, CachingAutotuner):
        kernel_fn = kernel_obj
    else:
        raise RuntimeError(f"Unexpected kernel object type: {type(kernel_obj)}")

    assert isinstance(kernel_fn, CachingAutotuner)

    inductor_meta = kernel_fn.inductor_meta
    inductor_meta["store_cubin"] = True

    # For TMA kernels, wrap tensor args with TMA descriptors
    args = _wrap_tma_args(args, kernel_fn)

    # Run the kernel with the provided arguments
    # This will trigger autotuning if there are multiple configs
    kernel_fn.run(*args, stream=stream)
    if not kernel_fn.launchers:
        raise RuntimeError("Kernel run did not produce any launchers")
    launcher = kernel_fn.launchers[0]

    cached_params: dict[str, Any] | None = CudaKernelParamCache.get(kernel_name)
    if cached_params is None:
        raise RuntimeError(f"Failed to get cached params for kernel {kernel_name}")

    from torch._inductor.codecache import get_cpp_wrapper_cubin_path_name

    cubin_path_name = get_cpp_wrapper_cubin_path_name()
    for key in (cubin_path_name, "mangled_name", "num_warps", "shared_mem"):
        if key not in cached_params:
            raise RuntimeError(f"{key} not found in cached params for {kernel_name}")
    cubin_path = cached_params[cubin_path_name]
    mangled_name = cached_params["mangled_name"]
    num_warps = cached_params["num_warps"]
    shared_mem = cached_params["shared_mem"]

    config = config_to_dict(launcher.config) if launcher.config else {}
    xblock = config.get("XBLOCK", 128)
    yblock = config.get("YBLOCK", 1)
    zblock = config.get("ZBLOCK", 1)
    r0block = config.get("R0_BLOCK", 1)
    rsplit = config.get("RSPLIT", 1)
    rsplit_size = config.get("RSPLIT_SIZE", 1)

    config_index = None
    grid_type = inductor_meta.get("grid_type") if inductor_meta else None
    if grid_type == "PrecomputedGrid" and inductor_meta:
        # PrecomputedGrid selects one of precomputed_grids. We use config_index
        # to remember which grid is chosen.
        precomputed_grids = inductor_meta.get("precomputed_grids", [])
        for idx, entry in enumerate(precomputed_grids):
            entry_config = entry.get("config", {})
            if all(config.get(k) == v for k, v in entry_config.items()):
                config_index = idx
                break

    global_scratch: int | None = cached_params.get("global_scratch")
    profile_scratch: int | None = cached_params.get("profile_scratch")

    log.debug(
        "Successfully autotuned Triton kernel: cubin_path=%s, mangled_name=%s, "
        "num_warps=%d, shared_mem=%d, xblock=%d, yblock=%d, zblock=%d, r0block=%d, "
        "rsplit=%d, rsplit_size=%d, config_index=%s, global_scratch=%s, profile_scratch=%s",
        cubin_path,
        mangled_name,
        num_warps,
        shared_mem,
        xblock,
        yblock,
        zblock,
        r0block,
        rsplit,
        rsplit_size,
        config_index,
        global_scratch,
        profile_scratch,
    )

    return TritonKernelCompileResult(
        cubin_path=cubin_path,
        mangled_name=mangled_name,
        num_warps=num_warps,
        shared_mem=shared_mem,
        xblock=xblock,
        yblock=yblock,
        zblock=zblock,
        r0block=r0block,
        rsplit=rsplit,
        rsplit_size=rsplit_size,
        config_index=config_index,
        global_scratch=global_scratch,
        profile_scratch=profile_scratch,
    )
