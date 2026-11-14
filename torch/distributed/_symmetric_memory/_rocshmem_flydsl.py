"""rocSHMEM device API for FlyDSL kernels. ROCm only.

Call these inside ``@flyc.kernel``. The first call links
``librocshmem_device_<arch>.bc`` and registers ``rocshmem_hipmodule_init``
as the post-load hook, which is the same device bring-up Triton does in
``_rocshmem_triton.py``.

FlyDSL ``ffi`` has no pointer type. Pointer arguments are lowered to i64
and a small wrapper bitcode ``inttoptr``s them before the rocSHMEM symbol.
``$ROCM_PATH/llvm/bin/ld.lld`` must exist. When ``ROCM_PATH`` is unset, this
module points it at the prefix that shipped the device bitcode if that
linker is there. An explicit ``ROCM_PATH`` is left unchanged.

``put``, ``get``, ``get_nbi``, ``putmem_signal_block``, ``barrier_all``, and
``sync_all`` call rocSHMEM workgroup symbols. Every thread in the block
must call them with the same arguments. ``wait_until`` and
``signal_wait_until`` call ``rocshmem_int_wait_until`` and
``rocshmem_uint64_wait_until``.
"""

import atexit
import os
import shutil
import tempfile
from typing import Any

import torch
from torch.distributed._symmetric_memory._rocshmem_triton import RocshmemLibFinder


_WRAPPER_BC: str | None = None

# AMDGPU datalayout of librocshmem_device_*.bc. The wrapper is linked next
# to that bitcode, so it has to use the same layout.
_WRAPPER_LL = r"""
target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

declare void @rocshmem_putmem_wg(ptr, ptr, i64, i32)
declare void @rocshmem_getmem_wg(ptr, ptr, i64, i32)
declare void @rocshmem_getmem_nbi_wg(ptr, ptr, i64, i32)
declare void @rocshmem_putmem_signal_wg(ptr, ptr, i64, ptr, i64, i32, i32)
declare void @rocshmem_int_wait_until(ptr, i32, i32)
declare void @rocshmem_uint64_wait_until(ptr, i32, i64)

define void @fly_rocshmem_putmem_wg(i64 %dest, i64 %source, i64 %nbytes, i32 %pe) {
  %dest_ptr = inttoptr i64 %dest to ptr
  %source_ptr = inttoptr i64 %source to ptr
  call void @rocshmem_putmem_wg(ptr %dest_ptr, ptr %source_ptr, i64 %nbytes, i32 %pe)
  ret void
}

define void @fly_rocshmem_getmem_wg(i64 %dest, i64 %source, i64 %nbytes, i32 %pe) {
  %dest_ptr = inttoptr i64 %dest to ptr
  %source_ptr = inttoptr i64 %source to ptr
  call void @rocshmem_getmem_wg(ptr %dest_ptr, ptr %source_ptr, i64 %nbytes, i32 %pe)
  ret void
}

define void @fly_rocshmem_getmem_nbi_wg(i64 %dest, i64 %source, i64 %nbytes, i32 %pe) {
  %dest_ptr = inttoptr i64 %dest to ptr
  %source_ptr = inttoptr i64 %source to ptr
  call void @rocshmem_getmem_nbi_wg(ptr %dest_ptr, ptr %source_ptr, i64 %nbytes, i32 %pe)
  ret void
}

define void @fly_rocshmem_putmem_signal_wg(i64 %dest, i64 %source, i64 %nbytes, i64 %signal, i64 %sig_val, i32 %sig_op, i32 %pe) {
  %dest_ptr = inttoptr i64 %dest to ptr
  %source_ptr = inttoptr i64 %source to ptr
  %signal_ptr = inttoptr i64 %signal to ptr
  call void @rocshmem_putmem_signal_wg(ptr %dest_ptr, ptr %source_ptr, i64 %nbytes, ptr %signal_ptr, i64 %sig_val, i32 %sig_op, i32 %pe)
  ret void
}

define void @fly_rocshmem_int_wait_until(i64 %ivar, i32 %cmp, i32 %val) {
  %ivar_ptr = inttoptr i64 %ivar to ptr
  call void @rocshmem_int_wait_until(ptr %ivar_ptr, i32 %cmp, i32 %val)
  ret void
}

define void @fly_rocshmem_uint64_wait_until(i64 %signal, i32 %cmp, i64 %val) {
  %signal_ptr = inttoptr i64 %signal to ptr
  call void @rocshmem_uint64_wait_until(ptr %signal_ptr, i32 %cmp, i64 %val)
  ret void
}
"""


def rocshmem_flydsl_module_init(module_handle: int) -> None:
    """Copy ROCSHMEM_CTX_DEFAULT into a FlyDSL-loaded HIP module."""
    from torch._C._distributed_c10d import _nvshmemx_cumodule_init

    _nvshmemx_cumodule_init(module_handle)


def _device_bc() -> str:
    # The finder searches $ROCM_PATH/lib. When the user set neither ROCM_PATH
    # nor ROCSHMEM_LIB_DIR, point that search at hipcc's prefix, then drop the
    # assignment. _ensure_rocm_path must still see ROCM_PATH as unset.
    added = False
    if not os.environ.get("ROCSHMEM_LIB_DIR") and not os.environ.get("ROCM_PATH"):
        hipcc = shutil.which("hipcc")
        if hipcc:
            os.environ["ROCM_PATH"] = os.path.dirname(
                os.path.dirname(os.path.realpath(hipcc))
            )
            added = True
    try:
        return RocshmemLibFinder.find_device_library()
    finally:
        if added:
            os.environ.pop("ROCM_PATH", None)


def _lld(prefix: str) -> str:
    return os.path.join(prefix, "llvm", "bin", "ld.lld")


def _ensure_rocm_path(device_bc: str) -> str:
    """Return a ROCm prefix whose ld.lld FlyDSL can run.

    An explicit ``ROCM_PATH`` is left unchanged. When it is unset, point it
    at the prefix that shipped the device bitcode if that prefix has ld.lld.
    """
    explicit = os.environ.get("ROCM_PATH")
    if explicit:
        if os.path.isfile(_lld(explicit)):
            return explicit
        raise RuntimeError(
            "FlyDSL's gpu-module-to-binary pass runs $ROCM_PATH/llvm/bin/ld.lld. "
            f"ROCM_PATH={explicit!r} has no such linker."
        )
    prefix = os.path.dirname(os.path.dirname(os.path.abspath(device_bc)))
    if os.path.isfile(_lld(prefix)):
        os.environ["ROCM_PATH"] = prefix
        return prefix
    current = os.environ.get("ROCM_HOME") or "/opt/rocm"
    if os.path.isfile(_lld(current)):
        os.environ["ROCM_PATH"] = current
        return current
    raise RuntimeError(
        "FlyDSL's gpu-module-to-binary pass runs $ROCM_PATH/llvm/bin/ld.lld. "
        f"That linker was not found next to {device_bc} or under {current!r}."
    )


def _llvm_as(prefix: str) -> str:
    for relative in (
        os.path.join("lib", "llvm", "bin", "llvm-as"),
        os.path.join("llvm", "bin", "llvm-as"),
    ):
        path = os.path.join(prefix, relative)
        if os.path.isfile(path):
            return path
    raise RuntimeError(f"llvm-as not found under {prefix}")


def _drop_wrapper_dir(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)


def _wrapper_bc(prefix: str) -> str:
    global _WRAPPER_BC
    if _WRAPPER_BC is not None and os.path.isfile(_WRAPPER_BC):
        return _WRAPPER_BC
    # A private directory per process. A fixed path under the temp dir can be
    # owned by another user or be a planted symlink, and a shared .bc lets
    # two ranks truncate each other.
    cache_dir = tempfile.mkdtemp(prefix="rocshmem_flydsl_")
    atexit.register(_drop_wrapper_dir, cache_dir)
    ll_path = os.path.join(cache_dir, "fly_rocshmem_wrappers.ll")
    bc_path = os.path.join(cache_dir, "fly_rocshmem_wrappers.bc")
    with open(ll_path, "w") as ll_file:
        ll_file.write(_WRAPPER_LL)
    import subprocess

    subprocess.check_call([_llvm_as(prefix), ll_path, "-o", bc_path])
    _WRAPPER_BC = bc_path
    return bc_path


def _bind() -> None:
    """Link the device bitcode and the pointer wrapper into the kernel being traced."""
    if torch.version.hip is None:
        raise RuntimeError(
            "torch.distributed._symmetric_memory._rocshmem_flydsl is a ROCm-only "
            "module (torch.version.hip is None)."
        )
    from flydsl.compiler.kernel_function import CompilationContext

    ctx = CompilationContext.get_current()
    if ctx is None:
        raise RuntimeError("rocSHMEM FlyDSL ops must be called inside @flyc.kernel")
    device_bc = _device_bc()
    prefix = _ensure_rocm_path(device_bc)
    # The wrapper has to be linked first. FlyDSL links bitcode with
    # LinkOnlyNeeded, so rocshmem_* symbols referenced only by the wrapper
    # are dropped if the device bitcode is linked before the wrapper.
    ctx.add_link_lib(_wrapper_bc(prefix))
    ctx.add_link_lib(device_bc)
    if rocshmem_flydsl_module_init not in ctx.post_load_processors:
        ctx.post_load_processors.append(rocshmem_flydsl_module_init)


def _ffi(symbol: str, arg_types: list[str], ret_type: str) -> Any:
    from flydsl.expr.extern import ffi

    return ffi(symbol, arg_types, ret_type)


def _as_addr(value: Any) -> Any:
    """Device pointer of a kernel memref argument, as i64.

    FlyDSL ``ffi`` has no pointer type, and kernel tensors arrive as
    ``!fly.memref`` rather than ``!fly.ptr``.
    """
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import fly as fly_d, llvm as llvm_d
    from flydsl.expr.utils.arith import ArithValue

    raw = value.__extract_to_ir_values__()[0]
    ptr = fly_d.extract_aligned_pointer_as_index(ir.Type.parse("!llvm.ptr<1>"), raw)
    return ArithValue(llvm_d.ptrtoint(ir.IntegerType.get_signless(64), ptr))


def _elem_type(value: Any) -> Any:
    elem = getattr(value, "element_type", None)
    if elem is None:
        return value.dtype
    return elem


def _mem_op(symbol: str, dest: Any, source: Any, nelems: Any, pe: Any) -> None:
    dest_ty = _elem_type(dest)
    source_ty = _elem_type(source)
    if dest_ty != source_ty:
        raise RuntimeError(
            f"dest and source element types must match, got {dest_ty} and {source_ty}"
        )
    _bind()
    _ffi(symbol, ["int64", "int64", "int64", "int32"], "void")(
        _as_addr(dest),
        _as_addr(source),
        nelems * (dest_ty.width // 8),
        pe,
    )


def put(dest: Any, source: Any, nelems: Any, pe: Any) -> None:
    """Put ``nelems`` elements from local ``source`` to ``dest`` on ``pe``.

    ``dest`` and ``source`` must have the same element type.
    Every thread in the block must call this with the same arguments.
    """
    _mem_op("fly_rocshmem_putmem_wg", dest, source, nelems, pe)


def get(dest: Any, source: Any, nelems: Any, pe: Any) -> None:
    """Get ``nelems`` elements from ``source`` on ``pe`` into local ``dest``.

    ``dest`` and ``source`` must have the same element type.
    Every thread in the block must call this with the same arguments.
    """
    _mem_op("fly_rocshmem_getmem_wg", dest, source, nelems, pe)


def get_nbi(dest: Any, source: Any, nelems: Any, pe: Any) -> None:
    """Non-blocking get. Call ``quiet`` before reading ``dest``.

    ``dest`` and ``source`` must have the same element type.
    Every thread in the block must call this with the same arguments.
    """
    _mem_op("fly_rocshmem_getmem_nbi_wg", dest, source, nelems, pe)


def putmem_signal_block(
    dest: Any,
    source: Any,
    nbytes: Any,
    signal: Any,
    sig_val: Any,
    sig_op: Any,
    pe: Any,
) -> None:
    """Put ``nbytes`` bytes and update a remote uint64 signal.

    ``sig_op`` is a ``ROCSHMEM_SIGNAL_OPS`` value: ``ROCSHMEM_SIGNAL_SET`` (0)
    or ``ROCSHMEM_SIGNAL_ADD`` (1). ``ROCSHMEM_SIGNAL_ADD`` is 1;
    ``NVSHMEM_SIGNAL_ADD`` is 5.
    ``sig_val`` is zero-extended to uint64. A kernel ``int`` is 32-bit, and
    sign-extending it would set the high half of any value with bit 31 set.
    Every thread in the block must call this with the same arguments.
    """
    _bind()
    _ffi(
        "fly_rocshmem_putmem_signal_wg",
        ["int64", "int64", "int64", "int64", "uint64", "int32", "int32"],
        "void",
    )(
        _as_addr(dest),
        _as_addr(source),
        nbytes,
        _as_addr(signal),
        sig_val,
        sig_op,
        pe,
    )


def _void(symbol: str) -> None:
    _bind()
    _ffi(symbol, [], "void")()


def quiet() -> None:
    """Wait for outstanding remote-memory operations."""
    _void("rocshmem_quiet")


def fence() -> None:
    """Order remote-memory operations to each target PE."""
    _void("rocshmem_fence")


def barrier_all() -> None:
    """Workgroup barrier across all PEs.

    Every thread in the block must call this with the same arguments.
    """
    _void("rocshmem_barrier_all_wg")


def sync_all() -> None:
    """Workgroup sync across all PEs.

    Every thread in the block must call this with the same arguments.
    """
    _void("rocshmem_sync_all_wg")


def _i32(symbol: str) -> Any:
    _bind()
    from flydsl.expr.utils.arith import ArithValue

    return ArithValue(_ffi(symbol, [], "int32")())


def my_pe() -> Any:
    """PE number of the caller."""
    return _i32("rocshmem_my_pe")


def n_pes() -> Any:
    """Number of PEs."""
    return _i32("rocshmem_n_pes")


def _width_bits(value: Any) -> int:
    elem = _elem_type(value)
    width = getattr(elem, "width", None)
    if width is None:
        return elem.itemsize * 8
    return width


def wait_until(ivar: Any, cmp_op: Any, cmp_val: Any) -> None:
    """Block until 32-bit ``ivar`` satisfies ``cmp_op`` against ``cmp_val``.

    ``cmp_op`` is a ``rocshmem_cmps`` value: ``ROCSHMEM_CMP_EQ`` (0),
    ``ROCSHMEM_CMP_NE`` (1), ``ROCSHMEM_CMP_GT`` (2), ``ROCSHMEM_CMP_GE`` (3),
    ``ROCSHMEM_CMP_LT`` (4), ``ROCSHMEM_CMP_LE`` (5).
    """
    width = _width_bits(ivar)
    if width != 32:
        raise RuntimeError(
            f"wait_until expects a 32-bit synchronization variable, got width {width}"
        )
    _bind()
    _ffi("fly_rocshmem_int_wait_until", ["int64", "int32", "int32"], "void")(
        _as_addr(ivar),
        cmp_op,
        cmp_val,
    )


def signal_wait_until(signal: Any, cmp_op: Any, cmp_val: Any) -> None:
    """Block until uint64 ``signal`` satisfies ``cmp_op`` against ``cmp_val``.

    ``cmp_op`` is a ``rocshmem_cmps`` value: ``ROCSHMEM_CMP_EQ`` (0),
    ``ROCSHMEM_CMP_NE`` (1), ``ROCSHMEM_CMP_GT`` (2), ``ROCSHMEM_CMP_GE`` (3),
    ``ROCSHMEM_CMP_LT`` (4), ``ROCSHMEM_CMP_LE`` (5).
    """
    width = _width_bits(signal)
    if width != 64:
        raise RuntimeError(
            f"signal_wait_until expects a 64-bit signal variable, got width {width}"
        )
    _bind()
    _ffi("fly_rocshmem_uint64_wait_until", ["int64", "int32", "uint64"], "void")(
        _as_addr(signal),
        cmp_op,
        cmp_val,
    )


def signal_op(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError(
        "rocshmem has no device-bitcode equivalent for signal_op. "
        "Use rocshmem_uint64_atomic_set or rocshmem_uint64_atomic_add instead."
    )


def alltoall(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError(
        "rocshmem_alltoallmem_wg is not available in current device bitcode. "
        "Use host-side rocshmem_alltoallmem_on_stream instead."
    )


def broadcast(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError(
        "rocshmem_broadcastmem_wg is not available in current device bitcode. "
        "Use host-side rocshmem_broadcastmem_on_stream instead."
    )


def reduce(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError(
        "rocshmem team reduce is not available in current device bitcode. "
        "Use host-side rocshmem reduce API instead."
    )


__all__ = [
    "alltoall",
    "barrier_all",
    "broadcast",
    "fence",
    "get",
    "get_nbi",
    "my_pe",
    "n_pes",
    "put",
    "putmem_signal_block",
    "quiet",
    "reduce",
    "rocshmem_flydsl_module_init",
    "signal_op",
    "signal_wait_until",
    "sync_all",
    "wait_until",
]
