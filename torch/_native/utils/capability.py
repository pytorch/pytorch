# Per-tensor capability gating shared by ALL CuteDSL native-op override families
# (reductions today, pointwise/etc. later). Every override's `cond` runs on EVERY
# eager dispatch, so these must be cheap and must NEVER raise (a throwing cond would
# crash the dispatcher instead of falling back to aten).
#
# These three checks are op-agnostic -- they say nothing about reductions, dims, or
# geometry, only "can our kernels run on this tensor's device/storage at all". Op
# families layer their own dtype / geometry / shape checks on top.

from __future__ import annotations

import torch
import torch._subclasses.fake_tensor


def is_cow(t: torch.Tensor) -> bool:
    # A copy-on-write tensor would be MATERIALIZED (a real copy) the moment our
    # launch path reads its data pointer via DLPack export -- which the autograd
    # backward contract forbids for a transparent op override. Decline COW inputs
    # and let aten handle them.
    return torch._C._is_cow_tensor(t)  # pyrefly: ignore[missing-attribute]


def dlpack_offset_ok(t: torch.Tensor) -> bool:
    """TEMPORARY: decline any tensor whose DLPack byte_offset would be nonzero.

    CuteDSL's tvm-ffi entry point loads `byte_offset` from the DLTensor purely to assert
    it is ZERO (base_dsl/tvm_ffi_builder/tvm_ffi_builder.py -- the value is never added
    to `data`, so the kernel addresses the storage BASE). pytorch#182924 then made DLPack
    export report `byte_offset = storage_offset * itemsize` for every device instead of
    folding it into `data`, so any view with a nonzero storage offset -- x[4:], chunk /
    split / narrow, an arena or KV-cache slice -- now trips that assert mid-call.

    Serving such a tensor is not an option: with the assert suppressed the kernel would
    silently read and write from the storage base. So this is a genuine CAPABILITY limit
    ("we cannot hand CuteDSL a nonzero byte_offset"), the same species as the neg/conj
    and 16-byte-alignment gates -- not a performance threshold. Declining costs only the
    acceleration; aten computes the same values.

    Remove this (and its call sites) once CuteDSL folds byte_offset into the pointer on
    ingest, which it already does for direct-address devices in
    tvm/ffi/container/tensor.h, or drops the assert. Tracked as task #38; standalone
    repro in agent_space/cutedsl_byte_offset_repro.py.
    """
    return t.storage_offset() == 0


def is_traced(t: torch.Tensor) -> bool:
    # FakeTensor (compile/export tracing) and meta tensors have no real storage to
    # launch a kernel on; the decomposition / shape-inference path should use aten's
    # reference, not our kernel. Decline so the compile router falls back. (Eager on
    # real tensors is unaffected -- that is the path our kernels target.)
    #
    # This cond runs on EVERY eager dispatch, and is_fake() is a ~0.5us Python
    # subclass walk. A FakeTensor / FunctionalTensor / traceable-wrapper subclass is
    # never EXACTLY torch.Tensor, so for a plain tensor (the hot path) we skip is_fake
    # entirely -- the only trace case left for an exact-type tensor is a meta tensor,
    # caught by the cheap device read (~0.24us total vs ~0.80us).
    if type(t) is torch.Tensor:
        return t.device.type == "meta"
    return torch._subclasses.fake_tensor.is_fake(t) or t.device.type == "meta"


# Per-device "is this an arch we target?" cache. get_device_capability queries
# device properties (~1.4us) and the answer is IMMUTABLE per device, but a cond runs
# on every eager call -- so memoize by device index. Absent = HIP / not-yet-seen.
_ARCH_OK: dict[int, bool] = {}


def device_ok(x: torch.Tensor) -> bool:
    # CUDA, not HIP, and a compute capability our kernels target (Hopper/Blackwell).
    if x.device.type != "cuda" or torch.version.hip is not None:
        return False
    idx = x.device.index
    ok = _ARCH_OK.get(idx)
    if ok is None:
        major, _ = torch.cuda.get_device_capability(x.device)
        ok = major in (9, 10)  # Hopper / Blackwell -- the archs the kernels target
        _ARCH_OK[idx] = ok
    return ok


def on_current_device(x: torch.Tensor) -> bool:
    # The compiled-kernel and stream caches are keyed/bound to the CURRENT CUDA
    # device; launching a cuda:0-compiled kernel on a cuda:1 tensor raises
    # cudaErrorInvalidResourceHandle. Until the caches are per-device, decline when
    # the input is not on the current device and let aten handle it.
    return x.device.index == torch.cuda.current_device()
