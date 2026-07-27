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
