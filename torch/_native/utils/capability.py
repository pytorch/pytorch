# Per-tensor capability gating shared by ALL CuteDSL native-op override families
# (reductions today, pointwise/etc. later). Every override's `cond` runs on EVERY
# eager dispatch, so these must be cheap and must NEVER raise (a throwing cond would
# crash the dispatcher instead of falling back to aten).
#
# These checks are op-agnostic -- they say nothing about reductions, dims, or
# geometry, only "can our kernels run on this tensor's device/storage at all". Op
# families layer their own dtype / geometry / shape checks on top.

from __future__ import annotations

import functools

import torch
import torch._subclasses.fake_tensor


def is_traced(t: torch.Tensor) -> bool:
    # FakeTensor (compile/export tracing) and meta tensors have no real storage to
    # launch a kernel on; the decomposition / shape-inference path should use aten's
    # reference, not our kernel. Decline so the compile router falls back. (Eager on
    # real tensors is unaffected -- that is the path our kernels target.)
    #
    # This cond runs on EVERY eager dispatch and is_fake() is a Python subclass walk, so an
    # exact-type tensor (the hot path) answers from cheap reads instead. Being exact-type is
    # NOT on its own enough to mean "not traced": a Python subclass (FakeTensor,
    # FunctionalTensor, traceable wrapper) is never exactly torch.Tensor, but the two C++-level
    # wrappers ARE -- C++ functionalization and a C++ fake, which is_fake reaches through its
    # own _is_functional_tensor / _is_fake_tensor branches. A functional tensor wrapping a fake
    # one is exact-type, is_fake() True, and was answered False here.
    #
    # MEASURED (200k reps, plain CPU tensor): 0.26us for the device read alone, 0.52us with both
    # wrapper bits, 1.01us for the full is_fake walk -- so the fast path keeps half its value.
    if type(t) is torch.Tensor:
        return (
            t.device.type == "meta"
            or torch._is_functional_tensor(t)
            or torch._C._is_fake_tensor(t)
        )
    return torch._subclasses.fake_tensor.is_fake(t) or t.device.type == "meta"


@functools.cache
def _arch_ok(idx: int, majors: tuple[int, ...]) -> bool:
    # get_device_capability queries device properties and the answer is IMMUTABLE per device,
    # while a cond runs on every eager call -- so memoize. The accepted set is part of the key:
    # two families with different sets must not be served each other's answer.
    try:
        major, _ = torch.cuda.get_device_capability(idx)
    except RuntimeError:
        # No usable device, e.g. tracing a CUDA model on a CPU-only box: a FAKE cuda tensor
        # normalizes to cuda:0 without initializing CUDA, so it reaches here and the query
        # raises. The contract at the top of this file is that a cond never raises -- decline.
        return False
    return major in majors


def device_ok(x: torch.Tensor, majors: tuple[int, ...]) -> bool:
    """CUDA (not HIP), on a compute-capability major the CALLER's kernels support.

    An explicit allow-list, not a minimum. `>=` would silently admit hardware nobody tested:
    SM 11.0 is Thor, which this tree already knows about (``common_cuda.IS_THOR``,
    ``cpp_extension``'s ``'11.0'``/``'11.0a'`` arches, ``sm_110a`` in ``cmake/Codegen.cmake``),
    and admitting it -- or any future major -- is exactly what a family enumerating
    ``(9, 10, 12)`` is refusing to do. The set belongs to the caller because the families
    genuinely differ; passing one here is a claim about which arches that family's kernels have
    been run on.

    Granularity is the MAJOR only, so an accepted major admits every minor within it, including
    ones that do not exist yet -- ``(9, 10, 12)`` accepts Rubin at 10.7
    (``cpp_extension.py``'s ``('Rubin', '10.7+PTX')``). A family that needs to distinguish
    minors has to check ``get_device_capability`` itself; this predicate cannot express it.
    """
    if x.device.type != "cuda" or torch.version.hip is not None:
        return False
    return _arch_ok(x.device.index, tuple(majors))


def on_current_device(x: torch.Tensor) -> bool:
    # The compiled-kernel and stream caches are keyed/bound to the CURRENT CUDA
    # device; launching a cuda:0-compiled kernel on a cuda:1 tensor raises
    # cudaErrorInvalidResourceHandle. Until the caches are per-device, decline when
    # the input is not on the current device and let aten handle it.
    #
    # The non-CUDA check comes first because current_device() RAISES without CUDA, and the
    # contract at the top of this file is that a cond never raises. That is not sufficient on
    # its own: a FAKE cuda tensor has device.type "cuda" in a process with no usable device
    # (tracing a CUDA model on a CPU-only box), so the query has to be guarded too.
    if x.device.type != "cuda":
        return False
    try:
        return x.device.index == torch.cuda.current_device()
    except RuntimeError:
        return False
