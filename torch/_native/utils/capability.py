# Per-tensor capability gating shared by every CuteDSL native-op override family. These run on EVERY
# eager dispatch, so they must be cheap and must NEVER raise -- a throwing cond crashes the dispatcher
# instead of falling back to aten. Op-agnostic: only "can our kernels run on this tensor at all",
# with each family layering its own dtype and geometry checks on top.

from __future__ import annotations

import functools

import torch
import torch._subclasses.fake_tensor


def is_traced(t: torch.Tensor) -> bool:
    # A fake or meta tensor has no storage to launch on, so decline and let the compile router use
    # aten's reference. The exact-type fast path avoids is_fake()'s subclass walk (0.52us against
    # 1.01us), but exact-type alone does NOT mean "not traced": the two C++-level wrappers --
    # functionalization and a C++ fake -- ARE exactly torch.Tensor, so both bits must be read too.
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
    # The compiled-kernel and stream caches are bound to the CURRENT device, and launching a
    # cuda:0-compiled kernel on a cuda:1 tensor raises cudaErrorInvalidResourceHandle, so decline
    # until they are per-device. The non-CUDA check must come first because current_device() raises
    # without CUDA -- and a FAKE cuda tensor reaches here on a CPU-only box, so guard the query too.
    if x.device.type != "cuda":
        return False
    try:
        return x.device.index == torch.cuda.current_device()
    except RuntimeError:
        return False
