"""Keep a copy of every cubin the CUDA driver loads.

Serializing a captured CUDA graph needs the device code its kernel nodes point at,
and there is no way to get it back out of the driver: the whole
``cuModule``/``cuLibrary``/``cuFunc``/``cuKernel`` query surface can report a
function's name, module and attributes but never its binary image. Recovering a
kernel by name from the fatbins in the host ``.so`` files covers most of ATen, but
not kernels that are generated at runtime -- cuBLASLt's ``nvjet_*`` kernels on
Blackwell exist in no fatbin on disk and are not written to ``CUDA_CACHE_PATH``.

Capture the images on the way in instead. CUPTI's ``RESOURCE`` /
``MODULE_LOADED`` callback hands over a pointer to the cubin the driver was given,
so copying it there covers every producer -- ATen, Triton, flash attention,
cuBLASLt's JIT -- with no dependence on where the code came from.

Two properties this relies on. Modules load lazily throughout a process, so the
callback has to be armed *before* the work whose kernels you intend to serialize;
arming late silently misses whatever already loaded. And the copy has to happen
inside the callback, because CUPTI only guarantees the pointer for the duration of
the call.

The lifecycle that follows from that::

    start()  # before any CUDA work
    ...  # capture every graph you intend to save
    stop()  # disarm; releases the CUPTI subscription, keeps the images
    graph.save(...)  # still works: saving reads the retained images
    clear()  # return the host memory

``stop()`` and :func:`clear` are separate for exactly that reason. Disarming early
is safe in the sense that it fails loudly -- a graph captured afterwards refuses to
save, naming the kernels it could not find -- but note ``start()`` after a
``stop()`` cannot recover what loaded in between, so capture cannot be cycled
around the interesting parts. It is once, early, until the last save.

Costs, measured on GB200 over a workload loading 31 modules / 203 MB of cubin:
dispatch alone is ~0.5 ms in total, copying the images adds ~18 ms. Images are
kept keyed by CUPTI module id and are not deduplicated -- in that same workload
every cubin was unique, so hashing would have been ~140 ms of pure cost.
Deduplication belongs where archives are written, across processes.
"""

from __future__ import annotations

from logging import getLogger
from typing import Any


logger = getLogger(__name__)


# Handler token from the monitor, carrying the (domain, cbid) it was registered
# for; kept so stop() can arm/disarm and unregister by the same key.
_handler: Any = None

# CUPTI module id -> cubin image. Insert-only while armed.
_modules: dict[int, bytes] = {}

# Modules announced with no usable image. Counted rather than warned about on the
# spot: this runs inside CUPTI's C dispatch, where warnings.warn could be configured
# to raise. Survives stop() so save can name it as a cause; cleared with the images.
_skipped_modules: int = 0


def _on_module_loaded(_domain: int, _cbid: int, cbdata: int) -> None:
    """Copy a freshly loaded module's cubin.

    Runs synchronously on the loading thread inside the CUDA call. ``cbdata`` is a
    raw ``CUpti_ResourceData*``; :func:`read_module_resource` unpacks the image
    from it and copies it, which has to happen here because CUPTI only guarantees
    the pointer for the duration of the callback.
    """
    global _skipped_modules
    from torch.profiler._cupti.cupti_python import read_module_resource

    module = read_module_resource(cbdata)
    if module is None:
        _skipped_modules += 1
        return
    module_id, image = module
    _modules[module_id] = image


def is_available() -> bool:
    """True when capture can be turned on right now: cupti-python is importable
    and CUPTI is usable. Does not create the monitor."""
    try:
        from cupti import (  # noqa: F401  # pyrefly: ignore[missing-import]
            cupti as _cupti,
        )

        from torch.profiler._cupti.monitor import CuptiMonitor  # noqa: F401
    except ImportError:
        return False
    return True


def start() -> bool:
    """Begin copying loaded modules' cubins, and return whether capture is on.

    Idempotent. Takes (or joins) the monitor's shared CUPTI subscription, so it
    composes with the other subscriber-callback consumers such as
    :mod:`torch.cuda._graph_node_callbacks`.

    Call this before the work whose kernels you intend to serialize: modules load
    lazily, and anything already loaded is not re-announced.
    """
    global _handler
    if _handler is not None:
        return True
    if not is_available():
        return False

    from cupti import cupti as _cupti  # pyrefly: ignore[missing-import]

    from torch.profiler._cupti.monitor import CuptiMonitor

    domain = int(_cupti.CallbackDomain.RESOURCE)
    cbid = int(_cupti.CallbackIdResource.MODULE_LOADED)
    try:
        monitor = CuptiMonitor()
        _handler = monitor.register_callback_handler(domain, cbid, _on_module_loaded)
        monitor.arm_callback(domain, cbid)
    except Exception:
        logger.warning("could not arm CUPTI module capture", exc_info=True)
        if _handler is not None:
            stop()
        return False
    return True


def stop() -> None:
    """Stop copying cubins, releasing the monitor's subscription if nothing else
    holds it. Idempotent. Captured images are kept; use :func:`clear` to drop them."""
    global _handler, _skipped_modules
    if _handler is None:
        return
    from torch.profiler._cupti.monitor import CuptiMonitor

    handler, _handler = _handler, None
    monitor = CuptiMonitor()
    monitor.disarm_callback(handler.domain, handler.cbid)
    monitor.unregister_callback_handler(handler)
    if _skipped_modules:
        logger.warning(
            "skipped %d module load(s) announced without a usable cubin image; "
            "kernels from those modules will not be serializable",
            _skipped_modules,
        )


def is_started() -> bool:
    return _handler is not None


def captured_modules() -> dict[int, bytes]:
    """The cubins captured so far, keyed by CUPTI module id."""
    return dict(_modules)


def clear() -> None:
    """Drop the captured images."""
    global _skipped_modules
    _modules.clear()
    _skipped_modules = 0


def skipped_modules() -> int:
    """How many module loads were announced without a usable ELF cubin."""
    return _skipped_modules
