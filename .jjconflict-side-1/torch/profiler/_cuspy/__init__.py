"""In-process CUPTI activity collection for torch.profiler.

``core`` owns the GIL-free CUPTI activity buffer collection (backed by the
native ``CuspyBuffers``); ``trace`` builds Chrome traces from it.

Early HES reserves a CUPTI subscriber across profiling windows. Applications must
pair ``core.enable_hes_early()`` with ``core.release_hes()``: enable before CUDA
initialization, configure collector options after parsing arguments, then close
all activity observers and callback handlers and release HES before CUDA/NCCL
teardown. Ordinary observer close deliberately keeps HES for the next window.

Example::

    from torch.profiler._cuspy import core

    core.enable_hes_early()
    try:
        # Parse arguments, configure options, and run profiling windows here.
        # Close every observer and callback handler in their owning scopes.
        ...
    finally:
        core.release_hes()  # before destroying CUDA/NCCL resources
"""
