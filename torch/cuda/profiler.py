# mypy: allow-untyped-defs
import contextlib

from . import check_error, cudart


__all__ = ["start", "stop", "profile"]

DEFAULT_FLAGS = [
    "gpustarttimestamp",
    "gpuendtimestamp",
    "gridsize3d",
    "threadblocksize",
    "streamid",
    "enableonstart 0",
    "conckerneltrace",
]


def start():
    r"""Starts cuda profiler data collection.

    .. warning::
        Raises CudaError in case of it is unable to start the profiler.
    """
    check_error(cudart().cudaProfilerStart())


def stop():
    r"""Stops cuda profiler data collection.

    .. warning::
        Raises CudaError in case of it is unable to stop the profiler.
    """
    check_error(cudart().cudaProfilerStop())


@contextlib.contextmanager
def profile():
    """
    Enable profiling.

    Context Manager to enabling profile collection by the active profiling tool from CUDA backend.
    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> import torch
        >>> model = torch.nn.Linear(20, 30).cuda()
        >>> inputs = torch.randn(128, 20).cuda()
        >>> with torch.cuda.profiler.profile() as prof:
        ...     model(inputs)
    """
    try:
        start()
        yield
    finally:
        stop()
