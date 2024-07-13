# mypy: allow-untyped-defs
import contextlib
import tempfile

import torch
from . import check_error, cudart

__all__ = ["init", "start", "stop", "profile"]

DEFAULT_FLAGS = [
    "gpustarttimestamp",
    "gpuendtimestamp",
    "gridsize3d",
    "threadblocksize",
    "streamid",
    "enableonstart 0",
    "conckerneltrace",
]


def init(output_file, flags=None, output_mode="key_value"):
    rt = cudart()
    if not hasattr(rt, "cudaOutputMode"):
        raise AssertionError("HIP does not support profiler initialization!")
    if (
        hasattr(torch.version, "cuda")
        and torch.version.cuda is not None
        and int(torch.version.cuda.split(".")[0]) >= 12
    ):
        # Check https://github.com/pytorch/pytorch/pull/91118
        # cudaProfilerInitialize is no longer needed after CUDA 12
        raise AssertionError("CUDA12+ does not need profiler initialization!")
    flags = DEFAULT_FLAGS if flags is None else flags
    if output_mode == "key_value":
        output_mode_enum = rt.cudaOutputMode.KeyValuePair
    elif output_mode == "csv":
        output_mode_enum = rt.cudaOutputMode.CSV
    else:
        raise RuntimeError(
            "supported CUDA profiler output modes are: key_value and csv"
        )
    with tempfile.NamedTemporaryFile(delete=True) as f:
        f.write(b"\n".join(f.encode("ascii") for f in flags))
        f.flush()
        check_error(rt.cudaProfilerInitialize(f.name, output_file, output_mode_enum))


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
