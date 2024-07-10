# mypy: allow-untyped-defs
from typing_extensions import deprecated

from .data_parallel import data_parallel, DataParallel
from .distributed import DistributedDataParallel
from .parallel_apply import parallel_apply
from .replicate import replicate
from .scatter_gather import gather, scatter


__all__ = [
    "replicate",
    "scatter",
    "parallel_apply",
    "gather",
    "data_parallel",
    "DataParallel",
    "DistributedDataParallel",
]


@deprecated(
    "`torch.nn.parallel.DistributedDataParallelCPU` is deprecated, "
    "please use `torch.nn.parallel.DistributedDataParallel` instead.",
    category=FutureWarning,
)
class DistributedDataParallelCPU(DistributedDataParallel):
    pass
