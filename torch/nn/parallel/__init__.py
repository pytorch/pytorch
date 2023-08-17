from .parallel_apply import parallel_apply
from .replicate import replicate
from .data_parallel import DataParallel, data_parallel
from .scatter_gather import gather, scatter
from .distributed import DistributedDataParallel

__all__ = ['replicate', 'scatter', 'parallel_apply', 'gather', 'data_parallel',
           'DataParallel', 'DistributedDataParallel']

def DistributedDataParallelCPU(*args, **kwargs):
    import warnings
    warnings.warn("torch.nn.parallel.DistributedDataParallelCPU is deprecated, "
                  "please use torch.nn.parallel.DistributedDataParallel instead.", stacklevel=2)
    return DistributedDataParallel(*args, **kwargs)
