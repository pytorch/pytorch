from .parallel_apply import parallel_apply as parallel_apply
from .replicate import replicate as replicate
from .data_parallel import DataParallel as DataParallel, data_parallel as data_parallel
from .scatter_gather import gather as gather, scatter as scatter
from .distributed import DistributedDataParallel as DistributedDataParallel

__all__ = ['replicate', 'scatter', 'parallel_apply', 'gather', 'data_parallel',
           'DataParallel', 'DistributedDataParallel']

def DistributedDataParallelCPU(*args, **kwargs):
    import warnings
    warnings.warn("torch.nn.parallel.DistributedDataParallelCPU is deprecated, "
                  "please use torch.nn.parallel.DistributedDataParallel instead.")
    return DistributedDataParallel(*args, **kwargs)
