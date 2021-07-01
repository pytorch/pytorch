from torch.utils.data.sampler import \
    (Sampler, SequentialSampler, RandomSampler,
     SubsetRandomSampler, WeightedRandomSampler, BatchSampler)
from torch.utils.data.dataset import \
    (Dataset, IterableDataset, TensorDataset, ConcatDataset, ChainDataset,
     Subset, random_split, Dataset as MapDataPipe, IterableDataset as IterDataPipe)
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import DataLoader, _DatasetKind, get_worker_info
from torch.utils.data._decorator import \
    (functional_datapipe, guaranteed_datapipes_determinism, non_deterministic,
     argument_validation, runtime_validation_disabled, runtime_validation)


Sampler.__module__ = "torch.utils.data"
SequentialSampler.__module__ = "torch.utils.data"
RandomSampler.__module__ = "torch.utils.data"
SubsetRandomSampler.__module__ = "torch.utils.data"
WeightedRandomSampler.__module__ = "torch.utils.data"
BatchSampler.__module__ = "torch.utils.data"

Dataset.__module__ = "torch.utils.data"
IterableDataset.__module__ = "torch.utils.data"
TensorDataset.__module__ = "torch.utils.data"
ConcatDataset.__module__ = "torch.utils.data"
ChainDataset.__module__ = "torch.utils.data"
Subset.__module__ = "torch.utils.data"
random_split.__module__ = "torch.utils.data"
MapDataPipe.__module__ = "torch.utils.data"
IterDataPipe.__module__ = "torch.utils.data"

DistributedSampler.__module__ = "torch.utils.data"

DataLoader.__module__ = "torch.utils.data"
_DatasetKind.__module__ = "torch.utils.data"
get_worker_info.__module__ = "torch.utils.data"

functional_datapipe.__module__ = "torch.utils.data"
guaranteed_datapipes_determinism.__module__ = "torch.utils.data"
non_deterministic.__module__ = "torch.utils.data"
argument_validation.__module__ = "torch.utils.data"
runtime_validation_disabled.__module__ = "torch.utils.data"
runtime_validation.__module__ = "torch.utils.data"


__all__ = ['Sampler', 'SequentialSampler', 'RandomSampler',
           'SubsetRandomSampler', 'WeightedRandomSampler', 'BatchSampler',
           'DistributedSampler', 'Dataset', 'IterableDataset', 'TensorDataset',
           'ConcatDataset', 'ChainDataset', 'Subset', 'random_split',
           'DataLoader', '_DatasetKind', 'get_worker_info',
           'IterDataPipe', 'MapDataPipe', 'functional_datapipe',
           'guaranteed_datapipes_determinism', 'non_deterministic',
           'argument_validation', 'runtime_validation_disabled',
           'runtime_validation']


################################################################################
# import subpackage
################################################################################
from torch.utils.data import datapipes
