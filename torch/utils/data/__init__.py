# TODO(VitalyFedyunin): Rearranging this imports leads to crash,
# need to cleanup dependencies and fix it
from torch.utils.data.sampler import (
    BatchSampler,
    RandomSampler,
    Sampler,
    SequentialSampler,
    SubsetRandomSampler,
    WeightedRandomSampler,
)
from torch.utils.data.dataset import (
    ChainDataset,
    ConcatDataset,
    Dataset,
    IterableDataset,
    Subset,
    TensorDataset,
    random_split,
)
from torch.utils.data.datapipes.datapipe import (
    DFIterDataPipe,
    DataChunk,
    IterDataPipe,
    MapDataPipe,
)
from torch.utils.data.dataloader import (
    DataLoader,
    _DatasetKind,
    get_worker_info,
)
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.datapipes._decorator import (
    argument_validation,
    functional_datapipe,
    guaranteed_datapipes_determinism,
    non_deterministic,
    runtime_validation,
    runtime_validation_disabled,
)
from torch.utils.data._utils.collate import (
    default_collate,
    default_convert,
    default_collate_fn_map,
    register_default_collate_for,
)


__all__ = ['BatchSampler',
           'ChainDataset',
           'ConcatDataset',
           'DFIterDataPipe',
           'DataChunk',
           'DataLoader',
           'Dataset',
           'DistributedSampler',
           'IterDataPipe',
           'IterableDataset',
           'MapDataPipe',
           'RandomSampler',
           'Sampler',
           'SequentialSampler',
           'Subset',
           'SubsetRandomSampler',
           'TensorDataset',
           'WeightedRandomSampler',
           '_DatasetKind',
           'argument_validation',
           'default_collate',
           'default_convert',
           'default_collate_fn_map',
           'register_default_collate_for',
           'functional_datapipe',
           'get_worker_info',
           'guaranteed_datapipes_determinism',
           'non_deterministic',
           'random_split',
           'runtime_validation',
           'runtime_validation_disabled']

# Please keep this list sorted
assert __all__ == sorted(__all__)
