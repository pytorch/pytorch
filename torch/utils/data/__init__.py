from torch.utils.data.dataloader import (
    _DatasetKind,
    DataLoader,
    default_collate,
    default_convert,
    get_worker_info,
)
from torch.utils.data.dataset import (
    ChainDataset,
    ConcatDataset,
    Dataset,
    IterableDataset,
    random_split,
    StackDataset,
    Subset,
    TensorDataset,
)
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import (
    BatchSampler,
    RandomSampler,
    Sampler,
    SequentialSampler,
    SubsetRandomSampler,
    WeightedRandomSampler,
)


# Lazy imports for datapipes symbols to avoid loading datapipes on every `import torch`
_DATAPIPES_DECORATOR_SYMBOLS = frozenset(
    {
        "argument_validation",
        "functional_datapipe",
        "guaranteed_datapipes_determinism",
        "non_deterministic",
        "runtime_validation",
        "runtime_validation_disabled",
    }
)

_DATAPIPES_DATAPIPE_SYMBOLS = frozenset(
    {
        "DataChunk",
        "DFIterDataPipe",
        "IterDataPipe",
        "MapDataPipe",
    }
)


def __getattr__(name):
    if name in _DATAPIPES_DECORATOR_SYMBOLS:
        from torch.utils.data.datapipes.utils.common import _warn_datapipes_deprecation

        _warn_datapipes_deprecation()
        from torch.utils.data.datapipes import _decorator

        return getattr(_decorator, name)
    if name in _DATAPIPES_DATAPIPE_SYMBOLS:
        from torch.utils.data.datapipes.utils.common import _warn_datapipes_deprecation

        _warn_datapipes_deprecation()
        from torch.utils.data.datapipes import datapipe

        return getattr(datapipe, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BatchSampler",
    "ChainDataset",
    "ConcatDataset",
    "DFIterDataPipe",
    "DataChunk",
    "DataLoader",
    "Dataset",
    "DistributedSampler",
    "IterDataPipe",
    "IterableDataset",
    "MapDataPipe",
    "RandomSampler",
    "Sampler",
    "SequentialSampler",
    "StackDataset",
    "Subset",
    "SubsetRandomSampler",
    "TensorDataset",
    "WeightedRandomSampler",
    "_DatasetKind",
    "argument_validation",
    "default_collate",
    "default_convert",
    "functional_datapipe",
    "get_worker_info",
    "guaranteed_datapipes_determinism",
    "non_deterministic",
    "random_split",
    "runtime_validation",
    "runtime_validation_disabled",
]

# Please keep this list sorted
if __all__ != sorted(__all__):
    raise AssertionError("__all__ is not sorted")
