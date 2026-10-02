from . import _extension
from .api import CheckpointException
from .default_planner import DefaultLoadPlanner, DefaultSavePlanner
from .filesystem import FileSystemReader, FileSystemWriter
from .hf_storage import HuggingFaceStorageReader, HuggingFaceStorageWriter
from .metadata import (
    BytesStorageMetadata,
    ChunkStorageMetadata,
    Metadata,
    TensorStorageMetadata,
)
from .optimizer import load_sharded_optimizer_state_dict
from .planner import LoadPlan, LoadPlanner, ReadItem, SavePlan, SavePlanner, WriteItem
from .protocol import CheckpointableTensor
from .quantized_hf_storage import QuantizedHuggingFaceStorageReader

# pyrefly: ignore [deprecated]
from .state_dict_loader import load, load_state_dict

# pyrefly: ignore [deprecated]
from .state_dict_saver import async_save, save, save_state_dict
from .storage import StorageReader, StorageWriter


def __getattr__(name: str):
    # Deferred so that importing torch.distributed.checkpoint does not require fsspec.
    if name in ("FsspecReader", "FsspecWriter"):
        from . import fsspec_filesystem

        return getattr(fsspec_filesystem, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
