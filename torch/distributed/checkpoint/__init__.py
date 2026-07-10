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
from .quantized_hf_storage import QuantizedHuggingFaceStorageReader

# pyrefly: ignore [deprecated]
from .state_dict_loader import load, load_state_dict

# pyrefly: ignore [deprecated]
from .state_dict_saver import async_save, save, save_state_dict
from .storage import StorageReader, StorageWriter

try:
    from ._fsspec_filesystem import FsspecReader, FsspecWriter
except ModuleNotFoundError as exc:
    if exc.name != "fsspec":
        raise

    _fsspec_import_error = exc

    class FsspecReader:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ModuleNotFoundError(
                "FsspecReader requires fsspec. Install it with `pip install fsspec`."
            ) from _fsspec_import_error

    class FsspecWriter:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ModuleNotFoundError(
                "FsspecWriter requires fsspec. Install it with `pip install fsspec`."
            ) from _fsspec_import_error
