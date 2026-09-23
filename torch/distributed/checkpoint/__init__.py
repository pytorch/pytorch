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
from . import state_dict as _state_dict
from ._grad_dtype import _init_optim_state, _patch_consolidate_hf_safetensors
from ._metadata_warning import install as _install_metadata_warning

_state_dict._init_optim_state = _init_optim_state
_patch_consolidate_hf_safetensors()
_install_metadata_warning()
