import abc

import io
import json
import os
from concurrent.futures import Future
from dataclasses import asdict, dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from contextlib import contextmanager
from typing import (
    Any,
    BinaryIO,
    Callable,
    Optional,
    TypeAlias,
    Union,
)

import torch
import torch.multiprocessing as mp

from torch.distributed.checkpoint._v2._metadata import Metadata, Param

from torch.types import FileLike, Storage

_MAP_LOCATION: TypeAlias = Optional[
    Union[Callable[[Storage, str], Storage], torch.device, str, dict[str, str]]
]
_STORAGE: TypeAlias = Union[Storage, torch.storage.TypedStorage, torch.UntypedStorage]


@dataclass
class CheckpointingConfig:
    save_manifest_with_checkpoint: bool = True
    filter_replicated_tensors_on_save: bool = False
    use_barrier_for_save_completion: bool = True
    barrier_timeout_on_save: int = 3600


@dataclass
class RankInfo:
    global_rank: int
    global_world_size: int
    role_rank: int
    role_world_size: int
    role_index: int
    role_replica_count: int  # to support PAFT with HSDP


@dataclass
class CheckpointContext:
    step: int
    extra_context: dict[str, Any]


class Checkpointer(abc.ABC):
    """
    This is a checkpointing solution to store and load models with parallelism at scale.

    Note: If you working with single rank models and do not need asynchronous checkpointing, we recommend
    using `torch.save` and `torch.load` for its simplicity.

    This class provides extension points for users to customize individual components as they
    need.

    .. warning::
        This feature is experimental and subject to removal/change.

    """

    @abc.abstractmethod
    def save(
        self,
        state_dict: dict[str, Any],
        context: CheckpointContext,
        root_dir: str,
        use_cached_manifest: bool = False,
    ) -> Optional[tuple[Future[None], Future[None]]]:
        """
        Save a checkpoint to a given path on storage. This optionally saves the metadata aggregated
        across all ranks to make resharding on load efficient and generic.

        Using module names as top level keys is optional but recommended. See the below example for
        typical usage with a state dict containing module names as keys and module state dicts as
        values.
            {
                "model": model_state_dict,
                "optimizer": optimizer_state_dict,
                "dataloader": dataloader_state_dict,
                "metrics": metrics_state_dict,
                "extra_state": extra_state_dict,
            }

        We expect most users to use a variant of async checkpointing implementation and so the API is
        designed to be easy to use for this case. A synchronous checkpointing implementation
        can still be implemented by on top of this API. EG: synchronous version can return a tuple of
        two futures which are set with a result in the save() method.

        Args:
            state_dict (dict[str, Any]): The state_dict to save.
            context (CheckpointContext): The context to save the checkpoint.
            root_dir (str): The path to save the checkpoint.
            use_cached_manifest (bool): Whether to use cached manifest. (Default: ``False``). If
                use_cached_manifest is True, it will use the compute the manifest to save the checkpoint.
                Otherwise, it will compute the manifest and save the checkpoint. If the checkpoint is saved
                successfully, it will return a tuple of two futures. The first future is a future for the
                manifest and the second future is a future for the checkpoint.

        Returns:
            tuple[Future, Future]: A tuple of two futures. The first future can be awaited on for completion
            of staging (D2H) the state_dict and the second for completion of checkpoint.
        """
        pass

    @abc.abstractmethod
    def load_manifest(self, path: str) -> dict[str, Any]:
        pass

    @abc.abstractmethod
    def close(self):
        pass


class CheckpointLoader(abc.ABC):
    """
    This is a checkpointing solution to load models with parallelism at scale.

    Note: If you working with single rank models and do not need asynchronous checkpointing, we recommend
    using `torch.save` and `torch.load` for its simplicity.

    This class provides extension points for users to customize individual components as they
    need.

    .. warning::
        This feature is experimental and subject to removal/change.

    """

    @abc.abstractmethod
    def load(
        self,
        path: str,
        fqns_to_load: Union[str, list[str], None],
        map_location: _MAP_LOCATION,
    ) -> dict[str, Any]:
        """
        Loads a checkpoint from a given path for the current rank.

        if fqns_to_load is None, this will load the entire state_dict written from this rank.
        if fqns_to_load is list or a str, this will load params for the fqns specified in the
        list and returns a dict of {fqn: obj}. If the fqn is not found in the checkpoint,
        an exception will be raised. (TODO: should we just skip as the user can do it themselves?)

        the checkpoint loader will load params to given map_location.
        """
        pass

    @abc.abstractmethod
    def load_in_place(
        self,
        path: str,
        storage_location: dict[str, Any],
        reshard_if_needed: bool = True,
    ) -> dict[str, Any]:
        """
        Loads a checkpoint from a given path for the current rank.

        if fqns_to_load is None, this will load the entire state_dict written from this rank.
        if fqns_to_load is list or a str, this will load params for the fqns specified in the
        list and returns a dict of {fqn: obj}. If the fqn is not found in the checkpoint,
        an exception will be raised. (TODO: should we just skip as the user can do it themselves?)

        the checkpoint loader will load params to given map_location.
        """
        pass


# A base class for storage backends
class ModelStore(abc.ABC):
    """
    Acts as an adaptor for storage backends and is also responsible for parallelizing
    writes/reads to storage backend if needed.

    Naming is a throwback to our beloved OG checkpointing solution.
    """

    @abc.abstractmethod
    def ls(self, path: str) -> list[Path]:
        pass

    @abc.abstractmethod
    def mkdir(self, path: str, recursive: bool, exists_ok: bool):
        pass

    @abc.abstractmethod
    def rmdir(self, path: str):
        pass

    @abc.abstractmethod
    @contextmanager
    def open(self, path: str) -> BinaryIO:
        pass

    @abc.abstractmethod
    def delete_obj(self, path: str):
        pass


class CheckpointLayout(abc.ABC):
    """
    This class is responsible for deciding the layout of the checkpoint on storage.

    TODO: The caller needs to ensure checkpoint loader is setup with options that are
    in sync with the options used to save the checkpoint. we can write metadata about
    the options we used to save and validate if needed? In comparison to DCP, This is
    similar to writing checkpoint with one storage writer and loading with another that
    is incompatible. so may be just document this as a requirement.
    """

    @abc.abstractmethod
    def get_metadata_path(
        self, config: CheckpointingConfig, rank_info: RankInfo, context: Any
    ) -> str:
        return "metadata.json"

    @abc.abstractmethod
    def get_file_mappings_for_write(
        self, rank: int, state_dict: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Examples usecases:

            1. To save state_dict from one rank in one file and all ranks in a single directory.
                `return {f"checkpoint_{rank}.pt": state_dict}`
            2. To save state_dict from one rank in one file and but use a separate directory
                for every 1000 ranks.
                `return {f"{rank/1000}/checkpoint_{rank}.pt": state_dict}`
            2. To save each module in a separate file.
                `
                return {
                    "model.pt": state_dict["model"],
                    "optimizer.pt": state_dict["optimizer"],
                    "dataloader.pt": state_dict["dataloader"],
                }
                `
            3. To save each param in a separate file.
                `
                return {
                    "model.weights.param1.pt": state_dict["model"]["weights"]["param1"],
                    "model.weights.param2.pt": state_dict["model"]["weights"]["param2"],
                }
                `
        """
        pass

    @abc.abstractmethod
    def get_all_file_mappings_to_read(self, rank: int) -> list[str]:
        """
        Return all files to be read for this rank.
        """
        # return f"checkpoint_{rank_info.global_rank}.pt"
        pass

    @abc.abstractmethod
    def get_file_mappings_to_read(
        self, rank_info: int, fqns_to_load: list[str]
    ) -> dict[str, list[str]]:
        """ """
        pass


class SerializationFormat(abc.ABC):
    # Can this be a callable? do we need an interface for this?
    @abc.abstractmethod
    def serialize(self, obj: object, f: FileLike) -> None:
        torch.save(obj, f)

    @abc.abstractmethod
    def deserialize(self, f: FileLike) -> object:
        pass


class TorchSerializationFormat(abc.ABC):
    @abc.abstractmethod
    def serialize(self, obj: object, f: FileLike) -> None:
        torch.save(obj, f)

    @abc.abstractmethod
    def deserialize(self, f: FileLike) -> object:
        return torch.load(f)


class Barrier(abc.ABC):
    def __init__(self, world_size: int, timeout: int):
        self.world_size = world_size

    def wait(self, timeout):
        pass


class CheckpointWriter:
    def __init__(
        self,
        config: Any,
        rank_info: RankInfo,
        storage: ModelStore,
        checkpoint_layout: CheckpointLayout,
        serialization_format: SerializationFormat,
        barrier: Barrier,
    ):
        """
        Writes the state_dict to storage.

        Args:
            config (Any): The config to use for the checkpoint.
            rank_info (RankInfo): The rank info to use for the checkpoint.
            storage (Storage): The storage to use for the checkpoint.
            checkpoint_layout (CheckpointLayout): The layout to use for the checkpoint.
            serialization_format (SerializationFormat): The serialization format to use for the checkpoint.
        """

        self._config = config
        self._rank_info = rank_info
        self._storage = storage
        self._layout = checkpoint_layout
        self._serialization_format = serialization_format
        self._barrier = barrier

    def write_checkpoint(
        self,
        state_dict: dict[str, Any],
        metadata: Metadata,
        context: Any,
        root_dir: str,
    ) -> None:
        """
        Writes the state_dict to storage.

        Args:
            state_dict (dict[str, Any]): The state_dict to write.
            manifest (Optional[Manifest]): The manifest to write.
            context (Any): The context to write.
            path (str): The path to write the checkpoint to.

        Returns:
            str: The path to the checkpoint.
        """
        # naive example for now
        metadata_path = self._layout.get_metadata_path(
            self._config, self._rank_info, context
        )
        if self._config.save_manifest_with_checkpoint:
            with self._storage.open(os.path.join(root_dir, metadata_path)) as f:
                f.write(json.dumps(asdict(metadata)))

        file_paths = self._layout.get_file_mappings_for_write(
            self._rank_info.global_rank, state_dict
        )

        for file_path, obj in file_paths.items():
            with self._storage.open(os.path.join(root_dir, file_path)) as f:
                self._serialization_format.serialize(obj, f)


        if self._config.use_barrier_for_save_completion:
            self._barrier.wait(self._config.barrier_timeout)


MANIFEST: TypeAlias = dict[str, list[Param]]


class ManifestBuilder(abc.ABC):
    """
    This class is responsible for building the manifest for a checkpoint.

    TODO alernate plan:
    We can also avoid this class by just providing a utility function to build the manifest from
    ShardedTensor and DTensor. If the user has a state_dict with custom impls of parallelisms, they
    can write their own logic. User can call the function before saving the checkpoint and pass the
    manifest to the checkpointing API. 

    TODO: is this a problem? Allowing the user to build the manifest or override the builder is
    flexible but has downside of not being able to ensure that manifest is built as expected which
    might affect resharding.

    TODO alernate plan:
    The alternate plan is to do the collective on load_with_resharding where each rank
    reads its own manifest (or from data.pkl in file saved with torch.save) in file and
    does the collective to build the global manifest. This is a bit more expensive if
    load_with_resharding is used by default.
    """

    def __init__(self, config: CheckpointingConfig, rank_info: RankInfo):
        self._config = config
        self._rank_info = rank_info
        self._manifest: MANIFEST = {}

    def buid_manifest(
        self,
        state_dict: dict[str, Any],
        context: CheckpointContext,
    ) -> MANIFEST:
        assert torch.distributed.is_initialized()
        # TODO

        # prepare global manifest
        # do torch.distributed.all_reduce()
        return {}
