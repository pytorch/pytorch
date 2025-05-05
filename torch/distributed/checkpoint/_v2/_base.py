import abc
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable, Optional, TypeAlias, Union

import torch

from torch.distributed.checkpoint._v2._metadata import Metadata, Param
from torch.types import FileLike, Storage


# TODO see if we can do this in types.py
_MAP_LOCATION: TypeAlias = Optional[
    Union[Callable[[Storage, str], Storage], torch.device, str, dict[str, str]]
]

"""
    This module provides a checkpointing solution to store and loads models with parallelism 
    at scale. 
    
    On load, resharding is done natively for most common changes in paralellism when using 
    ShardedTensors and DTensors. For more specifics on supported resharding usecases and 
    implications, see documentation for :class:`torch.distributed.checkpoint.CheckpointLoader`.

    The implementation focuses on efficiency and scale (up to 100Ks of GPUs) and supports 
    asynchronous checkpointing with zero-overhead checkpointing. See 
    :class:`torch.distributed.checkpoint.AsyncCheckpointer` for more details.

    Note: If you working with single rank models and do not need asynchronous checkpointing, we 
    recommend using `torch.save` and `torch.load` for its simplicity.

    We provide extension points for users to customize individual components as they need but 
    provide a common implementations for each component.

.. warning::
    This feature is experimental and subject to removal/change.
"""


@dataclass
class CheckpointingConfig:
    """
    Some configs for example. We would need to add more.
    TBD: Add apis for arsing or preparing configs.
    """

    save_manifest_with_checkpoint: bool = True
    filter_replicated_tensors_on_save: bool = False
    use_barrier_for_save_completion: bool = True
    barrier_timeout_on_save: int = 3600
    init_buffer_in_thread: bool = True
    stage_in_thread: bool = True
    extra_config: dict[str, Any] = {}
    async_checkpointing: bool = True


@dataclass
class RankInfo:
    """
    These attrs are defined in torchrun.

    When using ShardingStrategy.HYBRID_SHARD, depeending on the setup we may need role rank
    info.
    """

    global_rank: int
    global_world_size: int
    role_rank: int
    role_world_size: int
    role_index: int
    role_replica_count: int


@dataclass
class CheckpointContext:
    """
    Context around the checkpoint.  This is passed to the components so they can
    manipulate file paths/names, select configurations accordingly. This is also
    super useful for logging and debugging.

    TBD: May be model identifier, job name, etc could be added as first class attrs.
    """

    step: int
    extra_context: dict[str, Any]


# A base class for storage backends
class ModelStore(abc.ABC):
    """
    Acts as an adaptor for storage backends and is also responsible for parallelizing
    writes/reads to storage backend as needed.

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
    def open(self, path: str) -> BinaryIO:
        pass

    @abc.abstractmethod
    def delete_obj(self, path: str):
        pass


class SerializationFormat(abc.ABC):
    @abc.abstractmethod
    def serialize(self, obj: object, f: FileLike) -> None:
        pass

    @abc.abstractmethod
    def deserialize(self, f: FileLike) -> object:
        pass


class Barrier(abc.ABC):
    """
    A barrier to synchronize ranks after checkpointing is complete.
    """

    @abc.abstractmethod
    def wait(self, timeout: int) -> None:
        pass


class CheckpointWriterBase(abc.ABC):
    """
    Writes the state_dict to storage.
    """

    @abc.abstractmethod
    def write(
        self,
        state_dict: Union[Future[dict[str, Any]], dict[str, Any]],
        metadata: Metadata,
        context: Any,
        root_dir: str,
    ) -> Optional[Future[None]]:
        pass


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


class CheckpointerBase(abc.ABC):
    """
        This is a base class to save checkpoints in a training job. Typical usage is to
        initiate this at the begining of the training job and call save() as needed during
        the training and close() at the end. This allows implmentations (EG: AsyncCheckpointer)
        to create threads/processes, allocate additional memory and clean them up at the end of
        training.

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
                Otherwise, it will compute the manifest and save the checkpoint.

        Returns:
            None: when using synchronous checkpointing.
            tuple[Future, Future]: A tuple of two futures when using async checkpointing. The first
            future can be awaited on for completion of staging (D2H) the state_dict and the second
            for completion of checkpoint.
        """
        pass

    @abc.abstractmethod
    def close(self):
        pass


class CheckpointLoaderBase(abc.ABC):
    """
    This is a checkpointing solution to load models with parallelism at scale.

    Note: If you working with single rank models and do not need asynchronous checkpointing, we
    recommend using `torch.save` and `torch.load` for its simplicity.

    This class provides extension points for users to customize individual components as they
    need.

    .. warning::
        This feature is experimental and subject to removal/change.
    """

    @abc.abstractmethod
    def load(
        self,
        path: str,
        *,
        storage_location: Optional[dict[str, Any]] = None,
        load_only_storage_location_keys: bool = False,
        default_map_location: _MAP_LOCATION = None,
        reshard_if_needed: bool = True,
    ) -> dict[str, Any]:
        """
        Loads a checkpoint from a given path for this rank.

        User can provide a storage_location to load the data into. If
        load_only_storage_location_keys=False, all of the data present from the saved
        state_dict for this rank will be loaded otherwise only the keys present in the
        storage_location will be loaded.

        If storage_location is None or if a corresponding key is not found in storage_location
        or if location of the value for the key is a meta device, then data will be loaded into
        the default_map_location and returned in the dict.

        Resharding is supported only if reshard_if_needed is True and when a target storate layout
        is provided. Caller can pass in a tensor with meta device to allow resharding when the
        target state_dict is not available.

        Example usages :
            1. Load all the data from the checkpoint into the default_map_location and return a dict.
                state_dict = loader.load(path)

            2. Load only specified modules from the checkpoint into the default_map_location and return
            a dict.
                state_dict = loader.load(
                    path,
                    storage_location={"module1":None, "module2":None},
                    load_only_storage_location_keys=True,
                )

            3. Load all the data from the checkpoint but copy storages into storage_location with resharding.
                state_dict = loader.load(
                    path,
                    storage_location=module.state_dict(),
                    load_only_storage_location_keys=False,
                    reshard_if_needed=True,
                )
                module.load_state_dict(state_dict)

            4. Load a specified tensor from the checkpoint in to cpu.
                state_dict = loader.load(
                    path,
                    storage_location={"module1":{"a":torch.ones(5, device='meta')}},
                    load_only_storage_location_keys=False,
                    reshard_if_needed=True,
                )
                t = state_dict["module1"]["a"]

        TODO: It is possible we dont reshard_if_needed, as we can verify the need from metadata
        when available. Check if we can do this efficiently or and for cases when metadata exists
        and does not exist, partial and full loads.
        """
        pass


class TorchSerializationFormat(abc.ABC):
    @abc.abstractmethod
    def serialize(self, obj: object, f: FileLike) -> None:
        torch.save(obj, f)

    @abc.abstractmethod
    def deserialize(self, f: FileLike) -> object:
        return torch.load(f)
