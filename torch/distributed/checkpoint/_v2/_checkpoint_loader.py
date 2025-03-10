import abc
from typing import (
    BinaryIO,
    Callable,
    Generator,
    Optional,
    Union,
    Dict,
    Any,
    List,
    Generator,
)
import torch

from torch.distributed.checkpoint._v2._base import (
    CheckpointContext,
    CheckpointLoaderBase,
    CheckpointingConfig,
    RankInfo,
    Storage,
    CheckpointLayout,
    SerializationFormat,
    # CheckpointReader,
    _STORAGE,
    _MAP_LOCATION,
    ModelStore,
    get_state_dict_fqns,
)
from torch.serialization import MAP_LOCATION
from torch.types import FileLike


class TorchPartialLoader(abc.ABC):
    """
    TorchStreamingLoader or TorchPartialLoader will be implemented outside of checkpointing
    and almost certainly with different API. This is quick hack to show what we need
    from torch load to support checkpointing usecases: resharding, low memory footprint,
    random access to params, etc.

    The capabilities we need -
        1. Load a specified param without loading the entire model file into memory.
        2. Load specified params to a specified pre-allocated location in chunks or return
            a generator so caller can load the params in chunks.
    """

    def __init__(
        self,
        f: FileLike,
        pickle_module: Any = None,
        *,
        weights_only: Optional[bool] = None,
        mmap: Optional[bool] = None,
        **pickle_load_args: Any,
    ):
        # read the data.pkl file, load objects with dummy storage objects in to cpu and
        # compute the offset of each storage and build a map of {fqn: file, offset, size}
        # for each storage.
        # We could use do this for every file/rank and use the tensor metadata read for
        # resharding on the  fly if  we dont have a global metadata but that is not needed
        # for now.
        self.shallow_state_dict = torch.load(f, map_location="meta", **pickle_load_args)
        self.fqn_to_offset = {}
        pass

    def load_shallow(self) -> dict[str, Any]:
        """
        Load the state_dict from torch.save file but load all storages with meta device.  The
        returned state_dict will have the same structure as the original state_dict.
        """
        return self.shallow_state_dict

    @abc.abstractmethod
    def load_fqn(
        self,
        fqn: str,
        offset_bytes: int = 0,
        length: Optional[int] = None,
        storage_location: Optional[Storage] = None,
        MAP_LOCATION=None,
    ) -> Any:
        # map the fqn to object in shallow_state_dict. Find the corresponding storage key and find position in file.
        # add offset to the position and load the param there to storage/map location.

        # f.seek( self.fqn_to_offset[fqn] + offset_bytes)
        # f.read(length)
        pass


class CheckpointReader:
    def __init__(
        self,
        config: Any,
        rank_info: RankInfo,
        storage: ModelStore,
        checkpoint_layout: CheckpointLayout,
        serialization_format: SerializationFormat,
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

    def read_from_rank(
        self,
        root_dir: str,
        context: Any,
        rank: int,
        fqns_to_load: Union[str, List[str], None],
        map_location: _MAP_LOCATION,
    ) -> Any:

        # convert fqns to file paths

        # load all the file and filter for params (or load the respective params
        # by calculating offset of storages from zip metadata)
        pass


class CheckpointLoader(CheckpointLoaderBase):
    """
    Example for a sync checkpoint loader.
    """

    def __init__(
        self,
        config: CheckpointingConfig,
        rank_info: RankInfo,
        reader: CheckpointReader,
    ):
        self._config = config
        self._rank_info = rank_info
        self._reader = reader

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
        when available. Check if we can handle all cases - when metadata exists and does not
        exist, partial and full loads.

        """
        pass

        # Parts dealing with state_dict and manifest will be in the loader and rest of this
        # logic will be in the reader.

        if not reshard_if_needed:
            # get all fqns to load from storage_location

            fqns_to_load = []
            if storage_location is not None and load_only_storage_location_keys:
                fqns_to_load = get_state_dict_fqns(storage_location)

            if fqns_to_load is None and storage_location is None:
                return torch.load(
                    path,
                    map_location=default_map_location,
                )
            # we could easily load the entire checkpoint into memory and copy as needed.
            # if we really want to to in-place and load directly from storage to target,
            # we can do something like this for each file.
            # result = {}
            # torch_partial_loader = TorchPartialLoader()
            # for fqn in fqns_to_load:

            #     location = None
            #     if fqn in storage_location and storage_location[fqn] is not None and storage_location[fqn].device.type != "meta":
            #         # load the fqn into storage_location[fqn]
            #         location = storage_location[fqn]
            #     # load the fqn into storage_location[fqn]

            #     torch_partial_loader.load_fqn(fqn, location, default_map_location)
            #     result[fqn] = storage_location[fqn]
            pass

        else:
            # determine fqns to read from different ranks if reshard_if_needed is True

            # not all params need resharding . Check the target storage, find tensors and compare
            # with metadata written. This will give us a layout of tensors to read from each rank.

            # then run the same logic as above to load the data into the target storage.
            pass
        return {}
