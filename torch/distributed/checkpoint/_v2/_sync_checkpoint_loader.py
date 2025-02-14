import abc
from typing import BinaryIO, Callable, Generator, Optional, Union, Dict, Any, List, Generator


from torch.distributed.checkpoint._v2._checkpointing import (
    CheckpointContext,
    CheckpointingConfig,
    RankInfo,
    Storage,
    CheckpointLayout,
    SerializationFormat,
    # CheckpointReader,
    _STORAGE,
    _MAP_LOCATION,
    ModelStorage,
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
            2. Load specified params to a given pre-allocated location in chunks or return
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
        # compute the offset of each storage and build a map of {fqn: offset, size}
        # for each storage. we could use the tensor metadata read for resharding on the 
        # fly if  we dont have a global metadata but that is not needed for now.
        self.shallow_state_dict = {}
        pass
    
    @abc.abstractmethod
    def load_fqn(self, fqn: str, map_location: MAP_LOCATION) -> Any:
        # map the fqn to object in shallow_state_dict. Find the corresponding storage key.
        # Look up the map location and the param there load it.
        pass

    @abc.abstractmethod
    def load_in_chunks(self, fqn: str) -> BinaryIO:
        # map the fqn to object in shallow_state_dict. Find the corresponding storage key.
        # Look up the map location and the param there load it.
        pass


class CheckpointReader:
    def __init__(
        self,
        config: Any,
        rank_info: RankInfo,
        storage: ModelStorage,
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


class SyncCheckpointLoader(abc.ABC):
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
        root_dir: str,
        fqns_to_load: Union[str, List[str], None], 
        map_location: _MAP_LOCATION,
    ) -> Dict[str, Any]:
        """ 
            Loads a checkpoint from a given path for the current rank.

            if fqns_to_load is None, this will load the entire state_dict written from this rank.
            if fqns_to_load is list or a str, this will load params for the fqns specified in the 
            list and returns a dict of {fqn: obj}. If the fqn is not found in the checkpoint, 
            an exception will be raised. (TODO: should we just skip as the user can do it themselves?)

            the checkpoint loader will load params to given map_location.
        """

        self._reader.read_from_rank(root_dir, self._config, self._rank_info.global_rank, fqns_to_load, map_location)

        return {}

    def load_in_place(
        self,
        path: str,
        storage_location: Dict[str, Any],
        reshard_if_needed: bool = True,
    ) -> Dict[str, Any]:
        """ 
            Loads a checkpoint from a given path for the current rank.

            if fqns_to_load is None, this will load the entire state_dict written from this rank.
            if fqns_to_load is list or a str, this will load params for the fqns specified in the 
            list and returns a dict of {fqn: obj}. If the fqn is not found in the checkpoint, 
            an exception will be raised. (TODO: should we just skip as the user can do it themselves?)

            the checkpoint loader will load params to given map_location.
        """

        if not reshard_if_needed:
            # get all fqns to load from storage_location
            fqns_to_load = []

            return 
                # determine fqns to read from different ranks if reshard_if_needed is True

        # get all fqns to load from storage_location
        
        # iterate on each one and 

        return {}
        