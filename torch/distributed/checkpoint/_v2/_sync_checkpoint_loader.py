import abc
from typing import Callable, Optional, Union, Dict, Any, List

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

    def read(
        self,
        root_dir: str,
        context: Any,
        rank: int,
        fqns_to_load: Union[str, List[str], None], 
        map_location: _MAP_LOCATION,
    ) -> Any:
        """
        Writes the state_dict to storage.

        Args:
            state_dict (Dict[str, Any]): The state_dict to write.
            manifest (Optional[Manifest]): The manifest to write.
            context (Any): The context to write.
            path (str): The path to write the checkpoint to.

        Returns:
            str: The path to the checkpoint.
        """


        convert fqns to file paths 

        load all the file and filter for params (or load the respective params by calculating offset of storages from zip metadata)



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

        self._reader.read(root_dir, self._config, self._rank_info.global_rank, fqns_to_load, map_location)

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
                # determine fqns to read from different ranks 

        

        return {}
        