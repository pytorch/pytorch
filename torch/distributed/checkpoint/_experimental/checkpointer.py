import logging
from typing import Any, Optional, TypeVar

from .checkpoint_reader import CheckpointReader
from .checkpoint_writer import CheckpointWriter
from .types import STATE_DICT


logger = logging.getLogger(__name__)

LOG_INTERVAL = 60
T = TypeVar("T")


class Checkpointer:
    """
    WARNING: This class is experimental, and is created to validate certain ideas,
    and is subjected to change or deprecation and we strong discourage any usages at
    this time.

    Orchestrates the checkpointing process.

    This class coordinates the writing and loading of model state dictionaries to and from storage.
    It provides methods to save and load model states.

    Attributes:
        _writer: Writer for writing state dictionaries to storage.
        _reader: Reader for reading state dictionaries from storage.
    """

    def __init__(
        self,
        writer: CheckpointWriter,
        reader: CheckpointReader,
    ):
        """
        Initialize a Checkpointer.

        Args:
            writer: Writer for writing state dictionaries to storage.
            reader: Reader for reading state dictionaries from storage.
        """
        self._writer = writer
        self._reader = reader

    def save(
        self,
        state_dict: STATE_DICT,
        path: str,
        **kwargs: Any,
    ) -> None:
        """
        Save a state dictionary to storage.

        This method writes the state dictionary to storage at the specified path.
        It ensures that previous checkpoint operations have completed successfully
        before starting new ones.

        Args:
            state_dict: The state dictionary to save.
            path: The path where the checkpoint should be saved.
            **kwargs: Additional keyword arguments to pass to the writer.
        """
        logger.info("Initiating checkpoint save to %s.", path)
        self._writer.write(state_dict, path, **kwargs)

    def load(
        self,
        path: str,
        state_dict: Optional[STATE_DICT] = None,
        *,
        default_map_location: Any = None,
        strict: bool = False,
        **kwargs: Any,
    ) -> STATE_DICT:
        """
        Load a state dictionary from storage.

        This method reads a state dictionary from storage at the specified path.

        Args:
            path: The path from which to load the checkpoint.
            state_dict: Optional state dictionary to update with loaded values.
                        If provided, only keys in this dictionary will be loaded.
            default_map_location: Device mapping function or device name for relocating tensors.
            strict (bool): Whether to raise an error if the loaded state dictionary is missing keys.
            **kwargs: Additional keyword arguments to pass to the reader.

        Returns:
            The loaded state dictionary.
        """
        logger.info("Loading checkpoint from %s", path)

        # Otherwise, read the full checkpoint
        loaded_state_dict, missing_keys = self._reader.read(
            path=path,
            state_dict=state_dict,
            map_location=default_map_location,
            **kwargs,
        )
        if strict and missing_keys is not None and missing_keys != []:
            raise RuntimeError(f"Checkpoint at {path} is missing keys: {missing_keys}")
        return loaded_state_dict

    def close(self) -> None:
        """
        Close the checkpointer and release any resources.

        This method should be called when the checkpointer is no longer needed to ensure
        proper cleanup of resources.
        """
        self._writer.close()
        logger.info("Checkpointer closed")
