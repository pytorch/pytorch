import abc
import logging
from concurrent.futures import Future
from typing import Any, Optional, TypeVar

from .checkpoint_reader import CheckpointReader
from .checkpoint_writer import CheckpointWriter
from .types import STATE_DICT


logger = logging.getLogger(__name__)

LOG_INTERVAL = 60
T = TypeVar("T")


class Checkpointer(abc.ABC):
    """
    WARNING: This class is experimental, and is created to validate certain ideas,
    and is subjected to change or deprecation and we strong discourage any usages at
    this time.

    Abstract base class that defines the API for checkpointing.

    This class defines the interface for coordinating the writing and loading of model
    state dictionaries to and from storage. It provides abstract methods to save and load model states
    with support for both synchronous and asynchronous operations.

    Concrete implementations of this class must implement all the abstract methods.
    """

    @abc.abstractmethod
    def save(
        self,
        state_dict: STATE_DICT,
        path: str,
        **kwargs: dict[str, Any],
    ) -> Optional[tuple[Future, Future]]:
        """
        Save a state dictionary to storage.

        Args:
            state_dict: The state dictionary to save.
            path: The path where the checkpoint should be saved.
            **kwargs: Additional keyword arguments to pass to the writer.

        Returns:
            For synchronous implementations: None
            For asynchronous implementations: tuple of (stage_future, write_future)
                                            representing the staging and writing operations.
        """

    @abc.abstractmethod
    def load(
        self,
        path: str,
        state_dict: Optional[STATE_DICT] = None,
        *,
        default_map_location: Any = None,
        strict: bool = False,
        **kwargs: dict[str, Any],
    ) -> STATE_DICT:
        """
        Load a state dictionary from storage.

        Args:
            path: The path from which to load the checkpoint.
            state_dict: Optional state dictionary to update with loaded values.
                        If provided, only keys in this dictionary will be loaded.
            default_map_location: Device mapping function or device name for relocating tensors.
            strict: If True, raises an error when there are missing keys in the checkpoint.
            **kwargs: Additional keyword arguments to pass to the reader.

        Returns:
            The loaded state dictionary.
        """

    @abc.abstractmethod
    def close(self) -> None:
        """
        Close the checkpointer and release any resources.

        This method should be called when the checkpointer is no longer needed to ensure
        proper cleanup of resources.
        """


class SyncCheckpointer(Checkpointer):
    """
    Synchronous implementation of Checkpointer.

    This class coordinates the writing and loading of model state dictionaries to and from storage
    using only synchronous operations. It provides a simple, efficient interface for checkpoint
    operations without async overhead.

    Attributes:
        _writer: CheckpointWriter for writing state dictionaries to storage.
        _reader: CheckpointReader for reading state dictionaries from storage.

    Example:
        checkpointer = SyncCheckpointer(writer=writer, reader=reader)
        checkpointer.save(state_dict, path)
        loaded_state_dict = checkpointer.load(path)
    """

    def __init__(
        self,
        writer: CheckpointWriter,
        reader: CheckpointReader,
    ):
        """
        Initialize a synchronous checkpointer.

        Args:
            writer: CheckpointWriter for writing checkpoints to storage.
            reader: CheckpointReader for reading checkpoints from storage.
        """
        self._writer = writer
        self._reader = reader

    def save(
        self,
        state_dict: STATE_DICT,
        path: str,
        **kwargs: dict[str, Any],
    ) -> Optional[tuple[Future, Future]]:
        """
        Save a state dictionary to storage synchronously.

        Args:
            state_dict: The state dictionary to save.
            path: The path where the checkpoint should be saved.
            **kwargs: Additional keyword arguments to pass to the writer.

        Returns:
            Always returns None as operations are synchronous.

        Example:
            checkpointer.save(state_dict, "/path/to/checkpoint")
        """
        logger.debug("Saving checkpoint synchronously to %s", path)
        self._writer.write(state_dict, path, **kwargs)
        return None

    def load(
        self,
        path: str,
        state_dict: Optional[STATE_DICT] = None,
        *,
        default_map_location: Any = None,
        strict: bool = False,
        **kwargs: dict[str, Any],
    ) -> STATE_DICT:
        """
        Load a state dictionary from storage.

        Args:
            path: The path from which to load the checkpoint.
            state_dict: Optional state dictionary to update with loaded values.
                        If provided, only keys in this dictionary will be loaded.
            default_map_location: Device mapping function or device name for relocating tensors.
            strict: If True, raises an error when there are missing keys in the checkpoint.
            **kwargs: Additional keyword arguments to pass to the reader.

        Returns:
            The loaded state dictionary.

        Raises:
            RuntimeError: If strict=True and there are missing keys in the checkpoint.
            FileNotFoundError: If the checkpoint file is not found.
        """
        logger.info("Loading checkpoint from %s", path)

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
        logger.info("SyncCheckpointer closed")
