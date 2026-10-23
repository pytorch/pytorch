# pyre-strict
# mypy: allow-untyped-defs
import abc
import logging
import os
import threading
import traceback
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any

import torch.distributed as dist
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE
from torch.distributed.checkpoint.planner import SavePlanner
from torch.distributed.checkpoint.storage import StorageWriter


logger = logging.getLogger(__name__)


class _OwnedStateDictFuture(Future[Any]):
    def __init__(
        self,
        state_dict: STATE_DICT_TYPE,
        cleanup: Callable[[], None],
    ) -> None:
        super().__init__()
        self._state_dict: STATE_DICT_TYPE | None = state_dict
        self._cleanup: Callable[[], None] | None = cleanup
        self._release_lock = threading.Lock()

    def run(self, save_fn: Callable[..., Any]) -> None:
        if not self.set_running_or_notify_cancel():
            return

        state_dict = self._state_dict
        if state_dict is None:
            raise AssertionError("staged state dictionary was released before save")
        try:
            result = save_fn(staging_future_or_state_dict=state_dict)
        except BaseException as error:
            state_dict = None
            traceback.clear_frames(error.__traceback__)
            self._close()
            self.set_exception(error)
            return
        state_dict = None
        self._close()
        self.set_result(result)

    def _close(self) -> None:
        with self._release_lock:
            cleanup = self._cleanup
            self._cleanup = None
        try:
            if cleanup is not None:
                cleanup()
        except Exception:
            logger.exception("Failed to close internally owned async stager")

    def _release(self) -> None:
        with self._release_lock:
            self._state_dict = None

    def cancel(self) -> bool:
        cancelled = super().cancel()
        if cancelled:
            self._close()
            self._release()
        return cancelled

    def result(self, timeout: float | None = None) -> Any:
        try:
            return super().result(timeout)
        finally:
            if self.done():
                self._release()

    def exception(self, timeout: float | None = None) -> BaseException | None:
        try:
            return super().exception(timeout)
        finally:
            if self.done():
                self._release()


_STAGING_INPUT = STATE_DICT_TYPE | Future[STATE_DICT_TYPE] | _OwnedStateDictFuture


class _AsyncCheckpointExecutor(abc.ABC):
    @abc.abstractmethod
    def execute_save(
        self,
        staging_future_or_state_dict: _STAGING_INPUT,
        *,
        checkpoint_id: str | os.PathLike | None = None,
        storage_writer: StorageWriter | None = None,
        planner: SavePlanner | None = None,
        process_group: dist.ProcessGroup | None = None,
        no_dist: bool = False,
        use_collectives: bool = True,
    ) -> Future:
        """
        Execute the checkpoint save request asynchronously.

        This method is intended to be used as an abstraction for
        implementing async checkpointing. The actual checkpoint save
        operation is executed in a separate thread or process depending
        on the implementation of this interface.
        """
