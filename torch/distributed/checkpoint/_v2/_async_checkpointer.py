import abc

import io
import json
from concurrent.futures import Future
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, TypeAlias

import torch
import torch.multiprocessing as mp
from torch.types import FileLike

from torch.distributed.checkpoint._v2._checkpointer import (
    Checkpointer,
    StorageWriter,
    StagingMethod,
    CheckpointContext,
    CheckpointingConfig,
    RankInfo,
)


class A0CStagingMethod(StagingMethod):
    def __init__(self, rank_info: RankInfo, config: CheckpointingConfig):
        self._rank_info = rank_info
        self._config = config

    def initiate_staging(
        self,
        state_dict: Dict[str, Any],
        context: CheckpointContext,
    ) -> Any:
        pass

    def wait_for_staging(self):
        pass

    def close(self) -> None:
        pass


class AsyncCheckpointer(Checkpointer):

    def __init__(
        self,
        config: CheckpointingConfig,
        rank_info: RankInfo,
        staging_method: StagingMethod,
        storage_writer: StorageWriter,
    ):
        self._config = config
        self._rank_info = rank_info
        self._staging_method = staging_method
        self._storage_writer = storage_writer
        

    def save(
        self,
        state_dict: Dict[str, Any],
        context: CheckpointContext,
        root_dir: str,
        use_cached_metadata: bool = False,
    ) -> Optional[tuple[Future[None], Future[None]]]:
        pass
