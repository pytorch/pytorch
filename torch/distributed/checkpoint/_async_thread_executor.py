# pyre-strict
# mypy: allow-untyped-defs
import os
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional, Union

import torch.distributed as dist
from torch.distributed.checkpoint._async_executor import _AsyncCheckpointExecutor
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE
from torch.distributed.checkpoint.planner import SavePlanner
from torch.distributed.checkpoint.storage import StorageWriter


class _ThreadBasedAsyncCheckpointExecutor(_AsyncCheckpointExecutor):
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1)

    def execute_save(
        self,
        staged_state_dict: STATE_DICT_TYPE,
        *,
        checkpoint_id: Union[str, os.PathLike, None] = None,
        storage_writer: Optional[StorageWriter] = None,
        planner: Optional[SavePlanner] = None,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> Future:
        from torch.distributed.checkpoint.state_dict_saver import save

        f: Future = self._executor.submit(
            save,
            staged_state_dict,
            checkpoint_id=checkpoint_id,
            storage_writer=storage_writer,
            planner=planner,
            process_group=process_group,
        )
        f.add_done_callback(lambda f: self._executor.shutdown(wait=False))

        return f
