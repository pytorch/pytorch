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


def save_wrapper(
    staging_future_or_state_dict: Union[Future[STATE_DICT_TYPE], STATE_DICT_TYPE],
    *,
    checkpoint_id: Union[str, os.PathLike, None] = None,
    storage_writer: Optional[StorageWriter] = None,
    planner: Optional[SavePlanner] = None,
    process_group: Optional[dist.ProcessGroup] = None,
    no_dist: bool = False,
    use_collectives: bool = True,
) -> Future:
    from torch.distributed.checkpoint.state_dict_saver import save

    staged_dict = (
        staging_future_or_state_dict.result()
        if isinstance(staging_future_or_state_dict, Future)
        else staging_future_or_state_dict
    )
    return save(
        staged_dict,
        checkpoint_id=checkpoint_id,
        storage_writer=storage_writer,
        planner=planner,
        process_group=process_group,
        no_dist=no_dist,
        use_collectives=use_collectives,
    )


class _ThreadBasedAsyncCheckpointExecutor(_AsyncCheckpointExecutor):
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="AsyncCheckpointExecutor"
        )

    def execute_save(
        self,
        staging_future_or_state_dict: Union[Future[STATE_DICT_TYPE], STATE_DICT_TYPE],
        *,
        checkpoint_id: Union[str, os.PathLike, None] = None,
        storage_writer: Optional[StorageWriter] = None,
        planner: Optional[SavePlanner] = None,
        process_group: Optional[dist.ProcessGroup] = None,
        no_dist: bool = False,
        use_collectives: bool = True,
    ) -> Future:
        f: Future = self._executor.submit(
            save_wrapper,
            staging_future_or_state_dict=staging_future_or_state_dict,
            checkpoint_id=checkpoint_id,
            storage_writer=storage_writer,
            planner=planner,
            process_group=process_group,
            no_dist=no_dist,
            use_collectives=use_collectives,
        )
        f.add_done_callback(lambda f: self._executor.shutdown(wait=False))

        return f
