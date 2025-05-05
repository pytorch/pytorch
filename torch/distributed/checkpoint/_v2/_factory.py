from contextlib import contextmanager

from torch.distributed.checkpoint._v2.base import (
    CheckpointContext,
    CheckpointingConfig,
    RankInfo,
    StorageWriter,
    CheckpointWriter,
    Checkpointer,
    SyncCheckpointer,
    ModelStore,
)


def create_checkpointer(
    self,
    config: CheckpointingConfig,
    rank_info: RankInfo,
    storage: ModelStore,
) -> Checkpointer:
    """
    This is where we construct a checkpointer based on the config.

    TODO: returning a naive sync checkpointer for now but complete
    this with later with sync and async checkpointer.
    """

    return SyncCheckpointer(
        config=config,
        rank_info=rank_info,
        storage=storage,
    )
