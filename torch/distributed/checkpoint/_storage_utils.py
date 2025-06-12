import os
from typing import Union

from .filesystem import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.filesystem import FileSystemWriter

from .storage import StorageReader, StorageWriter


def _storage_setup(
    storage: Union[StorageReader, StorageWriter, None],
    checkpoint_id: Union[str, os.PathLike, None],
    reader: bool = False,
    block_on_staging: bool = False,
    use_shared_memory: bool = False,
) -> Union[None, StorageReader, StorageWriter]:
    if storage:
        if checkpoint_id is not None:
            storage.reset(checkpoint_id)
        return storage

    if not checkpoint_id:
        raise RuntimeError(
            "`checkpoint_id` must be specificed if "
            "storage_reader/storage_writer is None."
        )

    targets: list[type[Union[StorageReader, StorageWriter]]] = []
    if reader:
        targets = [
            FileSystemReader,
        ]
    else:
        targets = [
            FileSystemWriter,
        ]
    try:
        from ._fsspec_filesystem import FsspecReader, FsspecWriter

        targets.append(FsspecReader if reader else FsspecWriter)
    except Exception:
        pass

    for target in targets:
        if target.validate_checkpoint_id(checkpoint_id):
            if target == FileSystemWriter:
                storage = target(checkpoint_id, cache_staged_state_dict=not block_on_staging, share_memory=use_shared_memory, block_on_staging=block_on_staging)
            else:
                storage = target(checkpoint_id)  # type: ignore[call-arg]
            storage.reset(checkpoint_id)
            return storage

    raise RuntimeError(
        "Cannot detect which StorageReader or StorageWriter to use. "
        "Please specify the storage_reader/storage_writer."
    )

