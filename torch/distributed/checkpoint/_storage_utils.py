import os
from typing import Union

from .filesystem import FileSystemReader, FileSystemWriter
from .storage import StorageReader, StorageWriter


def _storage_setup(
    storage: Union[StorageReader, StorageWriter, None],
    checkpoint_id: Union[str, os.PathLike, None],
    reader: bool = False,
) -> Union[None, StorageReader, StorageWriter]:
    if storage:
        if checkpoint_id is not None:
            storage.reset(checkpoint_id)
        return storage

    if not checkpoint_id:
        raise RuntimeError(
            "`checkpoint_id` must be specified if "
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
            storage = target(checkpoint_id)  # type: ignore[call-arg]
            storage.reset(checkpoint_id)
            return storage

    raise RuntimeError(
        "Cannot detect which StorageReader or StorageWriter to use. "
        "Please specify the storage_reader/storage_writer."
    )
