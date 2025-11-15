import os
from typing import cast, Optional, Union

from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    TensorStorageMetadata,
)

from ._storage_utils import _storage_setup
from .storage import StorageReader


__all__ = ["list_stored_state_dict"]


def list_stored_state_dict(
    checkpoint_id: Union[str, os.PathLike, None] = None,
    storage_reader: Optional[StorageReader] = None,
) -> dict[str, Union[TensorStorageMetadata, BytesStorageMetadata]]:
    """
    List the stored checkpoint metadata.
    NB: The returned state-dict keys are flattened.
    """
    storage_reader = cast(
        StorageReader, _storage_setup(storage_reader, checkpoint_id, reader=True)
    )
    md = storage_reader.read_metadata()
    return md.state_dict_metadata  # flattened dict.
