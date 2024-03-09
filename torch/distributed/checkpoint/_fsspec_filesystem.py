# Mypy will not try inferring the types of any 3rd party libraries installed.
# mypy: ignore-errors

import logging

from torch.distributed.checkpoint.fsspec import (  # noqa: F401  # noqa: F401
    FsspecReader,
    FsspecWriter,
)

log = logging.getLogger(__name__)
log.warning(
    "FSSpec Filesystem has been made public, please update your "
    "import to torch.distributed.checkpoint"
)
