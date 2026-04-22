from __future__ import annotations

from typing import TYPE_CHECKING

from torch._logging import getArtifactLogger


if TYPE_CHECKING:
    import logging


log: logging.Logger = getArtifactLogger(__name__, "incremental")
