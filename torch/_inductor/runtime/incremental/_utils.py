from __future__ import annotations

import logging

from torch._logging import getArtifactLogger

log: logging.Logger = getArtifactLogger(__name__, "incremental")
