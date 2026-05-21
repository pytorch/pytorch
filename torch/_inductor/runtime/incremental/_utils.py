from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch._utils_internal

from torch._logging import getArtifactLogger

from .config import _INCREMENTAL_AUTOTUNE_VERSION


if TYPE_CHECKING:
    import logging


log: logging.Logger = getArtifactLogger(__name__, "incremental")


@functools.cache
def jk_passes() -> bool:
    """Return True if the JK gate allows incremental autotuning."""
    try:
        val = torch._utils_internal.justknobs_getval_int(
            "pytorch/inductor:incremental_autotune_version"
        )
    except AttributeError:
        return True
    return _INCREMENTAL_AUTOTUNE_VERSION >= val
