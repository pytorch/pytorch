import logging
from typing import List

from torch.distributed.logging_handlers import _log_handlers


print("in oss file")
__all__: List[str] = []

DCP_LOGGER_NAME = "dcp_logger"

_log_handlers.update(
    {
        DCP_LOGGER_NAME: logging.NullHandler(),
    }
)
