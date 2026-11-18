import logging

from torch.distributed.logging_handlers import _log_handlers


__all__: list[str] = []

DCP_LOGGER_NAME = "dcp_logger"

_log_handlers.update(
    {
        DCP_LOGGER_NAME: logging.NullHandler(),
    }
)
