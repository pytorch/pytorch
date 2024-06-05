import logging
import logging.config
import os
from typing import Optional

import torch.distributed as dist


LOGGING_CONFIG = {
    "version": 1,
    "formatters": {
        "spmd_format": {"format": "%(name)s: [%(levelname)s] %(message)s"},
        "graph_opt_format": {"format": "%(name)s: [%(levelname)s] %(message)s"},
    },
    "handlers": {
        "spmd_console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "spmd_format",
            "stream": "ext://sys.stdout",
        },
        "graph_opt_console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "graph_opt_format",
            "stream": "ext://sys.stdout",
        },
        "null_console": {
            "class": "logging.NullHandler",
        },
    },
    "loggers": {
        "spmd_exp": {
            "level": "DEBUG",
            "handlers": ["spmd_console"],
            "propagate": False,
        },
        "graph_opt": {
            "level": "DEBUG",
            "handlers": ["graph_opt_console"],
            "propagate": False,
        },
        "null_logger": {
            "handlers": ["null_console"],
            "propagate": False,
        },
        # TODO(anj): Add loggers for MPMD
    },
    "disable_existing_loggers": False,
}


def get_logger(log_type: str) -> Optional[logging.Logger]:
    from torch.distributed._spmd import config

    if "PYTEST_CURRENT_TEST" not in os.environ:
        logging.config.dictConfig(LOGGING_CONFIG)
        avail_loggers = list(LOGGING_CONFIG["loggers"].keys())  # type: ignore[attr-defined]
        assert (
            log_type in avail_loggers
        ), f"Unable to find {log_type} in the available list of loggers {avail_loggers}"

        if not dist.is_initialized():
            return logging.getLogger(log_type)

        if dist.get_rank() == 0:
            logger = logging.getLogger(log_type)
            logger.setLevel(config.log_level)
            if config.log_file_name is not None:
                log_file = logging.FileHandler(config.log_file_name)
                log_file.setLevel(config.log_level)
                logger.addHandler(log_file)
        else:
            logger = logging.getLogger("null_logger")

        return logger

    return logging.getLogger("null_logger")
