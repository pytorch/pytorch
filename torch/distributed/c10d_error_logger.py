#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Tuple

from torch.distributed.logging_handlers import _log_handlers

_c10d_error_logger = None


def _get_or_create_logger() -> logging.Logger:
    global _c10d_error_logger
    if _c10d_error_logger:
        return _c10d_error_logger
    logging_handler, log_handler_name = _get_logging_handler()
    _c10d_error_logger = logging.getLogger(f"c10d-collectives-{log_handler_name}")
    _c10d_error_logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s %(filename)s:%(lineno)s %(levelname)s p:%(processName)s t:%(threadName)s: %(message)s"
    )
    logging_handler.setFormatter(formatter)
    _c10d_error_logger.propagate = False
    _c10d_error_logger.addHandler(logging_handler)
    return _c10d_error_logger


def _get_logging_handler(destination: str = "default") -> Tuple[logging.Handler, str]:
    log_handler = _log_handlers[destination]
    log_handler_name = type(log_handler).__name__
    return (log_handler, log_handler_name)
