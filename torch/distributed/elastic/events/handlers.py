#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

_log_handlers: dict[str, logging.Handler] = {
    "console": logging.StreamHandler(),
    "dynamic_rendezvous": logging.StreamHandler() if os.environ.get("RECORD_RDZV_EVENT") else logging.NullHandler(),
    "null": logging.NullHandler(),
}


def get_logging_handler(destination: str = "null") -> logging.Handler:
    global _log_handlers
    return _log_handlers[destination]
