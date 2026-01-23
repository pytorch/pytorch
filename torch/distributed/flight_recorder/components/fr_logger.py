# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections.abc import Callable
from typing import Any


__all__ = ["FlightRecorderLogger"]


class FlightRecorderLogger:
    _instance: Any | None = None
    logger: logging.Logger

    def __init__(self) -> None:
        self.logger: logging.Logger = logging.getLogger("Flight Recorder")

    def __new__(cls) -> Any:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.logger = logging.getLogger("Flight Recorder")
            cls._instance.logger.setLevel(logging.INFO)
            formatter = logging.Formatter("%(message)s")
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            cls._instance.logger.addHandler(ch)
        return cls._instance

    def set_log_level(self, level: int) -> None:
        self.logger.setLevel(level)

    @property
    def debug(self) -> Callable[..., None]:
        return self.logger.debug

    @property
    def info(self) -> Callable[..., None]:
        return self.logger.info

    @property
    def warning(self) -> Callable[..., None]:
        return self.logger.warning

    @property
    def error(self) -> Callable[..., None]:
        return self.logger.error

    @property
    def critical(self) -> Callable[..., None]:
        return self.logger.critical
