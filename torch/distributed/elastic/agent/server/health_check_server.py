#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable

from torch.distributed.elastic.utils.logging import get_logger

log = get_logger(__name__)


def start_healthcheck_server(
    alive_callback: Callable[[], int],
    port: int,
    timeout: int,
) -> Callable[[], None]:
    """
    Unsupported functionality for Pytorch, doesn't start any health check server
    """
    log.info("No health check server started")

    return lambda: None
