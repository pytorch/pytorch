#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable

from torch.distributed.elastic.utils.logging import get_logger

log = get_logger(__name__)


class HealthCheckServer:
    """
    Interface for health check monitoring server, which can be extended
    by starting tcp/http server on the specified port.

    Args:

        alive_callback: Callable[[], int], callback to last progress time of agent

        port: int, port number to start tcp/http server

        timeout: int, timeout seconds to decide agent is alive/dead
    """

    _alive_callback: Callable[[], int]
    _port: int
    _timeout: int

    def __init__(
        self, alive_callback: Callable[[], int], port: int, timeout: int
    ) -> None:
        self._alive_callback = alive_callback
        self._port = port
        self._timeout = timeout

    def start(self) -> None:
        """
        Unsupported functionality for Pytorch, doesn't start any health check server
        """
        log.warning("No health check server started")

    def stop(self) -> None:
        """
        Function to stop health check server
        """
        log.info("Stopping noop health check server.")


def create_healthcheck_server(
    alive_callback: Callable[[], int],
    port: int,
    timeout: int,
) -> HealthCheckServer:
    """
    creates health check server object
    """
    return HealthCheckServer(alive_callback, port, timeout)
