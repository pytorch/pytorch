#!/usr/bin/env python3
# mypy: allow-untyped-defs

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import datetime
import os
import socket
from contextlib import closing
from typing import Optional

import torch.distributed as dist
from torch.distributed.elastic.utils.logging import get_logger
from torch.distributed.elastic.utils.store import barrier


__all__ = ["create_c10d_store", "get_free_port", "get_socket_with_port"]

logger = get_logger(__name__)

_ADDRESS_IN_USE = "Address already in use"
_SOCKET_TIMEOUT = "Socket Timeout"

_TCP_STORE_INIT = "_tcp_store/num_members"


def create_c10d_store(
    is_server: bool,
    server_addr: str,
    server_port: int = -1,
    world_size: int = 1,
    timeout: float = (60 * 10),  # 10 min
    wait_for_workers: bool = True,
    retries=3,
    use_libuv: Optional[bool] = None,
):
    if use_libuv is not None:
        logger.warning(
            "argument use_libuv is deprecated and ignored. Set USE_LIBUV environment "
            'variable to "0" to disable libuv, or "1" to enable it. If the env var '
            "is not set, libuv will be used by default."
        )

    # check os.environ for use_libuv
    use_libuv = os.environ.get("USE_LIBUV", "1") == "1"  # libuv is the default option

    if server_port == -1 and world_size > 1:
        raise ValueError(
            f"server_port must be specified when world_size > 1, got server_port={server_port}, world_size={world_size}"
        )

    if server_port != -1:
        logger.info("sever_port: %s, specified, ignoring retries", server_port)

    # only retry when server_port is NOT static
    attempt = retries if server_port == -1 else 1
    while True:
        if server_port != -1:
            port = server_port
        else:
            port = get_free_port()

        logger.info(
            "Creating c10d store on %s:%s\n"
            "  world_size  : %s\n"
            "  is_server   : %s\n"
            "  timeout(sec): %s\n"
            "  use_libuv   : %s\n",
            server_addr,
            port,
            world_size,
            is_server,
            timeout,
            use_libuv,
        )

        try:
            store = dist.TCPStore(
                host_name=server_addr,
                port=port,
                world_size=world_size,
                is_master=is_server,
                timeout=datetime.timedelta(seconds=timeout),
                wait_for_workers=wait_for_workers,
                use_libuv=use_libuv,
            )
            # skips full rank check when we don't have to wait for all workers
            if wait_for_workers:
                _check_full_rank(store, world_size, timeout=timeout)
            logger.info("Successfully created c10d store")
            return store
        except RuntimeError as e:
            # this is brittle, but the underlying exception type is not properly pybinded
            # so we parse the error msg for now, interestingly this is how torch itself
            # detects timeouts and port conflicts in their own unittests
            # see - caffe2/torch/testing/_internal/common_utils.py
            # TODO properly map the exceptions in pybind (c10d/init.cpp)
            if str(e) == _ADDRESS_IN_USE:  # this will only happen on the server
                if attempt < retries:
                    logger.warning(
                        "port: %s already in use, attempt: [%s/%s]",
                        port,
                        attempt,
                        retries,
                    )
                    attempt += 1
                else:
                    raise RuntimeError(
                        f"on {server_addr}, port: {port} already in use"
                    ) from e
            else:
                raise


def _check_full_rank(store, world_size, timeout):
    try:
        barrier(store, world_size, key_prefix=_TCP_STORE_INIT, barrier_timeout=timeout)
    except RuntimeError as e:
        if str(e) == _SOCKET_TIMEOUT:
            raise TimeoutError(
                f"timed out waiting for all {world_size} members to join"
            ) from e
        else:
            raise


def get_free_port():
    """
    Returns an unused port on localhost.

    This function finds an unused port on localhost by opening to socket to bind
    to a port and then closing it.

    Returns:
        int: an unused port on localhost

    Example:
        >>> # xdoctest: +SKIP("Nondeterministic")
        >>> get_free_port()
        63976

    ..note:
        The port returned by :func:`get_free_port` is not reserved and may be
        taken by another process after this function returns.
    """
    sock = get_socket_with_port()
    with closing(sock):
        return sock.getsockname()[1]


def get_socket_with_port() -> socket.socket:
    """
    Returns a free port on localhost that is "reserved" by binding a temporary
    socket on it. Close the socket before passing the port to the entity
    that requires it. Usage example

    ::

    sock = _get_socket_with_port()
    with closing(sock):
        port = sock.getsockname()[1]
        sock.close()
        # there is still a race-condition that some other process
        # may grab this port before func() runs
        func(port)
    """

    addrs = socket.getaddrinfo(
        host="localhost", port=None, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM
    )
    for addr in addrs:
        family, type, proto, _, _ = addr
        s = socket.socket(family, type, proto)
        try:
            s.bind(("localhost", 0))
            s.listen(0)
            return s
        except OSError as e:
            s.close()
            logger.warning("Socket creation attempt failed.", exc_info=e)
    raise RuntimeError("Failed to create a socket")
