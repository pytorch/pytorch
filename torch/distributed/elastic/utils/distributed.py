#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import datetime
import socket
from contextlib import closing

import torch.distributed as dist
from torch.distributed.elastic.utils.logging import get_logger


log = get_logger()

_ADDRESS_IN_USE = "Address already in use"
_CONNECT_TIMEOUT = "connect() timed out."
_SOCKET_TIMEOUT = "Socket Timeout"

_MEMBER_CHECKIN = "_tcp_store/num_members"
_LAST_MEMBER_CHECKIN = "_tcp_store/last_member"


def create_c10d_store(
    is_server: bool,
    server_addr: str,
    server_port: int = -1,
    world_size: int = 1,
    timeout: float = (60 * 10),  # 10 min
    retries=3,
):
    if server_port == -1 and world_size > 1:
        raise ValueError(
            f"server_port must be specified when world_size > 1, got server_port={server_port}, world_size={world_size}"
        )

    if server_port != -1:
        log.info(f"sever_port: {server_port}, specified, ignoring retries")

    # only retry when server_port is NOT static
    attempt = retries if server_port == -1 else 1
    while True:
        if server_port != -1:
            port = server_port
        else:
            port = get_free_port()

        log.info(
            f"Creating c10d store on {server_addr}:{port}\n"
            f"  world_size  : {world_size}\n"
            f"  is_server   : {is_server}\n"
            f"  timeout(sec): {timeout}\n"
        )

        try:
            store = dist.TCPStore(
                host_name=server_addr,
                port=port,
                world_size=world_size,
                is_master=is_server,
                timeout=datetime.timedelta(seconds=timeout),
            )
            _check_full_rank(store, world_size)
            log.info("Successfully created c10d store")
            return store
        except RuntimeError as e:
            # this is brittle, but the underlying exception type is not properly pybinded
            # so we parse the error msg for now, interestingly this is how torch itself
            # detects timeouts and port conflicts in their own unittests
            # see - caffe2/torch/testing/_internal/common_utils.py
            # TODO properly map the exceptions in pybind (c10d/init.cpp)
            if str(e) == _CONNECT_TIMEOUT and not is_server:
                raise TimeoutError(
                    f"timed out waiting for tcp store's server: {server_addr}:{port}"
                ) from e
            elif str(e) == _ADDRESS_IN_USE:  # this will only happen on the server
                if attempt < retries:
                    log.warning(
                        f"port: {port} already in use, attempt: [{attempt}/{retries}]"
                    )
                    attempt += 1
                else:
                    raise IOError(
                        f"on {server_addr}, port: {port} already in use"
                    ) from e
            else:
                raise


def _check_full_rank(store, world_size):
    idx = store.add(_MEMBER_CHECKIN, 1)
    if idx == world_size:
        store.set(_LAST_MEMBER_CHECKIN, "<val_ignored>")

    try:
        store.get(_LAST_MEMBER_CHECKIN)
    except RuntimeError as e:
        if str(e) == _SOCKET_TIMEOUT:
            raise TimeoutError(
                f"timed out waiting for all {world_size} members to join"
            ) from e
        else:
            raise


def get_free_port():
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
            log.info("Socket creation attempt failed.", exc_info=e)
    raise RuntimeError("Failed to create a socket")
