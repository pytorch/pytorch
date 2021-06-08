#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import socket
from string import Template
from typing import List, Any


def get_env_variable_or_raise(env_name: str) -> str:
    r"""
    Tries to retrieve environment variable. Raises ``ValueError``
    if no environment variable found.

    Args:
        env_name (str): Name of the env variable
    """
    value = os.environ.get(env_name, None)
    if value is None:
        msg = f"Environment variable {env_name} expected, but not set"
        raise ValueError(msg)
    return value


def get_socket_with_port() -> socket.socket:
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
    raise RuntimeError("Failed to create a socket")


class macros:
    """
    Defines simple macros for caffe2.distributed.launch cmd args substitution
    """

    local_rank = "${local_rank}"

    @staticmethod
    def substitute(args: List[Any], local_rank: str) -> List[str]:
        args_sub = []
        for arg in args:
            if isinstance(arg, str):
                sub = Template(arg).safe_substitute(local_rank=local_rank)
                args_sub.append(sub)
            else:
                args_sub.append(arg)
        return args_sub
