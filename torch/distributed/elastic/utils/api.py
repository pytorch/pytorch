#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import socket
from string import Template
from typing import Any


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
    """See :func:`torch.distributed.elastic.utils.distributed.get_socket_with_port`."""
    from torch.distributed.elastic.utils.distributed import (
        get_socket_with_port as _get_socket_with_port,
    )

    return _get_socket_with_port()


class macros:
    """
    Defines simple macros for caffe2.distributed.launch cmd args substitution
    """

    local_rank = "${local_rank}"

    @staticmethod
    def substitute(args: list[Any], local_rank: str) -> list[str]:
        args_sub = []
        for arg in args:
            if isinstance(arg, str):
                sub = Template(arg).safe_substitute(local_rank=local_rank)
                args_sub.append(sub)
            else:
                args_sub.append(arg)
        return args_sub
