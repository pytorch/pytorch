#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import socket


def _get_host_address() -> str:
    """
    Returns address of the host. This can be either hostname or IP address
    """
    return socket.getfqdn(socket.gethostname())
