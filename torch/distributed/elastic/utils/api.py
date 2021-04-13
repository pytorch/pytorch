#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os


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
