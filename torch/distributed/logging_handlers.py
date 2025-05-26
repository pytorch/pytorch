#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging


__all__: list[str] = []

_log_handlers: dict[str, logging.Handler] = {
    "default": logging.NullHandler(),
}
