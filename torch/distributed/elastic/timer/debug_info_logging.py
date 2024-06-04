#!/usr/bin/env python3
# mypy: allow-untyped-defs

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List

from torch.distributed.elastic.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["log_debug_info_for_expired_timers"]


def log_debug_info_for_expired_timers(
    run_id: str,
    expired_timers: Dict[int, List[str]],
):
    if expired_timers:
        logger.info("Timers expired for run:[%s] [%s].", run_id, expired_timers)
