#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import logging
import time
from typing import Any, Dict, List, Tuple

from torch.distributed.logging_handlers import _log_handlers
import torch.distributed as dist

__all__: List[str] = []


def _get_or_create_logger() -> logging.Logger:
    logging_handler, log_handler_name = _get_logging_handler()
    logger = logging.getLogger(f"c10d-{log_handler_name}")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s %(filename)s:%(lineno)s %(levelname)s p:%(processName)s t:%(threadName)s: %(message)s"
    )
    logging_handler.setFormatter(formatter)
    logger.propagate = False
    logger.addHandler(logging_handler)
    return logger


def _get_logging_handler(destination: str = "default") -> Tuple[logging.Handler, str]:
    log_handler = _log_handlers[destination]
    log_handler_name = type(log_handler).__name__
    return (log_handler, log_handler_name)


global _c10d_logger
_c10d_logger = _get_or_create_logger()


def get_msg_dict(*args, **kwargs) -> Dict[str, Any]:
    if dist.is_initialized():
        msg_dict = {
            "func_name": f"{func.__name__}",
            "args": f"{args}, {kwargs}",
            "pg_name": f"{_get_process_group_name(kwargs.get('group'))}",
            "backend": f"{dist.get_backend(kwargs.get('group'))}",
            "world_size": f"{dist.get_world_size(kwargs.get('group'))}",
            "group_size": f"{dist.get_world_size(kwargs.get('group'))}",
            "global_rank": f"{dist.get_rank()}",
            "local_rank": f"{dist.get_rank(kwargs.get('group'))}",
        }
    else:
        msg_dict = {
            "func_name": f"{func.__name__}",
            "args": f"{args}, {kwargs}",
        }
    return msg_dict


def _exception_logger(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as error:
            from torch.distributed.distributed_c10d import _get_process_group_name
            if dist.is_initialized():
                error_msg_dict = {
                    "func_name": f"{func.__name__}",
                    "args": f"{args}, {kwargs}",
                    "pg_name": f"{_get_process_group_name(kwargs.get('group'))}",
                    "backend": f"{dist.get_backend(kwargs.get('group'))}",
                    "world_size": f"{dist.get_world_size(kwargs.get('group'))}",
                    "group_size": f"{dist.get_world_size(kwargs.get('group'))}",
                    "global_rank": f"{dist.get_rank()}",
                    "local_rank": f"{dist.get_rank(kwargs.get('group'))}",
                    "error": f"{error}",
                }
            else:
                error_msg_dict = {
                    "func_name": f"{func.__name__}",
                    "args": f"{args}, {kwargs}",
                    "error": f"{error}",
                }
            _c10d_logger.debug(error_msg_dict)
            raise
    return wrapper


def _time_logger(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t1 = time.time_ns()
        func_return = func(*args, **kwargs)
        t2 = time.time_ns()

        if dist.is_initialized():
            msg_dict = {
                "func_name": f"{func.__name__}",
                "args": f"{args}, {kwargs}",
                "backend": f"{dist.get_backend(kwargs.get('group'))}",
                "world_size": f"{dist.get_world_size(kwargs.get('group'))}",
                "global_rank": f"{dist.get_rank()}",
                "local_rank": f"{dist.get_rank(kwargs.get('group'))}",
                "time_spent": f"{t2-t1}ns",
            }
        else:
            msg_dict = {
                "func_name": f"{func.__name__}",
                "args": f"{args}, {kwargs}",
                "time_spent": f"{t2-t1}",
            }
        _c10d_logger.debug(msg_dict)

        return func_return

    return wrapper
