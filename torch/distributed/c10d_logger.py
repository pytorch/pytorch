#!/usr/bin/env python3
# mypy: allow-untyped-defs

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import logging
from typing import Any, Callable, Dict, List, Tuple, TypeVar
from typing_extensions import ParamSpec

import torch
import torch.distributed as dist
from torch.distributed.logging_handlers import _log_handlers
from torch.monitor import _WaitCounter


__all__: List[str] = []

_DEFAULT_DESTINATION = "default"


def _get_or_create_logger(destination: str = _DEFAULT_DESTINATION) -> logging.Logger:
    logging_handler, log_handler_name = _get_logging_handler(destination)
    logger = logging.getLogger(f"c10d-{log_handler_name}")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s %(filename)s:%(lineno)s %(levelname)s p:%(processName)s t:%(threadName)s: %(message)s"
    )
    logging_handler.setFormatter(formatter)
    logger.propagate = False
    logger.addHandler(logging_handler)
    return logger


def _get_logging_handler(
    destination: str = _DEFAULT_DESTINATION,
) -> Tuple[logging.Handler, str]:
    log_handler = _log_handlers[destination]
    log_handler_name = f"{type(log_handler).__name__}-{destination}"
    return (log_handler, log_handler_name)


global _c10d_logger
_c10d_logger = _get_or_create_logger()


def _get_msg_dict(func_name, *args, **kwargs) -> Dict[str, Any]:
    if dist.is_initialized():
        group = kwargs.get("group") or kwargs.get("process_group")
        msg_dict = {
            "func_name": f"{func_name}",
            "pg_name": f"{dist._get_process_group_name(kwargs.get('pg'))}",  # type: ignore[arg-type]
            "backend": f"{dist.get_backend(group)}",
            "world_size": f"{dist.get_world_size()}",
            "group_size": f"{dist.get_world_size(group)}",
            "global_rank": f"{dist.get_rank()}",
            "local_rank": f"{dist.get_rank(group)}",
        }
        if msg_dict["backend"] == "nccl":
            nccl_version = torch.cuda.nccl.version()
            msg_dict["nccl_version"] = ".".join(str(v) for v in nccl_version)
    else:
        msg_dict = {
            "func_name": f"{func_name}",
        }
    return msg_dict


_T = TypeVar("_T")
_P = ParamSpec("_P")


def _exception_logger(func: Callable[_P, _T]) -> Callable[_P, _T]:
    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        try:
            return func(*args, **kwargs)
        except Exception as error:
            msg_dict = _get_msg_dict(func.__name__, *args, **kwargs)
            msg_dict["error"] = f"{error}"
            _c10d_logger.debug(msg_dict)
            raise

    return wrapper


def _time_logger(func: Callable[_P, _T]) -> Callable[_P, _T]:
    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        with _WaitCounter(f"pytorch.wait_counter.c10d.{func.__name__}").guard():
            func_return = func(*args, **kwargs)
        return func_return

    return wrapper
