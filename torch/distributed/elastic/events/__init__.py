#!/usr/bin/env/python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Module contains events processing mechanisms that are integrated with the standard python logging.

Example of usage:

::

  from torch.distributed.elastic import events
  event = events.Event(name="test_event", source=events.EventSource.WORKER, metadata={...})
  events.get_logging_handler(destination="console").info(event)

"""

import inspect
import logging
import os
import socket
import traceback
from enum import Enum
from typing import Dict, Optional

from torch.distributed.elastic.events.handlers import get_logging_handler

from .api import (  # noqa: F401
    Event,
    EventMetadataValue,
    EventSource,
    NodeState,
    RdzvEvent,
)

_events_loggers: Dict[str, logging.Logger] = {}

def _get_or_create_logger(destination: str = "null") -> logging.Logger:
    """
    Construct python logger based on the destination type or extends if provided.

    Available destination could be found in ``handlers.py`` file.
    The constructed logger does not propagate messages to the upper level loggers,
    e.g. root logger. This makes sure that a single event can be processed once.

    Args:
        destination: The string representation of the event handler.
            Available handlers found in ``handlers`` module
    """
    global _events_loggers

    if destination not in _events_loggers:
        _events_logger = logging.getLogger(f"torchelastic-events-{destination}")
        _events_logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))
        # Do not propagate message to the root logger
        _events_logger.propagate = False

        logging_handler = get_logging_handler(destination)
        _events_logger.addHandler(logging_handler)

        # Add the logger to the global dictionary
        _events_loggers[destination] = _events_logger

    return _events_loggers[destination]


def record(event: Event, destination: str = "null") -> None:
    _get_or_create_logger(destination).info(event.serialize())

def record_rdzv_event(event: RdzvEvent) -> None:
    _get_or_create_logger("dynamic_rendezvous").info(event.serialize())


def construct_and_record_rdzv_event(
    run_id: str,
    message: str,
    node_state: NodeState,
    name: str = "",
    hostname: str = "",
    pid: Optional[int] = None,
    master_endpoint: str = "",
    local_id: Optional[int] = None,
    rank: Optional[int] = None,
) -> None:
    # We don't want to perform an extra computation if not needed.
    if isinstance(get_logging_handler("dynamic_rendezvous"), logging.NullHandler):
        return

    # Set up parameters.
    if not hostname:
        hostname = socket.getfqdn()
    if not pid:
        pid = os.getpid()

    # Determines which file called this function.
    callstack = inspect.stack()
    filename = "no_file"
    if len(callstack) > 1:
        stack_depth_1 = callstack[1]
        filename = os.path.basename(stack_depth_1.filename)
        if not name:
            name = stack_depth_1.function

    # Delete the callstack variable. If kept, this can mess with python's
    # garbage collector as we are holding on to stack frame information in
    # the inspect module.
    del callstack

    # Set up error trace if this is an exception
    if node_state == NodeState.FAILED:
        error_trace = traceback.format_exc()
    else:
        error_trace = ""

    # Initialize event object
    event = RdzvEvent(
        name=f"{filename}:{name}",
        run_id=run_id,
        message=message,
        hostname=hostname,
        pid=pid,
        node_state=node_state,
        master_endpoint=master_endpoint,
        rank=rank,
        local_id=local_id,
        error_trace=error_trace,
    )

    # Finally, record the event.
    record_rdzv_event(event)
