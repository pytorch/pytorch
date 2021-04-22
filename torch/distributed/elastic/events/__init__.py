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
  event = Event(name="test_event", source=EventSource.WORKER, metadata={...})
  events.get_events_logger(destination="default").info(event)

"""

import logging

from torch.distributed.elastic.events.handlers import get_logging_handler

from .api import Event, EventSource, EventMetadataValue  # noqa: F401

_events_logger = None


def _get_or_create_logger(destination: str = "null") -> logging.Logger:
    """
    Constructs python logger based on the destination type or extends if provided.
    Available destination could be found in ``handlers.py`` file.
    The constructed logger does not propagate messages to the upper level loggers,
    e.g. root logger. This makes sure that a single event can be processed once.

    Args:
        destination: The string representation of the event handler.
            Available handlers found in ``handlers`` module
        logger: Logger to be extended with the events handler. Method constructs
            a new logger if None provided.
    """
    global _events_logger
    if _events_logger:
        return _events_logger
    logging_handler = get_logging_handler(destination)
    _events_logger = logging.getLogger(f"torchelastic-events-{destination}")
    _events_logger.setLevel(logging.DEBUG)
    # Do not propagate message to the root logger
    _events_logger.propagate = False
    _events_logger.addHandler(logging_handler)
    return _events_logger


def record(event: Event, destination: str = "console") -> None:
    _get_or_create_logger(destination).info(event.serialize())
