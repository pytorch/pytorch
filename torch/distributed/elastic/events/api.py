#!/usr/bin/env python3
# mypy: allow-untyped-defs

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional, Union


__all__ = ["EventSource", "Event", "NodeState", "RdzvEvent"]

EventMetadataValue = Union[str, int, float, bool, None]


class EventSource(str, Enum):
    """Known identifiers of the event producers."""

    AGENT = "AGENT"
    WORKER = "WORKER"


@dataclass
class Event:
    """
    The class represents the generic event that occurs during the torchelastic job execution.

    The event can be any kind of meaningful action.

    Args:
        name: event name.
        source: the event producer, e.g. agent or worker
        timestamp: timestamp in milliseconds when event occurred.
        metadata: additional data that is associated with the event.
    """

    name: str
    source: EventSource
    timestamp: int = 0
    metadata: dict[str, EventMetadataValue] = field(default_factory=dict)

    def __str__(self):
        return self.serialize()

    @staticmethod
    def deserialize(data: Union[str, "Event"]) -> "Event":
        if isinstance(data, Event):
            return data
        if isinstance(data, str):
            data_dict = json.loads(data)
        data_dict["source"] = EventSource[data_dict["source"]]  # type: ignore[possibly-undefined]
        # pyrefly: ignore  # unbound-name
        return Event(**data_dict)

    def serialize(self) -> str:
        return json.dumps(asdict(self))


class NodeState(str, Enum):
    """The states that a node can be in rendezvous."""

    INIT = "INIT"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


@dataclass
class RdzvEvent:
    """
    Dataclass to represent any rendezvous event.

    Args:
        name: Event name. (E.g. Current action being performed)
        run_id: The run id of the rendezvous
        message: The message describing the event
        hostname: Hostname of the node
        pid: The process id of the node
        node_state: The state of the node (INIT, RUNNING, SUCCEEDED, FAILED)
        master_endpoint: The master endpoint for the rendezvous store, if known
        rank: The rank of the node, if known
        local_id: The local_id of the node, if defined in dynamic_rendezvous.py
        error_trace: Error stack trace, if this is an error event.
    """

    name: str
    run_id: str
    message: str
    hostname: str
    pid: int
    node_state: NodeState
    master_endpoint: str = ""
    rank: Optional[int] = None
    local_id: Optional[int] = None
    error_trace: str = ""

    def __str__(self):
        return self.serialize()

    @staticmethod
    def deserialize(data: Union[str, "RdzvEvent"]) -> "RdzvEvent":
        if isinstance(data, RdzvEvent):
            return data
        if isinstance(data, str):
            data_dict = json.loads(data)
        data_dict["node_state"] = NodeState[data_dict["node_state"]]  # type: ignore[possibly-undefined]
        # pyrefly: ignore  # unbound-name
        return RdzvEvent(**data_dict)

    def serialize(self) -> str:
        return json.dumps(asdict(self))
