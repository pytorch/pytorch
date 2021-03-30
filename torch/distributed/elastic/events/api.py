#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Dict, Union


EventMetadataValue = Union[str, int, float, bool, None]


class EventSource(str, Enum):
    """
    Known identifiers of the event producers.
    """

    AGENT = "AGENT"
    WORKER = "WORKER"


@dataclass
class Event:
    """
    The class represents the generic event that occurs during the torchelastic
    job execution. The event can be any kind of meaningful action.

    Args:
        name: event name.
        source: the event producer, e.g. agent or worker
        timestamp: timestamp in milliseconds when event occured.
        metadata: additional data that is associated with the event.
    """

    name: str
    source: EventSource
    timestamp: int = 0
    metadata: Dict[str, EventMetadataValue] = field(default_factory=dict)

    def __str__(self):
        return self.serialize()

    @staticmethod
    def deserialize(data: Union[str, "Event"]) -> "Event":
        if isinstance(data, Event):
            return data
        if isinstance(data, str):
            data_dict = json.loads(data)
        data_dict["source"] = EventSource[data_dict["source"]]
        return Event(**data_dict)

    def serialize(self) -> str:
        return json.dumps(asdict(self))
