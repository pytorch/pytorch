# Defined in torch/csrc/monitor/python_init.cpp

from typing import List, Dict, Callable, Union
from enum import Enum
import datetime

class Aggregation(Enum):
    VALUE = ...
    MEAN = ...
    COUNT = ...
    SUM = ...
    MAX = ...
    MIN = ...

class Stat:
    name: str
    count: int
    def add(self, v: float) -> None: ...
    def get(self) -> Dict[Aggregation, float]: ...

class IntervalStat(Stat):
    def __init__(
        self,
        name: str,
        aggregations: List[Aggregation],
        window_size: datetime.timedelta,
    ) -> None: ...

class FixedCountStat(Stat):
    def __init__(
        self, name: str, aggregations: List[Aggregation], window_size: int
    ) -> None: ...

class Event:
    name: str
    timestamp: datetime.datetime
    data: Dict[str, Union[int, float, bool, str]]
    def __init__(
        self,
        name: str,
        timestamp: datetime.datetime,
        data: Dict[str, Union[int, float, bool, str]],
    ) -> None: ...

def log_event(e: Event) -> None: ...

class PythonEventHandler: ...

def register_event_handler(handler: Callable[[Event], None]) -> PythonEventHandler: ...
def unregister_event_handler(handle: PythonEventHandler) -> None: ...
