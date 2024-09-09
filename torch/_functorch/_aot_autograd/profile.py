from dataclasses import dataclass
from typing import List


@dataclass
class Event:
    name: str
    duration_sec: float


events: List[Event] = []


def reset() -> None:
    global events
    events.clear()


def add_event(e: Event) -> None:
    global events
    events.append(e)


def get_events() -> List[Event]:
    global events
    return events
