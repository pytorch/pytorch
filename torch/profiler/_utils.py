from collections import deque
from typing import Dict
from dataclasses import dataclass
from torch.autograd.profiler import profile


class EventKey:

    def __init__(self, event):
        self.event = event

    def __hash__(self):
        return hash(self.event.id)

    def __eq__(self, other):
        return self.event.id == other.event.id

    def __repr__(self):
        return f"<{self.event.name()} id={self.event.correlation_id}>"


@dataclass
class EventMetrics:
    self_time_ns: int = 0


def compute_self_time(prof: profile, metrics: Dict[EventKey, EventMetrics]):
    '''
    Computes event's self time(total time - time in child ops).

        Parameters:
            prof: autograd profile object
            metrics: dictionary of event key and event metrics
    '''
    assert (prof.kineto_results is not None)
    stack = deque(prof.kineto_results.experimental_event_tree())

    # standard iterating dfs
    while stack:
        curr_event = stack.pop()
        self_time = curr_event.duration_time_ns
        for child_event in curr_event.children:
            self_time -= child_event.duration_time_ns
            stack.append(child_event)

        assert EventKey(
            curr_event
        ) not in metrics, f"Duplicate id: {curr_event.id}, {curr_event.name()}"
        metrics[EventKey(curr_event)] = EventMetrics(self_time_ns=self_time)
