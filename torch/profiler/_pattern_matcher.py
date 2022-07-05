from collections import deque
import re
from typing import Dict, List

from torch.profiler import profile
from torch._C._autograd import _ProfilerEvent


class Pattern:
    '''
    Base class for all patterns, subclass this class and implement match()
    to define custom patterns.
    '''
    def __init__(self, prof: profile):
        self.prof = prof
        self.description = "Please specify a description for pattern"
        assert prof.profiler is not None and prof.profiler.kineto_results is not None
        self.event_tree = prof.profiler.kineto_results.experimental_event_tree(
        )
        self.tid_root: Dict[int, List[_ProfilerEvent]] = {}
        for event in self.event_tree:
            self.tid_root.setdefault(event.start_tid, []).append(event)

    def report(self, event: _ProfilerEvent):
        msg = f"{self.description}\n{source_code_location(event)}"
        return msg

    def match(self, event: _ProfilerEvent):
        '''
        Return True if the event matches the pattern.
        This method should be overriden in subclass.
        '''
        raise NotImplementedError

    def matched_events(self):
        matched_events = []
        for event in eventTreeDFS(self.event_tree):
            if self.match(event):
                matched_events.append(event)
        return matched_events

    def root_of(self, event: _ProfilerEvent):
        while event.parent:
            event = event.parent
        return event

    def siblings_of(self, event: _ProfilerEvent):
        if event.parent:
            children = event.parent.children
        else:
            children = self.tid_root[event.start_tid]
        index = children.index(event)
        return children[:index], children[index + 1:]

    def next_of(self, event: _ProfilerEvent):
        _, next_events = self.siblings_of(event)
        return next_events[0] if next_events else None

    def prev_of(self, event: _ProfilerEvent):
        prev_events, _ = self.siblings_of(event)
        return prev_events[-1] if prev_events else None


# Patterns


class NamePattern(Pattern):

    def __init__(self, prof: profile, name: str):
        super().__init__(prof)
        self.description = f"Matched Name Event: {name}"
        self.name = name

    def match(self, event: _ProfilerEvent):
        return re.search(self.name, event.name()) is not None


def eventTreeDFS(event_tree):
    stack = deque(event_tree)
    while stack:
        curr_event = stack.pop()
        yield curr_event
        for child_event in curr_event.children:
            stack.append(child_event)


def source_code_location(event: _ProfilerEvent):
    while (event is not None):
        match = re.search(r"\.py\(.*\)", event.name())
        if (match is None):
            event = event.parent
            continue
        return event.name()
    return "No source code location found"
