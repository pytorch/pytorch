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


class ExtraCUDACopyPattern(Pattern):
    '''
    This pattern identifies if we creates a constant tensor on CPU and immediately moves it to GPU.
    example: torch.zeros((100, 100)).to("cuda")

    Pattern:
    build-in method                 |build-in method
        ...                         |    aten::to
            aten::fill_/aten::zero_ |

    Algorithm:
    We start at node aten::to, go parent events' previous events,
    and check if we have a aten::fill_/aten::zero_ as we keep going down the tree.
    We always select the last child in the children list when we go down the tree.
    If at any step we failed, it is not a match.
    '''
    def __init__(self, prof: profile):
        assert prof.with_stack
        super().__init__(prof)
        self.description = "Filled a CPU tensor and immediately moved it to GPU. Please initalize it on GPU."

    def match(self, event):
        # TODO: We should also check tensor identities
        if event.name() != "aten::to":
            return False
        # Up one level
        event = event.parent
        if event is None:
            return False
        # Check if we have a aten::fill_ in previous leaf
        event = self.prev_of(event)
        if event is None:
            return False
        while event.children:
            event = event.children[-1]
            # aten::zero_ is a special optimzation case where fill_ is not called
            if event.name() == "aten::fill_" or event.name() == "aten::zero_":
                return True
        return event.name() == "aten::fill_" or event.name() == "aten::zero_"


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


def report_all_anti_patterns(prof):
    anti_patterns = [ExtraCUDACopyPattern(prof)]
    for anti_pattern in anti_patterns:
        for event in anti_pattern.matched_events():
            print(anti_pattern.report(event))
