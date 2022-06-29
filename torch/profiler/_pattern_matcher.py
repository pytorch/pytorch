from collections import deque
import re

from torch.profiler import profile


class Pattern:

    def __init__(self, prof: profile):
        self.prof = prof
        self.description = "Please specify a description for pattern"
        assert prof.profiler is not None and prof.profiler.kineto_results is not None
        self.event_tree = prof.profiler.kineto_results.experimental_event_tree(
        )

    def report(self, event):
        msg = f"{event.name()}\n{self.description}\n{source_code_location(event)}"
        return msg

    def match(self, event):
        raise NotImplementedError

    def matched_events(self):
        matched_events = []
        for event in EventTreeDFS(self.event_tree):
            if self.match(event):
                matched_events.append(event)
        return matched_events

    def root_of(self, event):
        while event.parent:
            event = event.parent
        return event

    def siblings_of(self, event):
        if event.parent:
            return event.parent.children
        else:
            return self.event_tree

    def next_of(self, event):
        siblings = self.siblings_of(event)
        index = siblings.index(event)
        if index + 1 < len(siblings):
            return siblings[index + 1]
        else:
            return None

    def prev_of(self, event):
        siblings = self.siblings_of(event)
        index = siblings.index(event)
        if index - 1 >= 0:
            return siblings[index - 1]
        else:
            return None


# Patterns


class NamePattern(Pattern):

    def __init__(self, prof: profile, name: str):
        super().__init__(prof)
        self.description = f"Matched Name Event: {name}"
        self.name = name

    def match(self, event):
        return re.search(self.name, event.name()) is not None


class ExtraCUDACopyPattern(Pattern):

    def __init__(self, prof: profile):
        assert prof.with_stack
        super().__init__(prof)
        self.description = "Filled a CPU tensor and immediately moved it to GPU. Please initalize it on GPU."

    def match(self, event):
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
            if event.name() == "aten::fill_" or event.name() == "aten::zero_":
                return True
        return event.name() == "aten::fill_" or event.name() == "aten::zero_"


def EventTreeDFS(event_tree):
    stack = deque(event_tree)
    while stack:
        curr_event = stack.pop()
        yield curr_event
        for child_event in curr_event.children:
            stack.append(child_event)


def source_code_location(event):
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
