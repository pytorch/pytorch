from collections import deque
import re


class Pattern:

    def match(self, event):
        raise NotImplementedError


class NamePattern(Pattern):

    def __init__(self, name):
        self.name = name

    def match(self, event):
        return re.search(self.name, event.name()) is not None


def and_(*args):

    class CompositePattern(Pattern):

        def match(self, event):
            return all(pattern.match(event) for pattern in args)

    return CompositePattern()


def or_(*args):

    class CompositePattern(Pattern):

        def match(self, event):
            return any(pattern.match(event) for pattern in args)

    return CompositePattern()


def EventTreeDFS(event_tree):
    stack = deque(event_tree)
    while stack:
        curr_event = stack.pop()
        yield curr_event
        for child_event in curr_event.children:
            stack.append(child_event)


# TODO: Think about How can we reuse the same pattern for multiple events?
def find_anti_pattern(prof, anti_pattern):
    for event in EventTreeDFS(
            prof.kineto_results.experimental_event_tree()):
        for pattern, description in anti_pattern:
            if pattern.match(event):
                print(f"{event.name()} {description}")
