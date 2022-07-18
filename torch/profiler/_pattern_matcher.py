from collections import deque
import re
from typing import Dict, List, Set

import torch
from torch.profiler import profile
from torch.profiler._utils import index_of_first_match
from torch._C._autograd import (_ProfilerEvent, _ExtraFields_TorchOp,
                                _ExtraFields_Backend, _ExtraFields_Allocation,
                                _ExtraFields_PyCCall, _ExtraFields_PyCall,
                                _EventType)


class Pattern:
    '''
    Base class for all patterns, subclass this class and implement match()
    to define custom patterns.

    In subclass, define description and skip property.
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

    @property
    def skip(self):
        return False

    def report(self, event: _ProfilerEvent):
        msg = f"{event.name()}\n{self.description}\n{source_code_location(event)}"
        return msg

    def eventTreeTraversal(self):
        '''
        Standard DFS traversal of the event tree.
        Override this method to customize the traversal order.
        '''
        stack = deque(self.event_tree)
        while stack:
            curr_event = stack.pop()
            yield curr_event
            for child_event in curr_event.children:
                stack.append(child_event)

    def match(self, event: _ProfilerEvent):
        '''
        Return True if the event matches the pattern.
        This method should be overriden in subclass.
        '''
        raise NotImplementedError

    def matched_events(self):
        if self.skip:
            return []
        matched_events = []
        for event in self.eventTreeTraversal():
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
            aten::fill_/aten::zero_ |        aten::_to_copy

    Algorithm:
    We start at node aten::to, go parent events' previous events,
    and check if we have a aten::fill_/aten::zero_ as we keep going down the tree.
    We always select the last child in the children list when we go down the tree.
    If at any step we failed, it is not a match.
    '''

    def __init__(self, prof: profile):
        super().__init__(prof)
        self.description = "Filled a CPU tensor and immediately moved it to GPU. Please initalize it on GPU."
        self.init_ops = {
            "aten::fill_", "aten::zero_", "aten::normal_", "aten::uniform_"
        }

    @property
    def skip(self):
        return self.prof.with_stack is False

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
            if event.name() in self.init_ops:
                return True
        return event.name() in self.init_ops
        # TODO: Check if tensor is reused


class ForLoopIndexingPattern(Pattern):
    '''
    This pattern identifies if we use a for loop to index a tensor that
    can be vectorized.
    example:
    tensor = torch.empty((100, 100))
    for i in range(100):
        tensor[i] = i

    Pattern:
    aten::select | ... | aten::select | ... (Repeat)

    Algorithm:
    We start at node aten::select, and we check if we can find this alternating patterns.
    We also keep a dictionary to avoid duplicate match in the for loop.
    '''

    def __init__(self, prof: profile):
        super().__init__(prof)
        self.description = "For loop indexing detected. Vectorization recommended."
        self.visited: Set[int] = set()

    def eventTreeTraversal(self):
        '''
        We need to use BFS traversal order to avoid duplicate match.
        '''
        stack = deque(self.event_tree)
        while stack:
            curr_event = stack.popleft()
            yield curr_event
            for child_event in curr_event.children:
                stack.append(child_event)

    def match(self, event: _ProfilerEvent):
        if event.name() != "aten::select":
            return False
        if event.id in self.visited:
            return False
        repeat_count = 1
        _, next = self.siblings_of(event)
        if len(next) <= 1:
            return False

        # Custom event list matching
        def same_ops(list1, list2):
            if len(list1) != len(list2):
                return False
            for op1, op2 in zip(list1, list2):
                if op1.name() != op2.name():
                    return False
            return True

        # Record the ops between two aten::select
        next_select_idx = index_of_first_match(
            next, lambda e: e.name() == "aten::select")
        if next_select_idx is None:
            return False
        indexing_ops = [event] + next[:next_select_idx]
        next = next[len(indexing_ops) - 1:]
        for i in range(0, len(next), len(indexing_ops)):
            if same_ops(indexing_ops, next[i:i + len(indexing_ops)]):
                repeat_count += 1
                self.visited.add(next[i].id)
            else:
                break
        return repeat_count >= 10


class FP32MatMulPattern(Pattern):

    def __init__(self, prof: profile):
        super().__init__(prof)
        self.description = (
            "You are currently using GPU that supports TF32. "
            "Please enable TF32 by setting 'torch.backends.cuda.matmul.allow_tf32 = True'"
        )

    @property
    def skip(self):
        has_tf32 = all(
            int(arch[3:]) >= 80 for arch in torch.cuda.get_arch_list())
        return has_tf32 is False

    def match(self, event: _ProfilerEvent):
        # If we saw this pattern once, we don't need to match it again
        if event_type(event) != _EventType.TorchOp:
            return False
        assert isinstance(event.extra_fields, _ExtraFields_TorchOp)
        if event.name() == "aten::mm":
            if event.extra_fields.allow_tf32_cublas is False:
                return True
        return False

    def report(self, event: _ProfilerEvent):
        return self.description


def source_code_location(event: _ProfilerEvent):
    while event:
        if event_type(event) == _EventType.PyCall or event_type(
                event) == _EventType.PyCCall:
            assert isinstance(event.extra_fields,
                              _ExtraFields_PyCall) or isinstance(
                                  event.extra_fields, _ExtraFields_PyCCall)
            return f"{event.extra_fields.caller.file_name}:{event.extra_fields.caller.line_number}"
        event = event.parent
    return "No source code location found"


def report_all_anti_patterns(prof):
    anti_patterns = [
        ExtraCUDACopyPattern(prof),
        ForLoopIndexingPattern(prof),
        FP32MatMulPattern(prof)
    ]
    reported = set()
    print(f"{'-'*40}TorchTidy Report{'-'*40}")
    for anti_pattern in anti_patterns:
        for event in anti_pattern.matched_events():
            report_msg = anti_pattern.report(event)
            if report_msg not in reported:
                print(report_msg)
                reported.add(report_msg)


def event_type(event: _ProfilerEvent):
    if isinstance(event.extra_fields, _ExtraFields_TorchOp):
        return _EventType.TorchOp
    elif isinstance(event.extra_fields, _ExtraFields_Backend):
        return _EventType.Backend
    elif isinstance(event.extra_fields, _ExtraFields_Allocation):
        return _EventType.Allocation
    elif isinstance(event.extra_fields, _ExtraFields_PyCall):
        return _EventType.PyCall
    elif isinstance(event.extra_fields, _ExtraFields_PyCCall):
        return _EventType.PyCCall
    else:
        raise Exception("Unknown event type")
