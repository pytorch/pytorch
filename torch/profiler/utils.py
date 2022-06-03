from hypothesis import event
import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
from collections import deque


class EventKey:
    def __init__(self, event):
        self.event = event

    def __hash__(self):
        return hash(self.event.id)

    def __repr__(self):
        return self.event.name()


def compute_self_time(event_tree):
    '''
    return a dictionary of EventKey to event's self time (total time - time in child ops).

        Parameters:
            event_tree: Profiler's kineto_results.experimental_event_tree
        
        Returns:
            result_dict: dictionary of EventKey to event's self time
    '''
    stack = deque()
    for event in event_tree:
        stack.append(event)

    result_dict = dict()
    # standard iterating dfs
    while stack:
        curr_event = stack.pop()
        self_time = curr_event.duration_time_ns
        if curr_event.children:
            for child_event in curr_event.children:
                self_time - child_event.duration_time_ns
                stack.append(child_event)
        result_dict[EventKey(curr_event)] = self_time

    return result_dict




if __name__ == '__main__':
    model = models.resnet18()
    inputs = torch.randn(5, 3, 224, 224)

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            model(inputs)
    print(compute_self_time(prof.profiler.kineto_results.experimental_event_tree()))
