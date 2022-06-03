<<<<<<< HEAD
from collections import deque
from dataclasses import dataclass
from torch.profiler import DeviceType
=======
from hypothesis import event
import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
from collections import deque
>>>>>>> Computed self time for events


class EventKey:
    def __init__(self, event):
        self.event = event

    def __hash__(self):
        return hash(self.event.id)

<<<<<<< HEAD
    def __eq__(self, other):
        return self.event.id == other.event.id

    def __repr__(self):
        return f"<{self.event.name()} id={self.event.correlation_id}>"


@dataclass
class EventMetrics:
    duration_time_ns: int = 0
    self_time_ns: int = 0
    idle_time_ns: int = 0
    def percent_idle_time(self):
        if self.duration_time_ns == 0:
            return 0
        return self.idle_time_ns / self.duration_time_ns


def compute_event_metrics(prof) -> dict[EventKey, EventMetrics]:
    metrics = dict()
    compute_self_time(prof, metrics)
    compute_idle_time(prof, metrics)
    return metrics


def compute_self_time(prof, metrics):
    '''
    Computes event's self time (total time - time in child ops).

        Parameters:
            event_tree: Profiler's kineto_results.experimental_event_tree
    '''
    stack = deque()
    event_tree = prof.profiler.kineto_results.experimental_event_tree()
    for event in event_tree:
        stack.append(event)

=======
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
>>>>>>> Computed self time for events
    # standard iterating dfs
    while stack:
        curr_event = stack.pop()
        self_time = curr_event.duration_time_ns
        if curr_event.children:
            for child_event in curr_event.children:
                self_time - child_event.duration_time_ns
                stack.append(child_event)
<<<<<<< HEAD
        if EventKey(curr_event) in metrics:
            metrics[EventKey(curr_event)].self_time_us = self_time
        else:
            metrics[EventKey(curr_event)] = EventMetrics(self_time_ns=self_time)
        metrics[EventKey(curr_event)].duration_time_ns = curr_event.duration_time_ns


def compute_idle_time(prof, metrics):
    event_tree = prof.profiler.kineto_results.experimental_event_tree()
    event_list = prof.profiler.kineto_results.events()

    def is_cuda_launch_kernel(e):
        return e.name() == "cudaLaunchKernel"
    
    def is_cuda_kernel(e):
        # TODO: find a better way to identify cudaLaunchKernel
        return e.device_type() == DeviceType.CUDA and e.name() != "[memory]" and e.name() != "Memset (Device)"

    # Record All the idle intervals
    queue_depth = 0
    idle_interval = []
    cuda_kernel_events = [event for event in event_list if is_cuda_launch_kernel(event) or is_cuda_kernel(event)]
    cuda_kernel_events.sort(key=lambda e: e.start_us())

    if len(cuda_kernel_events) == 1:
        return
    for i in range(1, len(cuda_kernel_events)):
        prev_event = cuda_kernel_events[i - 1]
        curr_event = cuda_kernel_events[i]
        if (is_cuda_launch_kernel(prev_event)):
            queue_depth += 1
        if (is_cuda_kernel(prev_event)):
            queue_depth -= 1
        if (prev_event.start_us() + prev_event.duration_us()) < curr_event.start_us() and queue_depth == 0:
            idle_interval.append((prev_event.start_us() + prev_event.duration_us(), curr_event.start_us()))

    # For every event, compute the absolute idle time and the percentage idle time
    # idle_interval Seems correct
    event_list = [e.event for e in metrics.keys()]
    for event in event_list:
        idle_time = 0
        for interval in idle_interval:
            overlap_start = max(event.start_time_ns, interval[0] * 1000)
            overlap_end = min(event.end_time_ns, interval[1] * 1000)

            if overlap_start < overlap_end:
                idle_time += overlap_end - overlap_start
        metrics[EventKey(event)].idle_time_ns = idle_time

def get_optimizable_events(prof):
    metrics = compute_event_metrics(prof)
    
    # Compute a list of events that that long idle time

    # Print the list of events in human-friendly format

=======
        result_dict[EventKey(curr_event)] = self_time

    return result_dict




if __name__ == '__main__':
    model = models.resnet18()
    inputs = torch.randn(5, 3, 224, 224)

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            model(inputs)
    print(compute_self_time(prof.profiler.kineto_results.experimental_event_tree()))
>>>>>>> Computed self time for events
