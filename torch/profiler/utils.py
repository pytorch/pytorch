from collections import deque
from dataclasses import dataclass
from torch.profiler import DeviceType
from torch.autograd.profiler import profile
import re


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
    duration_time_ns: int = 0
    self_time_ns: int = 0
    idle_time_ns: int = 0

    def fraction_idle_time(self):
        if self.duration_time_ns == 0:
            return 0.0
        return self.idle_time_ns / self.duration_time_ns

    # heuristic to determine which event is more impactful and optimizable
    def score(self):
        return self.fraction_idle_time()


def compute_event_metrics(prof: profile):
    metrics = dict()
    compute_self_time(prof, metrics)
    compute_idle_time(prof, metrics)
    return metrics


def compute_self_time(prof: profile, metrics: dict[EventKey, EventMetrics]):
    '''
    Computes event's self time(total time - time in child ops).

        Parameters:
            prof: profile object that we call kineto_results.experimental_event_tree() on
            metrics: dictionary of event key and event metrics
    '''
    assert(prof.kineto_results is not None)
    stack = deque(prof.kineto_results.experimental_event_tree())

    # standard iterating dfs
    while stack:
        curr_event = stack.pop()
        self_time = curr_event.duration_time_ns
        for child_event in curr_event.children:
            self_time - child_event.duration_time_ns
            stack.append(child_event)
        
        assert EventKey(curr_event) not in metrics, f"Duplicate id: {curr_event.id}, {curr_event.name()}"
        metrics[EventKey(curr_event)] = EventMetrics(self_time_ns=self_time)
        metrics[EventKey(curr_event)].duration_time_ns = curr_event.duration_time_ns


def compute_idle_time(prof: profile, metrics: dict[EventKey, EventMetrics]):
    assert(prof.kineto_results is not None)
    event_list = prof.kineto_results.events()

    def is_cuda_launch_kernel(e):
        # TODO: find a better way to identify cudaLaunchKernel
        return e.name() == "cudaLaunchKernel"

    def is_cuda_kernel(e):
        # TODO: find a better way to identify CUDA Kernel
        return e.device_type() == DeviceType.CUDA and e.name() != "[memory]" and e.name() != "Memset (Device)"

    # Record All the idle intervals
    queue_depth = 0
    idle_interval = []
    cuda_kernel_events = [event for event in event_list if is_cuda_launch_kernel(event) or is_cuda_kernel(event)]
    cuda_kernel_events.sort(key=lambda e: e.start_us())

    for prev_event, curr_event in zip(cuda_kernel_events, cuda_kernel_events[1:]):
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


def get_optimizable_events(prof: profile, length: int = 1):
    metrics = compute_event_metrics(prof)

    # Compute a list of events that that long idle time
    # H score: h(self_time, average fraction_idle_time, )
    event_list = [e for e in metrics.keys()]
    event_list.sort(key=lambda e: metrics[e].score(), reverse=True)
    event_list = event_list[:length]
    # Print the list of events in human-friendly format
    print("Optimizable events:")
    for event in event_list:
        print(f"Event:                {event}\nSource code location: {source_code_location(event.event)}")
    return event_list

def source_code_location(event):
    while(event is not None):
        match = re.search(".py(.*)", event.name())
        if (match is None):
            event = event.parent
            continue
        return event.name()
    return "No source code location found"
