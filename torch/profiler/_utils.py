from collections import deque
from dataclasses import dataclass
from typing import Dict, List

from torch.profiler import DeviceType
from torch.autograd.profiler import profile
from torch.autograd import _KinetoEvent


@dataclass
class EventMetrics:
    duration_time_ns: int = 0
    self_time_ns: int = 0
    idle_time_ns: int = 0

    @property
    def fraction_idle_time(self):
        if self.duration_time_ns == 0:
            return 0.0
        return self.idle_time_ns / self.duration_time_ns


@dataclass
class Interval:
    start: int
    end: int
    queue_depth: int = 0


class EventKey:

    def __init__(self, event):
        self.event = event

    def __hash__(self):
        return hash(self.event.id)

    def __eq__(self, other):
        return self.event.id == other.event.id

    def __repr__(self):
        return f"<{self.event.name()} id={self.event.correlation_id}>"

    def intervals_overlap(self, intervals: List[Interval]):
        overlap_time = 0
        intervals = sorted(intervals, key=lambda x: x.start)
        for i, interval in enumerate(intervals):
            if i + 1 < len(intervals):
                assert interval.end <= intervals[
                    i + 1].start, "Intervals must be disjoint"
            overlap_start = max(self.event.start_time_ns, interval.start)
            overlap_end = min(self.event.end_time_ns, interval.end)

            if overlap_start < overlap_end:
                overlap_time += overlap_end - overlap_start
        return overlap_time


class BasicEvaluation:

    def __init__(self, prof: profile):
        self.profile = prof
        self.metrics: Dict[EventKey, EventMetrics] = {}
        self.compute_self_time()

    def compute_self_time(self):
        '''
        Computes event's self time(total time - time in child ops).
        '''
        assert (self.profile.kineto_results is not None)
        stack = deque(self.profile.kineto_results.experimental_event_tree())

        # standard iterating dfs
        while stack:
            curr_event = stack.pop()
            self_time = curr_event.duration_time_ns
            for child_event in curr_event.children:
                self_time -= child_event.duration_time_ns
                stack.append(child_event)

            assert EventKey(
                curr_event
            ) not in self.metrics, f"Duplicate id: {curr_event.id}, {curr_event.name()}"
            self.metrics[EventKey(curr_event)] = EventMetrics(
                self_time_ns=self_time)
            self.metrics[EventKey(
                curr_event)].duration_time_ns = curr_event.duration_time_ns

    def compute_queue_depth(self):
        '''
        Computes event's idle time. Idle time is defined as the time when the CUDA kernel queue depth is 0.
        It also return a Time series of the queue depth data.
        qd = cuda kernel queue depth
        '''
        assert (self.profile.kineto_results is not None)
        cuda_event_list = self.profile.kineto_results.events()

        def is_cuda_launch_kernel(e):
            # TODO: find a better way to identify cudaLaunchKernel
            return e.name() == "cudaLaunchKernel"

        def is_cuda_kernel(e):
            # TODO: find a better way to identify CUDA Kernel
            return e.device_type() == DeviceType.CUDA and "mem" not in e.name(
            ).lower()

        # Record All the idle intervals
        idle_interval: List[Interval] = []
        queue_depth_list: List[Interval] = []

        cuda_launch_events = sorted(
            (e for e in cuda_event_list if is_cuda_launch_kernel(e)),
            key=lambda x: x.start_us())
        cuda_kernel_events = sorted(
            (e for e in cuda_event_list if is_cuda_kernel(e)),
            key=lambda x: x.start_us())

        kernel_mapping: Dict[_KinetoEvent, int] = {}
        last_mapped_kernel = 0
        for cuda_launch_event in cuda_launch_events:
            index = index_of_first_match(
                cuda_kernel_events,
                lambda x: x.linked_correlation_id(
                ) == cuda_launch_event.linked_correlation_id(),
                start=last_mapped_kernel)
            kernel_mapping[cuda_launch_event] = index
            last_mapped_kernel = index if index is not None else last_mapped_kernel

        current_kernel_index = -1
        spawned_kernel_index = None
        for cuda_launch_event in cuda_launch_events:
            # Find latest cuda kernel event
            while (current_kernel_index + 1 < len(cuda_kernel_events) and
                   cuda_kernel_events[current_kernel_index + 1].start_us() +
                   cuda_kernel_events[current_kernel_index + 1].duration_us() <
                   cuda_launch_event.start_us() +
                   cuda_launch_event.duration_us()):
                current_kernel_index += 1

            # Find current spawned cuda kernel event
            spawned_kernel_index = kernel_mapping[cuda_launch_event]
            if spawned_kernel_index is None:
                current_queue_depth = 0
            else:
                current_queue_depth = spawned_kernel_index - current_kernel_index

            queue_depth_list.append(
                Interval(
                    cuda_launch_event.start_us(),
                    cuda_launch_event.start_us() +
                    cuda_launch_event.duration_us(), current_queue_depth))

        event_list = [e.event for e in self.metrics.keys()]
        for event in event_list:
            self.metrics[EventKey(event)].idle_time_ns = EventKey(
                event).intervals_overlap(idle_interval)

        return queue_depth_list


def index_of_first_match(seq, predicate, start=0, end=None):
    if end is None or end >= len(seq):
        end = len(seq)
    for i in range(start, end):
        if predicate(seq[i]):
            return i
    return None
