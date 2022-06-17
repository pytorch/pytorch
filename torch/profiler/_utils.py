from collections import deque
from typing import Dict, List
from dataclasses import dataclass
from torch.profiler import DeviceType
from torch.autograd.profiler import profile
import re
import numpy as np

DEBUG = False


@dataclass
class EventMetrics:
    duration_time_ns: int = 0
    self_time_ns: int = 0
    idle_time_ns: int = 0

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
        for interval in intervals:
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
        self.qd_list = self.compute_idle_time_and_qd()

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

    def compute_idle_time_and_qd(self):
        '''
        Computes event's idle time. Idle time is defined as the time when the CUDA kernel queue depth is 0.
        It also return a Time series of the queue depth data.
        qd = cuda kernel queue depth
        '''
        assert (self.profile.kineto_results is not None)
        event_list = self.profile.kineto_results.events()

        def is_cuda_launch_kernel(e):
            # TODO: find a better way to identify cudaLaunchKernel
            return e.name() == "cudaLaunchKernel"

        def is_cuda_kernel(e):
            # TODO: find a better way to identify CUDA Kernel
            return e.device_type() == DeviceType.CUDA and "mem" not in e.name(
            ).lower()

        # Record All the idle intervals
        curr_qd = 0
        idle_interval = []
        qd_list = []
        idle_start = 0
        cuda_kernel_events = [
            event for event in event_list
            if is_cuda_launch_kernel(event) or is_cuda_kernel(event)
        ]
        cuda_kernel_events.sort(key=lambda e: e.start_us())

        for curr_event, next_event in zip(cuda_kernel_events,
                                          cuda_kernel_events[1:]):
            if (is_cuda_launch_kernel(curr_event)):
                if (curr_qd == 0):
                    if idle_start is not None:
                        idle_interval.append(
                            Interval(idle_start * 1000,
                                     curr_event.start_us() * 1000))
                curr_qd += 1
            if (is_cuda_kernel(curr_event)):
                curr_qd -= 1
                if (curr_qd == 0):
                    idle_start = curr_event.start_us()

            qd_list.append(
                Interval(curr_event.start_us() * 1000,
                         (next_event.start_us() + next_event.duration_us()) *
                         1000, curr_qd))

        event_list = [e.event for e in self.metrics.keys()]
        for event in event_list:
            self.metrics[EventKey(event)].idle_time_ns = EventKey(
                event).intervals_overlap(idle_interval)
        return qd_list

    def rank_events(self, length):
        '''
        Filter and Rank the events based on some heuristics:
        1) Events that are in the falling phase of the queue depth.
        2) Events that have a high idle_time, self_time difference.

        Parameters:
            length: The number of events to return.
        '''

        # Find the interval when qd is falling to 0
        qd_list = list(reversed(self.qd_list))
        qd_values = [e.queue_depth for e in qd_list]

        decrease_interval = []
        for i in range(len(qd_values) - 1):
            if qd_values[i] == 0:
                # Find next zero and if the max value between them exceeds
                # the threshold, then we have a falling interval
                for j in range(i + 1, len(qd_values)):
                    if qd_values[j] == 0:
                        peak_idx = argmax(qd_values, start=i + 1, end=j)
                        if peak_idx is None:
                            continue
                        # check for threshold
                        if qd_values[peak_idx] - qd_values[i] > 8:
                            decrease_interval.append(
                                Interval(qd_list[peak_idx].start,
                                         qd_list[i].start))
                            i = peak_idx

        # Filter out events that are not in the decrease interval
        event_list = [
            event for event in self.metrics.keys()
            if event.intervals_overlap(decrease_interval)
        ]

        self_time = np.array(
            [self.metrics[event].self_time_ns for event in event_list])
        idle_time = np.array(
            [self.metrics[event].idle_time_ns for event in event_list])

        normalized_gain = (idle_time - np.mean(idle_time)) / np.std(idle_time)
        normalized_self = (self_time - np.mean(self_time)) / np.std(self_time)
        heuristic_score_list = np.abs(0.4 * normalized_gain - normalized_self)

        # Sort events by heuristic
        event_list = [
            event for _, event in sorted(zip(heuristic_score_list, event_list),
                                         key=lambda x: x[0],
                                         reverse=True)
        ]
        event_list = event_list[:length]

        def plot_analysis_graph(filepath):
            import matplotlib.pyplot as plt
            plt.plot([x.start for x in self.qd_list],
                     [x for x in qd_values[::-1]])
            for interval in decrease_interval:
                plt.axvspan(interval.start,
                            interval.end,
                            color='orange',
                            alpha=0.5)
            for i, event in enumerate(event_list):
                y_loc = max(qd_values) - i * 20
                plt.plot([event.event.start_time_ns, event.event.end_time_ns],
                         [y_loc, y_loc],
                         label=event.event.name())
            plt.legend(bbox_to_anchor=(1, 0.5), loc='center left')
            plt.xlabel("Time (ns)")
            plt.ylabel("Queue Depth")
            plt.savefig(filepath, bbox_inches='tight')

        if DEBUG:
            plot_analysis_graph("qd.png")

        return event_list

    def get_optimizable_events(self, length: int = 1):
        event_list = self.rank_events(length)
        if len(event_list) == 0:
            print("No events to optimize")
            return []
        print("Optimizable events:")
        for event in event_list:
            print(f"""{'-'*80}
Event:                {event}
Source code location: {source_code_location(event.event)}
Percentage idle time: {self.metrics[event].fraction_idle_time() * 100:.2f}%
{'-'*80}""")
        return event_list


def index_of_first_match(seq, predicate, start=0, end=None):
    for i, x in enumerate(seq[start:end]):
        if predicate(x):
            return i + start
    return None


def argmax(seq, key=lambda x: x, start=0, end=None):
    seq = seq[start:end]
    if len(seq) == 0:
        return None
    return seq.index(max(seq, key=key)) + start


def source_code_location(event):
    while (event is not None):
        match = re.search("\.py\(.*\)", event.name())
        if (match is None):
            event = event.parent
            continue
        return event.name()
    return "No source code location found"
