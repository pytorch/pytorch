from collections import deque, namedtuple
from dataclasses import dataclass
from torch.profiler import DeviceType
from torch.autograd.profiler import profile
import re
import math
import matplotlib.pyplot as plt
import pandas as pd


class EventKey:
    def __init__(self, event):
        self.event = event

    def __hash__(self):
        return hash(self.event.id)

    def __eq__(self, other):
        return self.event.id == other.event.id

    def __repr__(self):
        return f"<{self.event.name()} id={self.event.correlation_id}>"
    
    def in_intervals(self, intervals):
        for interval in intervals:
            if self.event.start_time_ns >= interval[1] or self.event.end_time_ns <= interval[0]:
                return False
        return True


@dataclass
class EventMetrics:
    duration_time_ns: int = 0
    self_time_ns: int = 0
    idle_time_ns: int = 0

    def fraction_idle_time(self):
        if self.duration_time_ns == 0:
            return 0.0
        return self.idle_time_ns / self.duration_time_ns


class BasicEvaluation:
    def __init__(self, prof: profile):
        self.profile = prof
        self.metrics = dict()
        self.compute_self_time(self.profile, self.metrics)
        self.qd_list = self.compute_idle_time(self.profile, self.metrics)

    def score(self, event: EventKey):
        return self.metrics[event].duration_time_ns + self.metrics[event].idle_time_ns

    def rank_events(self):
        # Find the interval when qd is falling 0
        # plt.plot([x.queue_depth for x in self.qd_list])
        # plt.savefig("qd.png")
        increase_interval = []
        is_increasing = False
        qd_ma = pd.Series([x.queue_depth for x in self.qd_list]).rolling(30).mean().to_list()
        for i, (q0, q1) in enumerate(zip(qd_ma, qd_ma[1:])):
            if q1 > q0:
                if not is_increasing:
                    decrease_start = self.qd_list[i].start
                    is_increasing = True
            else:
                if is_increasing:
                    increase_interval.append((decrease_start, self.qd_list[i].end))
                    is_increasing = False
        # print(increase_interval)
        event_list = self.metrics.keys()
        event_list = [e for e in event_list if not e.in_intervals(increase_interval)]
        event_list = sorted(event_list, key=lambda e: self.score(e), reverse=True)
        return event_list

    def get_optimizable_events(self, length: int = 1):
        event_list = self.rank_events()
        if len(event_list) < length:
            length = len(event_list)
        event_list = event_list[:length]
        print("Optimizable events:")
        for event in event_list:
            print(
f"""--------------------------------------------------------------------------------
Event:                {event}
Source code location: {source_code_location(event.event)}
Percentage idle time: {self.metrics[event].fraction_idle_time() * 100:.2f}%
Heuristic score:      {self.score(event)}
--------------------------------------------------------------------------------"""
            )
        return event_list
    
    def compute_self_time(self, prof: profile, metrics: dict[EventKey, EventMetrics]):
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


    def compute_idle_time(self, prof: profile, metrics: dict[EventKey, EventMetrics]):
        assert(prof.kineto_results is not None)
        event_list = prof.kineto_results.events()

        def is_cuda_launch_kernel(e):
            # TODO: find a better way to identify cudaLaunchKernel
            return e.name() == "cudaLaunchKernel"

        def is_cuda_kernel(e):
            # TODO: find a better way to identify CUDA Kernel
            return e.device_type() == DeviceType.CUDA and "mem" not in e.name().lower()

        # Record All the idle intervals
        curr_qd = 0
        qd_data = namedtuple("qd_data", "queue_depth start end")
        idle_interval = []
        qd_list = []
        idle_start = 0
        cuda_kernel_events = [event for event in event_list if is_cuda_launch_kernel(event) or is_cuda_kernel(event)]
        cuda_kernel_events.sort(key=lambda e: e.start_us())

        for curr_event, next_event in zip(cuda_kernel_events, cuda_kernel_events[1:]):
            if (is_cuda_launch_kernel(curr_event)):
                if (curr_qd == 0):
                    idle_start = curr_event.start_us()
                curr_qd += 1
            if (is_cuda_kernel(curr_event)):
                curr_qd -= 1
                if (curr_qd == 0):
                    idle_interval.append((idle_start, curr_event.start_us()))
            qd_list.append(qd_data(curr_qd, curr_event.start_us() * 1000, (next_event.start_us() + next_event.duration_us()) * 1000))
        
        #plt.plot([x.start for x in qd_list], [x.queue_depth for x in qd_list])
        #plt.savefig("queue_depth.png")
        event_list = [e.event for e in metrics.keys()]
        for event in event_list:
            idle_time = 0
            for interval in idle_interval:
                overlap_start = max(event.start_time_ns, interval[0] * 1000)
                overlap_end = min(event.end_time_ns, interval[1] * 1000)

                if overlap_start < overlap_end:
                    idle_time += overlap_end - overlap_start
            metrics[EventKey(event)].idle_time_ns = idle_time
        return qd_list
        

class NaiveEvaluation(BasicEvaluation):
    def __init__(self, prof: profile):
        super().__init__(prof)

def source_code_location(event):
    while(event is not None):
        match = re.search(".py(.*)", event.name())
        if (match is None):
            event = event.parent
            continue
        return event.name()
    return "No source code location found"