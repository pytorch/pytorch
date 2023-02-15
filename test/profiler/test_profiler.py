# Owner(s): ["oncall: profiler"]
import collections
import gc
import io
import json
import os
import re
import tempfile
import textwrap
import threading
import unittest
from unittest.mock import patch
import weakref
from dataclasses import dataclass, field
from typing import List, Optional

import expecttest
import subprocess
import sys
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.utils.data.datapipes as dp
from torch.autograd import (
    _record_function_with_args_enter,
    _record_function_with_args_exit,
)
from torch.autograd.profiler import profile as _profile
from torch.autograd.profiler import KinetoStepTracker
from torch.autograd.profiler_legacy import profile as _profile_legacy
from torch.profiler import (
    _utils,
    DeviceType,
    ExecutionGraphObserver,
    kineto_available,
    profile,
    ProfilerAction,
    ProfilerActivity,
    record_function,
    supported_activities,
)
from torch._C._profiler import _TensorMetadata
from torch.profiler._pattern_matcher import (
    Conv2dBiasFollowedByBatchNorm2dPattern,
    ExtraCUDACopyPattern,
    ForLoopIndexingPattern,
    FP32MatMulPattern,
    GradNotSetToNonePattern,
    MatMulDimInFP16Pattern,
    NamePattern,
    OptimizerSingleTensorPattern,
    Pattern,
    report_all_anti_patterns,
    SynchronizedDataLoaderPattern,
)
from torch.testing._internal.common_cuda import TEST_MULTIGPU
from torch.testing._internal.common_device_type import skipCUDAVersionIn
from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TemporaryDirectoryName,
    TemporaryFileName,
    TEST_WITH_ASAN,
    TEST_WITH_CROSSREF,
    TEST_WITH_ROCM,
    TestCase,
)

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
import pickle

from torch._C._profiler import _ExperimentalConfig, _ExtraFields_PyCall


@instantiate_parametrized_tests
class TestProfiler(TestCase):


    @parametrize(
        "name,thread_spec",
        {
            "basic": ((False, False),),
            #"multiple_preexisting": ((False, False), ) * 2,
            #"open_in_scope": ((True, False),),
            #"close_in_scope": ((False, True),),
#            "complex": (
#                # Large number of background threads
#                (False, False),
#                (False, False),
#                (False, False),
#                (False, False),
#
#                # some of which finish during profiling
#                (False, True),
#                (False, True),
#
#                # And the profiled section is also multithreaded
#                (True, False),
#                (True, True),
#            ),
        }.items(),
        name_fn=lambda name, thread_spec: name
    )
    # @parametrize("work_in_main_thread", [True, False])
    @parametrize("work_in_main_thread", [True])
    def test_source_multithreaded(self, name, thread_spec, work_in_main_thread):
        """Test various threading configurations.

        `thread_spec` is a Tuple[Tuple[bool, bool], ...] where each pair is a
        thread. The first bool indicates if the thread should be started under
        the profiler context and the second is if it should be joined under the
        profiler context.
        """

        timeout = 15
        num_threads = len(thread_spec) + 1  # Main thread
        print(f"=== NUM: {num_threads}")
        start_barrier = threading.Barrier(num_threads, timeout=timeout)
        end_barrier = threading.Barrier(num_threads, timeout=timeout)

        class Task(threading.Thread):

            def __init__(self):
                self._end_gate = threading.Event()
                super().__init__(daemon=True)
                self.start()
                self.finished = False

            def run(self):
                self._run(self._end_gate)

            def release(self):
                self._end_gate.set()

            @staticmethod
            def _run(end_gate=None):

                def known_preexisting_function():
                    start_barrier.wait()

                # Fixed point that we can use to test capture of functions
                # which are already running when profiling is enabled.
                known_preexisting_function()

                model = torch.nn.Sequential(
                    torch.nn.Linear(10, 10),
                    torch.nn.ReLU(),
                )

                def invoked_during_run():
                    pass

                invoked_during_run()

                _ = model(torch.rand(4, 10))
                end_barrier.wait()

                if end_gate is not None:
                    end_gate.wait(timeout=timeout)

        threads = {}

        def add_threads(context: bool):
            for idx, (start_under_profiler, _) in enumerate(thread_spec):
                if start_under_profiler == context:
                    assert idx not in threads
                    threads[idx] = Task()

        def join_threads(context: bool):
            for idx, (_, end_under_profiler) in enumerate(thread_spec):
                if end_under_profiler == context:
                    threads[idx].release()

            for idx, (_, end_under_profiler) in enumerate(thread_spec):
                t = threads[idx]
                if end_under_profiler == context:
                    t.join(timeout=timeout)

        try:
            add_threads(False)
            with torch.profiler.profile(with_stack=True) as prof:
                # Threads added while the profiler are running will not be observed
                # since there is no way to hook into Python's thread start call to
                # register the observer. These are here purely to verify safety.
                add_threads(True)

                if work_in_main_thread:
                    Task._run()
                else:
                    start_barrier.wait()
                    end_barrier.wait()

                join_threads(True)
            join_threads(False)

        finally:
            # It is very important that we clean up everything because the
            # Python tracer will detect ALL active threads. (Even orphans from
            # prior failed tests.) If we don't clean up properly we can
            # contaminate subsequent tests.
            start_barrier.abort()
            end_barrier.abort()
            for t in threads.values():
                t.release()

            for t in threads.values():
                t.join(timeout=timeout)

            for t in threads.values():
                self.assertFalse(t.is_alive())

        roots = prof.profiler.kineto_results.experimental_event_tree()
        print(f"==== ROOT {roots}")
        nodes = [node for node in _utils.traverse_dfs(roots) if isinstance(node.extra_fields, _ExtraFields_PyCall)]
        print(f"==== NODE {nodes}")
        tid_counts = collections.Counter([node.start_tid for node in nodes])
        print(f"==== {[node.start_tid for node in nodes]}")
        print(f"==== {tid_counts}")
        print(f"==== THREAD: {thread_spec}")

        prior_threads = sum(not start_under_profiler for start_under_profiler, _ in thread_spec)
        print(f"==== GOT: {len(tid_counts)}")
        print(f"==== EXPECTED: {prior_threads + 1}")
        expected_threads = prior_threads + 1
        self.assertEqual(len(tid_counts), expected_threads, f"{expected_threads}, {tid_counts}")
        self.assertEqual(len(nodes), sum(tid_counts.values()))

        # Profiler uses uint64_t max as a placeholder until TID can be determined.
        no_tid = 2 ** 64 - 1
        self.assertFalse(no_tid in tid_counts)

        worker_threads = prior_threads + (1 if work_in_main_thread else 0)

        observed_preexisting = [node.start_tid for node in nodes if "known_preexisting_function" in node.name]
        self.assertEqual(len(observed_preexisting), worker_threads)
        self.assertEqual(len(observed_preexisting), len(set(observed_preexisting)))

        observed_during_run = [node.start_tid for node in nodes if "invoked_during_run" in node.name]
        self.assertEqual(len(observed_during_run), worker_threads)
        self.assertEqual(len(observed_during_run), len(set(observed_during_run)))

    def payload(self, use_cuda=False):
        x = torch.randn(10, 10)
        if use_cuda:
            x = x.cuda()
        y = torch.randn(10, 10)
        if use_cuda:
            y = y.cuda()
        z = torch.mm(x, y)
        z = z + y
        if use_cuda:
            z = z.cpu()


def find_node_with_name(nodes, name):
    for node in _utils.traverse_dfs(nodes):
        if node.name == name:
            return node

def find_node_with_regex(nodes, pattern):
    for node in _utils.traverse_dfs(nodes):
        if re.search(pattern, node.name):
            return node


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        return self.fc2(self.fc1(x))


if __name__ == '__main__':
    run_tests()
