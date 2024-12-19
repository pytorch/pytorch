# Owner(s): ["oncall: profiler"]
# ruff: noqa: F841

# if tqdm is not shutdown properly, it will leave the monitor thread alive.
# This causes an issue in the multithreading test because we check all events
# in that test with their tids. The events that correspond to these lingering
# threads all have TID of (uint64_t)(-1) which is invalid.
# The work around is turnning off monitoring thread when tqdm is loaded.
# Since these are unit tests, it is safe to turn off monitor thread.
try:
    import tqdm

    tqdm.tqdm.monitor_interval = 0
except ImportError:
    None

from typing import Any

import torch
import torch.optim
import torch.utils.data
import torch.utils.data.datapipes as dp
from torch.autograd import (
    _record_function_with_args_enter,
    _record_function_with_args_exit,
)
from torch.autograd.profiler import profile as _profile
from torch.profiler import kineto_available, record_function
from torch.testing._internal.common_utils import run_tests, TestCase


Json = dict[str, Any]


class TestRecordFunction(TestCase):
    def _record_function_with_param(self):
        u = torch.randn(3, 4, 5, requires_grad=True)
        with _profile(
            with_stack=True, use_kineto=kineto_available(), record_shapes=True
        ) as prof:
            with record_function("## TEST 1 ##", "1, 2, 3"):
                rf_handle = _record_function_with_args_enter(
                    "## TEST 2 ##", 1, False, 2.5, [u, u], "hello", u
                )
                _record_function_with_args_exit(rf_handle)
            with record_function("## TEST 3 ##"):
                rf_handle = _record_function_with_args_enter("## TEST 4 ##")
                _record_function_with_args_exit(rf_handle)
        return prof

    def test_record_function(self):
        prof_result = self._record_function_with_param()
        found_test_1 = False
        found_test_2 = False
        found_test_3 = False
        found_test_4 = False
        for e in prof_result.function_events:
            if "## TEST 1 ##" == e.name:
                found_test_1 = True
                self.assertTrue(e.input_shapes == [[]])
            elif "## TEST 2 ##" == e.name:
                found_test_2 = True
                self.assertTrue(e.input_shapes == [[], [], [], [], [], [3, 4, 5]])
            elif "## TEST 3 ##" == e.name:
                found_test_3 = True
                self.assertTrue(e.input_shapes == [])
            elif "## TEST 4 ##" == e.name:
                found_test_4 = True
                self.assertTrue(e.input_shapes == [])
        self.assertTrue(found_test_1)
        self.assertTrue(found_test_2)
        self.assertTrue(found_test_3)
        self.assertTrue(found_test_4)

    def test_datapipe_with_record_function(self):
        with _profile(
            with_stack=True, use_kineto=kineto_available(), record_shapes=True
        ) as prof:
            input_dp1 = dp.iter.IterableWrapper(range(4))
            input_dp2 = dp.iter.IterableWrapper(range(4, 8))
            input_dp3 = dp.iter.IterableWrapper(range(8, 12))
            output_dp = input_dp1.mux(input_dp2, input_dp3)
            output = list(output_dp)

        has_iter = False
        has_mux = False
        for e in prof.function_events:
            if has_iter and has_mux:
                break

            if not has_iter and "IterableWrapper" in e.name:
                has_iter = True
            if not has_mux and "Multiplexer" in e.name:
                has_mux = True
        self.assertTrue(has_iter)
        self.assertTrue(has_mux)

    def test_datapipe_delegation_with_profiler(self):
        class IDPIterator(torch.utils.data.IterDataPipe):
            def __init__(self) -> None:
                self.data = list(range(10))
                self._idx = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self._idx >= 10:
                    self._idx = 0
                    raise StopIteration
                self._idx += 1
                return self.data[self._idx - 1]

            def get_value(self, idx):
                return self.data[idx]

        dp1 = IDPIterator()  # The object itself is an iterator
        self.assertEqual(5, dp1.get_value(5))
        it_dp1 = iter(dp1)  # This creates the 1st iterator
        self.assertEqual(5, it_dp1.get_value(5))  # type: ignore[attr-defined]
        self.assertEqual(list(range(10)), list(it_dp1))

        class IDPDelegator(torch.utils.data.IterDataPipe):
            def __init__(self, datapipe):
                self.datapipe = datapipe

            def __iter__(self):
                return iter(self.datapipe)

        dp2 = IDPDelegator(dp1)
        it_dp2 = iter(dp2)
        self.assertEqual(5, it_dp2.get_value(5))
        self.assertEqual(list(range(10)), list(it_dp2))

    def test_datapipe_with_record_function_fork(self):
        with _profile(
            with_stack=True, use_kineto=kineto_available(), record_shapes=True
        ) as prof:
            input_dp = dp.iter.IterableWrapper(range(10))
            dp1, dp2, dp3 = input_dp.fork(num_instances=3)
            output1 = list(dp1)
        has_iter = False
        has_child = False
        for e in prof.function_events:
            if has_iter and has_child:
                break

            if not has_iter and "IterableWrapper" in e.name:
                has_iter = True
            if not has_child and "_ChildDataPipe" in e.name:
                has_child = True
        self.assertTrue(has_iter)
        self.assertTrue(has_child)


if __name__ == "__main__":
    run_tests()
