# Owner(s): ["oncall: profiler"]
# ruff: noqa: F841

import json
from typing import Any

import torch
import torch.optim
import torch.utils.data
import torch.utils.data.datapipes as dp
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.utils import ChromiumEventLogger
from torch.autograd import (
    _record_function_with_args_enter,
    _record_function_with_args_exit,
)
from torch.autograd.profiler import profile as _profile
from torch.profiler import kineto_available, profile, record_function
from torch.testing._internal.common_profiler import (
    _profiler_test_skip_reason,
    initialize_kineto_with_accelerator,
    skipIfProfilerUnsupported,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    skipIfTorchDynamo,
    TemporaryFileName,
    TestCase,
)


def setUpModule():
    if _profiler_test_skip_reason() is None:
        initialize_kineto_with_accelerator()


# if tqdm is not shutdown properly, it will leave the monitor thread alive.
# This causes an issue in the multithreading test because we check all events
# in that test with their tids. The events that correspond to these lingering
# threads all have TID of (uint64_t)(-1) which is invalid.
# The work around is turning off monitoring thread when tqdm is loaded.
# Since these are unit tests, it is safe to turn off monitor thread.
try:
    import tqdm

    tqdm.tqdm.monitor_interval = 0
except ImportError:
    pass

Json = dict[str, Any]


class TestRecordFunction(TestCase):
    hw_classification = HardwareClassification.GENERIC

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

    def test_python_dispatch_mode_record_function(self):
        from torch.utils._python_dispatch import TorchDispatchMode

        class TestDispatchMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}
                return func(*args, **kwargs)

        with _profile() as prof:
            with enable_python_dispatcher():
                with TestDispatchMode():
                    x = torch.randn(3, 4)
                    y = torch.sin(x)

        found_python_dispatch_mode = False
        for e in prof.function_events:
            if e.name == "PythonDispatchMode":
                found_python_dispatch_mode = True
                break
        self.assertTrue(
            found_python_dispatch_mode,
            "PythonDispatchMode record function not found in profiler events",
        )

    def test_python_subclass_record_function(self):
        class TestTensorSubclass(torch.Tensor):
            @staticmethod
            def __new__(cls, elem):
                r = torch.Tensor._make_wrapper_subclass(
                    cls,
                    elem.size(),
                    dtype=elem.dtype,
                    device=elem.device,
                    requires_grad=elem.requires_grad,
                )
                r.elem = elem
                return r

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}

                def unwrap(x):
                    return x.elem if isinstance(x, TestTensorSubclass) else x

                def wrap(x):
                    return TestTensorSubclass(x) if isinstance(x, torch.Tensor) else x

                unwrapped_args = tuple(unwrap(arg) for arg in args)
                unwrapped_kwargs = {k: unwrap(v) for k, v in kwargs.items()}
                result = func(*unwrapped_args, **unwrapped_kwargs)

                if isinstance(result, torch.Tensor):
                    return TestTensorSubclass(result)
                return result

        with _profile() as prof:
            with enable_python_dispatcher():
                x = TestTensorSubclass(torch.randn(3, 4))
                y = torch.sin(x)

        found_python_subclass = False
        for e in prof.function_events:
            if e.name == "PythonSubclass":
                found_python_subclass = True
                break
        self.assertTrue(
            found_python_subclass,
            "PythonSubclass record function not found in profiler events",
        )

    def test_chromium_event_logger_profiler_integration(self):
        """Verify ChromiumEventLogger emits RecordFunction events to profiler"""

        with _profile(use_kineto=kineto_available()) as prof:
            logger = ChromiumEventLogger()
            logger.log_event_start("test_compile_event", 0, {})
            logger.log_event_end(
                "test_compile_event", 1000, {}, 0, log_pt2_compile_event=False
            )

        found = False
        for e in prof.function_events:
            if e.name == "test_compile_event":
                found = True
                break
        self.assertTrue(
            found,
            "ChromiumEventLogger event not found in profiler output",
        )

    def test_chromium_logger_reset_cleans_open_events(self):
        """Verify reset() cleans up lingering record functions on exception"""
        from torch._dynamo.utils import ChromiumEventLogger

        with _profile(use_kineto=kineto_available()):
            logger = ChromiumEventLogger()

            logger.log_event_start("incomplete_event", 0, {})

            rf_dict = logger.get_record_functions()
            self.assertIn("incomplete_event", rf_dict)

            logger.reset()

            rf_dict = logger.get_record_functions()
            self.assertEqual(len(rf_dict), 0, "reset() failed to clean up open events")

    def test_chromium_logger_no_overhead_when_profiler_disabled(self):
        """Verify zero overhead when profiler is not enabled"""
        from torch._dynamo.utils import ChromiumEventLogger

        self.assertFalse(torch.autograd.profiler._is_profiler_enabled)

        logger = ChromiumEventLogger()

        logger.log_event_start("test_event", 0, {})
        logger.log_event_end("test_event", 1000, {}, 0, log_pt2_compile_event=False)

        rf_dict = logger.get_record_functions()
        self.assertEqual(
            len(rf_dict), 0, "RecordFunction created when profiler disabled"
        )

    @skipIfProfilerUnsupported
    def test_record_function_fast(self):
        x, y = (torch.rand((4, 4)) for _ in range(2))
        with profile(record_shapes=True) as p:
            for _ in range(4):
                # Test first with no optional args
                with torch._C._profiler._RecordFunctionFast("add_test_fast_rf1"):
                    x.add(y)

        self.assertGreaterEqual(
            len([e for e in p.events() if e.name == "add_test_fast_rf1"]), 4
        )
        for e in p.events():
            if e.name == "add_test_fast_rf1":
                self.assertTrue(e.input_shapes == [])
                self.assertTrue(e.kwinputs == {})
        with profile(record_shapes=True) as p:
            # add optional args
            cm = torch._C._profiler._RecordFunctionFast(
                "add_test_fast_rf2", [x, y], {"stream": 0, "grid": "lambda x : x + 1"}
            )
            for _ in range(4):
                with cm:
                    x.add(y)

        self.assertGreaterEqual(
            len([e for e in p.events() if e.name == "add_test_fast_rf2"]), 4
        )

        for e in p.events():
            if e.name == "add_test_fast_rf2":
                self.assertTrue(e.input_shapes == [[4, 4], [4, 4]])
                self.assertTrue(e.kwinputs == {"stream": 0, "grid": "lambda x : x + 1"})

        with profile(record_shapes=True) as p:
            cm = torch._C._profiler._RecordFunctionFast(
                "add_test_fast_rf3", input_values=["hi"], keyword_values={"hi": "hello"}
            )
            for _ in range(4):
                try:
                    with cm:
                        x.add(y)
                        raise ValueError
                        x.relu()
                except ValueError:
                    pass

        self.assertGreaterEqual(
            len([e for e in p.events() if e.name == "add_test_fast_rf3"]), 4
        )
        self.assertFalse(any((e.name and "relu" in e.name) for e in p.events()))

        for e in p.events():
            if e.name == "add_test_fast_rf3":
                self.assertTrue(e.input_shapes == [[]])

        with profile() as p:
            for _ in range(4):
                with torch._C._profiler._RecordFunctionFast(
                    "add_test_fast_rf4", [x, y]
                ):
                    x.add(y)
                    with torch._C._profiler._RecordFunctionFast("add_test_fast_rf5"):
                        x.relu()

        self.assertGreaterEqual(
            len([e for e in p.events() if e.name == "add_test_fast_rf4"]), 4
        )

        for e in p.events():
            if e.name == "add_test_fast_rf4":
                self.assertTrue(e.input_shapes == [])

        self.assertGreaterEqual(
            len([e for e in p.events() if e.name == "add_test_fast_rf5"]), 4
        )

        with profile(record_shapes=True) as p:
            # test optional args with tuple
            cm = torch._C._profiler._RecordFunctionFast(
                "add_test_fast_rf6",
                (
                    x,
                    y,
                ),
            )
            for _ in range(4):
                with cm:
                    x.add(y)

        self.assertGreaterEqual(
            len([e for e in p.events() if e.name == "add_test_fast_rf6"]), 4
        )

        for e in p.events():
            if e.name == "add_test_fast_rf6":
                self.assertTrue(e.input_shapes == [[4, 4], [4, 4]])

    @skipIfProfilerUnsupported
    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    def test_profiler_op_event_kwargs(self):
        x, y = (torch.rand((4, 4)) for _ in range(2))
        with profile(record_shapes=True) as p:
            cm = torch._C._profiler._RecordFunctionFast(
                "add_test_kwinputs",
                [x, y],
                {
                    "stream": 0,
                    "grid": "lambda x : x + 1",
                    "debug": 'debug"',
                    "boolean": True,
                },
            )
            for _ in range(4):
                with cm:
                    x.add(y)
        with TemporaryFileName(mode="w+") as fname:
            p.export_chrome_trace(fname)
            with open(fname) as f:
                j = json.load(f)
                op_events = [
                    e
                    for e in j["traceEvents"]
                    if e.get("name", "") == "add_test_kwinputs"
                ]
                self.assertTrue(len(op_events) > 0)
                for e in op_events:
                    args = e["args"]
                    self.assertTrue("stream" in args)
                    self.assertTrue("grid" in args)
                    self.assertTrue("boolean" in args)
                    self.assertTrue(args["stream"] == 0)
                    self.assertTrue(args["grid"] == "lambda x : x + 1")
                    self.assertTrue(args["debug"] == "None")
                    self.assertTrue(args["boolean"])
                    self.assertTrue(e["cat"] == "cpu_op")

        with profile(record_shapes=True) as p1:
            cm = torch._C._profiler._RecordFunctionFast(
                "add_test_kwinputs",
                [x, y],
                {"stream": "test", "grid": [1, 2], "scope": "user_scope"},
            )
            for _ in range(4):
                with cm:
                    x.add(y)
        with TemporaryFileName(mode="w+") as fname1:
            p1.export_chrome_trace(fname1)
            with open(fname1) as f1:
                j = json.load(f1)
                op_events = [
                    e
                    for e in j["traceEvents"]
                    if e.get("name", "") == "add_test_kwinputs"
                ]
                self.assertTrue(len(op_events) > 0)
                for e in op_events:
                    args = e["args"]
                    self.assertTrue("stream" not in args)
                    self.assertTrue("grid" not in args)
                    self.assertTrue(e["cat"] == "user_annotation")

    @skipIfProfilerUnsupported
    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    def test_profiler_op_event_kwargs_list_of_strings(self):
        x, y = (torch.rand((4, 4)) for _ in range(2))
        with profile(record_shapes=True) as p:
            cm = torch._C._profiler._RecordFunctionFast(
                "add_test_kwinputs_string_list",
                [x, y],
                {
                    "string_list": ["hello", "world", "test"],
                    "int_param": 42,
                    "string_param": "single_string",
                },
            )
            for _ in range(4):
                with cm:
                    x.add(y)
        with TemporaryFileName(mode="w+") as fname:
            p.export_chrome_trace(fname)
            with open(fname) as f:
                j = json.load(f)
                op_events = [
                    e
                    for e in j["traceEvents"]
                    if e.get("name", "") == "add_test_kwinputs_string_list"
                ]
                self.assertTrue(len(op_events) > 0)
                for e in op_events:
                    args = e["args"]
                    self.assertTrue("string_list" in args)
                    self.assertTrue("int_param" in args)
                    self.assertTrue("string_param" in args)
                    self.assertEqual(args["string_list"], ["hello", "world", "test"])
                    self.assertEqual(args["int_param"], 42)
                    self.assertEqual(args["string_param"], "single_string")
                    self.assertTrue(e["cat"] == "cpu_op")

        # Test mixed types that should be filtered out
        with profile(record_shapes=True) as p1:
            cm = torch._C._profiler._RecordFunctionFast(
                "add_test_kwinputs_string_list_filtered",
                [x, y],
                {
                    "valid_string_list": ["valid1", "valid2"],
                    "mixed_list": ["string", 123],  # Should be filtered out
                    "non_string_list": [1, 2, 3],  # Should be filtered out
                    "valid_int": 100,
                },
            )
            for _ in range(4):
                with cm:
                    x.add(y)
        with TemporaryFileName(mode="w+") as fname1:
            p1.export_chrome_trace(fname1)
            with open(fname1) as f1:
                j = json.load(f1)
                op_events = [
                    e
                    for e in j["traceEvents"]
                    if e.get("name", "") == "add_test_kwinputs_string_list_filtered"
                ]
                self.assertTrue(len(op_events) > 0)
                for e in op_events:
                    args = e["args"]
                    # Only valid types should be present
                    self.assertTrue("valid_string_list" in args)
                    self.assertTrue("valid_int" in args)
                    # Invalid lists should be filtered out
                    self.assertTrue("mixed_list" not in args)
                    self.assertTrue("non_string_list" not in args)
                    # Check values
                    self.assertEqual(args["valid_string_list"], ["valid1", "valid2"])
                    self.assertEqual(args["valid_int"], 100)
                    self.assertTrue(e["cat"] == "cpu_op")

    @skipIfProfilerUnsupported
    def test_guarded_record_function_fast(self):
        x, y = (torch.rand((4, 4)) for _ in range(2))

        with profile() as p:
            cm = torch._C._profiler._RecordFunctionFast("guarded_rff")
            for _ in range(4):
                if torch.autograd.profiler._is_profiler_enabled:
                    with cm:
                        x.add(y)
                else:
                    x.add(y)

        self.assertGreaterEqual(
            len([e for e in p.events() if e.name == "guarded_rff"]), 4
        )


if __name__ == "__main__":
    run_tests()
