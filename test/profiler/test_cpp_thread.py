# Owner(s): ["oncall: profiler"]

import os
import unittest
from unittest import skipIf

import torch
import torch.utils.cpp_extension
from torch._environment import is_fbcode
from torch.testing._internal.common_utils import IS_WINDOWS, run_tests, TestCase


if is_fbcode():
    import caffe2.test.profiler_test_cpp_thread_lib as cpp  # @manual=//caffe2/test:profiler_test_cpp_thread_lib
else:
    # cpp extensions use relative paths. Those paths are relative to
    # this file, so we'll change the working directory temporarily
    old_working_dir = os.getcwd()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    cpp = torch.utils.cpp_extension.load(
        name="profiler_test_cpp_thread_lib",
        sources=[
            "test_cpp_thread.cpp",
        ],
        verbose=True,
    )

    # return the working directory (see setUp)
    os.chdir(old_working_dir)


KinetoProfiler = None
IterationCount = 5
ActivateIteration = 2
device = None


def blueprint(text):
    print(f"\33[34m{text}\33[0m")


# onIterationStart() will be called by C++ training engine in cpp_thread_test_lib.cpp
class PythonProfilerEventHandler(cpp.ProfilerEventHandler):
    def onIterationStart(self, iteration: int) -> None:
        global KinetoProfiler, IterationCount
        # it is important to start the profiler on the same thread that step() is called
        # and yes, onIterationStart() will always be called on the same thread
        if iteration == 0:
            # this also means step() starts on iteration 1, not 0
            KinetoProfiler.start()
            blueprint("starting kineto profiler")
        elif iteration == IterationCount - 1:
            KinetoProfiler.stop()
            blueprint("stopping kineto profiler")
        else:
            blueprint("stepping kineto profiler")
            KinetoProfiler.step()

    def emulateTraining(self, iteration: int, thread_id: int) -> None:
        global device
        # blueprint(f"training iteration {iteration} in thread {thread_id}")
        torch_device = getattr(torch, device)
        assert hasattr(torch_device, "synchronize")
        sync_func = torch_device.synchronize

        with torch.autograd.profiler.record_function("user_function"):
            a = torch.ones(1, device=device)
            b = torch.ones(1, device=device)
            torch.add(a, b).cpu()
            sync_func()


class CppThreadTestCUDA(TestCase):
    ThreadCount = 20  # set to 2 for debugging
    EventHandler = None
    TraceObject = None

    @classmethod
    def setUpClass(cls) -> None:
        super(TestCase, cls).setUpClass()
        CppThreadTestCUDA.EventHandler = PythonProfilerEventHandler()
        cpp.ProfilerEventHandler.Register(CppThreadTestCUDA.EventHandler)

    @classmethod
    def tearDownClass(cls):
        if not is_fbcode():
            torch.testing._internal.common_utils.remove_cpp_extensions_build_root()

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("Test machine does not have cuda")
        global device
        device = "cuda"

        # this clears off events from initialization
        self.start_profiler(False)
        cpp.start_threads(1, IterationCount, False)

    def start_profiler(self, profile_memory):
        global KinetoProfiler
        KinetoProfiler = torch.profiler.profile(
            schedule=torch.profiler.schedule(
                wait=1, warmup=1, active=ActivateIteration, repeat=1
            ),
            on_trace_ready=self.set_trace,
            with_stack=True,
            profile_memory=profile_memory,
            record_shapes=True,
        )

    def set_trace(self, trace_obj) -> None:
        CppThreadTestCUDA.TraceObject = trace_obj

    def assert_text(self, condition, text, msg):
        if condition:
            print(f"\33[32m{text}\33[0m")
        else:
            print(f"\33[31m{text}\33[0m")
        self.assertTrue(condition, msg)

    def check_trace(self, expected, mem=False) -> None:
        blueprint("verifying trace")
        event_list = CppThreadTestCUDA.TraceObject.events()
        for key, values in expected.items():
            count = values[0]
            min_count = count * (ActivateIteration - 1)
            device = values[1]
            filtered = filter(
                lambda ev: ev.name == key
                and str(ev.device_type) == f"DeviceType.{device}",
                event_list,
            )

            if mem:
                actual = 0
                for ev in filtered:
                    sev = str(ev)
                    has_cuda_memory_usage = (
                        sev.find("cuda_memory_usage=0 ") < 0
                        and sev.find("cuda_memory_usage=") > 0
                    )
                    if has_cuda_memory_usage:
                        actual += 1
                self.assert_text(
                    actual >= min_count,
                    f"{key}: {actual} >= {min_count}",
                    "not enough event with cuda_memory_usage set",
                )
            else:
                actual = len(list(filtered))
                if count == 1:  # test_without
                    count *= ActivateIteration
                    self.assert_text(
                        actual == count,
                        f"{key}: {actual} == {count}",
                        "baseline event count incorrect",
                    )
                else:
                    self.assert_text(
                        actual >= min_count,
                        f"{key}: {actual} >= {min_count}",
                        "not enough event recorded",
                    )

    @skipIf(
        IS_WINDOWS,
        "Failing on windows cuda, see https://github.com/pytorch/pytorch/pull/130037 for slightly more context",
    )
    def test_with_enable_profiler_in_child_thread_cuda(self) -> None:
        self.start_profiler(False)
        cpp.start_threads(self.ThreadCount, IterationCount, True)
        self.check_trace(
            {
                "aten::add": [self.ThreadCount, "CPU"],
                "user_function": [self.ThreadCount, "CUDA"],
            }
        )

    @skipIf(
        IS_WINDOWS,
        "Failing on windows cuda, see https://github.com/pytorch/pytorch/pull/130037 for slightly more context",
    )
    def test_without_enable_profiler_in_child_thread_cuda(self) -> None:
        self.start_profiler(False)
        cpp.start_threads(self.ThreadCount, IterationCount, False)
        self.check_trace(
            {
                "aten::add": [1, "CPU"],
                "user_function": [1, "CUDA"],
            }
        )

    @skipIf(
        IS_WINDOWS,
        "Failing on windows cuda, see https://github.com/pytorch/pytorch/pull/130037 for slightly more context",
    )
    def test_profile_memory_cuda(self) -> None:
        self.start_profiler(True)
        cpp.start_threads(self.ThreadCount, IterationCount, True)
        self.check_trace(
            {
                "aten::add": [self.ThreadCount, "CPU"],
            },
            mem=True,
        )


# Here duplicate the CppThreadTest to enable the xpu cases because the
# instantiate_device_type_tests will call class method setUpClass.
# In function setUpClass, the instantiated class(e.g CppThreadTestCPU, CppThreadTestXPU)
# needs to be called to get it member EventHandler, while in this period,
# the input class in argument cls is CppThreadTest, which is not defined any more.
# We cannot detect which instantiated class is being created in setUpClass, so duplicate here
# for enabling xpu test cases
class CppThreadTestXPU(TestCase):
    ThreadCount = 20  # set to 2 for debugging
    EventHandler = None
    TraceObject = None

    @classmethod
    def setUpClass(cls) -> None:
        super(TestCase, cls).setUpClass()
        CppThreadTestXPU.EventHandler = PythonProfilerEventHandler()
        cpp.ProfilerEventHandler.Register(CppThreadTestXPU.EventHandler)

    @classmethod
    def tearDownClass(cls):
        if not is_fbcode():
            torch.testing._internal.common_utils.remove_cpp_extensions_build_root()

    def setUp(self) -> None:
        if not torch.xpu.is_available():
            self.skipTest("Test machine does not have xpu")
        global device
        device = "xpu"

        # this clears off events from initialization
        self.start_profiler(False)
        cpp.start_threads(1, IterationCount, False)

    def start_profiler(self, profile_memory):
        global KinetoProfiler
        KinetoProfiler = torch.profiler.profile(
            schedule=torch.profiler.schedule(
                wait=1, warmup=1, active=ActivateIteration, repeat=1
            ),
            on_trace_ready=self.set_trace,
            with_stack=True,
            profile_memory=profile_memory,
            record_shapes=True,
        )

    def set_trace(self, trace_obj) -> None:
        CppThreadTestXPU.TraceObject = trace_obj

    def assert_text(self, condition, text, msg):
        if condition:
            print(f"\33[32m{text}\33[0m")
        else:
            print(f"\33[31m{text}\33[0m")
        self.assertTrue(condition, msg)

    def check_trace(self, expected, mem=False) -> None:
        blueprint("verifying trace")
        event_list = CppThreadTestXPU.TraceObject.events()
        for key, values in expected.items():
            count = values[0]
            min_count = count * (ActivateIteration - 1)
            device = values[1]
            filtered = filter(
                lambda ev: ev.name == key
                and str(ev.device_type) == f"DeviceType.{device}",
                event_list,
            )

            if mem:
                actual = 0
                for ev in filtered:
                    sev = str(ev)
                    has_cuda_memory_usage = (
                        sev.find("xpu_memory_usage=0 ") < 0
                        and sev.find("xpu_memory_usage=") > 0
                    )
                    if has_cuda_memory_usage:
                        actual += 1
                self.assert_text(
                    actual >= min_count,
                    f"{key}: {actual} >= {min_count}",
                    "not enough event with xpu_memory_usage set",
                )
            else:
                actual = len(list(filtered))
                if count == 1:  # test_without
                    count *= ActivateIteration
                    self.assert_text(
                        actual == count,
                        f"{key}: {actual} == {count}",
                        "baseline event count incorrect",
                    )
                else:
                    self.assert_text(
                        actual >= min_count,
                        f"{key}: {actual} >= {min_count}",
                        "not enough event recorded",
                    )

    @unittest.skip(
        reason="The XPU Profiler will not cover this case for now. Will support it in next period."
    )
    def test_with_enable_profiler_in_child_thread_xpu(self) -> None:
        self.start_profiler(False)
        cpp.start_threads(self.ThreadCount, IterationCount, True)
        self.check_trace(
            {
                "aten::add": [self.ThreadCount, "CPU"],
                "user_function": [self.ThreadCount, "XPU"],
            }
        )

    @unittest.skip(
        reason="The XPU Profiler will not cover this case for now. Will support it in next period."
    )
    def test_without_enable_profiler_in_child_thread_xpu(self) -> None:
        self.start_profiler(False)
        cpp.start_threads(self.ThreadCount, IterationCount, False)
        self.check_trace(
            {
                "aten::add": [1, "CPU"],
                "user_function": [1, "XPU"],
            }
        )

    @unittest.skip(
        reason="The XPU Profiler will not cover this case for now. Will support it in next period."
    )
    def test_profile_memory_xpu(self) -> None:
        self.start_profiler(True)
        cpp.start_threads(self.ThreadCount, IterationCount, True)
        self.check_trace(
            {
                "aten::add": [self.ThreadCount, "CPU"],
            },
            mem=True,
        )


if __name__ == "__main__":
    run_tests()
