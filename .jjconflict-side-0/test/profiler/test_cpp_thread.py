# Owner(s): ["oncall: profiler"]

import os

import torch
import torch.utils.cpp_extension
from torch._environment import is_fbcode
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    skipCUDAIf,
    skipXPUIf,
)
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
        if not hasattr(torch_device, "synchronize"):
            raise AssertionError(f"Device {device} does not have synchronize method")
        sync_func = torch_device.synchronize

        with torch.autograd.profiler.record_function("user_function"):
            a = torch.ones(1, device=device)
            b = torch.ones(1, device=device)
            torch.add(a, b).cpu()
            sync_func()


class CppThreadTest(TestCase):
    ThreadCount = 20  # set to 2 for debugging
    EventHandler = None
    TraceObject = None

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.EventHandler = PythonProfilerEventHandler()
        cpp.ProfilerEventHandler.Register(cls.EventHandler)

    @classmethod
    def tearDownClass(cls):
        if not is_fbcode():
            torch.testing._internal.common_utils.remove_cpp_extensions_build_root()

    def setUp(self) -> None:
        super().setUp()
        global device
        device = self.device_type

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
        type(self).TraceObject = trace_obj

    def assert_text(self, condition, text, msg):
        if condition:
            print(f"\33[32m{text}\33[0m")
        else:
            print(f"\33[31m{text}\33[0m")
        self.assertTrue(condition, msg)

    def check_trace(self, expected, mem=False) -> None:
        blueprint("verifying trace")
        event_list = type(self).TraceObject.events()
        for key, values in expected.items():
            count = values[0]
            min_count = count * (ActivateIteration - 1)
            dev = values[1]
            filtered = filter(
                lambda ev: ev.name == key
                and str(ev.device_type) == f"DeviceType.{dev}",
                event_list,
            )

            if mem:
                mem_key = f"{self.device_type}_memory_usage"
                actual = 0
                for ev in filtered:
                    sev = str(ev)
                    has_device_memory_usage = (
                        sev.find(f"{mem_key}=0 ") < 0 and sev.find(f"{mem_key}=") > 0
                    )
                    if has_device_memory_usage:
                        actual += 1
                self.assert_text(
                    actual >= min_count,
                    f"{key}: {actual} >= {min_count}",
                    f"not enough event with {mem_key} set",
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

    @skipCUDAIf(
        IS_WINDOWS,
        "Failing on windows cuda, see https://github.com/pytorch/pytorch/pull/130037 for slightly more context",
    )
    @skipXPUIf(
        True,
        "The XPU Profiler will not cover this case for now. Will support it in next period.",
    )
    def test_with_enable_profiler_in_child_thread(self, device) -> None:
        self.start_profiler(False)
        cpp.start_threads(self.ThreadCount, IterationCount, True)
        self.check_trace(
            {
                "aten::add": [self.ThreadCount, "CPU"],
                "user_function": [self.ThreadCount, self.device_type.upper()],
            }
        )

    @skipCUDAIf(
        IS_WINDOWS,
        "Failing on windows cuda, see https://github.com/pytorch/pytorch/pull/130037 for slightly more context",
    )
    @skipXPUIf(
        True,
        "The XPU Profiler will not cover this case for now. Will support it in next period.",
    )
    def test_without_enable_profiler_in_child_thread(self, device) -> None:
        self.start_profiler(False)
        cpp.start_threads(self.ThreadCount, IterationCount, False)
        self.check_trace(
            {
                "aten::add": [1, "CPU"],
                "user_function": [1, self.device_type.upper()],
            }
        )

    @skipCUDAIf(
        IS_WINDOWS,
        "Failing on windows cuda, see https://github.com/pytorch/pytorch/pull/130037 for slightly more context",
    )
    @skipXPUIf(
        True,
        "The XPU Profiler will not cover this case for now. Will support it in next period.",
    )
    def test_profile_memory(self, device) -> None:
        self.start_profiler(True)
        cpp.start_threads(self.ThreadCount, IterationCount, True)
        self.check_trace(
            {
                "aten::add": [self.ThreadCount, "CPU"],
            },
            mem=True,
        )


instantiate_device_type_tests(CppThreadTest, globals(), only_for=("cuda", "xpu"))

if __name__ == "__main__":
    run_tests()
