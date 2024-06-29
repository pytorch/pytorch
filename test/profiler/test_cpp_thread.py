# Owner(s): ["oncall: profiler"]

import os
import shutil
import subprocess

import torch
import torch.utils.cpp_extension
from torch.testing._internal.common_utils import IS_WINDOWS, run_tests, TestCase


def remove_build_path():
    default_build_root = torch.utils.cpp_extension.get_default_build_root()
    if os.path.exists(default_build_root):
        if IS_WINDOWS:
            # rmtree returns permission error: [WinError 5] Access is denied
            # on Windows, this is a word-around
            subprocess.run(["rm", "-rf", default_build_root], stdout=subprocess.PIPE)
        else:
            shutil.rmtree(default_build_root)


def is_fbcode():
    return not hasattr(torch.version, "git_version")


if is_fbcode():
    import caffe2.test.profiler_test_cpp_thread_lib as cpp
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
        # blueprint(f"training iteration {iteration} in thread {thread_id}")
        device = torch.device("cuda")
        # device = torch.device("cpu")
        with torch.autograd.profiler.record_function("user_function"):
            a = torch.ones(1, device=device)
            b = torch.ones(1, device=device)
            torch.add(a, b).cpu()
            torch.cuda.synchronize()


class CppThreadTest(TestCase):
    ThreadCount = 20  # set to 2 for debugging
    EventHandler = None
    TraceObject = None

    @classmethod
    def setUpClass(cls) -> None:
        super(TestCase, cls).setUpClass()
        CppThreadTest.EventHandler = PythonProfilerEventHandler()
        cpp.ProfilerEventHandler.Register(CppThreadTest.EventHandler)

    @classmethod
    def tearDownClass(cls):
        if not is_fbcode():
            remove_build_path()

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("Test machine does not have cuda")

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
        CppThreadTest.TraceObject = trace_obj

    def assert_text(self, condition, text, msg):
        if condition:
            print(f"\33[32m{text}\33[0m")
        else:
            print(f"\33[31m{text}\33[0m")
        self.assertTrue(condition, msg)

    def check_trace(self, expected, mem=False) -> None:
        blueprint("verifying trace")
        event_list = CppThreadTest.TraceObject.events()
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

    def test_with_enable_profiler_in_child_thread(self) -> None:
        self.start_profiler(False)
        cpp.start_threads(self.ThreadCount, IterationCount, True)
        self.check_trace(
            {
                "aten::add": [self.ThreadCount, "CPU"],
                "user_function": [self.ThreadCount, "CUDA"],
            }
        )

    def test_without_enable_profiler_in_child_thread(self) -> None:
        self.start_profiler(False)
        cpp.start_threads(self.ThreadCount, IterationCount, False)
        self.check_trace(
            {
                "aten::add": [1, "CPU"],
                "user_function": [1, "CUDA"],
            }
        )

    def test_profile_memory(self) -> None:
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
