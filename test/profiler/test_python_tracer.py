# Owner(s): ["oncall: profiler"]

import json
import subprocess
import sys
import time

from torch.profiler import profile, ProfilerActivity
from torch.testing._internal.common_utils import (
    run_tests,
    skipIfPythonVersionMismatch,
    TemporaryFileName,
    TestCase,
)


class TestPythonTracer(TestCase):
    @skipIfPythonVersionMismatch(lambda major, minor, micro: major == 3 and minor == 12)
    def test_method_with_c_function(self):
        class A:
            method_with_c_function = classmethod(repr)

        def get_key(x):
            A().method_with_c_function()
            time.sleep(1.2)
            return len(x)

        names = ["Alice", "Bob"]

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True
        ) as prof:
            sorted(names, key=get_key)

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                events = json.load(f)["traceEvents"]
                found = False
                for event in events:
                    if (
                        event.get("cat", "") == "python_function"
                        and event.get("name", "") == "<built-in function sorted>"
                    ):
                        duration = event.get("dur", 0)
                        if duration >= 2000000:
                            found = True
                            break
                self.assertTrue(found)

    @skipIfPythonVersionMismatch(lambda major, minor, micro: major == 3 and minor == 12)
    def test_monitoring_callback(self):
        vi = sys.version_info
        from sys import monitoring

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True
        ):
            name = monitoring.get_tool(2)
            if vi.micro < 5:
                self.assertEqual(name, "PyTorch Profiler")
            else:
                self.assertEqual(name, None)
        name = monitoring.get_tool(2)
        self.assertEqual(name, None)

    def test_unexpected_c_return_events(self):
        code = """
import threading
import time
import torch

from threading import Event, Lock

lock = Lock()
lock.acquire()
event1 = Event()
event2 = Event()
event3 = Event()

def run():
    event1.set()
    event2.wait()
    lock.acquire()
    event3.set()

threading.Thread(target=run).start()

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_stack=True):
    event1.wait()
    event2.set()
    time.sleep(1)

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_stack=True):
    lock.release()
    event3.wait()
    """

        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )

        self.assertFalse(
            "Python replay stack is empty during pop operation" in result.stderr
        )


if __name__ == "__main__":
    run_tests()
