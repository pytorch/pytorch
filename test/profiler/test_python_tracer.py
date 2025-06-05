from torch.profiler import profile, ProfilerActivity
from torch.testing._internal.common_utils import run_tests, skipIfPythonVersionNotIn, TestCase, TemporaryFileName

import json
import sys
import time


class TestPythonTracer(TestCase):

    @skipIfPythonVersionNotIn("3.12", "3.13")
    def test_method_with_c_function(self):
        micro = sys.version_info.micro

        class A:
            method_with_c_function = classmethod(repr)

        def get_key(x):
            A().method_with_c_function()
            time.sleep(1)
            return len(x)

        names = ["Alice", "Bob"]

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof:
            sorted(names, key=get_key)

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with open(fname) as f:
                events = json.load(f)["traceEvents"]
                found = False
                for event in events:
                    if event.get("cat", "") == "python_function" and event.get("name", "") == "<built-in function sorted>":
                        duration = event.get("dur", 0)
                        if (duration >= 2000000):
                            found = True
                            break
                self.assertTrue(found)


if __name__ == "__main__":
    run_tests()
