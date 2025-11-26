# Owner(s): ["module: tests"]
import os
import threading

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


# This test is in its own file so that nothing tries to accesses the backend
# before this test does. This is required since, for the MPS backend at least,
# the pipeline state for each global unary function is cached and can't be
# reset by things like importlib.reload(). This also means that the test should
# only be run through run_test.py or with the --subprocess flag.
class TestFunctionsMultithreaded(TestCase):
    # Since this is a test for a race condition, the test is probabilistic and
    # may pass spuriously. When this was failing I'd only get ~30% failure rate
    # per function, so we test with a bunch of unary functions to increase the
    # likelihood of us catching the race. With 10 functions I haven't seen this
    # test pass when the bug is present.
    @parametrize(
        "unary_func",
        [
            torch.tanh,
            torch.abs,
            torch.sin,
            torch.cos,
            torch.tan,
            torch.asin,
            torch.acos,
            torch.atan,
            torch.sqrt,
            torch.rsqrt,
        ],
        name_fn=lambda fn: fn.__name__,
    )
    def test_unary_function_on_multiple_threads(self, device, unary_func):
        num_threads = max((os.cpu_count() or 0) // 2, 2)
        input = torch.rand(128).to(device)

        barrier = threading.Barrier(num_threads)

        def task():
            barrier.wait()
            unary_func(input)

        threads = [threading.Thread(target=task) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()


instantiate_device_type_tests(TestFunctionsMultithreaded, globals(), allow_mps=True)

if __name__ == "__main__":
    run_tests()
