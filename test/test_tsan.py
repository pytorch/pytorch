# Owner(s): ["module: ci"]

import threading

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestTSan(TestCase):
    def test_data_race(self):
        # Intentional data race: multiple threads doing unsynchronized
        # in-place operations on a shared tensor. This should trigger a
        # TSan error, confirming the sanitizer is working.
        for _ in range(100):
            shared = torch.zeros(64, 64)
            barrier = threading.Barrier(4)

            def fn():
                barrier.wait()
                for _ in range(100):
                    shared.add_(1)

            threads = [threading.Thread(target=fn) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()


if __name__ == "__main__":
    run_tests()
