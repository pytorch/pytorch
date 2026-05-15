# Owner(s): ["oncall: jit"]

from threading import Event
from time import sleep

import torch._lazy
import torch._lazy.ts_backend
from torch.testing._internal.common_utils import run_tests, skipIfFreeThreaded, TestCase


torch._lazy.ts_backend.init()


class ClosuresTest(TestCase):
    def test_synchronous(self):
        flag = Event()
        if flag.is_set():
            raise AssertionError("flag should not be set initially")

        def closure():
            sleep(1)
            if flag.is_set():
                raise AssertionError("flag should not be set during closure")
            flag.set()

        torch._lazy.add_step_closure(closure)
        torch._lazy.mark_step()

        # should not get to this part before closure is finished running
        if not flag.is_set():
            raise AssertionError("flag should be set after mark_step")

    def test_asynchronous(self):
        flag = Event()
        if flag.is_set():
            raise AssertionError("flag should not be set initially")

        def closure():
            sleep(1)
            if not flag.is_set():
                raise AssertionError("flag should be set by the time closure runs")

        torch._lazy.add_step_closure(closure, run_async=True)
        torch._lazy.mark_step()

        # should get to this part and complete before closure is finished running
        if flag.is_set():
            raise AssertionError("flag should not be set yet (async)")
        flag.set()

    def test_synchronous_exception(self):
        flag = Event()
        if flag.is_set():
            raise AssertionError("flag should not be set initially")

        try:

            def closure():
                flag.set()
                raise RuntimeError("Simulating exception in closure")

            torch._lazy.add_step_closure(closure)
            torch._lazy.mark_step()

            raise AssertionError("Should not reach here")
        except RuntimeError:
            if not flag.is_set():
                raise AssertionError(
                    "Should have caught exception from closure"
                ) from None

    @skipIfFreeThreaded(
        "Non-deterministic, fails more consistently in free threaded python",
    )
    def test_asynchronous_exception(self):
        flag = Event()
        if flag.is_set():
            raise AssertionError("flag should not be set initially")

        def closure1():
            flag.set()
            raise RuntimeError("Simulating exception in closure1")

        torch._lazy.add_step_closure(closure1, run_async=True)
        torch._lazy.mark_step()

        flag.wait(timeout=5)

        try:

            def closure2():  # Should never execute
                flag.clear()

            torch._lazy.add_step_closure(closure2, run_async=True)
            torch._lazy.mark_step()

            raise AssertionError("Should not reach here")
        except RuntimeError:
            # Should have caught exception from closure1
            pass

        if not flag.is_set():
            raise AssertionError("flag should still be set")


if __name__ == "__main__":
    run_tests()
