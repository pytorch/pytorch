# Owner(s): ["module: inductor"]

"""Calls of one compiled callable from several threads at once.

Aimed at free-threaded builds, but valid on the default build too.
"""

import contextvars
import sys
import sysconfig
import threading
import time
import traceback
from collections.abc import Callable

import torch
import torch._dynamo.config
import torch._inductor.config
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import HardwareClassification
from torch.testing._internal.triton_utils import requires_cuda_and_triton


NTHREADS = 8
ITERS = 50
# Under the runner's 10 minute timeout, so a deadlock fails with stacks.
TIMEOUT = 300


def all_stacks() -> str:
    return "\n".join(
        f"Thread {ident}:\n{''.join(traceback.format_stack(frame))}"
        for ident, frame in sys._current_frames().items()
    )


def run_threads(body: Callable[[int], None]) -> None:
    """Runs ``body(tid)`` on ``NTHREADS`` threads at once and re-raises the
    first failure, or fails with every thread's stack if they are still running
    after ``TIMEOUT`` seconds.
    """
    barrier = threading.Barrier(NTHREADS)
    contexts = [contextvars.copy_context() for _ in range(NTHREADS)]
    failures: list[BaseException] = []

    def worker(tid: int) -> None:
        try:
            barrier.wait()
            contexts[tid].run(body, tid)
        except BaseException as e:
            failures.append(e)

    threads = [
        threading.Thread(target=worker, args=(i,), daemon=True) for i in range(NTHREADS)
    ]
    for t in threads:
        t.start()
    deadline = time.monotonic() + TIMEOUT
    for t in threads:
        t.join(max(0.0, deadline - time.monotonic()))
    if any(t.is_alive() for t in threads):
        raise AssertionError(f"threads still running after {TIMEOUT}s\n{all_stacks()}")
    if failures:
        raise failures[0]
    if sysconfig.get_config_var("Py_GIL_DISABLED") and sys._is_gil_enabled():
        raise AssertionError("an extension module enabled the GIL")


def pointwise(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x.sin() * y + 1).relu()


def reduction(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x * y).softmax(dim=-1)


# In fbcode, even checking for a remote cache loads its client, and with it
# boost.python, whose destructors crash a free-threaded interpreter at exit.
@torch._inductor.config.patch(fx_graph_remote_cache=False, autotune_remote_cache=False)
@torch._dynamo.config.patch(automatic_dynamic_remote_pgo=False)
class FreeThreadingTestBase(TestCase):
    def check_shared_callable(self, device: str, warm: bool) -> None:
        """Calls each compiled function from every thread, each with its own
        inputs."""
        inputs = [
            (torch.randn(64, 512, device=device), torch.randn(64, 512, device=device))
            for _ in range(NTHREADS)
        ]
        for fn in (pointwise, reduction):
            compiled = torch.compile(fn, dynamic=False)
            if warm:
                compiled(*inputs[0])

            def body(tid: int) -> None:
                expected = fn(*inputs[tid])
                for _ in range(ITERS):
                    self.assertEqual(compiled(*inputs[tid]), expected)

            run_threads(body)


class FreeThreadingTest(FreeThreadingTestBase):
    hw_classification = HardwareClassification.GENERIC

    def test_warm_cpu(self) -> None:
        self.check_shared_callable("cpu", warm=True)

    def test_cold_cpu(self) -> None:
        self.check_shared_callable("cpu", warm=False)


class FreeThreadingCudaTest(FreeThreadingTestBase):
    hw_classification = HardwareClassification.CUDA

    @requires_cuda_and_triton
    @torch._inductor.config.patch(
        use_static_cuda_launcher=True, use_fast_triton_launcher=True
    )
    def test_warm_gpu(self) -> None:
        self.check_shared_callable("cuda", warm=True)


if __name__ == "__main__":
    run_tests()
