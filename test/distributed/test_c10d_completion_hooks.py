# Owner(s): ["oncall: distributed"]
#
# Tests for ProcessGroup.register_completion_hook.

import gc
import os
import signal
import subprocess
import sys
import tempfile
import threading
import unittest
import weakref
from datetime import timedelta
from unittest import mock

import torch
import torch.distributed as dist


if not dist.is_available():
    print("distributed package not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.testing._internal.common_distributed import MultiProcContinuousTest
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TEST_CUDA,
    TestCase,
)


TIMEOUT = 60

requires_nccl2 = unittest.skipIf(
    not dist.is_backend_available("nccl2"), "nccl2 backend is not available"
)


class _Collector:
    """Records completion_key -> duration_ms; waits without sleeping."""

    def __init__(self):
        self.cv = threading.Condition()
        self.calls = {}

    def __call__(self, args):
        with self.cv:
            self.calls[args.completion_key] = args.duration_ms
            self.cv.notify_all()

    def wait_for(self, keys):
        """Returns the durations of ``keys``; other keys may be reported too."""
        with self.cv:
            ok = self.cv.wait_for(lambda: set(keys) <= self.calls.keys(), TIMEOUT)
            calls = dict(self.calls)
        if not ok:
            raise AssertionError(f"expected {keys}, got {calls}")
        return [calls[key] for key in keys]


@unittest.skipIf(not TEST_CUDA, "requires a GPU")
@requires_nccl2
class CompletionHookNccl2Test(MultiProcContinuousTest):
    hw_classification = HardwareClassification.CUDA
    world_size = 1
    timeout = timedelta(seconds=TIMEOUT)

    @classmethod
    def backend_str(cls):
        return "nccl2"

    @classmethod
    def device_type(cls):
        return "cuda"

    @classmethod
    def _init_pg(cls, rank, world_size, rdvz_file):
        os.environ.pop("TORCH_NCCL_ENABLE_TIMING", None)
        os.environ.pop("NCCL_ENABLE_TIMING", None)
        torch.cuda.set_device(rank)
        super()._init_pg(rank, world_size, rdvz_file)

    @property
    def device(self):
        return torch.device("cuda", self.rank)

    def _register(self, pg, hook_id, hook):
        pg.register_completion_hook(hook_id, hook)
        self.addCleanup(pg.unregister_completion_hook, hook_id)

    def _all_reduce(self, group=None):
        t = torch.ones(1024, device=self.device)
        return dist.all_reduce(t, group=group, async_op=True).completion_key

    def test_correlates_with_post_hook(self):
        pg = dist.group.WORLD
        self.assertTrue(pg.supports_completion_hooks)
        ops = {}
        pg.register_post_hook(
            0, lambda args: ops.__setitem__(args.work.completion_key, args.name)
        )
        self.addCleanup(pg.unregister_post_hook, 0)
        collector = _Collector()
        self._register(pg, 0, collector)

        t = torch.ones(8, device=self.device)
        dist.all_reduce(t)
        dist.broadcast(t, src=0)
        torch.cuda.synchronize()

        self.assertEqual(
            sorted(op.name for op in ops.values()), ["ALLREDUCE", "BROADCAST"]
        )
        # nccl2 does not time ops unless asked to.
        self.assertEqual(collector.wait_for(list(ops)), [None, None])

    def test_duration_with_timing(self):
        # Own group: timing cannot be turned off again.
        pg = dist.new_group(backend=self.backend_str())
        self.addCleanup(dist.destroy_process_group, pg)
        pg._enable_collectives_timing()
        collector = _Collector()
        self._register(pg, 0, collector)
        keys = [self._all_reduce(pg) for _ in range(3)]

        for duration in collector.wait_for(keys):
            self.assertIsInstance(duration, float)
            self.assertGreaterEqual(duration, 0.0)
            self.assertLess(duration, TIMEOUT * 1000.0)

    def test_unregister(self):
        pg = dist.group.WORLD
        reference = _Collector()
        self._register(pg, 0, reference)

        first = _Collector()
        released = threading.Event()
        weakref.finalize(first, released.set)
        self._register(pg, 1, first)
        with self.assertRaisesRegex(ValueError, "already registered"):
            pg.register_completion_hook(1, _Collector())
        first.wait_for([self._all_reduce()])
        pg.unregister_completion_hook(1)
        # Unknown ids are ignored.
        pg.unregister_completion_hook(1)
        key = self._all_reduce()
        reference.wait_for([key])
        self.assertNotIn(key, first.calls)
        # Unregister released the callable.
        del first
        self.assertTrue(released.is_set())

        # Unregister from the hook itself releases it once it returns.
        released.clear()

        def remove_self(args):
            pg.unregister_completion_hook(2)

        self._register(pg, 2, remove_self)
        weakref.finalize(remove_self, released.set)
        del remove_self
        self._all_reduce()
        self.assertTrue(released.wait(TIMEOUT))

        # The id can be reused.
        again = _Collector()
        self._register(pg, 1, again)
        again.wait_for([self._all_reduce()])

    def test_raising_hook(self):
        pg = dist.group.WORLD
        collector = _Collector()

        def hook(args):
            collector(args)
            raise ValueError("hook failed")

        errors = []
        unraisablehook = sys.unraisablehook

        def record(unraisable):
            if str(unraisable.exc_value) != "hook failed":
                unraisablehook(unraisable)
                return
            errors.append(unraisable)

        with mock.patch.object(sys, "unraisablehook", record):
            self._register(pg, 0, hook)
            collector.wait_for([self._all_reduce(), self._all_reduce()])
            # Waits for the last call to report its exception.
            pg.unregister_completion_hook(0)
        self.assertGreaterEqual(len(errors), 2)
        for error in errors:
            self.assertIsInstance(error.exc_value, ValueError)

    def test_unregister_waits_for_running_call(self):
        pg = dist.group.WORLD
        entered = threading.Event()
        proceed = threading.Event()
        finished = threading.Event()

        def hook(args):
            entered.set()
            proceed.wait(TIMEOUT)
            finished.set()

        self._register(pg, 0, hook)
        self._all_reduce()
        self.assertTrue(entered.wait(TIMEOUT))
        finished_on_return = []

        def unregister():
            pg.unregister_completion_hook(0)
            finished_on_return.append(finished.is_set())

        thread = threading.Thread(target=unregister)
        thread.start()
        try:
            thread.join(1)
            self.assertTrue(thread.is_alive())
        finally:
            proceed.set()
        thread.join(TIMEOUT)
        self.assertEqual(finished_on_return, [True])

    def test_destroyed_group_released(self):
        pg = dist.new_group(backend=self.backend_str())
        hook = _Collector()
        released = threading.Event()
        weakref.finalize(hook, released.set)
        pg.register_completion_hook(0, hook)
        del hook
        dist.destroy_process_group(pg)
        del pg
        gc.collect()
        # The next registration stops hooks of destroyed groups.
        self._register(dist.group.WORLD, 0, _Collector())
        self.assertTrue(released.wait(TIMEOUT))


@unittest.skipIf(torch.cuda.device_count() < 2, "requires 2 GPUs")
@unittest.skipIf(
    not dist.is_backend_available("nccl-lazy"), "nccl-lazy backend is not available"
)
class CompletionHookNcclLazyP2PTest(MultiProcContinuousTest):
    hw_classification = HardwareClassification.CUDA
    world_size = 2
    timeout = timedelta(seconds=TIMEOUT)

    @classmethod
    def backend_str(cls):
        return "nccl-lazy"

    @classmethod
    def device_type(cls):
        return "cuda"

    @classmethod
    def _init_pg(cls, rank, world_size, rdvz_file):
        torch.cuda.set_device(rank)
        super()._init_pg(rank, world_size, rdvz_file)

    def _p2p(self):
        t = torch.ones(1024, device=torch.device("cuda", self.rank))
        if self.rank == 0:
            work = dist.isend(t, dst=1)
        else:
            work = dist.irecv(t, src=0)
        work.wait()
        return work.completion_key

    def test_p2p(self):
        pg = dist.group.WORLD
        # Registered before the pair comm exists: fanned out when it is created.
        before = _Collector()
        pg.register_completion_hook(0, before)
        self.addCleanup(pg.unregister_completion_hook, 0)
        first = self._p2p()
        before.wait_for([first])

        # Registered after: fanned out to the existing pair comm.
        after = _Collector()
        pg.register_completion_hook(1, after)
        second = self._p2p()
        before.wait_for([second])
        after.wait_for([second])

        pg.unregister_completion_hook(1)
        third = self._p2p()
        before.wait_for([third])
        self.assertNotIn(third, after.calls)


@unittest.skipIf(not dist.is_gloo_available(), "gloo is not available")
class CompletionHookUnsupportedTest(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_gloo_raises(self):
        dist.init_process_group("gloo", world_size=1, rank=0, store=dist.HashStore())
        self.addCleanup(dist.destroy_process_group)
        pg = dist.group.WORLD
        self.assertFalse(pg.supports_completion_hooks)
        with self.assertRaisesRegex(RuntimeError, "does not support"):
            pg.register_completion_hook(0, lambda args: None)
        with self.assertRaisesRegex(TypeError, "callable"):
            pg.register_completion_hook(0, 1)


_EXIT_SCRIPT = """
import atexit
import os
import sys

released = False

def late_register():
    # Registered before torch's exit handler, so it runs after it.
    if released:
        print("released before late register", flush=True)
    try:
        dist.group.WORLD.register_completion_hook(1, lambda args: None)
    except RuntimeError as e:
        if "shutting down" in str(e):
            print("late register raised", flush=True)

atexit.register(late_register)

import torch
import torch.distributed as dist

torch.cuda.set_device(0)
dist.init_process_group(
    "nccl2", rank=0, world_size=1, store=dist.FileStore(sys.argv[1], 1)
)

class Hook:
    def __call__(self, args):
        pass

    def __del__(self):
        global released
        released = True

dist.group.WORLD.register_completion_hook(0, Hook())
t = torch.ones(1024, device="cuda")
for _ in range(100):
    dist.all_reduce(t)
if sys.argv[2] == "fork":
    torch.cuda.synchronize()
    # The child has none of the parent's hook threads; exit must not wait for
    # them.
    pid = os.fork()
    if pid == 0:
        atexit._run_exitfuncs()
        os._exit(0)
    if os.waitpid(pid, 0)[1] == 0:
        print("child exited", flush=True)
    # nccl2 itself does not survive the parent's exit after a fork.
    os._exit(0)
"""


@unittest.skipIf(not TEST_CUDA, "requires a GPU")
@requires_nccl2
class CompletionHookExitTest(TestCase):
    hw_classification = HardwareClassification.CUDA

    def _run(self, mode, *expected):
        with tempfile.TemporaryDirectory() as tmp:
            # Own session, so a timeout also kills a forked child.
            proc = subprocess.Popen(
                [sys.executable, "-c", _EXIT_SCRIPT, os.path.join(tmp, "store"), mode],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=TIMEOUT * 2)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.communicate()
                raise
        self.assertEqual(proc.returncode, 0, msg=stderr)
        for e in expected:
            self.assertIn(e, stdout, msg=stderr)

    def test_exit_with_registered_hook(self):
        self._run("exit", "released before late register", "late register raised")

    def test_exit_forked_child(self):
        self._run("fork", "child exited")


if __name__ == "__main__":
    run_tests()
