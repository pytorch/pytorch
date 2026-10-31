# Owner(s): ["oncall: distributed"]
#
# Tests for torch._C._distributed_c10d.CompletionCallbackHook: per-op
# completion callbacks built on the ProcessGroup post and completion hooks.

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

from torch._C._distributed_c10d import CompletionCallbackHook, HookOpName
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    MultiProcessTestCase,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TEST_CUDA,
    TestCase,
)
from torch.testing._internal.distributed.fake_pg import FakeStore


TIMEOUT = 60

requires_2_gpus = unittest.skipIf(
    not TEST_CUDA or torch.cuda.device_count() < 2, "requires at least 2 GPUs"
)
requires_nccl2 = unittest.skipIf(
    not dist.is_backend_available("nccl2"), "nccl2 backend is not available"
)


class _Collector:
    """Records (name, op_id, duration_ms) per callback; waits without sleeping."""

    def __init__(self, on_info=None):
        self.cv = threading.Condition()
        self.infos = []
        self.on_info = on_info

    def __call__(self, info):
        with self.cv:
            self.infos.append((info.name, info.op_id, info.duration_ms))
            self.cv.notify_all()
        if self.on_info is not None:
            self.on_info(info)

    def wait_for(self, count, timeout=TIMEOUT):
        with self.cv:
            ok = self.cv.wait_for(lambda: len(self.infos) >= count, timeout)
            infos = list(self.infos)
        if not ok:
            raise AssertionError(f"expected {count} callbacks, got {infos}")
        return infos

    def names(self):
        with self.cv:
            return [name for name, _, _ in self.infos]


class _CompletionCallbackHookTestBase(MultiProcessTestCase):
    @property
    def world_size(self):
        return 2

    def setUp(self):
        super().setUp()
        # Tests that expect untimed ops must not inherit timing from the env.
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        for var in ("TORCH_NCCL_ENABLE_TIMING", "NCCL_ENABLE_TIMING"):
            os.environ.pop(var, None)
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def _init_pg(self, backend):
        dist.init_process_group(
            backend,
            world_size=self.world_size,
            rank=self.rank,
            store=dist.FileStore(self.file_name, self.world_size),
            timeout=timedelta(seconds=TIMEOUT),
        )
        return dist.group.WORLD


@requires_2_gpus
@requires_nccl2
class CompletionCallbackHookNccl2Test(MultiProcContinuousTest):
    """Tests whose callbacks never block share one group."""

    hw_classification = HardwareClassification.CUDA
    world_size = 2
    timeout = timedelta(seconds=TIMEOUT)

    @classmethod
    def backend_str(cls):
        return "nccl2"

    @classmethod
    def device_type(cls):
        return "cuda"

    @classmethod
    def _init_pg(cls, rank, world_size, rdvz_file):
        # Tests that expect untimed ops must not inherit timing from the env.
        for var in ("TORCH_NCCL_ENABLE_TIMING", "NCCL_ENABLE_TIMING"):
            os.environ.pop(var, None)
        torch.cuda.set_device(rank)
        super()._init_pg(rank, world_size, rdvz_file)

    @property
    def device(self):
        return torch.device("cuda", self.rank)

    def _init(self):
        return dist.group.WORLD

    def _attach(self, pg, callback):
        hook = CompletionCallbackHook.attach(pg, callback)
        self.addCleanup(hook.remove)
        return hook

    def _sync_all_reduce(self, group=None):
        dist.all_reduce(torch.ones(4, device=self.device), group=group)
        torch.cuda.synchronize()

    def test_reports_collectives(self):
        pg = self._init()
        collector = _Collector()
        hook = self._attach(pg, collector)
        self.assertEqual(hook.last_op_id, -1)

        n = self.world_size
        shard = torch.ones(8, device=self.device)
        flat = torch.ones(8 * n, device=self.device)
        dist.all_reduce(shard)
        dist.broadcast(shard, src=0)
        dist.all_gather_into_tensor(torch.empty_like(flat), shard)
        dist.reduce_scatter_tensor(torch.empty_like(shard), flat)
        torch.cuda.synchronize()

        # Report order is not guaranteed: nccl2 may report from two threads.
        infos = sorted(collector.wait_for(4), key=lambda info: info[1])
        self.assertEqual(
            [name for name, _, _ in infos],
            [
                HookOpName.ALLREDUCE,
                HookOpName.BROADCAST,
                HookOpName.ALLGATHER_BASE,
                HookOpName.REDUCE_SCATTER_BASE,
            ],
        )
        op_ids = [op_id for _, op_id, _ in infos]
        self.assertEqual(op_ids, sorted(set(op_ids)))
        self.assertEqual(hook.last_op_id, op_ids[-1])
        # nccl2 does not time ops unless asked to.
        self.assertEqual([d for _, _, d in infos], [None] * 4)

    def test_duration_from_backend_timing(self):
        # Own group: timing cannot be turned off again.
        pg = dist.new_group(backend="nccl2")
        self.addCleanup(dist.destroy_process_group, pg)
        # Before the first op: works created earlier carry untimed events.
        pg._enable_collectives_timing()
        collector = _Collector()
        self._attach(pg, collector)
        t = torch.ones(1024, device=self.device)
        for _ in range(3):
            dist.all_reduce(t, group=pg)
        torch.cuda.synchronize()

        for name, _, duration in collector.wait_for(3):
            self.assertEqual(name, HookOpName.ALLREDUCE)
            self.assertIsInstance(duration, float)
            self.assertGreater(duration, 0.0)
            self.assertLess(duration, TIMEOUT * 1000.0)

    def test_last_op_id_attributes_batches(self):
        # A caller marks the first op of a batch as last_op_id + 1 and
        # attributes every reported op_id at or above it to that batch.
        pg = self._init()
        collector = _Collector()
        hook = self._attach(pg, collector)
        self._sync_all_reduce()
        collector.wait_for(1)

        batch_start = hook.last_op_id + 1
        for _ in range(3):
            dist.all_reduce(torch.ones(4, device=self.device))
        torch.cuda.synchronize()
        batch = collector.wait_for(4)[1:]
        self.assertEqual(
            sorted(op_id for _, op_id, _ in batch),
            list(range(batch_start, batch_start + 3)),
        )
        self.assertEqual(hook.last_op_id, batch_start + 2)

    def test_reports_p2p(self):
        pg = self._init()
        collector = _Collector()
        self._attach(pg, collector)
        t = torch.ones(4, device=self.device)
        if self.rank == 0:
            dist.send(t, dst=1)
        else:
            dist.recv(t, src=0)
        torch.cuda.synchronize()
        expected = HookOpName.SEND if self.rank == 0 else HookOpName.RECV
        self.assertEqual(collector.wait_for(1)[0][0], expected)

    def test_coalescing(self):
        pg = self._init()
        collector = _Collector()
        self._attach(pg, collector)

        # Each phase is waited for before the next: async and synchronous ops
        # run on different streams, so their completions may be reported in
        # either order.
        # Collectives in a window are reported one by one.
        pg._start_coalescing(self.device)
        for _ in range(3):
            dist.all_reduce(torch.ones(4, device=self.device), async_op=True)
        pg._end_coalescing(self.device).wait()
        torch.cuda.synchronize()
        collector.wait_for(3)
        # A *_coalesced op is reported once.
        dist.all_reduce_coalesced([torch.ones(4, device=self.device)] * 2)
        torch.cuda.synchronize()
        collector.wait_for(4)
        # p2p in a window completes only as the window, which no post-hook
        # sees, so it is not reported; later ops still are.
        peer = 1 - self.rank
        ops = [
            dist.P2POp(dist.isend, torch.ones(4, device=self.device), peer),
            dist.P2POp(dist.irecv, torch.empty(4, device=self.device), peer),
        ]
        for work in dist.batch_isend_irecv(ops):
            work.wait()
        torch.cuda.synchronize()
        dist.all_reduce(torch.ones(4, device=self.device))
        torch.cuda.synchronize()

        collector.wait_for(5)
        self.assertEqual(
            collector.names(),
            [HookOpName.ALLREDUCE] * 3
            + [HookOpName.ALLREDUCE_COALESCED, HookOpName.ALLREDUCE],
        )

    def test_completion_before_post_hook(self):
        # A fast op can complete before the post-hook maps its completion key.
        # Constructed, not raced: post-hooks run in hook_id order, so the
        # hand-registered hook below runs before the callback hook's and holds
        # the window open until a sentinel issued behind the target has been
        # reported. Both run on the current stream, which the backend retires
        # in order, so the target's completion has arrived by then, before its
        # key was registered.
        pg = self._init()
        collector = _Collector()
        self._attach(pg, collector)
        self._sync_all_reduce()
        collector.wait_for(1)
        state = {"running": False}

        def hold_post_hook_open(args):
            if args.work is None or state["running"]:
                return
            state["running"] = True
            try:
                dist.all_reduce(torch.ones(8, device=self.device))
                collector.wait_for(2)
            finally:
                state["running"] = False

        pg.register_post_hook(1, hold_post_hook_open)
        try:
            dist.broadcast(torch.ones(4, device=self.device), src=0)
            torch.cuda.synchronize()
        finally:
            pg.unregister_post_hook(1)

        infos = collector.wait_for(3)
        self.assertEqual(
            [name for name, _, _ in infos[1:]],
            [HookOpName.ALLREDUCE, HookOpName.BROADCAST],
        )
        # op_ids are assigned at issue, so the sentinel's is the later one.
        self.assertGreater(infos[1][1], infos[2][1])

    def test_non_default_backend_not_reported(self):
        # Completion hooks are registered on the default backend only.
        pg = dist.new_group(backend="cpu:gloo,cuda:nccl2")
        self.addCleanup(dist.destroy_process_group, pg)
        collector = _Collector()
        hook = self._attach(pg, collector)
        dist.all_reduce(torch.ones(4), group=pg)
        self._sync_all_reduce(group=pg)
        infos = collector.wait_for(1)
        self.assertEqual(infos, [(HookOpName.ALLREDUCE, hook.last_op_id, None)])

    def test_cuda_graph_not_reported(self):
        # The backend does not track captured ops, and replays issue no op.
        pg = dist.new_group(backend="nccl2")
        self.addCleanup(dist.destroy_process_group, pg)
        collector = _Collector()
        hook = self._attach(pg, collector)
        t = torch.ones(4, device=self.device)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        # Warm up outside the capture.
        with torch.cuda.stream(stream):
            dist.all_reduce(t, group=pg)
        torch.cuda.synchronize()
        collector.wait_for(1)
        graph = torch.cuda.CUDAGraph()
        # Release NCCL graph resources before the group is destroyed.
        self.addCleanup(graph.reset)
        with torch.cuda.graph(graph, stream=stream):
            dist.all_reduce(t, group=pg)
        captured = hook.last_op_id
        self.assertGreater(captured, collector.wait_for(1)[0][1])
        graph.replay()
        self._sync_all_reduce(group=pg)
        # One more on the capture stream, so a late report of the captured op
        # would arrive before this one.
        with torch.cuda.stream(stream):
            dist.all_reduce(t, group=pg)
        torch.cuda.synchronize()
        infos = collector.wait_for(3)
        self.assertEqual(len(infos), 3)
        self.assertNotIn(captured, [op_id for _, op_id, _ in infos])
        self.assertEqual(max(op_id for _, op_id, _ in infos), hook.last_op_id)

    def test_callback_exception_is_unraisable(self):
        pg = self._init()
        unraisable = []
        unraisable_cv = threading.Condition()

        def record_unraisable(args):
            with unraisable_cv:
                unraisable.append(args.exc_value)
                unraisable_cv.notify_all()

        def raise_once(info):
            if not unraisable:
                raise ValueError("callback failed")

        collector = _Collector(on_info=raise_once)
        old_hook = sys.unraisablehook
        sys.unraisablehook = record_unraisable
        try:
            self._attach(pg, collector)
            self._sync_all_reduce()
            with unraisable_cv:
                self.assertTrue(unraisable_cv.wait_for(lambda: unraisable, TIMEOUT))
            self.assertIsInstance(unraisable[0], ValueError)
            # Delivery continues after a failed callback.
            self._sync_all_reduce()
            collector.wait_for(2)
            self.assertEqual(len(unraisable), 1)
        finally:
            sys.unraisablehook = old_hook

    def test_remove_stops_delivery(self):
        pg = self._init()
        removed = _Collector()
        hook = self._attach(pg, removed)
        sentinel = _Collector()
        self._attach(pg, sentinel)
        self._sync_all_reduce()
        removed.wait_for(1)
        sentinel.wait_for(1)

        hook.remove()
        hook.remove()  # idempotent
        last = hook.last_op_id
        self._sync_all_reduce()
        sentinel.wait_for(2)
        # Post-hooks run at issue, so this is deterministic.
        self.assertEqual(hook.last_op_id, last)
        self.assertEqual(len(removed.names()), 1)

    def test_remove_from_callback(self):
        pg = self._init()
        handle = []
        removed_done = threading.Event()
        attach_errors = []

        def remove_self(info):
            try:
                CompletionCallbackHook.attach(pg, _Collector())
            except RuntimeError as e:
                attach_errors.append(str(e))
            handle[0].remove()
            removed_done.set()

        removed = _Collector(on_info=remove_self)
        handle.append(self._attach(pg, removed))
        sentinel = _Collector()
        self._attach(pg, sentinel)
        self._sync_all_reduce()
        self.assertTrue(removed_done.wait(TIMEOUT))
        self.assertEqual(len(attach_errors), 1)
        self.assertIn("from a completion callback", attach_errors[0])
        # The removed hook stays registered but inert.
        last = handle[0].last_op_id
        self._sync_all_reduce()
        sentinel.wait_for(2)
        self.assertGreater(handle[0].last_op_id, last)
        self.assertEqual(len(removed.names()), 1)
        # A later remove() outside a callback unregisters it.
        handle[0].remove()
        last = handle[0].last_op_id
        self._sync_all_reduce()
        self.assertEqual(handle[0].last_op_id, last)


@requires_2_gpus
@requires_nccl2
class CompletionCallbackHookNccl2BlockingTest(_CompletionCallbackHookTestBase):
    """Tests with blocked callbacks, each in fresh processes."""

    hw_classification = HardwareClassification.CUDA

    @property
    def device(self):
        return torch.device("cuda", self.rank)

    def _init(self):
        torch.cuda.set_device(self.rank)
        return self._init_pg("nccl2")

    def _sync_all_reduce(self, group=None):
        dist.all_reduce(torch.ones(4, device=self.device), group=group)
        torch.cuda.synchronize()

    def test_remove_waits_for_running_callback(self):
        pg = self._init()
        entered = threading.Event()
        release = threading.Event()
        finished = threading.Event()

        def block(info):
            if entered.is_set():
                return
            entered.set()
            release.wait()
            finished.set()

        blocked = _Collector(on_info=block)
        hook = CompletionCallbackHook.attach(pg, blocked)
        sentinel = _Collector()
        sentinel_hook = CompletionCallbackHook.attach(pg, sentinel)
        remover = threading.Thread(target=hook.remove, daemon=True)
        try:
            for _ in range(3):
                dist.all_reduce(torch.ones(4, device=self.device))
            torch.cuda.synchronize()
            self.assertTrue(entered.wait(TIMEOUT))
            sentinel.wait_for(3)
            remover.start()
            # Bounded, and only fails if remove() returns early.
            remover.join(0.5)
            self.assertTrue(remover.is_alive())
        finally:
            release.set()
        remover.join(TIMEOUT)
        self.assertFalse(remover.is_alive())
        self.assertTrue(finished.is_set())
        # Queued and later callbacks are dropped.
        self._sync_all_reduce()
        sentinel.wait_for(4)
        self.assertEqual(len(blocked.names()), 1)
        sentinel_hook.remove()
        dist.destroy_process_group()

    def test_abort_with_blocked_callback(self):
        self._init()
        sub = dist.new_group(backend="nccl2", use_local_synchronization=True)
        entered = threading.Event()
        release = threading.Event()

        def block(info):
            entered.set()
            release.wait()

        blocked = _Collector(on_info=block)
        hook = CompletionCallbackHook.attach(sub, blocked)
        try:
            dist.all_reduce(torch.ones(4, device=self.device), group=sub)
            torch.cuda.synchronize()
            self.assertTrue(entered.wait(TIMEOUT))

            # Neither the backend nor abort waits on a running callback.
            aborter = threading.Thread(
                target=dist.distributed_c10d._abort_process_group,
                args=(sub,),
                daemon=True,
            )
            aborter.start()
            aborter.join(TIMEOUT)
            self.assertFalse(aborter.is_alive(), "abort waited on the callback")
            self._sync_all_reduce()
        finally:
            release.set()
        hook.remove()
        self.assertEqual(blocked.names(), [HookOpName.ALLREDUCE])
        dist.destroy_process_group()

    def test_destroy_releases_hook(self):
        # The group owns the hook. Destroying and dropping the group while a
        # callback is blocked waits on neither it nor remove(): queued
        # callbacks are still delivered, then the callback is released, even
        # though it references the handle (a cycle the GC cannot see).
        self._init()
        sub = dist.new_group(backend="nccl2", use_local_synchronization=True)
        entered = threading.Event()
        release = threading.Event()
        freed = threading.Event()

        def block(info):
            entered.set()
            release.wait()

        blocked = _Collector(on_info=block)
        handles = []
        callback = lambda info, handles=handles: blocked(info)  # noqa: E731
        weakref.finalize(callback, freed.set)
        handles.append(CompletionCallbackHook.attach(sub, callback))
        del callback, handles
        try:
            for _ in range(3):
                dist.all_reduce(torch.ones(4, device=self.device), group=sub)
            torch.cuda.synchronize()
            self.assertTrue(entered.wait(TIMEOUT))
            destroyer = threading.Thread(
                target=dist.destroy_process_group, args=(sub,), daemon=True
            )
            destroyer.start()
            destroyer.join(TIMEOUT)
            self.assertFalse(destroyer.is_alive(), "destroy waited on the callback")
            del sub, destroyer
            gc.collect()
            # Still running.
            self.assertFalse(freed.is_set())
        finally:
            release.set()
        self.assertEqual(
            [name for name, _, _ in blocked.wait_for(3)], [HookOpName.ALLREDUCE] * 3
        )
        self.assertTrue(freed.wait(TIMEOUT), "callback not released")
        self._sync_all_reduce()
        dist.destroy_process_group()


@requires_2_gpus
@unittest.skipIf(
    not dist.is_backend_available("nccl-lazy"), "nccl-lazy backend is not available"
)
class CompletionCallbackHookNcclLazyTest(_CompletionCallbackHookTestBase):
    hw_classification = HardwareClassification.CUDA

    def test_reports_ops(self):
        torch.cuda.set_device(self.rank)
        pg = self._init_pg("nccl-lazy")
        collector = _Collector()
        hook = CompletionCallbackHook.attach(pg, collector)
        device = torch.device("cuda", self.rank)
        dist.all_reduce(torch.ones(4, device=device))
        dist.broadcast(torch.ones(4, device=device), src=0)
        # p2p runs on a separate pair communicator.
        t = torch.ones(4, device=device)
        if self.rank == 0:
            dist.send(t, dst=1)
        else:
            dist.recv(t, src=0)
        torch.cuda.synchronize()
        infos = sorted(collector.wait_for(3), key=lambda info: info[1])
        self.assertEqual(
            [name for name, _, _ in infos],
            [
                HookOpName.ALLREDUCE,
                HookOpName.BROADCAST,
                HookOpName.SEND if self.rank == 0 else HookOpName.RECV,
            ],
        )
        self.assertEqual(infos[-1][1], hook.last_op_id)
        hook.remove()
        dist.destroy_process_group()


class CompletionCallbackHookUnsupportedTest(TestCase):
    """Backends without completion hooks raise, so callers can fall back."""

    hw_classification = HardwareClassification.GENERIC

    def _init_pg(self, backend, store):
        dist.init_process_group(backend, world_size=1, rank=0, store=store)
        self.addCleanup(dist.destroy_process_group)
        return dist.group.WORLD

    def test_gloo_raises(self):
        pg = self._init_pg("gloo", dist.HashStore())
        self.assertFalse(pg.supports_completion_hooks)
        with self.assertRaisesRegex(RuntimeError, "does not support completion hooks"):
            CompletionCallbackHook.attach(pg, lambda info: None)

    def test_rejects_non_callable(self):
        pg = self._init_pg("gloo", dist.HashStore())
        with self.assertRaisesRegex(TypeError, "callable"):
            CompletionCallbackHook.attach(pg, 1)

    def test_fake_raises(self):
        pg = self._init_pg("fake", FakeStore())
        with self.assertRaisesRegex(RuntimeError, "does not support completion hooks"):
            CompletionCallbackHook.attach(pg, lambda info: None)


@unittest.skipIf(not TEST_CUDA, "requires a GPU")
@unittest.skipIf(not dist.is_nccl_available(), "nccl is not available")
class CompletionCallbackHookLegacyNcclTest(TestCase):
    hw_classification = HardwareClassification.CUDA

    def test_legacy_nccl_raises(self):
        dist.init_process_group(
            "nccl-legacy", world_size=1, rank=0, store=dist.HashStore()
        )
        self.addCleanup(dist.destroy_process_group)
        pg = dist.group.WORLD
        self.assertFalse(pg.supports_completion_hooks)
        with self.assertRaisesRegex(RuntimeError, "does not support completion hooks"):
            CompletionCallbackHook.attach(pg, lambda info: None)


_SHUTDOWN_SCRIPT = """
import atexit
import os
import sys
import threading

mode = sys.argv[2]

def late_attach():
    # Registered before torch's handler, so it runs after it.
    try:
        CompletionCallbackHook.attach(dist.group.WORLD, lambda info: None)
    except RuntimeError as e:
        if "exiting" in str(e):
            print("late attach raised", flush=True)

if mode == "exit":
    atexit.register(late_attach)

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import CompletionCallbackHook

torch.cuda.set_device(0)
dist.init_process_group(
    "nccl2", rank=0, world_size=1, store=dist.FileStore(sys.argv[1], 1)
)
entered = threading.Event()
release = threading.Event()

class Callback:
    def __call__(self, info):
        entered.set()
        if mode in ("blocked", "fork"):
            release.wait()

    # Runs only if the callback is released before the interpreter finalizes.
    def __del__(self):
        print("released", flush=True)

hook = CompletionCallbackHook.attach(dist.group.WORLD, Callback())
t = torch.ones(1024, device="cuda")
# fork: one op, so none is still being reported when the child is forked.
for _ in range(1 if mode == "fork" else 100):
    dist.all_reduce(t)
if mode == "blocked":
    # Exit with a callback running and more queued behind it. Registered after
    # torch's handler, so it runs first.
    torch.cuda.synchronize()
    entered.wait()
    atexit.register(release.set)
if mode == "destroy":
    dist.destroy_process_group()
if mode == "fork":
    # The child has none of the parent's dispatch threads, one of which is in a
    # callback; neither remove() nor exit may wait for them.
    torch.cuda.synchronize()
    entered.wait()
    pid = os.fork()
    if pid == 0:
        hook.remove()
        sys.exit(0)
    if os.waitpid(pid, 0)[1] == 0:
        print("child exited", flush=True)
    # nccl2 itself does not survive the parent's exit after a fork.
    os._exit(0)
"""


@unittest.skipIf(not TEST_CUDA, "requires a GPU")
@requires_nccl2
class CompletionCallbackHookShutdownTest(TestCase):
    hw_classification = HardwareClassification.CUDA

    def _run(self, mode, *expected):
        # Exits with ops in flight and callbacks queued or running.
        with tempfile.TemporaryDirectory() as tmp:
            # Own session, so a timeout also kills a forked child.
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    _SHUTDOWN_SCRIPT,
                    os.path.join(tmp, "store"),
                    mode,
                ],
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
        for e in expected or ("released",):
            self.assertIn(e, stdout, msg=stderr)

    def test_exit_without_destroy(self):
        self._run("exit", "released", "late attach raised")

    def test_exit_after_destroy(self):
        self._run("destroy")

    def test_exit_with_running_callback(self):
        self._run("blocked")

    def test_exit_forked_child(self):
        self._run("fork", "child exited")


if __name__ == "__main__":
    run_tests()
