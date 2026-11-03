# Owner(s): ["oncall: distributed"]

import os
import subprocess
import sys
import tempfile
import threading
import weakref
from unittest import mock

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import HookOpName
from torch.distributed._hooks.collective_annotations import (
    register_collective_annotation_hooks,
    unregister_collective_annotation_hooks,
)
from torch.distributed.distributed_c10d import _get_default_group
from torch.testing._internal.common_distributed import MultiProcContinuousTest
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)


# The annotation hooks only register on NCCL groups; let them hook these Gloo groups.
_PGS_TO_ANNOTATE = "torch.distributed._hooks.collective_annotations._pgs_to_annotate"

TIMEOUT = 60


class _CompletionCollector:
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


class TestProcessGroupHooks(MultiProcContinuousTest):
    """
    Verify that pre/post collective hooks registered on a ProcessGroup fire for
    every collective issued through it. The hooks are wired into the c10d
    dispatcher kernels (Ops.cpp), so they fire wherever a c10d op is dispatched
    -- including replay of a captured graph that re-dispatches the raw op (see
    test_hooks_fire_on_captured_graph_replay). Uses a real gloo backend; a
    pure-Python or fake ProcessGroup overrides the collective methods and
    dispatches without going through the c10d ops, so it bypasses these hooks.
    """

    hw_classification = HardwareClassification.GENERIC

    world_size = 2

    @classmethod
    def backend_str(cls):
        return "gloo"

    def test_hooks_fire_for_all_collectives(self):
        pg = _get_default_group()
        # The pre-hook fires before the op is issued, the post-hook after, with
        # a matching op_id correlating the two.
        pre_ops: list[HookOpName] = []
        post_ops: list[HookOpName] = []
        pre_op_ids: list[int] = []
        post_op_ids: list[int] = []
        pg.register_pre_hook(
            0,
            lambda args: (pre_ops.append(args.name), pre_op_ids.append(args.op_id)),
        )
        pg.register_post_hook(
            0,
            lambda args: (post_ops.append(args.name), post_op_ids.append(args.op_id)),
        )

        rank = self.rank
        ws = self.world_size

        dist.broadcast(torch.ones(2), src=0)
        dist.all_reduce(torch.ones(2))
        dist.reduce(torch.ones(2), dst=0)
        dist.all_gather([torch.zeros(2) for _ in range(ws)], torch.ones(2))
        dist.all_gather_into_tensor(torch.zeros(2 * ws), torch.ones(2))
        dist.reduce_scatter(torch.zeros(2), [torch.ones(2) for _ in range(ws)])
        dist.reduce_scatter_tensor(torch.zeros(2), torch.ones(2 * ws))
        dist.all_to_all_single(torch.zeros(ws), torch.arange(ws, dtype=torch.float32))
        dist.all_to_all(
            [torch.zeros(1) for _ in range(ws)],
            [torch.ones(1) for _ in range(ws)],
        )
        dist.scatter(
            torch.zeros(2),
            [torch.ones(2) for _ in range(ws)] if rank == 0 else None,
            src=0,
        )
        dist.gather(
            torch.ones(2),
            [torch.zeros(2) for _ in range(ws)] if rank == 0 else None,
            dst=0,
        )
        dist.barrier()
        # send/recv need a peer; order send/recv by parity to avoid deadlock.
        peer = (rank + 1) % ws
        if rank % 2 == 0:
            dist.send(torch.ones(2), dst=peer)
            dist.recv(torch.zeros(2), src=peer)
        else:
            dist.recv(torch.zeros(2), src=peer)
            dist.send(torch.ones(2), dst=peer)

        expected = {
            HookOpName.BROADCAST,
            HookOpName.ALLREDUCE,
            HookOpName.REDUCE,
            HookOpName.ALLGATHER,
            HookOpName.REDUCE_SCATTER,
            HookOpName.ALLTOALL,
            HookOpName.SCATTER,
            HookOpName.GATHER,
            HookOpName.BARRIER,
            HookOpName.SEND,
            HookOpName.RECV,
        }
        for op in expected:
            self.assertIn(
                op, pre_ops, lambda msg: f"{msg}\npre-hook did not fire for {op}"
            )
            self.assertIn(
                op, post_ops, lambda msg: f"{msg}\npost-hook did not fire for {op}"
            )

        # Every issued collective fires exactly one pre and one post hook, and
        # each post correlates with its pre via op_id.
        self.assertEqual(len(pre_ops), len(post_ops))
        self.assertEqual(pre_op_ids, post_op_ids)

        # After unregistering, no further hooks fire.
        pg.unregister_pre_hook(0)
        pg.unregister_post_hook(0)
        pre_count, post_count = len(pre_ops), len(post_ops)
        dist.all_reduce(torch.ones(2))
        self.assertEqual(len(pre_ops), pre_count)
        self.assertEqual(len(post_ops), post_count)

        dist.barrier()

    def test_hooks_fire_on_captured_graph_replay(self):
        # make_fx traces at the dispatcher (op) level and records the raw
        # c10d.allreduce_ op. Replaying that graph re-dispatches the op directly,
        # bypassing ProcessGroup::allreduce -- so this only fires hooks because
        # they live in the dispatcher kernels (Ops.cpp), not on ProcessGroup.
        from torch.fx.experimental.proxy_tensor import make_fx

        pg = _get_default_group()
        pre_ops: list[HookOpName] = []
        pg.register_pre_hook(0, lambda args: pre_ops.append(args.name))

        def fn(x):
            dist.all_reduce(x)
            return x

        graph = make_fx(fn, tracing_mode="real")(torch.ones(2))
        self.assertTrue(
            any("allreduce_" in str(n.target) for n in graph.graph.nodes),
            "expected make_fx to record the raw c10d.allreduce_ op",
        )

        # Replaying the captured graph re-dispatches the raw op; the hook fires.
        pre_ops.clear()
        graph(torch.ones(2))
        self.assertIn(HookOpName.ALLREDUCE, pre_ops)
        pg.unregister_pre_hook(0)

        dist.barrier()

    def test_collective_annotation_metadata(self):
        pg = _get_default_group()
        seq = pg._get_sequence_number_for_group()
        ws = self.world_size
        peer = (self.rank + 1) % ws
        annotator, push, pop = object(), mock.Mock(), mock.Mock()
        with mock.patch(_PGS_TO_ANNOTATE, return_value=[pg]):
            register_collective_annotation_hooks(annotator, push, pop)
        try:
            dist.all_reduce(torch.ones(3))
            dist.all_gather_into_tensor(torch.zeros(2 * ws), torch.ones(2))
            if self.rank == 0:
                dist.send(torch.ones(2), dst=peer)
            else:
                dist.recv(torch.zeros(2), src=peer)
        finally:
            with mock.patch(_PGS_TO_ANNOTATE, return_value=[pg]):
                unregister_collective_annotation_hooks(annotator)

        self.assertEqual(pop.call_count, push.call_count)
        allreduce, allgather, p2p = (c.args[0] for c in push.call_args_list)
        self.assertEqual(
            allreduce,
            {
                "Collective name": "all_reduce",
                "In msg nelems": 3,
                "Out msg nelems": 3,
                "Group size": ws,
                "Process Group Name": pg.group_name,
                "Process Group Description": pg.group_desc,
                "Process Group Ranks": str(list(range(ws))),
                "Is asynchronized op": False,
                "Rank": self.rank,
                "dtype": "Float",
                "Seq": seq + 1,
            },
        )
        self.assertEqual(allgather["Collective name"], "all_gather")
        self.assertEqual(allgather["In msg nelems"], 2)
        self.assertEqual(allgather["Out msg nelems"], 2 * ws)
        self.assertEqual(allgather["Seq"], seq + 2)
        if self.rank == 0:
            self.assertEqual(p2p["Collective name"], "send")
            self.assertEqual(p2p["Dst Rank"], peer)
        else:
            self.assertEqual(p2p["Collective name"], "recv")
            self.assertEqual(p2p["Src Rank"], peer)
        self.assertEqual(p2p["Rank"], peer)
        self.assertNotIn("Seq", p2p)

        dist.barrier()

    def test_collective_annotation_hooks(self):
        group = dist.new_group()
        annotator, calls = object(), mock.Mock()
        with mock.patch(_PGS_TO_ANNOTATE, return_value=[group]):
            register_collective_annotation_hooks(annotator, calls.push, calls.pop)
            dist.all_reduce(torch.ones(1), group=group)
            unregister_collective_annotation_hooks(annotator)
            unregister_collective_annotation_hooks(annotator)  # idempotent
        dist.all_reduce(torch.ones(1), group=group)
        self.assertEqual([name for name, _, _ in calls.mock_calls], ["push", "pop"])
        metadata = calls.push.call_args.args[0]
        self.assertEqual(metadata["Process Group Name"], group.group_name)
        dist.destroy_process_group(group)

        dist.barrier()

    def test_completion_hooks_unsupported(self):
        pg = _get_default_group()
        self.assertFalse(pg.supports_completion_hooks)
        with self.assertRaisesRegex(RuntimeError, "does not support"):
            pg.register_completion_hook(0, lambda args: None)
        with self.assertRaisesRegex(TypeError, "callable"):
            pg.register_completion_hook(0, 1)


_EXIT_SCRIPT = """
import atexit
import sys
import weakref


def check():
    # Registered before torch's exit handler, so it runs after it.
    try:
        dist.group.WORLD.register_completion_hook(1, print)
    except RuntimeError as e:
        if "shutting down" in str(e) and ref() is None:
            print("exit ok", flush=True)


atexit.register(check)

import torch
import torch.distributed as dist


def hook(args):
    pass


torch.cuda.set_device(0)
dist.init_process_group(
    sys.argv[1], rank=0, world_size=1, store=dist.FileStore(sys.argv[2], 1)
)
ref = weakref.ref(hook)
dist.group.WORLD.register_completion_hook(0, hook)
del hook
for _ in range(100):
    dist.all_reduce(torch.ones(1024, device="cuda"))
"""


@instantiate_parametrized_tests
class TestProcessGroupCompletionHooks(MultiProcContinuousTest):
    """Completion hooks on each backend that supports them."""

    hw_classification = HardwareClassification.CUDA

    world_size = 2

    @classmethod
    def backend_str(cls):
        return "gloo"

    def _skip_unless(self, backend):
        if not dist.is_backend_available(backend):
            self.skipTest(f"{backend} is not available")
        if torch.cuda.device_count() < self.world_size:
            self.skipTest(f"requires {self.world_size} GPUs")

    @parametrize("backend", ["nccl2", "nccl-lazy"])
    def test_completion_hooks(self, backend):
        self._skip_unless(backend)
        device = torch.device("cuda", self.rank)
        torch.cuda.set_device(device)
        env = mock.patch.dict(os.environ, {"TORCH_NCCL_ENABLE_TIMING": "0"})
        env.start()
        self.addCleanup(env.stop)
        pg = dist.new_group(backend=backend)
        self.addCleanup(dist.destroy_process_group, pg)
        self.assertTrue(pg.supports_completion_hooks)
        # nccl-lazy has no _enable_collectives_timing.
        timed = backend == "nccl2"
        if timed:
            pg._enable_collectives_timing()

        ops = {}
        pg.register_post_hook(
            0, lambda args: ops.__setitem__(args.work.completion_key, args.name)
        )
        self.addCleanup(pg.unregister_post_hook, 0)

        def run_ops():
            ops.clear()
            t = torch.ones(8, device=device)
            dist.all_reduce(t, group=pg)
            if self.rank == 0:
                dist.send(t, dst=1, group=pg)
            else:
                dist.recv(t, src=0, group=pg)
            return dict(ops)

        # Registered before the p2p comm exists.
        first = _CompletionCollector()
        pg.register_completion_hook(0, first)
        self.addCleanup(pg.unregister_completion_hook, 0)
        keys = run_ops()
        p2p = "SEND" if self.rank == 0 else "RECV"
        self.assertEqual(sorted(op.name for op in keys.values()), ["ALLREDUCE", p2p])
        for duration in first.wait_for(keys):
            if not timed:
                self.assertIsNone(duration)
                continue
            self.assertIsInstance(duration, float)
            self.assertGreaterEqual(duration, 0.0)

        # Registered after the p2p comm exists.
        second = _CompletionCollector()
        pg.register_completion_hook(1, second)
        self.addCleanup(pg.unregister_completion_hook, 1)
        with self.assertRaisesRegex(ValueError, "already registered"):
            pg.register_completion_hook(1, second)
        keys = run_ops()
        first.wait_for(keys)
        second.wait_for(keys)

        released = weakref.ref(second)
        del second
        pg.unregister_completion_hook(1)
        self.assertIsNone(released())
        # Unknown ids are ignored.
        pg.unregister_completion_hook(1)
        # The id can be reused.
        again = _CompletionCollector()
        pg.register_completion_hook(1, again)
        again.wait_for(run_ops())

        dist.barrier()

    @parametrize("backend", ["nccl2", "nccl-lazy"])
    def test_exit_with_registered_hook(self, backend):
        self._skip_unless(backend)
        if self.rank == 0:
            with tempfile.TemporaryDirectory() as tmp:
                proc = subprocess.run(
                    [sys.executable, "-c", _EXIT_SCRIPT, backend, f"{tmp}/store"],
                    capture_output=True,
                    text=True,
                    timeout=TIMEOUT,
                )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertIn("exit ok", proc.stdout, msg=proc.stderr)
        dist.barrier()


if __name__ == "__main__":
    run_tests()
