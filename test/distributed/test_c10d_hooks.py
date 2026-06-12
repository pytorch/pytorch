# Owner(s): ["oncall: distributed"]

import os

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import HookOpName
from torch.distributed.distributed_c10d import _get_default_group
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import run_tests


class TestProcessGroupHooks(MultiProcessTestCase):
    """
    Verify that pre/post collective hooks registered on a ProcessGroup fire for
    every collective issued through it. The hooks are wired into the c10d
    dispatcher kernels (Ops.cpp), so they fire wherever a c10d op is dispatched
    -- including replay of a captured graph that re-dispatches the raw op (see
    test_hooks_fire_on_captured_graph_replay). Uses a real gloo backend; a
    pure-Python or fake ProcessGroup overrides the collective methods and
    dispatches without going through the c10d ops, so it bypasses these hooks.
    """

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @property
    def world_size(self):
        return 2

    def test_hooks_fire_for_all_collectives(self):
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="gloo",
            store=store,
            rank=self.rank,
            world_size=self.world_size,
        )

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
            self.assertIn(op, pre_ops, f"pre-hook did not fire for {op}")
            self.assertIn(op, post_ops, f"post-hook did not fire for {op}")

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
        dist.destroy_process_group()

    def test_hooks_fire_on_captured_graph_replay(self):
        # make_fx traces at the dispatcher (op) level and records the raw
        # c10d.allreduce_ op. Replaying that graph re-dispatches the op directly,
        # bypassing ProcessGroup::allreduce -- so this only fires hooks because
        # they live in the dispatcher kernels (Ops.cpp), not on ProcessGroup.
        from torch.fx.experimental.proxy_tensor import make_fx

        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="gloo",
            store=store,
            rank=self.rank,
            world_size=self.world_size,
        )
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

        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()
