# Owner(s): ["oncall: distributed"]

import gc
import tempfile
import weakref

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch._C._distributed_c10d import (
    AllgatherOptions,
    AllreduceOptions,
    AllToAllOptions,
    BroadcastOptions,
    ReduceScatterOptions,
    ScatterOptions,
)
from torch.distributed._virtual_pg import LocalProcessGroup, VirtualProcessGroup
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import run_tests, skipIfRocm, TestCase


if not dist.is_available():
    print("distributed package not available, skipping tests")
    import sys

    sys.exit(0)


class TestLocalProcessGroup(TestCase):
    def _make_pg(self, rank=1, world=4):
        pg = LocalProcessGroup(rank, world)
        self.addCleanup(pg.unregister)
        return pg

    def test_construction(self):
        pg = self._make_pg(rank=3, world=8)
        self.assertEqual(pg.rank(), 3)
        self.assertEqual(pg.size(), 8)
        self.assertFalse(dist.is_initialized())

    def test_invalid_rank(self):
        with self.assertRaisesRegex(ValueError, "invalid rank"):
            LocalProcessGroup(4, 4)

    def test_allreduce_identity(self):
        pg = self._make_pg()
        t = torch.arange(4.0)
        work = pg.allreduce([t], AllreduceOptions())
        self.assertTrue(work.wait())
        self.assertEqual(t, torch.arange(4.0))

    def test_broadcast_identity(self):
        pg = self._make_pg()
        t = torch.arange(4.0)
        opts = BroadcastOptions()
        opts.rootRank = 0
        pg.broadcast([t], opts).wait()
        self.assertEqual(t, torch.arange(4.0))

    def test_all_gather_single_replicates(self):
        pg = self._make_pg()
        inp = torch.arange(2.0)
        out = torch.zeros(8)
        pg.all_gather_single(out, inp, AllgatherOptions()).wait()
        self.assertEqual(out, inp.repeat(4))

    def test_allgather_replicates(self):
        pg = self._make_pg()
        inp = torch.arange(3.0)
        outs = [[torch.zeros(3) for _ in range(4)]]
        pg.allgather(outs, [inp], AllgatherOptions()).wait()
        for o in outs[0]:
            self.assertEqual(o, inp)

    def test_reduce_scatter_single_takes_own_chunk(self):
        pg = self._make_pg(rank=1, world=4)
        inp = torch.arange(8.0)
        out = torch.zeros(2)
        pg.reduce_scatter_single(out, inp, ReduceScatterOptions()).wait()
        self.assertEqual(out, torch.tensor([2.0, 3.0]))

    def test_scatter_root_takes_own_chunk(self):
        pg = self._make_pg(rank=1, world=4)
        ins = [[torch.full((2,), float(r)) for r in range(4)]]
        out = torch.zeros(2)
        opts = ScatterOptions()
        opts.rootRank = 1
        pg.scatter([out], ins, opts).wait()
        self.assertEqual(out, torch.full((2,), 1.0))

    def test_alltoall_identity(self):
        pg = self._make_pg()
        ins = [torch.full((2,), float(i)) for i in range(4)]
        outs = [torch.zeros(2) for _ in range(4)]
        pg.alltoall(outs, ins, AllToAllOptions()).wait()
        for o, i in zip(outs, ins):
            self.assertEqual(o, i)

    def test_all_to_all_single_even(self):
        pg = self._make_pg()
        inp = torch.arange(8.0)
        out = torch.zeros(8)
        pg.all_to_all_single(out, inp, [], [], AllToAllOptions()).wait()
        self.assertEqual(out, inp)

    def test_requires_grad_input(self):
        pg = self._make_pg()
        inp = torch.ones(2, requires_grad=True)
        out = torch.zeros(8)
        pg.all_gather_single(out, inp, AllgatherOptions()).wait()
        self.assertEqual(out, torch.ones(8))

    def test_dist_all_reduce_group_kwarg(self):
        pg = self._make_pg()
        t = torch.ones(4)
        dist.all_reduce(t, group=pg)
        self.assertEqual(t, torch.ones(4))

    def test_dist_all_reduce_async(self):
        pg = self._make_pg()
        t = torch.ones(4)
        work = dist.all_reduce(t, group=pg, async_op=True)
        self.assertTrue(work.wait())

    def test_hook_sees_normalized_collective(self):
        calls = []

        class RecordingPG(LocalProcessGroup):
            def run_collective(self, coll):
                calls.append(
                    (coll.op, [t.shape for t in coll.inputs], coll.reduce_op, coll.root)
                )
                return None

        pg = RecordingPG(0, 2)
        self.addCleanup(pg.unregister)
        t = torch.ones(3)
        opts = AllreduceOptions()
        opts.reduceOp = dist.ReduceOp.MAX
        pg.allreduce([t], opts).wait()
        out = torch.zeros(6)
        pg.all_gather_single(out, t, AllgatherOptions()).wait()
        self.assertEqual(calls[0][0], "allreduce")
        self.assertEqual(calls[0][1], [torch.Size([3])])
        self.assertEqual(calls[0][2], dist.ReduceOp.MAX)
        self.assertEqual(calls[1][0], "all_gather_single")

    def test_funcol_by_name(self):
        pg = self._make_pg()
        y = funcol.wait_tensor(funcol.all_reduce(torch.ones(2), "sum", pg.group_name))
        self.assertEqual(y, torch.ones(2))

    def test_funcol_by_object(self):
        pg = self._make_pg()
        y = funcol.wait_tensor(funcol.all_gather_single(torch.ones(2), 0, pg))
        self.assertEqual(y, torch.ones(8))

    def test_funcol_reduce_scatter(self):
        pg = self._make_pg(rank=1, world=4)
        y = funcol.wait_tensor(
            funcol.reduce_scatter_tensor(torch.arange(8.0), "sum", 0, pg.group_name)
        )
        self.assertEqual(y, torch.tensor([2.0, 3.0]))

    def test_work_lifetime(self):
        wrefs = []

        class TrackedWorkPG(LocalProcessGroup):
            def run_collective(self, coll):
                work = super().run_collective(coll)
                return work

        pg = TrackedWorkPG(0, 2)
        self.addCleanup(pg.unregister)
        t = torch.ones(2)
        work = pg.allreduce([t], AllreduceOptions())
        wrefs.append(weakref.ref(work))
        work.wait()
        del work
        gc.collect()
        self.assertIsNone(wrefs[0]())

    def test_pg_not_leaked_after_unregister(self):
        pg = LocalProcessGroup(0, 2)
        name = pg.group_name
        pg.unregister()
        wref = weakref.ref(pg)
        del pg
        gc.collect()
        self.assertIsNone(wref())
        with self.assertRaises(Exception):
            torch._C._distributed_c10d._resolve_process_group(name)

    def test_split_local(self):
        pg = self._make_pg(rank=1, world=4)
        sub = pg.split_local([1, 3])
        self.addCleanup(sub.unregister)
        self.assertEqual(sub.rank(), 0)
        self.assertEqual(sub.size(), 2)
        self.assertIsNone(pg.split_local([0, 2]))

    def test_split_local_validation(self):
        pg = self._make_pg()
        with self.assertRaisesRegex(ValueError, "out of range"):
            pg.split_local([1, 7])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            pg.split_local([1, 1])

    def test_split_group_virtual_method(self):
        pg = self._make_pg(rank=1, world=4)
        sub = pg.splitGroup([0, 1], None, None, "sub_name", "sub_desc", None)
        self.addCleanup(sub.unregister)
        self.assertEqual(sub.rank(), 1)
        self.assertEqual(sub.size(), 2)
        self.assertEqual(sub.group_name, "sub_name")
        self.assertEqual(sub.group_desc, "sub_desc")

    def test_nested_split(self):
        pg = self._make_pg(rank=2, world=8)
        sub = pg.split_local([0, 2, 4, 6])
        self.addCleanup(sub.unregister)
        self.assertEqual(sub.rank(), 1)
        subsub = sub.split_local([1, 3])
        self.addCleanup(subsub.unregister)
        self.assertEqual(subsub.rank(), 0)
        self.assertEqual(subsub.size(), 2)

    def test_funcol_on_split_group(self):
        pg = self._make_pg(rank=1, world=4)
        sub = pg.split_local([1, 3])
        self.addCleanup(sub.unregister)
        y = funcol.wait_tensor(
            funcol.all_gather_single(torch.ones(2), 0, sub.group_name)
        )
        self.assertEqual(y, torch.ones(4))

    def test_compile_funcol(self):
        pg = self._make_pg()

        def f(x):
            y = funcol.all_reduce(x * 3, "sum", pg.group_name)
            return funcol.wait_tensor(y) + 1

        x = torch.ones(4)
        eager = f(x)
        compiled = torch.compile(f, fullgraph=True)(x)
        self.assertEqual(eager, compiled)

    def test_compile_funcol_group_object(self):
        pg = self._make_pg()

        def f(x):
            return funcol.wait_tensor(funcol.all_gather_single(x, 0, pg)).sum()

        x = torch.ones(2)
        eager = f(x)
        compiled = torch.compile(f, fullgraph=True)(x)
        self.assertEqual(eager, compiled)

    def test_compile_bakes_logical_rank(self):
        # dist.get_rank(pg) translates through the default group's global
        # rank, so it cannot work for a standalone PG; pg.rank() is the
        # logical-rank query that dynamo constant-folds.
        pg = self._make_pg(rank=2, world=4)

        def f(x):
            return x + pg.rank()

        compiled = torch.compile(f, fullgraph=True)(torch.zeros(2))
        self.assertEqual(compiled, torch.full((2,), 2.0))


class TestVirtualProcessGroup(TestCase):
    def _make_phys(self):
        f = tempfile.NamedTemporaryFile(delete=False)  # noqa: SIM115
        store = dist.FileStore(f.name, 1)
        return dist.ProcessGroupGloo(store, 0, 1)

    def _make_pg(self, rank=1, world=4, **kwargs):
        pg = VirtualProcessGroup(rank, world, self._make_phys(), **kwargs)
        self.addCleanup(pg.unregister)
        return pg

    def test_logical_vs_physical_rank(self):
        pg = self._make_pg(rank=2, world=4)
        self.assertEqual(pg.rank(), 2)
        self.assertEqual(pg.size(), 4)
        self.assertEqual(pg.physical_group.size(), 1)

    def test_wait_reaches_physical_work(self):
        wait_counts = []

        class PhysWork(dist._Work):
            def __init__(self):
                super().__init__()
                self.waited = 0
                wait_counts.append(self)

            def wait(self, timeout=None):
                self.waited += 1
                return True

        class CountingPhys(dist.ProcessGroup):
            def __init__(self):
                super().__init__(0, 1)

            def getBackendName(self):
                return "counting"

            def allreduce(self, tensors, opts):
                return PhysWork()

        pg = VirtualProcessGroup(1, 4, CountingPhys())
        self.addCleanup(pg.unregister)
        t = torch.full((3,), 5.0)
        work = pg.allreduce([t], AllreduceOptions())
        self.assertEqual(len(wait_counts), 1)
        self.assertEqual(wait_counts[0].waited, 0)
        work.wait()
        self.assertEqual(wait_counts[0].waited, 1)
        self.assertEqual(t, torch.full((3,), 5.0))

    def test_allgather_data_and_mirror(self):
        pg = self._make_pg()
        inp = torch.arange(3.0)
        out = torch.zeros(12)
        pg.all_gather_single(out, inp, AllgatherOptions()).wait()
        self.assertEqual(out, inp.repeat(4))

    def test_reduce_scatter_mirror(self):
        pg = self._make_pg(rank=1, world=4)
        inp = torch.arange(8.0)
        out = torch.zeros(2)
        pg.reduce_scatter_single(out, inp, ReduceScatterOptions()).wait()
        self.assertEqual(out, torch.tensor([2.0, 3.0]))

    def test_mirror_records_physical_collectives(self):
        issued = []

        class RecordingPhys(dist.ProcessGroup):
            def __init__(self):
                super().__init__(0, 1)

            def getBackendName(self):
                return "recording"

            def allreduce(self, tensors, opts):
                issued.append(("allreduce", tensors[0].numel()))
                fut = torch.futures.Future()
                fut.set_result(tensors)
                return torch._C._distributed_c10d._create_work_from_future(fut)

            def all_gather_single(self, out, inp, opts):
                issued.append(("all_gather_single", inp.numel()))
                fut = torch.futures.Future()
                fut.set_result(out)
                return torch._C._distributed_c10d._create_work_from_future(fut)

        phys = RecordingPhys()
        pg = VirtualProcessGroup(1, 4, phys)
        self.addCleanup(pg.unregister)
        t = torch.ones(5)
        pg.allreduce([t], AllreduceOptions()).wait()
        out = torch.zeros(8)
        pg.all_gather_single(out, torch.ones(2), AllgatherOptions()).wait()
        self.assertEqual(issued, [("allreduce", 5), ("all_gather_single", 2)])

    def test_scratch_buffers_are_stable(self):
        pg = self._make_pg()
        t = torch.ones(4)
        pg.allreduce([t], AllreduceOptions()).wait()
        key = next(iter(pg._scratch))
        buf_ptr = pg._scratch[key][0].data_ptr()
        pg.allreduce([t], AllreduceOptions()).wait()
        self.assertEqual(len(pg._scratch), 1)
        self.assertEqual(pg._scratch[key][0].data_ptr(), buf_ptr)

    def test_funcol_by_name(self):
        pg = self._make_pg()
        y = funcol.wait_tensor(funcol.all_reduce(torch.ones(2), "sum", pg.group_name))
        self.assertEqual(y, torch.ones(2))

    def test_split_reuses_physical_group(self):
        pg = self._make_pg(rank=1, world=4)
        sub = pg.split_local([1, 3])
        self.addCleanup(sub.unregister)
        self.assertEqual(sub.rank(), 0)
        self.assertEqual(sub.size(), 2)
        self.assertIs(sub.physical_group, pg.physical_group)
        y = funcol.wait_tensor(
            funcol.all_gather_single(torch.ones(2), 0, sub.group_name)
        )
        self.assertEqual(y, torch.ones(4))

    def test_compile_funcol(self):
        pg = self._make_pg()

        def f(x):
            return (
                funcol.wait_tensor(funcol.all_reduce(x * 2, "sum", pg.group_name)) + 1
            )

        x = torch.ones(4)
        eager = f(x)
        compiled = torch.compile(f, fullgraph=True)(x)
        self.assertEqual(eager, compiled)

    def test_hook_normalization_unchanged(self):
        seen = []
        orig = VirtualProcessGroup.run_collective

        class SpyPG(VirtualProcessGroup):
            def run_collective(self, coll):
                seen.append(coll.op)
                return orig(self, coll)

        pg = SpyPG(0, 4, self._make_phys())
        self.addCleanup(pg.unregister)
        t = torch.ones(2)
        pg.allreduce([t], AllreduceOptions()).wait()
        pg.barrier(torch._C._distributed_c10d.BarrierOptions()).wait()
        self.assertEqual(seen, ["allreduce", "barrier"])


class TestProjectedMode(TestCase):
    """output_mode="projected": physical collective on views of app tensors."""

    def _make_pg(self, rank=1, world=4, **kwargs):
        f = tempfile.NamedTemporaryFile(delete=False)  # noqa: SIM115
        store = dist.FileStore(f.name, 1)
        phys = dist.ProcessGroupGloo(store, 0, 1)
        kwargs.setdefault("output_mode", "projected")
        pg = VirtualProcessGroup(rank, world, phys, **kwargs)
        self.addCleanup(pg.unregister)
        return pg

    def test_invalid_output_mode(self):
        with self.assertRaisesRegex(ValueError, "output_mode"):
            self._make_pg(output_mode="bogus")

    def test_allreduce_in_place_on_logical_tensor(self):
        seen_ptrs = []

        class SpyPhys(dist.ProcessGroup):
            def __init__(self):
                super().__init__(0, 1)

            def getBackendName(self):
                return "spy"

            def allreduce(self, tensors, opts):
                seen_ptrs.append(tensors[0].data_ptr())
                fut = torch.futures.Future()
                fut.set_result(tensors)
                return torch._C._distributed_c10d._create_work_from_future(fut)

        pg = VirtualProcessGroup(1, 4, SpyPhys(), output_mode="projected")
        self.addCleanup(pg.unregister)
        t = torch.ones(8)
        pg.allreduce([t], AllreduceOptions()).wait()
        self.assertEqual(seen_ptrs, [t.data_ptr()])

    def test_all_gather_projects_onto_logical_buffers(self):
        seen = {}

        class SpyPhys(dist.ProcessGroup):
            def __init__(self):
                super().__init__(0, 1)

            def getBackendName(self):
                return "spy"

            def all_gather_single(self, out, inp, opts):
                seen["out_ptr"] = out.data_ptr()
                seen["in_ptr"] = inp.data_ptr()
                seen["out_numel"] = out.numel()
                out.copy_(inp)  # physical world = 1
                fut = torch.futures.Future()
                fut.set_result(out)
                return torch._C._distributed_c10d._create_work_from_future(fut)

        pg = VirtualProcessGroup(1, 4, SpyPhys(), output_mode="projected")
        self.addCleanup(pg.unregister)
        inp = torch.arange(2.0)
        out = torch.zeros(8)
        pg.all_gather_single(out, inp, AllgatherOptions()).wait()
        self.assertEqual(seen["in_ptr"], inp.data_ptr())
        self.assertEqual(seen["out_ptr"], out.data_ptr())
        # physical world 1 => reachable prefix is 1 * inp.numel()
        self.assertEqual(seen["out_numel"], 2)
        self.assertEqual(out[:2], inp)
        # no local fake fill: the rest of the logical output was not written
        self.assertEqual(out[2:], torch.zeros(6))
        self.assertEqual(len(pg._scratch), 0)

    def test_reduce_scatter_projects_onto_logical_buffers(self):
        pg = self._make_pg()
        inp = torch.arange(8.0)
        out = torch.zeros(2)
        pg.reduce_scatter_single(out, inp, ReduceScatterOptions()).wait()
        # physical world 1: reduce_scatter over first 1*2 elements of input
        self.assertEqual(out, torch.tensor([0.0, 1.0]))
        self.assertEqual(len(pg._scratch), 0)

    def test_projected_requires_contiguous(self):
        from torch.distributed._virtual_pg import ProjectionError

        pg = self._make_pg()
        inp = torch.zeros(4, 4).t()  # non-contiguous
        out = torch.zeros(4, 4)
        with self.assertRaises(ProjectionError):
            pg.all_gather_single(out, inp, AllgatherOptions())

    def test_projected_rejects_undersized_output(self):
        from torch.distributed._virtual_pg import ProjectionError

        class TwoRankPhys(dist.ProcessGroup):
            def __init__(self):
                super().__init__(0, 2)

            def getBackendName(self):
                return "two"

        pg = VirtualProcessGroup(0, 4, TwoRankPhys(), output_mode="projected")
        self.addCleanup(pg.unregister)
        # logical output sized for one physical rank only; needs 2*3=6
        with self.assertRaises(ProjectionError):
            pg.all_gather_single(torch.zeros(3), torch.zeros(3), AllgatherOptions())

    def test_projected_never_falls_back_to_scratch(self):
        from torch.distributed._virtual_pg import ProjectionError

        pg = self._make_pg()
        outs = [[torch.zeros(2) for _ in range(4)]]
        with self.assertRaisesRegex(ProjectionError, "scratch"):
            pg.allgather(outs, [torch.zeros(2)], AllgatherOptions())
        self.assertEqual(len(pg._scratch), 0)

    def test_scratch_mode_untouched_logical_output(self):
        pg = self._make_pg(output_mode="scratch")
        inp = torch.arange(2.0)
        out = torch.full((8,), -1.0)
        pg.all_gather_single(out, inp, AllgatherOptions()).wait()
        # scratch mode: logical output not written at all
        self.assertEqual(out, torch.full((8,), -1.0))
        self.assertGreater(len(pg._scratch), 0)

    def test_local_fake_mode_fills_output(self):
        pg = self._make_pg(output_mode="local_fake")
        inp = torch.arange(2.0)
        out = torch.zeros(8)
        pg.all_gather_single(out, inp, AllgatherOptions()).wait()
        self.assertEqual(out, inp.repeat(4))

    def test_split_inherits_output_mode(self):
        pg = self._make_pg(rank=1, world=4)
        sub = pg.split_local([1, 3])
        self.addCleanup(sub.unregister)
        self.assertEqual(sub.output_mode, "projected")

    def test_p2p_requires_peer_map_in_projected(self):
        from torch.distributed._virtual_pg import ProjectionError

        pg = self._make_pg()
        with self.assertRaisesRegex(ProjectionError, "physical_peer_map"):
            pg.send([torch.ones(2)], 3, 0)

    def test_p2p_mirrors_with_peer_map(self):
        sent = []

        class SpyPhys(dist.ProcessGroup):
            def __init__(self):
                super().__init__(0, 1)

            def getBackendName(self):
                return "spy"

            def send(self, tensors, dst, tag):
                sent.append((tensors[0].data_ptr(), dst, tag))
                fut = torch.futures.Future()
                fut.set_result(tensors)
                return torch._C._distributed_c10d._create_work_from_future(fut)

        pg = VirtualProcessGroup(
            0, 4, SpyPhys(), output_mode="projected", physical_peer_map={3: 0}
        )
        self.addCleanup(pg.unregister)
        t = torch.ones(2)
        pg.send([t], 3, 7).wait()
        self.assertEqual(sent, [(t.data_ptr(), 0, 7)])

    def test_profiler_metadata_has_logical_name(self):
        pg = self._make_pg()
        inp, out = torch.ones(2), torch.zeros(8)
        with torch.profiler.profile() as prof:
            pg.all_gather_single(out, inp, AllgatherOptions()).wait()
        names = [e.name for e in prof.events()]
        self.assertTrue(
            any(f"virtual_pg::{pg.group_name}::all_gather_single" in n for n in names),
            names,
        )


class DeferredWork(dist._Work):
    """Work that records when wait() happens; completes only on wait."""

    log: list = []

    def __init__(self, label):
        super().__init__()
        self.label = label
        DeferredWork.log.append(("issue", label))

    def wait(self, timeout=None):
        DeferredWork.log.append(("wait", self.label))
        return True


class DeferredPhys(dist.ProcessGroup):
    """Physical PG returning DeferredWork so wait placement is observable."""

    def __init__(self):
        super().__init__(0, 1)
        self._name = ""

    def getBackendName(self):
        return "deferred"

    def setGroupName(self, name):
        self._name = name

    def getGroupName(self):
        return self._name

    def allreduce(self, tensors, opts):
        return DeferredWork("allreduce")

    def all_gather_single(self, out, inp, opts):
        out[: inp.numel()].copy_(inp)
        return DeferredWork("all_gather")

    def reduce_scatter_single(self, out, inp, opts):
        out.copy_(inp[: out.numel()])
        return DeferredWork("reduce_scatter")


class TestWorkSemantics(TestCase):
    """The physical Work is joined at the consumer, not at issue time."""

    def setUp(self):
        super().setUp()
        DeferredWork.log = []

    def _make_pg(self, rank=1, world=4):
        pg = VirtualProcessGroup(rank, world, DeferredPhys(), output_mode="projected")
        self.addCleanup(pg.unregister)
        return pg

    def test_no_wait_at_issue_eager_async(self):
        pg = self._make_pg()
        t = torch.ones(4)
        work = dist.all_reduce(t, group=pg, async_op=True)
        self.assertEqual(DeferredWork.log, [("issue", "allreduce")])
        work.wait()
        self.assertEqual(
            DeferredWork.log, [("issue", "allreduce"), ("wait", "allreduce")]
        )

    def test_funcol_defers_wait_to_wait_tensor(self):
        pg = self._make_pg()
        y = funcol.all_reduce(torch.ones(2), "sum", pg.group_name)
        self.assertEqual(DeferredWork.log, [("issue", "allreduce")])
        # consuming via wait_tensor joins the physical work
        funcol.wait_tensor(y)
        self.assertEqual(
            DeferredWork.log, [("issue", "allreduce"), ("wait", "allreduce")]
        )

    def test_funcol_registers_work_against_logical_output(self):
        pg = self._make_pg()
        before = torch._C._distributed_c10d._get_work_registry_size()
        y = funcol.all_reduce(torch.ones(2), "sum", pg.group_name)
        self.assertEqual(
            torch._C._distributed_c10d._get_work_registry_size(), before + 1
        )
        funcol.wait_tensor(y)
        self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), before)

    def test_async_collective_tensor_waits_on_use(self):
        pg = self._make_pg()
        act = funcol.all_reduce(torch.ones(2), "sum", pg.group_name)
        self.assertEqual([e[0] for e in DeferredWork.log], ["issue"])
        # first real use of the AsyncCollectiveTensor triggers the wait
        _ = act + 1
        self.assertIn(("wait", "allreduce"), DeferredWork.log)

    def test_work_gc_after_wait(self):
        pg = self._make_pg()
        t = torch.ones(4)
        work = pg.allreduce([t], AllreduceOptions())
        wref = weakref.ref(work)
        work.wait()
        del work
        gc.collect()
        self.assertIsNone(wref())

    def test_compiled_funcol_defers_wait(self):
        pg = self._make_pg()

        def f(x):
            y = funcol.all_reduce(x, "sum", pg.group_name)
            return funcol.wait_tensor(y) + 1

        compiled = torch.compile(f, fullgraph=True)
        DeferredWork.log = []
        compiled(torch.ones(2))
        self.assertEqual(
            DeferredWork.log, [("issue", "allreduce"), ("wait", "allreduce")]
        )


class TestInstallVirtualWorld(TestCase):
    """install_virtual_world: unmodified apps see the logical world."""

    def _install(self, rank=2, world=8, **kwargs):
        from torch.distributed._virtual_pg import (
            install_virtual_world,
            uninstall_virtual_world,
        )

        f = tempfile.NamedTemporaryFile(delete=False)  # noqa: SIM115
        phys = dist.ProcessGroupGloo(dist.FileStore(f.name, 1), 0, 1)
        pg = VirtualProcessGroup(rank, world, phys, **kwargs)
        install_virtual_world(pg)
        self.addCleanup(uninstall_virtual_world)
        self.addCleanup(pg.unregister)
        return pg

    def test_rank_and_world_size(self):
        self._install(rank=2, world=8)
        self.assertEqual(dist.get_rank(), 2)
        self.assertEqual(dist.get_world_size(), 8)

    def test_requires_uninitialized(self):
        from torch.distributed._virtual_pg import install_virtual_world

        pg = self._install()
        with self.assertRaisesRegex(RuntimeError, "uninitialized"):
            install_virtual_world(pg)

    def test_default_group_collective(self):
        self._install()
        t = torch.ones(4)
        dist.all_reduce(t)
        self.assertEqual(t, torch.ones(4))

    def test_new_group_creates_virtual_child(self):
        self._install(rank=2, world=8)
        sub = dist.new_group([0, 2, 4])
        self.assertIsInstance(sub, VirtualProcessGroup)
        self.assertEqual(sub.rank(), 1)
        self.assertEqual(sub.size(), 3)
        self.assertEqual(dist.new_group([1, 3]), dist.GroupMember.NON_GROUP_MEMBER)

    def test_new_group_rank_translation(self):
        self._install(rank=2, world=8)
        sub = dist.new_group([0, 2, 4])
        self.assertEqual(dist.get_group_rank(sub, 2), 1)
        self.assertEqual(dist.get_global_rank(sub, 1), 2)
        self.assertEqual(dist.get_rank(sub), 1)

    def test_device_mesh_dim_groups_are_virtual(self):
        from torch.distributed.device_mesh import init_device_mesh

        self._install(rank=2, world=8)
        mesh = init_device_mesh("cpu", (2, 4), mesh_dim_names=("dp", "tp"))
        dp, tp = mesh.get_group("dp"), mesh.get_group("tp")
        self.assertIsInstance(dp, VirtualProcessGroup)
        self.assertIsInstance(tp, VirtualProcessGroup)
        self.assertEqual(dp.size(), 2)
        self.assertEqual(tp.size(), 4)
        y = funcol.wait_tensor(funcol.all_reduce(torch.ones(2), "sum", tp))
        self.assertEqual(y, torch.ones(2))

    def test_child_groups_share_physical_by_default(self):
        pg = self._install(rank=2, world=8)
        sub1 = dist.new_group([0, 2])
        sub2 = dist.new_group([2, 4])
        self.assertIs(sub1.physical_group, pg.physical_group)
        self.assertIs(sub2.physical_group, pg.physical_group)

    def test_new_group_called_by_all_for_physical_split(self):
        # In mirror_split="split" mode the physical child is created before
        # the logical membership check, so non-member processes still
        # participate in the (deterministic-order) physical split.
        split_calls = []

        class SplitSpyPhys(dist.ProcessGroup):
            def __init__(self):
                super().__init__(0, 1)
                self._name = ""

            def getBackendName(self):
                return "splitspy"

            def setGroupName(self, name):
                self._name = name

            def getGroupName(self):
                return self._name

            def splitGroup(self, ranks, timeout, opts, name, desc, devices):
                split_calls.append(list(ranks))
                child = SplitSpyPhys()
                return child

        from torch.distributed._virtual_pg import (
            install_virtual_world,
            uninstall_virtual_world,
        )

        pg = VirtualProcessGroup(2, 8, SplitSpyPhys(), mirror_split="split")
        install_virtual_world(pg)
        self.addCleanup(uninstall_virtual_world)
        self.addCleanup(pg.unregister)
        member = dist.new_group([0, 2])
        nonmember = dist.new_group([1, 3])
        self.assertIsInstance(member, VirtualProcessGroup)
        self.assertEqual(nonmember, dist.GroupMember.NON_GROUP_MEMBER)
        # both calls split the physical parent over ALL physical ranks
        self.assertEqual(split_calls, [[0], [0]])
        self.assertIsNot(member.physical_group, pg.physical_group)


class TestVirtualProcessGroupNccl(TestCase):
    def _make_pg(self, rank=1, world=4):
        if not TEST_CUDA or not dist.is_nccl_available():
            self.skipTest("CUDA and NCCL required")
        f = tempfile.NamedTemporaryFile(delete=False)  # noqa: SIM115
        store = dist.FileStore(f.name, 1)
        phys = dist.ProcessGroupNCCL(store, 0, 1)
        pg = VirtualProcessGroup(rank, world, phys)
        self.addCleanup(pg.unregister)
        return pg

    @skipIfRocm
    def test_nccl_mirror_eager(self):
        pg = self._make_pg()
        t = torch.full((4,), 3.0, device="cuda")
        work = pg.allreduce([t], AllreduceOptions())
        work.wait()
        torch.cuda.synchronize()
        self.assertEqual(t, torch.full((4,), 3.0, device="cuda"))
        out = torch.zeros(8, device="cuda")
        pg.all_gather_single(
            out, torch.ones(2, device="cuda"), AllgatherOptions()
        ).wait()
        torch.cuda.synchronize()
        self.assertEqual(out, torch.ones(8, device="cuda"))

    @skipIfRocm
    def test_nccl_mirror_cuda_graph(self):
        pg = self._make_pg()
        t = torch.ones(4, device="cuda")
        # warmup on a side stream (allocates scratch, inits communicator)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                pg.allreduce([t], AllreduceOptions()).wait()
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            work = pg.allreduce([t], AllreduceOptions())
            work.wait()
            y = t * 2
        t.fill_(3.0)
        g.replay()
        torch.cuda.synchronize()
        self.assertEqual(y, torch.full((4,), 6.0, device="cuda"))


def _has_cuda_bindings():
    try:
        from cuda.bindings import runtime  # noqa: F401

        return True
    except ImportError:
        return False


def _collective_nodes(nodes):
    """Graph nodes issued by NCCL: device kernels (multi-rank) or memcpys
    (single-rank NCCL lowers self-collectives to copies)."""
    out = []
    for n in nodes:
        name = n.get("kernel_name") or ""
        if "nccl" in name.lower() or n["node_type"] == "memcpy":
            out.append(n)
    return out


def _reaches(nodes, src_idx, dst_idx):
    """Whether dst depends (transitively) on src in the captured graph."""
    deps = {n["index"]: set(n["dependencies"]) for n in nodes}
    seen, stack = set(), [dst_idx]
    while stack:
        i = stack.pop()
        if i == src_idx:
            return True
        if i in seen:
            continue
        seen.add(i)
        stack.extend(deps.get(i, ()))
    return False


class TestGraphStructureNccl(TestCase):
    """Structural CUDA-graph fidelity: the mirrored NCCL collective must sit
    on the application's producer->consumer dependency path, on views of the
    application tensors, with distinct communicators per logical group."""

    # One NCCL parent communicator per process: a second independent NCCL
    # init on the same device deadlocks, and shutting a split parent down
    # while children exist crashes later splits, so tests share the parent.
    _phys = None

    @classmethod
    def _physical_parent(cls):
        if cls._phys is None:
            from torch.distributed._virtual_pg import create_physical_group

            f = tempfile.NamedTemporaryFile(delete=False)  # noqa: SIM115
            store = dist.FileStore(f.name, 1)
            cls._phys = create_physical_group(store, 0, 1, torch.device("cuda:0"))
        return cls._phys

    def _make_world(self):
        if not TEST_CUDA or not dist.is_nccl_available():
            self.skipTest("CUDA and NCCL required")
        device = torch.device("cuda:0")
        phys = self._physical_parent()
        vpg = VirtualProcessGroup(
            1, 4, phys, mirror_split="split", output_mode="projected"
        )
        self.addCleanup(vpg.unregister)
        return vpg, phys, device

    @skipIfRocm
    def test_projected_all_gather_uses_app_buffers_in_graph(self):
        vpg, phys, device = self._make_world()
        nccl = phys._get_backend(device)
        splits_before = nccl.comm_split_count()
        g1 = vpg.new_group([0, 1], group_name="lg1").register()
        g2 = vpg.new_group([1, 3], group_name="lg2").register()
        self.addCleanup(g1.unregister)
        self.addCleanup(g2.unregister)
        self.assertIsNot(g1.physical_group, g2.physical_group)
        self.assertEqual(nccl.comm_split_count(), splits_before + 2)

        x1 = torch.ones(4, device=device)
        x2 = torch.ones(4, device=device)
        out1 = torch.zeros(4, device=device)
        out2 = torch.zeros(4, device=device)

        def step():
            p1 = x1 * 2
            w1 = g1.all_gather_single(out1, p1, AllgatherOptions())
            p2 = x2 * 3
            w2 = g2.all_gather_single(out2, p2, AllgatherOptions())
            w1.wait()
            c1 = out1 + 1
            w2.wait()
            c2 = out2 + 1
            return c1, c2

        # warmup: communicators + allocations before capture
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                step()
        torch.cuda.current_stream().wait_stream(s)

        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            c1, c2 = step()
        graph.instantiate()
        graph.replay()
        torch.cuda.synchronize()
        # projected data: physical world 1 wrote the first chunk of out{1,2}
        self.assertEqual(c1[:1], torch.tensor([3.0], device=device))
        self.assertEqual(c2[:1], torch.tensor([4.0], device=device))
        # no scratch was used: collectives ran on the app tensors
        self.assertEqual(len(g1._scratch), 0)
        self.assertEqual(len(g2._scratch), 0)

        if not _has_cuda_bindings():
            return
        nodes = graph.get_graph_data()["nodes"]
        colls = _collective_nodes(nodes)
        self.assertEqual(len(colls), 2, nodes)
        first, second = colls[0]["index"], colls[1]["index"]
        kernels = [n["index"] for n in nodes if n["node_type"] == "kernel"]
        producers, consumers = kernels[:2], kernels[-2:]
        # producer -> NCCL -> consumer paths exist
        self.assertTrue(_reaches(nodes, producers[0], first))
        self.assertTrue(_reaches(nodes, first, consumers[0]))
        self.assertTrue(_reaches(nodes, second, consumers[1]))
        # the two collective branches are independent of each other
        self.assertFalse(_reaches(nodes, first, second))
        self.assertFalse(_reaches(nodes, second, first))

    @skipIfRocm
    def test_scratch_mode_detached_from_app_buffers_in_graph(self):
        vpg, phys, device = self._make_world()
        spg = VirtualProcessGroup(
            1, 4, vpg.physical_group, group_name="scr", output_mode="scratch"
        )
        self.addCleanup(spg.unregister)
        x = torch.ones(4, device=device)
        out = torch.zeros(16, device=device)

        def step():
            p = x * 2
            w = spg.all_gather_single(out, p, AllgatherOptions())
            w.wait()
            return out + 1

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                step()  # grows scratch before capture
        torch.cuda.current_stream().wait_stream(s)
        self.assertGreater(len(spg._scratch), 0)

        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            c = step()
        graph.instantiate()
        graph.replay()
        torch.cuda.synchronize()
        # scratch mode: the logical output tensor was never written
        self.assertEqual(c, torch.ones(16, device=device))

    @skipIfRocm
    def test_event_node_counts_recorded(self):
        # Count NCCL-related event nodes so graph configurations can be
        # compared; exact counts vary by NCCL version so only record shape.
        if not _has_cuda_bindings():
            self.skipTest("cuda-bindings required")
        vpg, phys, device = self._make_world()
        g1 = vpg.new_group([0, 1], group_name="ev1").register()
        self.addCleanup(g1.unregister)
        x = torch.ones(4, device=device)
        out = torch.zeros(4, device=device)

        def step():
            w = g1.all_gather_single(out, x * 2, AllgatherOptions())
            w.wait()
            return out + 1

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            step()
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            step()
        graph.instantiate()
        nodes = graph.get_graph_data()["nodes"]
        by_type: dict = {}
        for n in nodes:
            by_type[n["node_type"]] = by_type.get(n["node_type"], 0) + 1
        # every node type observed must be accounted for; event/wait nodes
        # appear only in some NCCL configurations
        self.assertLessEqual(
            set(by_type), {"kernel", "memcpy", "memset", "event_record", "wait_event"}
        )


class TestVirtualWorldTwoProcGloo(MultiProcessTestCase):
    """Same two-physical-process topology as the NCCL test, on gloo/CPU, so
    the install + new_group + projection flow runs in NCCL-less CI."""

    @property
    def world_size(self) -> int:
        return 2

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def test_two_proc_virtual_world_projected(self):
        from torch.distributed._virtual_pg import (
            create_physical_group,
            install_virtual_world,
            uninstall_virtual_world,
        )

        store = dist.FileStore(self.file_name, self.world_size)
        phys = create_physical_group(store, self.rank, self.world_size)
        pw = self.world_size
        vpg = VirtualProcessGroup(2, 8, phys, output_mode="projected")
        install_virtual_world(vpg)
        try:
            self.assertEqual(dist.get_rank(), 2)
            self.assertEqual(dist.get_world_size(), 8)
            g1 = dist.new_group([0, 2])
            self.assertIsInstance(g1, VirtualProcessGroup)
            self.assertEqual(g1.rank(), 1)

            inp = torch.full((4,), float(self.rank + 1))
            out = torch.zeros(g1.size() * 4)
            w = g1.all_gather_single(out, inp, AllgatherOptions())
            w.wait()
            expect = torch.cat([torch.full((4,), float(r + 1)) for r in range(pw)])
            self.assertEqual(out[: pw * 4], expect)
            self.assertEqual(len(g1._scratch), 0)
        finally:
            uninstall_virtual_world()
            vpg.unregister()


class TestVirtualWorldTwoProcNccl(MultiProcessTestCase):
    """End-to-end: two physical NCCL ranks hosting a logical world of 8.

    Both processes present the SAME logical rank to the application (rank 2
    of 8) while having distinct physical ranks, install the virtual PG as
    the default world, create two logical subgroups through dist.new_group
    with distinct ncclCommSplit child communicators, and capture
    producer -> projected all-gather -> consumer in a CUDA graph on views of
    the application tensors.
    """

    @property
    def world_size(self) -> int:
        return 2

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_two_proc_virtual_world_cuda_graph(self):
        from torch.distributed._virtual_pg import (
            create_physical_group,
            install_virtual_world,
            uninstall_virtual_world,
        )

        device = torch.device(f"cuda:{self.rank}")
        torch.cuda.set_device(device)
        store = dist.FileStore(self.file_name, self.world_size)
        phys = create_physical_group(store, self.rank, self.world_size, device)
        pw = self.world_size

        # same logical rank on both physical processes, logical world of 8
        vpg = VirtualProcessGroup(
            2, 8, phys, mirror_split="split", output_mode="projected"
        )
        install_virtual_world(vpg)
        try:
            self.assertEqual(dist.get_rank(), 2)
            self.assertEqual(dist.get_world_size(), 8)

            # two logical subgroups -> two distinct physical split children;
            # every physical process calls new_group in the same order
            g1 = dist.new_group([0, 2])
            g2 = dist.new_group([2, 5, 7])
            self.assertIsInstance(g1, VirtualProcessGroup)
            self.assertIsInstance(g2, VirtualProcessGroup)
            self.assertIsNot(g1.physical_group, g2.physical_group)
            self.assertIsNot(g1.physical_group, phys)
            nccl = phys._get_backend(device)
            self.assertEqual(nccl.comm_split_count(), 2)

            x1 = torch.full((4,), float(self.rank + 1), device=device)
            x2 = torch.full((4,), float(self.rank + 1), device=device)
            out1 = torch.zeros(g1.size() * 4, device=device)
            out2 = torch.zeros(g2.size() * 4, device=device)

            def step():
                p1 = x1 * 2
                w1 = g1.all_gather_single(out1, p1, AllgatherOptions())
                p2 = x2 * 3
                w2 = g2.all_gather_single(out2, p2, AllgatherOptions())
                w1.wait()
                c1 = out1 + 1
                w2.wait()
                c2 = out2 + 1
                return c1, c2

            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    step()
            torch.cuda.current_stream().wait_stream(s)

            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph):
                c1, c2 = step()
            graph.instantiate()
            x1.fill_(float(self.rank + 10))
            graph.replay()
            torch.cuda.synchronize()

            # projected: physical all-gather wrote the first pw*4 elements of
            # the logical output from the app tensors (views, not scratch)
            expect1 = torch.cat(
                [torch.full((4,), (r + 10) * 2 + 1.0, device=device) for r in range(pw)]
            )
            self.assertEqual(c1[: pw * 4], expect1)
            expect2 = torch.cat(
                [torch.full((4,), (r + 1) * 3 + 1.0, device=device) for r in range(pw)]
            )
            self.assertEqual(c2[: pw * 4], expect2)
            self.assertEqual(len(g1._scratch), 0)
            self.assertEqual(len(g2._scratch), 0)

            if _has_cuda_bindings():
                nodes = graph.get_graph_data()["nodes"]
                kernels = [n["index"] for n in nodes if n["node_type"] == "kernel"]
                names = [
                    (n["index"], (n.get("kernel_name") or "").lower()) for n in nodes
                ]
                nccl_nodes = [i for i, kn in names if "nccl" in kn]
                # two collectives from two communicators inside the graph
                self.assertEqual(len(nccl_nodes), 2, names)
                first, second = nccl_nodes
                # producer -> NCCL -> consumer, branches independent
                self.assertTrue(_reaches(nodes, kernels[0], first))
                self.assertTrue(_reaches(nodes, first, kernels[-2]))
                self.assertFalse(_reaches(nodes, first, second))
                self.assertFalse(_reaches(nodes, second, first))
        finally:
            uninstall_virtual_world()
            vpg.unregister()


if __name__ == "__main__":
    run_tests()
