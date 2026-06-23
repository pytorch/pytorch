# Owner(s): ["oncall: profiler"]
"""Tests for the CUPTI collective (comms) monitor: the ``CommsObserver`` record
producer and its lifecycle (on_schedule / on_start / on_progress / on_end /
on_wait), the ``CommMonitorHook`` torchcomms tagging, and the hang watchdog. The
schema serializer plugins (FlightRecorder / CommDump / Clog) live in
``test_cupti_comms_plugins.py``.
"""

import json
import threading
import time
import unittest

import torch
from torch.testing._internal.common_utils import run_tests, TEST_WITH_ROCM, TestCase
from torch.utils._import_utils import _check_module_exists


TEST_CUDA = torch.cuda.is_available()
TEST_CUPTI_PYTHON = _check_module_exists("cupti") and not TEST_WITH_ROCM


def _cupti_version() -> int:
    if not TEST_CUPTI_PYTHON:
        return 0
    try:
        from torch.profiler._cupti.cupti_python import pylibcupti

        return pylibcupti().get_version()
    except Exception:
        return 0


TEST_CUPTI_V13_3 = TEST_CUPTI_PYTHON and _cupti_version() >= 130300


def _have_node_tools_id() -> bool:
    # Graph-node correlation needs cudaGraphNodeGetToolsId, gated by the driver /
    # cuda-compat (NewerDriver below ~13.3); without it the graph node-walk no-ops.
    if not TEST_CUDA:
        return False
    try:
        from torch.profiler._cupti.utils.graph_nodes import HAVE_NODE_TOOLS_ID

        return HAVE_NODE_TOOLS_ID
    except Exception:
        return False


TEST_NODE_TOOLS_ID = _have_node_tools_id()


def _symm_mem_measurer_worker(rank: int, world: int) -> None:
    # Subprocess body for TestCuptiCommsCUDA.test_symm_mem_collective_via_measurer.
    # A real symm-mem one_shot_all_reduce (its kernel is NOT nccl-named) wrapped in
    # CollectiveMeasurer must be recorded by a CommsObserver purely by its mark (the
    # observer has no name heuristic) -- exercising the measure CM (eager external id +
    # metadata). Asserts on rank 0; mp.spawn re-raises in parent.
    import os

    import torch.distributed as dist

    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE=str(world),
        LOCAL_RANK=str(rank),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT="29563",
    )
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    try:
        from torch.distributed._symmetric_memory import empty as symm_empty, rendezvous
        from torch.profiler._cupti import monitor as cupti_monitor
        from torch.profiler._cupti.comms import CollectiveMeasurer
        from torch.profiler._cupti.observers.comms import CommsObserver

        gname = dist.group.WORLD.group_name
        t = symm_empty(1024, dtype=torch.float32, device=f"cuda:{rank}")
        t.fill_(float(rank + 1))
        rendezvous(t, gname)
        obs = CommsObserver()
        assert obs.available, "CUPTI monitor unavailable in subprocess"
        mon = obs._monitor
        measurer = CollectiveMeasurer(mon)
        try:
            for _ in range(4):
                with measurer.measure(
                    "symm_one_shot_all_reduce", dtype=str(t.dtype), numel=t.numel()
                ):
                    torch.ops.symm_mem.one_shot_all_reduce(t, "sum", gname)
            torch.cuda.synchronize()
            mon.flush(sync=True)
            records = obs.poll()
        finally:
            obs.close()
            cupti_monitor._instance = None
        if rank == 0:
            named = [
                r
                for r in records
                if r.metadata.get("name") == "symm_one_shot_all_reduce"
            ]
            assert named, f"symm-mem collective not recorded (total={len(records)})"
            assert all(r.end_ns >= r.start_ns for r in records), "non-monotonic timing"
    finally:
        dist.destroy_process_group()


def _symm_mem_dispatch_worker(rank: int, world: int) -> None:
    # Like _symm_mem_measurer_worker, but the symm-mem op is captured AUTOMATICALLY by
    # SymmMemDispatchMode (it operates on a symm-mem tensor) -- no manual measure().
    import os

    import torch.distributed as dist

    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE=str(world),
        LOCAL_RANK=str(rank),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT="29564",
    )
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    try:
        from torch.distributed._symmetric_memory import empty as symm_empty, rendezvous
        from torch.profiler._cupti import monitor as cupti_monitor
        from torch.profiler._cupti.comms import (
            disable_symm_mem_dispatch,
            enable_symm_mem_dispatch,
        )
        from torch.profiler._cupti.observers.comms import CommsObserver

        gname = dist.group.WORLD.group_name
        t = symm_empty(1024, dtype=torch.float32, device=f"cuda:{rank}")
        t.fill_(float(rank + 1))
        rendezvous(t, gname)
        obs = CommsObserver()
        assert obs.available, "CUPTI monitor unavailable in subprocess"
        mon = obs._monitor
        enable_symm_mem_dispatch(mon)  # process-wide; no per-op wrapping
        try:
            for _ in range(4):
                torch.ops.symm_mem.one_shot_all_reduce(t, "sum", gname)
            torch.cuda.synchronize()
            mon.flush(sync=True)
            records = obs.poll()
        finally:
            disable_symm_mem_dispatch()
            obs.close()
            cupti_monitor._instance = None
        if rank == 0:
            named = [
                r
                for r in records
                if r.metadata.get("name") == "symm_mem::one_shot_all_reduce"
            ]
            assert named, f"symm-mem op not auto-recorded (total={len(records)})"
            assert all(r.end_ns >= r.start_ns for r in records), "non-monotonic timing"
    finally:
        dist.destroy_process_group()


def _symm_mem_dispatch_graph_worker(rank: int, world: int) -> None:
    # Like _symm_mem_dispatch_worker, but the symm-mem op is captured into a CUDA graph.
    # During capture SymmMemDispatchMode routes through _GraphCommAnchor (start event off
    # the critical path + a node-walk for the kernel's graph_node_id); replays carry no
    # external id, so the observer keeps and attributes them by graph_node_id. Exercises
    # the graph keep path (drain_collectives' metadata-resolver keep, lag-free).
    import os

    import torch.distributed as dist

    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE=str(world),
        LOCAL_RANK=str(rank),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT="29565",
    )
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    try:
        from torch.distributed._symmetric_memory import empty as symm_empty, rendezvous
        from torch.profiler._cupti import monitor as cupti_monitor
        from torch.profiler._cupti.comms import (
            disable_symm_mem_dispatch,
            enable_symm_mem_dispatch,
        )
        from torch.profiler._cupti.comms.hook import _GraphCommAnchor
        from torch.profiler._cupti.observers.comms import CommsObserver

        gname = dist.group.WORLD.group_name
        t = symm_empty(1024, dtype=torch.float32, device=f"cuda:{rank}")
        t.fill_(float(rank + 1))
        rendezvous(t, gname)
        obs = CommsObserver(start_events=True)
        assert obs.available, "CUPTI monitor unavailable in subprocess"
        mon = obs._monitor
        anchor = _GraphCommAnchor(mon)
        obs.set_event_resolver(anchor.event_resolver)
        obs.set_metadata_resolver(anchor.metadata_resolver)
        enable_symm_mem_dispatch(mon, anchor=anchor)  # anchor -> graph capture path
        try:
            for _ in range(3):  # warmup eager before capture
                torch.ops.symm_mem.one_shot_all_reduce(t, "sum", gname)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s), torch.cuda.graph(g):
                torch.ops.symm_mem.one_shot_all_reduce(t, "sum", gname)
            torch.cuda.current_stream().wait_stream(s)
            anchor.finalize(g)  # remap nodes + learn start-event ids before replay
            for _ in range(4):
                g.replay()
            torch.cuda.synchronize()
            mon.flush(sync=True)
            records = obs.poll()
        finally:
            disable_symm_mem_dispatch()
            anchor.close()
            obs.close()
            cupti_monitor._instance = None
        if rank == 0:
            graph = [
                r
                for r in records
                if r.graph_node_id
                and r.metadata.get("name") == "symm_mem::one_shot_all_reduce"
            ]
            assert len(graph) >= 4, f"graph replays not recorded ({len(graph)})"
            assert all(r.end_ns >= r.start_ns for r in records), "non-monotonic timing"
    finally:
        dist.destroy_process_group()


@unittest.skipIf(not TEST_CUPTI_PYTHON, "requires cupti-python")
class TestCuptiComms(TestCase):
    def test_comms_observer_correlate_kernels(self):
        # The per-collective kernel correlation: each eager collective kernel (graph_node
        # 0) is attributed to its collective by correlation_id -> external_id; a kernel
        # whose id is not a marked collective is dropped. Host-side.
        import numpy as np

        from torch.profiler._cupti.observers.comms import (
            _attribution,
            _correlate_kernels,
        )

        # kernels: AllReduce (corr 100), an elementwise (corr 200), RS (corr 300); all
        # eager, so graph_node 0.
        kernels = [
            (
                np.array([12, 11, 30], dtype="<i8"),  # start
                np.array([20, 15, 40], dtype="<i8"),  # end
                np.array([100, 200, 300], dtype="<u8"),  # correlation_id
                np.array([0, 0, 0], dtype="<u8"),  # graph_node
                np.array(
                    [
                        "ncclDevKernel_AllReduce",
                        "vectorized_elementwise",
                        "ncclDevKernel_ReduceScatter",
                    ],
                    dtype=object,
                ),  # name
            )
        ]
        # Each collective kernel carries its collective's (innermost) external id;
        # the elementwise (corr 200) has no comms tag.
        ext = [
            (
                np.array([42, 43], dtype="<u8"),  # external_id
                np.array([100, 300], dtype="<u8"),  # correlation_id
            )
        ]

        out = sorted(
            _correlate_kernels(
                kernels, _attribution(ext, keep_ext_ids=frozenset({42, 43}))
            ),
            key=lambda r: r["external_id"],
        )
        # corr100->42, corr300->43 (both marked collectives); the elementwise (corr200,
        # no ext tag, so not a marked collective) is dropped.
        self.assertEqual(len(out), 2)
        self.assertEqual(
            out[0],
            {
                "external_id": 42,
                "start_ns": 12,
                "end_ns": 20,
                "graph_node_id": 0,
                "name": "ncclDevKernel_AllReduce",
            },
        )
        self.assertEqual(out[1]["external_id"], 43)
        self.assertEqual((out[1]["start_ns"], out[1]["end_ns"]), (30, 40))

    def test_comms_observer_correlate_kernels_nested_chain(self):
        # The collective's external id need not be the kernel's innermost active id: a
        # nested region (e.g. a tracer) pushed inside the collective makes its id the one
        # CUPTI tags. _correlate_kernels walks the innermost id's push chain
        # (chain_resolver) and attributes the kernel to the collective id active when that
        # id was pushed. Host-side.
        import numpy as np

        from torch.profiler._cupti.observers.comms import (
            _attribution,
            _correlate_kernels,
        )

        # One collective kernel tagged with innermost id 99 (a nested region); the
        # collective is id 42, an ancestor in 99's push chain.
        kernels = [
            (
                np.array([12], dtype="<i8"),
                np.array([20], dtype="<i8"),
                np.array([100], dtype="<u8"),  # correlation_id
                np.array([0], dtype="<u8"),  # graph_node
                np.array(["ncclDevKernel_AllReduce"], dtype=object),
            )
        ]
        ext = [
            (np.array([99], dtype="<u8"), np.array([100], dtype="<u8"))
        ]  # corr100->99
        chains = {99: (42, 99)}  # 42 (collective) encloses 99 (innermost, nested)

        out = _correlate_kernels(
            kernels,
            _attribution(
                ext,
                keep_ext_ids=frozenset({42}),
                chain_resolver=lambda i: chains.get(i, (i,)),
            ),
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["external_id"], 42)  # the collective, not innermost 99
        self.assertEqual((out[0]["start_ns"], out[0]["end_ns"]), (12, 20))

        # Innermost-only (no chain): 99 is not a collective, so the kernel is dropped.
        self.assertEqual(
            _correlate_kernels(
                kernels, _attribution(ext, keep_ext_ids=frozenset({42}))
            ),
            [],
        )

    def test_comms_observer_keeps_only_marked_kernels(self):
        # A raw symm-mem kernel (not nccl-named) bracketed by CollectiveMeasurer is
        # tagged as a collective -- eager by external id, graph by a registered node.
        # _correlate_kernels keeps a kernel only when that mark is supplied; an unmarked
        # kernel (no collective external id, no registered graph node) is always dropped.
        import numpy as np

        from torch.profiler._cupti.observers.comms import (
            _attribution,
            _correlate_kernels,
        )

        kernels = [
            (
                np.array([12, 30], dtype="<i8"),  # start
                np.array([20, 45], dtype="<i8"),  # end
                np.array([100, 555], dtype="<u8"),  # correlation_id
                np.array([0, 8589934592], dtype="<u8"),  # graph_node (eager 0 / graph)
                np.array(
                    ["_triton_symm_mem_barrier", "_triton_all_gather_kernel"],
                    dtype=object,
                ),
            )
        ]
        ext = [
            (np.array([42], dtype="<u8"), np.array([100], dtype="<u8"))
        ]  # corr100->42

        # No marks: neither is a tagged collective, so both are dropped.
        self.assertEqual(_correlate_kernels(kernels, _attribution(ext)), [])

        # Marked: eager by external id 42, graph by node 8589934592 -> both kept.
        out = _correlate_kernels(
            kernels,
            _attribution(
                ext,
                keep_ext_ids=frozenset({42}),
                keep_graph_nodes=frozenset({8589934592}),
            ),
        )
        self.assertEqual(len(out), 2)
        by_ext = {r["external_id"]: r for r in out}
        self.assertEqual(by_ext[42]["name"], "_triton_symm_mem_barrier")
        self.assertEqual((by_ext[42]["start_ns"], by_ext[42]["end_ns"]), (12, 20))
        self.assertEqual(by_ext[0]["graph_node_id"], 8589934592)
        self.assertEqual(by_ext[0]["name"], "_triton_all_gather_kernel")

    def test_enable_disable_symm_mem_dispatch(self):
        # Process-wide enable is idempotent and disables cleanly (host-side; a fake
        # monitor is fine -- no ops dispatch here, so __torch_dispatch__ never fires).
        from torch.profiler._cupti.comms import (
            disable_symm_mem_dispatch,
            enable_symm_mem_dispatch,
            SymmMemDispatchMode,
        )

        mode = enable_symm_mem_dispatch(object())
        try:
            self.assertIsInstance(mode, SymmMemDispatchMode)
            self.assertIs(enable_symm_mem_dispatch(object()), mode)  # idempotent
        finally:
            disable_symm_mem_dispatch()
            disable_symm_mem_dispatch()  # idempotent no-op

    def test_comms_observer_graph_collective(self):
        # A CUDA-graph collective kernel has no EXTERNAL_CORRELATION record at replay,
        # so it is attributed by graph_node_id (external_id 0) -- kept via its registered
        # node, still timed for the graph metadata resolver to name. Eager kernels (with
        # a marked external id) take that path.
        import numpy as np

        from torch.profiler._cupti.observers.comms import (
            _attribution,
            _correlate_kernels,
        )

        kernels = [
            (
                np.array([10, 30], dtype="<i8"),  # start
                np.array([20, 45], dtype="<i8"),  # end
                np.array([100, 555], dtype="<u8"),  # correlation_id
                np.array([0, 8589934592], dtype="<u8"),  # graph_node
                np.array(
                    ["ncclDevKernel_AllReduce", "ncclDevKernel_ReduceScatter"],
                    dtype=object,
                ),
            )
        ]
        ext = [
            (np.array([42], dtype="<u8"), np.array([100], dtype="<u8"))
        ]  # eager only

        out = sorted(
            _correlate_kernels(
                kernels,
                _attribution(
                    ext,
                    keep_ext_ids=frozenset({42}),
                    keep_graph_nodes=frozenset({8589934592}),
                ),
            ),
            key=lambda r: r["graph_node_id"],
        )
        self.assertEqual(len(out), 2)
        # Eager: external id from the ext record, no graph node.
        self.assertEqual(
            out[0],
            {
                "external_id": 42,
                "start_ns": 10,
                "end_ns": 20,
                "graph_node_id": 0,
                "name": "ncclDevKernel_AllReduce",
            },
        )
        # Graph replay: no ext record -> external_id 0, keyed by graph_node_id.
        self.assertEqual(out[1]["external_id"], 0)
        self.assertEqual(out[1]["graph_node_id"], 8589934592)
        self.assertEqual(out[1]["name"], "ncclDevKernel_ReduceScatter")

    def test_comm_record_join(self):
        # The CommRecord join: eager metadata (in-flight, keyed by external_id) is
        # READ onto its timing record (NOT popped here -- poll() drops the entry after
        # this read, keyed by the completing kernel's external_id); graph metadata is
        # resolved by graph_node_id; an unknown collective becomes a timing-only
        # record. Pure host-side.
        from torch.profiler._cupti.observers.comms import _join_record, _parse

        in_flight = {42: {"func": "AllReduce", "count": 4096}}
        eager = _join_record(
            {
                "external_id": 42,
                "start_ns": 10,
                "end_ns": 30,
                "graph_node_id": 7,
                "name": "ncclDevKernel_AllReduce",
            },
            in_flight,
            None,
        )
        self.assertEqual(eager.coll_id, 42)
        self.assertEqual(eager.metadata, {"func": "AllReduce", "count": 4096})
        self.assertEqual(eager.latency_ns, 20)
        self.assertIn(42, in_flight)  # NOT popped by _join_record (poll() clears it)

        registry = {8: json.dumps({"func": "ReduceScatter"})}
        graph = _join_record(
            {
                "external_id": 0,
                "start_ns": 5,
                "end_ns": 9,
                "graph_node_id": 8,
                "name": "ncclDevKernel_RS",
            },
            {},
            registry.get,
        )
        self.assertEqual(graph.coll_id, 8)
        self.assertEqual(graph.metadata, {"func": "ReduceScatter"})

        unknown = _join_record(
            {
                "external_id": 99,
                "start_ns": 1,
                "end_ns": 2,
                "graph_node_id": 0,
                "name": "ncclX",
            },
            {},
            None,
        )
        self.assertEqual(unknown.coll_id, 99)
        self.assertEqual(unknown.metadata, {})

        self.assertEqual(_parse(None), {})
        self.assertEqual(_parse("not json"), {})
        self.assertEqual(_parse(json.dumps([1, 2])), {})  # non-object -> {}

    def test_on_start_fires_eager(self):
        # on_start for EAGER: a CUDA_EVENT (start) record keyed by correlation_id ->
        # external_id fires on_start, after on_schedule. A start arriving in a poll
        # AFTER on_end (records arrive in undefined order) still fires on_start without
        # re-firing on_schedule. Synthetic columns through _on_activities/poll.
        import numpy as np

        from torch.profiler._cupti.comms import CommRecordPlugin
        from torch.profiler._cupti.records import CudaEvent, ExternalCorrelation, Kernel

        calls: list = []

        class _Recorder(CommRecordPlugin):
            def on_schedule(self, coll_id, metadata):
                calls.append(("schedule", coll_id))

            def on_start(self, coll_id, metadata):
                calls.append(("start", coll_id, metadata.get("func")))

            def on_end(self, record):
                calls.append(("end", record.coll_id))

            def on_progress(self, in_flight):
                pass

        obs = self._bare_observer()
        obs.add_plugin(_Recorder())
        # In flight: collective 42 issued (metadata recorded by the plugin/store).
        obs._in_flight = {42: {"func": "AllReduce"}}

        from cupti.cupti import ActivityKind  # pyrefly: ignore[missing-import]

        # Poll 1: the start event (corr 100 -> ext 42) lands while the kernel is still
        # in flight. The EXTERNAL_CORRELATION record maps corr 100 -> ext 42.
        obs._on_activities(
            {
                ActivityKind.EXTERNAL_CORRELATION: {
                    int(ExternalCorrelation.EXTERNAL_ID): np.array([42], dtype="<u8"),
                    int(ExternalCorrelation.CORRELATION_ID): np.array(
                        [100], dtype="<u8"
                    ),
                },
                ActivityKind.CUDA_EVENT: {
                    int(CudaEvent.EVENT_ID): np.array([7], dtype="<u8"),
                    int(CudaEvent.CORRELATION_ID): np.array([100], dtype="<u8"),
                },
            }
        )
        obs.poll()
        # on_schedule fired before on_start; on_end has not (kernel not yet timed).
        self.assertEqual(calls, [("schedule", 42), ("start", 42, "AllReduce")])

        # Poll 2 (end-before-start): the kernel completes (on_end) and a SECOND start
        # record for the same collective arrives in the same poll AFTER the end loop.
        # on_start must still fire; on_schedule must NOT fire again.
        calls.clear()
        obs._on_activities(
            {
                ActivityKind.CONCURRENT_KERNEL: {
                    int(Kernel.START): np.array([10], dtype="<i8"),
                    int(Kernel.END): np.array([20], dtype="<i8"),
                    int(Kernel.CORRELATION_ID): np.array([100], dtype="<u8"),
                    int(Kernel.GRAPH_NODE_ID): np.array([0], dtype="<u8"),
                    int(Kernel.NAME): np.array(
                        ["ncclDevKernel_AllReduce"], dtype=object
                    ),
                },
                ActivityKind.CUDA_EVENT: {
                    int(CudaEvent.EVENT_ID): np.array([7], dtype="<u8"),
                    int(CudaEvent.CORRELATION_ID): np.array([100], dtype="<u8"),
                },
            }
        )
        obs.poll()
        # end fired; the late start still fired (no second schedule). Its metadata is
        # empty: the in-flight entry was cleared by on_end before the start processed.
        starts = [c for c in calls if c[0] == "start"]
        self.assertIn(("end", 42), calls)
        self.assertEqual(starts, [("start", 42, None)])
        self.assertNotIn("schedule", [c[0] for c in calls])
        # on_end precedes the late on_start (start processed after the on_end loop).
        self.assertLess(calls.index(("end", 42)), calls.index(starts[0]))

    def test_on_start_fires_graph_via_event_resolver(self):
        # on_start for the GRAPH path: a CUDA_EVENT whose event_id resolves (via the
        # event_resolver) to a coll_id fires on_start with the resolved metadata, and the
        # observer records the resolved start in its graph in-flight accounting (the
        # collective is outstanding until its kernel completion arrives).
        import numpy as np
        from cupti.cupti import ActivityKind  # pyrefly: ignore[missing-import]

        from torch.profiler._cupti.comms import CommRecordPlugin
        from torch.profiler._cupti.records import CudaEvent

        role = (5, frozenset({200, 201}), {"func": "ReduceScatter"})
        calls: list = []

        class _Recorder(CommRecordPlugin):
            def on_start(self, coll_id, metadata):
                calls.append((coll_id, metadata.get("func")))

        obs = self._bare_observer(
            event_resolver=lambda eid: role if eid == 99 else None
        )
        obs.add_plugin(_Recorder())

        obs._on_activities(
            {
                ActivityKind.CUDA_EVENT: {
                    int(CudaEvent.EVENT_ID): np.array([99], dtype="<u8"),
                    # No EXTERNAL_CORRELATION for graph replay -> no corr->ext mapping, so
                    # the eager branch misses and the resolver is used. Correlation id is a
                    # fresh runtime id with no ext record.
                    int(CudaEvent.CORRELATION_ID): np.array([555], dtype="<u8"),
                },
            }
        )
        obs.poll()
        self.assertEqual(calls, [(5, "ReduceScatter")])
        # The graph start is recorded in the observer's in-flight accounting: collective
        # 5 has an outstanding start (no kernel completion fed yet), so it is in flight.
        self.assertEqual(obs.in_flight(), {5: {"func": "ReduceScatter"}})

    def test_comm_monitor_hook(self):
        # The native torchcomms-hook integration: a
        # pre-hook pushes the external-corr id and records host-side metadata
        # (sizes/dtype/peer/is_p2p + comm identity) read from the typed hook args; the
        # post-hook pops. A fake comm drives the pre/post hooks the way torchcomms
        # does. Host-side, CPU tensors, no CUDA.
        import types

        from torch.profiler._cupti.comms import CommMonitorHook

        class FakeMonitor:
            def __init__(self):
                self.depth = 0
                self.added = []  # (fields, depth_when_added)

            def push_external_correlation_id(self):
                self.depth += 1
                return self.depth

            def pop_external_correlation_id(self):
                self.depth -= 1

            def add_collective_metadata(self, **fields):
                self.added.append((dict(fields), self.depth))

        class FakeComm:
            name = "default"

            def __init__(self):
                self._pre = self._post = None

            def register_pre_hook(self, cb):
                self._pre = cb
                return object()

            def register_post_hook(self, cb):
                self._post = cb
                return object()

            def get_rank(self):
                return 0

            @property
            def ranks(self):
                return [0, 1]

            def all_reduce(self, tensor):  # drive pre -> enqueue -> post
                self._pre("all_reduce", 1, types.SimpleNamespace(tensor=tensor))
                self._post(1, None)

            def send(self, tensor, peer):
                self._pre("send", 2, types.SimpleNamespace(tensor=tensor, peer=peer))
                self._post(2, None)

        mon = FakeMonitor()
        comm = FakeComm()
        CommMonitorHook(monitor=mon).register_with_comm(comm)

        t = torch.ones(1024, dtype=torch.bfloat16)
        comm.all_reduce(t)
        self.assertEqual(mon.depth, 0)  # push/pop balanced across pre/post
        self.assertEqual(len(mon.added), 1)
        meta, depth = mon.added[0]
        self.assertEqual(depth, 1)  # recorded inside the push window
        self.assertEqual(meta["process_group"], ["default", ""])
        self.assertEqual(meta["rank"], 0)
        self.assertEqual(meta["process_group_ranks"], [0, 1])  # from comm.ranks
        self.assertEqual(meta["input_sizes"], [[1024]])
        self.assertEqual(meta["output_sizes"], [[1024]])
        self.assertEqual(meta["input_dtypes"], ["torch.bfloat16"])
        self.assertFalse(meta["is_p2p"])

        # p2p: is_p2p + peer, recorded from the send args.
        comm.send(t, 3)
        meta2, _ = mon.added[1]
        self.assertTrue(meta2["is_p2p"])
        self.assertEqual(meta2["peer"], 3)

    def test_comm_monitor_hook_capture_frames(self):
        # capture_frames=True records the Python call stack at the collective call site
        # under "frames" (innermost-first {name, filename, line} dicts), with the hook's
        # own internal frames trimmed; the default keeps "frames" absent. Host-side.
        import types

        from torch.profiler._cupti.comms import CommMonitorHook

        class FakeMonitor:
            def __init__(self):
                self.added = []

            def push_external_correlation_id(self):
                return 1

            def pop_external_correlation_id(self):
                pass

            def add_collective_metadata(self, **fields):
                self.added.append(dict(fields))

        class FakeComm:
            name = "default"

            def register_pre_hook(self, cb):
                self._pre = cb

            def register_post_hook(self, cb):
                self._post = cb

            def all_reduce(self, tensor):  # the user's collective call site
                self._pre("all_reduce", 1, types.SimpleNamespace(tensor=tensor))
                self._post(1, None)

        t = torch.ones(8)

        # Default: no frames captured.
        mon = FakeMonitor()
        comm = FakeComm()
        CommMonitorHook(monitor=mon).register_with_comm(comm)
        comm.all_reduce(t)
        self.assertNotIn("frames", mon.added[0])

        # Opt-in: non-empty list of well-formed frames; the top frame is the call site
        # (the comm method), not the hook's internals.
        mon = FakeMonitor()
        comm = FakeComm()
        CommMonitorHook(monitor=mon, capture_frames=True).register_with_comm(comm)
        comm.all_reduce(t)  # the call site we expect at frames[0]
        frames = mon.added[0]["frames"]
        self.assertGreater(len(frames), 0)
        for f in frames:
            self.assertEqual(set(f), {"name", "filename", "line"})
            self.assertIsInstance(f["line"], int)
        names = [f["name"] for f in frames]
        self.assertEqual(frames[0]["name"], "all_reduce")  # innermost = call site
        self.assertNotIn("_pre_hook", names)  # hook internals trimmed
        self.assertNotIn("_capture_frames", names)
        self.assertIn("test_comm_monitor_hook_capture_frames", names)  # this test

        # max_frames keeps the most-recent N (innermost).
        mon = FakeMonitor()
        comm = FakeComm()
        CommMonitorHook(
            monitor=mon, capture_frames=True, max_frames=1
        ).register_with_comm(comm)
        comm.all_reduce(t)
        self.assertEqual(len(mon.added[0]["frames"]), 1)
        self.assertEqual(mon.added[0]["frames"][0]["name"], "all_reduce")

    def test_quiescence_stall_trips_hang_detector(self):
        # The observer's quiescence model: when no lifecycle callback has fired for the
        # timeout while collectives are still in flight, the background tick publishes the
        # in-flight set via on_progress, and HangDetectorPlugin trips once per stuck
        # collective. The suspicion flush is deferred a tick (CUPTI delivery is async).
        # Deterministic via the bare observer + injected clock; ticks driven directly.
        from torch.profiler._cupti.comms import HangDetectorPlugin

        clock = [0.0]
        tripped: list = []
        obs = self._bare_observer(quiescence_timeout_s=5.0, clock=lambda: clock[0])
        obs.add_plugin(
            HangDetectorPlugin(
                lambda cid, meta, el: tripped.append((cid, meta.get("func"))),
                clock=lambda: clock[0],
            )
        )
        obs._in_flight = {7: {"func": "AllReduce"}}
        obs.drain_collectives = lambda flush=False: []  # nothing ever completes

        obs._quiescence_tick()  # t=0: in flight but not past the timeout -> quiet
        self.assertEqual(tripped, [])
        clock[0] = 6.0
        obs._quiescence_tick()  # past timeout -> suspicion flush, verdict deferred
        self.assertEqual(tripped, [])
        clock[0] = 7.0
        obs._quiescence_tick()  # still stalled a tick later -> emit on_progress, trip 7
        self.assertEqual(tripped, [(7, "AllReduce")])
        clock[0] = 12.0
        obs._quiescence_tick()  # still stuck but already tripped -> no double-fire
        self.assertEqual(tripped, [(7, "AllReduce")])

    def test_quiescence_undelivered_completion_is_not_a_hang(self):
        # A completion merely sitting in an undelivered buffer must not be flagged: the
        # suspicion flush delivers it (on_end fires -> resets the deadline, clears the
        # collective), so the next tick sees nothing in flight and never trips.
        from torch.profiler._cupti.comms import HangDetectorPlugin

        clock = [0.0]
        tripped: list = []
        obs = self._bare_observer(quiescence_timeout_s=5.0, clock=lambda: clock[0])
        obs.add_plugin(
            HangDetectorPlugin(
                lambda cid, meta, el: tripped.append(cid), clock=lambda: clock[0]
            )
        )
        obs._in_flight = {7: {"func": "AllReduce"}}
        obs._scheduled = {7}  # issued earlier (its on_schedule already fired, not now)
        # 7's completion is delivered only by the flush (undelivered until then).
        obs.drain_collectives = lambda flush=False: (
            [
                {
                    "external_id": 7,
                    "start_ns": 1,
                    "end_ns": 2,
                    "graph_node_id": 0,
                    "name": "ncclDevKernel_AllReduce",
                }
            ]
            if flush
            else []
        )
        clock[0] = 6.0
        obs._quiescence_tick()  # stalled -> suspicion flush delivers 7 -> on_end clears it
        self.assertEqual(obs.in_flight(), {})
        clock[0] = 7.0
        obs._quiescence_tick()  # nothing in flight -> not a stall
        self.assertEqual(tripped, [])

    def test_quiescence_thread_start_stop(self):
        # The observer's background thread starts, runs ticks, and stops cleanly
        # (idempotent). Requires a (stub) monitor since start() no-ops without one.
        obs = self._bare_observer(quiescence_timeout_s=1.0)
        obs._monitor = object()
        obs._quiescence_interval_s = 0.005
        ticks = [0]
        obs._quiescence_tick = lambda: ticks.__setitem__(0, ticks[0] + 1)
        obs.start()
        obs.start()  # idempotent
        time.sleep(0.05)
        obs.stop()
        obs.stop()  # idempotent
        self.assertIsNone(obs._thread)
        self.assertGreater(ticks[0], 0)  # the thread actually ran ticks

    def test_graph_collective_in_flight_and_stall(self):
        # Graph-replay accounting lives in the observer: each resolved start bumps a
        # collective's start count and records its kernel graph_node_ids; each kernel
        # completion bumps that node's count. A collective is in flight while a node has
        # more starts than completions -- so a replay whose kernel never completes shows
        # up in in_flight() and trips the hang detector at quiescence.
        from torch.profiler._cupti.comms import HangDetectorPlugin

        clock = [0.0]
        tripped: list = []
        roles = {
            10: (1, frozenset({100}), {"func": "AllReduce"}),
            20: (2, frozenset({200}), {"func": "Reduce"}),
        }
        obs = self._bare_observer(
            event_resolver=lambda ev: roles.get(ev),
            quiescence_timeout_s=5.0,
            clock=lambda: clock[0],
        )
        obs.add_plugin(
            HangDetectorPlugin(
                lambda cid, meta, el: tripped.append((cid, meta.get("func"))),
                clock=lambda: clock[0],
            )
        )

        # Replay 1: both collectives start (events 10, 20) and both kernels complete.
        obs._cuda_events = [([10, 20], [0, 0])]
        obs.drain_collectives = lambda flush=False: [
            {
                "external_id": 0,
                "start_ns": 1,
                "end_ns": 2,
                "graph_node_id": 100,
                "name": "ncclDevKernel_AllReduce",
            },
            {
                "external_id": 0,
                "start_ns": 1,
                "end_ns": 2,
                "graph_node_id": 200,
                "name": "ncclDevKernel_Reduce",
            },
        ]
        obs.poll()
        self.assertEqual(obs.in_flight(), {})  # all kernels caught up

        # Replay 2: both start again, but collective 2's kernel (200) never completes.
        obs._cuda_events = [([10, 20], [0, 0])]
        obs.drain_collectives = lambda flush=False: [
            {
                "external_id": 0,
                "start_ns": 3,
                "end_ns": 4,
                "graph_node_id": 100,
                "name": "ncclDevKernel_AllReduce",
            },
        ]
        obs.poll()
        self.assertEqual(set(obs.in_flight()), {2})  # collective 2 outstanding

        # Quiescence: nothing more completes -> stall heartbeat trips collective 2 only.
        obs.drain_collectives = lambda flush=False: []
        clock[0] = 6.0
        obs._quiescence_tick()  # suspected
        clock[0] = 7.0
        obs._quiescence_tick()  # confirmed -> trip 2
        self.assertEqual(tripped, [(2, "Reduce")])

    @staticmethod
    def _bare_observer(*, event_resolver=None, quiescence_timeout_s=None, clock=None):
        # A CommsObserver with its state set up but NOT registered with the monitor
        # (no CUDA), so _on_activities + poll + _quiescence_tick can be driven host-side.
        import collections
        import time

        from torch.profiler._cupti.observers.comms import CommsObserver

        obs = CommsObserver.__new__(CommsObserver)
        obs._resolver = None
        obs._meta_resolver = None
        obs._event_resolver = event_resolver
        obs._wait_source = None
        obs._capture_events = True
        obs._monitor = None
        obs._quiescence_timeout_s = quiescence_timeout_s
        obs._quiescence_interval_s = 5.0
        obs._clock = clock or time.monotonic
        obs._last_activity = obs._clock()
        obs._suspected = False
        obs._stop = threading.Event()
        obs._thread = None
        obs._lock = threading.Lock()
        obs._kernels = []
        obs._ext = []
        obs._cuda_events = []
        obs._corr_to_ext = collections.OrderedDict()
        obs._in_flight = {}
        obs._graph_start_count = {}
        obs._graph_coll_nodes = {}
        obs._graph_complete_count = {}
        obs._graph_meta = {}
        obs._scheduled = set()
        obs._wait_meta = collections.OrderedDict()
        obs._past = collections.deque(maxlen=16)
        obs._plugins = []
        return obs

    def test_observer_plugin_lifecycle_order(self):
        # The per-collective lifecycle dispatch in poll(): each collective gets
        # on_schedule exactly once and always before its on_end (even when issued+
        # completed in one poll); the still-in-flight one is scheduled after the
        # completion loop. With no start events fed, on_start does not fire. on_progress
        # is NOT fired by poll() -- it's the quiescence-thread stall heartbeat (covered
        # separately). Pure host-side via a stubbed drain.
        from torch.profiler._cupti.comms import CommRecordPlugin

        calls: list = []

        class _Recorder(CommRecordPlugin):
            def on_schedule(self, coll_id, metadata):
                calls.append(("schedule", coll_id))

            def on_start(self, coll_id, metadata):
                calls.append(("start", coll_id))

            def on_progress(self, in_flight):
                calls.append(("progress", sorted(in_flight)))

            def on_end(self, record):
                calls.append(("end", record.coll_id))

        obs = self._bare_observer()
        obs.add_plugin(_Recorder())

        # Poll 1: a collective issued+completed in the same poll, plus one left in
        # flight. The completed one must get schedule-before-end; the in-flight one is
        # scheduled after the completion loop. poll() never fires on_progress.
        obs._in_flight = {7: {"func": "AllReduce"}, 8: {"func": "Reduce"}}
        completed = [
            {
                "external_id": 7,
                "start_ns": 10,
                "end_ns": 20,
                "graph_node_id": 0,
                "name": "ncclDevKernel_AllReduce",
            }
        ]
        obs.drain_collectives = lambda flush=False: completed
        recs = obs.poll()

        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0].coll_id, 7)
        self.assertEqual(
            calls,
            [
                ("schedule", 7),
                ("end", 7),  # schedule-before-end for the same-poll one
                ("schedule", 8),  # still-in-flight scheduled after the loop
            ],
        )
        self.assertNotIn("start", [c[0] for c in calls])  # no start events fed
        self.assertNotIn("progress", [c[0] for c in calls])  # poll() never emits it

        # Poll 2: nothing new completes; 8 already scheduled -> no callbacks at all.
        calls.clear()
        obs.drain_collectives = lambda flush=False: []
        obs.poll()
        self.assertEqual(calls, [])

        # Poll 3: 8 completes -> end only (it was scheduled in poll 1).
        calls.clear()
        obs.drain_collectives = lambda flush=False: [
            {
                "external_id": 8,
                "start_ns": 30,
                "end_ns": 50,
                "graph_node_id": 0,
                "name": "ncclDevKernel_Reduce",
            }
        ]
        obs.poll()
        self.assertEqual(calls, [("end", 8)])
        self.assertEqual(obs.in_flight(), {})

    def test_collective_wait_side_channel(self):
        # The comms hook owns the CPU-wait channel (not the generic monitor): the
        # per-work wait hook records waited-on external ids via _note_wait on the
        # (arbitrary) waiting thread; drain_waits swaps them out on the poll thread.
        # Thread-safe -- waits from worker threads all surface in one drain. Host-side.

        from torch.profiler._cupti.comms import CommMonitorHook

        hook = CommMonitorHook(monitor=object())  # monitor unused by the wait channel
        self.assertEqual(hook.drain_waits(), [])  # empty before any wait

        hook._note_wait(11)
        hook._note_wait(22)
        self.assertEqual(hook.drain_waits(), [11, 22])
        self.assertEqual(hook.drain_waits(), [])  # drained -> reset

        # Concurrent recorders: every id surfaces exactly once across drains.
        recorded = list(range(100))
        barrier = threading.Barrier(len(recorded))

        def _record(eid):
            barrier.wait()
            hook._note_wait(eid)

        threads = [threading.Thread(target=_record, args=(e,)) for e in recorded]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(sorted(hook.drain_waits()), recorded)

    def test_observer_on_wait_fires_with_cached_metadata(self):
        # poll() drains the wait source (the comms hook's drain_waits) and fires
        # on_wait(eid, metadata) per wait occurrence, resolving metadata from the
        # recent-metadata cache that on_schedule/on_end seed (so a wait arriving after
        # completion still has it). Drive poll() with a fake wait source -- host-side.

        from torch.profiler._cupti.comms import CommRecordPlugin

        calls: list = []

        class _Recorder(CommRecordPlugin):
            def on_schedule(self, coll_id, metadata):
                calls.append(("schedule", coll_id))

            def on_wait(self, coll_id, metadata):
                calls.append(("wait", coll_id, dict(metadata)))

            def on_end(self, record):
                calls.append(("end", record.coll_id))

        class _FakeMonitor:
            def take_external_metadata(self):
                return {}

        pending_waits: list[int] = []

        def _drain_waits():
            waits, pending_waits[:] = list(pending_waits), []
            return waits

        obs = self._bare_observer()
        obs._monitor = _FakeMonitor()
        obs._wait_source = _drain_waits
        obs.add_plugin(_Recorder())

        # Poll 1: collective 5 issued+completed (seeds _wait_meta via on_schedule/on_end),
        # and a wait on it recorded the same poll -> on_wait fires with the metadata.
        meta = {"func": "AllReduce", "coll_id": 5}
        obs._in_flight = {5: meta}
        obs.drain_collectives = lambda flush=False: [
            {
                "external_id": 5,
                "start_ns": 1,
                "end_ns": 9,
                "graph_node_id": 0,
                "name": "ncclDevKernel_AllReduce",
            }
        ]
        pending_waits[:] = [5]
        obs.poll()
        self.assertIn(("wait", 5, meta), calls)
        self.assertLess(  # on_wait fires after on_end (drained at end of poll)
            calls.index(("end", 5)), calls.index(("wait", 5, meta))
        )

        # Poll 2: 5 is long gone from in-flight, but a late wait still resolves its
        # cached metadata; an unknown id yields {} (no cache entry). Per-occurrence:
        # two waits on 5 fire on_wait twice.
        calls.clear()
        obs.drain_collectives = lambda flush=False: []
        pending_waits[:] = [5, 5, 999]
        obs.poll()
        self.assertEqual(
            [c for c in calls if c[0] == "wait"],
            [("wait", 5, meta), ("wait", 5, meta), ("wait", 999, {})],
        )


@unittest.skipIf(not TEST_CUDA, "CUDA required")
class TestCuptiCommsCUDA(TestCase):
    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    def test_comms_observer_records_join_metadata(self):
        # End-to-end on device (no NCCL needed): push an external id around each
        # "collective", inject its metadata into the store (as the NCCL plugin would
        # at startEvent), run a kernel, and confirm CommsObserver.poll() pairs the GPU
        # timing with the metadata and notifies a plugin. Each collective's kernel
        # completes (non-zero end), so poll() clears it from in-flight; a hung
        # collective (no completing kernel) would instead linger and be flagged.
        # Each kernel is kept by its mark (the pushed external id carries metadata), so
        # the workload's relu/matmul stand in for collectives.
        from torch.profiler._cupti import monitor as cupti_monitor
        from torch.profiler._cupti.comms import CommRecordPlugin
        from torch.profiler._cupti.observers.comms import CommsObserver

        # Drop the singleton after teardown so later tests get a fresh monitor.
        self.addCleanup(setattr, cupti_monitor, "_instance", None)
        obs = CommsObserver()
        if not obs.available:
            self.skipTest("CUPTI monitor unavailable (v2 subscribe failed)")

        seen: list = []

        class _Collector(CommRecordPlugin):
            def on_end(self, record):
                seen.append(record)

        obs.add_plugin(_Collector())
        native = torch._C._profiler._cupti_monitor
        mon = obs._monitor
        try:
            x = torch.randn(128, 128, device="cuda")
            for i in range(4):
                mon.push_external_correlation_id()
                native.metadata_put_external(json.dumps({"func": "FakeColl", "seq": i}))
                x = torch.relu(x @ x)
                mon.pop_external_correlation_id()
            x.sum().item()
            torch.cuda.synchronize()
            mon.flush(sync=True)  # deterministic delivery of metadata + timing
            records = obs.poll()
        finally:
            obs.close()

        self.assertGreater(len(records), 0)
        self.assertTrue(all(r.end_ns >= r.start_ns for r in records))
        # Metadata paired onto timing (one record per pushed id carries the blob).
        named = [r for r in records if r.metadata.get("func") == "FakeColl"]
        self.assertGreater(len(named), 0)
        # Every collective's kernel completed, so its in-flight entry was cleared.
        self.assertEqual(len(obs.in_flight()), 0)
        self.assertEqual(len(obs.past()), len(records))
        self.assertEqual(seen, records)  # the plugin saw every completed record

    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    @unittest.skipIf(torch.cuda.device_count() < 2, "requires 2 GPUs")
    def test_symm_mem_collective_via_measurer(self):
        # End-to-end on 2 ranks: a raw symm-mem collective (one_shot_all_reduce, whose
        # kernel is not "nccl"-named) wrapped in CollectiveMeasurer is recorded by the
        # CommsObserver purely by its mark (no name heuristic) -- exercising the measure
        # context manager (eager external id + metadata). The worker asserts; mp.spawn
        # re-raises any subprocess failure here.
        import torch.multiprocessing as mp

        mp.spawn(_symm_mem_measurer_worker, args=(2,), nprocs=2)

    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    @unittest.skipIf(torch.cuda.device_count() < 2, "requires 2 GPUs")
    def test_symm_mem_collective_via_dispatch_mode(self):
        # End-to-end on 2 ranks: a symm_mem.* dispatcher op (one_shot_all_reduce) is
        # auto-measured by SymmMemDispatchMode because it operates on a symm-mem tensor,
        # and recorded by the CommsObserver -- no manual measure(). The worker asserts;
        # mp.spawn re-raises any subprocess failure here.
        import torch.multiprocessing as mp

        mp.spawn(_symm_mem_dispatch_worker, args=(2,), nprocs=2)

    @unittest.skipIf(not TEST_CUPTI_V13_3, "requires libcupti >= 13.3")
    @unittest.skipIf(not TEST_NODE_TOOLS_ID, "requires cudaGraphNodeGetToolsId")
    @unittest.skipIf(torch.cuda.device_count() < 2, "requires 2 GPUs")
    def test_symm_mem_collective_via_dispatch_mode_graph(self):
        # End-to-end on 2 ranks: a symm_mem.* dispatcher op captured into a CUDA graph is
        # auto-measured by SymmMemDispatchMode through the _GraphCommAnchor, and its
        # replays are recorded by the CommsObserver and attributed by graph_node_id -- no
        # manual measure(). The worker asserts; mp.spawn re-raises any subprocess failure.
        import torch.multiprocessing as mp

        mp.spawn(_symm_mem_dispatch_graph_worker, args=(2,), nprocs=2)


if __name__ == "__main__":
    run_tests()
