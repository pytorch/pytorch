# Owner(s): ["oncall: distributed"]

import gc
import os
import time
import unittest
import weakref
from datetime import timedelta

import test_c10d_common

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.nn as nn
from torch._C._distributed_c10d import _create_work_from_future, FakeProcessGroup
from torch.distributed._process_group_subclass import _ALIAS_PAIRS, _NORMALIZED_METHODS
from torch.distributed.distributed_c10d import (
    _coalescing_manager,
    _get_default_group,
    _register_process_group,
    _unregister_process_group,
    ReconfigureOptions,
)
from torch.futures import Future
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    MultiThreadedTestCase,
)
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed.fake_pg import FakeStore


def create_work(result):
    future = Future()
    future.set_result(result)
    return _create_work_from_future(future)


class MyWork(dist._Work):
    def __init__(self, result, pg):
        super().__init__()
        self.result_ = result
        self.future_ = torch.futures.Future()
        self.future_.set_result(result)
        self.pg_ = weakref.ref(pg)

    def wait(self, timeout):
        self.pg_().wait_count += 1
        return True

    def get_future(self):
        self.pg_().get_future_count += 1
        return self.future_


class LonelyRankProcessGroup(dist.ProcessGroup):
    """
    This PG only supports world_size of 1
    """

    def __init__(self, rank, world, use_wrapper):
        super().__init__(rank, world)
        if rank != 0:
            raise AssertionError(f"Expected rank == 0, got {rank}")
        if world != 1:
            raise AssertionError(f"Expected world == 1, got {world}")

        self._rank = rank
        self._world = world
        self.wait_count = 0
        self.get_future_count = 0
        self.use_wrapper = use_wrapper
        self._work = []

    def broadcast(self, tensor_list, opts):
        if self.use_wrapper:
            return create_work(tensor_list)
        res = MyWork(tensor_list, self)
        self._work.append(res)
        return res

    def allgather(self, output_tensors, input_tensor, opts):
        for o, i in zip(output_tensors[0], input_tensor):
            o.copy_(i)
        if self.use_wrapper:
            return create_work(output_tensors)

        res = MyWork(output_tensors, self)
        self._work.append(res)

        return res

    def allreduce(self, tensors, opts):
        if self.use_wrapper:
            return create_work(tensors)
        res = MyWork(tensors, self)
        self._work.append(res)
        return res

    def getSize(self):
        return self._world

    def getBackendName(self):
        return "lonely-pg"

    def __repr__(self):
        return f"PLG w:{self._world} r:{self._rank}"


class DummyAttrProcessGroup(dist.ProcessGroup):
    def getRank(self):
        return 123

    def getSize(self):
        return 456

    def getBackendName(self):
        return "dummy-attr"

    def setGroupName(self, name) -> None:
        self._group_name = "py:" + name

    def getGroupName(self) -> str:
        return self._group_name

    def setGroupDesc(self, group_desc) -> None:
        self._group_desc = "py:" + group_desc

    def getGroupDesc(self) -> str:
        return self._group_desc


class StoreProcessGroup(dist.ProcessGroup):
    """
    A ProcessGroup constructed with the 3-arg (store, rank, size) constructor.
    Used to verify the store is accessible and that a Python subclass built
    this way is routed through the PyProcessGroup trampoline.
    """

    def __init__(self, store, rank, world):
        super().__init__(store, rank, world)
        self._rank = rank
        self._world = world

    def getBackendName(self):
        return "store-pg"


class CoalescingProcessGroup(test_c10d_common.DummyProcessGroup):
    """
    A ProcessGroup that advertises coalescing support and records coalescing
    calls, so both the coalescing manager and the batch_isend_irecv coalescing
    path can be exercised against a pure-Python backend. send/recv (which add 1
    and 2 respectively) are inherited from DummyProcessGroup.
    """

    def __init__(self, rank, world):
        super().__init__(rank, world)
        self.start_coalescing_count = 0
        self.end_coalescing_count = 0

    @property
    def supports_coalescing(self):
        return True

    def start_coalescing(self, device):
        self.start_coalescing_count += 1

    def end_coalescing(self, device):
        self.end_coalescing_count += 1
        return create_work([])


class ReconfigurableProcessGroup(dist.ProcessGroup):
    """
    A Python ProcessGroup that records reconfigure calls. Used to verify the
    torch.distributed reconfigure helpers delegate to ProcessGroup.
    """

    def __init__(self, rank, world):
        super().__init__(rank, world)
        self.reconfigure_opts = None

    @property
    def supports_reconfigure(self):
        return True

    def get_reconfigure_handle(self):
        return "handle-for-rank-0"

    def reconfigure(self, opts):
        self.reconfigure_opts = opts
        return create_work(None)


class WindowProcessGroup(dist.ProcessGroup):
    """
    A Python ProcessGroup that records new_window calls. Used to verify the
    torch.distributed window helpers delegate to ProcessGroup.
    """

    def __init__(self, rank, world):
        super().__init__(rank, world)
        self.new_window_tensor = "unset"

    @property
    def supports_window(self):
        return True

    def new_window(self, tensor=None):
        self.new_window_tensor = tensor
        return "fake-window"


# We cannot use parametrize as some tests are defined on the base class and use _get_process_group
class AbstractDDPSingleRank(test_c10d_common.CommonDistributedDataParallelTest):
    def setUp(self):
        super().setUp()
        self._spawn_threads()

    @property
    def world_size(self):
        return 1

    def _get_process_group(self):
        return LonelyRankProcessGroup(self.rank, self.world_size, self.use_wrapper)

    def test_ddp_invoke_work_object(self):
        pg = self._get_process_group()

        torch.manual_seed(123)
        model = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
        wrapped_model = model
        input_tensor = torch.rand(2)
        model = DDP(model, process_group=pg)
        model(input_tensor).sum().backward()

        ddp_grad = wrapped_model[0].bias.grad.clone()

        wrapped_model.zero_grad()
        wrapped_model(input_tensor).sum().backward()
        self.assertEqual(wrapped_model[0].bias.grad, ddp_grad)
        if not self.use_wrapper:
            self.assertTrue(pg.wait_count > 0)
            self.assertTrue(pg.get_future_count > 0)

    def test_ddp_with_pypg(self):
        pg = self._get_process_group()

        self._test_ddp_with_process_group(pg, [torch.device("cpu")], device_ids=None)

    def test_ddp_with_pypg_with_grad_views(self):
        pg = self._get_process_group()

        self._test_ddp_with_process_group(
            pg, [torch.device("cpu")], device_ids=None, gradient_as_bucket_view=True
        )

    def test_ddp_no_init_sync(self):
        pg = self._get_process_group()

        model = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
        model = DDP(model, process_group=pg, init_sync=False)

        self.assertEqual(pg.wait_count, 0)
        self.assertEqual(pg.get_future_count, 0)

    def test_manual_backward_finalization_pickle_compatibility(self):
        # Inherited from CommonDistributedDataParallelTest via
        # _create_ddp_model(), which here constructs DDP with the custom,
        # non-default process group returned by _get_process_group() (see
        # above). DDP pickling is only supported with the default process
        # group (DistributedDataParallel._check_default_group()), so this
        # inherited test does not apply to these process-group-wrapping
        # test classes.
        self.skipTest(
            "DDP is constructed with a non-default process group in "
            "AbstractDDPSingleRank subclasses; pickling requires the "
            "default process group"
        )


class TestDDPWithWorkSubclass(AbstractDDPSingleRank, MultiThreadedTestCase):
    @property
    def use_wrapper(self):
        return False


class TestDDPWithWorkWrapper(AbstractDDPSingleRank, MultiThreadedTestCase):
    @property
    def use_wrapper(self):
        return True


class BlockWork(dist._Work):
    """
    Dummy work that is used to test blocking the current stream.
    """

    def __init__(self):
        super().__init__()
        self.future_ = torch.futures.Future()

    def get_future(self):
        return self.future_


class TestPyProcessGroup(TestCase):
    def test_attr_overrides(self):
        pg = DummyAttrProcessGroup(0, 1)
        self.assertEqual(pg.name(), "dummy-attr")
        self.assertEqual(pg.rank(), 123)
        self.assertEqual(pg.size(), 456)

        pg._set_group_name("name")
        self.assertEqual(pg.group_name, "py:name")

        pg._set_group_desc("desc")
        self.assertEqual(pg.group_desc, "py:desc")

    def test_store_constructor(self):
        store = dist.HashStore()
        store.set("test_key", "test_value")

        pg = StoreProcessGroup(store, 0, 1)

        # The store passed to the 3-arg constructor is accessible via the PG
        # and is the same store object we passed in.
        group_store = pg.get_group_store()
        self.assertIs(group_store, store)
        self.assertEqual(group_store.get("test_key"), b"test_value")

        # A Python subclass built via the 3-arg constructor must be routed
        # through the PyProcessGroup trampoline so the getBackendName override
        # dispatches back into Python; a raw C++ ProcessGroup would return the
        # base backend name ("undefined") instead.
        self.assertEqual(pg.name(), "store-pg")
        self.assertEqual(pg.rank(), 0)
        self.assertEqual(pg.size(), 1)

    def test_all_gather_single(self):
        # all_gather_into_tensor_out calls group->all_gather_single(...) in C++,
        # which dispatches through the PyProcessGroup trampoline into the Python
        # DummyProcessGroup.all_gather_single override (it copies the input into
        # each chunk). Without the override the base impl needs a backend and
        # raises, so a correct output proves the override the PR adds was hit.
        pg = test_c10d_common.DummyProcessGroup(0, 1)
        _register_process_group("test_all_gather_single", pg)
        try:
            input = torch.ones(2)
            output = torch.empty(2)
            torch.ops._c10d_functional.all_gather_into_tensor_out(
                input, 1, "test_all_gather_single", out=output
            )
            torch.ops._c10d_functional.wait_tensor(output)
            self.assertIn("all_gather_single", pg.collectives_called)
            self.assertEqual(output, torch.ones(2))
        finally:
            _unregister_process_group("test_all_gather_single")

    def test_reduce_scatter_single(self):
        # No functional collective dispatches to reduce_scatter_single (they use
        # the coalesced variant), so this is a direct sanity check that the
        # override runs and forwards its buffers.
        pg = test_c10d_common.DummyProcessGroup(0, 1)
        input = torch.ones(2)
        output = torch.zeros(2)
        pg.reduce_scatter_single(output, input).wait()
        self.assertIn("reduce_scatter_single", pg.collectives_called)
        self.assertEqual(output, torch.ones(2))

    # reduce / gather / scatter / alltoall share their Python name with the
    # bound method, so a plain instance call hits the Python override directly
    # (a sanity check that it runs and forwards its buffers; no functional
    # collective dispatches to these as ProcessGroup virtuals). recvAnysource is
    # bound under the non-shadowed name recv_anysource, so pg.recv_anysource()
    # routes through the C++ virtual into the trampoline. DummyProcessGroup
    # records each dispatched collective and applies a recognizable mutation
    # (reduce adds 3, recvAnysource adds 4, gather/scatter/alltoall copy).
    def test_reduce(self):
        pg = test_c10d_common.DummyProcessGroup(0, 1)
        tensor = torch.zeros(4)
        pg.reduce([tensor]).wait()
        self.assertIn("reduce", pg.collectives_called)
        self.assertEqual(tensor, torch.zeros(4) + 3)

    def test_gather(self):
        pg = test_c10d_common.DummyProcessGroup(0, 1)
        input = torch.arange(4, dtype=torch.float32)
        output = torch.zeros(4)
        pg.gather([[output]], [input]).wait()
        self.assertIn("gather", pg.collectives_called)
        self.assertEqual(output, input)

    def test_scatter(self):
        pg = test_c10d_common.DummyProcessGroup(0, 1)
        input = torch.arange(4, dtype=torch.float32)
        output = torch.zeros(4)
        pg.scatter([output], [[input]]).wait()
        self.assertIn("scatter", pg.collectives_called)
        self.assertEqual(output, input)

    def test_alltoall(self):
        pg = test_c10d_common.DummyProcessGroup(0, 1)
        input = torch.arange(4, dtype=torch.float32)
        output = torch.zeros(4)
        pg.alltoall([output], [input]).wait()
        self.assertIn("alltoall", pg.collectives_called)
        self.assertEqual(output, input)

    def test_recv_anysource(self):
        # recv_anysource is bound to the recvAnysource virtual under a different
        # name, so this routes through the C++ trampoline into the override.
        pg = test_c10d_common.DummyProcessGroup(0, 1)
        tensor = torch.zeros(4)
        pg.recv_anysource([tensor], 7).wait()
        self.assertIn("recvAnysource", pg.collectives_called)
        self.assertEqual(tensor, torch.zeros(4) + 4)

    def test_coalescing_manager(self):
        # The coalescing manager calls _start_coalescing / _end_coalescing, which
        # route through the C++ virtual into the PyProcessGroup trampoline and
        # dispatch to the start_coalescing / end_coalescing overrides; the work
        # returned by end_coalescing is collected.
        pg = CoalescingProcessGroup(0, 1)
        device = torch.device("cpu")
        with _coalescing_manager(pg, device, async_ops=True) as cm:
            pass
        cm.wait()

        self.assertEqual(pg.start_coalescing_count, 1)
        self.assertEqual(pg.end_coalescing_count, 1)
        self.assertEqual(len(cm.works), 1)

    def test_abort_shutdown(self) -> None:
        # verify this are noops
        pg = DummyAttrProcessGroup(0, 1)
        pg.abort()
        pg.shutdown()

    def test_reconfigure_delegation(self) -> None:
        pg = ReconfigurableProcessGroup(0, 1)

        self.assertTrue(dist._supports_reconfigure(group=pg))
        self.assertEqual(dist._get_reconfigure_handle(group=pg), "handle-for-rank-0")

        timeout = timedelta(seconds=30)
        work = dist._reconfigure(
            uuid=7,
            handles=["a", "b"],
            group=pg,
            timeout=timeout,
            hints={"k": "v"},
        )
        self.assertIsNotNone(work)

        # The helper builds a ReconfigureOptions and forwards it unchanged.
        opts = pg.reconfigure_opts
        self.assertIsNotNone(opts)
        self.assertEqual(opts.uuid, 7)
        self.assertEqual(opts.handles, ["a", "b"])
        self.assertEqual(opts.timeout, timeout)
        self.assertEqual(opts.hints, {"k": "v"})

    def test_reconfigure_rejects_multiple_backends(self) -> None:
        pg = dist.ProcessGroup(0, 1)
        pg._register_backend(torch.device("cpu"), dist.ProcessGroup.BackendType.GLOO)
        pg._register_backend(torch.device("cuda"), dist.ProcessGroup.BackendType.NCCL)

        msg = "multiple backends"
        with self.assertRaisesRegex(RuntimeError, msg):
            pg.supports_reconfigure
        with self.assertRaisesRegex(RuntimeError, msg):
            pg.get_reconfigure_handle()
        with self.assertRaisesRegex(RuntimeError, msg):
            pg.reconfigure(ReconfigureOptions())

    def test_reconfigure_cpp_dispatch(self):
        pg = ReconfigurableProcessGroup(0, 1)
        self.assertTrue(dist.ProcessGroup.supports_reconfigure.__get__(pg))
        self.assertEqual(
            dist.ProcessGroup.get_reconfigure_handle(pg), "handle-for-rank-0"
        )
        opts = ReconfigureOptions()
        opts.uuid = 13
        opts.handles = ["first", "second"]
        opts.timeout = timedelta(seconds=7)
        opts.hints = {"key": "value"}
        work = dist.ProcessGroup.reconfigure(pg, opts)
        self.assertTrue(work.wait())
        self.assertIsNone(work.get_future().wait())
        self.assertEqual(pg.reconfigure_opts.uuid, opts.uuid)
        self.assertEqual(pg.reconfigure_opts.handles, opts.handles)
        self.assertEqual(pg.reconfigure_opts.timeout, opts.timeout)
        self.assertEqual(pg.reconfigure_opts.hints, opts.hints)

    def test_reconfigure_cpp_python_work_lifetime(self):
        class PG(ReconfigurableProcessGroup):
            def reconfigure(self, opts):
                work = MyWork([], self)
                self.work_ref = weakref.ref(work)
                return work

        pg = PG(0, 1)
        pg.wait_count = 0
        pg.get_future_count = 0
        work = dist.ProcessGroup.reconfigure(pg, ReconfigureOptions())
        gc.collect()
        self.assertIsNotNone(pg.work_ref())
        self.assertTrue(work.wait())
        self.assertEqual(work.get_future().wait(), [])
        self.assertEqual(pg.wait_count, 1)
        self.assertEqual(pg.get_future_count, 1)
        del work
        gc.collect()
        self.assertIsNone(pg.work_ref())

    def test_reconfigure_cpp_capability_false(self):
        class PG(ReconfigurableProcessGroup):
            @property
            def supports_reconfigure(self):
                return False

        self.assertFalse(dist.ProcessGroup.supports_reconfigure.__get__(PG(0, 1)))

    def test_reconfigure_cpp_fallback(self):
        class PG(dist.ProcessGroup):
            pass

        class DelegatingPG(PG):
            @property
            def supports_reconfigure(self):
                return super().supports_reconfigure

            def get_reconfigure_handle(self):
                return super().get_reconfigure_handle()

            def reconfigure(self, opts):
                return super().reconfigure(opts)

        for cls in (PG, DelegatingPG):
            with self.subTest(cls=cls):
                store = dist.HashStore()
                pg = cls(store, 0, 1)
                backend = dist.ProcessGroupGloo(store, 0, 1, enable_reconfigure=True)
                pg._set_default_backend(dist.ProcessGroup.BackendType.GLOO)
                pg._register_backend(
                    torch.device("cpu"), dist.ProcessGroup.BackendType.GLOO, backend
                )
                try:
                    self.assertTrue(dist.ProcessGroup.supports_reconfigure.__get__(pg))
                    handle = dist.ProcessGroup.get_reconfigure_handle(pg)
                    self.assertEqual(handle, backend.get_reconfigure_handle())
                    opts = ReconfigureOptions()
                    opts.uuid = 14
                    opts.handles = [handle]
                    self.assertTrue(dist.ProcessGroup.reconfigure(pg, opts).wait())
                finally:
                    pg.shutdown()

    def test_reconfigure_cpp_errors(self):
        class PG(dist.ProcessGroup):
            @property
            def supports_reconfigure(self):
                raise ValueError("capability error")

            def get_reconfigure_handle(self):
                raise ValueError("handle error")

            def reconfigure(self, opts):
                raise ValueError("reconfigure error")

        pg = PG(0, 1)
        # Repeated property failures must not leave the recursion guard active.
        for _ in range(2):
            with self.assertRaisesRegex(ValueError, "capability error"):
                dist.ProcessGroup.supports_reconfigure.__get__(pg)
        with self.assertRaisesRegex(ValueError, "handle error"):
            dist.ProcessGroup.get_reconfigure_handle(pg)
        with self.assertRaisesRegex(ValueError, "reconfigure error"):
            dist.ProcessGroup.reconfigure(pg, ReconfigureOptions())

    def test_window_delegation(self) -> None:
        pg = WindowProcessGroup(0, 1)

        self.assertTrue(dist._supports_window(group=pg))

        # With no tensor, new_window is called with tensor=None.
        self.assertEqual(dist._new_window(group=pg), "fake-window")
        self.assertIsNone(pg.new_window_tensor)

        # A tensor is forwarded through to ProcessGroup.new_window.
        t = torch.zeros(4)
        self.assertEqual(dist._new_window(t, group=pg), "fake-window")
        self.assertIs(pg.new_window_tensor, t)

    @unittest.skipIf(not TEST_CUDA, "no cuda/xpu")
    def test_block_current_stream(self) -> None:
        torch.cuda.synchronize()

        stream = torch.cuda.Stream()
        with stream:
            # nothing in queue so instantly resolves
            event1 = torch.cuda.Event()
            event1.record()
            event1.synchronize()
            self.assertTrue(event1.query())

            work = BlockWork()
            work.block_current_stream()

            # stream is blocked so doesn't resolve
            event = torch.cuda.Event()
            event.record()
            time.sleep(0.1)
            self.assertFalse(event.query())

            # resolve the work
            work.get_future().set_result(None)

            stream.synchronize()
            self.assertTrue(event.query())

    @unittest.skipIf(not TEST_CUDA, "no cuda/xpu")
    def test_block_current_stream_use_after_free(self) -> None:
        """
        This tests that the CPU control tensor is not freed before the CUDA kernel executes.
        """
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        with stream:
            a = BlockWork()
            a.block_current_stream()

            b = BlockWork()
            b.block_current_stream()

            # unblock b first though a is still blocking
            b.get_future().set_result(None)
            # delete b
            del b

            # a is still blocking so this doesn't resolve
            event = torch.cuda.Event()
            event.record()
            time.sleep(0.1)
            self.assertFalse(event.query())

            # unblock a
            a.get_future().set_result(None)

            stream.synchronize()
            self.assertTrue(event.query())


class FakeBackedProcessGroup(dist.ProcessGroup):
    """
    A Python ProcessGroup whose collectives are executed by a FakeProcessGroup
    backend. Subclasses override collectives, record what they receive, and
    delegate to the backend through super().
    """

    def __init__(self, store, rank, size):
        super().__init__(store, rank, size)
        backend = FakeProcessGroup._create_internal(rank, world_size=size)
        self._register_backend(
            torch.device("cpu"), dist.ProcessGroup.BackendType.CUSTOM, backend
        )
        self._set_default_backend(dist.ProcessGroup.BackendType.CUSTOM)
        self.calls = []


class CanonicalProcessGroup(FakeBackedProcessGroup):
    """Overrides every collective that has convenience overloads using exactly
    the signature of the C++ virtual."""

    def allreduce(self, tensors, opts):
        self.calls.append(("allreduce", tensors, opts))
        return super().allreduce(tensors, opts)

    def broadcast(self, tensors, opts):
        self.calls.append(("broadcast", tensors, opts))
        return super().broadcast(tensors, opts)

    def reduce(self, tensors, opts):
        self.calls.append(("reduce", tensors, opts))
        return super().reduce(tensors, opts)

    def allgather(self, output_tensors, input_tensors, opts):
        self.calls.append(("allgather", output_tensors, input_tensors, opts))
        return super().allgather(output_tensors, input_tensors, opts)

    def gather(self, output_tensors, input_tensors, opts):
        self.calls.append(("gather", output_tensors, input_tensors, opts))
        return super().gather(output_tensors, input_tensors, opts)

    def scatter(self, output_tensors, input_tensors, opts):
        self.calls.append(("scatter", output_tensors, input_tensors, opts))
        return super().scatter(output_tensors, input_tensors, opts)

    def reduce_scatter(self, output_tensors, input_tensors, opts):
        self.calls.append(("reduce_scatter", output_tensors, input_tensors, opts))
        return super().reduce_scatter(output_tensors, input_tensors, opts)

    def all_to_all_single(self, output, input, output_splits, input_splits, opts):
        self.calls.append(
            ("all_to_all_single", output, input, output_splits, input_splits, opts)
        )
        return super().all_to_all_single(
            output, input, output_splits, input_splits, opts
        )

    def barrier(self, opts):
        self.calls.append(("barrier", opts))
        return super().barrier(opts)


class VarargsProcessGroup(FakeBackedProcessGroup):
    """Overrides written before normalization existed, which parse the
    convenience forms themselves."""

    def allreduce(self, *args, **kwargs):
        tensors = args[0] if args else kwargs["tensors"]
        if isinstance(tensors, torch.Tensor):
            tensors = [tensors]
        opts = args[1] if len(args) > 1 else kwargs.get("opts", kwargs.get("op"))
        if not isinstance(opts, dist.AllreduceOptions):
            op = dist.ReduceOp.SUM if opts is None else opts
            opts = dist.AllreduceOptions()
            opts.reduceOp = op
        self.calls.append(("allreduce", tensors, opts))
        return super().allreduce(tensors, opts)

    def broadcast(self, tensor_list, opts=None):
        self.calls.append(("broadcast", tensor_list, opts))
        return create_work(tensor_list)

    def barrier(self, opts=None):
        # Like ProcessLocalGroup, call a collective with a keyword argument that
        # only this override's signature knows about.
        return self.broadcast(tensor_list=[torch.ones(1)])


class DeprecatedFallbackProcessGroup(FakeBackedProcessGroup):
    """Overrides only the deprecated name of a pair the trampoline falls back
    to."""

    def alltoall_base(self, output, input, output_splits, input_splits, opts):
        self.calls.append(
            ("alltoall_base", output, input, output_splits, input_splits, opts)
        )
        return super().alltoall_base(output, input, output_splits, input_splits, opts)


class TestProcessGroupSubclassNormalization(TestCase):
    def _init_pg(self, pg_cls):
        def create(common_opts, backend_options):
            return pg_cls(
                common_opts.store, common_opts.group_rank, common_opts.group_size
            )

        dist.Backend.register_backend(
            "pypg_subclass", create, extended_api=True, devices=["cpu"]
        )
        dist.init_process_group(
            "pypg_subclass", rank=0, world_size=2, store=FakeStore()
        )
        self.addCleanup(dist.destroy_process_group)
        pg = dist.group.WORLD
        self.assertIsInstance(pg, pg_cls)
        return pg

    def _pop_call(self, pg, name):
        self.assertEqual(len(pg.calls), 1, pg.calls)
        call = pg.calls.pop()
        self.assertEqual(call[0], name)
        return call[1:]

    def test_canonical_override_convenience_overloads(self):
        pg = self._init_pg(CanonicalProcessGroup)
        unset = timedelta(milliseconds=-1)
        timeout = timedelta(seconds=5)
        t = torch.ones(2)
        scalar = torch.tensor(1.0)
        o0, o1, i0, i1 = (torch.zeros(2) for _ in range(4))

        # allreduce(tensor, op, timeout) and allreduce(tensors, op, timeout)
        pg.allreduce(scalar, op=dist.ReduceOp.MAX).wait()
        tensors, opts = self._pop_call(pg, "allreduce")
        self.assertEqual(len(tensors), 1)
        self.assertIs(tensors[0], scalar)
        self.assertEqual(opts.reduceOp.op, dist.ReduceOp.MAX)
        self.assertEqual(opts.timeout, unset)

        pg.allreduce(t, dist.ReduceOp.MIN, timeout).wait()
        tensors, opts = self._pop_call(pg, "allreduce")
        self.assertIs(tensors[0], t)
        self.assertEqual(opts.reduceOp.op, dist.ReduceOp.MIN)
        self.assertEqual(opts.timeout, timeout)

        pg.allreduce([t], dist.ReduceOp.PRODUCT).wait()
        tensors, opts = self._pop_call(pg, "allreduce")
        self.assertIs(tensors[0], t)
        self.assertEqual(opts.reduceOp.op, dist.ReduceOp.PRODUCT)

        # A canonical call without opts gets the default options.
        pg.allreduce([t]).wait()
        tensors, opts = self._pop_call(pg, "allreduce")
        self.assertEqual(opts.reduceOp.op, dist.ReduceOp.SUM)
        self.assertEqual(opts.timeout, unset)

        # broadcast(tensor, root, timeout)
        pg.broadcast(scalar, 1, timeout).wait()
        tensors, opts = self._pop_call(pg, "broadcast")
        self.assertIs(tensors[0], scalar)
        self.assertEqual(opts.rootRank, 1)
        self.assertEqual(opts.timeout, timeout)

        # reduce(tensor, root, op, timeout)
        pg.reduce(scalar, root=1, op=dist.ReduceOp.MAX).wait()
        tensors, opts = self._pop_call(pg, "reduce")
        self.assertIs(tensors[0], scalar)
        self.assertEqual(opts.rootRank, 1)
        self.assertEqual(opts.reduceOp.op, dist.ReduceOp.MAX)

        # allgather(list, Tensor, timeout)
        pg.allgather([o0, o1], t, timeout=timeout).wait()
        outputs, inputs, opts = self._pop_call(pg, "allgather")
        self.assertEqual(len(outputs), 1)
        self.assertIs(outputs[0][0], o0)
        self.assertIs(outputs[0][1], o1)
        self.assertEqual(len(inputs), 1)
        self.assertIs(inputs[0], t)
        self.assertEqual(opts.timeout, timeout)

        # gather(list, Tensor, root, timeout); an empty output list (non-root)
        # becomes an empty list of lists.
        pg.gather([o0, o1], t, 0).wait()
        outputs, inputs, opts = self._pop_call(pg, "gather")
        self.assertIs(outputs[0][1], o1)
        self.assertIs(inputs[0], t)
        self.assertEqual(opts.rootRank, 0)
        pg.gather([], t, root=1).wait()
        outputs, inputs, opts = self._pop_call(pg, "gather")
        self.assertEqual(outputs, [])
        self.assertEqual(opts.rootRank, 1)

        # scatter(Tensor, list, root, timeout)
        pg.scatter(t, [i0, i1], 0).wait()
        outputs, inputs, opts = self._pop_call(pg, "scatter")
        self.assertIs(outputs[0], t)
        self.assertIs(inputs[0][1], i1)
        self.assertEqual(opts.rootRank, 0)
        pg.scatter(t, [], root=1).wait()
        outputs, inputs, opts = self._pop_call(pg, "scatter")
        self.assertEqual(inputs, [])
        self.assertEqual(opts.rootRank, 1)

        # reduce_scatter(Tensor, list, op, timeout)
        pg.reduce_scatter(t, [i0, i1], op=dist.ReduceOp.MAX).wait()
        outputs, inputs, opts = self._pop_call(pg, "reduce_scatter")
        self.assertIs(outputs[0], t)
        self.assertIs(inputs[0][0], i0)
        self.assertIs(inputs[0][1], i1)
        self.assertEqual(opts.reduceOp.op, dist.ReduceOp.MAX)

        # all_to_all_single(output, input, output_splits, input_splits, timeout)
        pg.all_to_all_single(o0, t, [1, 1], [1, 1], timeout).wait()
        output, input, output_splits, input_splits, opts = self._pop_call(
            pg, "all_to_all_single"
        )
        self.assertIs(output, o0)
        self.assertIs(input, t)
        self.assertEqual(output_splits, [1, 1])
        self.assertEqual(input_splits, [1, 1])
        self.assertIsInstance(opts, dist.AllToAllOptions)
        self.assertEqual(opts.timeout, timeout)

        # barrier(timeout) and barrier()
        pg.barrier(timeout=timeout).wait()
        (opts,) = self._pop_call(pg, "barrier")
        self.assertEqual(opts.timeout, timeout)
        pg.barrier().wait()
        (opts,) = self._pop_call(pg, "barrier")
        self.assertEqual(opts.timeout, unset)

    def test_canonical_call_passes_through(self):
        pg = self._init_pg(CanonicalProcessGroup)
        tensors = [torch.ones(2)]
        opts = dist.AllreduceOptions()
        pg.allreduce(tensors, opts).wait()
        self.assertEqual(self._pop_call(pg, "allreduce"), (tensors, opts))
        pg.allreduce(tensors, opts=opts).wait()
        self.assertEqual(self._pop_call(pg, "allreduce"), (tensors, opts))
        barrier_opts = dist.BarrierOptions()
        pg.barrier(barrier_opts).wait()
        self.assertEqual(self._pop_call(pg, "barrier"), (barrier_opts,))

    def test_canonical_override_dist_callers(self):
        pg = self._init_pg(CanonicalProcessGroup)
        t = torch.ones(2)

        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        tensors, opts = self._pop_call(pg, "allreduce")
        self.assertIs(tensors[0], t)
        self.assertEqual(opts.reduceOp.op, dist.ReduceOp.MAX)

        dist.broadcast(t, src=1)
        tensors, opts = self._pop_call(pg, "broadcast")
        self.assertEqual(opts.rootRank, 1)

        dist.reduce(t, dst=1)
        self._pop_call(pg, "reduce")

        dist.all_gather([torch.zeros(2), torch.zeros(2)], t)
        self._pop_call(pg, "allgather")

        dist.gather(t, [torch.zeros(2), torch.zeros(2)], dst=0)
        self._pop_call(pg, "gather")

        dist.scatter(t, [torch.zeros(2), torch.zeros(2)], src=0)
        self._pop_call(pg, "scatter")

        dist.reduce_scatter(t, [torch.zeros(2), torch.zeros(2)])
        self._pop_call(pg, "reduce_scatter")

        dist.all_to_all_single(torch.zeros(2), t)
        self._pop_call(pg, "all_to_all_single")

        dist.barrier()
        self._pop_call(pg, "barrier")

        # Functional collectives dispatch from C++ through the trampoline.
        funcol.wait_tensor(funcol.all_reduce(t, "avg", pg))
        tensors, opts = self._pop_call(pg, "allreduce")
        self.assertEqual(opts.reduceOp.op, dist.ReduceOp.AVG)

        funcol.wait_tensor(funcol.broadcast(t, 1, pg))
        tensors, opts = self._pop_call(pg, "broadcast")
        self.assertEqual(opts.rootRank, 1)

        funcol.wait_tensor(funcol.all_to_all_single(t, [1, 1], [1, 1], pg))
        call = self._pop_call(pg, "all_to_all_single")
        self.assertEqual(call[2:4], ([1, 1], [1, 1]))

    def test_varargs_override(self):
        pg = self._init_pg(VarargsProcessGroup)
        scalar = torch.tensor(1.0)

        pg.allreduce(scalar, op=dist.ReduceOp.MAX).wait()
        tensors, opts = self._pop_call(pg, "allreduce")
        self.assertIs(tensors[0], scalar)
        self.assertEqual(opts.reduceOp.op, dist.ReduceOp.MAX)

        dist.all_reduce(scalar, op=dist.ReduceOp.MIN)
        tensors, opts = self._pop_call(pg, "allreduce")
        self.assertEqual(opts.reduceOp.op, dist.ReduceOp.MIN)

        funcol.wait_tensor(funcol.all_reduce(scalar, "sum", pg))
        self._pop_call(pg, "allreduce")

        # Keyword arguments that match no pybind overload reach the override
        # unchanged.
        pg.barrier()
        tensors, opts = self._pop_call(pg, "broadcast")
        self.assertEqual(tensors, [torch.ones(1)])
        self.assertIsNone(opts)

    def test_alias_pair_overrides_rejected(self):
        for canonical, deprecated, falls_back in _ALIAS_PAIRS:
            with self.subTest(canonical=canonical, falls_back=falls_back):
                namespace = {
                    canonical: lambda self, *args: None,
                    deprecated: lambda self, *args: None,
                }
                with self.assertRaisesRegex(TypeError, f"Override only {canonical}"):
                    type("PG", (dist.ProcessGroup,), namespace)

    def test_deprecated_fallback_override(self):
        pg = self._init_pg(DeprecatedFallbackProcessGroup)
        output, input = torch.zeros(2), torch.ones(2)
        timeout = timedelta(seconds=5)

        # Convenience form of the deprecated name is normalized.
        pg.alltoall_base(output, input, [1, 1], [1, 1], timeout).wait()
        call = self._pop_call(pg, "alltoall_base")
        self.assertIs(call[0], output)
        self.assertEqual(call[4].timeout, timeout)

        # The canonical name, dist.* and functional collectives reach the
        # deprecated override through the trampoline's fallback.
        pg.all_to_all_single(output, input, [], [], timeout).wait()
        self._pop_call(pg, "alltoall_base")
        dist.all_to_all_single(output, input)
        self._pop_call(pg, "alltoall_base")
        funcol.wait_tensor(funcol.all_to_all_single(input, [1, 1], [1, 1], pg))
        self._pop_call(pg, "alltoall_base")

    def test_deprecated_non_fallback_override_warns(self):
        with self.assertWarnsRegex(
            FutureWarning, "only reached by direct calls to _reduce_scatter_base"
        ):

            class PG(FakeBackedProcessGroup):
                def _reduce_scatter_base(self, output, input, opts):
                    self.calls.append(("_reduce_scatter_base", output, input, opts))
                    return super()._reduce_scatter_base(output, input, opts)

        pg = self._init_pg(PG)
        output, input = torch.zeros(1), torch.ones(2)
        pg._reduce_scatter_base(output, input, dist.ReduceScatterOptions()).wait()
        self._pop_call(pg, "_reduce_scatter_base")

        dist.reduce_scatter_tensor(output, input)
        self.assertEqual(pg.calls, [])

    def test_normalized_methods_match_bindings(self):
        # Every ProcessGroup method with pybind convenience overloads must be
        # normalized for subclasses.
        overloaded = {
            name
            for name in dir(dist.ProcessGroup)
            if not name.startswith("__")
            and "Overloaded function"
            in (getattr(getattr(dist.ProcessGroup, name), "__doc__", None) or "")
        }
        self.assertEqual(overloaded, set(_NORMALIZED_METHODS))


class TestBatchSendRecv(MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @staticmethod
    def create_dummy(store, group_rank, group_size, timeout):
        return test_c10d_common.DummyProcessGroup(group_rank, group_size)

    @staticmethod
    def create_coalescing(store, group_rank, group_size, timeout):
        return CoalescingProcessGroup(group_rank, group_size)

    def test_batch_isend_irecv(self):
        # batch_isend_irecv over a Python ProcessGroup that does not advertise
        # coalescing falls back to per-op send/recv. DummyProcessGroup.send adds
        # 1 and recv adds 2 to verify the ops are dispatched into Python.
        dist.Backend.register_backend("dummy", TestBatchSendRecv.create_dummy)

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "6789"
        dist.init_process_group("dummy", rank=self.rank, world_size=self.world_size)

        peer = (self.rank + 1) % self.world_size
        send_tensor = torch.zeros(2, 2)
        recv_tensor = torch.zeros(2, 2)
        reqs = dist.batch_isend_irecv(
            [
                dist.P2POp(dist.isend, send_tensor, peer),
                dist.P2POp(dist.irecv, recv_tensor, peer),
            ]
        )
        for req in reqs:
            req.wait()

        self.assertEqual(send_tensor, torch.zeros(2, 2) + 1)
        self.assertEqual(recv_tensor, torch.zeros(2, 2) + 2)

        dist.barrier()
        dist.destroy_process_group()

    def test_batch_isend_irecv_coalescing(self):
        # A Python ProcessGroup that advertises supports_coalescing routes
        # batch_isend_irecv through the coalescing manager: start/endCoalescing
        # wrap the sends/recvs and the single end-coalescing work is returned.
        dist.Backend.register_backend("coalescing", TestBatchSendRecv.create_coalescing)

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "6789"
        dist.init_process_group(
            "coalescing", rank=self.rank, world_size=self.world_size
        )

        pg = _get_default_group()
        peer = (self.rank + 1) % self.world_size
        send_tensor = torch.zeros(2, 2)
        recv_tensor = torch.zeros(2, 2)
        works = dist.batch_isend_irecv(
            [
                dist.P2POp(dist.isend, send_tensor, peer),
                dist.P2POp(dist.irecv, recv_tensor, peer),
            ]
        )
        for work in works:
            work.wait()

        self.assertEqual(pg.start_coalescing_count, 1)
        self.assertEqual(pg.end_coalescing_count, 1)
        self.assertEqual(len(works), 1)
        self.assertEqual(send_tensor, torch.zeros(2, 2) + 1)
        self.assertEqual(recv_tensor, torch.zeros(2, 2) + 2)

        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()
