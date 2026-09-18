# Owner(s): ["oncall: distributed"]

import gc
import inspect
import os
import weakref
from datetime import timedelta

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import (
    Backend as C10DBackend,
    ErrorType,
    ReconfigureOptions,
)
from torch.distributed.distributed_c10d import (
    _coalescing_manager,
    _get_default_group,
    _time_estimator,
    _world,
)
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


_CONFIG_COLLECTIVES = {
    "broadcast": "broadcast",
    "all_reduce": "allreduce",
    "all_reduce_coalesced": "allreduce_coalesced",
    "reduce": "reduce",
    "all_gather": "allgather",
    "all_gather_single": "all_gather_single",
    "all_gather_into_tensor": "all_gather_single",
    "_all_gather_base": "all_gather_single",
    "all_gather_coalesced": "allgather_coalesced",
    "gather_single": "gather_single",
    "gather_into_tensor": "gather_single",
    "reduce_scatter": "reduce_scatter",
    "reduce_scatter_single": "reduce_scatter_single",
    "reduce_scatter_tensor": "reduce_scatter_single",
    "_reduce_scatter_base": "reduce_scatter_single",
    "all_to_all_single": "all_to_all_single",
}


def _collective_inputs(name):
    tensor = torch.ones(2)
    output = torch.empty_like(tensor)
    return {
        "broadcast": ((tensor,), {"group_src": 0}),
        "allreduce": ((tensor,), {"op": dist.ReduceOp.SUM}),
        "allreduce_coalesced": (([tensor],), {"op": dist.ReduceOp.SUM}),
        "reduce": ((tensor,), {"group_dst": 0, "op": dist.ReduceOp.SUM}),
        "allgather": (([output], tensor), {}),
        "all_gather_single": ((output, tensor), {}),
        "allgather_coalesced": (([[output]], [tensor]), {}),
        "gather_single": ((tensor, output), {"group_dst": 0}),
        "reduce_scatter": ((output, [tensor]), {"op": dist.ReduceOp.SUM}),
        "reduce_scatter_single": ((output, tensor), {"op": dist.ReduceOp.SUM}),
        "all_to_all_single": ((output, tensor), {}),
    }[_CONFIG_COLLECTIVES[name]]


class RecordingWork(dist._Work):
    def __init__(self, result, backend):
        super().__init__()
        self.result_ = result
        self.future_ = torch.futures.Future()
        self.future_.set_result(result)
        self.backend_ = weakref.ref(backend)

    def wait(self, timeout=timedelta(0)):
        self.backend_().wait_count += 1
        return True

    def get_future(self):
        self.backend_().get_future_count += 1
        return self.future_


class RecordingBackend(C10DBackend):
    def __init__(self, rank, world, name="python-backend"):
        super().__init__(rank, world)
        self._name = name
        self._options = C10DBackend.Options(name, timeout=timedelta(seconds=5))
        self.wait_count = 0
        self.get_future_count = 0
        self.calls = []
        self._work = []
        self.reconfigure_opts = None
        self.start_coalescing_count = 0
        self.end_coalescing_count = 0
        self.time_estimate_started = False
        self.sequence_number = 0
        self.registered_hook = None
        self.wait_for_pending_works_count = 0
        self.collectives_timing_enabled = False
        self.eager_device = None
        self.tensor_alloc_device_idx = None
        self.allocate_args = None
        self.aborted = False
        self.shut_down = False
        self.suspended = False
        self.resumed = False

    def _new_work(self, result=None):
        work = RecordingWork(result, self)
        self._work.append(work)
        return work

    @property
    def supports_splitting(self):
        return True

    @property
    def supports_coalescing(self):
        return True

    @property
    def _supports_time_estimate(self):
        return True

    @property
    def supports_shrinking(self):
        return True

    @property
    def supports_reconfigure(self):
        return True

    @property
    def supports_window(self):
        return True

    @property
    def options(self):
        return self._options

    def getBackendName(self):
        return self._name

    def set_timeout(self, timeout):
        self.calls.append(("set_timeout", timeout))

    def _start_time_estimate(self):
        self.time_estimate_started = True

    def _end_time_estimate(self):
        self.time_estimate_started = False
        return 1.25

    def shrink(self, ranks_to_exclude, shrink_flags=0, opts_override=None):
        self.calls.append(("shrink", ranks_to_exclude, shrink_flags, opts_override))
        return RecordingBackend(self.rank(), self.size(), "shrunk-python-backend")

    def get_reconfigure_handle(self):
        return "handle-for-python-backend"

    def reconfigure(self, opts):
        self.reconfigure_opts = opts
        return self._new_work(None)

    def start_coalescing(self):
        self.start_coalescing_count += 1

    def end_coalescing(self):
        self.end_coalescing_count += 1
        return self._new_work([])

    def broadcast(self, tensor_list, opts):
        self.calls.append(("broadcast", opts))
        for tensor in tensor_list:
            tensor.add_(1)
        return self._new_work(tensor_list)

    def allreduce(self, tensor_list, opts):
        self.calls.append(("allreduce", opts))
        for tensor in tensor_list:
            tensor.add_(2)
        return self._new_work(tensor_list)

    def allreduce_sparse(self, tensor_list, opts):
        self.calls.append(("allreduce_sparse", opts))
        return self._new_work(tensor_list)

    def allreduce_coalesced(self, tensor_list, opts):
        self.calls.append(("allreduce_coalesced", opts))
        for tensor in tensor_list:
            tensor.add_(3)
        return self._new_work(tensor_list)

    def reduce(self, tensor_list, opts):
        self.calls.append(("reduce", opts))
        for tensor in tensor_list:
            tensor.add_(4)
        return self._new_work(tensor_list)

    def allgather(self, output_tensors, input_tensors, opts):
        self.calls.append(("allgather", opts))
        for output_tensor_list, input_tensor in zip(output_tensors, input_tensors):
            for output_tensor in output_tensor_list:
                output_tensor.copy_(input_tensor)
        return self._new_work(output_tensors)

    def all_gather_single(self, output_tensor, input_tensor, opts):
        self.calls.append(("all_gather_single", opts))
        output_tensor.copy_(input_tensor)
        return self._new_work(output_tensor)

    def all_gather_single_coalesced(self, outputs, inputs, opts):
        self.calls.append(("all_gather_single_coalesced", opts))
        for output, input in zip(outputs, inputs):
            output.copy_(input)
        return self._new_work(outputs)

    def allgather_coalesced(self, output_lists, input_list, opts):
        self.calls.append(("allgather_coalesced", opts))
        for output_list, input in zip(output_lists, input_list):
            for output in output_list:
                output.copy_(input)
        return self._new_work(output_lists)

    def gather(self, output_tensors, input_tensors, opts):
        self.calls.append(("gather", opts))
        if output_tensors:
            for output, input in zip(output_tensors[0], input_tensors):
                output.copy_(input)
        return self._new_work(output_tensors)

    def gather_single(self, output_tensor, input_tensor, opts):
        self.calls.append(("gather_single", opts))
        if self.rank() == opts.rootRank:
            output_tensor.copy_(input_tensor)
        return self._new_work(output_tensor)

    def scatter(self, output_tensors, input_tensors, opts):
        self.calls.append(("scatter", opts))
        if input_tensors:
            for output, input in zip(output_tensors, input_tensors[0]):
                output.copy_(input)
        return self._new_work(output_tensors)

    def reduce_scatter(self, output_tensors, input_tensors, opts):
        self.calls.append(("reduce_scatter", opts))
        for output, input_list in zip(output_tensors, input_tensors):
            output.copy_(input_list[self.rank()])
        return self._new_work(output_tensors)

    def reduce_scatter_single(self, output_tensor, input_tensor, opts):
        self.calls.append(("reduce_scatter_single", opts))
        output_tensor.copy_(input_tensor.narrow(0, 0, output_tensor.numel()))
        return self._new_work(output_tensor)

    def reduce_scatter_single_coalesced(self, outputs, inputs, opts):
        self.calls.append(("reduce_scatter_single_coalesced", opts))
        for output, input in zip(outputs, inputs):
            output.copy_(input.narrow(0, 0, output.numel()))
        return self._new_work(outputs)

    def all_to_all_single(
        self, output_tensor, input_tensor, output_split_sizes, input_split_sizes, opts
    ):
        self.calls.append(
            ("all_to_all_single", output_split_sizes, input_split_sizes, opts)
        )
        output_tensor.copy_(input_tensor)
        return self._new_work(output_tensor)

    def alltoall(self, output_tensors, input_tensors, opts):
        self.calls.append(("alltoall", opts))
        for output, input in zip(output_tensors, input_tensors):
            output.copy_(input)
        return self._new_work(output_tensors)

    def send(self, tensor_list, dst, tag=0):
        self.calls.append(("send", dst, tag))
        for tensor in tensor_list:
            tensor.add_(1)
        return self._new_work(tensor_list)

    def recv(self, tensor_list, src, tag=0):
        self.calls.append(("recv", src, tag))
        for tensor in tensor_list:
            tensor.add_(2)
        return self._new_work(tensor_list)

    def recv_anysource(self, tensor_list, tag=0):
        self.calls.append(("recv_anysource", tag))
        for tensor in tensor_list:
            tensor.add_(5)
        return self._new_work(tensor_list)

    def barrier(self, opts):
        self.calls.append(("barrier", opts))
        return self._new_work(None)

    def monitored_barrier(self, opts, wait_all_ranks=False):
        self.calls.append(("monitored_barrier", opts, wait_all_ranks))

    def _set_sequence_number_for_group(self):
        self.sequence_number = 123

    def _get_sequence_number_for_group(self):
        return self.sequence_number

    def _register_on_completion_hook(self, hook):
        self.registered_hook = hook

    def _wait_for_pending_works(self):
        self.wait_for_pending_works_count += 1

    def _enable_collectives_timing(self):
        self.collectives_timing_enabled = True

    def split(self, store, ranks, opts):
        self.calls.append(("split", store, ranks, opts))
        return RecordingBackend(ranks.index(self.rank()), len(ranks), "split-backend")

    def merge(self, store, opts, rank, size):
        self.calls.append(("merge", store, opts, rank, size))
        return RecordingBackend(rank, size, "merged-backend")

    def eager_connect_single_device(self, device):
        self.eager_device = device

    def get_error(self):
        return ErrorType.SUCCESS

    def supports_tensor_alloc(self, device_idx):
        self.tensor_alloc_device_idx = device_idx
        return True

    def allocate_tensor(self, size, dtype, device):
        self.allocate_args = (size, dtype, device)
        return torch.empty(size, dtype=dtype, device=device)

    def suspend(self):
        self.suspended = True

    def resume(self):
        self.resumed = True

    def memory_stats(self):
        return {"allocated": 7}

    def abort(self):
        self.aborted = True

    def shutdown(self):
        self.shut_down = True


def create_process_group(backend):
    group = dist.ProcessGroup(dist.HashStore(), backend.rank(), backend.size())
    group._register_backend(
        torch.device("cpu"), dist.ProcessGroup.BackendType.CUSTOM, backend
    )
    group._set_default_backend(dist.ProcessGroup.BackendType.CUSTOM)
    group._set_group_name("pybackend-test")
    group._set_group_desc("python backend test")
    return group


@instantiate_parametrized_tests
class TestPyBackend(TestCase):
    @parametrize("name", _CONFIG_COLLECTIVES)
    @parametrize("async_op", [False, True])
    @parametrize("config_kind", ["omitted", "none", "object"])
    def test_collective_config(self, name, async_op, config_kind) -> None:
        backend = RecordingBackend(0, 1, "nccl2")
        # PyBackend has no gather_single trampoline; test Python forwarding directly.
        group = (
            backend
            if _CONFIG_COLLECTIVES[name] == "gather_single"
            else create_process_group(backend)
        )
        config = object() if config_kind == "object" else None
        args, kwargs = _collective_inputs(name)
        if config_kind != "omitted":
            kwargs["config"] = config
        result = getattr(dist, name)(*args, group=group, async_op=async_op, **kwargs)
        self.assertEqual(
            [call[0] for call in backend.calls], [_CONFIG_COLLECTIVES[name]]
        )
        opts = backend.calls[0][-1]
        self.assertIs(opts.config, config)
        self.assertEqual(opts.asyncOp, async_op)
        if "op" in kwargs:
            self.assertEqual(opts.reduceOp, kwargs["op"])
        if "group_src" in kwargs or "group_dst" in kwargs:
            self.assertEqual(opts.rootRank, 0)
        self.assertEqual(backend.wait_count, int(not async_op))
        if not async_op:
            self.assertIsNone(result)
            return
        if name in ("all_reduce_coalesced", "all_gather_coalesced"):
            self.assertIsInstance(result, torch.Future)
            self.assertEqual(backend.get_future_count, 1)
        else:
            self.assertIsInstance(result, dist.Work)
        result.wait()

    @parametrize("name", _CONFIG_COLLECTIVES)
    def test_collective_config_unsupported_backend(self, name) -> None:
        backend = RecordingBackend(0, 1)
        group = create_process_group(backend)
        args, kwargs = _collective_inputs(name)
        with self.assertRaisesRegex(
            RuntimeError, "only supported by the nccl2 backend"
        ):
            getattr(dist, name)(*args, group=group, config=object(), **kwargs)
        self.assertEqual(backend.calls, [])

    def test_collective_config_selected_backend(self) -> None:
        cpu_backend = RecordingBackend(0, 1)
        cuda_backend = RecordingBackend(0, 1, "nccl2")
        group = create_process_group(cpu_backend)
        group._register_backend(
            torch.device("cuda"), dist.ProcessGroup.BackendType.NCCL, cuda_backend
        )
        group._set_default_backend(dist.ProcessGroup.BackendType.NCCL)
        with self.assertRaisesRegex(
            RuntimeError, "only supported by the nccl2 backend"
        ):
            dist.all_reduce(torch.ones(2), group=group, config=object())
        self.assertEqual(cpu_backend.calls, [])
        self.assertEqual(cuda_backend.calls, [])

    def test_collective_config_compile_fallback(self) -> None:
        backend = RecordingBackend(0, 1, "nccl2")
        group = create_process_group(backend)
        config = object()

        def fn(tensor):
            output = tensor.clone()
            dist.all_reduce(output, group=group, config=config)
            return output + 1

        result = torch.compile(fn, backend="eager")(torch.zeros(2))
        self.assertEqual(result, torch.full((2,), 3.0))
        self.assertEqual([call[0] for call in backend.calls], ["allreduce"])
        self.assertIs(backend.calls[0][-1].config, config)

    @parametrize("name", _CONFIG_COLLECTIVES)
    @parametrize("config_kind", ["omitted", "none", "object"])
    def test_collective_config_torch_function(self, name, config_kind) -> None:
        config = object() if config_kind == "object" else None
        calls = []

        class Mode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                calls.append(kwargs)
                return None

        args, kwargs = _collective_inputs(name)
        if config_kind != "omitted":
            kwargs["config"] = config
        with Mode():
            getattr(dist, name)(*args, **kwargs)
        self.assertEqual(len(calls), 1)
        if config is None:
            self.assertNotIn("config", calls[0])
        else:
            self.assertIs(calls[0]["config"], config)

    @parametrize("name", _CONFIG_COLLECTIVES)
    def test_collective_config_keyword_only(self, name) -> None:
        param = inspect.signature(getattr(dist, name)).parameters["config"]
        self.assertEqual(param.kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIsNone(param.default)

    @parametrize(
        "options",
        [
            "BroadcastOptions",
            "AllreduceOptions",
            "AllreduceCoalescedOptions",
            "ReduceOptions",
            "AllgatherOptions",
            "GatherOptions",
            "ReduceScatterOptions",
            "AllToAllOptions",
        ],
    )
    def test_collective_config_lifetime(self, options) -> None:
        class Config:
            pass

        opts = getattr(torch._C._distributed_c10d, options)()
        self.assertIsNone(opts.config)
        config = Config()
        ref = weakref.ref(config)
        opts.config = config
        self.assertIs(opts.config, config)
        del config
        gc.collect()
        self.assertIsNotNone(ref())
        opts.config = None
        gc.collect()
        self.assertIsNone(opts.config)
        self.assertIsNone(ref())

    @parametrize("name", ["all_reduce", "all_gather_single", "reduce_scatter_single"])
    @parametrize("device", [None, torch.device("cpu")])
    @parametrize("pending", [False, True])
    def test_collective_config_coalescing(self, name, device, pending) -> None:
        backend = RecordingBackend(0, 1, "nccl2")
        group = create_process_group(backend)
        args, kwargs = _collective_inputs(name)
        collective = getattr(dist, name)
        with self.assertRaisesRegex(
            NotImplementedError, "configuration is not supported with coalescing"
        ):
            with _coalescing_manager(group, device):
                if pending:
                    collective(*args, group=group, **kwargs)
                collective(*args, group=group, config=object(), **kwargs)
        self.assertNotIn(group, _world.pg_coalesce_state)
        self.assertEqual(backend.calls, [])
        self.assertEqual(backend.start_coalescing_count, int(device is not None))
        self.assertEqual(backend.end_coalescing_count, int(device is not None))
        collective(*args, group=group, **kwargs)
        self.assertEqual(
            [call[0] for call in backend.calls], [_CONFIG_COLLECTIVES[name]]
        )

    def test_collective_config_time_estimator_cleanup(self) -> None:
        backend = RecordingBackend(0, 1, "nccl2")
        group = create_process_group(backend)
        tensor = torch.zeros(2)
        with self.assertRaisesRegex(NotImplementedError, "during time estimation"):
            with _time_estimator(group, torch.device("cpu")):
                self.assertTrue(backend.time_estimate_started)
                dist.all_reduce(tensor, group=group, config=object())
        self.assertFalse(backend.time_estimate_started)
        self.assertEqual(backend.calls, [])
        dist.all_reduce(tensor, group=group, config=object())
        self.assertEqual(tensor, torch.full((2,), 2.0))
        self.assertEqual([call[0] for call in backend.calls], ["allreduce"])

    def test_attr_overrides(self) -> None:
        backend = RecordingBackend(0, 1)
        group = create_process_group(backend)

        self.assertEqual(group.rank(), 0)
        self.assertEqual(group.size(), 1)

        for attr in (
            "supports_splitting",
            "supports_coalescing",
            "_supports_time_estimate",
            "supports_shrinking",
            "supports_reconfigure",
            "supports_window",
        ):
            self.assertTrue(getattr(backend, attr))

        self.assertTrue(group.supports_reconfigure)
        self.assertTrue(group.supports_window)
        self.assertEqual(
            group._get_backend(torch.device("cpu")).options.backend, "python-backend"
        )

        group._set_group_name("name")
        group._set_group_desc("desc")
        self.assertEqual(group.group_name, "name")
        self.assertEqual(group.group_desc, "desc")

        group.bound_device_id = torch.device("cpu:0")
        self.assertEqual(group.bound_device_id, torch.device("cpu:0"))

        group.use_pg_for_symm_mem_rendezvous = True
        self.assertTrue(group.use_pg_for_symm_mem_rendezvous)

    def test_collective_overrides(self) -> None:
        backend = RecordingBackend(0, 1)
        group = create_process_group(backend)

        allreduce_tensor = torch.zeros(2)
        work = group.allreduce([allreduce_tensor])
        self.assertTrue(work.wait())
        self.assertEqual(allreduce_tensor, torch.full((2,), 2.0))

        broadcast_tensor = torch.zeros(2)
        group.broadcast([broadcast_tensor]).wait()
        self.assertEqual(broadcast_tensor, torch.ones(2))

        output = torch.zeros(2)
        input = torch.ones(2)
        group.all_gather_single(output, input).wait()
        self.assertEqual(output, input)
        self.assertEqual(backend.calls[-1][0], "all_gather_single")

        outputs = [torch.zeros(2), torch.zeros(2)]
        inputs = [torch.ones(2), torch.full((2,), 3.0)]
        group.all_gather_single_coalesced(outputs, inputs).wait()
        self.assertEqual(outputs, inputs)
        self.assertEqual(backend.calls[-1][0], "all_gather_single_coalesced")

        rs_output = torch.zeros(2)
        rs_input = torch.arange(2.0)
        group.reduce_scatter_single(rs_output, rs_input).wait()
        self.assertEqual(rs_output, rs_input)
        self.assertEqual(backend.calls[-1][0], "reduce_scatter_single")

        rs_outputs = [torch.zeros(2), torch.zeros(2)]
        rs_inputs = [torch.arange(4.0), torch.arange(2.0, 6.0)]
        group.reduce_scatter_single_coalesced(rs_outputs, rs_inputs).wait()
        self.assertEqual(rs_outputs, [torch.arange(2.0), torch.arange(2.0, 4.0)])
        self.assertEqual(backend.calls[-1][0], "reduce_scatter_single_coalesced")

        a2a_output = torch.zeros(2)
        a2a_input = torch.ones(2)
        group.all_to_all_single(a2a_output, a2a_input, [], []).wait()
        self.assertEqual(a2a_output, a2a_input)
        self.assertEqual(backend.calls[-1][0], "all_to_all_single")

        recv_any = torch.zeros(2)
        group.recv_anysource([recv_any], 0).wait()
        self.assertEqual(recv_any, torch.full((2,), 5.0))

        self.assertGreater(backend.wait_count, 0)

    def test_collective_returns_none(self) -> None:
        backend = RecordingBackend(0, 1)
        backend.allreduce = lambda tensors, opts: [t.add_(2) for t in tensors] and None
        backend.broadcast = lambda tensors, opts: [t.add_(1) for t in tensors] and None
        backend.barrier = lambda opts: None
        group = create_process_group(backend)

        t = torch.zeros(2)
        self.assertIsNone(group.allreduce([t]))
        self.assertEqual(t, torch.full((2,), 2.0))

        t = torch.zeros(2)
        self.assertIsNone(group.broadcast([t]))
        self.assertEqual(t, torch.ones(2))

        self.assertIsNone(group.barrier())

        backend.recv_anysource = (
            lambda tensors, tag: [t.add_(5) for t in tensors] and None
        )
        t = torch.zeros(2)
        self.assertIsNone(group.recv_anysource([t], 0))
        self.assertEqual(t, torch.full((2,), 5.0))

        backend.end_coalescing = lambda: None
        group._start_coalescing(torch.device("cpu"))
        self.assertIsNone(group._end_coalescing(torch.device("cpu")))

    def test_backend_only_overrides(self) -> None:
        backend = RecordingBackend(0, 1)
        group = create_process_group(backend)

        timeout = timedelta(seconds=3)
        group.set_timeout(timeout)
        self.assertEqual(backend.calls[-1], ("set_timeout", timeout))

        shrunk = group._get_backend(torch.device("cpu")).shrink([1], 7, None)
        self.assertEqual(shrunk.name(), "shrunk-python-backend")
        self.assertEqual(backend.calls[-1], ("shrink", [1], 7, None))

        self.assertEqual(group.get_reconfigure_handle(), "handle-for-python-backend")
        opts = ReconfigureOptions()
        group.reconfigure(opts).wait()
        self.assertIs(backend.reconfigure_opts, opts)

        group._start_coalescing(torch.device("cpu"))
        group._end_coalescing(torch.device("cpu")).wait()
        self.assertEqual(backend.start_coalescing_count, 1)
        self.assertEqual(backend.end_coalescing_count, 1)

        with dist._time_estimator(group, torch.device("cpu")) as estimator:
            self.assertTrue(backend.time_estimate_started)
        self.assertFalse(backend.time_estimate_started)
        self.assertEqual(estimator.estimated_time, 1.25)

        backend._set_sequence_number_for_group()
        self.assertEqual(backend._get_sequence_number_for_group(), 123)

        group.monitored_barrier(timeout, True)
        self.assertEqual(backend.calls[-1][0], "monitored_barrier")

        group._register_on_completion_hook(lambda work_info: None)
        self.assertTrue(group._has_hooks())
        self.assertIsNotNone(backend.registered_hook)

        group._wait_for_pending_works()
        group._enable_collectives_timing()
        self.assertEqual(backend.wait_for_pending_works_count, 1)
        self.assertTrue(backend.collectives_timing_enabled)

        store = dist.HashStore()
        split_opts = C10DBackend.Options("split")
        split = group.split_group([0], opts=split_opts)
        self.assertEqual(
            split._get_backend(torch.device("cpu")).name(), "split-backend"
        )

        merge = group.merge_remote_group(store, 1)
        self.assertEqual(
            merge._get_backend(torch.device("cpu")).name(), "merged-backend"
        )

        device = torch.device("cpu:0")
        backend.eager_connect_single_device(device)
        self.assertEqual(backend.eager_device, device)

        self.assertEqual(backend.get_error(), ErrorType.SUCCESS)
        self.assertTrue(backend.supports_tensor_alloc(device))
        self.assertEqual(backend.tensor_alloc_device_idx, device)

        allocated = backend.allocate_tensor(
            4,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        self.assertEqual(allocated.shape, (4,))
        self.assertEqual(allocated.dtype, torch.float64)
        self.assertEqual(backend.allocate_args, (4, torch.float64, torch.device("cpu")))

        backend.suspend()
        backend.resume()
        self.assertTrue(backend.suspended)
        self.assertTrue(backend.resumed)
        self.assertEqual(backend.memory_stats(), {"allocated": 7})

        group.abort()
        group.shutdown()
        self.assertTrue(backend.aborted)
        self.assertTrue(backend.shut_down)


class TestPyBackendProcessGroup(MultiProcessTestCase):
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
    def create_backend(store, group_rank, group_size, timeout):
        return RecordingBackend(group_rank, group_size, "registered-python-backend")

    def test_init_process_group_with_pybackend(self):
        dist.Backend.register_backend(
            "pybackend", TestPyBackendProcessGroup.create_backend, devices=["cpu"]
        )

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "6789"
        dist.init_process_group("pybackend", rank=self.rank, world_size=self.world_size)

        pg = _get_default_group()
        backend = pg._get_backend(torch.device("cpu"))
        self.assertIsInstance(backend, C10DBackend)
        self.assertNotIsInstance(backend, dist.ProcessGroup)
        self.assertEqual(backend.name(), "registered-python-backend")

        allreduce_tensor = torch.zeros(2)
        dist.all_reduce(allreduce_tensor)
        self.assertEqual(allreduce_tensor, torch.full((2,), 2.0))

        sync_tensor = torch.zeros(2)
        dist.all_reduce(sync_tensor, async_op=False)
        self.assertEqual(sync_tensor, torch.full((2,), 2.0))

        async_tensor = torch.zeros(2)
        work = dist.all_reduce(async_tensor, async_op=True)
        self.assertIsNotNone(work)
        work.wait()
        self.assertEqual(async_tensor, torch.full((2,), 2.0))

        future_tensor = torch.ones(2)
        work = dist.all_reduce(future_tensor, async_op=True)
        fut = work.get_future()
        fut.wait()
        self.assertEqual(future_tensor, torch.full((2,), 3.0))

        async_tensors = [torch.zeros(2) for _ in range(4)]
        works = [dist.all_reduce(t, async_op=True) for t in async_tensors]
        for w in works:
            w.wait()
        for t in async_tensors:
            self.assertEqual(t, torch.full((2,), 2.0))

        peer = (self.rank + 1) % self.world_size
        send_tensor = torch.zeros(2)
        recv_tensor = torch.zeros(2)
        works = dist.batch_isend_irecv(
            [
                dist.P2POp(dist.isend, send_tensor, peer),
                dist.P2POp(dist.irecv, recv_tensor, peer),
            ]
        )
        for work in works:
            work.wait()

        self.assertEqual(backend.start_coalescing_count, 1)
        self.assertEqual(backend.end_coalescing_count, 1)
        self.assertEqual(send_tensor, torch.ones(2))
        self.assertEqual(recv_tensor, torch.full((2,), 2.0))

        dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()
