# Owner(s): ["oncall: distributed"]

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
from torch.distributed.distributed_c10d import _get_default_group
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import run_tests, TestCase


class RecordingWork(dist._Work):
    def __init__(self, result, backend):
        super().__init__()
        self.result_ = result
        self.future_ = torch.futures.Future()
        self.future_.set_result(result)
        self.backend_ = weakref.ref(backend)

    def wait(self, timeout):
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
    def supports_time_estimate(self):
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


class TestPyBackend(TestCase):
    def test_attr_overrides(self) -> None:
        backend = RecordingBackend(0, 1)
        group = create_process_group(backend)

        self.assertEqual(group.rank(), 0)
        self.assertEqual(group.size(), 1)

        for attr in (
            "supports_splitting",
            "supports_coalescing",
            "supports_time_estimate",
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
