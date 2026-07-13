# Owner(s): ["oncall: distributed"]

import os
import sys

import torch
import torch.cuda
import torch.cuda.nccl as nccl
import torch.distributed as c10d
import torch.distributed._symmetric_memory as symm_mem
from torch.testing._internal.common_cuda import TEST_CUDA, TEST_MULTIGPU
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    requires_nccl_version,
    skip_if_lt_x_gpu,
    TEST_SKIPS,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_WINDOWS,
    load_tests,
    NoTest,
    parametrize,
    requires_cuda_p2p_access,
    run_tests,
    skip_but_pass_in_sandcastle_if,
    TEST_WITH_ROCM,
    TestCase,
)


# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests  # noqa: PLW0127

nGPUs = torch.cuda.device_count()
if not TEST_CUDA:
    print("CUDA not available, skipping tests", file=sys.stderr)
    TestCase = NoTest


datatypes = [torch.float]
if (
    TEST_CUDA and c10d.is_nccl_available() and nccl.version() >= (2, 10)
) or TEST_WITH_ROCM:
    datatypes.append(torch.bfloat16)

# Broadcast (and alltoall) support float8, while reduce and allreduce do not support float8 currently
broadcast_dtypes = (
    datatypes + [torch.float8_e4m3fnuz, torch.float8_e5m2fnuz]
    if TEST_WITH_ROCM
    else [torch.float8_e4m3fn, torch.float8_e5m2]
)


class TestNCCL(TestCase):
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    def test_unique_id(self, device):
        uid = nccl.unique_id()
        self.assertIsInstance(uid, bytes)
        self.assertGreater(len(uid), 1)

    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "only one GPU detected")
    @dtypes(*broadcast_dtypes)
    def test_broadcast(self, device, dtype):
        expected = torch.zeros(128).uniform_().to(dtype=dtype)
        tensors = [expected.cuda()]
        for device in range(1, torch.cuda.device_count()):
            tensors.append(torch.zeros(128, dtype=dtype, device=device))

        nccl.broadcast(tensors)
        for i in range(torch.cuda.device_count()):
            self.assertEqual(tensors[i], expected)

        # Test with tuple
        tensors = [expected.cuda()]
        for device in range(1, torch.cuda.device_count()):
            tensors.append(torch.zeros(128, dtype=dtype, device=device))

        nccl.broadcast(tuple(tensors))
        for i in range(torch.cuda.device_count()):
            self.assertEqual(tensors[i], expected)

        # Test with a non-zero root (regression test for #179908)
        root = nGPUs - 1
        expected = torch.zeros(128).uniform_().to(dtype=dtype)
        tensors = [
            expected.cuda(device)
            if device == root
            else torch.zeros(128, dtype=dtype, device=device)
            for device in range(nGPUs)
        ]

        nccl.broadcast(tensors, root=root)
        for i in range(nGPUs):
            self.assertEqual(tensors[i], expected)

    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "only one GPU detected")
    @dtypes(*datatypes)
    def test_reduce(self, device, dtype):
        cpu_tensors = [
            torch.zeros(128).uniform_().to(dtype=dtype) for i in range(nGPUs)
        ]
        expected = torch.zeros(128, dtype=dtype)
        for t in cpu_tensors:
            expected.add_(t)

        tensors = [cpu_tensors[i].cuda(i) for i in range(nGPUs)]
        nccl.reduce(tensors)

        self.assertEqual(tensors[0], expected)

        # Test with tuple
        tensors = [cpu_tensors[i].cuda(i) for i in range(nGPUs)]
        nccl.reduce(tuple(tensors))

        self.assertEqual(tensors[0], expected)

    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "only one GPU detected")
    @dtypes(*datatypes)
    def test_all_reduce(self, device, dtype):
        cpu_tensors = [
            torch.zeros(128).uniform_().to(dtype=dtype) for i in range(nGPUs)
        ]
        expected = torch.zeros(128, dtype=dtype)
        for t in cpu_tensors:
            expected.add_(t)

        tensors = [cpu_tensors[i].cuda(i) for i in range(nGPUs)]
        nccl.all_reduce(tensors)

        for tensor in tensors:
            self.assertEqual(tensor, expected)

        # Test with tuple.
        tensors = tuple(cpu_tensors[i].cuda(i) for i in range(nGPUs))
        nccl.all_reduce(tensors)

        for tensor in tensors:
            self.assertEqual(tensor, expected)

        # Test with set.
        tensors = {cpu_tensors[i].cuda(i) for i in range(nGPUs)}
        nccl.all_reduce(tensors)

        for tensor in tensors:
            self.assertEqual(tensor, expected)

    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    def test_collective_errors(self, device):
        t = torch.rand(10).cuda(0)
        with self.assertRaisesRegex(
            TypeError, "Inputs should be a collection of tensors"
        ):
            nccl.all_reduce(t)

        with self.assertRaisesRegex(
            TypeError, "Inputs should be a collection of tensors"
        ):
            nccl.reduce(t)

        with self.assertRaisesRegex(
            TypeError, "Inputs should be a collection of tensors"
        ):
            nccl.broadcast(t)

        with self.assertRaisesRegex(
            TypeError, "Inputs should be a collection of tensors"
        ):
            nccl.all_gather(t, t)

        with self.assertRaisesRegex(
            TypeError, "Inputs should be a collection of tensors"
        ):
            nccl.reduce_scatter(t, t)

    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "only one GPU detected")
    @dtypes(*datatypes)
    def test_all_gather(self, device, dtype):
        cpu_inputs = [torch.zeros(128).uniform_().to(dtype=dtype) for i in range(nGPUs)]
        expected = torch.cat(cpu_inputs, 0)

        inputs = [cpu_inputs[i].cuda(i) for i in range(nGPUs)]
        outputs = [
            torch.zeros(128 * nGPUs, device=i, dtype=dtype) for i in range(nGPUs)
        ]
        nccl.all_gather(inputs, outputs)

        for tensor in outputs:
            self.assertEqual(tensor, expected)

        # Test with tuple.
        inputs = [cpu_inputs[i].cuda(i) for i in range(nGPUs)]
        outputs = [
            torch.zeros(128 * nGPUs, device=i, dtype=dtype) for i in range(nGPUs)
        ]
        nccl.all_gather(tuple(inputs), tuple(outputs))

        for tensor in outputs:
            self.assertEqual(tensor, expected)

    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "only one GPU detected")
    @dtypes(*datatypes)
    def test_reduce_scatter(self, device, dtype):
        in_size = 32 * nGPUs
        out_size = 32

        cpu_inputs = [
            torch.zeros(in_size).uniform_().to(dtype=dtype) for i in range(nGPUs)
        ]
        expected = torch.zeros(in_size, dtype=dtype)
        for t in cpu_inputs:
            expected.add_(t)
        expected = expected.view(nGPUs, 32)

        inputs = [cpu_inputs[i].cuda(i) for i in range(nGPUs)]
        outputs = [torch.zeros(out_size, device=i, dtype=dtype) for i in range(nGPUs)]
        nccl.reduce_scatter(inputs, outputs)

        for i in range(nGPUs):
            self.assertEqual(outputs[i], expected[i])

        # Test with tuple
        inputs = [cpu_inputs[i].cuda(i) for i in range(nGPUs)]
        outputs = [torch.zeros(out_size, device=i, dtype=dtype) for i in range(nGPUs)]
        nccl.reduce_scatter(tuple(inputs), tuple(outputs))

        for i in range(nGPUs):
            self.assertEqual(outputs[i], expected[i])


@instantiate_parametrized_tests
@requires_cuda_p2p_access()
class NCCLSymmetricMemoryTest(MultiProcContinuousTest):
    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    @classmethod
    def _init_pg(cls, rank, world_size, rdvz_file):
        # Eager NCCL communicator init via device_id, so symm_mem rendezvous
        # does not require a separate warm-up collective.
        if rdvz_file is None:
            raise AssertionError("Expected rdvz_file to not be None")
        os.environ["LOCAL_RANK"] = str(rank)
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
        store = c10d.FileStore(rdvz_file, world_size)
        c10d.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=rank,
            store=store,
            timeout=cls.timeout,
            device_id=device,
        )
        cls.pg = c10d.distributed_c10d._get_default_group()

    @skip_but_pass_in_sandcastle_if(TEST_WITH_ROCM, "Skip NCCL tests for ROCm")
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version((2, 27), "NCCL Symmetric Memory support from nccl 2.27")
    @skip_if_lt_x_gpu(2)
    def test_nccl_symmem_alloc(self):
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        # Need this all_reduce to initialize NCCL communicator. Otherwise, the
        # test will hang.  TODO: investigate how NCCLSymmetricMemory can
        # initialize NCCL communicator.
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        dtype = torch.float
        numel = 1024

        def foo():
            inp = symm_mem.empty(numel, dtype=dtype, device=self.device)
            symm_mem.rendezvous(inp, group=group_name)

        foo()

        out = symm_mem.empty(numel, dtype=dtype, device=self.device)
        symm_mem.rendezvous(out, group=group_name)

    @skip_but_pass_in_sandcastle_if(TEST_WITH_ROCM, "Skip NCCL tests for ROCm")
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version((2, 27), "NCCL Symmetric Memory support from nccl 2.27")
    @skip_if_lt_x_gpu(2)
    def test_nccl_symmem_rendezvous_many_allocations(self):
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        tensors = [
            symm_mem.empty(1, dtype=torch.float, device=self.device) for _ in range(256)
        ]

        # Rendezvous a subset twice so the repeated lookup path is covered
        # while many allocations are still live.
        sampled_tensors = tensors[::16]
        for tensor in sampled_tensors:
            handle = symm_mem.rendezvous(tensor, group=group_name)
            self.assertEqual(handle.rank, self.rank)
            self.assertEqual(handle.world_size, self.world_size)
        for tensor in sampled_tensors:
            symm_mem.rendezvous(tensor, group=group_name)

        result = torch.ops.symm_mem.one_shot_all_reduce(
            tensors[-1].fill_(self.rank), "sum", group_name
        )
        self.assertEqual(
            result, torch.full_like(result, (self.world_size - 1) * self.world_size / 2)
        )

    @skip_but_pass_in_sandcastle_if(TEST_WITH_ROCM, "Skip NCCL tests for ROCm")
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @skip_if_lt_x_gpu(2)
    def test_nccl_symmem_rendezvous_world(self):
        symm_mem.set_backend("NCCL")
        group_name = c10d.group.WORLD.group_name

        t = symm_mem.empty(64, device=self.device)
        handle = symm_mem.rendezvous(t, group=group_name)

        self.assertEqual(handle.world_size, self.world_size)
        self.assertEqual(handle.rank, self.rank)

        t.fill_(self.rank)
        c10d.barrier()

        peer_rank = (self.rank + 1) % self.world_size
        buf = handle.get_buffer(peer_rank, (64,), torch.float32)
        self.assertTrue(buf.eq(peer_rank).all())

    @skip_but_pass_in_sandcastle_if(TEST_WITH_ROCM, "Skip NCCL tests for ROCm")
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version((2, 27), "NCCL Symmetric Memory support from nccl 2.27")
    @skip_if_lt_x_gpu(2)
    def test_nccl_symmem_barrier(self):
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        numel = 64
        t = symm_mem.empty(numel, dtype=torch.float32, device=self.device).fill_(
            self.rank
        )
        handle = symm_mem.rendezvous(t, group=group_name)
        self.assertEqual(handle.rank, self.rank)
        self.assertEqual(handle.world_size, self.world_size)

        handle.barrier()
        for peer in range(self.world_size):
            buf = handle.get_buffer(peer, (numel,), torch.float32)
            self.assertTrue(buf.eq(peer).all())
        handle.barrier()

    @skip_but_pass_in_sandcastle_if(TEST_WITH_ROCM, "Skip NCCL tests for ROCm")
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @skip_if_lt_x_gpu(2)
    def test_nccl_symmem_rendezvous_subgroup(self):
        symm_mem.set_backend("NCCL")

        subgroup = c10d.new_group(list(range(self.world_size)))

        t = symm_mem.empty(64, device=self.device)
        handle = symm_mem.rendezvous(t, group=subgroup)

        self.assertEqual(handle.world_size, self.world_size)
        self.assertEqual(handle.rank, self.rank)

        t.fill_(self.rank)
        c10d.barrier(group=subgroup)

        peer_rank = (self.rank + 1) % self.world_size
        buf = handle.get_buffer(peer_rank, (64,), torch.float32)
        self.assertTrue(buf.eq(peer_rank).all())

    @skip_but_pass_in_sandcastle_if(TEST_WITH_ROCM, "Skip NCCL tests for ROCm")
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version(
        (2, 28), "NCCL Symmetric Memory support device API from nccl 2.28"
    )
    @skip_if_lt_x_gpu(2)
    def test_nccl_symmem_collective(self):
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        # Need this all_reduce to initialize NCCL communicator. Otherwise, the
        # test will hang.  TODO: investigate how NCCLSymmetricMemory can
        # initialize NCCL communicator.
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        dtype = torch.float
        numel = 1024

        out = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(self.rank)
        symm_mem.rendezvous(out, group=group_name)
        c10d.all_reduce(out)
        torch.cuda.synchronize()
        self.assertEqual(
            out, torch.full_like(out, (self.world_size - 1) * self.world_size / 2)
        )

        inp = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(self.rank)
        symm_mem.rendezvous(inp, group=group_name)
        res = torch.ops.symm_mem.one_shot_all_reduce(inp, "sum", group_name)
        self.assertEqual(out, res)

    @skip_but_pass_in_sandcastle_if(TEST_WITH_ROCM, "Skip NCCL tests for ROCm")
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version(
        (2, 28), "NCCL Symmetric Memory support device API from nccl 2.28"
    )
    @skip_if_lt_x_gpu(2)
    def test_nccl_symmem_collective_cuda_graph(self):
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        # Need this all_reduce to initialize NCCL communicator. Otherwise, the
        # test will hang.  TODO: investigate how NCCLSymmetricMemory can
        # initialize NCCL communicator.
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        dtype = torch.float
        numel = 1024

        out = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(self.rank)
        symm_mem.rendezvous(out, group=group_name)
        graph_all_reduce = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph_all_reduce):
            c10d.all_reduce(out)
        graph_all_reduce.replay()
        torch.cuda.synchronize()
        self.assertEqual(
            out, torch.full_like(out, (self.world_size - 1) * self.world_size / 2)
        )

        inp = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(self.rank)
        symm_mem.rendezvous(inp, group=group_name)
        graph_one_shot_all_reduce = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph_one_shot_all_reduce):
            res = torch.ops.symm_mem.one_shot_all_reduce(inp, "sum", group_name)
        graph_one_shot_all_reduce.replay()
        self.assertEqual(out, res)

        for repeat in range(3):
            offset = 13 + repeat
            inp.fill_(self.rank + offset)
            out.fill_(self.rank + offset)
            res.fill_(0.0)
            expected_sum = float(
                self.world_size * offset + self.world_size * (self.world_size - 1) / 2
            )
            graph_all_reduce.replay()
            graph_one_shot_all_reduce.replay()
            self.assertEqual(out, torch.full_like(out, expected_sum))
            self.assertEqual(res, out)

    @skip_but_pass_in_sandcastle_if(TEST_WITH_ROCM, "Skip NCCL tests for ROCm")
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version(
        (2, 28), "NCCL Symmetric Memory support device API from nccl 2.28"
    )
    @skip_if_lt_x_gpu(2)
    def test_nccl_symmem_put(self):
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        # Need this all_reduce to initialize NCCL communicator. Otherwise, the
        # test will hang.  TODO: investigate how NCCLSymmetricMemory can
        # initialize NCCL communicator.
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        dtype = torch.float
        numel = 1024
        tensor = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(self.rank)
        # This is needed to make sure we don't get blocked the second time we call rendezvous
        # for the same tensor because it will be cached by that moment.
        symm_mem.rendezvous(tensor, group=group_name)
        signal_val = 5
        c10d.barrier()

        if self.rank == 1:
            torch.ops.symm_mem.nccl_put_with_signal(tensor, signal_val, 0)
        elif self.rank == 0:
            torch.ops.symm_mem.nccl_wait_for_signal(tensor, signal_val)
            torch.testing.assert_close(
                tensor, torch.ones(numel, dtype=dtype, device=self.device)
            )
        c10d.barrier()
        if self.rank == 1:
            tensor *= 2
            torch.ops.symm_mem.nccl_put(tensor, 0)
            c10d.barrier()
        else:
            c10d.barrier()
        if self.rank == 0:
            torch.testing.assert_close(
                tensor, torch.ones(numel, dtype=dtype, device=self.device) * 2
            )

    @skip_but_pass_in_sandcastle_if(TEST_WITH_ROCM, "Skip NCCL tests for ROCm")
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version((2, 29), "NCCL one-sided host API support from nccl 2.29")
    @skip_if_lt_x_gpu(2)
    def test_nccl_symmem_handle_signal(self):
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        # need this all_reduce to initialize NCCL communicator. Otherwise, the
        # test will hang.  TODO: investigate how NCCLSymmetricMemory can
        # initialize NCCL communicator.
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        dtype = torch.float
        numel = 1024
        tensor = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(self.rank)
        handle = symm_mem.rendezvous(tensor, group=group_name)

        channel = 0
        world_size = handle.world_size

        c10d.barrier()

        # Pair up ranks: odd ranks send to even ranks
        # This allows the test to work with any number of GPUs
        if self.rank % 2 == 1:
            # Odd rank: send signal to previous even rank
            dst_rank = self.rank - 1
            handle.put_signal(dst_rank=dst_rank, channel=channel)
            torch.cuda.synchronize()
        elif self.rank % 2 == 0 and self.rank + 1 < world_size:
            # Even rank: wait for signal from next odd rank (if it exists)
            src_rank = self.rank + 1
            # wait_signal blocks until the signal arrives
            # If this completes without hanging, the test passes
            handle.wait_signal(src_rank=src_rank, channel=channel)
            torch.cuda.synchronize()

        c10d.barrier()

    @skip_but_pass_in_sandcastle_if(TEST_WITH_ROCM, "Skip NCCL tests for ROCm")
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @skip_if_lt_x_gpu(2)
    def test_nccl_symmem_get(self):
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        # Need this all_reduce to initialize NCCL communicator. Otherwise, the
        # test will hang.  TODO: investigate how NCCLSymmetricMemory can
        # initialize NCCL communicator.
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        dtype = torch.float
        numel = 1024
        tensor = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(self.rank)
        # This is needed to make sure we don't get blocked the second time we call rendezvous
        # for the same tensor because it will be cached by that moment.
        symm_mem.rendezvous(tensor, group=group_name)
        c10d.barrier()
        if self.rank == 0:
            torch.ops.symm_mem.nccl_get(tensor, 1)
            # TODO: remove after we have wait_signal
            c10d.barrier()
            torch.testing.assert_close(
                tensor, torch.ones(numel, dtype=dtype, device=self.device)
            )
        else:
            # handle.wait_signal(src_rank=0)
            # TODO: remove after we have wait_signal
            c10d.barrier()

    @skip_but_pass_in_sandcastle_if(TEST_WITH_ROCM, "Skip NCCL tests for ROCm")
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version(
        (2, 28), "NCCL Symmetric Memory device API support from nccl 2.28"
    )
    @skip_if_lt_x_gpu(2)
    def test_get(self):
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        dtype = torch.float
        numel = 1024

        # Full-buffer get from a peer's allocation.
        src = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(self.rank)
        hdl = symm_mem.rendezvous(src, group=group_name)
        c10d.barrier()

        if self.rank == 0:
            dst = torch.empty_like(src)
            symm_mem.get(dst, hdl, peer=1)
            torch.testing.assert_close(dst, torch.ones_like(dst))

        c10d.barrier()

        # Offset get: copy a sub-region of the peer's allocation.
        src_base = symm_mem.empty(2 * numel, dtype=dtype, device=self.device)
        src_base.copy_(
            torch.arange(2 * numel, dtype=dtype, device=self.device)
            + self.rank * 2 * numel
        )
        hdl = symm_mem.rendezvous(src_base, group=group_name)
        c10d.barrier()

        if self.rank == 0:
            offset = numel // 2
            dst = torch.empty(numel, dtype=dtype, device=self.device)
            symm_mem.get(dst, hdl, peer=1, offset=offset)
            expected = (
                torch.arange(offset, offset + numel, dtype=dtype, device=self.device)
                + 2 * numel
            )
            torch.testing.assert_close(dst, expected)

            # Filling a sub-region: pass a view; the rest of dst is untouched.
            larger_dst = torch.full((numel + 1,), -1, dtype=dtype, device=self.device)
            symm_mem.get(larger_dst[:numel], hdl, peer=1, offset=offset)
            self.assertEqual(larger_dst[:numel], expected)
            self.assertEqual(larger_dst[numel], -1)

            noncontig_dst = torch.empty(2 * numel, dtype=dtype, device=self.device)[::2]
            with self.assertRaisesRegex(ValueError, "contiguous"):
                symm_mem.get(noncontig_dst, hdl, peer=1)

            with self.assertRaisesRegex(ValueError, "non-negative"):
                symm_mem.get(
                    torch.empty(numel, dtype=dtype, device=self.device),
                    hdl,
                    peer=1,
                    offset=-1,
                )

            with self.assertRaisesRegex(ValueError, "exceeds"):
                symm_mem.get(
                    torch.empty(1, dtype=dtype, device=self.device),
                    hdl,
                    peer=1,
                    offset=hdl.buffer_size // dst.element_size(),
                )

            with self.assertRaisesRegex(ValueError, "invalid peer"):
                symm_mem.get(
                    torch.empty(numel, dtype=dtype, device=self.device),
                    hdl,
                    peer=hdl.world_size,
                )

        c10d.barrier()

    @skip_but_pass_in_sandcastle_if(TEST_WITH_ROCM, "Skip NCCL tests for ROCm")
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version(
        (2, 29, 7), "nccl_reduce_scatter_offset requires nccl 2.29.7"
    )
    @skip_if_lt_x_gpu(2)
    @parametrize("experts_per_rank", [1, 2])
    @parametrize("dim", [0, 1])
    def test_reduce_scatter_offset(self, experts_per_rank: int, dim: int):
        """reduce_scatter_offset: each expert gradient is reduced to its
        destination rank and written to a separate contiguous tensor; the source
        Grouped GEMM buffer is left unmodified."""
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        rows, cols = 64, 32
        n_experts = experts_per_rank * self.world_size

        # dim=1: experts laid out as column blocks [rows, n_experts * cols]
        # dim=0: experts laid out as row blocks    [n_experts * rows, cols]
        if dim == 1:
            buf = symm_mem.empty(
                rows, n_experts * cols, dtype=torch.float, device=self.device
            )
            for i in range(n_experts):
                buf[:, i * cols : (i + 1) * cols] = float((self.rank + 1) * (i + 1))
        else:
            buf = symm_mem.empty(
                n_experts * rows, cols, dtype=torch.float, device=self.device
            )
            for i in range(n_experts):
                buf[i * rows : (i + 1) * rows, :] = float((self.rank + 1) * (i + 1))
        symm_mem.rendezvous(buf, group=group_name)

        # Round-robin: expert i is reduced to rank i % world_size.
        dst_ranks = [i % self.world_size for i in range(n_experts)]
        n_owned = sum(r == self.rank for r in dst_ranks)
        out = [
            torch.zeros(rows, cols, dtype=torch.float, device=self.device)
            for _ in range(n_owned)
        ]
        block_size = cols if dim == 1 else rows
        offsets = [i * block_size for i in range(1, n_experts + 1)]

        symm_mem.reduce_scatter_offset(
            buf, out, group_name, dim=dim, offsets=offsets, dst_ranks=dst_ranks
        )
        torch.cuda.synchronize()

        # out[j] corresponds to expert (rank + j * world_size); expected value is
        # (expert_idx + 1) * sum(r + 1 for r in range(world_size)).
        rank_sum = float(sum(r + 1 for r in range(self.world_size)))
        for j in range(n_owned):
            expert_idx = self.rank + j * self.world_size
            expected = float(expert_idx + 1) * rank_sum
            self.assertEqual(
                out[j],
                torch.full_like(out[j], expected),
                msg=lambda msg: f"{msg}\nrank {self.rank}: out[{j}] should contain the reduced sum",
            )
        # Source buffer must be unmodified.
        for i in range(n_experts):
            if dim == 1:
                src_slice = buf[:, i * cols : (i + 1) * cols]
            else:
                src_slice = buf[i * rows : (i + 1) * rows, :]
            self.assertEqual(
                src_slice,
                torch.full(
                    (rows, cols),
                    float((self.rank + 1) * (i + 1)),
                    dtype=torch.float,
                    device=self.device,
                ),
                msg=lambda msg: f"{msg}\nrank {self.rank}: source buffer block {i} should be unchanged",
            )

    @skip_but_pass_in_sandcastle_if(TEST_WITH_ROCM, "Skip NCCL tests for ROCm")
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version(
        (2, 29, 7), "nccl_reduce_scatter_offset requires nccl 2.29.7"
    )
    @skip_if_lt_x_gpu(2)
    @parametrize("dim", [0, 1])
    def test_reduce_scatter_offset_uneven(self, dim: int):
        """reduce_scatter_offset with uneven block sizes: j=0 and j=1 own blocks
        of different sizes, verifying that out[j] shapes differ across j."""
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        rows, cols = 64, 32
        # j=0 blocks have size_0 along dim; j=1 blocks have size_1 along dim.
        # Arrange blocks as [size_0] * world_size + [size_1] * world_size so
        # that round-robin assigns each rank exactly one block of each size.
        size_0, size_1 = 16, 48
        block_sizes = [size_0] * self.world_size + [size_1] * self.world_size
        offsets = []
        total = 0
        for sz in block_sizes:
            total += sz
            offsets.append(total)

        n_experts = 2 * self.world_size
        if dim == 1:
            buf = symm_mem.empty(rows, total, dtype=torch.float, device=self.device)
            pos = 0
            for i, sz in enumerate(block_sizes):
                buf[:, pos : pos + sz] = float((self.rank + 1) * (i + 1))
                pos += sz
        else:
            buf = symm_mem.empty(total, cols, dtype=torch.float, device=self.device)
            pos = 0
            for i, sz in enumerate(block_sizes):
                buf[pos : pos + sz, :] = float((self.rank + 1) * (i + 1))
                pos += sz
        symm_mem.rendezvous(buf, group=group_name)

        dst_ranks = [i % self.world_size for i in range(n_experts)]
        if dim == 1:
            out = [
                torch.zeros(rows, size_0, dtype=torch.float, device=self.device),
                torch.zeros(rows, size_1, dtype=torch.float, device=self.device),
            ]
        else:
            out = [
                torch.zeros(size_0, cols, dtype=torch.float, device=self.device),
                torch.zeros(size_1, cols, dtype=torch.float, device=self.device),
            ]

        symm_mem.reduce_scatter_offset(
            buf, out, group_name, dim=dim, offsets=offsets, dst_ranks=dst_ranks
        )
        torch.cuda.synchronize()

        rank_sum = float(sum(r + 1 for r in range(self.world_size)))
        for j in range(2):
            expert_idx = self.rank + j * self.world_size
            expected = float(expert_idx + 1) * rank_sum
            self.assertEqual(
                out[j],
                torch.full_like(out[j], expected),
                msg=lambda msg: f"{msg}\nrank {self.rank}: out[{j}] should contain the reduced sum",
            )

    @skip_but_pass_in_sandcastle_if(TEST_WITH_ROCM, "Skip NCCL tests for ROCm")
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version((2, 28, 0), "nccl_all_to_all_nd requires nccl 2.28")
    @skip_if_lt_x_gpu(2)
    @parametrize(
        "scatter_gather,out_2d,input_3d",
        [
            ((1, 0), False, False),
            ((1, 0), True, False),
            ((1, 0), False, True),
            ((1, 0), True, True),
            ((0, 1), False, False),
            ((0, 1), True, False),
            ((0, 1), False, True),
            ((0, 1), True, True),
        ],
    )
    def test_all_to_all_nd(self, scatter_gather, out_2d, input_3d):
        """all_to_all_nd: (1,0)/(0,1); 3-D input [rows,G,loc] or [G,loc,cols] where supported."""
        scatter_dim, gather_dim = scatter_gather
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        p = self.world_size
        dtype = torch.float

        if scatter_dim == 1 and gather_dim == 0:
            local_cols = 4
            rows = 8
            if input_3d:
                buf = symm_mem.empty(
                    rows, p, local_cols, dtype=dtype, device=self.device
                ).fill_(float(self.rank))
            else:
                buf = symm_mem.empty(
                    rows, p * local_cols, dtype=dtype, device=self.device
                ).fill_(float(self.rank))
            symm_mem.rendezvous(buf, group=group_name)
            if out_2d:
                out = torch.empty(p * rows, local_cols, dtype=dtype, device=self.device)
            else:
                out = torch.empty(p, rows, local_cols, dtype=dtype, device=self.device)
            symm_mem.all_to_all_nd(
                buf,
                out,
                scatter_dim=scatter_dim,
                gather_dim=gather_dim,
                group=group_name,
            )
            torch.cuda.synchronize()
            out_view = out.view(p, rows, local_cols) if out_2d else out
            for j in range(p):
                self.assertEqual(
                    out_view[j],
                    torch.full(
                        (rows, local_cols),
                        float(j),
                        dtype=dtype,
                        device=self.device,
                    ),
                    msg=f"rank {self.rank}: out[{j}] should be peer {j}'s column block",
                )
        else:
            local_rows = 4
            cols = 4
            if input_3d:
                buf = symm_mem.empty(
                    p, local_rows, cols, dtype=dtype, device=self.device
                ).fill_(float(self.rank))
            else:
                buf = symm_mem.empty(
                    p * local_rows, cols, dtype=dtype, device=self.device
                ).fill_(float(self.rank))
            symm_mem.rendezvous(buf, group=group_name)
            if out_2d:
                out = torch.empty(local_rows, p * cols, dtype=dtype, device=self.device)
            else:
                out = torch.empty(local_rows, p, cols, dtype=dtype, device=self.device)
            symm_mem.all_to_all_nd(
                buf,
                out,
                scatter_dim=scatter_dim,
                gather_dim=gather_dim,
                group=group_name,
            )
            torch.cuda.synchronize()
            out_view = out.view(local_rows, p, cols) if out_2d else out
            for j in range(p):
                self.assertEqual(
                    out_view[:, j, :],
                    torch.full(
                        (local_rows, cols),
                        float(j),
                        dtype=dtype,
                        device=self.device,
                    ),
                    msg=f"rank {self.rank}: out[:, {j}, :] should be peer {j}'s row block",
                )

    @skip_but_pass_in_sandcastle_if(TEST_WITH_ROCM, "Skip NCCL tests for ROCm")
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version((2, 29), "NCCL one-sided host API support from nccl 2.29")
    @skip_if_lt_x_gpu(2)
    def test_put_wait_signal(self):
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        # Use this barrier to make sure all ranks are initialized.
        c10d.barrier()
        group_name = c10d.group.WORLD.group_name

        dtype = torch.float
        numel = 1024
        src = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(self.rank)
        dst = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        symm_mem.rendezvous(src, group=group_name)
        hdl = symm_mem.rendezvous(dst, group=group_name)

        # Pair ranks: odd ranks send to previous even ranks.
        if self.rank % 2 == 1:
            dst_rank = self.rank - 1
            symm_mem.put_signal(src, hdl, dst_rank)
        elif self.rank % 2 == 0 and self.rank + 1 < self.world_size:
            src_rank = self.rank + 1
            symm_mem.wait_signal(hdl, src_rank)
            self.assertEqual(dst, torch.full_like(dst, float(src_rank)))

        c10d.barrier()

    @skip_but_pass_in_sandcastle_if(TEST_WITH_ROCM, "Skip NCCL tests for ROCm")
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @skip_if_lt_x_gpu(2)
    def test_mempool_tensor_factory(self):
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        # Need this all_reduce to initialize NCCL communicator. Otherwise, the
        # test will hang.  TODO: investigate how NCCLSymmetricMemory can
        # initialize NCCL communicator.
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        dtype = torch.float
        numel = 1024

        mempool = symm_mem.get_mem_pool(self.device)

        with torch.cuda.use_mem_pool(mempool):
            tensor = torch.arange(numel, dtype=dtype, device=self.device)

        # Rendezvous should not error out
        symm_mem.rendezvous(tensor, group=group_name)
        tensor = torch.ops.symm_mem.one_shot_all_reduce(tensor, "sum", group_name)
        expected = (
            torch.arange(numel, dtype=dtype, device=self.device) * self.world_size
        )
        self.assertEqual(tensor, expected)

    @skip_but_pass_in_sandcastle_if(TEST_WITH_ROCM, "Skip NCCL tests for ROCm")
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @skip_if_lt_x_gpu(2)
    def test_mempool_compute_ops(self):
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        # Need this all_reduce to initialize NCCL communicator. Otherwise, the
        # test will hang.  TODO: investigate how NCCLSymmetricMemory can
        # initialize NCCL communicator.
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        dtype = torch.float
        dim = 1024
        w = torch.ones(dim, dim, dtype=dtype, device=self.device)
        x = torch.ones(1, dim, dtype=dtype, device=self.device)

        mempool = symm_mem.get_mem_pool(self.device)

        with torch.cuda.use_mem_pool(mempool):
            y = torch.mm(x, w)

        # One-shot all-reduce should not error out
        y = torch.ops.symm_mem.one_shot_all_reduce(y, "sum", group_name)
        expected = torch.mm(x, w) * self.world_size
        self.assertEqual(y, expected)

    @skip_but_pass_in_sandcastle_if(TEST_WITH_ROCM, "Skip NCCL tests for ROCm")
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version((2, 27), "NCCL Symmetric Memory support from nccl 2.27")
    @skip_if_lt_x_gpu(2)
    def test_mempool_recycled_barrier(self):
        # Regression test for the signal-pad pollution bug: ncclMemAlloc does
        # not zero memory, so a MemPool-recycled NCCL symmetric allocation could
        # start the CAS-based barrier() protocol from a non-zero signal pad and
        # deadlock. alloc() zeros the pad up front. Allocate from the SymmMem
        # MemPool, run a barrier / buffer round-trip, free, then allocate the
        # same size again (recycling the freed block) and confirm the round-trip
        # still works on the recycled region.
        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name
        mempool = symm_mem.get_mem_pool(self.device)
        numel, dtype = 1024, torch.float

        def barrier_roundtrip():
            with torch.cuda.use_mem_pool(mempool):
                t = torch.empty(numel, dtype=dtype, device=self.device)
            hdl = symm_mem.rendezvous(t, group=group_name)
            t.fill_(self.rank)
            # Bounded barriers so a polluted-pad regression fails cleanly
            # instead of hanging.
            hdl.barrier(timeout_ms=60000)
            for peer in range(self.world_size):
                buf = hdl.get_buffer(peer, (numel,), dtype)
                self.assertTrue(buf.eq(peer).all())
            hdl.barrier(timeout_ms=60000)
            return t, hdl

        t1, hdl1 = barrier_roundtrip()
        del hdl1, t1
        t2, hdl2 = barrier_roundtrip()
        del hdl2, t2

    @skip_but_pass_in_sandcastle_if(TEST_WITH_ROCM, "Skip NCCL tests for ROCm")
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @skip_but_pass_in_sandcastle_if(
        os.environ.get("NCCL_NVLS_ENABLE", "1") == "0",
        "NCCL_NVLS_ENABLE=0",
    )
    @skip_if_lt_x_gpu(2)
    @requires_nccl_version(
        (2, 29), "NCCL Symmetric Memory multicast support from nccl 2.29"
    )
    def test_multicast_ptr(self) -> None:
        """
        Get the multicast pointer
        """
        from torch._C._autograd import DeviceType
        from torch._C._distributed_c10d import _SymmetricMemory

        symm_mem.set_backend("NCCL")
        torch.cuda.set_device(self.rank)
        c10d.all_reduce(torch.ones(1, device=self.device))
        group_name = c10d.group.WORLD.group_name

        tensor = symm_mem.empty(1, device=self.device)
        handle = symm_mem.rendezvous(tensor, group_name)
        if _SymmetricMemory.has_multicast_support(DeviceType.CUDA, self.device.index):
            self.assertNotEqual(handle.multicast_ptr, 0)
        else:
            self.assertEqual(handle.multicast_ptr, 0)


@requires_cuda_p2p_access()
class NCCLSymmemExpandableSegmentsTest(MultiProcContinuousTest):
    """NCCL symmetric memory tests using the CUDA Caching Allocator (CCA)
    expandable-segments path.

    When ``expandable_segments`` is enabled, ``NCCLSymmetricMemoryAllocator``
    allocates via ``raw_alloc``/``raw_delete`` instead of ``ncclMemAlloc``/
    ``ncclMemFree``. The caching allocator recycles freed addresses, so a freed
    allocation's virtual address can be handed back out for a later (possibly
    differently sized) allocation. These tests exercise that recycling.

    The whole class runs with ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``
    set *before* the worker processes are spawned, so every worker uses
    expandable segments from its first allocation (matching how a user launches
    the job, and avoiding per-test runtime toggling). ``symm_mem.empty`` skips
    the implicit MemPool automatically when expandable_segments is enabled (the
    two are incompatible). It is a separate class so that the configuration in
    this test doesn't pollute other test suites.
    """

    _prior_alloc_conf = None

    @classmethod
    def setUpClass(cls):
        # Spawning is deferred to the first setUp, so setting the env here (in
        # the dispatcher process) ensures the spawned workers inherit it.
        cls._prior_alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        if cls._prior_alloc_conf is None:
            os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        else:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = cls._prior_alloc_conf

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    @classmethod
    def _init_pg(cls, rank, world_size, rdvz_file):
        # Eager NCCL communicator init via device_id, so symm_mem rendezvous
        # does not require a separate warm-up collective.
        if rdvz_file is None:
            raise AssertionError("Expected rdvz_file to not be None")
        os.environ["LOCAL_RANK"] = str(rank)
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
        store = c10d.FileStore(rdvz_file, world_size)
        c10d.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=rank,
            store=store,
            timeout=cls.timeout,
            device_id=device,
        )
        cls.pg = c10d.distributed_c10d._get_default_group()

    def _expandable_segments_active(self) -> bool:
        settings = torch.cuda.memory._snapshot()["allocator_settings"]
        return bool(settings.get("expandable_segments", False))

    def _segment_for_ptr(self, ptr: int):
        # Find this rank's CCA snapshot segment whose mapped VA range covers
        # `ptr`, or None if `ptr` is not backed by any caching-allocator
        # segment (e.g. it came from the ncclMemAlloc path instead of
        # raw_alloc). The snapshot reports one segment per contiguous mapped
        # range, so a multi-chunk allocation that is mapped contiguously is a
        # single segment spanning all its chunks.
        for seg in torch.cuda.memory._snapshot()["segments"]:
            if seg["device"] != self.rank:
                continue
            if seg["address"] <= ptr < seg["address"] + seg["total_size"]:
                return seg
        return None

    def _setup_group(self) -> str:
        torch.cuda.set_device(self.rank)
        if not self._expandable_segments_active():
            # expandable_segments is not supported on this platform (e.g. no
            # driver-API support), so the raw_alloc path these tests target
            # cannot be exercised. Skip the whole test rather than running it
            # silently on the ncclMemAlloc path. In MultiProcContinuousTest a
            # body-level skip must use sys.exit with a TEST_SKIPS code;
            # self.skipTest() raised here would be reported as a failure. The
            # generic skip message points at this subprocess log for the reason.
            print(
                "Skipping NCCLSymmemExpandableSegmentsTest: expandable_segments "
                "is not active on this platform.",
                file=sys.stderr,
                flush=True,
            )
            sys.exit(TEST_SKIPS["generic"].exit_code)
        symm_mem.set_backend("NCCL")
        # Warm up the NCCL communicator before symm_mem rendezvous.
        c10d.all_reduce(torch.ones(1, device=self.device))
        return c10d.group.WORLD.group_name

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_ROCM, "expandable_segments is not supported on ROCm"
    )
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version(
        (2, 30), "NCCL multi-segment symmetric memory support from nccl 2.30"
    )
    @skip_if_lt_x_gpu(2)
    def test_nccl_symmem_alloc(self):
        # Functional check that the expandable segments CCA raw_alloc path,
        # rendezvouses, and runs a collective correctly.
        group_name = self._setup_group()
        torch.cuda.synchronize()
        dtype = torch.float
        numel = 1024

        inp = symm_mem.empty(numel, dtype=dtype, device=self.device)
        symm_mem.rendezvous(inp, group=group_name)
        result = torch.ops.symm_mem.one_shot_all_reduce(
            inp.fill_(self.rank), "sum", group_name
        )
        # Expected all-reduce sum: sum of ranks 0..world_size-1.
        self.assertEqual(
            result,
            torch.full_like(result, (self.world_size - 1) * self.world_size / 2),
        )

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_ROCM, "expandable_segments is not supported on ROCm"
    )
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version(
        (2, 30), "NCCL multi-segment symmetric memory support from nccl 2.30"
    )
    @skip_if_lt_x_gpu(2)
    def test_nccl_symmem_address_reuse_loop(self):
        # Repeated alloc -> rendezvous -> collective -> free of the SAME size.
        # The caching allocator hands the freed address back each iteration, so
        # this stresses window register/deregister bookkeeping keyed by ptr. If
        # recycling is mishandled the results corrupt, hang, or IMA.
        group_name = self._setup_group()
        dtype = torch.float
        numel = 1024

        seen_ptrs = set()
        for _ in range(8):
            t = symm_mem.empty(numel, dtype=dtype, device=self.device)
            seen_ptrs.add(t.data_ptr())
            symm_mem.rendezvous(t, group=group_name)
            result = torch.ops.symm_mem.one_shot_all_reduce(
                t.fill_(self.rank), "sum", group_name
            )
            self.assertEqual(
                result,
                torch.full_like(result, (self.world_size - 1) * self.world_size / 2),
            )
            del t
        # Address reuse is deterministic here: `t` is freed before the next
        # alloc and every allocation is the same size, so the caching allocator
        # hands the single freed (and only outstanding) address back each
        # iteration. Hence exactly one unique ptr across all iterations.
        self.assertEqual(len(seen_ptrs), 1)

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_ROCM, "expandable_segments is not supported on ROCm"
    )
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version(
        (2, 30), "NCCL multi-segment symmetric memory support from nccl 2.30"
    )
    @skip_if_lt_x_gpu(2)
    def test_nccl_symmem_address_reuse_grow(self):
        # Allocate a symm tensor, rendezvous, free it, then allocate a LARGER
        # one that spans MULTIPLE expandable-segment chunks and recycles the
        # same base virtual address. The caching allocator maps physical memory
        # in fixed chunks (20 MiB for the large pool), so an allocation larger
        # than one chunk is backed by several chunks. With expandable segments
        # the allocator grows the freed segment in place by mapping additional
        # chunks contiguously, so the base VA is reused and the segment extends
        # to back the full large allocation. If the ptr-keyed bookkeeping
        # retains stale size/window state from the first allocation, collectives
        # over the full large buffer will be wrong.
        #
        # Both allocations are intentionally in the caching allocator's LARGE
        # pool (> 1 MiB). Base-VA reuse only holds within a single pool: the
        # small and large pools reserve separate expandable virtual address
        # ranges, so a small-pool allocation and a large-pool allocation can
        # never share a base address.
        group_name = self._setup_group()
        dtype = torch.float
        # first: one 20 MiB chunk; large: two chunks (40 MiB) after grow.
        first_numel = 2 * 1024 * 1024  # 8 MiB (large pool, single chunk)
        large_numel = 8 * 1024 * 1024  # 32 MiB > 20 MiB chunk (multi-chunk)
        chunk_bytes = 20 * 1024 * 1024

        first = symm_mem.empty(first_numel, dtype=dtype, device=self.device)
        first_ptr = first.data_ptr()
        first_seg = self._segment_for_ptr(first_ptr)
        self.assertIsNotNone(
            first_seg,
            "symm_mem tensor is not backed by a caching-allocator segment; "
            "expected the expandable_segments raw_alloc path.",
        )
        self.assertTrue(first_seg["is_expandable"])
        symm_mem.rendezvous(first, group=group_name)
        # Run a collective over `first` before freeing it, so the smaller
        # allocation also goes through the full register -> collective -> free
        # -> recycle path (not just window registration) before its virtual
        # address is recycled by the larger allocation below.
        torch.ops.symm_mem.one_shot_all_reduce(
            first.fill_(self.rank), "sum", group_name
        )
        torch.cuda.synchronize()
        del first

        large = symm_mem.empty(large_numel, dtype=dtype, device=self.device)
        large_ptr = large.data_ptr()
        # Base VA reuse is deterministic here: `first` is the only outstanding
        # large-pool allocation when it is freed, so the larger `large` recycles
        # the same base VA, with the expandable segment grown in place.
        self.assertEqual(first_ptr, large_ptr)

        large_seg = self._segment_for_ptr(large_ptr)
        self.assertIsNotNone(large_seg)
        self.assertTrue(large_seg["is_expandable"])
        # The first segment is reused (same base address) and grown in place to
        # back the full multi-chunk allocation, so it now spans more than one
        # 20 MiB chunk.
        self.assertEqual(large_seg["address"], first_seg["address"])
        self.assertGreaterEqual(
            large_seg["total_size"], large_numel * large.element_size()
        )
        self.assertGreater(large_seg["total_size"], chunk_bytes)

        handle = symm_mem.rendezvous(large, group=group_name)
        result = torch.ops.symm_mem.one_shot_all_reduce(
            large.fill_(self.rank), "sum", group_name
        )
        self.assertEqual(
            result,
            torch.full_like(result, (self.world_size - 1) * self.world_size / 2),
        )

        # all_gather over the grown buffer via standard c10d, which dispatches
        # to NCCL's symmetric-memory (window-registered) collective kernels
        # (network-capable, runs over IB on multi-node) rather than the
        # intra-node one_shot_all_reduce above. Validates the full multi-chunk
        # window is registered and usable by a real NCCL collective.
        large.fill_(self.rank)
        gathered = symm_mem.empty(
            large_numel * self.world_size, dtype=dtype, device=self.device
        )
        symm_mem.rendezvous(gathered, group=group_name)
        c10d.all_gather_single(gathered, large)
        torch.cuda.synchronize()
        expected = gathered.new_empty(large_numel * self.world_size)
        for r in range(self.world_size):
            expected[r * large_numel : (r + 1) * large_numel] = r
        self.assertEqual(gathered, expected)

        # The full large buffer (not just a stale small window) must be visible
        # to peers.
        large.fill_(self.rank)
        c10d.barrier()
        peer_rank = (self.rank + 1) % self.world_size
        buf = handle.get_buffer(peer_rank, (large_numel,), dtype)
        self.assertTrue(buf.eq(peer_rank).all())
        c10d.barrier()

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_ROCM, "expandable_segments is not supported on ROCm"
    )
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @requires_nccl_version(
        (2, 30), "NCCL multi-segment symmetric memory support from nccl 2.30"
    )
    @skip_if_lt_x_gpu(2)
    def test_nccl_symmem_address_reuse_shrink(self):
        # Shrinking counterpart of test_nccl_symmem_address_reuse_grow: allocate
        # a LARGE multi-chunk buffer, run a collective over it, free it, then
        # allocate a SMALLER buffer that recycles the same base virtual address
        # and run a collective over it. Both allocations are in the large pool
        # so the base VA is recycled (the smaller one reuses the head of the
        # freed segment). This exercises the full register -> collective -> free
        # -> recycle -> register path in the shrinking direction; if the larger
        # allocation's window/teardown state is not cleaned up before its VA is
        # recycled, the smaller allocation's collective corrupts or faults.
        group_name = self._setup_group()
        dtype = torch.float
        large_numel = 8 * 1024 * 1024  # 32 MiB, large pool, multi-chunk
        small_numel = 2 * 1024 * 1024  # 8 MiB, large pool (so the VA recycles)

        large = symm_mem.empty(large_numel, dtype=dtype, device=self.device)
        large_ptr = large.data_ptr()
        large_seg = self._segment_for_ptr(large_ptr)
        self.assertIsNotNone(
            large_seg,
            "symm_mem tensor is not backed by a caching-allocator segment; "
            "expected the expandable_segments raw_alloc path.",
        )
        self.assertTrue(large_seg["is_expandable"])
        symm_mem.rendezvous(large, group=group_name)
        torch.ops.symm_mem.one_shot_all_reduce(
            large.fill_(self.rank), "sum", group_name
        )
        torch.cuda.synchronize()
        del large

        small = symm_mem.empty(small_numel, dtype=dtype, device=self.device)
        small_ptr = small.data_ptr()
        # The smaller allocation recycles the head of the freed large segment.
        self.assertEqual(large_ptr, small_ptr)
        small_seg = self._segment_for_ptr(small_ptr)
        self.assertIsNotNone(small_seg)
        self.assertTrue(small_seg["is_expandable"])
        symm_mem.rendezvous(small, group=group_name)
        result = torch.ops.symm_mem.one_shot_all_reduce(
            small.fill_(self.rank), "sum", group_name
        )
        self.assertEqual(
            result,
            torch.full_like(result, (self.world_size - 1) * self.world_size / 2),
        )

        # all_gather over the recycled small buffer via standard c10d, which
        # dispatches to NCCL's symmetric-memory (window-registered) collective
        # kernels (network-capable, runs over IB on multi-node), validating the
        # recycled window is usable by a real NCCL collective.
        small.fill_(self.rank)
        gathered = symm_mem.empty(
            small_numel * self.world_size, dtype=dtype, device=self.device
        )
        symm_mem.rendezvous(gathered, group=group_name)
        c10d.all_gather_single(gathered, small)
        torch.cuda.synchronize()
        expected = gathered.new_empty(small_numel * self.world_size)
        for r in range(self.world_size):
            expected[r * small_numel : (r + 1) * small_numel] = r
        self.assertEqual(gathered, expected)


instantiate_device_type_tests(TestNCCL, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
