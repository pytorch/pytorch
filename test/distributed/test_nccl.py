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
    TestCase = NoTest  # noqa: F811


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
                msg=f"rank {self.rank}: out[{j}] should contain the reduced sum",
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
                msg=f"rank {self.rank}: source buffer block {i} should be unchanged",
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
                msg=f"rank {self.rank}: out[{j}] should contain the reduced sum",
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


instantiate_device_type_tests(TestNCCL, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
