# Owner(s): ["oncall: distributed"]

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
    IS_WINDOWS,
    load_tests,
    NoTest,
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
