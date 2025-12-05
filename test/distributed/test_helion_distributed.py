# Owner(s): ["module: c10d"]

"""
Helion Distributed Tests for Symmetric Memory
==============================================

This test suite validates that Helion distributed kernels work correctly
with PyTorch's symmetric memory infrastructure. These tests require:
- 4 GPUs with P2P access
- NVSHMEM support
- Helion package installed (pip install helion)
"""

from unittest import skipUnless

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed._symmetric_memory import enable_symm_mem_for_group
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
    skip_if_rocm_multiprocess,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    requires_cuda_p2p_access,
    run_tests,
)


@instantiate_parametrized_tests
@requires_cuda_p2p_access()
class HelionDistributedTest(MultiProcessTestCase):
    """
    Test class for Helion distributed operations using symmetric memory.

    These tests validate that Helion kernels can correctly use PyTorch's
    symmetric memory infrastructure for distributed operations like
    all-gather matmul and all-reduce.
    """

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    @property
    def world_size(self) -> int:
        return 4

    @property
    def device(self) -> torch.device:
        return torch.device(f"cuda:{self.rank}")

    def _init_process(self):
        torch.cuda.set_device(self.device)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        torch.manual_seed(42 + self.rank)

    @skip_if_rocm_multiprocess
    @skip_if_lt_x_gpu(4)
    def test_helion_all_gather_matmul(self):
        """
        Test all-gather matrix multiplication using Helion kernel.

        This test verifies that:
        1. Symmetric memory allocation works correctly
        2. All-gather with progress tracking operates correctly
        3. Matrix multiplication with progress-based synchronization produces correct results
        4. Results match PyTorch's fused_all_gather_matmul reference implementation
        """
        self._init_process()

        # Import Helion modules
        from helion._testing import EXAMPLES_DIR, import_path

        mod = import_path(EXAMPLES_DIR / "all_gather_matmul.py")

        M, N, K = 4096, 6656, 16384

        a_shared = symm_mem.empty(
            M // self.world_size, K, dtype=torch.bfloat16, device=self.device
        ).normal_()

        b = (
            torch.randn((K, N), device=self.device, dtype=torch.bfloat16)
            .T.contiguous()
            .T
        )

        symm_mem_group = dist.group.WORLD
        if symm_mem_group is None:
            raise RuntimeError("No symmetric memory group available")
        symm_mem_hdl = symm_mem.rendezvous(a_shared, group=symm_mem_group)
        a_shape = list(a_shared.shape)
        a_shape[0] *= symm_mem_hdl.world_size
        a_out = torch.empty(a_shape, dtype=a_shared.dtype, device=a_shared.device)
        progress = torch.zeros(
            symm_mem_hdl.world_size,
            dtype=torch.uint32,
            device=a_shared.device,
        )
        backend_stream = mod.copy_engine_all_gather_w_progress(
            a_out, a_shared, progress, 1
        )

        result = mod.helion_matmul_w_progress(
            a_out, a_shared, b, progress, 1, symm_mem_hdl.rank
        )

        golden_a = a_shared.clone()
        ag_golden, mm_golden = torch.ops.symm_mem.fused_all_gather_matmul(
            golden_a, [b], gather_dim=0, group_name=symm_mem_group.group_name
        )

        torch.testing.assert_close(result, mm_golden[0], rtol=1e-1, atol=1e-1)
        torch.testing.assert_close(a_out, ag_golden)

        torch.cuda.current_stream().wait_stream(backend_stream)
        dist.destroy_process_group()

    @skip_if_rocm_multiprocess
    @skip_if_lt_x_gpu(4)
    def test_helion_all_reduce(self):
        """
        Test one-shot all-reduce using Helion kernel.

        This test verifies that:
        1. NVSHMEM backend can be set correctly
        2. Symmetric memory with signal pads works for cross-device synchronization
        3. Remote tensor access via get_remote_tensors operates correctly
        4. All-reduce results match the reference implementation
        """
        self._init_process()

        # Import Helion modules
        from helion._testing import EXAMPLES_DIR, import_path

        mod = import_path(EXAMPLES_DIR / "all_reduce.py")

        # Only NVSHMEM backend implements `get_remote_tensor` for now.
        symm_mem.set_backend("NVSHMEM")
        group = dist.group.WORLD
        enable_symm_mem_for_group(group.group_name)

        N = 16384
        dtype = torch.bfloat16

        a_shared = symm_mem.empty(
            N // self.world_size, dtype=dtype, device=self.device
        ).normal_()

        symm_mem_hdl = symm_mem.rendezvous(a_shared, group=group)
        local_signal_pad = symm_mem_hdl.get_signal_pad(
            symm_mem_hdl.rank, dtype=torch.int32
        ).view(-1, symm_mem_hdl.world_size)
        signal_pad_addrs = mod.dev_array_to_tensor_short(
            symm_mem_hdl.signal_pad_ptrs_dev,
            (symm_mem_hdl.world_size,),
            dtype=torch.uint64,
            device=a_shared.device,
        )

        result = mod.one_shot_all_reduce_kernel(
            signal_pad_addrs,
            local_signal_pad,
            a_shared,
            symm_mem_hdl.rank,
            group.group_name,
        )

        a_shared_ref = symm_mem.empty(
            N // self.world_size, dtype=dtype, device=self.device
        )
        a_shared_ref.copy_(a_shared)
        expected = mod.reference_one_shot_all_reduce(a_shared_ref)

        torch.testing.assert_close(result, expected, rtol=1e-1, atol=1e-1)

        dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()
