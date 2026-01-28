# Owner(s): ["oncall: distributed"]
"""
Test hierarchical all-reduce Triton kernel with NVSHMEM backend.

This test file contains multiprocess tests that verify the hierarchical
all-reduce kernel works correctly with NVSHMEMSymmComm wrapper objects.

The hierarchical all-reduce algorithm currently implements:
1. Intra-LSA Reduce-Scatter: Pull-based reduction within LSA domain
2. Intra-LSA Broadcast: Push-based broadcast to distribute results

NOTE: Inter-domain (multi-node) ring all-reduce (Phase 2) requires NVSHMEM 3.0+
with nvshmemx_putmem_signal_nbi support and is not yet implemented.

Tests both:
1. Dynamic dispatch (BACKEND_DEFAULT) - runtime dispatch based on SymmContext type
2. Explicit NVSHMEM dispatch (BACKEND_NVSHMEM) - direct dispatch to NVSHMEM backend

NOTE: NCCL tests are skipped because NCCL does not provide a device bitcode library
(libnccl_device.bc) that can be linked with Triton kernels. Only NVSHMEM backend
is functional since it provides libnvshmem_device.bc.
"""

import ctypes
import sys
import unittest
from collections.abc import Callable

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
)


if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip ASAN as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


# Check for Triton availability
try:
    import triton  # noqa: F401

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


def requires_triton(func: Callable) -> Callable:
    """Decorator to skip tests if Triton is not available."""
    return unittest.skipUnless(TRITON_AVAILABLE, "Triton not available")(func)


def requires_nvshmem(func: Callable) -> Callable:
    """Decorator to skip tests if NVSHMEM is not available."""
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            from torch.distributed._symmetric_memory import is_nvshmem_available

            if not is_nvshmem_available():
                raise unittest.SkipTest("NVSHMEM not available")
            return func(*args, **kwargs)
        except ImportError:
            raise unittest.SkipTest("NVSHMEM not available") from None

    return wrapper


# Import Triton kernels if available
if TRITON_AVAILABLE:
    from torch._extern_triton._torch_symm_triton import (
        BACKEND_DEFAULT,
        BACKEND_NVSHMEM,
    )
    from torch._extern_triton._all_reduce_hierarchical import (
        all_reduce_hierarchical_dynamic,
        all_reduce_hierarchical_nvshmem,
    )


class TestAllReduceHierarchical(MultiProcessTestCase):
    """
    Multiprocess test case for hierarchical all-reduce Triton kernel.

    Tests the NVSHMEM backend with the hierarchical all-reduce kernel
    using both dynamic dispatch and explicit NVSHMEM dispatch.

    The test verifies that the three-phase algorithm correctly computes
    the all-reduce sum across all ranks.
    """

    @property
    def world_size(self) -> int:
        return 2

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()

    def _init_process_group(self):
        """Initialize the process group with NCCL backend."""
        torch.cuda.set_device(self.rank)
        dist.init_process_group(
            backend="nccl",
            init_method=f"file://{self.file_name}",
            world_size=self.world_size,
            rank=self.rank,
        )

    def _cleanup_process_group(self):
        """Clean up the process group."""
        if dist.is_initialized():
            dist.destroy_process_group()

    def _run_hierarchical_all_reduce_test(
        self,
        comm_class,
        kernel_func,
        num_elements: int = 256,
        backend_hint: int = 0,
        block_size: int = 64,
    ):
        """
        Helper method to run hierarchical all-reduce test.

        Args:
            comm_class: Either NCCLSymmComm or NVSHMEMSymmComm class
            kernel_func: The Triton kernel to use for the test
            num_elements: Number of float32 elements to reduce
            backend_hint: Backend hint to pass to the kernel (0=DEFAULT, 2=NVSHMEM)
            block_size: Triton block size for the kernel
        """
        self._init_process_group()
        try:
            device = torch.device("cuda", self.rank)

            # Calculate buffer sizes
            # Data buffer: num_elements * sizeof(float32)
            # Scratch buffer: same size for inter-node ring algorithm
            data_buffer_size = num_elements * 4  # float32 = 4 bytes
            scratch_buffer_size = num_elements * 4

            group_name = dist.group.WORLD.group_name

            # Create communicator with sufficient buffer for data + scratch
            total_buffer_size = data_buffer_size + scratch_buffer_size
            comm = comm_class(group_name, total_buffer_size, self.rank)

            # Get buffer and context pointers
            buffer_ptr = comm.get_buffer_ptr()
            ctx_ptr = comm.get_context_ptr()

            # Get topology information from comm
            my_rank = comm.get_rank()
            world_size = comm.get_world_size()
            # For single-machine testing, LSA size = world_size (all GPUs in same domain)
            lsa_size = world_size

            # Get team pointer for multicast operations
            team_ptr = comm.get_team_ptr()

            # Scratch buffer starts after data buffer
            scratch_ptr = buffer_ptr + data_buffer_size

            # Initialize local tensor with (rank + 1) values
            local_tensor = torch.full(
                (num_elements,),
                float(self.rank + 1),
                dtype=torch.float32,
                device=device,
            )

            # Copy local data to the symmetric buffer
            cuda_rt = ctypes.CDLL("libcudart.so")

            # Wait for all initialization to complete
            torch.cuda.synchronize(device)
            dist.barrier()

            # Copy data to symmetric buffer
            cuda_rt.cudaMemcpy(
                ctypes.c_void_p(buffer_ptr),
                ctypes.c_void_p(local_tensor.data_ptr()),
                num_elements * 4,
                1,  # cudaMemcpyDeviceToDevice
            )

            # Zero out scratch buffer
            cuda_rt.cudaMemset(
                ctypes.c_void_p(scratch_ptr),
                0,
                scratch_buffer_size,
            )

            torch.cuda.synchronize(device)

            # Barrier before kernel to ensure all data is ready
            dist.barrier()

            # Launch kernel with single CTA
            # Use cooperative grid for NVSHMEM barriers to work correctly
            kernel_func[(1,)](
                ctx_ptr,
                buffer_ptr,
                scratch_ptr,
                team_ptr,
                num_elements,
                my_rank,
                world_size,
                lsa_size,
                block_size,
                backend_hint,
                launch_cooperative_grid=True,
                num_ctas=1,
            )

            # Synchronize after kernel
            torch.cuda.synchronize(device)
            dist.barrier()

            # Copy result back
            result_tensor = torch.zeros(
                num_elements, dtype=torch.float32, device=device
            )
            cuda_rt.cudaMemcpy(
                ctypes.c_void_p(result_tensor.data_ptr()),
                ctypes.c_void_p(buffer_ptr),
                num_elements * 4,
                1,  # cudaMemcpyDeviceToDevice
            )
            torch.cuda.synchronize(device)

            # Verify result
            # Expected: sum of all ranks' values = 1 + 2 = 3 for world_size=2
            expected_sum = float(self.world_size * (self.world_size + 1) / 2)
            expected = torch.full(
                (num_elements,), expected_sum, dtype=torch.float32, device=device
            )

            torch.testing.assert_close(
                result_tensor,
                expected,
                rtol=1e-5,
                atol=1e-5,
                msg=f"Hierarchical all-reduce result mismatch on rank {self.rank}",
            )

        finally:
            self._cleanup_process_group()

    @skip_if_lt_x_gpu(2)
    @requires_triton
    @requires_nvshmem
    @parametrize("num_elements", [128, 256, 512])
    def test_nvshmem_hierarchical_all_reduce_dynamic(self, num_elements: int):
        """
        Test hierarchical all-reduce with NVSHMEM backend using dynamic dispatch.

        Uses NVSHMEMSymmComm to create symmetric memory and NVSHMEMSymmContext,
        then calls all_reduce_hierarchical with BACKEND_DEFAULT via Triton kernel.
        The unified dispatcher routes to the NVSHMEM backend based on context type.

        This tests the runtime dispatch path where the backend is determined
        by examining the SymmContext type field.
        """
        from torch._C._distributed_c10d import NVSHMEMSymmComm

        self._run_hierarchical_all_reduce_test(
            NVSHMEMSymmComm,
            all_reduce_hierarchical_dynamic,
            num_elements,
            backend_hint=BACKEND_DEFAULT,
        )

    @skip_if_lt_x_gpu(2)
    @requires_triton
    @requires_nvshmem
    @parametrize("num_elements", [128, 256, 512])
    def test_nvshmem_hierarchical_all_reduce_explicit(self, num_elements: int):
        """
        Test hierarchical all-reduce with explicit NVSHMEM backend dispatch.

        Uses NVSHMEMSymmComm to create symmetric memory and NVSHMEMSymmContext,
        then calls all_reduce_hierarchical with BACKEND_NVSHMEM via Triton kernel.
        This bypasses the runtime dispatch and calls the NVSHMEM backend directly.

        This tests the compile-time dispatch path where BACKEND_NVSHMEM is
        specified as a constexpr argument, avoiding runtime type checking.
        """
        from torch._C._distributed_c10d import NVSHMEMSymmComm

        self._run_hierarchical_all_reduce_test(
            NVSHMEMSymmComm,
            all_reduce_hierarchical_nvshmem,
            num_elements,
            backend_hint=BACKEND_NVSHMEM,
        )

    @skip_if_lt_x_gpu(2)
    @requires_triton
    @requires_nvshmem
    @parametrize("block_size", [32, 64, 128])
    def test_nvshmem_hierarchical_all_reduce_block_sizes(self, block_size: int):
        """
        Test hierarchical all-reduce with different Triton block sizes.

        Verifies that the kernel works correctly with various block sizes,
        which affects the parallelism and memory access patterns.
        """
        from torch._C._distributed_c10d import NVSHMEMSymmComm

        num_elements = 512  # Fixed size, divisible by all block sizes

        self._run_hierarchical_all_reduce_test(
            NVSHMEMSymmComm,
            all_reduce_hierarchical_nvshmem,
            num_elements,
            backend_hint=BACKEND_NVSHMEM,
            block_size=block_size,
        )


class TestAllReduceHierarchicalLargeScale(MultiProcessTestCase):
    """
    Larger scale tests for hierarchical all-reduce with 4 GPUs.

    These tests exercise the multi-node ring algorithm when running
    on a single machine with 4+ GPUs, treating pairs of GPUs as
    separate "nodes" (LSA domains).
    """

    @property
    def world_size(self) -> int:
        return 4

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()

    def _init_process_group(self):
        """Initialize the process group with NCCL backend."""
        torch.cuda.set_device(self.rank)
        dist.init_process_group(
            backend="nccl",
            init_method=f"file://{self.file_name}",
            world_size=self.world_size,
            rank=self.rank,
        )

    def _cleanup_process_group(self):
        """Clean up the process group."""
        if dist.is_initialized():
            dist.destroy_process_group()

    @skip_if_lt_x_gpu(4)
    @requires_triton
    @requires_nvshmem
    @parametrize("num_elements", [256, 512, 1024])
    def test_nvshmem_hierarchical_all_reduce_4_gpus(self, num_elements: int):
        """
        Test hierarchical all-reduce with 4 GPUs.

        This test exercises the full hierarchical algorithm with
        multiple "nodes" (LSA domains), testing the inter-node
        ring reduce-scatter and all-gather phases.
        """
        from torch._C._distributed_c10d import NVSHMEMSymmComm

        self._init_process_group()
        try:
            device = torch.device("cuda", self.rank)

            # Buffer sizes
            data_buffer_size = num_elements * 4
            scratch_buffer_size = num_elements * 4
            total_buffer_size = data_buffer_size + scratch_buffer_size

            group_name = dist.group.WORLD.group_name
            comm = NVSHMEMSymmComm(group_name, total_buffer_size, self.rank)

            buffer_ptr = comm.get_buffer_ptr()
            ctx_ptr = comm.get_context_ptr()

            # Get topology information from comm
            my_rank = comm.get_rank()
            world_size = comm.get_world_size()
            # For single-machine testing, LSA size = world_size (all GPUs in same domain)
            lsa_size = world_size

            # Get team pointer for multicast operations
            team_ptr = comm.get_team_ptr()

            scratch_ptr = buffer_ptr + data_buffer_size

            # Initialize with rank-specific values
            local_tensor = torch.full(
                (num_elements,),
                float(self.rank + 1),
                dtype=torch.float32,
                device=device,
            )

            cuda_rt = ctypes.CDLL("libcudart.so")
            torch.cuda.synchronize(device)
            dist.barrier()

            # Copy data to symmetric buffer
            cuda_rt.cudaMemcpy(
                ctypes.c_void_p(buffer_ptr),
                ctypes.c_void_p(local_tensor.data_ptr()),
                num_elements * 4,
                1,
            )
            cuda_rt.cudaMemset(ctypes.c_void_p(scratch_ptr), 0, scratch_buffer_size)
            torch.cuda.synchronize(device)
            dist.barrier()

            # Launch kernel with single CTA (cooperative grid requirement)
            block_size = 64

            all_reduce_hierarchical_nvshmem[(1,)](
                ctx_ptr,
                buffer_ptr,
                scratch_ptr,
                team_ptr,
                num_elements,
                my_rank,
                world_size,
                lsa_size,
                block_size,
                BACKEND_NVSHMEM,
                launch_cooperative_grid=True,
                num_ctas=1,
            )

            torch.cuda.synchronize(device)
            dist.barrier()

            # Copy result back
            result_tensor = torch.zeros(
                num_elements, dtype=torch.float32, device=device
            )
            cuda_rt.cudaMemcpy(
                ctypes.c_void_p(result_tensor.data_ptr()),
                ctypes.c_void_p(buffer_ptr),
                num_elements * 4,
                1,
            )
            torch.cuda.synchronize(device)

            # Verify: sum should be 1 + 2 + 3 + 4 = 10
            expected_sum = float(self.world_size * (self.world_size + 1) / 2)
            expected = torch.full(
                (num_elements,), expected_sum, dtype=torch.float32, device=device
            )

            torch.testing.assert_close(
                result_tensor,
                expected,
                rtol=1e-5,
                atol=1e-5,
                msg=f"Hierarchical all-reduce result mismatch on rank {self.rank}",
            )

        finally:
            self._cleanup_process_group()


instantiate_parametrized_tests(TestAllReduceHierarchical)
instantiate_parametrized_tests(TestAllReduceHierarchicalLargeScale)


if __name__ == "__main__":
    run_tests()
