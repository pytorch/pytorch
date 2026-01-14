# Owner(s): ["oncall: distributed"]
"""
Test symmetric all-reduce Triton kernel with NVSHMEM backend.

This test file contains multiprocess tests that verify the symm_all_reduce_sum_f32
Triton extern function works correctly with NVSHMEMSymmComm wrapper objects.

NOTE: NCCL tests are skipped because NCCL does not provide a device bitcode library
(libnccl_device.bc) that can be linked with Triton kernels. Only NVSHMEM backend
is functional since it provides libnvshmem_device.bc.
"""

import os
import sys
import unittest
from typing import Callable

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
    TestCase,
)


if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip ASAN as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


# Check for Triton availability
try:
    import triton
    from triton import language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    triton = None
    tl = None


def requires_triton(func: Callable) -> Callable:
    """Decorator to skip tests if Triton is not available."""
    return unittest.skipUnless(TRITON_AVAILABLE, "Triton not available")(func)


def requires_nccl_symm_mem(func: Callable) -> Callable:
    """Decorator to skip tests if NCCL symmetric memory is not available."""
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            from torch._C._distributed_c10d import NCCLSymmComm  # noqa: F401

            return func(*args, **kwargs)
        except ImportError:
            raise unittest.SkipTest(
                "NCCLSymmComm not available (requires NCCL >= 2.28)"
            )

    return wrapper


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
            raise unittest.SkipTest("NVSHMEM not available")

    return wrapper


# Define Triton test kernel if Triton is available
if TRITON_AVAILABLE:
    from torch._extern_triton._symm_all_reduce_triton import (
        requires_symm_all_reduce,
        symm_all_reduce_sum_f32,
        SymmAllReduceLibFinder,
    )

    @requires_symm_all_reduce
    @triton.jit
    def symm_all_reduce_test_kernel(
        ctx_ptr,
        buffer_ptr,
        num_elements: tl.constexpr,
    ):
        """
        Test kernel that calls symm_all_reduce_sum_f32.

        This kernel performs an all-reduce sum operation on the symmetric buffer.
        Each rank should have its local buffer initialized with (rank + 1) values,
        and after the all-reduce, all elements should be the sum across all ranks.

        The unified symm_all_reduce_sum_f32 function dispatches to the correct
        backend (NCCL or NVSHMEM) based on the SymmContext type.
        """
        # Call the unified all-reduce function
        # byte_offset = 0, num_elements from parameter
        byte_offset: tl.int64 = 0
        n_elems: tl.int64 = num_elements
        result = symm_all_reduce_sum_f32(
            ctx_ptr,
            buffer_ptr,
            byte_offset,
            n_elems,
        )


class TestSymmAllReduceTriton(MultiProcessTestCase):
    """
    Multiprocess test case for symmetric all-reduce Triton kernels.

    Tests the NVSHMEM backend with the unified symm_all_reduce_sum_f32
    Triton extern function.

    NOTE: NCCL tests are skipped because NCCL does not provide a device
    bitcode library that can be linked with Triton kernels.
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

    def _run_all_reduce_test(
        self,
        comm_class,
        kernel_func,
        num_elements: int = 128,
    ):
        """
        Helper method to run all-reduce test with given communicator class.

        Args:
            comm_class: Either NCCLSymmComm or NVSHMEMSymmComm class
            kernel_func: The Triton kernel to use for the test
            num_elements: Number of float32 elements to reduce
        """
        self._init_process_group()
        try:
            device = torch.device("cuda", self.rank)

            # Create communicator
            buffer_size = num_elements * 4  # float32 = 4 bytes
            group_name = dist.group.WORLD.group_name
            comm = comm_class(group_name, buffer_size, self.rank)

            # Get buffer and context pointers
            buffer_ptr = comm.get_buffer_ptr()
            ctx_ptr = comm.get_context_ptr()

            # Initialize local tensor with (rank + 1) values
            local_tensor = torch.full(
                (num_elements,),
                float(self.rank + 1),
                dtype=torch.float32,
                device=device,
            )

            # Copy local data to the symmetric buffer
            import ctypes

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
            torch.cuda.synchronize(device)

            # Barrier before kernel to ensure all data is ready
            dist.barrier()

            # Launch kernel
            # Use cooperative grid for NVSHMEM barriers to work correctly
            kernel_func[(1,)](
                ctx_ptr,
                buffer_ptr,
                num_elements,
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
                msg=f"All-reduce result mismatch on rank {self.rank}",
            )

        finally:
            self._cleanup_process_group()

    @skip_if_lt_x_gpu(2)
    @requires_triton
    @requires_nccl_symm_mem
    @parametrize("num_elements", [128, 512, 2048])
    @unittest.skip("NCCL does not provide a device bitcode library (libnccl_device.bc)")
    def test_nccl_symm_all_reduce(self, num_elements: int):
        """
        Test symmetric all-reduce with NCCL backend.

        Uses NCCLSymmComm to create symmetric memory and NCCLSymmContext,
        then calls symm_all_reduce_sum_f32 via Triton kernel.

        NOTE: This test is skipped because NCCL does not provide a device
        bitcode library (libnccl_device.bc) that can be linked with Triton
        kernels. The NCCL backend implementation in symm_all_reduce.cu is
        guarded by NCCL_HAS_DEVICE_BITCODE which is set to 0.
        """
        from torch._C._distributed_c10d import NCCLSymmComm

        self._run_all_reduce_test(
            NCCLSymmComm, symm_all_reduce_test_kernel, num_elements
        )

    @skip_if_lt_x_gpu(2)
    @requires_triton
    @requires_nvshmem
    @parametrize("num_elements", [128, 512, 2048])
    def test_nvshmem_symm_all_reduce(self, num_elements: int):
        """
        Test symmetric all-reduce with NVSHMEM backend.

        Uses NVSHMEMSymmComm to create symmetric memory and NVSHMEMSymmContext,
        then calls symm_all_reduce_sum_f32 via the unified Triton kernel.
        The unified dispatcher routes to the NVSHMEM backend based on context type.
        """
        from torch._C._distributed_c10d import NVSHMEMSymmComm

        self._run_all_reduce_test(
            NVSHMEMSymmComm, symm_all_reduce_test_kernel, num_elements
        )


class TestSymmCommAvailability(TestCase):
    """Unit tests for checking communicator availability."""

    def test_nccl_symm_comm_import(self):
        """Test that NCCLSymmComm can be imported when NCCL >= 2.28 is available."""
        try:
            from torch._C._distributed_c10d import NCCLSymmComm  # noqa: F401

            self.assertTrue(True, "NCCLSymmComm imported successfully")
        except ImportError:
            self.skipTest("NCCLSymmComm not available (requires NCCL >= 2.28)")

    def test_nvshmem_symm_comm_import(self):
        """Test that NVSHMEMSymmComm can be imported when NVSHMEM is available."""
        try:
            from torch.distributed._symmetric_memory import is_nvshmem_available

            if not is_nvshmem_available():
                self.skipTest("NVSHMEM not available")

            from torch._C._distributed_c10d import NVSHMEMSymmComm  # noqa: F401

            self.assertTrue(True, "NVSHMEMSymmComm imported successfully")
        except ImportError as e:
            self.skipTest(f"NVSHMEMSymmComm not available: {e}")

    @requires_triton
    def test_symm_all_reduce_lib_finder(self):
        """Test that SymmAllReduceLibFinder can find the bitcode library."""
        from torch._extern_triton._symm_all_reduce_triton import SymmAllReduceLibFinder

        if not SymmAllReduceLibFinder.is_available():
            self.skipTest("symm_all_reduce.bc not available")

        path = SymmAllReduceLibFinder.find_device_library()
        self.assertTrue(os.path.isfile(path), f"Library not found at {path}")

    @requires_triton
    def test_nvshmem_device_lib_finder(self):
        """Test that NVSHMEM device library can be found."""
        from torch._extern_triton._symm_all_reduce_triton import SymmAllReduceLibFinder

        if not SymmAllReduceLibFinder.is_nvshmem_available():
            self.skipTest("libnvshmem_device.bc not available")

        path = SymmAllReduceLibFinder.find_nvshmem_device_library()
        self.assertTrue(os.path.isfile(path), f"Library not found at {path}")


instantiate_parametrized_tests(TestSymmAllReduceTriton)


if __name__ == "__main__":
    run_tests()
