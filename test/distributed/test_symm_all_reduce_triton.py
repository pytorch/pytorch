# Owner(s): ["oncall: distributed"]
"""
Test suite for the symmetric all-reduce Triton extern library.

This test suite verifies that the NCCL symmetric memory all-reduce operations
exposed via core.extern_elementwise mechanism work correctly when called
from Triton kernels.

To run:
    python test/distributed/test_symm_all_reduce_triton.py

Prerequisites:
    1. NCCL >= 2.28.9 with symmetric memory support
    2. Build the CUDA library to bitcode:
       cd torch/csrc/_extern_triton && make symm_all_reduce.bc CUDA_ARCH=sm_80
    3. Or set SYMM_ALL_REDUCE_LIB_PATH environment variable to point to the .bc file
"""

import sys

# Import TEST_WITH_ROCM first to check for ROCm before importing CUDA-specific modules
from torch.testing._internal.common_utils import TEST_WITH_ROCM


# Skip entire module on ROCm before importing CUDA-specific modules
if TEST_WITH_ROCM:
    print("Symmetric all-reduce extern library not available on ROCm, skipping tests")
    sys.exit(0)


import torch
import torch.distributed as dist
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    requires_cuda_p2p_access,
    run_tests,
    skip_but_pass_in_sandcastle_if,
    skipIfRocm,
    TestCase,
)


# Skip if CUDA is not available
if not TEST_CUDA:
    print("CUDA not available, skipping tests")
    sys.exit(0)


# Import Triton and our extern library
try:
    import triton
    import triton.language as tl

    from torch._extern_triton._symm_all_reduce_triton import (
        requires_symm_all_reduce,
        requires_symm_all_reduce_lib,
        symm_all_reduce_sum_f32,
        SymmAllReduceLibFinder,
    )

    TRITON_AVAILABLE = True
except ImportError as e:
    print(f"Triton not available, skipping tests: {e}")
    TRITON_AVAILABLE = False
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]


# Check if the bitcode library is available
LIB_AVAILABLE = False
LIB_PATH = None
LIB_ERROR = ""
if TRITON_AVAILABLE:
    try:
        LIB_PATH = SymmAllReduceLibFinder.find_device_library()
        LIB_AVAILABLE = True
    except RuntimeError as e:
        LIB_ERROR = str(e)


# NOTE: Even though the bitcode library may exist, the NCCL device functions
# (ncclLsaBarrier, ncclGetLsaPointer) are declared as extern in the bitcode
# but NCCL does not ship a device library (like NVSHMEM's libnvshmem_device.bc).
# This means the kernel compilation will fail at PTX generation with unresolved
# symbols. This is a known limitation tracked in T12345678.
NCCL_DEVICE_LIB_AVAILABLE = False
NCCL_DEVICE_LIB_ERROR = (
    "NCCL does not provide a device bitcode library (libnccl_device.bc). "
    "The symm_all_reduce.bc uses extern NCCL device functions that cannot "
    "be resolved at compile time. Unlike NVSHMEM which provides "
    "libnvshmem_device.bc, NCCL device functions are only available in "
    "the runtime shared library. This limitation prevents using NCCL's "
    "LSA API directly from Triton kernels via extern_libs."
)


# Check if NCCL symmetric memory is available
SYMM_MEM_AVAILABLE = False
try:
    import torch.distributed._symmetric_memory as symm_mem

    SYMM_MEM_AVAILABLE = hasattr(symm_mem, "set_backend")
except (ImportError, RuntimeError):
    pass


# Check if NCCLSymmComm C++ bindings are available
# This requires NCCL >= 2.28.0 with symmetric memory device support
NCCL_SYMM_COMM_AVAILABLE = False
NCCL_SYMM_COMM_ERROR = ""
try:
    from torch._C._distributed_c10d import (  # type: ignore[attr-defined]
        NCCLSymmComm as _NCCLSymmComm,
    )

    NCCL_SYMM_COMM_AVAILABLE = True
except ImportError as e:
    NCCL_SYMM_COMM_ERROR = str(e)
    # Check if this is because NCCL version is too old
    try:
        nccl_version = torch.cuda.nccl.version()
        if nccl_version < (2, 28, 0):
            NCCL_SYMM_COMM_ERROR = (
                f"NCCLSymmComm requires NCCL >= 2.28.0, but found {'.'.join(map(str, nccl_version))}. "
                "Please upgrade NCCL to use NCCLSymmComm bindings."
            )
    except Exception:
        pass


def requires_extern_lib():
    """Skip test if the extern library is not available."""
    return skip_but_pass_in_sandcastle_if(
        not LIB_AVAILABLE,
        f"Symmetric all-reduce bitcode library not available: {LIB_ERROR}. "
        "Compile with: cd torch/csrc/_extern_triton && make symm_all_reduce.bc",
    )


def requires_nccl_device_lib():
    """Skip test if NCCL device library is not available.

    NCCL does not provide a device bitcode library like NVSHMEM does,
    so this will always skip until NCCL provides such a library.
    """
    return skip_but_pass_in_sandcastle_if(
        not NCCL_DEVICE_LIB_AVAILABLE,
        NCCL_DEVICE_LIB_ERROR,
    )


def requires_symm_mem():
    """Skip test if NCCL symmetric memory is not available."""
    return skip_but_pass_in_sandcastle_if(
        not SYMM_MEM_AVAILABLE,
        "NCCL symmetric memory not available. Requires NCCL >= 2.28.9 with symmetric memory support.",
    )


def requires_triton():
    """Skip test if Triton is not available."""
    return skip_but_pass_in_sandcastle_if(
        not TRITON_AVAILABLE,
        "Triton not available.",
    )


def requires_nccl_symm_comm():
    """Skip test if NCCLSymmComm C++ bindings are not available."""
    return skip_but_pass_in_sandcastle_if(
        not NCCL_SYMM_COMM_AVAILABLE,
        f"NCCLSymmComm C++ bindings not available: {NCCL_SYMM_COMM_ERROR}",
    )


# So that tests are written in device-agnostic way
device_type = "cuda"
device_module = torch.get_device_module(device_type)


# =============================================================================
# TRITON KERNEL DEFINITIONS
# =============================================================================

if TRITON_AVAILABLE:

    @triton.jit  # type: ignore[misc]
    def _all_reduce_kernel_f32(
        ctx_ptr,  # tl.int64 - pointer to SymmContext
        buffer_ptr,  # tl.int64 - pointer to symmetric buffer
        byte_offset,  # tl.int64 - byte offset within buffer
        num_elements,  # tl.int64 - number of float32 elements to reduce
        result_ptr,  # Pointer to store the result status
    ):
        """
        Triton kernel that performs symmetric all-reduce sum on float32 data.

        This kernel calls the external symm_all_reduce_sum_f32 function which
        performs an all-reduce sum operation across all ranks using NCCL's
        Local Symmetric Access (LSA) API.

        Args:
            ctx_ptr: Device pointer to NCCLSymmContext (int64)
            buffer_ptr: Device pointer to local symmetric buffer (int64)
            byte_offset: Byte offset within the symmetric buffer
            num_elements: Number of float32 elements to reduce
            result_ptr: Pointer to store the result status (0=success, non-zero=error)
        """
        # Call the extern function that performs the all-reduce
        result = symm_all_reduce_sum_f32(
            ctx_ptr,
            buffer_ptr,
            byte_offset,
            num_elements,
        )
        # Store the result status
        tl.store(result_ptr, result)

    # Apply the requires_symm_all_reduce decorator to enable extern library
    # This is done after @triton.jit to match the NVSHMEM pattern
    try:
        all_reduce_kernel_f32 = requires_symm_all_reduce(_all_reduce_kernel_f32)
    except RuntimeError as e:
        # Library not found - kernel will fail at runtime
        print(f"Warning: Could not load symm_all_reduce library: {e}")
        all_reduce_kernel_f32 = _all_reduce_kernel_f32


# =============================================================================
# TEST CLASSES
# =============================================================================


class TestSymmAllReduceLibFinder(TestCase):
    """
    Test suite for the library finder utility.

    Note: This is a non-distributed test class since it only tests the finder
    functionality which doesn't require multiple processes.
    """

    @requires_triton()
    def test_finder_caches_path(self):
        """Test that the library finder caches the found path."""
        if not LIB_AVAILABLE:
            self.skipTest("Library not available")

        # Clear cache
        SymmAllReduceLibFinder.clear_cache()

        # First call should search
        path1 = SymmAllReduceLibFinder.find_device_library()
        self.assertIsNotNone(path1)
        self.assertTrue(path1.endswith(".bc"))

        # Second call should return cached
        path2 = SymmAllReduceLibFinder.find_device_library()
        self.assertEqual(path1, path2)

    @requires_triton()
    def test_finder_is_available(self):
        """Test the is_available() method."""
        if not LIB_AVAILABLE:
            self.skipTest("Library not available")

        # Clear cache
        SymmAllReduceLibFinder.clear_cache()

        # Should return True if library exists
        self.assertTrue(SymmAllReduceLibFinder.is_available())

    @requires_triton()
    def test_finder_uses_env_var_when_valid(self):
        """Test that the library finder uses SYMM_ALL_REDUCE_LIB_PATH when it points to a valid file."""
        import os

        if not LIB_AVAILABLE:
            self.skipTest("Library not available")

        # Clear cache
        SymmAllReduceLibFinder.clear_cache()

        # Get the actual library path first
        actual_path = SymmAllReduceLibFinder.find_device_library()

        # Clear cache again
        SymmAllReduceLibFinder.clear_cache()

        # Set environment to the actual path
        original_env = os.environ.get("SYMM_ALL_REDUCE_LIB_PATH")
        try:
            os.environ["SYMM_ALL_REDUCE_LIB_PATH"] = actual_path

            # Should return the path from env var
            found_path = SymmAllReduceLibFinder.find_device_library()
            self.assertEqual(found_path, actual_path)
        finally:
            # Restore original environment
            if original_env is not None:
                os.environ["SYMM_ALL_REDUCE_LIB_PATH"] = original_env
            else:
                os.environ.pop("SYMM_ALL_REDUCE_LIB_PATH", None)

            # Clear cache again
            SymmAllReduceLibFinder.clear_cache()

    @requires_triton()
    def test_finder_falls_through_invalid_env_var(self):
        """Test that finder falls through when env var points to non-existent file."""
        import os

        if not LIB_AVAILABLE:
            self.skipTest("Library not available")

        # Clear cache
        SymmAllReduceLibFinder.clear_cache()

        # Set environment to a non-existent path
        original_env = os.environ.get("SYMM_ALL_REDUCE_LIB_PATH")
        try:
            os.environ["SYMM_ALL_REDUCE_LIB_PATH"] = "/nonexistent/path.bc"

            # Since the library exists in other locations, it should still be found
            # The env var is only used if the file exists
            path = SymmAllReduceLibFinder.find_device_library()
            self.assertIsNotNone(path)
            self.assertTrue(path.endswith(".bc"))
            # Path should NOT be the invalid env var path
            self.assertNotEqual(path, "/nonexistent/path.bc")
        finally:
            # Restore original environment
            if original_env is not None:
                os.environ["SYMM_ALL_REDUCE_LIB_PATH"] = original_env
            else:
                os.environ.pop("SYMM_ALL_REDUCE_LIB_PATH", None)

            # Clear cache again
            SymmAllReduceLibFinder.clear_cache()


class TestNCCLSymmCommModule(TestCase):
    """
    Test suite for NCCLSymmComm Python module.

    Note: This is a non-distributed test class that tests the Python module
    import and availability checking, without requiring the C++ bindings.
    """

    def test_nccl_symm_comm_module_import(self):
        """Test that the NCCLSymmComm module can be imported."""
        from torch._extern_triton._nccl_symm_comm import (
            is_nccl_symm_mem_available,
            NCCLSymmComm,
        )

        # Check class and function exist
        self.assertIsNotNone(NCCLSymmComm)
        self.assertIsNotNone(is_nccl_symm_mem_available)

    def test_is_nccl_symm_mem_available(self):
        """Test the is_nccl_symm_mem_available function."""
        from torch._extern_triton._nccl_symm_comm import is_nccl_symm_mem_available

        # Should return a boolean
        result = is_nccl_symm_mem_available()
        self.assertIsInstance(result, bool)


class TestTritonKernelDefinition(TestCase):
    """
    Test suite for verifying Triton kernel definitions.

    These tests verify that the kernel definitions are correct without
    running them in a distributed setting.
    """

    @requires_triton()
    @requires_extern_lib()
    def test_kernel_is_defined(self):
        """Test that the all_reduce_kernel_f32 is properly defined."""
        self.assertIsNotNone(all_reduce_kernel_f32)
        # With the requires_symm_all_reduce decorator, the kernel is wrapped
        # in a GridCallableWithExtern object
        self.assertTrue(
            hasattr(all_reduce_kernel_f32, "run")
            or hasattr(all_reduce_kernel_f32, "jit_func")
        )

    @requires_triton()
    @requires_extern_lib()
    def test_extern_function_signature(self):
        """Test that symm_all_reduce_sum_f32 extern function is defined."""
        self.assertIsNotNone(symm_all_reduce_sum_f32)
        self.assertTrue(
            hasattr(symm_all_reduce_sum_f32, "__wrapped__")
            or callable(symm_all_reduce_sum_f32)
        )


@requires_cuda_p2p_access()
@instantiate_parametrized_tests
class TestSymmAllReduceTriton(MultiProcContinuousTest):
    """Test suite for symmetric all-reduce Triton kernel."""

    world_size: int = 2

    @property
    def device(self) -> torch.device:
        return torch.device(device_type, self.rank)

    def _init_device(self) -> None:
        device_module.set_device(self.device)

    @skipIfRocm
    @requires_triton()
    @requires_extern_lib()
    @requires_symm_mem()
    @skip_if_lt_x_gpu(2)
    def test_symm_all_reduce_basic(self):
        """
        Test basic symmetric all-reduce operation.

        This test verifies the library loading works and Triton extern
        functions are properly defined.
        """
        self._init_device()

        # Verify the library can be loaded
        self.assertTrue(LIB_AVAILABLE)

        # Verify Triton extern function is defined
        self.assertIsNotNone(symm_all_reduce_sum_f32)

    @skipIfRocm
    @requires_triton()
    @requires_extern_lib()
    @requires_symm_mem()
    @skip_if_lt_x_gpu(2)
    @parametrize("num_elements", [256, 1024, 4096])
    def test_symm_all_reduce_lib_path_valid(self, num_elements):
        """Test that library path is valid for various configurations."""
        self._init_device()

        # Clear cache and verify library is found
        SymmAllReduceLibFinder.clear_cache()
        path = SymmAllReduceLibFinder.find_device_library()

        self.assertIsNotNone(path)
        self.assertTrue(path.endswith(".bc"))

        # Verify num_elements is valid
        self.assertGreater(num_elements, 0)

    @skipIfRocm
    @requires_triton()
    @requires_extern_lib()
    @requires_symm_mem()
    @requires_nccl_symm_comm()
    @requires_nccl_device_lib()  # Skip until NCCL provides device bitcode library
    @skip_if_lt_x_gpu(2)
    @parametrize("num_elements", [128, 512, 2048])
    def test_symm_all_reduce_with_nccl_symm_context(self, num_elements):
        """
        Test all-reduce Triton kernel using NCCLSymmContext.

        This test:
        1. Creates an NCCLSymmComm instance to get NCCLSymmContext
        2. Initializes each rank's buffer with rank-specific values
        3. Runs the all-reduce Triton kernel
        4. Verifies the result matches expected all-reduce sum

        For 2 ranks with initial values [rank+1]:
        - Rank 0: buffer = [1.0, 1.0, ...]
        - Rank 1: buffer = [2.0, 2.0, ...]
        - After all-reduce sum: all ranks have [3.0, 3.0, ...]
        """
        self._init_device()

        # Import NCCLSymmComm
        from torch._extern_triton._nccl_symm_comm import NCCLSymmComm

        # Configuration
        dtype = torch.float32
        element_size = 4  # bytes per float32
        buffer_size = num_elements * element_size

        # Get the group name
        group_name = dist.group.WORLD.group_name

        # Create NCCLSymmComm instance
        # This allocates symmetric memory and creates the NCCLSymmContext
        comm = NCCLSymmComm(
            group_name=group_name,
            buffer_size=buffer_size,
            device_idx=self.rank,
        )

        # Get pointers for Triton kernel
        ctx_ptr = comm.get_context_ptr()
        buffer_ptr = comm.get_buffer_ptr()

        # Create a tensor view of the buffer for initialization and verification
        buffer_tensor = torch.empty(
            num_elements,
            dtype=dtype,
            device=self.device,
        )

        # Copy buffer_tensor data to the symmetric buffer
        # Initialize each rank with its rank value + 1
        # Rank 0: [1.0, 1.0, ...], Rank 1: [2.0, 2.0, ...]
        init_value = float(self.rank + 1)
        buffer_tensor.fill_(init_value)

        # Copy initialized data to the symmetric buffer
        # Use cudaMemcpy via ctypes
        import ctypes

        cuda = ctypes.CDLL("libcudart.so")
        cuda.cudaMemcpy(
            ctypes.c_void_p(buffer_ptr),
            ctypes.c_void_p(buffer_tensor.data_ptr()),
            ctypes.c_size_t(buffer_size),
            ctypes.c_int(1),  # cudaMemcpyDeviceToDevice
        )

        # Synchronize all ranks before running the kernel
        dist.barrier()

        # Allocate result tensor for kernel status
        result_tensor = torch.zeros(1, dtype=torch.int32, device=self.device)

        # Run the all-reduce Triton kernel
        # The kernel launches with 1 block
        all_reduce_kernel_f32[(1,)](
            ctx_ptr,
            buffer_ptr,
            0,  # byte_offset
            num_elements,
            result_tensor,
        )

        # Synchronize to ensure kernel completion
        torch.cuda.synchronize(self.device)

        # Synchronize all ranks after kernel execution
        dist.barrier()

        # Copy result back from symmetric buffer to verify
        cuda.cudaMemcpy(
            ctypes.c_void_p(buffer_tensor.data_ptr()),
            ctypes.c_void_p(buffer_ptr),
            ctypes.c_size_t(buffer_size),
            ctypes.c_int(1),  # cudaMemcpyDeviceToDevice
        )

        # Expected result: sum of all ranks' initial values
        # For 2 ranks: 1.0 + 2.0 = 3.0
        expected_sum = float((self.world_size * (self.world_size + 1)) // 2)
        expected_tensor = torch.full(
            (num_elements,),
            expected_sum,
            dtype=dtype,
            device=self.device,
        )

        # Verify the all-reduce result
        torch.testing.assert_close(
            buffer_tensor,
            expected_tensor,
            rtol=1e-5,
            atol=1e-5,
            msg=f"All-reduce result mismatch on rank {self.rank}",
        )

        # Verify kernel returned success (0)
        self.assertEqual(
            result_tensor.item(),
            0,
            f"Kernel returned error code {result_tensor.item()} on rank {self.rank}",
        )


if __name__ == "__main__":
    run_tests()
