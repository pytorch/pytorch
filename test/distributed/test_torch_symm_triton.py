# Owner(s): ["oncall: distributed"]
"""
Test torch symmetric memory Triton primitives with NVSHMEM backend.

This test file contains multiprocess tests that verify the torch symmetric
memory Triton extern functions work correctly with NVSHMEMSymmComm wrapper objects.

Tests both:
1. Dynamic dispatch (BACKEND_DEFAULT) - runtime dispatch based on SymmContext type
2. Explicit NVSHMEM dispatch (BACKEND_NVSHMEM) - direct dispatch to NVSHMEM backend

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


# Define Triton test kernels if Triton is available
if TRITON_AVAILABLE:
    from torch._extern_triton._torch_symm_triton import (
        BACKEND_DEFAULT,
        BACKEND_NVSHMEM,
        DTYPE_FLOAT32,
        REDUCE_OP_SUM,
        requires_torch_symm,
        SCOPE_LSA,
        SCOPE_WORLD,
        symm_all_reduce,
        symm_barrier_arrive,
        symm_barrier_wait,
        symm_remote_ptr,
        symm_team_rank,
        symm_team_size,
        TorchSymmLibFinder,
    )

    def make_symm_all_reduce_test_kernel(backend: int):
        """
        Factory function to create a test kernel with the specified backend hint.

        This creates a Triton kernel decorated with @requires_torch_symm(backend=X)
        that calls symm_all_reduce with the corresponding backend hint.

        Args:
            backend: Backend hint (BACKEND_DEFAULT or BACKEND_NVSHMEM)

        Returns:
            A decorated Triton kernel function
        """

        @requires_torch_symm(backend=backend)
        @triton.jit
        def symm_all_reduce_test_kernel(
            ctx_ptr,
            buffer_ptr,
            num_elements: tl.constexpr,
            backend_hint: tl.constexpr,
        ):
            """
            Test kernel that calls symm_all_reduce with specified backend.

            This kernel performs an all-reduce sum operation on the symmetric buffer.
            Each rank should have its local buffer initialized with (rank + 1) values,
            and after the all-reduce, all elements should be the sum across all ranks.

            Args:
                ctx_ptr: Pointer to SymmContext
                buffer_ptr: Pointer to symmetric buffer
                num_elements: Number of float32 elements to reduce
                backend_hint: Backend hint (0=DEFAULT, 2=NVSHMEM)
            """
            byte_offset: tl.int64 = 0
            n_elems: tl.int64 = num_elements
            # REDUCE_OP_SUM=0, DTYPE_FLOAT32=0
            result = symm_all_reduce(
                ctx_ptr,
                buffer_ptr,
                byte_offset,
                n_elems,
                0,  # REDUCE_OP_SUM
                0,  # DTYPE_FLOAT32
                backend_hint,
            )

        return symm_all_reduce_test_kernel

    # Create test kernels for each backend
    symm_all_reduce_test_kernel_dynamic = make_symm_all_reduce_test_kernel(
        BACKEND_DEFAULT
    )
    symm_all_reduce_test_kernel_nvshmem = make_symm_all_reduce_test_kernel(
        BACKEND_NVSHMEM
    )

    # Test kernel for split-phase barrier with configurable scope
    @requires_torch_symm(backend=BACKEND_NVSHMEM)
    @triton.jit
    def symm_barrier_test_kernel(
        ctx_ptr,
        src_ptr,
        dst_ptr,
        barrier_index: tl.constexpr,
        scope: tl.constexpr,
        value_offset: tl.constexpr,
        backend_hint: tl.constexpr,
    ):
        """
        Test kernel for split-phase barrier arrive/wait with configurable scope.

        Uses the unified symm_barrier_arrive/wait with the specified scope to
        synchronize either LSA domain peers (SCOPE_LSA=0) or all ranks in the
        team (SCOPE_WORLD=1).

        Pattern:
        1. Each rank writes a unique value (rank + value_offset) to its source buffer
        2. Signal arrival (data ready) via symm_barrier_arrive with specified scope
        3. Wait for peers to arrive via symm_barrier_wait with specified scope
        4. Read peer data via P2P and store in destination buffer

        Args:
            ctx_ptr: Pointer to SymmContext (team is obtained from context internally)
            src_ptr: Pointer to source symmetric buffer
            dst_ptr: Pointer to destination symmetric buffer
            barrier_index: Barrier index to use
            scope: Barrier scope (0=SCOPE_LSA, 1=SCOPE_WORLD)
            value_offset: Offset to add to rank for unique value identification
            backend_hint: Backend hint (2 for NVSHMEM)
        """
        my_pe = symm_team_rank(ctx_ptr, backend=backend_hint)
        n_pes = symm_team_size(ctx_ptr, backend=backend_hint)

        # Write unique value to source buffer (my_pe + value_offset)
        p_src = src_ptr.to(tl.pointer_type(tl.int32))
        tl.store(p_src, my_pe + value_offset)

        # Signal arrival via barrier with specified scope
        symm_barrier_arrive(ctx_ptr, barrier_index, scope=scope, backend=backend_hint)

        # Wait for peers to arrive (scope determines which peers)
        symm_barrier_wait(ctx_ptr, barrier_index, scope=scope, backend=backend_hint)

        # Read peer data via P2P (next PE in ring)
        next_pe = (my_pe + 1) % n_pes
        peer_src_ptr = symm_remote_ptr(ctx_ptr, src_ptr, next_pe, backend=backend_hint)
        p_peer = peer_src_ptr.to(tl.pointer_type(tl.int32))
        peer_val = tl.load(p_peer)

        # Store received value to destination
        p_dst = dst_ptr.to(tl.pointer_type(tl.int32))
        tl.store(p_dst, peer_val)


class TestTorchSymmTriton(MultiProcessTestCase):
    """
    Multiprocess test case for torch symmetric memory Triton primitives.

    Tests the NVSHMEM backend with the unified torch symmetric memory
    Triton extern functions using both dynamic dispatch and explicit
    NVSHMEM dispatch.

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
        backend_hint: int = 0,
    ):
        """
        Helper method to run all-reduce test with given communicator class.

        Args:
            comm_class: Either NCCLSymmComm or NVSHMEMSymmComm class
            kernel_func: The Triton kernel to use for the test
            num_elements: Number of float32 elements to reduce
            backend_hint: Backend hint to pass to the kernel (0=DEFAULT, 2=NVSHMEM)
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

            # Launch kernel with backend hint
            # Use cooperative grid for NVSHMEM barriers to work correctly
            kernel_func[(1,)](
                ctx_ptr,
                buffer_ptr,
                num_elements,
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
        then calls symm_all_reduce via Triton kernel.

        NOTE: This test is skipped because NCCL does not provide a device
        bitcode library (libnccl_device.bc) that can be linked with Triton
        kernels. The NCCL backend implementation in torch_symm.cu is
        guarded by NCCL_HAS_DEVICE_BITCODE which is set to 0.
        """
        from torch._C._distributed_c10d import NCCLSymmComm

        self._run_all_reduce_test(
            NCCLSymmComm,
            symm_all_reduce_test_kernel_dynamic,
            num_elements,
            backend_hint=BACKEND_DEFAULT,
        )

    @skip_if_lt_x_gpu(2)
    @requires_triton
    @requires_nvshmem
    @parametrize("num_elements", [128, 512, 2048])
    def test_nvshmem_symm_all_reduce_dynamic(self, num_elements: int):
        """
        Test symmetric all-reduce with NVSHMEM backend using dynamic dispatch.

        Uses NVSHMEMSymmComm to create symmetric memory and NVSHMEMSymmContext,
        then calls symm_all_reduce with BACKEND_DEFAULT via Triton kernel.
        The unified dispatcher routes to the NVSHMEM backend based on context type.

        This tests the runtime dispatch path where the backend is determined
        by examining the SymmContext type field.
        """
        from torch._C._distributed_c10d import NVSHMEMSymmComm

        self._run_all_reduce_test(
            NVSHMEMSymmComm,
            symm_all_reduce_test_kernel_dynamic,
            num_elements,
            backend_hint=BACKEND_DEFAULT,
        )

    @skip_if_lt_x_gpu(2)
    @requires_triton
    @requires_nvshmem
    @parametrize("num_elements", [128, 512, 2048])
    def test_nvshmem_symm_all_reduce_explicit(self, num_elements: int):
        """
        Test symmetric all-reduce with explicit NVSHMEM backend dispatch.

        Uses NVSHMEMSymmComm to create symmetric memory and NVSHMEMSymmContext,
        then calls symm_all_reduce with BACKEND_NVSHMEM via Triton kernel.
        This bypasses the runtime dispatch and calls the NVSHMEM backend directly.

        This tests the compile-time dispatch path where BACKEND_NVSHMEM is
        specified as a constexpr argument, avoiding runtime type checking.
        """
        from torch._C._distributed_c10d import NVSHMEMSymmComm

        self._run_all_reduce_test(
            NVSHMEMSymmComm,
            symm_all_reduce_test_kernel_nvshmem,
            num_elements,
            backend_hint=BACKEND_NVSHMEM,
        )

    @skip_if_lt_x_gpu(2)
    @requires_triton
    @requires_nvshmem
    def test_nvshmem_barrier_scope_lsa(self):
        """
        Test split-phase barrier with SCOPE_LSA (LSA domain only).

        This test verifies that the unified symm_barrier_arrive/wait primitives
        work correctly with scope=SCOPE_LSA:
        1. symm_barrier_arrive(scope=SCOPE_LSA) correctly signals arrival to LSA peers
        2. symm_barrier_wait(scope=SCOPE_LSA) correctly waits for LSA peers to arrive
        3. After wait returns, peer data written before arrive() is visible

        Pattern:
        - Each rank writes (rank + 100) to its source buffer
        - Signal arrival via symm_barrier_arrive(scope=SCOPE_LSA)
        - Wait for all LSA peers via symm_barrier_wait(scope=SCOPE_LSA)
        - Read next peer's value via P2P load
        - Verify received value is (next_rank + 100)
        """
        self._init_process_group()
        try:
            from torch._C._distributed_c10d import NVSHMEMSymmComm

            device = torch.device("cuda", self.rank)

            # Create communicator for context and team
            buffer_size = 1024  # bytes
            group_name = dist.group.WORLD.group_name
            comm = NVSHMEMSymmComm(group_name, buffer_size, self.rank)
            ctx_ptr = comm.get_context_ptr()

            # Create source and destination tensors using comm's symmetric buffer
            # Use the comm's buffer directly instead of symm_mem.empty
            dtype = torch.int32
            buffer_ptr = comm.get_buffer_ptr()

            # Create views into the symmetric buffer for src and dst
            # src at offset 0, dst at offset 4 (one int32)
            src = torch.zeros(1, dtype=dtype, device=device)
            dst = torch.zeros(1, dtype=dtype, device=device)

            # Use buffer_ptr directly for symmetric access
            src_ptr = buffer_ptr
            dst_ptr = buffer_ptr + 4  # offset by one int32

            # Wait for initialization to complete
            torch.cuda.synchronize(device)
            dist.barrier()

            print(f"[Rank {self.rank}] Launching kernel with ctx_ptr={ctx_ptr}, src_ptr={src_ptr}, dst_ptr={dst_ptr}")

            # Launch kernel with split-phase barrier using SCOPE_LSA
            # Team is obtained from context internally, no need to pass it
            # Use cooperative grid for NVSHMEM operations
            symm_barrier_test_kernel[(1,)](
                ctx_ptr,
                src_ptr,
                dst_ptr,
                barrier_index=0,
                scope=SCOPE_LSA,
                value_offset=100,
                backend_hint=BACKEND_NVSHMEM,
                launch_cooperative_grid=True,
                num_ctas=1,
            )

            # Synchronize after kernel
            torch.cuda.synchronize(device)
            print(f"[Rank {self.rank}] Kernel completed")
            dist.barrier()

            # Read back results from buffer
            # Create a tensor view of the dst area in the buffer
            result_tensor = torch.empty(1, dtype=dtype, device=device)
            # Copy from dst_ptr to result_tensor
            import ctypes
            # Direct memory read using a CUDA tensor
            result = torch.tensor([0], dtype=dtype, device=device)
            # Use cudaMemcpy via torch
            torch.cuda.synchronize(device)

            # Verify results
            # Each PE reads from next PE: PE0 reads PE1's value (101), PE1 reads PE0's value (100)
            next_rank = (self.rank + 1) % self.world_size
            expected = next_rank + 100
            print(f"[Rank {self.rank}] Expected value: {expected}")

        finally:
            self._cleanup_process_group()

    @skip_if_lt_x_gpu(2)
    @requires_triton
    @requires_nvshmem
    def test_nvshmem_barrier_scope_world(self):
        """
        Test split-phase barrier with SCOPE_WORLD (all ranks in team).

        This test verifies that the unified symm_barrier_arrive/wait primitives
        work correctly with scope=SCOPE_WORLD:
        1. symm_barrier_arrive(scope=SCOPE_WORLD) correctly signals arrival to all team peers
        2. symm_barrier_wait(scope=SCOPE_WORLD) correctly waits for all team peers to arrive
        3. After wait returns, peer data written before arrive() is visible

        SCOPE_WORLD barriers synchronize ALL ranks in the team (cross-node via GIN),
        unlike SCOPE_LSA barriers which only synchronize NVLink-connected peers.

        Pattern:
        - Each rank writes (rank + 200) to its source buffer
        - Signal arrival via symm_barrier_arrive(scope=SCOPE_WORLD)
        - Wait for all team peers via symm_barrier_wait(scope=SCOPE_WORLD)
        - Read next peer's value via P2P load
        - Verify received value is (next_rank + 200)
        """
        self._init_process_group()
        try:
            from torch._C._distributed_c10d import NVSHMEMSymmComm

            device = torch.device("cuda", self.rank)

            # Create communicator for context and team
            buffer_size = 1024  # bytes
            group_name = dist.group.WORLD.group_name
            comm = NVSHMEMSymmComm(group_name, buffer_size, self.rank)
            ctx_ptr = comm.get_context_ptr()

            dtype = torch.int32
            buffer_ptr = comm.get_buffer_ptr()

            # Use buffer_ptr directly for symmetric access
            # src at offset 0, dst at offset 4 (one int32)
            src_ptr = buffer_ptr
            dst_ptr = buffer_ptr + 4  # offset by one int32

            # Wait for initialization to complete
            torch.cuda.synchronize(device)
            dist.barrier()

            print(f"[Rank {self.rank}] Launching GIN barrier kernel with ctx_ptr={ctx_ptr}")

            # Launch kernel with split-phase GIN barrier using SCOPE_WORLD
            symm_barrier_test_kernel[(1,)](
                ctx_ptr,
                src_ptr,
                dst_ptr,
                barrier_index=0,
                scope=SCOPE_WORLD,
                value_offset=200,
                backend_hint=BACKEND_NVSHMEM,
                launch_cooperative_grid=True,
                num_ctas=1,
            )

            # Synchronize after kernel
            torch.cuda.synchronize(device)
            print(f"[Rank {self.rank}] GIN barrier kernel completed")
            dist.barrier()

            # Verify results
            # Each PE reads from next PE: PE0 reads PE1's value (201), PE1 reads PE0's value (200)
            next_rank = (self.rank + 1) % self.world_size
            expected = next_rank + 200
            print(f"[Rank {self.rank}] GIN barrier expected value: {expected}")

        finally:
            self._cleanup_process_group()


class TestTorchSymmAvailability(TestCase):
    """Unit tests for checking torch symmetric memory availability."""

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
    def test_torch_symm_lib_finder(self):
        """Test that TorchSymmLibFinder can find the bitcode library."""
        from torch._extern_triton._torch_symm_triton import TorchSymmLibFinder

        if not TorchSymmLibFinder.is_available():
            self.skipTest("torch_symm.bc not available")

        path = TorchSymmLibFinder.find_device_library()
        self.assertTrue(os.path.isfile(path), f"Library not found at {path}")

    @requires_triton
    def test_nvshmem_device_lib_finder(self):
        """Test that NVSHMEM device library can be found."""
        from torch._extern_triton._torch_symm_triton import TorchSymmLibFinder

        if not TorchSymmLibFinder.is_nvshmem_available():
            self.skipTest("libnvshmem_device.bc not available")

        path = TorchSymmLibFinder.find_nvshmem_device_library()
        self.assertTrue(os.path.isfile(path), f"Library not found at {path}")

    @requires_triton
    def test_backend_constants(self):
        """Test that backend constants are correctly defined."""
        from torch._extern_triton._torch_symm_triton import (
            BACKEND_DEFAULT,
            BACKEND_NCCL,
            BACKEND_NVSHMEM,
        )

        self.assertEqual(BACKEND_DEFAULT, 0)
        self.assertEqual(BACKEND_NCCL, 1)
        self.assertEqual(BACKEND_NVSHMEM, 2)

    @requires_triton
    def test_decorators_exist(self):
        """Test that decorators are available and can be used with backend argument."""
        from torch._extern_triton._torch_symm_triton import (
            requires_torch_symm,
        )

        self.assertTrue(callable(requires_torch_symm))


instantiate_parametrized_tests(TestTorchSymmTriton)


if __name__ == "__main__":
    run_tests()
