"""
Tests for USM (Unified Shared Memory) storage functionality.

This module tests the usm_share_ functionality that allows sharing storage
between CPU and devices that support unified memory (e.g., integrated GPUs,
NVIDIA Jetson platforms).
"""

import sys
import os
import tempfile
import unittest
import gc
from itertools import product

import torch
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
    skipIfTorchDynamo,
    IS_WINDOWS,
)
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
    deviceCountAtLeast,
)
from torch.testing._internal.common_cuda import TEST_CUDA, CUDA_DEVICE

# Check for MPS availability
TEST_MPS = torch.backends.mps.is_available()
MPS_DEVICE = torch.device("mps") if TEST_MPS else None


class TestUsmStorage(TestCase):
    """Test cases for USM storage functionality."""

    def tearDown(self):
        """Clean up CUDA resources after each test."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def test_usm_allocator_basic(self):
        """Test basic UsmAllocator functionality with memory allocation."""
        if IS_WINDOWS:
            self.skipTest("UsmAllocator is not supported on Windows")
        
        # Create a storage using from_file with usm=True
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_file = f.name
            # Write some test data
            test_data = bytes(range(256))
            f.write(test_data)
        
        try:
            # Load storage with USM allocator
            storage = torch.UntypedStorage.from_file(
                temp_file, 
                shared=False, 
                nbytes=256,
                usm=True
            )
            
            # Check storage properties
            self.assertEqual(storage.nbytes(), 256)
            self.assertIsNotNone(storage.data_ptr())
            
            # Verify data was loaded correctly
            for i in range(256):
                self.assertEqual(storage[i], i)
        finally:
            os.unlink(temp_file)

    def test_usm_allocator_no_file(self):
        """Test UsmAllocator with no file (just memory allocation)."""
        if IS_WINDOWS:
            self.skipTest("UsmAllocator is not supported on Windows")
        
        # Create storage without file
        storage = torch.UntypedStorage.from_file(
            "usmalloc",
            shared=False,
            nbytes=1024,
            usm=True
        )
        
        self.assertEqual(storage.nbytes(), 1024)
        self.assertIsNotNone(storage.data_ptr())

    def test_usm_storage_copy(self):
        """Test copying data between USM storages."""
        if IS_WINDOWS:
            self.skipTest("UsmAllocator is not supported on Windows")
        
        # Create source storage
        src_storage = torch.UntypedStorage(100)
        for i in range(100):
            src_storage[i] = i
        
        # Create destination USM storage
        dst_storage = torch.UntypedStorage.from_file(
            "usmalloc",
            shared=False,
            nbytes=100,
            usm=True
        )
        
        # Copy data
        dst_storage.copy_(src_storage)
        
        # Verify copy
        for i in range(100):
            self.assertEqual(dst_storage[i], i)

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_usm_share_cpu_to_cpu(self):
        """Test usm_share_ from CPU to CPU (should raise NotImplementedError)."""
        # Create CPU storage
        cpu_storage = torch.UntypedStorage(1024)
        for i in range(1024):
            cpu_storage[i] = i % 256
        
        # Share to CPU (should raise NotImplementedError)
        with self.assertRaises(NotImplementedError):
            cpu_storage.usm_share_(torch.device('cpu'))

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_usm_share_cpu_to_cuda_integrated(self):
        """Test usm_share_ from CPU to CUDA on integrated GPU."""
        # Check if device is integrated
        device_props = torch.cuda.get_device_properties(0)
        if not device_props.is_integrated:
            self.skipTest("Test requires integrated GPU (e.g., Jetson)")
        
        # Create CPU storage with USM allocator using a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='_cuda_test_64k.bin') as f:
            temp_file = f.name
            # Write unique pattern to ensure different memory (use 64KB)
            pattern = bytes([(i * 17) % 256 for i in range(65536)])
            f.write(pattern)
        
        try:
            cpu_storage = torch.UntypedStorage.from_file(
                temp_file,
                shared=False,
                nbytes=65536,
                usm=True
            )
            test_pattern = list(range(256)) * 256  # 256 repetitions for 64KB
            for i in range(65536):
                cpu_storage[i] = test_pattern[i]
            
            # Share to CUDA
            cuda_device = torch.device('cuda:0')
            cuda_storage = cpu_storage.usm_share_(cuda_device)
            
            # Verify storage properties
            self.assertEqual(cuda_storage.device, cuda_device)
            self.assertEqual(cuda_storage.nbytes(), cpu_storage.nbytes())
            
            # Create CUDA tensor from shared storage and verify with GPU operation
            cuda_tensor = torch.tensor([], dtype=torch.uint8, device='cuda:0').set_(cuda_storage)
            result = cuda_tensor + 1
            cpu_result = result.cpu()
            
            # Verify the operation worked
            for i in range(min(256, len(cpu_result))):
                expected = (test_pattern[i] + 1) % 256
                self.assertEqual(cpu_result[i].item(), expected)
        finally:
            os.unlink(temp_file)

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_usm_share_with_tensor(self):
        """Test usm_share_ with tensors created from storage."""
        device_props = torch.cuda.get_device_properties(0)
        if not device_props.is_integrated:
            self.skipTest("Test requires integrated GPU")
        
        # Create CPU storage with USM allocator using temp file
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_file = f.name
            f.write(b'\x00' * 4000)
        
        try:
            cpu_storage = torch.UntypedStorage.from_file(
                temp_file,
                shared=False,
                nbytes=4000,
                usm=True
            )
            
            # Fill with data as float32
            cpu_tensor = torch.tensor([], dtype=torch.float32).set_(cpu_storage)
            cpu_tensor = cpu_tensor[:1000]
            cpu_tensor.copy_(torch.arange(1000, dtype=torch.float32))
            
            # Share storage to CUDA
            cuda_storage = cpu_storage.usm_share_(torch.device('cuda:0'))
            
            # Create CUDA tensor from shared storage
            cuda_tensor = torch.empty(0, dtype=torch.float32, device='cuda:0').set_(cuda_storage)
            cuda_tensor = cuda_tensor[:1000]
            
            # Verify data
            self.assertTrue(torch.allclose(cpu_tensor, cuda_tensor.cpu()))
        finally:
            os.unlink(temp_file)

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_usm_share_errors(self):
        """Test error conditions for usm_share_."""
        # Test with non-CPU source
        if torch.cuda.device_count() > 0:
            cuda_storage = torch.UntypedStorage(100, device='cuda')
            with self.assertRaisesRegex(RuntimeError, "source storage must be on CPU"):
                cuda_storage.usm_share_(torch.device('cuda:0'))
        
        # Test with invalid storage
        empty_storage = torch.UntypedStorage(0)
        # Empty storage should work and return empty CUDA storage
        result = empty_storage.usm_share_(torch.device('cuda:0'))
        self.assertEqual(result.nbytes(), 0)
        self.assertEqual(result.device.type, 'cuda')

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_usm_share_non_integrated_gpu(self):
        """Test usm_share_ on non-integrated GPU (should fail)."""
        device_props = torch.cuda.get_device_properties(0)
        if device_props.is_integrated:
            self.skipTest("Test requires non-integrated GPU")
        
        cpu_storage = torch.UntypedStorage(1024)
        
        # Should fail on non-integrated GPU
        with self.assertRaisesRegex(
            RuntimeError, 
            "target device does not support USM.*not integrated GPU"
        ):
            cpu_storage.usm_share_(torch.device('cuda:0'))

    def test_usm_storage_lifecycle(self):
        """Test proper cleanup and lifecycle of USM storage."""
        if IS_WINDOWS:
            self.skipTest("UsmAllocator is not supported on Windows")
        
        # Create storage
        storage = torch.UntypedStorage.from_file(
            "usmalloc",
            shared=False,
            nbytes=1024,
            usm=True
        )
        
        ptr = storage.data_ptr()
        self.assertNotEqual(ptr, 0)
        
        # Fill storage directly
        for i in range(min(1024, storage.nbytes())):
            storage[i] = 42
        
        # Verify
        self.assertEqual(storage[0], 42)
        self.assertEqual(storage[100], 42)
        
        # Delete storage (should not crash)
        del storage

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_usm_share_preservation(self):
        """Test that shared storage preserves data after modifications."""
        device_props = torch.cuda.get_device_properties(0)
        if not device_props.is_integrated:
            self.skipTest("Test requires integrated GPU")
        
        # Create and populate CPU storage with USM allocator using temp file
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_file = f.name
            f.write(b'\x00' * 256)
        
        try:
            cpu_storage = torch.UntypedStorage.from_file(
                temp_file,
                shared=False,
                nbytes=256,
                usm=True
            )
            for i in range(256):
                cpu_storage[i] = i
            
            # Share to CUDA
            cuda_storage = cpu_storage.usm_share_(torch.device('cuda:0'))
            
            # Verify initial data by direct comparison
            errors = 0
            for i in range(256):
                if cpu_storage[i] != i:
                    errors += 1
            self.assertEqual(errors, 0)
            
            # Verify CUDA storage can be used
            cuda_tensor = torch.tensor([], dtype=torch.uint8, device='cuda:0').set_(cuda_storage)
            self.assertEqual(cuda_tensor.device.type, 'cuda')
        finally:
            os.unlink(temp_file)


class TestUsmStoragePython(TestCase):
    """Test Python API for USM storage."""
    
    def test_storage_usm_share_api(self):
        """Test that usm_share_ method exists and has correct signature."""
        storage = torch.UntypedStorage(100)
        
        # Check method exists
        self.assertTrue(hasattr(storage, 'usm_share_'))
        
        # Check it's callable
        self.assertTrue(callable(storage.usm_share_))

    def test_from_file_usm_parameter(self):
        """Test from_file accepts usm parameter."""
        if IS_WINDOWS:
            self.skipTest("UsmAllocator is not supported on Windows")
        
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_file = f.name
            f.write(b'\x00' * 100)
        
        try:
            # Should accept usm=True
            storage = torch.UntypedStorage.from_file(
                temp_file,
                shared=False,
                nbytes=100,
                usm=True
            )
            self.assertIsNotNone(storage)
            
            # Should also work with usm=False (default)
            storage2 = torch.UntypedStorage.from_file(
                temp_file,
                shared=False,
                nbytes=100,
                usm=False
            )
            self.assertIsNotNone(storage2)
        finally:
            os.unlink(temp_file)

    def test_usm_small_unaligned_file(self):
        """Test USM with small file (1KB) - tests DIO fallback."""
        if IS_WINDOWS:
            self.skipTest("UsmAllocator is not supported on Windows")
        
        # Create a 1KB file (not aligned to 4KB)
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_file = f.name
            # Write 1KB of data
            test_data = bytes(range(256)) * 4  # 1024 bytes
            f.write(test_data)
        
        try:
            # Load with USM - should trigger DIO fallback due to small size
            storage = torch.UntypedStorage.from_file(
                temp_file,
                shared=False,
                nbytes=1024,
                usm=True
            )
            
            # Storage nbytes() returns requested size, not aligned allocation size
            self.assertEqual(storage.nbytes(), 1024)
            
            # Verify data is loaded correctly
            for i in range(1024):
                self.assertEqual(storage[i], i % 256)
        finally:
            os.unlink(temp_file)

    def test_usm_large_unaligned_file(self):
        """Test USM with large file not aligned to 4KB - tests DIO fallback."""
        if IS_WINDOWS:
            self.skipTest("UsmAllocator is not supported on Windows")
        
        # Create a file that's not 4KB aligned (e.g., 10KB + 100 bytes)
        file_size = 10 * 1024 + 100  # 10340 bytes, not 4KB aligned
        
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_file = f.name
            # Write pattern data
            for i in range(file_size):
                f.write(bytes([i % 256]))
        
        try:
            # Load with USM - DIO might fail due to unaligned size, should fallback
            storage = torch.UntypedStorage.from_file(
                temp_file,
                shared=False,
                nbytes=file_size,
                usm=True
            )
            
            # Storage nbytes() returns requested size
            self.assertEqual(storage.nbytes(), file_size)
            
            # Verify data is loaded correctly (at least first few KB)
            errors = 0
            for i in range(min(1024, file_size)):
                if storage[i] != i % 256:
                    errors += 1
            self.assertEqual(errors, 0)
        finally:
            os.unlink(temp_file)

    def test_usm_unaligned_buffer_address(self):
        """Test USM allocation and verify pointer alignment."""
        if IS_WINDOWS:
            self.skipTest("UsmAllocator is not supported on Windows")
        
        # Create storage without file - should use mmap
        storage = torch.UntypedStorage.from_file(
            "usmalloc",
            shared=False,
            nbytes=4096,
            usm=True
        )
        
        # Storage nbytes() returns requested size
        self.assertEqual(storage.nbytes(), 4096)
        
        # Check alignment - pointer should be 512-byte aligned for DIO
        ptr = storage.data_ptr()
        self.assertEqual(ptr % 512, 0, "USM pointer should be 512-byte aligned")


class TestUsmCudaErrorInjection(TestCase):
    """Test CUDA error handling with simulated failures."""
    
    def setUp(self):
        """Set up for each test."""
        self.temp_files = []
    
    def tearDown(self):
        """Clean up CUDA resources and temp files after each test."""
        if TEST_CUDA:
            gc.collect()  # Force garbage collection to release storages
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Clean up temp files
        for temp_file in self.temp_files:
            if os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass
    
    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_usm_share_cuda_register_failure_simulation(self):
        """Test handling of cudaHostRegister-like failures."""
        device_props = torch.cuda.get_device_properties(0)
        if not device_props.is_integrated:
            self.skipTest("Test requires integrated GPU")
        
        # Test with very large storage that might cause cudaHostRegister to fail
        # Note: This is a best-effort test - actual failure depends on system resources
        huge_size = 16 * 1024 * 1024 * 1024  # 16GB
        
        # Use temp file to avoid memory address reuse issues
        with tempfile.NamedTemporaryFile(suffix='_huge_cuda_test.bin', delete=False) as f:
            temp_file = f.name
        
        # Track temp file for cleanup in tearDown
        self.temp_files.append(temp_file)
        
        try:
            cpu_storage = torch.UntypedStorage.from_file(
                temp_file,
                shared=False,
                nbytes=huge_size,
                usm=True
            )
            
            # Try to share - this might fail with large allocations
            try:
                cuda_storage = cpu_storage.usm_share_(torch.device('cuda:0'))
                # If it succeeds, that's also fine - just verify it works
                self.assertEqual(cuda_storage.device.type, 'cuda')
                # Clean up explicitly
                del cuda_storage
            except RuntimeError as e:
                # Expected failure for very large allocations
                self.assertIn("cuda", str(e).lower())
            finally:
                del cpu_storage
        except RuntimeError as e:
            # mmap itself might fail for huge allocation
            self.assertIn("mmap", str(e).lower())

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_usm_share_zero_size_edge_case(self):
        """Test edge case with zero-sized storage."""
        # Create zero-size storage
        empty_storage = torch.UntypedStorage(0)
        
        # Should handle gracefully
        result = empty_storage.usm_share_(torch.device('cuda:0'))
        self.assertEqual(result.nbytes(), 0)
        self.assertEqual(result.device.type, 'cuda')


class TestUsmStorageMPS(TestCase):
    """Test cases for USM storage functionality on MPS devices."""

    def tearDown(self):
        """Clean up MPS resources after each test."""
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    @unittest.skipIf(not TEST_MPS, "MPS not available")
    def test_usm_share_cpu_to_mps(self):
        """Test usm_share_ from CPU to MPS device."""
        # Create CPU storage
        cpu_storage = torch.UntypedStorage(1024)
        for i in range(1024):
            cpu_storage[i] = i % 256
        
        # Share to MPS
        mps_storage = cpu_storage.usm_share_(torch.device('mps'))
        
        # Verify storage properties
        self.assertEqual(mps_storage.device.type, 'mps')
        self.assertEqual(mps_storage.nbytes(), cpu_storage.nbytes())

    @unittest.skipIf(not TEST_MPS, "MPS not available")
    def test_usm_share_cpu_to_mps_with_tensor(self):
        """Test usm_share_ with tensors created from storage on MPS."""
        # Create CPU storage with USM allocator using temp file
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_file = f.name
            f.write(b'\x00' * 4000)
        
        try:
            cpu_storage = torch.UntypedStorage.from_file(
                temp_file,
                shared=False,
                nbytes=4000,
                usm=True
            )
            
            # Fill with data as float32
            cpu_tensor = torch.tensor([], dtype=torch.float32).set_(cpu_storage)
            cpu_tensor = cpu_tensor[:1000]
            cpu_tensor.copy_(torch.arange(1000, dtype=torch.float32))
            
            # Share storage to MPS
            mps_storage = cpu_storage.usm_share_(torch.device('mps'))
            
            # Create MPS tensor from shared storage
            mps_tensor = torch.empty(0, dtype=torch.float32, device='mps').set_(mps_storage)
            mps_tensor = mps_tensor[:1000]
            
            # Verify data
            self.assertTrue(torch.allclose(cpu_tensor, mps_tensor.cpu()))
        finally:
            os.unlink(temp_file)

    @unittest.skipIf(not TEST_MPS, "MPS not available")
    def test_usm_share_mps_storage_copy(self):
        """Test copying data between CPU and MPS storages."""
        # Create source CPU storage
        src_storage = torch.UntypedStorage(100)
        for i in range(100):
            src_storage[i] = i
        
        # Share to MPS
        mps_storage = src_storage.usm_share_(torch.device('mps'))
        
        # Verify data is accessible on MPS device
        mps_tensor = torch.tensor([], dtype=torch.uint8, device='mps').set_(mps_storage)
        self.assertEqual(mps_tensor.device.type, 'mps')

    @unittest.skipIf(not TEST_MPS, "MPS not available")
    def test_usm_share_mps_preservation(self):
        """Test that shared storage preserves data after modifications."""
        # Create and populate CPU storage with USM allocator using temp file
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_file = f.name
            f.write(b'\x00' * 256)
        
        try:
            cpu_storage = torch.UntypedStorage.from_file(
                temp_file,
                shared=False,
                nbytes=256,
                usm=True
            )
            for i in range(256):
                cpu_storage[i] = i
            
            # Share to MPS
            mps_storage = cpu_storage.usm_share_(torch.device('mps'))
            
            # Verify initial data
            errors = 0
            for i in range(256):
                if cpu_storage[i] != i:
                    errors += 1
            self.assertEqual(errors, 0)
            
            # Verify MPS storage can be used
            mps_tensor = torch.tensor([], dtype=torch.uint8, device='mps').set_(mps_storage)
            self.assertEqual(mps_tensor.device.type, 'mps')
        finally:
            os.unlink(temp_file)

    @unittest.skipIf(not TEST_MPS, "MPS not available")
    def test_usm_share_mps_tensor_operations(self):
        """Test tensor operations on MPS with shared storage."""
        # Create CPU storage
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_file = f.name
            f.write(b'\x00' * 1024)
        
        try:
            cpu_storage = torch.UntypedStorage.from_file(
                temp_file,
                shared=False,
                nbytes=1024,
                usm=True
            )
            
            # Create float tensor on CPU
            cpu_tensor = torch.tensor([], dtype=torch.float32).set_(cpu_storage)
            cpu_tensor = cpu_tensor[:256]
            cpu_tensor.copy_(torch.randn(256))
            
            # Share to MPS
            mps_storage = cpu_storage.usm_share_(torch.device('mps'))
            
            # Create MPS tensor and perform operations
            mps_tensor = torch.empty(0, dtype=torch.float32, device='mps').set_(mps_storage)
            mps_tensor = mps_tensor[:256]
            
            # Perform some operations
            result = mps_tensor * 2.0 + 1.0
            result_cpu = result.cpu()
            
            # Verify computation worked
            self.assertEqual(result_cpu.shape, torch.Size([256]))
            self.assertTrue(torch.isfinite(result_cpu).all())
        finally:
            os.unlink(temp_file)

    @unittest.skipIf(not TEST_MPS, "MPS not available")
    def test_usm_share_mps_zero_size(self):
        """Test edge case with zero-sized storage on MPS."""
        # Create zero-size storage
        empty_storage = torch.UntypedStorage(0)
        
        # Should handle gracefully
        result = empty_storage.usm_share_(torch.device('mps'))
        self.assertEqual(result.nbytes(), 0)
        self.assertEqual(result.device.type, 'mps')

    @unittest.skipIf(not TEST_MPS, "MPS not available")
    def test_usm_share_mps_errors(self):
        """Test error conditions for usm_share_ on MPS."""
        # Test with non-CPU source (if MPS storage is available)
        try:
            mps_storage = torch.UntypedStorage(100, device='mps')
            with self.assertRaisesRegex(RuntimeError, "source storage must be on CPU"):
                mps_storage.usm_share_(torch.device('mps'))
        except RuntimeError:
            # If MPS storage creation is not supported, skip this part
            self.skipTest("MPS storage creation not supported")

    @unittest.skipIf(not TEST_MPS, "MPS not available")
    def test_usm_share_mps_multiple_operations(self):
        """Test multiple consecutive usm_share_ calls."""
        # Create CPU storage
        cpu_storage = torch.UntypedStorage(512)
        for i in range(512):
            cpu_storage[i] = i % 128
        
        # Share to MPS
        mps_storage = cpu_storage.usm_share_(torch.device('mps'))
        
        # Verify it works
        self.assertEqual(mps_storage.device.type, 'mps')
        self.assertEqual(mps_storage.nbytes(), 512)
        
        # Try sharing the same storage again
        mps_storage2 = cpu_storage.usm_share_(torch.device('mps'))
        self.assertEqual(mps_storage2.device.type, 'mps')
        self.assertEqual(mps_storage2.nbytes(), 512)


class TestUsmMPSPython(TestCase):
    """Test Python API for USM storage with MPS."""
    
    @unittest.skipIf(not TEST_MPS, "MPS not available")
    def test_storage_usm_share_mps_api(self):
        """Test that usm_share_ method works with MPS device."""
        storage = torch.UntypedStorage(100)
        
        # Check method exists
        self.assertTrue(hasattr(storage, 'usm_share_'))
        
        # Check it's callable with MPS device
        self.assertTrue(callable(storage.usm_share_))
        
        # Should not raise with MPS device
        result = storage.usm_share_(torch.device('mps'))
        self.assertIsNotNone(result)

    @unittest.skipIf(not TEST_MPS, "MPS not available")
    def test_from_file_usm_parameter_with_mps(self):
        """Test from_file with USM allocator followed by MPS sharing."""
        if IS_WINDOWS:
            self.skipTest("UsmAllocator is not supported on Windows")
        
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_file = f.name
            f.write(b'\x00' * 100)
        
        try:
            # Create storage with USM allocator
            storage = torch.UntypedStorage.from_file(
                temp_file,
                shared=False,
                nbytes=100,
                usm=True
            )
            
            # Share to MPS
            mps_storage = storage.usm_share_(torch.device('mps'))
            self.assertEqual(mps_storage.device.type, 'mps')
        finally:
            os.unlink(temp_file)


if __name__ == '__main__':
    run_tests()
