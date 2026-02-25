# Owner(s): ["module: PrivateUse1"]

import multiprocessing
import threading
from concurrent.futures import ThreadPoolExecutor

import torch
from torch.testing._internal.common_dtype import get_all_dtypes
from torch.testing._internal.common_utils import run_tests, skipIfWindows, TestCase


def _multiprocessing_worker(device_idx, result_queue):
    """Worker function for multiprocessing spawn test"""
    try:
        torch.accelerator.set_device_index(device_idx)
        x = torch.randn(10, 10, device="openreg")
        y = torch.randn(10, 10, device="openreg")
        z = torch.matmul(x, y)
        result_queue.put(("success", device_idx, torch.sum(z).item()))
    except Exception as e:
        result_queue.put(("error", device_idx, str(e)))


class TestDevice(TestCase):
    def test_device_count(self):
        """Test device count query"""
        count = torch.accelerator.device_count()
        self.assertEqual(count, 2)

    def test_device_switch(self):
        """Test switching between devices"""
        torch.accelerator.set_device_index(1)
        self.assertEqual(torch.accelerator.current_device_index(), 1)

        torch.accelerator.set_device_index(0)
        self.assertEqual(torch.accelerator.current_device_index(), 0)

    def test_device_context(self):
        """Test device context manager"""
        device = torch.accelerator.current_device_index()
        with torch.accelerator.device_index(None):
            self.assertEqual(torch.accelerator.current_device_index(), device)
        self.assertEqual(torch.accelerator.current_device_index(), device)

        with torch.accelerator.device_index(1):
            self.assertEqual(torch.accelerator.current_device_index(), 1)
        self.assertEqual(torch.accelerator.current_device_index(), device)

    def test_invalid_device_index(self):
        """Test error handling for invalid device index"""
        with self.assertRaisesRegex(RuntimeError, "The device index is out of range"):
            torch.accelerator.set_device_index(2)

    def test_device_capability(self):
        capability = torch.accelerator.get_device_capability("openreg:0")
        supported_dtypes = capability["supported_dtypes"]
        expected_dtypes = get_all_dtypes(include_complex32=True, include_qint=True)

        self.assertTrue(all(dtype in supported_dtypes for dtype in expected_dtypes))

    def test_tensor_device(self):
        """Test tensor device assignment"""
        x = torch.randn(2, 3, device="openreg")
        self.assertEqual(x.device.type, "openreg")

        x = torch.randn(2, 3, device="openreg:1")
        self.assertEqual(x.device.type, "openreg")
        self.assertEqual(x.device.index, 1)

    def test_device_guard(self):
        """Test device guard context manager"""
        original_device = torch.accelerator.current_device_index()

        with torch.accelerator.device_index(1):
            self.assertEqual(torch.accelerator.current_device_index(), 1)

        self.assertEqual(torch.accelerator.current_device_index(), original_device)

    def test_device_count_consistency(self):
        """Test device count consistency"""
        count = torch.accelerator.device_count()
        self.assertEqual(count, 2)

        # Test that we can access all devices
        for i in range(count):
            torch.accelerator.set_device_index(i)
            self.assertEqual(torch.accelerator.current_device_index(), i)

    @skipIfWindows(msg="Fork not available on Windows")
    def test_device_poison_fork(self):
        # First, initialize in the parent process
        torch.openreg.init()

        def child(q):
            try:
                # Second, try to initialize in the child process
                torch.openreg.init()
            except Exception as e:
                q.put(e)

        ctx = multiprocessing.get_context("fork")
        q = ctx.Queue()
        p = ctx.Process(target=child, args=(q,))
        p.start()
        p.join()

        self.assertTrue(not q.empty())

        exc = q.get()
        with self.assertRaisesRegex(
            RuntimeError,
            (
                "Cannot re-initialize OpenReg in forked subprocess. "
                "To use OpenReg with multiprocessing, you must use the 'spawn' start method"
            ),
        ):
            raise exc

    def test_device_properties(self):
        """Test querying device name and memory information"""
        # Query device name
        device = torch.device("openreg:0")
        self.assertEqual(device.type, "openreg")
        self.assertEqual(device.index, 0)
        device_str = str(device)
        self.assertIn("openreg", device_str)
        self.assertIn("0", device_str)

        # Query device memory information
        try:
            memory_info = torch.accelerator.get_memory_info()
            self.assertIsInstance(memory_info, tuple)
            # Verify memory_info has expected structure if available
            if "allocated" in memory_info:
                self.assertGreaterEqual(memory_info["allocated"], 0)
        except (AttributeError, NotImplementedError):
            # If get_memory_info is not available or not implemented, skip memory check
            self.skipTest("get_memory_info not implemented")

    def test_current_device_after_operations(self):
        """Test that current device remains consistent after operations and device switching persists"""
        original_device = torch.accelerator.current_device_index()
        try:
            # Test 1: Operations don't change the current device
            x = torch.randn(2, 3, device="openreg")
            y = torch.randn(2, 3, device="openreg")
            z = x + y
            result = torch.sum(z)

            # Verify tensors are on correct device
            self.assertEqual(x.device.index, original_device)
            self.assertEqual(z.device.index, original_device)
            self.assertEqual(result.device.index, original_device)

            # Device should remain unchanged after operations
            self.assertEqual(torch.accelerator.current_device_index(), original_device)

            # Test 2: Device switch persists across operations
            torch.accelerator.set_device_index(1)
            self.assertEqual(torch.accelerator.current_device_index(), 1)

            # Perform operations on the new device
            a = torch.randn(5, 5, device="openreg")
            result2 = torch.matmul(a, a)

            # Verify tensors are on device 1
            self.assertEqual(a.device.index, 1)
            self.assertEqual(result2.device.index, 1)

            # Device should still be 1 after operations
            self.assertEqual(torch.accelerator.current_device_index(), 1)
        finally:
            # Restore original device
            torch.accelerator.set_device_index(original_device)

    def test_device_context_restoration(self):
        """Test that device context is properly restored after exceptions"""
        original_device = torch.accelerator.current_device_index()

        # Test exception handling in context manager
        try:
            with torch.accelerator.device_index(1):
                self.assertEqual(torch.accelerator.current_device_index(), 1)
                # Create tensor in context
                x = torch.randn(2, 2, device="openreg")
                self.assertEqual(x.device.index, 1)
                # Simulate an exception
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Device should be restored even after exception
        self.assertEqual(torch.accelerator.current_device_index(), original_device)

        # Test with nested contexts and exceptions
        try:
            with torch.accelerator.device_index(1):
                self.assertEqual(torch.accelerator.current_device_index(), 1)
                with torch.accelerator.device_index(0):
                    self.assertEqual(torch.accelerator.current_device_index(), 0)
                    raise RuntimeError("Nested exception")
        except RuntimeError:
            pass

        # Should restore to original device
        self.assertEqual(torch.accelerator.current_device_index(), original_device)

    def test_concurrent_device_operations(self):
        """Test concurrent device operations from multiple threads"""
        original_device = torch.accelerator.current_device_index()
        num_threads = 4
        num_operations = 50
        errors = []
        barrier = threading.Barrier(num_threads)

        def worker_thread(thread_id):
            try:
                barrier.wait()  # Synchronize start

                # Each thread performs device operations
                for i in range(num_operations):
                    # Switch to device based on thread_id
                    device_idx = thread_id % 2
                    torch.accelerator.set_device_index(device_idx)

                    # Verify device was set correctly
                    current = torch.accelerator.current_device_index()
                    if current != device_idx:
                        errors.append(
                            f"Thread {thread_id}: Expected device {device_idx}, got {current}"
                        )

                    # Perform some operations
                    x = torch.randn(10, 10, device="openreg")
                    y = torch.randn(10, 10, device="openreg")
                    z = torch.matmul(x, y)
                    result = torch.sum(z)

                    # Verify tensors are on correct device
                    if x.device.index != device_idx:
                        errors.append(
                            f"Thread {thread_id}: Tensor x on wrong device, expected {device_idx}, got {x.device.index}"
                        )
                    if result.device.index != device_idx:
                        errors.append(
                            f"Thread {thread_id}: Result tensor on wrong device, expected {device_idx}, got {result.device.index}"
                        )

                    # Verify device didn't change unexpectedly
                    if torch.accelerator.current_device_index() != device_idx:
                        errors.append(
                            f"Thread {thread_id}: Device changed during operation"
                        )
            except Exception as e:
                errors.append(f"Thread {thread_id}: {str(e)}")

        # Launch threads
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_thread, i) for i in range(num_threads)]
            # Wait for all threads to complete
            for future in futures:
                future.result()

        # Verify no errors occurred
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")

        # Restore original device
        torch.accelerator.set_device_index(original_device)

    def test_device_multiprocessing_spawn(self):
        """Test device with multiprocessing spawn method (safe path)"""
        # test device operations in spawned subprocess (safe for multiprocessing)
        ctx = multiprocessing.get_context("spawn")
        result_queue = ctx.Queue()

        p = ctx.Process(target=_multiprocessing_worker, args=(0, result_queue))
        p.start()
        p.join()

        self.assertFalse(result_queue.empty())
        status, device_idx, result = result_queue.get()
        self.assertEqual(status, "success")
        self.assertEqual(device_idx, 0)

    def test_device_state_after_errors(self):
        """Test device state consistency after error conditions"""
        original_device = torch.accelerator.current_device_index()

        # test that device state is preserved after invalid operations
        x = torch.randn(5, 5, device="openreg")
        y = torch.randn(3, 3, device="openreg")
        # this should raise an error (shape mismatch)
        with self.assertRaises(RuntimeError):
            _ = torch.matmul(x, y)

        # device should still be valid
        self.assertEqual(torch.accelerator.current_device_index(), original_device)

        # can still create tensors after error
        z = torch.randn(5, 5, device="openreg")
        self.assertEqual(z.device.index, original_device)
        # Verify tensor operations still work
        w = torch.randn(5, 5, device="openreg")
        _ = torch.matmul(z, w)  # Operation should complete successfully

    def test_device_synchronization(self):
        """Test device synchronization operations"""
        original_device = torch.accelerator.current_device_index()
        try:
            torch.accelerator.set_device_index(1)

            # Perform operations
            x = torch.randn(100, 100, device="openreg")
            y = torch.randn(100, 100, device="openreg")
            z = torch.matmul(x, y)

            # Synchronize device
            torch.accelerator.synchronize()

            # Verify device index is still correct after operations
            self.assertEqual(torch.accelerator.current_device_index(), 1)

            # Verify operations completed and result is on correct device
            result = torch.sum(z)
            self.assertEqual(result.device.type, "openreg")
            self.assertEqual(result.device.index, 1)
        finally:
            torch.accelerator.set_device_index(original_device)


if __name__ == "__main__":
    run_tests()
