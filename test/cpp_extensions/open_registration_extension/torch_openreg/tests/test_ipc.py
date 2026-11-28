# Owner(s): ["module: PrivateUse1"]
import multiprocessing

import torch
import torch_openreg
from torch.testing._internal.common_utils import run_tests, skipIfWindows, TestCase


# ---------------------------------------------------------------------------
# All worker functions must be at module level so 'spawn' can pickle them.
# Each producer waits on a done Event so its storage stays alive until the
# consumer has finished using and released the IPC-mapped memory.
# ---------------------------------------------------------------------------


def _producer_basic(queue, done):
    t = torch.arange(1.0, 5.0, device="openreg:0")  # [1., 2., 3., 4.]
    queue.put(t)
    done.wait(timeout=30)
    del t


def _consumer_basic(queue, result_queue, done):
    t = queue.get()
    assert t.device.type == "openreg", f"Expected openreg device, got {t.device.type}"
    result = t.cpu().tolist()
    del t  # Decrement IPC ref counter before signaling producer
    result_queue.put(result)
    done.set()


def _producer_offset(queue, done):
    base = torch.arange(0.0, 8.0, device="openreg:0")
    # base[4:] has storage_offset=4, size=(4,), stride=(1,); values [4..7].
    t = base[4:]
    queue.put(t)
    done.wait(timeout=30)
    del t
    del base


def _consumer_offset(queue, result_queue, done):
    t = queue.get()
    assert t.device.type == "openreg", f"Expected openreg device, got {t.device.type}"
    result = t.cpu().tolist()
    del t  # Decrement IPC ref counter before signaling producer
    result_queue.put(result)
    done.set()


def _producer_strided(queue, done):
    base = torch.arange(0.0, 8.0, device="openreg:0")
    # base[::2] has storage_offset=0, size=(4,), stride=(2,); values [0,2,4,6].
    t = base[::2]
    queue.put(t)
    done.wait(timeout=30)
    del t
    del base


def _consumer_strided(queue, result_queue, done):
    t = queue.get()
    assert t.device.type == "openreg", f"Expected openreg device, got {t.device.type}"
    result = {
        "values": t.cpu().tolist(),
        "size": list(t.shape),
        "stride": list(t.stride()),
    }
    del t  # Decrement IPC ref counter before signaling producer
    result_queue.put(result)
    done.set()


def _producer_reshare(queue, done):
    t = torch.arange(1.0, 5.0, device="openreg:0")
    queue.put(t)
    done.wait(timeout=30)
    del t


def _consumer_reshare(queue, result_queue, done):
    t = queue.get()
    assert t.device.type == "openreg", f"Expected openreg device, got {t.device.type}"
    try:
        t.untyped_storage()._share_device_()
        result_queue.put(None)  # sentinel: no error raised; assertIsNotNone will catch this
    except RuntimeError as e:
        result_queue.put(str(e))
    finally:
        del t  # Decrement IPC ref counter before signaling producer
        done.set()


def _producer_limbo(queue, done):
    t = torch.arange(1.0, 5.0, device="openreg:0")
    queue.put(t)
    done.wait(timeout=30)
    # del t intentionally AFTER done.wait() but consumer signals done
    # before releasing t, so counter > 0 when DeviceIPCSentDataDelete
    # fires here → DeviceIPCSentData goes into the limbo list.
    del t


def _consumer_limbo(queue, done):
    t = queue.get()
    # Signal the producer before releasing the IPC storage so that
    # the producer's del t sees counter > 0 and enters the limbo path.
    done.set()
    del t


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@skipIfWindows(msg="IPC via shm_open is POSIX-only; _WIN32 guard in OpenRegHooks")
class TestIpcWorkflow(TestCase):
    """Full two-process IPC workflow tests using the OpenReg reference backend."""

    def _run(self, prod_fn, cons_fn):
        """
        Spawn one producer and one consumer process sharing a multiprocessing
        Event for producer lifetime. Returns whatever the consumer puts into
        result_queue (with a 5-second timeout so a crash surfaces cleanly).
        """
        ctx = multiprocessing.get_context("spawn")
        queue = ctx.Queue()
        result_queue = ctx.Queue()
        done = ctx.Event()

        prod = ctx.Process(target=prod_fn, args=(queue, done))
        cons = ctx.Process(target=cons_fn, args=(queue, result_queue, done))
        prod.start()
        cons.start()
        prod.join(timeout=30)
        cons.join(timeout=30)

        self.assertFalse(prod.is_alive(), "Producer process timed out (hung)")
        self.assertFalse(cons.is_alive(), "Consumer process timed out (hung)")
        self.assertEqual(
            prod.exitcode, 0, f"Producer exited with code {prod.exitcode}"
        )
        self.assertEqual(
            cons.exitcode, 0, f"Consumer exited with code {cons.exitcode}"
        )
        return result_queue.get(timeout=5)

    def test_basic_tensor_roundtrip(self):
        """float32 tensor arrives at the consumer with identical values."""
        result = self._run(_producer_basic, _consumer_basic)
        self.assertEqual(result, [1.0, 2.0, 3.0, 4.0])

    def test_offset_tensor_roundtrip(self):
        """
        A sliced tensor (non-zero storage_offset) is transferred correctly.
        reduce_tensor records tensor_offset; rebuild_device_tensor re-applies it.
        """
        result = self._run(_producer_offset, _consumer_offset)
        self.assertEqual(result, [4.0, 5.0, 6.0, 7.0])

    def test_strided_tensor_roundtrip(self):
        """
        A non-contiguous tensor preserves its shape, stride, and values.
        The full backing storage is transferred; rebuild_device_tensor
        reconstructs the strided view using the original size and stride.
        """
        result = self._run(_producer_strided, _consumer_strided)
        self.assertEqual(result["values"], [0.0, 2.0, 4.0, 6.0])
        self.assertEqual(result["size"], [4])
        self.assertEqual(result["stride"], [2])

    def test_received_via_ipc_cannot_be_reshared(self):
        """
        _new_shared_device sets received_via_ipc_=True on the rebuilt StorageImpl.
        Calling _share_device_() on that storage must raise RuntimeError with the
        guard message from StorageSharing.cpp (TORCH_CHECK on received_via_ipc_).
        """
        result = self._run(_producer_reshare, _consumer_reshare)
        self.assertIsNotNone(result, "Expected RuntimeError but no error was raised")
        self.assertIn("received from another process", result)

    def test_producer_exits_cleanly_with_tensor_in_limbo(self):
        """
        Verify that the producer process exits with code 0 even when it
        calls del t while the consumer still holds the IPC-mapped storage.
        DeviceIPCSentData goes into the DeviceIPCSentDataLimbo list and is
        cleaned up at process exit.  The warning fires but is not fatal.
        This mirrors test_cuda_ipc_limbo_cleanup_at_exit in
        test_multiprocessing.py.
        """
        ctx = multiprocessing.get_context("spawn")
        queue = ctx.Queue()
        done = ctx.Event()

        prod = ctx.Process(target=_producer_limbo, args=(queue, done))
        cons = ctx.Process(target=_consumer_limbo, args=(queue, done))
        prod.start()
        cons.start()
        prod.join(timeout=30)
        cons.join(timeout=30)

        self.assertFalse(prod.is_alive(), "Producer process timed out (hung)")
        self.assertFalse(cons.is_alive(), "Consumer process timed out (hung)")
        # Producer must exit cleanly despite the limbo warning.
        self.assertEqual(
            prod.exitcode, 0, f"Producer exited with code {prod.exitcode}"
        )
        self.assertEqual(
            cons.exitcode, 0, f"Consumer exited with code {cons.exitcode}"
        )


if __name__ == "__main__":
    run_tests()
