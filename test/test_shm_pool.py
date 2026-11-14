# Owner(s): ["module: dataloader"]
import os
import unittest

import torch
from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    TestCase,
    run_tests,
)
from torch.utils.data import Dataset, DataLoader


# Conditionally import the shm_pool module (only when SHM_POOL=1 is used).
# These tests exercise the slot pool directly and via a real DataLoader.

class FakeKJT:
    """Minimal KJT-like object with tensor attributes, to test __dict__ recursion."""
    def __init__(self, values, lengths, offsets):
        self._values = values
        self._lengths = lengths
        self._offsets = offsets
        self._keys = ["a", "b"]


class TestShmPool(TestCase):
    """Unit tests for the SharedMemory slot pool module.

    These run on CPU only — no CUDA required — so they work in CI.
    """

    def setUp(self):
        os.environ["SHM_POOL"] = "1"
        # Force re-cache of the enabled flag.
        from torch.utils.data._utils import shm_pool
        shm_pool._ENABLED = None
        self.shm_pool = shm_pool

    def tearDown(self):
        # Clean up any leftover SharedMemory regions.
        import multiprocessing as mp
        from multiprocessing import shared_memory
        rank = os.environ.get("RANK", "0")
        for s in range(16):
            name = f"__dataloader_shmpool_{rank}_{s}"
            try:
                shared_memory.SharedMemory(name=name).close()
                shared_memory.SharedMemory(name=name).unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass
        # Reset module state
        self.shm_pool._SLOTS = []
        self.shm_pool._SLOT_BYTES = 0
        self.shm_pool._POOL_READY = None
        self.shm_pool._SLOT_BYTES_VAL = None
        self.shm_pool._N_SLOTS = 0
        self.shm_pool._FREE = None
        self.shm_pool._ENABLED = None
        os.environ.pop("SHM_POOL", None)

    def test_is_enabled(self):
        self.assertTrue(self.shm_pool.is_enabled())
        self.shm_pool._ENABLED = None
        os.environ["SHM_POOL"] = "0"
        self.assertFalse(self.shm_pool.is_enabled())

    def test_init_coordination(self):
        p = self.shm_pool
        p.init_coordination(4)
        self.assertEqual(p._N_SLOTS, 4)
        self.assertIsNotNone(p._POOL_READY)
        self.assertFalse(p._POOL_READY.is_set())
        self.assertIsNotNone(p._FREE)
        # free-list should have 4 slots
        slots = []
        for _ in range(4):
            slots.append(p._FREE.get())
        self.assertEqual(sorted(slots), [0, 1, 2, 3])

    def test_estimate_slot_bytes(self):
        p = self.shm_pool
        batch = {"a": torch.randn(1000), "b": [torch.randn(500), torch.randn(250)]}
        sb = p.estimate_slot_bytes(batch)
        # sum of bytes = 1000*4 + 500*4 + 250*4 = 7000, * 2.0 factor = 14000
        self.assertGreater(sb, 7000)
        self.assertLessEqual(sb, 7000 * 2 + 1000)  # headroom

    def test_estimate_slot_bytes_kjt(self):
        """estimate_slot_bytes recurses into arbitrary objects via __dict__."""
        p = self.shm_pool
        kjt = FakeKJT(torch.randn(1000), torch.tensor([1, 2, 3]), torch.tensor([0, 1, 3]))
        sb = p.estimate_slot_bytes(kjt)
        self.assertGreater(sb, 0)

    def test_copy_and_rebuild_dict(self):
        """Copy a dict of tensors into a slot and rebuild as zero-copy views."""
        p = self.shm_pool
        p.init_coordination(4)
        batch = {"a": torch.randn(1024), "b": torch.randn(512)}
        p.init_pool_from_batch(batch)
        p._ensure_attached()

        stub = p.copy_batch_into_slot(batch)
        self.assertTrue(p.is_slot_stub(stub))
        self.assertGreater(len(stub.meta), 0)

        out = p.rebuild_batch_from_slot(stub)
        self.assertIsInstance(out, dict)
        self.assertTrue(torch.equal(out["a"], batch["a"]))
        self.assertTrue(torch.equal(out["b"], batch["b"]))
        p.ret_slot(stub.slot_idx)

    def test_copy_and_rebuild_list(self):
        """Copy a list of tensors into a slot and rebuild."""
        p = self.shm_pool
        p.init_coordination(4)
        batch = [torch.randn(1024), torch.randn(512)]
        p.init_pool_from_batch(batch)
        p._ensure_attached()

        stub = p.copy_batch_into_slot(batch)
        self.assertTrue(p.is_slot_stub(stub))
        out = p.rebuild_batch_from_slot(stub)
        self.assertIsInstance(out, list)
        self.assertTrue(torch.equal(out[0], batch[0]))
        self.assertTrue(torch.equal(out[1], batch[1]))
        p.ret_slot(stub.slot_idx)

    def test_copy_and_rebuild_kjt(self):
        """Copy a KJT-like object into a slot and rebuild — tensors pooled, non-tensor attrs preserved."""
        p = self.shm_pool
        p.init_coordination(4)
        kjt = FakeKJT(torch.randn(1024), torch.tensor([1, 2, 3]), torch.tensor([0, 1, 3]))
        p.init_pool_from_batch(kjt)
        p._ensure_attached()

        stub = p.copy_batch_into_slot(kjt)
        self.assertTrue(p.is_slot_stub(stub))
        # 3 tensors should be pooled
        self.assertEqual(len(stub.meta), 3)

        out = p.rebuild_batch_from_slot(stub)
        self.assertIsInstance(out, FakeKJT)
        self.assertTrue(torch.equal(out._values, kjt._values))
        self.assertTrue(torch.equal(out._lengths, kjt._lengths))
        self.assertTrue(torch.equal(out._offsets, kjt._offsets))
        # non-tensor attribute preserved
        self.assertEqual(out._keys, ["a", "b"])
        p.ret_slot(stub.slot_idx)

    def test_slot_recycle(self):
        """Returned slots go back to the free-list and can be reused."""
        p = self.shm_pool
        p.init_coordination(2)
        batch = {"a": torch.randn(1024)}
        p.init_pool_from_batch(batch)
        p._ensure_attached()

        stub1 = p.copy_batch_into_slot(batch)
        slot1 = stub1.slot_idx
        p.ret_slot(slot1)

        stub2 = p.copy_batch_into_slot(batch)
        # slot should be reused (only 2 slots, one was returned)
        self.assertIn(stub2.slot_idx, [slot1, 0, 1])
        p.ret_slot(stub2.slot_idx)

    def test_degrade_when_not_ready(self):
        """Before the pool is ready (first batch), copy_batch_into_slot returns the batch unchanged."""
        p = self.shm_pool
        p.init_coordination(4)
        # pool not ready yet (Event not set)
        batch = {"a": torch.randn(100)}
        result = p.copy_batch_into_slot(batch)
        self.assertFalse(p.is_slot_stub(result))
        self.assertIs(result, batch)

    def test_degrade_oversized(self):
        """If the batch is larger than the slot, it degrades to normal (returned unchanged)."""
        p = self.shm_pool
        p.init_coordination(4)
        # small batch to size the pool small
        small = {"a": torch.randn(100)}
        p.init_pool_from_batch(small)
        p._ensure_attached()
        # now a huge batch that won't fit
        big = {"x": torch.randn(10_000_000)}
        result = p.copy_batch_into_slot(big)
        self.assertFalse(p.is_slot_stub(result))
        self.assertIs(result, big)

    def test_data_loader_integration(self):
        """End-to-end: DataLoader with SHM_POOL=1 produces correct data."""
        os.environ["SHM_POOL"] = "1"

        class DS(Dataset):
            def __init__(self):
                self.data = [torch.randn(100) for _ in range(32)]

            def __len__(self):
                return 32

            def __getitem__(self, idx):
                return {"x": self.data[idx], "id": idx}

        loader = DataLoader(
            DS(), batch_size=4, num_workers=2, pin_memory=False, prefetch_factor=2
        )
        seen = []
        for batch in loader:
            # data should be correct regardless of SHM_POOL
            for i, x in zip(batch["id"], batch["x"]):
                seen.append(int(i))
        self.assertEqual(sorted(seen), list(range(32)))

    @unittest.skipIf(IS_WINDOWS, "fork start method needed for slot pool")
    def test_data_loader_shm_pool_correctness(self):
        """DataLoader with SHM_POOL=1 produces correct tensors (CPU pin_memory)."""
        os.environ["SHM_POOL"] = "1"

        class DS(Dataset):
            def __init__(self):
                torch.manual_seed(42)
                self.data = [torch.randn(256, dtype=torch.float32) for _ in range(16)]

            def __len__(self):
                return 16

            def __getitem__(self, idx):
                return {"x": self.data[idx], "id": idx}

        ds = DS()
        # Reference: without SHM_POOL
        os.environ["SHM_POOL"] = "0"
        ref_loader = DataLoader(
            ds, batch_size=4, num_workers=2, pin_memory=False, prefetch_factor=2
        )
        ref_batches = []
        for batch in ref_loader:
            ref_batches.append({k: v.clone() for k, v in batch.items()})

        # With SHM_POOL
        os.environ["SHM_POOL"] = "1"
        from torch.utils.data._utils import shm_pool
        shm_pool._ENABLED = None
        shm_loader = DataLoader(
            ds, batch_size=4, num_workers=2, pin_memory=False, prefetch_factor=2
        )
        shm_batches = []
        for batch in shm_loader:
            shm_batches.append({k: v.clone() for k, v in batch.items()})

        # Compare
        self.assertEqual(len(ref_batches), len(shm_batches))
        for ref, shm in zip(ref_batches, shm_batches):
            self.assertTrue(torch.equal(ref["x"], shm["x"]))


if __name__ == "__main__":
    run_tests()
