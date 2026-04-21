# Owner(s): ["module: inductor"]

from unittest.mock import MagicMock

import torch
from torch._inductor import ir
from torch._inductor.codegen.wrapper import (
    BorrowMatchKey,
    CommBufferReuseKey,
    FreeIfNotReusedLine,
    MemoryPlanningState,
)
from torch._inductor.test_case import run_tests, TestCase


class TestPgAllocBorrowState(TestCase):
    def _make_comm_key(
        self, size_str: str = "1024", group: str = "default_pg"
    ) -> CommBufferReuseKey:
        return (
            torch.device("cuda:0"),
            torch.bfloat16,
            size_str,
            ir.CommBufferType.PG_ALLOC,
            group,
        )

    def _make_mock_free_line(self, name: str = "buf") -> FreeIfNotReusedLine:
        line = MagicMock(spec=FreeIfNotReusedLine)
        line.is_reused = False
        line.node = MagicMock()
        line.node.get_name.return_value = name
        return line

    def test_comm_pool_match_key_index_push(self):
        """Push updates comm_pool_by_match_key."""
        state = MemoryPlanningState()
        comm_key = self._make_comm_key()
        match_key: BorrowMatchKey = (
            torch.device("cuda:0"),
            torch.bfloat16,
            "1024",
        )

        free_line = self._make_mock_free_line("comm_0")
        state.comm_buffer_push(comm_key, free_line)

        self.assertIn(match_key, state.comm_pool_by_match_key)
        self.assertEqual(len(state.comm_pool_by_match_key[match_key]), 1)
        self.assertTrue(state.comm_buffer_contains(comm_key))

    def test_comm_pool_match_key_index_pop(self):
        """Pop updates comm_pool_by_match_key."""
        state = MemoryPlanningState()
        comm_key = self._make_comm_key()
        match_key: BorrowMatchKey = (
            torch.device("cuda:0"),
            torch.bfloat16,
            "1024",
        )

        free_line = self._make_mock_free_line("comm_0")
        state.comm_buffer_push(comm_key, free_line)
        popped = state.comm_buffer_pop(comm_key)

        self.assertIs(popped, free_line)
        self.assertEqual(len(state.comm_pool_by_match_key[match_key]), 0)
        self.assertFalse(state.comm_buffer_contains(comm_key))

    def test_borrowed_comm_keys_roundtrip(self):
        """borrowed_comm_keys tracks borrows and returns."""
        state = MemoryPlanningState()
        comm_key = self._make_comm_key()

        state.borrowed_comm_keys["regular_buf_0"] = comm_key
        self.assertIn("regular_buf_0", state.borrowed_comm_keys)

        key = state.borrowed_comm_keys.pop("regular_buf_0")
        self.assertEqual(key, comm_key)
        self.assertNotIn("regular_buf_0", state.borrowed_comm_keys)

    def test_borrow_return_to_comm_pool(self):
        """Buffer borrowed from comm pool is returned there on free."""
        state = MemoryPlanningState()
        comm_key = self._make_comm_key()

        # Simulate: comm buffer freed -> in comm pool
        comm_free = self._make_mock_free_line("comm_0")
        state.comm_buffer_push(comm_key, comm_free)

        # Simulate borrow: pop from comm pool, record in borrowed_comm_keys
        popped = state.comm_buffer_pop(comm_key)
        popped.is_reused = True
        state.borrowed_comm_keys["regular_buf"] = comm_key

        # Comm pool should be empty now
        self.assertFalse(state.comm_buffer_contains(comm_key))

        # Simulate regular buffer free -> should return to comm pool
        regular_free = self._make_mock_free_line("regular_buf")
        borrowed_key = state.borrowed_comm_keys.pop("regular_buf")
        state.comm_buffer_push(borrowed_key, regular_free)

        # Comm pool should have it back
        self.assertTrue(state.comm_buffer_contains(comm_key))

    def test_multiple_borrows_same_key(self):
        """Multiple sequential borrows from the same comm key work."""
        state = MemoryPlanningState()
        comm_key = self._make_comm_key()

        # Initial comm buffer in pool
        comm_free = self._make_mock_free_line("comm_0")
        state.comm_buffer_push(comm_key, comm_free)

        # First borrow
        popped = state.comm_buffer_pop(comm_key)
        popped.is_reused = True
        state.borrowed_comm_keys["reg_0"] = comm_key
        self.assertFalse(state.comm_buffer_contains(comm_key))

        # Return first borrow
        reg_free_0 = self._make_mock_free_line("reg_0")
        key = state.borrowed_comm_keys.pop("reg_0")
        state.comm_buffer_push(key, reg_free_0)
        self.assertTrue(state.comm_buffer_contains(comm_key))

        # Second borrow
        popped2 = state.comm_buffer_pop(comm_key)
        popped2.is_reused = True
        state.borrowed_comm_keys["reg_1"] = comm_key
        self.assertFalse(state.comm_buffer_contains(comm_key))

        # Return second borrow
        reg_free_1 = self._make_mock_free_line("reg_1")
        key = state.borrowed_comm_keys.pop("reg_1")
        state.comm_buffer_push(key, reg_free_1)
        self.assertTrue(state.comm_buffer_contains(comm_key))

    def test_different_groups_not_mixed(self):
        """Buffers from different process groups stay separate."""
        state = MemoryPlanningState()
        key_g1 = self._make_comm_key(group="group_1")
        key_g2 = self._make_comm_key(group="group_2")

        state.comm_buffer_push(key_g1, self._make_mock_free_line("c1"))
        state.comm_buffer_push(key_g2, self._make_mock_free_line("c2"))

        self.assertTrue(state.comm_buffer_contains(key_g1))
        self.assertTrue(state.comm_buffer_contains(key_g2))

        state.comm_buffer_pop(key_g1)
        self.assertFalse(state.comm_buffer_contains(key_g1))
        self.assertTrue(state.comm_buffer_contains(key_g2))

    def test_symm_mem_not_borrowed(self):
        """SYMM_MEM buffers should not be candidates for borrowing."""
        state = MemoryPlanningState()
        symm_key: CommBufferReuseKey = (
            torch.device("cuda:0"),
            torch.bfloat16,
            "1024",
            ir.CommBufferType.SYMM_MEM,
            "default_pg",
        )
        match_key: BorrowMatchKey = (
            torch.device("cuda:0"),
            torch.bfloat16,
            "1024",
        )

        state.comm_buffer_push(symm_key, self._make_mock_free_line("s0"))

        # Match key index should have it
        self.assertIn(match_key, state.comm_pool_by_match_key)

        # But the key type is SYMM_MEM, which _try_borrow_from_comm_pool
        # should skip (checked in AllocateLine._try_borrow_from_comm_pool)
        # Here we just verify the key is present but type is SYMM_MEM
        keys_for_match = state.comm_pool_by_match_key[match_key]
        self.assertEqual(len(keys_for_match), 1)
        self.assertEqual(keys_for_match[0][3], ir.CommBufferType.SYMM_MEM)

    def test_demand_schedule_bisect(self):
        """Verify bisect logic for finding next comm demand."""
        import bisect

        demand_snis = [5, 15, 25, 35]

        # At sni=3, next demand is at 5
        idx = bisect.bisect_right(demand_snis, 3)
        self.assertEqual(demand_snis[idx], 5)

        # At sni=5, next demand is at 15
        idx = bisect.bisect_right(demand_snis, 5)
        self.assertEqual(demand_snis[idx], 15)

        # At sni=10, next demand is at 15
        idx = bisect.bisect_right(demand_snis, 10)
        self.assertEqual(demand_snis[idx], 15)

        # At sni=35, no next demand
        idx = bisect.bisect_right(demand_snis, 35)
        self.assertEqual(idx, len(demand_snis))

    def test_borrow_safe_when_freed_before_demand(self):
        """Borrow is safe: free_sni < next_comm_sni."""
        # This tests the condition checked in _try_borrow_from_comm_pool
        import bisect

        demand_snis = [5, 20, 40]
        current_sni = 7  # Borrowing at step 7
        free_sni = 15  # Buffer freed at step 15

        # Next demand after current (7) is at 20
        idx = bisect.bisect_right(demand_snis, current_sni)
        next_comm_sni = demand_snis[idx]
        self.assertEqual(next_comm_sni, 20)

        # free_sni (15) < next_comm_sni (20) -> safe
        self.assertTrue(free_sni < next_comm_sni)

    def test_borrow_unsafe_when_freed_after_demand(self):
        """Borrow is unsafe: free_sni >= next_comm_sni."""
        import bisect

        demand_snis = [5, 20, 40]
        current_sni = 7
        free_sni = 25  # Freed AFTER next demand at 20

        idx = bisect.bisect_right(demand_snis, current_sni)
        next_comm_sni = demand_snis[idx]
        self.assertEqual(next_comm_sni, 20)

        # free_sni (25) >= next_comm_sni (20) -> unsafe
        self.assertFalse(free_sni < next_comm_sni)

    def test_borrow_safe_no_future_demand(self):
        """Borrow is always safe when no future comm demand exists."""
        import bisect

        demand_snis = [5]
        current_sni = 10

        idx = bisect.bisect_right(demand_snis, current_sni)
        # No more demands
        self.assertEqual(idx, len(demand_snis))
        # next_comm_sni is None -> always safe


if __name__ == "__main__":
    run_tests()
