# Owner(s): ["oncall: distributed"]

import torch
import torch.nn as nn
from torch.distributed.fsdp import CPUOffloadPolicy
from torch.distributed.fsdp._fully_shard._fsdp_api import (
    _select_largest_first,
    cpu_offload_by_budget,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


class _ThreeParam(nn.Module):
    # float32 byte sizes: big=400, mid=200, small=40 (total 640)
    def __init__(self) -> None:
        super().__init__()
        self.big = nn.Parameter(torch.zeros(100))
        self.mid = nn.Parameter(torch.zeros(50))
        self.small = nn.Parameter(torch.zeros(10))


@instantiate_parametrized_tests
class TestOffloadSelector(TestCase):
    def _items(self) -> list[tuple[str, int]]:
        return [("a", 40), ("b", 100), ("c", 30), ("d", 100)]

    def test_boundaries(self) -> None:
        items = self._items()
        total = sum(nbytes for _, nbytes in items)
        self.assertEqual(_select_largest_first(items, 0), [])
        self.assertEqual(set(_select_largest_first(items, total)), {"a", "b", "c", "d"})
        self.assertEqual(
            set(_select_largest_first(items, total + 1000)), {"a", "b", "c", "d"}
        )

    def test_largest_first_with_tie_break(self) -> None:
        # b and d tie at 100; the tie breaks on FQN, so b precedes d.
        self.assertEqual(_select_largest_first(self._items(), 100), ["b"])
        self.assertEqual(_select_largest_first(self._items(), 200), ["b", "d"])
        # a (40) is larger than c (30) and is taken first among the remainder.
        self.assertEqual(_select_largest_first(self._items(), 240), ["b", "d", "a"])

    def test_never_overshoots(self) -> None:
        items = self._items()
        for budget in range(0, sum(n for _, n in items) + 5):
            selected = set(_select_largest_first(items, budget))
            used = sum(n for fqn, n in items if fqn in selected)
            self.assertLessEqual(used, budget)

    def test_monotonic_superset(self) -> None:
        items = self._items()
        prev: set[str] = set()
        for budget in range(0, sum(n for _, n in items) + 5):
            cur = set(_select_largest_first(items, budget))
            self.assertTrue(prev.issubset(cur))
            prev = cur

    def test_deterministic(self) -> None:
        items = self._items()
        first = _select_largest_first(items, 170)
        for _ in range(5):
            self.assertEqual(_select_largest_first(items, 170), first)
        # Input order must not change the result.
        self.assertEqual(_select_largest_first(list(reversed(items)), 170), first)

    @parametrize(
        "budget,expected",
        [
            (0, set()),
            (399, set()),
            (400, {"big"}),
            (600, {"big", "mid"}),
            (640, {"big", "mid", "small"}),
            (10**9, {"big", "mid", "small"}),
        ],
    )
    def test_cpu_offload_by_budget_map(self, budget: int, expected: set) -> None:
        offload_map = cpu_offload_by_budget(_ThreeParam(), budget)
        self.assertEqual(set(offload_map), expected)
        self.assertNotIn("missing", offload_map)
        for policy in offload_map.values():
            self.assertIsInstance(policy, CPUOffloadPolicy)
            self.assertTrue(policy.pin_memory)

    def test_cpu_offload_by_budget_pin_memory(self) -> None:
        offload_map = cpu_offload_by_budget(_ThreeParam(), 10**9, pin_memory=False)
        self.assertTrue(all(not p.pin_memory for p in offload_map.values()))

    def test_cpu_offload_by_budget_rejects_negative(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            cpu_offload_by_budget(_ThreeParam(), -1)


if __name__ == "__main__":
    run_tests()
