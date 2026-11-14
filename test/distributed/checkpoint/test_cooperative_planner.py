# Owner(s): ["oncall: distributed"]

"""Unit tests for single-process-group CooperativeLoadPlanner MVP."""

from __future__ import annotations

import torch
from torch.distributed.checkpoint.cooperative_planner import (
    _create_cooperative_plans,
    _read_signature,
)
from torch.distributed.checkpoint.metadata import MetadataIndex
from torch.distributed.checkpoint.planner import (
    LoadItemType,
    LoadPlan,
    ReadItem,
)
from torch.testing._internal.common_utils import run_tests, TestCase


def _read_item(fqn: str, length: int = 8) -> ReadItem:
    index = MetadataIndex(fqn, (0,))
    return ReadItem(
        type=LoadItemType.TENSOR,
        dest_index=index,
        dest_offsets=torch.Size([0]),
        storage_index=index,
        storage_offsets=torch.Size([0]),
        lengths=torch.Size([length]),
    )


def _byte_io_item(fqn: str) -> ReadItem:
    index = MetadataIndex(fqn, None)
    return ReadItem(
        type=LoadItemType.BYTE_IO,
        dest_index=index,
        dest_offsets=torch.Size([]),
        storage_index=index,
        storage_offsets=torch.Size([]),
        lengths=torch.Size([]),
    )


def _plan(
    *items: ReadItem,
    shape=(8,),
    dtype="torch.float32",
    device="cpu",
    nbytes=32,
) -> LoadPlan:
    metadata = {
        _read_signature(item): (shape, dtype, device, nbytes)
        for item in items
        if item.type == LoadItemType.TENSOR
    }
    return LoadPlan(items=list(items), planner_data=metadata)


def _fqns(plan: LoadPlan) -> list[str]:
    return [item.dest_index.fqn for item in plan.items]


def _schedule_fqns(schedule) -> list[str]:
    return [task.read_item.dest_index.fqn for task in schedule]


class TestCooperativeLoadPlanner(TestCase):
    def test_deduplicates_and_byte_balances_readers(self):
        weight = _read_item("model.weight")
        bias = _read_item("model.bias")
        plans = _create_cooperative_plans(
            [_plan(weight, bias), _plan(bias, weight)],
            global_ranks=(0, 1),
            min_tensor_bytes=0,
        )

        self.assertEqual(2, sum(len(plan.items) for plan in plans))
        schedules = [plan.planner_data for plan in plans]
        self.assertEqual(schedules[0], schedules[1])
        self.assertEqual({0, 1}, {task.src for task in schedules[0]})
        self.assertEqual(
            ["model.bias", "model.weight"],
            _schedule_fqns(schedules[0]),
        )

    def test_keeps_native_reads_on_metadata_mismatch(self):
        weight = _read_item("model.weight")
        plans = [_plan(weight, shape=(8,)), _plan(weight, shape=(4,))]
        result = _create_cooperative_plans(
            plans,
            global_ranks=(0, 1),
            min_tensor_bytes=0,
        )

        self.assertEqual([1, 1], [len(plan.items) for plan in result])
        self.assertEqual([(), ()], [plan.planner_data for plan in result])

    def test_partial_overlap_keeps_subset_native(self):
        # Rank0/1 share A+B; Rank2 has A+C; Rank3 has A+D. Only A is full-group.
        a, b, c, d = (_read_item(x) for x in ("A", "B", "C", "D"))
        plans = _create_cooperative_plans(
            [_plan(a, b), _plan(a, b), _plan(a, c), _plan(a, d)],
            global_ranks=(10, 11, 12, 13),
            min_tensor_bytes=0,
        )
        owner = next(i for i, p in enumerate(plans) if "A" in _fqns(p))
        self.assertEqual(1, sum(1 for p in plans if "A" in _fqns(p)))
        self.assertEqual(["B"], [f for f in _fqns(plans[0]) if f != "A"])
        self.assertEqual(["B"], [f for f in _fqns(plans[1]) if f != "A"])
        self.assertEqual(["C"], [f for f in _fqns(plans[2]) if f != "A"])
        self.assertEqual(["D"], [f for f in _fqns(plans[3]) if f != "A"])
        schedules = [p.planner_data for p in plans]
        self.assertTrue(all(s == schedules[0] for s in schedules))
        self.assertEqual(["A"], _schedule_fqns(schedules[0]))
        self.assertEqual(schedules[0][0].src, (10, 11, 12, 13)[owner])

    def test_spec_example_only_a_cooperates(self):
        items = {name: _read_item(name) for name in list("ABCDEFGH")}
        plans = _create_cooperative_plans(
            [
                _plan(items["A"], items["B"], items["C"]),
                _plan(items["A"], items["B"], items["D"]),
                _plan(items["A"], items["E"], items["F"]),
                _plan(items["A"], items["G"], items["H"]),
            ],
            global_ranks=(0, 1, 2, 3),
            min_tensor_bytes=0,
        )
        schedules = [p.planner_data for p in plans]
        self.assertEqual(["A"], _schedule_fqns(schedules[0]))
        self.assertTrue(all(s == schedules[0] for s in schedules))
        self.assertEqual(1, sum(1 for p in plans if "A" in _fqns(p)))
        self.assertIn("B", _fqns(plans[0]))
        self.assertIn("B", _fqns(plans[1]))
        self.assertNotIn("B", _fqns(plans[2]))

    def test_no_duplicates_empty_schedule(self):
        plans = _create_cooperative_plans(
            [_plan(_read_item("A")), _plan(_read_item("B"))],
            global_ranks=(0, 1),
            min_tensor_bytes=0,
        )
        self.assertEqual([1, 1], [len(p.items) for p in plans])
        self.assertEqual([(), ()], [p.planner_data for p in plans])

    def test_min_tensor_bytes_gate(self):
        a = _read_item("A")
        plans = _create_cooperative_plans(
            [_plan(a, nbytes=16), _plan(a, nbytes=16)],
            global_ranks=(0, 1),
            min_tensor_bytes=32,
        )
        self.assertEqual([(), ()], [p.planner_data for p in plans])
        self.assertEqual([1, 1], [len(p.items) for p in plans])

    def test_byte_io_never_cooperative(self):
        tensor = _read_item("T")
        blob = _byte_io_item("blob")
        meta = {_read_signature(tensor): ((8,), "torch.float32", "cpu", 32)}
        plans = _create_cooperative_plans(
            [
                LoadPlan(items=[tensor, blob], planner_data=meta),
                LoadPlan(items=[tensor, blob], planner_data=dict(meta)),
            ],
            global_ranks=(0, 1),
            min_tensor_bytes=0,
        )
        self.assertEqual(
            1,
            sum(1 for p in plans for i in p.items if i.type == LoadItemType.TENSOR),
        )
        self.assertEqual(
            2,
            sum(1 for p in plans for i in p.items if i.type == LoadItemType.BYTE_IO),
        )

    def test_duplicate_rank_occurrence_rejected(self):
        a = _read_item("A")
        plans = _create_cooperative_plans(
            [_plan(a, a), _plan(a)],
            global_ranks=(0, 1),
            min_tensor_bytes=0,
        )
        self.assertEqual([(), ()], [p.planner_data for p in plans])
        self.assertEqual([2, 1], [len(p.items) for p in plans])

    def test_two_four_six_rank_full_replicate_identical_schedule(self):
        for world in (2, 4, 6):
            tensors = [_read_item(f"T{i}") for i in range(3)]

            def make_plan():
                metadata = {
                    _read_signature(item): ((8,), "torch.float32", "cpu", 32 * (i + 1))
                    for i, item in enumerate(tensors)
                }
                return LoadPlan(items=list(tensors), planner_data=metadata)

            plans = _create_cooperative_plans(
                [make_plan() for _ in range(world)],
                global_ranks=tuple(range(world)),
                min_tensor_bytes=0,
            )
            self.assertEqual(3, sum(len(p.items) for p in plans), msg=f"world={world}")
            schedules = [p.planner_data for p in plans]
            self.assertTrue(
                all(s == schedules[0] for s in schedules), msg=f"world={world}"
            )
            self.assertEqual(3, len(schedules[0]), msg=f"world={world}")
            # Greedy least-loaded owners: distinct src count is min(items, world).
            self.assertEqual(
                min(3, world),
                len({task.src for task in schedules[0]}),
                msg=f"world={world}",
            )

    def test_native_item_order_preserved_when_dropping_coop(self):
        # Rank-local natives stay native; only full-group COOP is dropped on non-owner.
        coop = _read_item("COOP")
        n0 = _read_item("N0")
        n1 = _read_item("N1")
        x = _read_item("X")
        y = _read_item("Y")
        plans = _create_cooperative_plans(
            [
                _plan(n0, coop, n1, nbytes=64),
                _plan(x, coop, y, nbytes=64),
            ],
            global_ranks=(0, 1),
            min_tensor_bytes=0,
        )
        owner = next(i for i, p in enumerate(plans) if "COOP" in _fqns(p))
        if owner == 0:
            self.assertEqual(["N0", "COOP", "N1"], _fqns(plans[0]))
            self.assertEqual(["X", "Y"], _fqns(plans[1]))
        else:
            self.assertEqual(["N0", "N1"], _fqns(plans[0]))
            self.assertEqual(["X", "COOP", "Y"], _fqns(plans[1]))

    def test_signature_boundary_mismatches_stay_native(self):
        def variant(**kwargs):
            base = dict(
                fqn="T",
                length=8,
                dest_offset=(0,),
                storage_fqn="T",
                storage_offset=(0,),
                storage_offsets=(0,),
                dest_offsets=(0,),
            )
            base.update(kwargs)
            dest = MetadataIndex(base["fqn"], base["dest_offset"])
            storage = MetadataIndex(base["storage_fqn"], base["storage_offset"])
            return ReadItem(
                type=LoadItemType.TENSOR,
                dest_index=dest,
                dest_offsets=torch.Size(list(base["dest_offsets"])),
                storage_index=storage,
                storage_offsets=torch.Size(list(base["storage_offsets"])),
                lengths=torch.Size([base["length"]]),
            )

        cases = [
            (variant(), variant(storage_fqn="OTHER")),
            (variant(), variant(storage_offset=(1,))),
            (variant(), variant(length=4)),
            (variant(), variant(fqn="OTHER")),
            (variant(), variant(dest_offset=(1,))),
            (variant(), variant(dest_offsets=(1,))),
            (variant(), variant(storage_offsets=(1,))),
        ]
        for left, right in cases:
            with self.subTest(left=_read_signature(left), right=_read_signature(right)):
                meta_l = {_read_signature(left): ((8,), "torch.float32", "cpu", 32)}
                meta_r = {_read_signature(right): ((8,), "torch.float32", "cpu", 32)}
                result = _create_cooperative_plans(
                    [
                        LoadPlan(items=[left], planner_data=meta_l),
                        LoadPlan(items=[right], planner_data=meta_r),
                    ],
                    global_ranks=(0, 1),
                    min_tensor_bytes=0,
                )
                self.assertEqual([(), ()], [p.planner_data for p in result])
                self.assertEqual([1, 1], [len(p.items) for p in result])

        # Destination metadata gates (same ReadSignature, different metadata).
        item = variant()
        for meta_a, meta_b in [
            (((8,), "torch.float32", "cpu", 32), ((4,), "torch.float32", "cpu", 16)),
            (((8,), "torch.float32", "cpu", 32), ((8,), "torch.float16", "cpu", 16)),
            (((8,), "torch.float32", "cpu", 32), ((8,), "torch.float32", "cuda", 32)),
            (((8,), "torch.float32", "cpu", 32), ((8,), "torch.float32", "cpu", 16)),
        ]:
            with self.subTest(meta_a=meta_a, meta_b=meta_b):
                result = _create_cooperative_plans(
                    [
                        LoadPlan(
                            items=[item],
                            planner_data={_read_signature(item): meta_a},
                        ),
                        LoadPlan(
                            items=[item],
                            planner_data={_read_signature(item): meta_b},
                        ),
                    ],
                    global_ranks=(0, 1),
                    min_tensor_bytes=0,
                )
                self.assertEqual([(), ()], [p.planner_data for p in result])

    def test_min_tensor_bytes_threshold_boundaries(self):
        a = _read_item("A")
        for threshold, nbytes, expect_coop in [
            (0, 16, True),
            (16, 15, False),
            (16, 16, True),
            (16, 17, True),
        ]:
            with self.subTest(threshold=threshold, nbytes=nbytes):
                plans = _create_cooperative_plans(
                    [_plan(a, nbytes=nbytes), _plan(a, nbytes=nbytes)],
                    global_ranks=(0, 1),
                    min_tensor_bytes=threshold,
                )
                if expect_coop:
                    self.assertEqual(1, len(plans[0].planner_data))
                    self.assertEqual(1, sum(len(p.items) for p in plans))
                else:
                    self.assertEqual([(), ()], [p.planner_data for p in plans])
                    self.assertEqual([1, 1], [len(p.items) for p in plans])

    def test_owner_byte_balancing_largest_first_least_loaded(self):
        # Distinct nbytes: 300, 200, 100.
        items = [_read_item("L"), _read_item("M"), _read_item("S")]
        sizes = {"L": 300, "M": 200, "S": 100}

        def make_plan(order):
            ordered = [
                next(i for i in items if i.dest_index.fqn == name) for name in order
            ]
            metadata = {
                _read_signature(item): (
                    (8,),
                    "torch.float32",
                    "cpu",
                    sizes[item.dest_index.fqn],
                )
                for item in ordered
            }
            return LoadPlan(items=ordered, planner_data=metadata)

        # Different local orders must not change owners / schedule identity order.
        plans = _create_cooperative_plans(
            [make_plan(["S", "L", "M"]), make_plan(["M", "S", "L"])],
            global_ranks=(0, 1),
            min_tensor_bytes=0,
        )
        schedule = plans[0].planner_data
        self.assertEqual(schedule, plans[1].planner_data)
        by_fqn = {t.read_item.dest_index.fqn: t.src for t in schedule}
        # After L→0 (300) and M→1 (200), S(100) goes to least-loaded rank1.
        self.assertEqual({"L": 0, "M": 1, "S": 1}, by_fqn)
        self.assertEqual(["L", "M", "S"], _schedule_fqns(schedule))


if __name__ == "__main__":
    run_tests()
