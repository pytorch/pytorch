from __future__ import annotations

import unittest

from tools.zippo.sharding import (
    assign_shards,
    filter_by_marker,
)


def _entries(
    prefix: str, count: int, markers: tuple[str, ...] = ("cuda",)
) -> list[dict]:
    return [
        {"test": f"{prefix}::test_{i}", "markers": list(markers)}
        for i in range(count)
    ]


def _ids(prefix: str, count: int) -> list[str]:
    return [f"{prefix}::test_{i}" for i in range(count)]


class TestAssignShards(unittest.TestCase):
    def test_invalid_n_shards(self) -> None:
        with self.assertRaises(ValueError):
            assign_shards({"a.py": _entries("a.py", 1)}, 0)

    def test_empty(self) -> None:
        self.assertEqual(assign_shards({}, 3), [[], [], []])

    def test_even_split(self) -> None:
        # 6 tests across 3 shards = 2 per shard.
        files = {"a.py": _entries("a.py", 6)}
        shards = assign_shards(files, 3)
        self.assertEqual([len(s) for s in shards], [2, 2, 2])

    def test_uneven_split_remainder_to_first(self) -> None:
        # 5 tests across 3 shards: first 2 shards get 2, last gets 1.
        files = {"a.py": _entries("a.py", 5)}
        shards = assign_shards(files, 3)
        self.assertEqual([len(s) for s in shards], [2, 2, 1])

    def test_splits_large_file_across_shards(self) -> None:
        # A 1000-test file across 4 shards must be split into 4 contiguous chunks.
        shards = assign_shards({"big.py": _entries("big.py", 1000)}, 4)
        self.assertEqual([len(s) for s in shards], [250, 250, 250, 250])
        # Order is preserved end-to-end: concatenation reproduces input node IDs.
        self.assertEqual(
            [nid for s in shards for nid in s], _ids("big.py", 1000)
        )

    def test_balanced_many_files(self) -> None:
        files = {f"f{i}.py": _entries(f"f{i}.py", 50) for i in range(10)}
        shards = assign_shards(files, 5)
        self.assertEqual([len(s) for s in shards], [100, 100, 100, 100, 100])
        # Every test ID appears exactly once.
        flat = [nid for s in shards for nid in s]
        self.assertEqual(
            sorted(flat),
            sorted(e["test"] for entries in files.values() for e in entries),
        )

    def test_preserves_file_order(self) -> None:
        # Files are flattened in sorted-path order.
        files = {
            "z.py": _entries("z.py", 3),
            "a.py": _entries("a.py", 3),
            "m.py": _entries("m.py", 3),
        }
        shards = assign_shards(files, 1)
        self.assertEqual(
            shards[0],
            _ids("a.py", 3) + _ids("m.py", 3) + _ids("z.py", 3),
        )

    def test_determinism(self) -> None:
        files = {f"f{i}.py": _entries(f"f{i}.py", i + 1) for i in range(20)}
        first = assign_shards(files, 4)
        for _ in range(5):
            self.assertEqual(assign_shards(files, 4), first)

    def test_skips_empty_files(self) -> None:
        files = {"a.py": [], "b.py": _entries("b.py", 5), "c.py": []}
        shards = assign_shards(files, 2)
        flat = [nid for s in shards for nid in s]
        self.assertEqual(sorted(flat), sorted(_ids("b.py", 5)))

    def test_more_shards_than_tests(self) -> None:
        # 3 tests, 5 shards: first 3 shards get 1 each, last 2 are empty.
        shards = assign_shards({"a.py": _entries("a.py", 3)}, 5)
        self.assertEqual([len(s) for s in shards], [1, 1, 1, 0, 0])

    def test_partition_invariant(self) -> None:
        # Every input test ID appears in exactly one shard.
        files = {
            f"f{i:02d}.py": _entries(f"f{i:02d}.py", 7 * i + 3) for i in range(13)
        }
        all_inputs = [e["test"] for entries in files.values() for e in entries]
        for n in (1, 3, 7, 13, 50):
            shards = assign_shards(files, n)
            flat = [nid for s in shards for nid in s]
            self.assertEqual(
                sorted(flat), sorted(all_inputs), msg=f"n_shards={n}"
            )
            seen: set[str] = set()
            for s in shards:
                for nid in s:
                    self.assertNotIn(nid, seen, msg=f"duplicate in n={n}: {nid}")
                    seen.add(nid)


class TestFilterByMarker(unittest.TestCase):
    def test_empty_input(self) -> None:
        self.assertEqual(filter_by_marker({}, "cuda"), {})

    def test_identity_when_all_match(self) -> None:
        files = {"a.py": _entries("a.py", 3, markers=("cuda",))}
        self.assertEqual(filter_by_marker(files, "cuda"), files)

    def test_mixed_markers_kept_and_dropped(self) -> None:
        files = {
            "a.py": _entries("a.py", 2, markers=("cpu",)),
            "b.py": _entries("b.py", 2, markers=("cuda",)),
            "c.py": _entries("c.py", 2, markers=("cpu", "cuda")),
        }
        cuda = filter_by_marker(files, "cuda")
        self.assertEqual(set(cuda), {"b.py", "c.py"})
        self.assertEqual([e["test"] for e in cuda["b.py"]], _ids("b.py", 2))
        self.assertEqual([e["test"] for e in cuda["c.py"]], _ids("c.py", 2))

        cpu = filter_by_marker(files, "cpu")
        self.assertEqual(set(cpu), {"a.py", "c.py"})

    def test_multi_marker_test_matches_each(self) -> None:
        files = {"a.py": _entries("a.py", 1, markers=("cpu", "cuda"))}
        self.assertEqual(set(filter_by_marker(files, "cpu")), {"a.py"})
        self.assertEqual(set(filter_by_marker(files, "cuda")), {"a.py"})

    def test_unknown_marker_returns_empty(self) -> None:
        files = {"a.py": _entries("a.py", 3, markers=("cuda",))}
        self.assertEqual(filter_by_marker(files, "mps"), {})

    def test_drops_files_with_zero_matches(self) -> None:
        files = {
            "a.py": _entries("a.py", 2, markers=("cpu",)),
            "b.py": _entries("b.py", 2, markers=("cuda",)),
        }
        self.assertNotIn("a.py", filter_by_marker(files, "cuda"))
        self.assertNotIn("b.py", filter_by_marker(files, "cpu"))

    def test_missing_markers_field_does_not_match(self) -> None:
        # Defensive: if `markers` is absent or empty, no filter ever matches.
        files = {"a.py": [{"test": "a.py::test_0"}]}
        self.assertEqual(filter_by_marker(files, "cuda"), {})

    def test_count_after_filter(self) -> None:
        files = {
            "a.py": _entries("a.py", 600, markers=("cuda",)),
            "b.py": _entries("b.py", 200, markers=("cpu",)),
        }
        cuda = filter_by_marker(files, "cuda")
        cpu = filter_by_marker(files, "cpu")
        self.assertEqual(sum(len(v) for v in cuda.values()), 600)
        self.assertEqual(sum(len(v) for v in cpu.values()), 200)


if __name__ == "__main__":
    unittest.main()
