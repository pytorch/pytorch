from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest import mock

from tools.ramjet_schedule import load_history, order_shards, update_history
from tools.ramjet_source import BOOTSTRAP

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


CPU = "tci.platform:linux_x86_64"
L4 = "tci.platform:linux_x86_64_cuda_l4"
FIRST = "test/test_first.py::Suite::test_first"
SECOND = "test/test_second.py::Suite::test_second"


def result(nodes, seconds, platform=CPU):
    return {
        "returncode": 0,
        "nodes": nodes,
        "platform": platform,
        "started_at": 100,
        "finished_at": 100 + seconds,
    }


@instantiate_parametrized_tests
class TestRamjetSchedule(TestCase):
    root: Path
    path: Path

    def setUp(self):
        super().setUp()
        temporary = tempfile.TemporaryDirectory(prefix="ramjet-schedule-tests-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "timings.json"

    def test_longest_batches_first_preserves_every_input_and_ties(self):
        unknown = ["test/test_unknown.py::test_unknown"]
        fast = [FIRST]
        slow = [SECOND, SECOND + "[parameter::value]"]
        tied = ["test/test_tied.py::test_tied"]
        source = self.root / "run-previous"
        self.assertEqual(
            update_history(
                self.path,
                [result(fast, 10), result(slow, 40), result(tied, 10)],
                source,
            ),
            3,
        )
        reversed_slow = list(reversed(slow))
        batches = [unknown, fast, reversed_slow, tied, unknown]
        ordered, counts = order_shards(batches, CPU, load_history(self.path))
        self.assertEqual(ordered, [reversed_slow, fast, tied, unknown, unknown])
        self.assertEqual(sorted(map(id, ordered)), sorted(map(id, batches)))
        self.assertEqual(batches, [unknown, fast, reversed_slow, tied, unknown])
        self.assertEqual(
            counts,
            {
                "exact_batches": 3,
                "estimated_batches": 0,
                "unknown_batches": 2,
                "sources": [str(source)],
                "estimated_worker_seconds": 60,
            },
        )

    def test_partial_history_estimates_changed_batches_by_file(self):
        original = [FIRST + str(i) for i in range(4)]
        update_history(
            self.path,
            [result(original, 20), result([FIRST], 10), result([SECOND], 6)],
            self.root / "run-previous",
        )
        changed: list[str] = [FIRST + "_new", FIRST + "_another"]
        batches: list[list[str]] = [[SECOND], changed, [FIRST]]
        ordered, counts = order_shards(batches, CPU, load_history(self.path))
        self.assertEqual(ordered, [changed, [FIRST], [SECOND]])
        self.assertEqual(counts["exact_batches"], 2)
        self.assertEqual(counts["estimated_batches"], 1)
        self.assertEqual(counts["estimated_worker_seconds"], 28)

    def test_updates_preserve_other_batches_and_platforms(self):
        first_source = self.root / "run-first"
        second_source = self.root / "run-second"
        update_history(
            self.path,
            [result([FIRST], 50), result([SECOND], 5), result([FIRST], 1, L4)],
            first_source,
        )
        update_history(self.path, [result([SECOND], 60)], second_source)
        history = load_history(self.path)
        batches = [[FIRST], [SECOND]]
        cpu, counts = order_shards(batches, CPU, history)
        self.assertEqual(cpu, [[SECOND], [FIRST]])
        self.assertEqual(counts["sources"], [str(first_source), str(second_source)])
        gpu, counts = order_shards(batches, L4, history)
        self.assertEqual(gpu, batches)
        self.assertEqual(counts["exact_batches"], 1)
        self.assertEqual(counts["unknown_batches"], 1)
        self.assertEqual(counts["estimated_worker_seconds"], 1)

    def test_no_history_preserves_existing_order_and_empty_selection(self):
        batches = [[SECOND], [FIRST], [SECOND]]
        history = load_history(self.path)
        ordered, counts = order_shards(batches, CPU, history)
        self.assertEqual(ordered, batches)
        self.assertEqual(counts["unknown_batches"], 3)
        self.assertEqual(counts["sources"], [])
        self.assertEqual(order_shards([], CPU, history)[0], [])
        self.assertEqual(update_history(self.path, [], self.root), 0)
        self.assertFalse(self.path.exists())

    @parametrize(
        "content",
        [
            "{truncated",
            "[]",
            '{"version":2,"platforms":{}}',
            '{"version":1,"platforms":[]}',
            '{"version":1,"platforms":{"cpu":{}}}',
        ],
    )
    def test_corrupt_history_uses_existing_order(self, content):
        self.path.write_text(content)
        with mock.patch("builtins.print") as warning:
            history = load_history(self.path)
        batches = [[SECOND], [FIRST]]
        self.assertEqual(order_shards(batches, CPU, history)[0], batches)
        warning.assert_called_once()

    @parametrize("seconds", [-1, float("nan"), float("inf"), True, "10"])
    def test_invalid_cached_duration_uses_existing_order(self, seconds):
        update_history(self.path, [result([FIRST], 20)], self.root)
        history = json.loads(self.path.read_text())
        history["platforms"][CPU]["files"]["test/test_first.py"]["seconds"] = seconds
        self.path.write_text(json.dumps(history))
        with mock.patch("builtins.print"):
            history = load_history(self.path)
        batches = [[SECOND], [FIRST]]
        self.assertEqual(order_shards(batches, CPU, history)[0], batches)

    @parametrize(
        "changes",
        [
            {"returncode": 1},
            {"returncode": False},
            {"nodes": []},
            {"nodes": [FIRST, FIRST]},
            {"nodes": [FIRST, None]},
            {"nodes": [FIRST, SECOND]},
            {"nodes": ["/test/test_first.py::test_first"]},
            {"nodes": ["../test_first.py::test_first"]},
            {"nodes": ["test/test_first.py"]},
            {"nodes": ["test/test_first.py::"]},
            {"platform": None},
            {"platform": "cpu"},
            {"started_at": float("nan")},
            {"started_at": -1},
            {"started_at": 200},
            {"finished_at": float("inf")},
            {"finished_at": None},
            {"worker_errors": [{"error": "mismatched input"}]},
        ],
    )
    def test_invalid_or_failed_records_do_not_replace_previous_timings(self, changes):
        update_history(self.path, [result([FIRST], 30)], self.root)
        original = self.path.read_bytes()
        record = {**result([FIRST], 10), **changes}
        self.assertEqual(update_history(self.path, [record], self.root), 0)
        self.assertEqual(self.path.read_bytes(), original)

    def test_write_replaces_the_complete_cache_atomically(self):
        update_history(self.path, [result([FIRST], 10)], self.root)
        original = self.path.read_bytes()
        replace = os.replace

        def publish(source, destination):
            self.assertEqual(self.path.read_bytes(), original)
            self.assertEqual(Path(source).parent, self.path.parent)
            contents = json.loads(Path(source).read_text())
            self.assertEqual(len(contents["platforms"][CPU]["batches"]), 2)
            replace(source, destination)

        with mock.patch("tools.ramjet_schedule.os.replace", side_effect=publish):
            update_history(self.path, [result([SECOND], 20)], self.root)
        self.assertEqual(list(self.root.iterdir()), [self.path])
        self.assertEqual(len(load_history(self.path)["platforms"][CPU]["batches"]), 2)

    def test_failed_publish_keeps_previous_cache_and_cleans_temporary_file(self):
        update_history(self.path, [result([FIRST], 10)], self.root)
        original = self.path.read_bytes()
        with mock.patch(
            "tools.ramjet_schedule.os.replace", side_effect=OSError("full")
        ):
            with self.assertRaisesRegex(OSError, "full"):
                update_history(self.path, [result([SECOND], 20)], self.root)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.root.iterdir()), [self.path])

    def test_scheduler_is_frozen_with_runner_bootstrap(self):
        self.assertIn("tools/ramjet_schedule.py", BOOTSTRAP)


if __name__ == "__main__":
    run_tests()
