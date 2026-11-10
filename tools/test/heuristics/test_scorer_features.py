from __future__ import annotations

import functools
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT))

import tools.stats.import_test_stats as import_test_stats
import tools.testing.do_target_determination_for_s3 as td_job
import tools.testing.target_determination.heuristics.interface as interface
from tools.test.heuristics import scorer_fixture as fixture
from tools.testing.target_determination.scorer import features, infer
from tools.testing.target_determination.scorer.model import (
    load_model,
    MAX_FILE_BYTES,
    Model,
    RiskScore,
)
from tools.testing.test_run import TestRun


sys.path.remove(str(REPO_ROOT))

TESTS = ["test_a", "test_b", "test_c", "dir/test_d"]
REGISTERED = list(features.SERVED_FEATURES.values()) + ["Profiling"]


def make_heuristic(name: str) -> interface.HeuristicInterface:
    class Heuristic(interface.HeuristicInterface):
        def __init__(self) -> None:
            super().__init__()

        def get_prediction_confidence(
            self, tests: list[str]
        ) -> interface.TestPrioritizations:
            return interface.TestPrioritizations(tests, {})

    heuristic: interface.HeuristicInterface = type(name, (Heuristic,), {})()
    return heuristic


def make_aggregated(
    scores: dict[str, dict[str, float]], tests: list[str] = TESTS
) -> interface.AggregatedHeuristics:
    aggregated = interface.AggregatedHeuristics(tests)
    for name, by_test in scores.items():
        aggregated.add_heuristic_results(
            make_heuristic(name),
            interface.TestPrioritizations(
                tests, {TestRun(t): s for t, s in by_test.items()}
            ),
        )
    return aggregated


def fake_sources(
    changed: list[str] | None = None,
    pr_number: int | None = 123,
    numstat: str | Exception = "",
) -> features.Sources:
    def git(args: list[str], timeout: float) -> str:
        if isinstance(numstat, Exception):
            raise numstat
        return numstat

    return features.Sources(
        pr_number=lambda: pr_number,
        changed_files=lambda: list(changed or []),
        merge_base=lambda: "base",
        git=git,
    )


TIMES = {"linux": {"default": {"test_a": 10.0, "test_b": 20.0}}}
ALL_SCORES = {
    "PreviouslyFailedInPR": {"test_a": 1.0},
    "EditedByPR": {"test_b": 1.0},
    "MentionedInPR": {"test_c": 1.0},
    "CorrelatedWithHistoricalFailures": {"test_a": 0.25, "test_b": 0.125},
    "LLM": {"dir/test_d": 0.25},
    "Filepath": {"test_c": 0.1875},
    "PublicBindings": {"test_b": 1.0},
}


class TestAccessor(unittest.TestCase):
    def test_get_heuristic_results(self) -> None:
        aggregated = make_aggregated({"EditedByPR": {"test_b": 1.0}})
        results = aggregated.get_heuristic_results("EditedByPR")
        if results is None:
            self.fail("no results for EditedByPR")
        self.assertEqual(results.get_scored_tests()[0], (1.0, TestRun("test_b")))
        self.assertIsNone(aggregated.get_heuristic_results("LLM"))


class TestFeatures(unittest.TestCase):
    def test_each_served_feature(self) -> None:
        names = list(features.SERVED_FEATURES)
        unit = features.Unit(fake_sources())
        rows = features.build_rows(
            names, TESTS, make_aggregated(ALL_SCORES), {}, unit, REGISTERED
        )
        for name, heuristic in features.SERVED_FEATURES.items():
            column = [row[names.index(name)] for row in rows]
            expected = [ALL_SCORES[heuristic].get(t, 0.0) for t in TESTS]
            self.assertEqual(column, expected, name)

    def test_class_level_scores_are_not_the_file_score(self) -> None:
        aggregated = interface.AggregatedHeuristics(TESTS)
        aggregated.add_heuristic_results(
            make_heuristic("EditedByPR"),
            interface.TestPrioritizations(TESTS, {TestRun("test_a::TestX"): 1.0}),
        )
        rows = features.build_rows(
            ["edited"], TESTS, aggregated, {}, features.Unit(fake_sources()), REGISTERED
        )
        self.assertEqual([row[0] for row in rows], [0.0, 0.0, 0.0, 0.0])

    def test_docs_only_filepath(self) -> None:
        changed = fixture.STATS_CASE["docs_only_changed"]
        aggregated = make_aggregated({"Filepath": dict.fromkeys(TESTS, -1.0)})
        unit = features.Unit(fake_sources(changed))
        rows = features.build_rows(
            ["filepath_s", "docs_only"], TESTS, aggregated, {}, unit, REGISTERED
        )
        self.assertEqual(rows, [[-1.0, fixture.STATS_CASE["docs_only_expected"]]] * 4)

    def test_raised_heuristic_is_zero(self) -> None:
        scores = {k: v for k, v in ALL_SCORES.items() if k != "LLM"}
        rows = features.build_rows(
            ["llm_n", "edited"],
            TESTS,
            make_aggregated(scores),
            {},
            features.Unit(fake_sources()),
            REGISTERED,
        )
        self.assertEqual([row[0] for row in rows], [0.0] * 4)
        self.assertEqual([row[1] for row in rows], [0.0, 1.0, 0.0, 0.0])

    def test_unregistered_heuristic_raises(self) -> None:
        with self.assertRaises(features.UnregisteredHeuristic):
            features.build_rows(
                ["llm_n"],
                TESTS,
                make_aggregated(ALL_SCORES),
                {},
                features.Unit(fake_sources()),
                [h for h in REGISTERED if h != "LLM"],
            )

    def test_test_seconds_is_median_over_entries(self) -> None:
        times: dict[str, dict[str, dict[str, float]]] = {
            "env1": {
                "default": {"test_a": 10.0, "test_b": 4.0},
                "slow": {"test_a": 30},
            },
            "env2": {"default": {"test_a": 20.0}},
        }
        rows = features.build_rows(
            ["test_seconds"],
            TESTS,
            make_aggregated({}),
            times,
            features.Unit(fake_sources()),
            REGISTERED,
        )
        self.assertEqual([row[0] for row in rows], [20.0, 4.0, 0.0, 0.0])

    def test_unit_features(self) -> None:
        names = list(features.UNIT_FEATURES)
        unit = features.Unit(fake_sources(fixture.STATS_CASE["changed"] + [""]))
        rows = features.build_rows(
            names, TESTS, make_aggregated({}), {}, unit, REGISTERED
        )
        expected = {"is_pr": 1.0, **fixture.STATS_CASE["expected"]}
        self.assertEqual(rows[0], [expected[name] for name in names])
        unit = features.Unit(fake_sources(pr_number=None))
        self.assertEqual(unit.value("is_pr"), 0.0)

    def test_change_stats_fixture(self) -> None:
        self.assertEqual(
            features.change_stats(fixture.STATS_CASE["changed"]),
            fixture.STATS_CASE["expected"],
        )
        self.assertEqual(
            features.change_stats(fixture.STATS_CASE["docs_only_changed"])["docs_only"],
            fixture.STATS_CASE["docs_only_expected"],
        )

    def test_changed_files_failure_propagates(self) -> None:
        def boom() -> list[str]:
            raise RuntimeError("Unable to get changed files")

        sources = fake_sources()
        sources.changed_files = boom
        with self.assertRaises(RuntimeError):
            features.Unit(sources).value("n_files")

    def test_known_names(self) -> None:
        self.assertNotIn("coedit_n", features.MODEL_FEATURES)
        self.assertNotIn("profiling_n", features.MODEL_FEATURES)
        self.assertEqual(len(features.MODEL_FEATURES), 7 + 1 + 1 + 10)
        self.assertEqual(len(features.RISK_FEATURES), 14)
        for name in ("sum_p", "max_p", "max_llm"):
            self.assertNotIn(name, features.RISK_FEATURES)
        self.assertLessEqual(set(features.PATH_FEATURES), features.RISK_FEATURES)


def numstat_z(records: list[dict[str, Any]]) -> str:
    """`git diff --numstat -z -M` output for the fixture's records."""
    out = []
    for record in records:
        added = "-" if record["added"] is None else str(record["added"])
        deleted = "-" if record["deleted"] is None else str(record["deleted"])
        if record["old_path"] is None:
            out.append(f"{added}\t{deleted}\t{record['path']}\0")
        else:
            out.append(
                f"{added}\t{deleted}\t\0{record['old_path']}\0{record['path']}\0"
            )
    return "".join(out)


class TestNumstat(unittest.TestCase):
    def test_fixture_case(self) -> None:
        case = fixture.NUMSTAT_CASE
        added, deleted = features.parse_numstat(numstat_z(case["records"]))
        self.assertEqual(added, case["expected"]["sig_added"])
        self.assertEqual(deleted, case["expected"]["sig_deleted"])
        self.assertAlmostEqual(
            math.log1p(added), case["expected"]["log_sig_added"], delta=1e-11
        )
        self.assertAlmostEqual(
            math.log1p(deleted), case["expected"]["log_sig_deleted"], delta=1e-11
        )

    def test_rename_counts_under_new_path(self) -> None:
        to_docs = "4\t2\t\0torch/a.py\0docs/a.md\0"
        self.assertEqual(features.parse_numstat(to_docs), (0.0, 0.0))
        from_docs = "4\t2\t\0docs/a.md\0torch/a.py\0"
        self.assertEqual(features.parse_numstat(from_docs), (4.0, 2.0))

    def test_parse_edge_cases(self) -> None:
        text = "-\t-\ttorch/lib.so\0garbage\0001\t4\taten/x.cpp\0"
        self.assertEqual(features.parse_numstat(text), (1.0, 4.0))
        self.assertEqual(features.parse_numstat(""), (0.0, 0.0))
        self.assertEqual(
            features.parse_numstat("3\t0\ttorch/tab\there.py\0"), (3.0, 0.0)
        )
        self.assertEqual(features.parse_numstat("4\t2\t\0torch/a.py"), (0.0, 0.0))

    def test_unit_lines_and_command(self) -> None:
        calls: list[tuple[list[str], float]] = []

        def git(args: list[str], timeout: float) -> str:
            calls.append((args, timeout))
            return "3\t1\ttorch/a.py\0"

        sources = fake_sources()
        sources.git = git
        self.assertEqual(features.Unit(sources).significant_lines(), (3.0, 1.0))
        args, timeout = calls[0]
        self.assertEqual(args, ["diff", "--numstat", "-z", "-M", "base", "HEAD"])
        self.assertEqual(timeout, features.GIT_TIMEOUT_SECONDS)

    def test_timeout_gives_nan(self) -> None:
        error = subprocess.TimeoutExpired(["git"], 60)
        with mock.patch("sys.stdout", io.StringIO()):
            added, deleted = features.Unit(
                fake_sources(numstat=error)
            ).significant_lines()
        self.assertTrue(math.isnan(added) and math.isnan(deleted))

    def test_risk_values(self) -> None:
        unit = features.Unit(fake_sources(["torch/a.py"], numstat="3\t0\ttorch/a.py\0"))
        values = features.risk_values(features.RISK_FEATURES, unit, CASE_PATHS)
        self.assertEqual(set(values), set(features.RISK_FEATURES))
        self.assertEqual(values["log_files"], math.log1p(1))
        self.assertEqual(values["log_sig_added"], math.log1p(3))
        self.assertEqual(values["path_revert_max"], math.log(0.12))
        self.assertEqual(values["path_revert_mean"], math.log(0.12))
        self.assertEqual(features.risk_values([], unit, CASE_PATHS), {})
        only_paths = features.risk_values(features.PATH_FEATURES, unit, CASE_PATHS)
        self.assertEqual(set(only_paths), set(features.PATH_FEATURES))
        self.assertEqual(features.risk_values(features.PATH_FEATURES, unit, None), {})


CASE_PATHS = features.Paths(
    fixture.PATHS_CASE["paths"]["base"], fixture.PATHS_CASE["paths"]["nodes"]
)


class TestPathValues(unittest.TestCase):
    def assert_values(self, files: list[str], expected: dict[str, float]) -> None:
        values = features.path_values(CASE_PATHS, files)
        self.assertEqual(set(values), set(features.PATH_FEATURES))
        for name in features.PATH_FEATURES:
            self.assertAlmostEqual(values[name], expected[name], delta=1e-11)

    def test_fixture_cases(self) -> None:
        case = fixture.PATHS_CASE
        self.assert_values(case["changed"], case["expected"])
        self.assert_values(case["docs_changed"], case["docs_expected"])
        self.assert_values(case["long_changed"], case["long_expected"])

    def test_no_files_is_nan(self) -> None:
        values = features.path_values(CASE_PATHS, [])
        self.assertTrue(all(math.isnan(v) for v in values.values()))
        self.assertEqual(set(values), set(features.PATH_FEATURES))

    def test_lookup_order(self) -> None:
        rate = functools.partial(features.path_rate, CASE_PATHS)
        self.assertEqual(rate("torch/_inductor/ir.py"), 0.4)
        self.assertEqual(rate("torch/_inductor/x/y.py"), 0.25)
        self.assertEqual(rate("torch/nn/a.py"), 0.12)
        self.assertEqual(rate("setup.py"), 0.1)
        self.assertEqual(rate("torchx/a.py"), 0.1)
        self.assertEqual(rate("docs/a.md"), 0.0)
        paths = features.Paths(0.1, {"/": 0.3, "a//": 0.2})
        self.assertEqual(features.path_rate(paths, "/x.py"), 0.3)
        self.assertEqual(features.path_rate(paths, "a//b.py"), 0.2)

    def test_keys_longer_than_the_limit_never_match(self) -> None:
        long_dir = "a/" + "b" * 600 + "/"
        paths = features.Paths(0.1, {long_dir: 0.9, long_dir + "c.py": 0.8})
        self.assertEqual(features.path_rate(paths, long_dir + "c.py"), 0.1)
        edge = "d" * (features.MAX_PATH_KEY_CHARS - 1) + "/"
        paths = features.Paths(0.1, {edge: 0.7})
        self.assertEqual(features.path_rate(paths, edge + "e.py"), 0.7)

    def test_files_deduplicated_sorted_and_capped(self) -> None:
        paths = features.Paths(0.0, {"z/": 1.0, "a/": 0.5})
        many = [f"a/{i:05d}.py" for i in range(features.MAX_CHANGED_PATHS)]
        values = features.path_values(paths, ["z/last.py", *many])
        self.assertEqual(values["path_revert_max"], math.log(0.5))
        values = features.path_values(paths, ["z/x.py", "a/y.py", "a/y.py"])
        self.assertAlmostEqual(values["path_revert_mean"], math.log(0.75))


class ScorerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)
        self.model_path = self.root / "td_scorer.json"
        self.ratings_path = self.root / "test-file-ratings.json"
        self.ratings_path.write_text(json.dumps({"torch/a.py": {"test_a": 1.0}}))
        self.stdout = mock.patch("sys.stdout", io.StringIO())
        self.stdout.start()

    def tearDown(self) -> None:
        self.stdout.stop()
        self.dir.cleanup()

    def write_model(self, doc: Any = fixture.DOC) -> None:
        self.model_path.write_text(json.dumps(doc))

    def compute(self, **kwargs: Any) -> infer.ScorerResult:
        args: dict[str, Any] = {
            "unit": features.Unit(fake_sources(["torch/a.py"])),
            "registered": REGISTERED,
            "ratings_path": self.ratings_path,
        }
        args.update(kwargs)
        aggregated = args.pop("aggregated", make_aggregated(ALL_SCORES))
        times = args.pop("times", TIMES)
        return infer.compute_scorer(TESTS, aggregated, times, self.model_path, **args)


class TestComputeScorer(ScorerTestCase):
    def test_block(self) -> None:
        self.write_model()
        block, metric = self.compute()
        self.assertEqual(set(block["p"]), set(TESTS))
        self.assertEqual(block["trained_at"], "2026-11-10T00:00:00Z")
        self.assertIsNotNone(block["risk"])
        self.assertEqual(block["budget_share"], metric["budget_share"])
        self.assertIsNone(metric["fallback"])
        self.assertEqual(metric["heuristics_raised"], ["Profiling"])
        self.assertEqual(metric["n_tests"], len(TESTS))
        json.dumps(block, allow_nan=False)

    def test_fallbacks(self) -> None:
        self.assertEqual(self.compute().block, {"fallback": "missing"})
        self.write_model({**fixture.DOC, "format": 2})
        block, metric = self.compute()
        self.assertEqual(block, {"fallback": "format"})
        self.assertEqual(metric["fallback"], "format")
        self.write_model(
            {**fixture.DOC, "features": ["llm_n", "test_seconds", "is_pr"]}
        )
        reason = self.compute(registered=["Filepath"]).block["fallback"]
        self.assertTrue(reason.startswith("UnregisteredHeuristic: llm_n"))

    def test_heuristics_raised(self) -> None:
        self.write_model()
        scores = {k: v for k, v in ALL_SCORES.items() if k != "LLM"}
        block, metric = self.compute(aggregated=make_aggregated(scores))
        self.assertIn("p", block)
        self.assertEqual(metric["heuristics_raised"], ["LLM", "Profiling"])

    def test_no_test_times(self) -> None:
        self.write_model()
        for times in ({}, {"linux": {"default": {"other_test": 5.0}}}):
            block, metric = self.compute(times=times)
            self.assertEqual(block, {"fallback": "no test times"})
            self.assertEqual(metric["fallback"], "no test times")

    def test_no_llm_scores(self) -> None:
        self.write_model(
            {**fixture.DOC, "features": ["fail_n", "test_seconds", "llm_n"]}
        )
        self.assertIn("p", self.compute().block)
        for scores in (
            {k: v for k, v in ALL_SCORES.items() if k != "LLM"},
            {**ALL_SCORES, "LLM": {}},
        ):
            block = self.compute(aggregated=make_aggregated(scores)).block
            self.assertEqual(block, {"fallback": "no LLM scores"})

    def test_no_failure_ratings(self) -> None:
        self.write_model()
        for contents in (None, "{}", "[1]", "{bad"):
            if contents is None:
                self.ratings_path.unlink()
            else:
                self.ratings_path.write_text(contents)
            block = self.compute().block
            self.assertEqual(block, {"fallback": "no failure ratings"}, contents)

    def test_checks_only_features_the_model_reads(self) -> None:
        self.write_model({**fixture.DOC, "features": ["edited", "is_pr", "n_files"]})
        self.ratings_path.unlink()
        block = self.compute(times={}, aggregated=make_aggregated({})).block
        self.assertIn("p", block)

    def test_exception_is_truncated(self) -> None:
        self.write_model()
        with mock.patch.object(
            infer, "build_rows", side_effect=RuntimeError("x" * 1000)
        ):
            block, _ = self.compute()
        self.assertEqual(len(block["fallback"]), infer.MAX_REASON_CHARS)
        self.assertTrue(block["fallback"].startswith("RuntimeError: xxx"))

    def test_non_finite_risk_falls_back(self) -> None:
        self.write_model()
        with mock.patch.object(
            Model, "risk", return_value=RiskScore(math.nan, 0.5, 0.1)
        ):
            block, _ = self.compute()
        self.assertTrue(block["fallback"].startswith("ValueError"))

    def test_risk_reads_the_change_alone(self) -> None:
        self.write_model()
        changed = ["torch/_inductor/ir.py", "aten/x.cpp"]
        block = self.compute(unit=features.Unit(fake_sources(changed))).block
        values = {
            "log_files": math.log1p(2),
            **features.path_values(CASE_PATHS, changed),
        }
        expected = load_model(
            self.model_path, features.MODEL_FEATURES, features.RISK_FEATURES
        ).risk(values)
        if expected is None:
            self.fail("no risk score")
        self.assertEqual(block["risk"], expected.risk)
        self.assertEqual(block["budget_share"], expected.share)
        with mock.patch.object(Model, "predict", return_value=[0.9] * len(TESTS)):
            again = self.compute(unit=features.Unit(fake_sources(changed))).block
        self.assertEqual(again["risk"], block["risk"])

    def test_risk_failure_keeps_p(self) -> None:
        self.write_model()

        def boom() -> list[str]:
            raise RuntimeError("no git")

        sources = fake_sources()
        sources.changed_files = boom
        block, _ = self.compute(unit=features.Unit(sources))
        self.assertEqual(set(block["p"]), set(TESTS))
        self.assertIsNone(block["risk"])


class TestTDJob(ScorerTestCase):
    def run_job(
        self, ci: bool = True, emit_side_effect: Any = None
    ) -> tuple[str, mock.MagicMock]:
        aggregated = make_aggregated(ALL_SCORES)
        (self.root / ".additional_ci_files").mkdir(exist_ok=True)
        compute = functools.partial(
            infer.compute_scorer,
            unit=features.Unit(fake_sources(["torch/a.py"])),
            registered=REGISTERED,
            ratings_path=self.ratings_path,
        )
        downloads: dict[str, Any] = {
            name: mock.patch.object(
                td_job, name, return_value=TIMES if name == "get_test_times" else {}
            )
            for name in (
                "get_test_times",
                "get_test_class_times",
                "get_test_file_ratings",
                "get_test_class_ratings",
                "get_td_heuristic_historial_edited_files_json",
                "get_td_heuristic_profiling_json",
                "copy_pytest_cache",
                "copy_additional_previous_failures",
                "get_td_scorer",
            )
        }
        with (
            mock.patch.object(td_job, "REPO_ROOT", self.root),
            mock.patch.object(td_job, "TESTS", TESTS),
            mock.patch.object(
                td_job, "get_test_prioritizations", return_value=aggregated
            ),
            mock.patch.object(td_job, "compute_scorer", compute),
            mock.patch.object(
                td_job, "emit_metric", side_effect=emit_side_effect
            ) as emit,
            mock.patch.dict(os.environ, {"CI": "true" if ci else "false"}),
        ):
            for patch in downloads.values():
                patch.start()
            try:
                td_job.main()
            finally:
                for patch in downloads.values():
                    patch.stop()
        return (self.root / "td_results.json").read_text(), emit

    def td_only(self) -> str:
        tp = make_aggregated(ALL_SCORES).get_aggregated_priorities()
        return json.dumps(tp.to_json())

    def test_td_scores_unchanged_and_scorer_added(self) -> None:
        (self.root / ".additional_ci_files").mkdir()
        (self.root / ".additional_ci_files" / "td_scorer.json").write_text(
            json.dumps(fixture.DOC)
        )
        text, emit = self.run_job()
        self.assertTrue(text.startswith(self.td_only()[:-1] + ", "))
        result = json.loads(text)
        self.assertEqual(set(result["scorer"]["p"]), set(TESTS))
        names = [c.args[0] for c in emit.call_args_list]
        self.assertEqual(
            names,
            [
                "td_results_final_test_prioritizations",
                "td_results_aggregated_heuristics",
                "td_scorer",
            ],
        )
        self.assertIsNone(emit.call_args_list[2].args[1]["fallback"])
        imported = self.import_results_from(result)
        self.assertEqual(
            imported.get_all_tests(),
            make_aggregated(ALL_SCORES).get_aggregated_priorities().get_all_tests(),
        )

    def import_results_from(self, result: dict[str, Any]) -> Any:
        ci_files = self.root / ".additional_ci_files"
        (ci_files / "td_results.json").write_text(json.dumps(result))
        with mock.patch.object(td_job, "REPO_ROOT", self.root):
            return td_job.import_results()

    def test_missing_model_falls_back(self) -> None:
        text, _ = self.run_job()
        self.assertEqual(json.loads(text)["scorer"], {"fallback": "missing"})
        self.assertTrue(text.startswith(self.td_only()[:-1] + ", "))

    def test_no_metrics_outside_ci(self) -> None:
        text, emit = self.run_job(ci=False)
        emit.assert_not_called()
        self.assertEqual(json.loads(text)["scorer"], {"fallback": "missing"})

    def test_td_scorer_metric_failure_does_not_stop_the_job(self) -> None:
        def emit_error(name: str, info: dict[str, Any]) -> None:
            if name == "td_scorer":
                raise RuntimeError("upload failed")

        text, emit = self.run_job(emit_side_effect=emit_error)
        self.assertEqual(emit.call_count, 3)
        self.assertEqual(json.loads(text)["scorer"], {"fallback": "missing"})


class TestDownload(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)
        self.path = self.root / ".additional_ci_files" / "td_scorer.json"
        patches: list[Any] = [
            mock.patch.object(import_test_stats, "REPO_ROOT", self.root),
            mock.patch("sys.stdout", io.StringIO()),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def tearDown(self) -> None:
        self.dir.cleanup()

    def urlopen(self, contents: bytes) -> mock.MagicMock:
        response = mock.MagicMock()
        response.__enter__.return_value.read.side_effect = lambda n: contents[:n]
        return mock.MagicMock(return_value=response)

    def test_downloads(self) -> None:
        opener = self.urlopen(b'{"format": 1}')
        with mock.patch.object(import_test_stats, "urlopen", opener):
            import_test_stats.get_td_scorer()
        self.assertEqual(self.path.read_bytes(), b'{"format": 1}')
        self.assertIn("timeout", opener.call_args.kwargs)

    def test_too_large_leaves_no_file(self) -> None:
        self.path.parent.mkdir()
        self.path.write_text("stale")
        opener = self.urlopen(b" " * (MAX_FILE_BYTES + 10))
        with mock.patch.object(import_test_stats, "urlopen", opener):
            import_test_stats.get_td_scorer()
        self.assertFalse(self.path.exists())

    def test_failure_leaves_no_file(self) -> None:
        self.path.parent.mkdir()
        self.path.write_text("stale")
        opener = mock.MagicMock(side_effect=OSError("network down"))
        with mock.patch.object(import_test_stats, "urlopen", opener):
            import_test_stats.get_td_scorer()
        self.assertFalse(self.path.exists())
        self.assertEqual(opener.call_count, 3)


if __name__ == "__main__":
    unittest.main()
