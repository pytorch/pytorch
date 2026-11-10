from __future__ import annotations

import copy
import json
import math
import sys
import tempfile
import unittest
from email.message import Message
from pathlib import Path
from typing import Any, Self
from unittest import mock
from urllib.error import HTTPError, URLError


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT))

import tools.testing.do_target_determination_for_s3 as td_job
from tools.stats import import_test_stats
from tools.testing.target_determination import scorer, scorer_model
from tools.testing.target_determination.heuristics.interface import (
    AggregatedHeuristics,
    HeuristicInterface,
    TestPrioritizations,
)
from tools.testing.target_determination.scorer_model import ModelError
from tools.testing.test_run import TestRun


sys.path.remove(str(REPO_ROOT))


def make_model(**overrides: Any) -> dict[str, Any]:
    model: dict[str, Any] = {
        "schema": 1,
        "version": "2026-11-11",
        "enabled": True,
        "overhead_seconds": 10.0,
        "scorer": {
            "intercept": -2.0,
            "cost_weight": 0.5,
            "heuristics": {
                "HeuristicA": {"weight": 1.0, "x": [0.0, 1.0], "y": [0.1, 0.5]},
                "HeuristicB": {"weight": 2.0, "x": [0.0], "y": [0.2]},
            },
            "build_env": {"env1": 0.3},
            "test_config": {"default": -0.1},
        },
        "risk": {
            "intercept": 0.5,
            "features": {"n_files": {"weight": 1.0, "mean": math.log(2), "scale": 1.0}},
            "quantiles": [i / 100 * 2 for i in range(101)],
        },
    }
    model.update(overrides)
    return model


def make_heuristic(name: str) -> HeuristicInterface:
    class Heuristic(HeuristicInterface):
        def __init__(self) -> None:
            super().__init__()

        def get_prediction_confidence(self, tests: list[str]) -> TestPrioritizations:
            return TestPrioritizations(tests, {})

    return type(name, (Heuristic,), {})()  # type: ignore[no-any-return]


HEURISTIC_A = make_heuristic("HeuristicA")
HEURISTIC_B = make_heuristic("HeuristicB")
TESTS = ["test_a", "test_b", "test_c"]


def make_aggregated() -> AggregatedHeuristics:
    aggregated = AggregatedHeuristics(TESTS)
    aggregated.add_heuristic_results(
        HEURISTIC_A,
        TestPrioritizations(
            TESTS, {TestRun("test_a::TestX"): 0.5, TestRun("test_b"): 0.25}
        ),
    )
    return aggregated


def encode(model: Any) -> bytes:
    return json.dumps(model).encode()


class TestValidation(unittest.TestCase):
    def assert_invalid(self, model: Any, message: str) -> None:
        with self.assertRaisesRegex(ModelError, message):
            scorer_model.parse_model(encode(model))

    def mutated(self, path: list[Any], value: Any) -> dict[str, Any]:
        model = make_model()
        target = model
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        return model

    def test_valid_model_round_trips(self) -> None:
        model = make_model()
        self.assertEqual(scorer_model.parse_model(encode(model)), model)

    def test_non_finite_literals(self) -> None:
        raw = encode(make_model()).decode()
        for literal in ("NaN", "Infinity", "-Infinity"):
            bad = raw.replace('"intercept": -2.0', f'"intercept": {literal}')
            with self.assertRaisesRegex(ModelError, "non-finite"):
                scorer_model.parse_model(bad.encode())

    def test_overflowing_numbers(self) -> None:
        raw = encode(make_model()).decode()
        for literal in ("1e999", "1" + "0" * 400):
            bad = raw.replace('"intercept": -2.0', f'"intercept": {literal}')
            with self.assertRaisesRegex(ModelError, "not finite"):
                scorer_model.parse_model(bad.encode())

    def test_bool_is_not_a_number(self) -> None:
        self.assert_invalid(self.mutated(["scorer", "cost_weight"], True), "number")
        self.assert_invalid(self.mutated(["schema"], True), "schema")

    def test_top_level_keys(self) -> None:
        self.assert_invalid(make_model(extra=1), "keys")
        model = make_model()
        del model["risk"]
        self.assert_invalid(model, "keys")
        self.assert_invalid([make_model()], "not an object")

    def test_nested_unknown_keys(self) -> None:
        h = {"weight": 1.0, "x": [0.0], "y": [0.5], "bias": 0.0}
        self.assert_invalid(self.mutated(["scorer", "heuristics", "H"], h), "keys")
        self.assert_invalid(self.mutated(["scorer", "extra"], 0.0), "keys")
        self.assert_invalid(self.mutated(["risk", "extra"], 0.0), "keys")

    def test_scalar_fields(self) -> None:
        self.assert_invalid(self.mutated(["schema"], 2), "schema")
        self.assert_invalid(self.mutated(["version"], 1), "version")
        self.assert_invalid(self.mutated(["enabled"], 1), "enabled")
        self.assert_invalid(self.mutated(["overhead_seconds"], -1), "overhead")
        self.assert_invalid(self.mutated(["overhead_seconds"], 0), "overhead")
        self.assert_invalid(self.mutated(["overhead_seconds"], 601), "overhead")

    def test_heuristic_tables(self) -> None:
        def h(**kw: Any) -> dict[str, Any]:
            return {"weight": 1.0, "x": [0.0, 1.0], "y": [0.1, 0.2], **kw}

        cases = [
            (h(weight=-0.1), "negative"),
            (h(x=[0.1, 1.0]), r"x\[0\]"),
            (h(x=[0.0, 0.0]), "strictly increasing"),
            (h(y=[0.2, 0.1]), "non-decreasing"),
            (h(y=[0.0, 0.1]), "outside"),
            (h(y=[0.1, 1.0]), "outside"),
            (h(y=[0.1]), "pairs"),
            (h(x=[], y=[]), "pairs"),
            (h(x=[float(i) for i in range(513)], y=[0.5] * 513), "pairs"),
            (h(x="0"), "not a list"),
        ]
        for entry, message in cases:
            with self.subTest(entry=entry):
                model = self.mutated(["scorer", "heuristics", "H"], entry)
                self.assert_invalid(model, message)

    def test_name_caps(self) -> None:
        h = {"weight": 1.0, "x": [0.0], "y": [0.5]}
        many = {f"H{i}": h for i in range(65)}
        self.assert_invalid(self.mutated(["scorer", "heuristics"], many), "more than")
        self.assert_invalid(self.mutated(["scorer", "build_env"], {"": 0.0}), "name")
        long_name = {"x" * 201: 0.0}
        self.assert_invalid(self.mutated(["scorer", "test_config"], long_name), "name")
        too_many = {f"c{i}": 0.0 for i in range(501)}
        self.assert_invalid(self.mutated(["scorer", "test_config"], too_many), "more")

    def test_risk(self) -> None:
        f = {"weight": 1.0, "mean": 0.0, "scale": 0.0}
        self.assert_invalid(self.mutated(["risk", "features", "n_files"], f), "scale")
        f = {"weight": 1.0, "mean": 0.0, "scale": 1.0}
        self.assert_invalid(self.mutated(["risk", "features", "lines"], f), "unknown")
        self.assert_invalid(self.mutated(["risk", "quantiles"], [0.0] * 100), "length")
        q = [0.0] * 100 + [-1.0]
        self.assert_invalid(self.mutated(["risk", "quantiles"], q), "non-decreasing")

    def test_oversize_bytes(self) -> None:
        raw = encode(make_model()) + b" " * scorer_model.MAX_MODEL_BYTES
        with self.assertRaisesRegex(ModelError, "larger"):
            scorer_model.parse_model(raw)

    def test_not_json(self) -> None:
        with self.assertRaisesRegex(ModelError, "not JSON"):
            scorer_model.parse_model(b"{not json")
        with self.assertRaisesRegex(ModelError, "not JSON"):
            scorer_model.parse_model(b"\xff\xfe")


class TestEvaluation(unittest.TestCase):
    def test_interp_clamps_and_interpolates(self) -> None:
        xs, ys = [0.0, 1.0, 3.0], [0.1, 0.3, 0.5]
        self.assertEqual(scorer_model.interp(xs, ys, -1.0), 0.1)
        self.assertEqual(scorer_model.interp(xs, ys, 0.0), 0.1)
        self.assertAlmostEqual(scorer_model.interp(xs, ys, 0.5), 0.2)
        self.assertAlmostEqual(scorer_model.interp(xs, ys, 2.0), 0.4)
        self.assertEqual(scorer_model.interp(xs, ys, 9.0), 0.5)
        self.assertEqual(scorer_model.interp([0.0], [0.2], 5.0), 0.2)

    def test_file_probability_hand_computed(self) -> None:
        # z = -2 + 0.5 ln 20 + 0.3 - 0.1 + ln(0.3 / 0.7) + 2 ln(0.2 / 0.8)
        p = scorer_model.file_probability(
            make_model(), {"HeuristicA": 0.5, "Unknown": 1.0}, 20.0, "env1", "default"
        )
        self.assertAlmostEqual(p, 0.019416578761671233, places=15)

    def test_raised_heuristic_contributes_nothing(self) -> None:
        model = make_model()
        p = scorer_model.file_probability(
            model, {"HeuristicA": 0.5}, 20.0, "env2", "other", raised=["HeuristicB"]
        )
        z = -2 + 0.5 * math.log(20) + math.log(0.3 / 0.7)
        self.assertAlmostEqual(p, 1 / (1 + math.exp(-z)), places=15)

    def test_extreme_z_does_not_overflow(self) -> None:
        model = make_model()
        model["scorer"]["intercept"] = -1e6
        self.assertEqual(scorer_model.file_probability(model, {}, 1.0, "", ""), 0.0)
        model["scorer"]["intercept"] = 1e6
        self.assertEqual(scorer_model.file_probability(model, {}, 1.0, "", ""), 1.0)

    def test_change_features(self) -> None:
        paths = [
            "test/test_nn.py",
            "torch/nn/modules/linear.py",
            "docs/source/nn.rst",
            "README.md",
            "third_party/foo/bar.c",
            "requirements.txt",
        ]
        self.assertEqual(
            scorer_model.change_features(paths),
            {
                "n_files": 6,
                "n_test_files": 1,
                "n_source_files": 1,
                "n_areas": 6,
                "docs_only": 0,
            },
        )
        docs = scorer_model.change_features(["docs/a.py", "x/README.md"])
        self.assertEqual(docs["docs_only"], 1)
        self.assertEqual(docs["n_areas"], 2)

    def test_risk_score(self) -> None:
        model = make_model()
        # raw = 0.5 + (ln 2 - ln 2) = 0.5, at percentile 25 of 0..2 -> r = -0.5
        self.assertAlmostEqual(scorer_model.risk_score(model, {"n_files": 1}), -0.5)
        model["risk"]["intercept"] = -5.0
        self.assertEqual(scorer_model.risk_score(model, {"n_files": 1}), -1.0)
        model["risk"]["intercept"] = 2.0
        self.assertEqual(scorer_model.risk_score(model, {"n_files": 1}), 1.0)

    def test_risk_score_flat_quantiles(self) -> None:
        model = make_model()
        model["risk"]["quantiles"] = [0.0] * 50 + [1.0] * 51
        model["risk"]["intercept"] = 0.0
        r = scorer_model.risk_score(model, {"n_files": 1})
        self.assertAlmostEqual(r, 2 * 0.49 - 1)


class TestScorerBlock(unittest.TestCase):
    def build(
        self,
        fetch: Any,
        changed: Any = ("test/test_a.py",),
        aggregated: AggregatedHeuristics | None = None,
    ) -> dict[str, Any]:
        with mock.patch("builtins.print") as printed:
            block = scorer.build_scorer_block(
                aggregated or make_aggregated(),
                [HEURISTIC_A, HEURISTIC_B],
                fetch,
                None if changed is None else list(changed),
            )
        self.printed = [" ".join(map(str, c.args)) for c in printed.call_args_list]
        if block is None:
            self.fail("block omitted")
        return block

    def test_ok_block(self) -> None:
        block = self.build(lambda: encode(make_model()))
        self.assertEqual(block["reason"], "ok")
        self.assertEqual(block["model"], make_model())
        self.assertEqual(
            block["heuristics"], {"HeuristicA": {"test_a": 0.5, "test_b": 0.25}}
        )
        self.assertEqual(block["raised"], ["HeuristicB"])
        self.assertEqual(block["change"]["n_files"], 1)
        self.assertAlmostEqual(block["risk"], -0.5)
        self.assertEqual(
            self.printed,
            [f"TD scorer: reason='ok' version='2026-11-11' risk={block['risk']!r}"],
        )

    def test_printed_line_escapes_model_strings(self) -> None:
        model = make_model(version="v\nTD scorer: reason='ok'")
        h = {"weight": -1.0, "x": [0.0], "y": [0.5]}
        bad = make_model()
        bad["scorer"]["heuristics"]["evil\nResults from X"] = h
        for fetch in (lambda: encode(model), lambda: encode(bad)):
            self.build(fetch)
            self.assertEqual(len(self.printed), 1)
            self.assertNotIn("\n", self.printed[0])
            self.assertTrue(self.printed[0].startswith("TD scorer: reason="))
        self.assertIn("evil", self.printed[0])

    def test_missing_invalid_disabled(self) -> None:
        cases = [
            (lambda: None, "missing"),
            (lambda: b"{}", "invalid: "),
            (lambda: encode(make_model(enabled=False)), "disabled"),
        ]
        for fetch, reason in cases:
            with self.subTest(reason=reason):
                block = self.build(fetch)
                self.assertTrue(block["reason"].startswith(reason))
                self.assertIsNone(block["model"])
                self.assertIsNone(block["risk"])
                self.assertIn("HeuristicA", block["heuristics"])

    def test_errors_are_caught(self) -> None:
        def boom() -> bytes:
            raise RuntimeError("network down")

        block = self.build(boom)
        self.assertEqual(set(block), {"model", "reason"})
        self.assertTrue(block["reason"].startswith("error: RuntimeError"))

    def test_risk_failure_keeps_the_model(self) -> None:
        for changed in (None, [1]):
            with self.subTest(changed=changed):
                block = self.build(lambda: encode(make_model()), changed)
                self.assertEqual(block["reason"], "ok")
                self.assertEqual(block["model"], make_model())
                self.assertIsNone(block["risk"])
                self.assertIsNone(block["change"])
        model = make_model()
        model["risk"]["features"]["n_files"]["weight"] = 1e308
        model["risk"]["features"]["n_test_files"] = {
            "weight": -1e308,
            "mean": -10.0,
            "scale": 1e-300,
        }
        block = self.build(lambda: encode(model))
        self.assertEqual((block["reason"], block["risk"]), ("ok", None))
        self.assertEqual(block["change"]["n_files"], 1)

    def test_non_finite_heuristic_score_is_dropped(self) -> None:
        aggregated = make_aggregated()
        aggregated.add_heuristic_results(
            HEURISTIC_B,
            TestPrioritizations(
                TESTS, {TestRun("test_c"): math.nan, TestRun("test_b"): 0.5}
            ),
        )
        block = self.build(lambda: None, aggregated=aggregated)
        self.assertEqual(block["reason"], "missing")
        self.assertEqual(block["heuristics"]["HeuristicB"], {"test_b": 0.5})


class FakeResponse:
    def __init__(self, data: bytes) -> None:
        self.data = data

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def read(self, n: int) -> bytes:
        return self.data[:n]


class TestGetTdScorer(unittest.TestCase):
    def fetch(self, outcomes: list[Any], max_bytes: int = 10) -> bytes | None:
        calls = iter(outcomes)

        def fake_urlopen(url: str, timeout: int) -> FakeResponse:
            self.urls.append(url)
            outcome = next(calls)
            if isinstance(outcome, Exception):
                raise outcome
            return FakeResponse(outcome)

        self.urls: list[str] = []
        with (
            mock.patch.object(import_test_stats, "urlopen", fake_urlopen),
            mock.patch("builtins.print"),
        ):
            return import_test_stats.get_td_scorer(max_bytes)

    def test_404_gives_none_after_three_tries(self) -> None:
        not_found = HTTPError("u", 404, "Not Found", Message(), None)
        self.assertIsNone(self.fetch([not_found] * 3))
        self.assertEqual(len(self.urls), 3)
        self.assertTrue(self.urls[0].endswith("/generated-stats/stats/td-scorer.json"))

    def test_transient_error_is_retried(self) -> None:
        self.assertEqual(self.fetch([URLError("reset"), b"{}"]), b"{}")
        self.assertEqual(len(self.urls), 2)

    def test_reads_at_most_max_bytes_plus_one(self) -> None:
        self.assertEqual(self.fetch([b"x" * 50]), b"x" * 11)


class TestTDJob(unittest.TestCase):
    def run_main(self, fetch: Any, changed: Any = None) -> dict[str, Any]:
        self.changed_calls = 0

        def query_changed_files() -> list[str]:
            self.changed_calls += 1
            if changed is not None:
                return changed()  # type: ignore[no-any-return]
            return ["test/test_a.py"]

        with tempfile.TemporaryDirectory() as tmp:
            patches: list[Any] = [
                mock.patch.object(td_job, "REPO_ROOT", Path(tmp)),
                mock.patch.object(td_job, "TESTS", TESTS),
                mock.patch.object(
                    td_job, "get_test_prioritizations", lambda _: make_aggregated()
                ),
                mock.patch.object(td_job, "HEURISTICS", [HEURISTIC_A, HEURISTIC_B]),
                mock.patch.object(td_job, "get_td_scorer", fetch),
                mock.patch.object(td_job, "query_changed_files", query_changed_files),
                mock.patch.dict("os.environ", {"CI": "false"}),
                mock.patch("builtins.print"),
            ]
            for name in (
                "get_test_times",
                "get_test_class_times",
                "get_test_file_ratings",
                "get_test_class_ratings",
                "get_td_heuristic_historial_edited_files_json",
                "get_td_heuristic_profiling_json",
                "copy_pytest_cache",
                "copy_additional_previous_failures",
            ):
                patches.append(mock.patch.object(td_job, name, lambda: None))
            for p in patches:
                p.start()
            try:
                td_job.main()
            finally:
                for p in reversed(patches):
                    p.stop()
            td_results: dict[str, Any] = json.loads(
                (Path(tmp) / "td_results.json").read_text()
            )
            return td_results

    def assert_td_scores(self, td_results: dict[str, Any]) -> None:
        expected = json.loads(
            json.dumps(make_aggregated().get_aggregated_priorities().to_json())
        )
        self.assertEqual(td_results["_test_scores"], expected["_test_scores"])
        self.assertEqual(
            sorted(td_results["_original_tests"]), sorted(expected["_original_tests"])
        )

    def test_model_is_embedded(self) -> None:
        td_results = self.run_main(lambda _: encode(make_model()))
        self.assert_td_scores(td_results)
        self.assertEqual(td_results["scorer"]["reason"], "ok")
        self.assertEqual(td_results["scorer"]["change"]["n_files"], 1)
        self.assertEqual(self.changed_calls, 1)

    def test_changed_files_failure_keeps_the_model(self) -> None:
        def no_git() -> list[str]:
            raise RuntimeError("Unable to get changed files")

        td_results = self.run_main(lambda _: encode(make_model()), no_git)
        self.assert_td_scores(td_results)
        self.assertEqual(td_results["scorer"]["reason"], "ok")
        self.assertIsNone(td_results["scorer"]["risk"])
        self.assertEqual(self.changed_calls, 1)

    def test_bad_or_raising_model_keeps_td_scores(self) -> None:
        def boom(_: int) -> bytes:
            raise RuntimeError("boom")

        bad = copy.deepcopy(make_model())
        bad["scorer"]["heuristics"]["HeuristicA"]["weight"] = -1.0
        for fetch, reason in (
            (lambda _: encode(bad), "invalid: "),
            (boom, "error: RuntimeError"),
        ):
            with self.subTest(reason=reason):
                td_results = self.run_main(fetch)
                self.assert_td_scores(td_results)
                self.assertTrue(td_results["scorer"]["reason"].startswith(reason))
                self.assertIsNone(td_results["scorer"]["model"])

    def test_scorer_omitted_when_even_the_error_block_fails(self) -> None:
        class Unprintable(Exception):
            def __str__(self) -> str:
                raise RuntimeError("no str")

        def boom(_: int) -> bytes:
            raise Unprintable

        td_results = self.run_main(boom)
        self.assert_td_scores(td_results)
        self.assertNotIn("scorer", td_results)


if __name__ == "__main__":
    unittest.main()
