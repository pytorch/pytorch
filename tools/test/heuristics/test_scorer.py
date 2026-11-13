from __future__ import annotations

import copy
import json
import math
import random
import statistics
import sys
import tempfile
import unittest
from datetime import date, datetime, timedelta, UTC
from email.message import Message
from pathlib import Path
from typing import Any, TYPE_CHECKING
from unittest import mock
from urllib.error import HTTPError, URLError


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT))

import tools.testing.do_target_determination_for_s3 as td_job
from tools.stats import import_test_stats
from tools.testing.discover_tests import TESTS as ALL_TESTS
from tools.testing.target_determination import (
    gen_artifact,
    scorer,
    scorer_model,
    scorer_paths,
)
from tools.testing.target_determination.heuristics.interface import (
    AggregatedHeuristics,
    HeuristicInterface,
    TestPrioritizations,
)
from tools.testing.target_determination.scorer_model import ModelError
from tools.testing.test_run import TestRun


sys.path.remove(str(REPO_ROOT))

if TYPE_CHECKING:
    from typing import Self


START_UP = 10.0
TODAY = date(2026, 11, 11)
TD_JOB_NOW = datetime(2026, 11, 12, 1, tzinfo=UTC)


def frozen_utc_now(now: datetime) -> mock.Mock:
    frozen = mock.Mock(wraps=datetime)
    frozen.now.return_value = now
    return frozen


def make_model(**overrides: Any) -> dict[str, Any]:
    model: dict[str, Any] = {
        "schema": 2,
        "version": "2026-11-11",
        "enabled": True,
        "start_up_seconds": {"env1": {"default": START_UP, "slow": 2.5}},
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


def make_budget(share: float = 0.25, spread: float = 0.0, a: float = 1.0) -> Any:
    return {"share": share, "spread": spread, "cost_exponent": a}


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
        self.assert_invalid(self.mutated(["schema"], 1), "schema 1 is not 2")
        self.assert_invalid(self.mutated(["schema"], 3), "schema")
        self.assert_invalid(self.mutated(["version"], 1), "version")
        self.assert_invalid(self.mutated(["enabled"], 1), "enabled")

    def test_schema_1_model_is_rejected(self) -> None:
        old = make_model(schema=1, overhead_seconds=24.0)
        del old["start_up_seconds"]
        self.assert_invalid(old, "keys")
        self.assert_invalid(make_model(overhead_seconds=24.0), "keys")

    def test_start_up_table(self) -> None:
        tables: list[dict[str, dict[str, float]]] = [
            {},
            {"env1": {}},
            {"e": {"a": 1.0, "b": 600.0, "c": 7}},
        ]
        for table in tables:
            with self.subTest(table=table):
                model = make_model(start_up_seconds=table)
                self.assertEqual(scorer_model.parse_model(encode(model)), model)

        def at(k: Any) -> dict[str, Any]:
            return self.mutated(["start_up_seconds", "env1", "default"], k)

        self.assert_invalid(at(0.99), "outside")
        self.assert_invalid(at(0), "outside")
        self.assert_invalid(at(-5.0), "outside")
        self.assert_invalid(at(600.01), "outside")
        self.assert_invalid(at(True), "not a number")
        self.assert_invalid(at("10"), "not a number")
        self.assert_invalid(at(None), "not a number")
        with self.assertRaisesRegex(ModelError, "not finite"):
            scorer_model.validate_model(at(math.nan))
        raw = encode(make_model()).decode().replace('"default": 10.0', '"default": NaN')
        with self.assertRaisesRegex(ModelError, "non-finite"):
            scorer_model.parse_model(raw.encode())
        self.assert_invalid(make_model(start_up_seconds=[]), "not an object")
        self.assert_invalid(make_model(start_up_seconds={"env1": 10.0}), "object")

    def test_start_up_name_caps(self) -> None:
        long_name = "x" * 201
        cases = [
            ({"": {"default": 10.0}}, "name"),
            ({long_name: {"default": 10.0}}, "name"),
            ({"env1": {long_name: 10.0}}, "name"),
            ({f"e{i}": {} for i in range(2001)}, "more than 2000"),
            ({"env1": {f"c{i}": 10.0 for i in range(501)}}, "more than 500"),
        ]
        for table, message in cases:
            with self.subTest(message=message):
                self.assert_invalid(make_model(start_up_seconds=table), message)
        at_caps: dict[str, dict[str, float]] = {f"e{i}": {} for i in range(2000)}
        at_caps["env1"] = {f"c{i}": 10.0 for i in range(500)}
        del at_caps["e0"]
        scorer_model.validate_model(make_model(start_up_seconds=at_caps))

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

    def test_start_up_seconds_lookup(self) -> None:
        model = make_model()
        model["start_up_seconds"]["env2"] = {"default": 7}
        lookup = scorer_model.start_up_seconds
        self.assertEqual(lookup(model, "env1", "default"), START_UP)
        self.assertEqual(lookup(model, "env1", "slow"), 2.5)
        k = lookup(model, "env2", "default")
        self.assertEqual((k, type(k)), (7.0, float))
        self.assertIsNone(lookup(model, "env1", "other"))
        self.assertIsNone(lookup(model, "env3", "default"))
        self.assertIsNone(lookup(make_model(start_up_seconds={}), "env1", "default"))

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
                "n_source_files": 2,
                "n_areas": 6,
                "docs_only": 0,
            },
        )
        docs = scorer_model.change_features(["docs/a.rst", "x/README.md"])
        self.assertEqual(docs["docs_only"], 1)
        self.assertEqual(docs["n_areas"], 2)
        self.assertEqual(docs["n_source_files"], 0)

    def test_docs_are_md_and_rst_only_in_any_case(self) -> None:
        def features(paths: list[str]) -> tuple[int, int, int]:
            f = scorer_model.change_features(paths)
            return f["n_files"], f["docs_only"], f["n_source_files"]

        cases = [
            (["CMakeLists.txt"], (1, 0, 1)),
            ([".github/ci_commit_pins/triton.txt"], (1, 0, 1)),
            (["requirements.txt"], (1, 0, 1)),
            (["README.MD"], (1, 1, 0)),
            (["torch/Notes.Rst"], (1, 1, 0)),
            (["docs/a.py"], (1, 0, 0)),
            (["README.md", ""], (1, 1, 0)),
            ([""], (0, 0, 0)),
            ([], (0, 0, 0)),
        ]
        for paths, expected in cases:
            with self.subTest(paths=paths):
                self.assertEqual(features(paths), expected)
        empty = scorer_model.change_features([""])
        self.assertEqual(set(empty.values()), {0})

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
        today: date | None = TODAY,
    ) -> dict[str, Any]:
        with mock.patch("builtins.print") as printed:
            block = scorer.build_scorer_block(
                aggregated or make_aggregated(),
                iter([HEURISTIC_A, HEURISTIC_B]),
                fetch,
                None if changed is None else list(changed),
                today,
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
        self.assertEqual(block["unknown"], [])
        self.assertEqual(block["change"]["n_files"], 1)
        self.assertAlmostEqual(block["risk"], -0.5)
        self.assertEqual(
            self.printed,
            [
                "TD scorer: reason='ok' version='2026-11-11' "
                f"risk={block['risk']!r} unknown=[]"
            ],
        )

    def test_printed_line_escapes_model_strings(self) -> None:
        model = make_model(version="v\nTD scorer: reason='ok'")
        h = {"weight": -1.0, "x": [0.0], "y": [0.5]}
        bad = make_model()
        bad["scorer"]["heuristics"]["evil\nResults from X"] = h
        renamed = make_model()
        renamed["scorer"]["heuristics"]["evil\nTD scorer: reason='ok'"] = {
            "weight": 1.0,
            "x": [0.0],
            "y": [0.5],
        }
        for fetch, reason, injected in (
            (lambda: encode(model), "stale: v\nTD scorer: reason='ok'", "stale: v\\n"),
            (lambda: encode(bad), "invalid: ", "evil"),
            (lambda: encode(renamed), "ok", "evil"),
        ):
            with self.subTest(reason=reason):
                block = self.build(fetch)
                self.assertTrue(block["reason"].startswith(reason))
                self.assertEqual(len(self.printed), 1)
                self.assertNotIn("\n", self.printed[0])
                self.assertTrue(self.printed[0].startswith("TD scorer: reason="))
                self.assertIn(injected, self.printed[0])
        self.assertIn("unknown=[\"evil\\nTD scorer: reason='ok'\"]", self.printed[0])

    def test_unknown_lists_model_heuristics_not_registered(self) -> None:
        model = make_model()
        model["scorer"]["heuristics"]["Renamed"] = {
            "weight": 1.0,
            "x": [0.0],
            "y": [0.5],
        }
        block = self.build(lambda: encode(model))
        self.assertEqual(block["reason"], "ok")
        self.assertEqual(block["unknown"], ["Renamed"])
        self.assertTrue(self.printed[0].endswith(" unknown=['Renamed']"))
        block = self.build(lambda: None)
        self.assertIsNone(block["unknown"])
        self.assertTrue(self.printed[0].endswith(" unknown=None"))

    def test_stale_model_is_dropped(self) -> None:
        cases = [
            ("2026-11-04", "ok"),
            ("2026-11-03", "stale: 2026-11-03"),
            ("2026-11-12", "ok"),
            ("2026-11-13", "stale: 2026-11-13"),
            ("2026-11-11", "ok"),
            ("2026-02-30", "stale: 2026-02-30"),
            ("refit-7", "stale: refit-7"),
            ("", "stale: "),
        ]
        for version, reason in cases:
            with self.subTest(version=version):
                block = self.build(lambda: encode(make_model(version=version)))
                self.assertEqual(block["reason"], reason)
                if reason == "ok":
                    self.assertEqual(block["model"]["version"], version)
                else:
                    self.assertIsNone(block["model"])
                    self.assertIsNone(block["risk"])
                    self.assertIsNone(block["unknown"])
                    self.assertIn("HeuristicA", block["heuristics"])

    def test_model_age_defaults_to_the_utc_date(self) -> None:
        for now, reason in (
            (datetime(2026, 11, 18, 23, 59, tzinfo=UTC), "ok"),
            (datetime(2026, 11, 19, 0, 1, tzinfo=UTC), "stale: 2026-11-11"),
        ):
            frozen = frozen_utc_now(now)
            with self.subTest(now=now), mock.patch.object(scorer, "datetime", frozen):
                block = self.build(lambda: encode(make_model()), today=None)
                self.assertEqual(block["reason"], reason)
                frozen.now.assert_called_once_with(UTC)

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
    def fetch(
        self, outcomes: list[Any], max_bytes: int = 10, name: str | None = None
    ) -> bytes | None:
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
            if name is None:
                return import_test_stats.get_td_scorer(max_bytes)
            return import_test_stats.get_td_scorer(max_bytes, name)

    def test_budget_file_name(self) -> None:
        self.assertEqual(
            self.fetch([b"{}"], name=import_test_stats.TD_SCORER_BUDGET_FILE), b"{}"
        )
        self.assertTrue(
            self.urls[0].endswith("/generated-stats/stats/td-scorer-budget.json")
        )

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
    def run_main(
        self, fetch: Any, changed: Any = None, budget: Any = None
    ) -> dict[str, Any]:
        """td_job.main with `fetch` serving td-scorer.json and `budget` (None: a
        404) serving td-scorer-budget.json; self.fetched lists the files asked."""
        self.changed_calls = 0
        self.fetched: list[str] = []

        def get_td_scorer(max_bytes: int, name: str = td_job.TD_SCORER_FILE) -> Any:
            self.fetched.append(name)
            if name == td_job.TD_SCORER_BUDGET_FILE:
                return None if budget is None else budget(max_bytes)
            return fetch(max_bytes)

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
                mock.patch.object(td_job, "get_td_scorer", get_td_scorer),
                mock.patch.object(td_job, "query_changed_files", query_changed_files),
                mock.patch.dict("os.environ", {"CI": "false"}),
                mock.patch.object(scorer, "datetime", frozen_utc_now(TD_JOB_NOW)),
            ]
            print_patch = mock.patch("builtins.print")
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
            printed = print_patch.start()
            try:
                td_job.main()
            finally:
                print_patch.stop()
                for p in reversed(patches):
                    p.stop()
            self.printed = [" ".join(map(str, c.args)) for c in printed.call_args_list]
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
        self.assertEqual(td_results["scorer"]["unknown"], [])
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

    def test_budget_file_used_when_both_files_are_usable(self) -> None:
        plain = make_model()
        with_budget = make_model(budget=make_budget())
        td_results = self.run_main(
            lambda _: encode(plain), budget=lambda _: encode(with_budget)
        )
        self.assertEqual(td_results["scorer"]["model"], with_budget)
        self.assertEqual(
            self.fetched, [td_job.TD_SCORER_FILE, td_job.TD_SCORER_BUDGET_FILE]
        )
        self.assertIn("TD scorer: model from td-scorer-budget.json", self.printed)

    def test_unusable_budget_file_falls_back_to_the_plain_file(self) -> None:
        def boom(_: int) -> bytes:
            raise RuntimeError("boom")

        bad = copy.deepcopy(make_model(budget=make_budget()))
        bad["budget"]["share"] = 5.0
        cases: list[tuple[str, Any]] = [
            ("missing", None),
            ("invalid", lambda _: encode(bad)),
            ("disabled", lambda _: encode(make_model(enabled=False))),
            ("stale", lambda _: encode(make_model(version="2026-10-01"))),
            ("raising", boom),
            ("another refit", lambda _: encode(make_model(version="2026-11-10"))),
        ]
        for why, budget in cases:
            with self.subTest(why=why):
                td_results = self.run_main(
                    lambda _: encode(make_model()), budget=budget
                )
                self.assertEqual(td_results["scorer"]["reason"], "ok")
                self.assertEqual(td_results["scorer"]["model"], make_model())
                self.assertIn("TD scorer: model from td-scorer.json", self.printed)

    def test_plain_file_is_the_master_switch(self) -> None:
        def boom(_: int) -> bytes:
            raise RuntimeError("boom")

        bad = copy.deepcopy(make_model())
        bad["schema"] = 1
        with_budget = make_model(budget=make_budget())
        cases: list[tuple[str, Any]] = [
            ("raising", boom),
            ("missing", lambda _: None),
            ("invalid", lambda _: encode(bad)),
            ("disabled", lambda _: encode(make_model(enabled=False))),
            ("stale", lambda _: encode(make_model(version="2026-10-01"))),
        ]
        for why, plain in cases:
            with self.subTest(why=why):
                td_results = self.run_main(plain, budget=lambda _: encode(with_budget))
                self.assertIsNone(td_results["scorer"]["model"])
                self.assertEqual(self.fetched, [td_job.TD_SCORER_FILE])
                self.assertFalse(any("model from" in line for line in self.printed))

    def test_a_block_in_the_plain_file_never_applies(self) -> None:
        with_budget = make_model(budget=make_budget())
        for budget in (None, lambda _: encode(make_model(enabled=False))):
            with self.subTest(budget=budget):
                td_results = self.run_main(lambda _: encode(with_budget), budget=budget)
                self.assertEqual(td_results["scorer"]["model"], make_model())

    def test_choosing_the_file_never_raises(self) -> None:
        real = td_job.build_scorer_block
        plain = make_model()

        def boom(*args: Any) -> Any:
            raise RuntimeError("glue")

        def second_call_boom(*args: Any) -> Any:
            if len(self.calls) == 1:
                raise RuntimeError("glue")
            self.calls.append(1)
            return real(*args)

        for name, patched, has_model in (
            ("first", boom, None),
            ("second", second_call_boom, True),
            ("none", lambda *args: None, None),
        ):
            self.calls: list[int] = []
            with (
                self.subTest(call=name),
                mock.patch.object(td_job, "build_scorer_block", patched),
            ):
                td_results = self.run_main(
                    lambda _: encode(plain),
                    budget=lambda _: encode(make_model(budget=make_budget())),
                )
                self.assert_td_scores(td_results)
                if has_model:
                    self.assertEqual(td_results["scorer"]["model"], plain)
                else:
                    self.assertNotIn("scorer", td_results)

    def test_scorer_omitted_when_even_the_error_block_fails(self) -> None:
        class Unprintable(Exception):
            def __str__(self) -> str:
                raise RuntimeError("no str")

        def boom(_: int) -> bytes:
            raise Unprintable

        td_results = self.run_main(boom)
        self.assert_td_scores(td_results)
        self.assertNotIn("scorer", td_results)


def make_block(model: Any = None, **scores: dict[str, float]) -> dict[str, Any]:
    return {
        "model": make_model() if model is None else model,
        "reason": "ok",
        "heuristics": scores,
        "raised": [],
        "change": {"n_files": 1},
        "risk": 0.0,
    }


def file_cost(seconds: float) -> float:
    return seconds + START_UP


def td_cut(
    tests: list[str], scores: dict[str, float]
) -> tuple[list[TestRun], list[TestRun]]:
    tp = TestPrioritizations(tests, {TestRun(t): s for t, s in scores.items()})
    return tp.get_top_per_tests(25)


def names(tests: list[TestRun]) -> list[str]:
    return [str(t) for t in tests]


class TestSelection(unittest.TestCase):
    def select(
        self, block: Any, include: list[TestRun], exclude: list[TestRun], times: Any
    ) -> tuple[list[TestRun], list[TestRun], dict[str, Any]]:
        return scorer.select_tests(block, include, exclude, times, "env1", "default")

    def test_ranks_by_p_over_cost_within_td_cost(self) -> None:
        tests = [f"t{i:02d}" for i in range(12)]
        times = dict.fromkeys(tests, 50.0)
        times["t05"] = 5000.0
        include, exclude = td_cut(tests, {"t05": 1.0, "t07": 0.5})
        self.assertEqual(names(include), ["t05", "t07", "t00", "t01"])
        block = make_block(HeuristicA={"t05": 1.0, "t07": 0.5, "t09": 0.25})
        new_include, new_exclude, info = self.select(block, include, exclude, times)
        self.assertEqual(info["budget_ms"], 5_190_000)
        self.assertEqual(names(new_include)[:2], ["t07", "t09"])
        self.assertEqual(names(new_exclude), ["t05"])
        self.assertEqual(info["cost_ms"], 660_000)
        self.assertEqual(info["start_up"], START_UP)
        self.assertEqual(info["cut"], "scorer")
        self.assertEqual(info["version"], "2026-11-11")
        self.assertEqual((info["n"], info["seconds"]), (11, 550.0))
        self.assertEqual((info["td_n"], info["td_seconds"]), (4, 5150.0))
        self.assertEqual(sorted(names(new_include) + names(new_exclude)), sorted(tests))

    def test_never_exceeds_budget(self) -> None:
        rng = random.Random(0)
        for trial in range(200):
            tests = [f"t{i}" for i in range(rng.randint(1, 60))]
            times = {t: rng.choice([0.0, 1.0, rng.uniform(0, 3000)]) for t in tests}
            scores = {t: rng.random() for t in tests if rng.random() < 0.3}
            include, exclude = td_cut(tests, scores)
            block = make_block(HeuristicA=scores)
            new_include, new_exclude, info = self.select(block, include, exclude, times)
            with self.subTest(trial=trial):
                self.assertLessEqual(info["cost_ms"], info["budget_ms"])
                cost = sum(
                    round(file_cost(times[t.test_file]) * 1000) for t in new_include
                )
                self.assertEqual(cost, info["cost_ms"])
                self.assertEqual(len(new_include) + len(new_exclude), len(tests))

    def test_every_file_costs_its_seconds_plus_the_start_up(self) -> None:
        tests = [f"t{i}" for i in range(8)]
        times = dict.fromkeys(tests, 0.04)
        times.update(t5=0.99, t6=100.0, t7=1.0)
        include, exclude = td_cut(tests, {"t6": 1.0, "t7": 1.0})
        self.assertEqual(names(include), ["t6", "t7", "t0"])
        block = make_block(HeuristicA={"t0": 1.0, "t1": 1.0, "t2": 1.0})
        new_include, _, info = self.select(block, include, exclude, times)
        self.assertEqual(info["budget_ms"], 110_000 + 11_000 + 10_040)
        self.assertEqual(names(new_include), ["t0", "t1", "t2", "t3", "t4", "t5", "t7"])
        self.assertEqual(info["cost_ms"], 5 * 10_040 + 10_990 + 11_000)

    def test_start_up_comes_from_the_job_config(self) -> None:
        times = {"a": 1.0, "b": 0.0049, "c": 4.0}
        block = make_block()
        _, _, info = scorer.select_tests(
            block, [TestRun("a"), TestRun("b")], [TestRun("c")], times, "env1", "slow"
        )
        self.assertEqual(info["start_up"], 2.5)
        self.assertEqual(info["budget_ms"], 3_500 + 2_505)
        self.assertEqual(info["cost_ms"], 3_500 + 2_505)
        self.assertEqual(info["n"], 2)

    def test_zero_second_file_costs_the_start_up(self) -> None:
        costs = []

        def spy(*args: Any, **kwargs: Any) -> float:
            costs.append(args[2])
            return scorer_model.file_probability(*args, **kwargs)

        with mock.patch.object(scorer, "file_probability", spy):
            include, exclude, info = self.select(
                make_block(), [TestRun("a")], [TestRun("b")], {"a": 0.0, "b": 0.0}
            )
        self.assertEqual(costs, [START_UP, START_UP])
        self.assertEqual((info["budget_ms"], info["cost_ms"]), (10_000, 10_000))
        self.assertEqual((len(include), len(exclude)), (1, 1))

    def test_unlisted_job_raises_no_model(self) -> None:
        for env, config in (("env1", "other"), ("env9", "default"), ("", "")):
            with self.subTest(env=env, config=config):
                with self.assertRaisesRegex(
                    scorer.NoModel, f"^no start-up for {env}/{config}$"
                ):
                    scorer.select_tests(
                        make_block(), [TestRun("a")], [], {"a": 1.0}, env, config
                    )

    def test_deterministic_ties_by_name(self) -> None:
        tests = ["b", "d", "a", "c", "e", "f"]
        times = dict.fromkeys(tests, 5.0)
        include, exclude = td_cut(tests, {"f": 1.0})
        block = make_block()
        first = self.select(block, include, exclude, times)
        self.assertEqual(names(first[0]), ["a", "b"])
        self.assertEqual(names(first[1]), ["c", "d", "e", "f"])
        shuffled = dict(reversed(list(times.items())))
        again = self.select(copy.deepcopy(block), include, exclude, shuffled)
        self.assertEqual(first, again)

    def test_trial_mode_heuristics_score_nothing(self) -> None:
        self.assertEqual(scorer.TRIAL_HEURISTICS, {"HistoricalClassFailurCorrelation"})
        tests = ["b", "d", "a", "c", "e", "f"]
        times = dict.fromkeys(tests, 5.0)
        include, exclude = td_cut(tests, {"f": 1.0})
        block = make_block(HeuristicA={"f": 1.0})
        self.assertEqual(
            names(self.select(block, include, exclude, times)[0]), ["f", "a"]
        )
        raised = []

        def spy(*args: Any) -> float:
            raised.append(args[5])
            return scorer_model.file_probability(*args)

        trial = frozenset({"HeuristicA"})
        with (
            mock.patch.object(scorer, "TRIAL_HEURISTICS", trial),
            mock.patch.object(scorer, "file_probability", spy),
        ):
            new_include, _, _ = self.select(block, include, exclude, times)
        self.assertEqual(names(new_include), ["a", "b"])
        self.assertEqual(raised, [{"HeuristicA"}] * len(tests))
        self.assertEqual(block["raised"], [])

    def test_missing_times_use_the_median(self) -> None:
        times = {"a": 10.0, "b": 20.0, "c": 90.0}
        block = make_block()
        _, _, info = self.select(block, [TestRun("d")], [TestRun("a")], times)
        self.assertEqual(info["budget_ms"], 30_000)
        self.assertEqual(info["td_seconds"], 20.0)
        _, _, info = self.select(block, [TestRun("d")], [], {})
        self.assertEqual(info["budget_ms"], 10_000)

    def test_build_env_one_hot_is_keyed_by_the_job_base_name(self) -> None:
        job = "dynamo-unittest / dynamo-test (3.12)"
        model = make_model()
        model["scorer"]["build_env"] = {job: 5.0, "linux-jammy-py3.12-clang21": -5.0}
        model["start_up_seconds"][job] = {"default": START_UP}
        z = -2 + 0.5 * math.log(410) + 5.0 - 0.1
        z += math.log(0.1 / 0.9) + 2 * math.log(0.2 / 0.8)
        p = scorer_model.file_probability(model, {}, 410.0, job, "default")
        self.assertAlmostEqual(p, 1 / (1 + math.exp(-z)), places=15)

        keys = []

        def spy(*args: Any) -> float:
            keys.append(args[3])
            return scorer_model.file_probability(*args)

        with mock.patch.object(scorer, "file_probability", spy):
            self.select(make_block(model=model), [TestRun("a")], [], {"a": 1.0})
            scorer.select_tests(
                make_block(model=model), [TestRun("a")], [], {}, job, "default"
            )
        self.assertEqual(keys, ["env1", job])

    def test_td_include_set_refits_exactly(self) -> None:
        model = make_model()
        model["scorer"]["heuristics"]["HeuristicA"] = {
            "weight": 10.0,
            "x": [0.0, 0.01, 1.0],
            "y": [1e-6, 0.5, 1 - 1e-6],
        }
        rng = random.Random(3)
        for trial in range(200):
            tests = [f"t{i:02d}" for i in range(rng.randint(4, 80))]
            k = 25 * len(tests) // 100 + 1
            scores = {t: rng.uniform(0.5, 1.0) for t in rng.sample(tests, k)}
            times = {t: rng.uniform(0, 500) for t in tests}
            include, exclude = td_cut(tests, scores)
            block = make_block(model=model, HeuristicA=scores)
            new_include, _, info = self.select(block, include, exclude, times)
            with self.subTest(trial=trial):
                self.assertEqual(sorted(names(new_include)), sorted(names(include)))
                self.assertEqual(info["cost_ms"], info["budget_ms"])

    def test_must_run_files_come_first(self) -> None:
        model = make_model()
        for h in model["scorer"]["heuristics"].values():
            h["weight"] = 0.0
        tests = [f"t{i}" for i in range(8)]
        times = dict.fromkeys(tests, 10.0)
        times["t6"] = 1000.0
        include, exclude = td_cut(tests, {})
        block = make_block(
            model=model,
            EditedByPR={"t6": 1.0},
            PreviouslyFailedInPR={"t5": 0.5, "t3": 0.0},
        )
        new_include, _, info = self.select(block, include, exclude, times)
        self.assertEqual(names(new_include), ["t5", "t6"])
        self.assertGreater(info["cost_ms"], info["budget_ms"])

        times["t6"] = 10.0
        new_include, _, _ = self.select(block, include, exclude, times)
        self.assertEqual(names(new_include), ["t5", "t6", "t0"])

        block["raised"] = ["EditedByPR"]
        new_include, _, _ = self.select(block, include, exclude, times)
        self.assertEqual(names(new_include), ["t5", "t0", "t1"])

    def test_class_level_test_runs_are_whole_files(self) -> None:
        include = [TestRun("a::TestX"), TestRun("b")]
        exclude = [TestRun("a", excluded=["TestX"]), TestRun("c")]
        times = {"a": 30.0, "b": 30.0, "c": 30.0}
        block = make_block(HeuristicA={"c": 1.0})
        new_include, new_exclude, info = self.select(block, include, exclude, times)
        self.assertEqual(info["budget_ms"], 80_000)
        self.assertEqual(names(new_include), ["c", "a"])
        self.assertEqual(names(new_exclude), ["b"])
        self.assertTrue(all(t.is_full_file() for t in new_include + new_exclude))

    def test_realistic_fixture(self) -> None:
        rng = random.Random(1)
        tests = sorted(ALL_TESTS)
        times = {t: rng.lognormvariate(4, 1.5) for t in tests if rng.random() < 0.95}
        scores = {t: rng.random() for t in rng.sample(tests, 80)}
        include, exclude = td_cut(tests, scores)
        block = make_block(HeuristicA=scores)
        new_include, _, info = self.select(block, include, exclude, times)
        median = statistics.median(times.values())
        td_cost = sum(file_cost(times.get(t.test_file, median)) for t in include)
        self.assertAlmostEqual(info["budget_ms"] / 1000, td_cost, delta=1.0)
        self.assertLessEqual(info["cost_ms"], info["budget_ms"])
        self.assertGreater(len(new_include), 0)


class TestScorerCut(unittest.TestCase):
    def cut(
        self, block: Any, build_env_key: str = "env1"
    ) -> tuple[Any, Any, dict[str, Any]]:
        self.include, self.exclude = td_cut(
            ["a", "b", "c", "d", "e"], {"e": 1.0, "d": 0.5}
        )
        times: dict[str, float] = dict.fromkeys("abcde", 5.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "td_results.json"
            if block is not None:
                path.write_text(json.dumps({"_test_scores": [], "scorer": block}))
            return scorer.scorer_cut(
                path, self.include, self.exclude, times, build_env_key, "default"
            )

    def assert_td_cut(self, block: Any, reason: str, env: str = "env1") -> None:
        include, exclude, info = self.cut(block, env)
        self.assertEqual(include, self.include)
        self.assertEqual(exclude, self.exclude)
        self.assertEqual(info["cut"], "td")
        self.assertTrue(info["reason"].startswith(reason), info["reason"])
        self.assertEqual(info["n"], len(self.include))
        self.assertEqual(len(info["hash"]), 12)
        self.assertIsNone(info["start_up"])
        self.assertIsNone(info["budget_ms"])

    def test_fallbacks_give_td_cut_exactly(self) -> None:
        bad = make_model()
        bad["scorer"]["heuristics"]["HeuristicA"]["weight"] = -1.0
        no_heuristics = make_block()
        del no_heuristics["heuristics"]
        cases: list[tuple[Any, str]] = [
            (None, "no scorer block"),
            ({"model": None, "reason": "error: boom"}, "no model: error: boom"),
            (make_block(model=bad), "invalid: "),
            (make_block(model=make_model(enabled=False)), "disabled"),
            (no_heuristics, "error: KeyError"),
            ([], "no scorer block"),
        ]
        for block, reason in cases:
            with self.subTest(reason=reason):
                self.assert_td_cut(block, reason)
        with mock.patch.object(scorer, "load_scorer_block", lambda _: make_block()):
            include, _, info = scorer.scorer_cut(
                Path("unused"), [TestRun("a")], [], {"a": -20.0}, "env1", "default"
            )
        self.assertEqual((names(include), info["cut"]), (["a"], "td"))
        self.assertTrue(info["reason"].startswith("error: ValueError"))

    def test_fallback_without_usable_times_keeps_td_cut(self) -> None:
        include, exclude, info = scorer.scorer_cut(
            Path("missing/td_results.json"),
            [TestRun("a")],
            [TestRun("b")],
            {"a": None, "b": 1.0},  # type: ignore[dict-item]
            "",
            "",
        )
        self.assertEqual((names(include), names(exclude)), (["a"], ["b"]))
        self.assertEqual(info["cut"], "td")
        self.assertIsNone(info["td_seconds"])

    def test_unlisted_job_gives_td_cut(self) -> None:
        self.assert_td_cut(make_block(), "no start-up for env9/default", env="env9")

    def test_malformed_budget_gives_td_cut(self) -> None:
        cases: list[Any] = [
            make_block(model=make_model(budget={"share": "x"})),
            make_block(model=make_model(budget=make_budget(share=2.0))),
            make_block(model=make_model(budget=None)),
        ]
        for block in cases:
            with self.subTest(budget=block["model"]["budget"]):
                self.assert_td_cut(block, "invalid: ")
                _, _, info = self.cut(block)
                self.assertEqual(
                    {k: info[k] for k in ("share", "risk", "cost_exponent")},
                    {"share": None, "risk": None, "cost_exponent": None},
                )
                self.assertEqual(info["budget_mode"], "error")

    def test_odd_budget_fields_keep_the_scorer(self) -> None:
        block = make_block(model=make_model(budget=make_budget()))
        block.update(build_files=[1], risk={"r": 1}, budget_reason=["stale"])
        include, _, info = self.cut(block)
        self.assertEqual(info["cut"], "scorer")
        self.assertEqual(info["budget_mode"], "build_paths")
        block.update(build_files=0, risk="0.5")
        _, _, info = self.cut(block)
        self.assertEqual((info["cut"], info["budget_mode"]), ("scorer", "unknown_risk"))

    def test_scorer_cut_with_model(self) -> None:
        include, exclude, info = self.cut(make_block(HeuristicA={"a": 1.0}))
        self.assertEqual(info["cut"], "scorer")
        self.assertEqual(names(include)[0], "a")
        self.assertEqual(info["budget_ms"], 30_000)
        self.assertEqual(info["start_up"], START_UP)

    def test_artifact_carries_scorer_info(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "test" / "test-reports").mkdir(parents=True)
            with mock.patch.object(gen_artifact, "REPO_ROOT", Path(tmp)):
                gen_artifact.gen_ci_artifact([{"test_file": "a"}], [], {"cut": "td"})
                gen_artifact.gen_ci_artifact([{"test_file": "a"}], [])
            written = [
                json.loads(p.read_text())
                for p in (Path(tmp) / "test" / "test-reports").iterdir()
            ]
        self.assertCountEqual(
            written,
            [
                {
                    "included": [{"test_file": "a"}],
                    "excluded": [],
                    "scorer": {"cut": "td"},
                },
                {"included": [{"test_file": "a"}], "excluded": []},
            ],
        )


class TestBudgetValidation(unittest.TestCase):
    def assert_invalid(self, model: Any, message: str) -> None:
        with self.assertRaisesRegex(ModelError, message):
            scorer_model.parse_model(encode(model))

    def assert_valid(self, model: Any) -> None:
        self.assertEqual(scorer_model.parse_model(encode(model)), model)

    def test_contract_constants(self) -> None:
        self.assertEqual(
            (
                scorer_model.MIN_SHARE,
                scorer_model.MAX_SHARE,
                scorer.BUDGET_STALE_DAYS,
                scorer_paths.MAX_CHANGED_FILES,
            ),
            (0.10, 1.0, 3, 3000),
        )

    def test_budget_is_optional(self) -> None:
        self.assert_valid(make_model())
        self.assert_valid(make_model(budget=make_budget()))
        self.assert_invalid(make_model(budget=make_budget(), extra=1), "keys")

    def test_budget_keys_and_types(self) -> None:
        cases: list[tuple[Any, str]] = [
            (None, "not an object"),
            ([0.25, 0.0, 1.0], "not an object"),
            ({**make_budget(), "floor": 0.1}, "keys"),
            ({"share": 0.25, "spread": 0.0}, "keys"),
            ({**make_budget(), "share": True}, "number"),
            ({**make_budget(), "spread": False}, "number"),
            ({**make_budget(), "cost_exponent": True}, "number"),
            ({**make_budget(), "share": "0.25"}, "number"),
            ({**make_budget(), "share": math.nan}, "not finite"),
            ({**make_budget(), "cost_exponent": math.inf}, "not finite"),
        ]
        for budget, message in cases:
            with self.subTest(budget=budget):
                with self.assertRaisesRegex(ModelError, message):
                    scorer_model.validate_model(make_model(budget=budget))
        raw = encode(make_model(budget=make_budget())).decode()
        bad = raw.replace('"share": 0.25', '"share": NaN')
        with self.assertRaisesRegex(ModelError, "non-finite"):
            scorer_model.parse_model(bad.encode())

    def test_budget_bounds(self) -> None:
        valid = [
            make_budget(share=0.1),
            make_budget(share=1.0),
            make_budget(share=0.2, spread=0.5),
            make_budget(spread=0.0, a=0.0),
            make_budget(a=1.0),
            make_budget(share=1.0, spread=0.8),
            {"share": 1, "spread": 0, "cost_exponent": 1},
        ]
        for budget in valid:
            with self.subTest(budget=budget):
                self.assert_valid(make_model(budget=budget))
        invalid = [
            (make_budget(share=0.0999), "below"),
            (make_budget(share=0.15, spread=0.5), "below"),
            (make_budget(share=1.0, spread=0.9), "below"),
            (make_budget(share=-1.0), "below"),
            (make_budget(share=1.01), "above"),
            (make_budget(spread=1.0), "spread"),
            (make_budget(spread=-0.01), "spread"),
            (make_budget(a=-0.01), "cost_exponent"),
            (make_budget(a=1.01), "cost_exponent"),
        ]
        for budget, message in invalid:
            with self.subTest(budget=budget):
                self.assert_invalid(make_model(budget=budget), message)

    def test_budget_keeps_the_start_up_bounds(self) -> None:
        for k in (1.0, 600.0):
            with self.subTest(k=k):
                start_up = {"env1": {"default": k}}
                self.assert_valid(
                    make_model(start_up_seconds=start_up, budget=make_budget())
                )
        for k in (0.99, 600.01):
            with self.subTest(k=k):
                start_up = {"env1": {"default": k}}
                self.assert_invalid(
                    make_model(start_up_seconds=start_up, budget=make_budget()),
                    "outside",
                )


class TestBudgetShare(unittest.TestCase):
    def share(self, build_files: Any = 0, risk: Any = 0.0, **budget: float) -> Any:
        model = make_model(budget=make_budget(**budget))
        return scorer_model.budget_share(model, build_files, risk)

    def test_no_budget_block(self) -> None:
        self.assertEqual(
            scorer_model.budget_share(make_model(), 0, 0.5), (None, "none")
        )

    def test_share_at_each_risk(self) -> None:
        for r in (-1, 0, 1, -1.0, 0.0, 1.0, 0.3):
            with self.subTest(r=r):
                expected = 0.3 * (1 + 0.5 * r)
                self.assertEqual(
                    self.share(risk=r, share=0.3, spread=0.5), (expected, "share")
                )
        self.assertEqual(self.share(risk=-1, share=0.3, spread=0.5)[0], 0.15)
        self.assertEqual(self.share(risk=0, share=0.3, spread=0.5)[0], 0.3)
        self.assertAlmostEqual(self.share(risk=1, share=0.3, spread=0.5)[0], 0.45)

    def test_unknown_risk_gets_the_top_of_the_window(self) -> None:
        top = 0.3 * (1 + 0.5)
        self.assertEqual(self.share(risk=1.0, share=0.3, spread=0.5), (top, "share"))
        for r in (None, math.nan, math.inf, -math.inf, True, False, "0.5", [0.5]):
            with self.subTest(r=r):
                self.assertEqual(
                    self.share(risk=r, share=0.3, spread=0.5), (top, "unknown_risk")
                )
        self.assertEqual(
            self.share(risk=10**400, share=0.3, spread=0.5), (top, "unknown_risk")
        )

    def test_build_files_get_the_top_of_the_window(self) -> None:
        top = 0.3 * (1 + 0.5)
        for build_files in (None, True, False, 1, 7, 1.0, 0.0, "0", []):
            with self.subTest(build_files=build_files):
                self.assertEqual(
                    self.share(build_files=build_files, share=0.3, spread=0.5),
                    (top, "build_paths"),
                )
        self.assertEqual(self.share(build_files=0), (0.25, "share"))
        self.assertEqual(
            self.share(build_files=1, risk=None, share=0.3, spread=0.5),
            (top, "build_paths"),
        )

    def test_top_of_the_window_is_clamped(self) -> None:
        self.assertEqual(
            self.share(build_files=1, share=1.0, spread=0.5), (1.0, "build_paths")
        )
        self.assertEqual(
            self.share(risk=None, share=0.8, spread=0.5), (1.0, "unknown_risk")
        )

    def test_clamp(self) -> None:
        high = 0.4 * (1 + 0.5 * 1.0)
        self.assertEqual(self.share(risk=5.0, share=0.4, spread=0.5), (high, "share"))
        self.assertEqual(self.share(risk=-5, share=0.4, spread=0.5), (0.2, "share"))
        self.assertEqual(self.share(risk=1.0, share=1.0, spread=0.5), (1.0, "share"))
        unvalidated = make_model(budget=make_budget(share=0.05))
        self.assertEqual(
            scorer_model.budget_share(unvalidated, 0, 0.0),
            (scorer_model.MIN_SHARE, "share"),
        )


class TestBuildPaths(unittest.TestCase):
    def test_each_list(self) -> None:
        build = [d + "x/y.py" for d in scorer_paths.BUILD_DIRS]
        build += list(scorer_paths.BUILD_FILES)
        for name in scorer_paths.BUILD_BASENAMES:
            build += [name, f"torch/csrc/{name}"]
        build += [
            "requirements.txt",
            "requirements-ci.txt",
            "torch/requirements_dev.txt",
            "a/b.cmake",
            "x.bzl",
            "aten/src/ATen/native/native_functions.yaml",
            "aten/src/ATen/templates/Functions.h",
            "tools/jit/gen_unboxing.py",
            "tools/pyi/gen_pyi.py",
            "tools/amd_build/build_amd.py",
            "MODULE.bazel",
            ".bazelrc",
            ".bazelversion",
        ]
        for path in build:
            with self.subTest(path=path):
                self.assertTrue(scorer_paths.is_build_path(path))
        not_build = [
            "torch/nn/modules/linear.py",
            "test/test_nn.py",
            "test/inductor/test_x.py",
            "docs/source/nn.rst",
            "README.md",
            "foo/test/run_test.py",
            "tools/testing_extra/x.py",
            "third_party_like/x.c",
            "my_requirements.txt",
            "requirements.in",
            "requirements.txt.bak",
            "a/b.cmake.in",
            "setup.py.orig",
            "x/CMakeLists.txt.in",
            "torch/testing/_comparison.py",
            "aten/src/ATen/native/Linear.cpp",
            "tools/jit_extra/x.py",
            "MODULE.bazel.lock",
        ]
        for path in not_build:
            with self.subTest(path=path):
                self.assertFalse(scorer_paths.is_build_path(path))

    def test_count_build_files(self) -> None:
        paths = ["torch/a.py", "CMakeLists.txt", ".github/x.yml", "test/test_a.py"]
        self.assertEqual(scorer_paths.count_build_files(paths), 2)
        for empty in ([], [""], ["", ""]):
            self.assertIsNone(scorer_paths.count_build_files(empty))
        self.assertEqual(scorer_paths.count_build_files(["", "setup.py"]), 1)
        many = [f"torch/f{i}.py" for i in range(2999)]
        self.assertEqual(scorer_paths.count_build_files(many), 0)
        self.assertEqual(scorer_paths.count_build_files([*many, "torch/x.py"]), 3000)
        self.assertEqual(
            scorer_paths.count_build_files([*many, "torch/x.py", "torch/y.py"]), 3001
        )


class TestBudgetStaleness(unittest.TestCase):
    def build(
        self, version: str, today: date | None = TODAY, changed: Any = None
    ) -> dict[str, Any]:
        model = make_model(version=version, budget=make_budget())
        with mock.patch("builtins.print") as printed:
            block = scorer.build_scorer_block(
                make_aggregated(),
                [HEURISTIC_A, HEURISTIC_B],
                lambda: encode(model),
                ["test/test_a.py"] if changed is None else changed,
                today=today,
            )
        self.printed = [" ".join(map(str, c.args)) for c in printed.call_args_list]
        if block is None:
            self.fail("block omitted")
        self.assertEqual(block["reason"], "ok")
        without = make_model(version=version)
        self.assertEqual(
            {k: v for k, v in block["model"].items() if k != "budget"}, without
        )
        return block

    def assert_dropped(self, block: dict[str, Any], why: str) -> None:
        self.assertNotIn("budget", block["model"])
        self.assertTrue(block["budget_reason"].startswith("stale: "))
        self.assertIn(why, block["budget_reason"])
        self.assertEqual(
            self.printed[0], f"TD scorer: budget dropped: {block['budget_reason']!r}"
        )

    def test_dropped_budgets(self) -> None:
        for days in (4, 5, 7):
            version = (TODAY - timedelta(days=days)).isoformat()
            with self.subTest(version=version):
                block = self.build(version)
                self.assert_dropped(block, f"{version} is over 3 days before {TODAY}")
                self.assertAlmostEqual(block["risk"], -0.5)

    def test_kept_budgets(self) -> None:
        for days in (-3, -1, 0, 1):
            version = (TODAY + timedelta(days=days)).isoformat()
            with self.subTest(version=version):
                block = self.build(version)
                self.assertEqual(block["model"]["budget"], make_budget())
                self.assertIsNone(block["budget_reason"])

    def test_a_stale_model_drops_everything(self) -> None:
        for version in ("x", (TODAY - timedelta(days=8)).isoformat()):
            model = make_model(version=version, budget=make_budget())
            with self.subTest(version=version), mock.patch("builtins.print"):
                block = scorer.build_scorer_block(
                    make_aggregated(),
                    [HEURISTIC_A],
                    lambda: encode(model),
                    ["test/test_a.py"],
                    today=TODAY,
                )
                if block is None:
                    self.fail("block omitted")
                self.assertIsNone(block["model"])
                self.assertEqual(block["reason"], f"stale: {version}")
                self.assertIsNone(block["budget_reason"])

    def test_default_today_is_the_utc_date(self) -> None:
        today = datetime.now(UTC).date()
        block = self.build(today.isoformat(), today=None)
        self.assertIn("budget", block["model"])
        block = self.build((today - timedelta(days=5)).isoformat(), today=None)
        self.assert_dropped(block, "is over 3 days before")

    def test_build_files(self) -> None:
        cases: list[tuple[Any, Any]] = [
            (["test/test_a.py"], 0),
            (["CMakeLists.txt", "torch/a.py", "x.bzl"], 2),
            ([""], None),
            ([1], None),
        ]
        for changed, expected in cases:
            with self.subTest(changed=changed):
                block = self.build(TODAY.isoformat(), changed=changed)
                self.assertEqual(block["build_files"], expected)
        model = make_model()
        with mock.patch("builtins.print"):
            block = scorer.build_scorer_block(
                make_aggregated(), [HEURISTIC_A], lambda: encode(model), None
            )
        if block is None:
            self.fail("block omitted")
        self.assertIsNone(block["build_files"])
        self.assertIsNone(block["budget_reason"])

    def test_td_job_writes_the_budget_fields(self) -> None:
        job = TestTDJob()
        today = TD_JOB_NOW.date()
        for version, kept in (
            ((today - timedelta(days=5)).isoformat(), False),
            (today.isoformat(), True),
        ):
            with self.subTest(version=version):
                plain = make_model(version=version)
                model = make_model(version=version, budget=make_budget())
                td_results = job.run_main(
                    lambda _: encode(plain), budget=lambda _: encode(model)
                )
                block = td_results["scorer"]
                self.assertEqual(block["reason"], "ok")
                self.assertEqual("budget" in block["model"], kept)
                self.assertEqual(block["budget_reason"] is None, kept)
                self.assertEqual(block["build_files"], 0)


def budget_block(
    model: Any, build_files: Any = 0, risk: Any = 0.0, **scores: dict[str, float]
) -> dict[str, Any]:
    block = make_block(model=model, **scores)
    block.update(build_files=build_files, risk=risk, budget_reason=None)
    return block


def reference_cut(
    block: dict[str, Any], tests: list[str], include: list[TestRun], times: Any
) -> tuple[list[str], int, int, int]:
    """The contract's cut, written from its text: selection, budget_ms, cost_ms and
    the cost of the top-ranked file that a share forces in (0 when none)."""
    model = block["model"]
    td = list(dict.fromkeys(t.test_file for t in include))
    start_up = model["start_up_seconds"]["env1"]["default"]
    cost = {f: times[f] + start_up for f in tests}
    cost_ms = {f: round(c * 1000) for f, c in cost.items()}
    share, _ = scorer_model.budget_share(model, block["build_files"], block["risk"])
    if share is None:
        a = 1.0
        budget_ms = sum(cost_ms[f] for f in td)
    else:
        a = model["budget"]["cost_exponent"]
        budget_ms = round(share * sum(cost_ms.values()))
    must = sorted(
        f for f in tests if block["heuristics"].get("EditedByPR", {}).get(f, 0) != 0
    )

    def key(f: str) -> tuple[float, str]:
        scores = {h: s.get(f, 0.0) for h, s in block["heuristics"].items()}
        p = scorer_model.file_probability(model, scores, cost[f], "env1", "default")
        return -(p / cost[f] ** a), f

    order = sorted((f for f in tests if f not in must), key=key)
    selected = list(must)
    used = sum(cost_ms[f] for f in must)
    forced = 0
    for i, f in enumerate(order):
        if used + cost_ms[f] <= budget_ms:
            selected.append(f)
            used += cost_ms[f]
        elif i == 0 and share is not None:
            selected.append(f)
            used += cost_ms[f]
            forced = cost_ms[f]
    if not selected and order:
        selected = [order[0]]
        used = cost_ms[order[0]]
    return selected, budget_ms, used, forced


class TestJobBudget(unittest.TestCase):
    def test_modes(self) -> None:
        model = make_model(budget=make_budget(0.2, 0.5, 0.25))
        cases: list[tuple[Any, Any, Any, dict[str, Any]]] = [
            (make_model(), 0, 0.5, {"share": None, "mode": "none"}),
            (make_model(), 0, 0.5, {"share": None, "mode": "stale"}),
            (model, 0, 2.0, {"share": 0.2 * 1.5, "mode": "share", "risk": 1.0}),
            (model, 1, 0.5, {"share": 0.2 * 1.5, "mode": "build_paths", "risk": None}),
            (
                model,
                0,
                None,
                {"share": 0.2 * 1.5, "mode": "unknown_risk", "risk": None},
            ),
        ]
        for i, (m, build_files, risk, want) in enumerate(cases):
            reason = "stale: x" if want["mode"] == "stale" else None
            got = scorer_model.job_budget(m, build_files, risk, reason, 500, 2000)
            with self.subTest(case=i):
                self.assertEqual({k: got[k] for k in want}, want)
                if want["share"] is None:
                    self.assertEqual(
                        (got["risk"], got["cost_exponent"], got["budget_ms"]),
                        (None, 1.0, 500),
                    )
                else:
                    self.assertEqual(got["cost_exponent"], 0.25)
                    self.assertEqual(got["budget_ms"], round(want["share"] * 2000))
        odd = scorer_model.job_budget(make_model(), 0, 0.0, 5, 500, 2000)
        self.assertEqual(odd["mode"], "none")


class TestBudgetSelection(unittest.TestCase):
    def select(
        self, block: Any, include: list[TestRun], exclude: list[TestRun], times: Any
    ) -> tuple[list[TestRun], list[TestRun], dict[str, Any]]:
        return scorer.select_tests(block, include, exclude, times, "env1", "default")

    def test_matches_the_contract_and_never_overspends(self) -> None:
        rng = random.Random(7)
        modes = set()
        for trial in range(300):
            tests = [f"t{i:02d}" for i in range(rng.randint(1, 50))]
            times = {t: rng.choice([0.0, 1.0, rng.uniform(0, 3000)]) for t in tests}
            scores = {t: rng.random() for t in tests if rng.random() < 0.3}
            edited = {t: 1.0 for t in tests if rng.random() < 0.05}
            spread = rng.choice([0.0, 0.5, rng.uniform(0, 0.85)])
            share = rng.uniform(0.1001 / (1 - spread), 1.0)
            a = rng.choice([0.0, 0.25, 0.5, 0.75, 1.0, rng.random()])
            start_up = rng.choice([1.0, START_UP, rng.uniform(1, 600)])
            model = make_model(
                start_up_seconds={"env1": {"default": start_up}},
                budget=make_budget(share, spread, a),
            )
            block = budget_block(
                model,
                build_files=rng.choice([0, 0, 0, 2, None]),
                risk=rng.choice([None, rng.uniform(-1.5, 1.5), rng.uniform(-1, 1)]),
                HeuristicA=scores,
                EditedByPR=edited,
            )
            include, exclude = td_cut(tests, scores)
            new_include, new_exclude, info = self.select(block, include, exclude, times)
            expected, budget_ms, cost_ms, forced = reference_cut(
                block, tests, include, times
            )
            modes.add(info["budget_mode"])
            with self.subTest(trial=trial):
                self.assertEqual(names(new_include), expected)
                self.assertEqual(
                    (info["budget_ms"], info["cost_ms"]), (budget_ms, cost_ms)
                )
                self.assertEqual(sorted(names(new_include) + names(new_exclude)), tests)
                must_ms = sum(
                    round((times[f] + start_up) * 1000) for f in sorted(edited)
                )
                forced_top = not edited and len(new_include) == 1
                if not forced_top:
                    self.assertLessEqual(
                        info["cost_ms"], max(info["budget_ms"], must_ms) + forced
                    )
        self.assertEqual(modes, {"share", "build_paths", "unknown_risk"})

    def test_share_of_all_files(self) -> None:
        tests = [f"t{i}" for i in range(10)]
        times = dict.fromkeys(tests, 30.0)
        include, exclude = td_cut(tests, {"t3": 1.0})
        block = budget_block(
            make_model(budget=make_budget(0.4, 0.5, 0.5)),
            risk=0.5,
            HeuristicA={"t3": 1.0},
        )
        new_include, _, info = self.select(block, include, exclude, times)
        self.assertEqual(info["budget_ms"], round(0.4 * (1 + 0.5 * 0.5) * 400_000))
        self.assertEqual(info["budget_ms"], 200_000)
        self.assertEqual(len(new_include), 5)
        self.assertEqual(names(new_include)[0], "t3")
        self.assertEqual(
            {k: info[k] for k in ("share", "risk", "cost_exponent", "budget_mode")},
            {"share": 0.5, "risk": 0.5, "cost_exponent": 0.5, "budget_mode": "share"},
        )

    def test_risk_reported_is_the_clamped_float(self) -> None:
        tests = ["a", "b"]
        include, exclude = td_cut(tests, {})
        block = budget_block(make_model(budget=make_budget(0.5, 0.5)), risk=3)
        _, _, info = self.select(block, include, exclude, dict.fromkeys(tests, 1.0))
        self.assertEqual((info["risk"], info["share"]), (1.0, 0.75))

    def test_top_file_when_nothing_fits(self) -> None:
        tests = ["a", "b", "c"]
        times = {"a": 100.0, "b": 100.0, "c": 100.0}
        include, exclude = td_cut(tests, {"b": 1.0})
        block = budget_block(make_model(budget=make_budget(0.1)), HeuristicA={"b": 1.0})
        new_include, new_exclude, info = self.select(block, include, exclude, times)
        self.assertEqual(names(new_include), ["b"])
        self.assertEqual(names(new_exclude), ["a", "c"])
        self.assertEqual(info["budget_ms"], 33_000)
        self.assertEqual(info["cost_ms"], 110_000)

        # With a share the top-ranked file runs even past must-run files.
        block["heuristics"]["EditedByPR"] = {"c": 1.0}
        new_include, new_exclude, info = self.select(block, include, exclude, times)
        self.assertEqual(names(new_include), ["c", "b"])
        self.assertEqual(names(new_exclude), ["a"])

    def test_top_ranked_file_runs_past_the_budget_with_a_share(self) -> None:
        # "big" ranks first by p / cost but costs more than the whole budget.
        tests = ["big", "s1", "s2", "s3"]
        times = {"big": 290.0, "s1": 0.0, "s2": 0.0, "s3": 0.0}
        scores = {"big": 1.0}
        include, exclude = td_cut(tests, {"s1": 1.0, "s2": 0.5})
        model = make_model(budget=make_budget(0.1, 0.0, 0.0))
        new_include, _, info = self.select(
            budget_block(model, HeuristicA=scores), include, exclude, times
        )
        self.assertEqual(info["budget_ms"], 33_000)
        self.assertEqual(names(new_include), ["big"])
        self.assertEqual(info["cost_ms"], 300_000)
        # Without a share, big is not forced in: it does not fit TD's cost.
        old_include, _, old = self.select(
            make_block(HeuristicA=scores), include, exclude, times
        )
        self.assertNotIn("big", names(old_include))
        self.assertLessEqual(old["cost_ms"], old["budget_ms"])

    def test_total_cost_zero(self) -> None:
        block = budget_block(make_model(budget=make_budget()))
        new_include, new_exclude, info = self.select(block, [], [], {})
        self.assertEqual((new_include, new_exclude), ([], []))
        self.assertEqual((info["budget_ms"], info["cost_ms"], info["n"]), (0, 0, 0))

        model = make_model(
            start_up_seconds={"env1": {"default": 5.0}}, budget=make_budget()
        )
        times = {"a": -4.9999999, "b": -4.9999999}
        include, exclude = td_cut(["a", "b"], {})
        new_include, _, info = self.select(budget_block(model), include, exclude, times)
        self.assertEqual(names(new_include), ["a", "b"])
        self.assertEqual((info["budget_ms"], info["cost_ms"]), (0, 0))

    def test_no_block_or_stale_gives_the_td_cost_cut(self) -> None:
        rng = random.Random(11)
        for trial in range(100):
            tests = [f"t{i:02d}" for i in range(rng.randint(1, 40))]
            times = {t: rng.uniform(0, 1000) for t in tests}
            scores = {t: rng.random() for t in tests if rng.random() < 0.3}
            include, exclude = td_cut(tests, scores)
            old = self.select(make_block(HeuristicA=scores), include, exclude, times)
            stale = budget_block(make_model(), HeuristicA=scores)
            stale["budget_reason"] = "stale: version 'x' is not a date"
            cases = [
                (budget_block(make_model(), HeuristicA=scores), "none"),
                (budget_block(make_model(), None, None, HeuristicA=scores), "none"),
                (stale, "stale"),
            ]
            for block, mode in cases:
                with self.subTest(trial=trial, mode=mode):
                    new = self.select(block, include, exclude, times)
                    self.assertEqual(new[:2], old[:2])
                    for key in ("budget_ms", "cost_ms", "n", "seconds", "hash"):
                        self.assertEqual(new[2][key], old[2][key])
                    self.assertEqual(
                        (new[2]["share"], new[2]["risk"], new[2]["cost_exponent"]),
                        (None, None, 1.0),
                    )
                    self.assertEqual(new[2]["budget_mode"], mode)
        self.assertEqual(old[2]["budget_mode"], "none")

    def test_build_paths_and_unknown_risk_get_the_top_of_the_window(self) -> None:
        rng = random.Random(13)
        model = make_model(budget=make_budget(0.3, 0.5, 0.25))
        top = 0.3 * (1 + 0.5)
        differs = 0
        for trial in range(100):
            tests = [f"t{i:02d}" for i in range(rng.randint(1, 40))]
            times = {t: rng.uniform(0, 1000) for t in tests}
            scores = {t: rng.random() for t in tests if rng.random() < 0.3}
            include, exclude = td_cut(tests, scores)
            old = self.select(make_block(HeuristicA=scores), include, exclude, times)
            at_r1 = self.select(
                budget_block(model, 0, 1.0, HeuristicA=scores), include, exclude, times
            )
            self.assertEqual(at_r1[2]["share"], top)
            differs += at_r1[2]["budget_ms"] != old[2]["budget_ms"]
            cases = [
                (budget_block(model, 1, HeuristicA=scores), "build_paths"),
                (budget_block(model, None, HeuristicA=scores), "build_paths"),
                (budget_block(model, 0, None, HeuristicA=scores), "unknown_risk"),
                (budget_block(model, 0, math.nan, HeuristicA=scores), "unknown_risk"),
            ]
            for block, mode in cases:
                with self.subTest(trial=trial, mode=mode):
                    new = self.select(block, include, exclude, times)
                    self.assertEqual(new[:2], at_r1[:2])
                    for key in ("budget_ms", "cost_ms", "n", "seconds", "hash"):
                        self.assertEqual(new[2][key], at_r1[2][key])
                    self.assertEqual(
                        (new[2]["share"], new[2]["risk"], new[2]["cost_exponent"]),
                        (top, None, 0.25),
                    )
                    self.assertEqual(new[2]["budget_mode"], mode)
        self.assertGreater(differs, 90)

    def test_unlisted_job_keeps_td_cut_with_a_budget(self) -> None:
        block = budget_block(make_model(budget=make_budget()))
        include, exclude = td_cut(["a", "b"], {})
        with self.assertRaisesRegex(scorer.NoModel, "no start-up for env9/default"):
            scorer.select_tests(block, include, exclude, {}, "env9", "default")

    def test_odd_block_fields_do_not_raise(self) -> None:
        include, exclude = td_cut(["a", "b"], {})
        block = budget_block(make_model(budget=make_budget()), "3", "x")
        block["budget_reason"] = 5
        _, _, info = self.select(block, include, exclude, {"a": 1.0, "b": 1.0})
        self.assertEqual(info["budget_mode"], "build_paths")
        del block["build_files"], block["budget_reason"]
        _, _, info = self.select(block, include, exclude, {"a": 1.0, "b": 1.0})
        self.assertEqual(info["budget_mode"], "build_paths")

    def test_cost_exponent_scalar_parity(self) -> None:
        model = make_model()
        model["scorer"]["cost_weight"] = 0.0
        model["scorer"]["intercept"] = 0.0
        model["scorer"]["build_env"] = {}
        model["scorer"]["test_config"] = {}
        model["scorer"]["heuristics"] = {
            "HeuristicA": {"weight": 1.0, "x": [0.0, 1.0], "y": [0.001, 0.02]}
        }
        times = {"big": 390.0, "s1": 0.0, "s2": 0.0}
        p_big = scorer_model.file_probability(model, {"HeuristicA": 1.0}, 400.0, "", "")
        p_small = scorer_model.file_probability(model, {}, 10.0, "", "")
        self.assertAlmostEqual(p_big, 0.02)
        self.assertAlmostEqual(p_small, 0.001)
        include, exclude = td_cut(list(times), {"big": 1.0})
        for a, hand, expected in (
            (0.5, (0.02 / 20, 0.001 / 3.1622776601683795), ["big"]),
            (0.75, (0.02 / 89.44271909999159, 0.001 / 5.623413251903491), ["big"]),
            (1.0, (0.02 / 400, 0.001 / 10), ["s1", "s2"]),
        ):
            with self.subTest(a=a):
                self.assertEqual(
                    p_big / 400.0**a > p_small / 10.0**a, hand[0] > hand[1]
                )
                self.assertAlmostEqual(p_big / 400.0**a, hand[0])
                self.assertAlmostEqual(p_small / 10.0**a, hand[1])
                block = budget_block(
                    make_model(**{**model, "budget": make_budget(405 / 420, 0.0, a)}),
                    HeuristicA={"big": 1.0},
                )
                new_include, _, info = self.select(block, include, exclude, times)
                self.assertEqual(info["budget_ms"], 405_000)
                self.assertEqual(sorted(names(new_include)), expected)
                self.assertEqual(info["cost_exponent"], a)


if __name__ == "__main__":
    unittest.main()
