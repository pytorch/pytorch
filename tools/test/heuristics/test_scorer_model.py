from __future__ import annotations

import copy
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT))

from tools.test.heuristics import scorer_fixture as fixture
from tools.testing.target_determination.scorer import model as scorer_model
from tools.testing.target_determination.scorer.features import path_values, Paths


sys.path.remove(str(REPO_ROOT))

KNOWN = {"fail_n", "test_seconds", "is_pr", "edited"}
KNOWN_RISK = {"log_files", "path_revert_max", "path_revert_mean"}
TOLERANCE = 1e-11
CASE_PATHS = Paths(
    fixture.PATHS_CASE["paths"]["base"], fixture.PATHS_CASE["paths"]["nodes"]
)
RISK_VALUES = {
    "log_files": math.log1p(len(set(fixture.RISK_CASE["changed"]))),
    **path_values(CASE_PATHS, fixture.RISK_CASE["changed"]),
}


def parse(doc: Any) -> scorer_model.Model:
    return scorer_model.parse_model(doc, KNOWN, KNOWN_RISK)


def chain_tree(depth: int) -> dict[str, Any]:
    """A tree whose leftmost leaf is `depth` edges below the root."""
    feature, threshold, missing, left, right, value, leaf = [], [], [], [], [], [], []
    for level in range(depth):
        node = 2 * level
        feature.append(0)
        threshold.append(0.5)
        missing.append(True)
        left.append(node + 2 if level < depth - 1 else node + 1)
        right.append(node + 1)
        value.append(0.0)
        leaf.append(False)
        # the right child of every split is a leaf
        feature.append(-1)
        threshold.append(0.0)
        missing.append(False)
        left.append(0)
        right.append(0)
        value.append(0.1)
        leaf.append(True)
    # the deepest split's left child
    feature.append(-1)
    threshold.append(0.0)
    missing.append(False)
    left.append(0)
    right.append(0)
    value.append(0.2)
    leaf.append(True)
    last = 2 * (depth - 1)
    left[last] = len(leaf) - 1
    return {
        "feature": feature,
        "threshold": threshold,
        "missing_left": missing,
        "left": left,
        "right": right,
        "value": value,
        "leaf": leaf,
    }


class TestParity(unittest.TestCase):
    def test_predict_matches_fixture(self) -> None:
        model = parse(fixture.DOC)
        self.assertEqual(model.trained_at, "2026-11-10T00:00:00Z")
        for p, expected in zip(model.predict(fixture.ROWS), fixture.EXPECTED_P):
            self.assertAlmostEqual(p, expected, delta=TOLERANCE)

    def test_risk_matches_fixture(self) -> None:
        model = parse(fixture.DOC)
        if model.risk_model is None:
            self.fail(f"no risk model: {model.risk_error}")
        self.assertEqual(model.risk_model.paths, CASE_PATHS)
        score = model.risk(RISK_VALUES)
        if score is None:
            self.fail("no risk score")
        case = fixture.RISK_CASE
        self.assertAlmostEqual(score.risk, case["expected_risk"], delta=TOLERANCE)
        self.assertEqual(score.percentile, case["expected_percentile"])
        self.assertAlmostEqual(score.share, case["expected_share"], delta=TOLERANCE)

    def test_risk_nan_uses_fill(self) -> None:
        model = parse(fixture.DOC)
        nan = model.risk({"log_files": math.nan, "path_revert_max": math.nan})
        filled = model.risk(
            {"log_files": 0.0, "path_revert_max": -2.0, "path_revert_mean": -2.5}
        )
        self.assertIsNotNone(filled)
        self.assertEqual(nan, filled)
        self.assertEqual(model.risk({}), filled)

    def test_risk_infinite_value_is_none(self) -> None:
        self.assertIsNone(
            parse(fixture.DOC).risk({**RISK_VALUES, "log_files": math.inf})
        )

    def test_very_negative_risk_logit_is_zero(self) -> None:
        doc = copy.deepcopy(fixture.DOC)
        doc["risk"]["intercept"] = -1e6
        score = parse(doc).risk(RISK_VALUES)
        self.assertEqual(score, scorer_model.RiskScore(0.0, 0.0, 0.1))

    def test_predict_rejects_wrong_row_length(self) -> None:
        with self.assertRaises(scorer_model.ModelError):
            parse(fixture.DOC).predict([[0.0, 1.0]])

    def test_predict_rejects_overflowing_platt(self) -> None:
        doc = copy.deepcopy(fixture.DOC)
        doc["platt"]["coef"] = 1e300
        with self.assertRaises(scorer_model.ModelError):
            parse(doc).predict(fixture.ROWS)

    def test_very_negative_raw_does_not_overflow(self) -> None:
        doc = copy.deepcopy(fixture.DOC)
        doc["scorer"]["baseline"] = -1e6
        p = parse(doc).predict(fixture.ROWS)
        self.assertTrue(all(math.isfinite(x) and x > 0 for x in p))


class TestValidation(unittest.TestCase):
    def assert_rejects(self, doc: Any, reason: str) -> None:
        with self.assertRaises(scorer_model.ModelError) as ctx:
            parse(doc)
        self.assertIn(reason, ctx.exception.reason)

    def mutate(self, path: list[Any], value: Any) -> dict[str, Any]:
        doc = copy.deepcopy(fixture.DOC)
        target = doc
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        return doc

    def test_format(self) -> None:
        self.assert_rejects(self.mutate(["format"], 2), "format")
        self.assert_rejects(self.mutate(["format"], True), "format")
        self.assert_rejects([], "format")

    def test_disabled(self) -> None:
        self.assert_rejects(self.mutate(["enabled"], False), "disabled")
        self.assert_rejects(self.mutate(["enabled"], 1), "disabled")

    def test_trained_at_length(self) -> None:
        self.assertEqual(
            parse(self.mutate(["trained_at"], "x" * 32)).trained_at, "x" * 32
        )
        self.assert_rejects(self.mutate(["trained_at"], "x" * 33), "format")

    def test_unknown_feature(self) -> None:
        self.assert_rejects(
            self.mutate(["features"], ["fail_n", "nope", "is_pr"]),
            "unknown feature: nope",
        )

    def test_features_count_and_duplicates(self) -> None:
        self.assert_rejects(self.mutate(["features"], []), "invalid features")
        self.assert_rejects(
            self.mutate(["features"], ["fail_n", "fail_n", "is_pr"]), "duplicates"
        )

    def test_bad_child_index(self) -> None:
        doc = self.mutate(["scorer", "trees", 0, "right", 0], 9)
        self.assert_rejects(doc, "invalid tree: 0")

    def test_child_not_greater(self) -> None:
        doc = self.mutate(["scorer", "trees", 0, "left", 2], 1)
        self.assert_rejects(doc, "invalid tree: 0")

    def test_feature_out_of_range(self) -> None:
        doc = self.mutate(["scorer", "trees", 1, "feature", 0], 3)
        self.assert_rejects(doc, "invalid tree: 1")

    def test_unequal_lists(self) -> None:
        doc = self.mutate(["scorer", "trees", 1, "value"], [0.0, 0.1])
        self.assert_rejects(doc, "invalid tree: 1")

    def test_depth(self) -> None:
        parse(self.mutate(["scorer", "trees"], [chain_tree(8)]))
        self.assert_rejects(
            self.mutate(["scorer", "trees"], [chain_tree(9)]), "deeper than 8"
        )

    def test_too_many_nodes(self) -> None:
        leaf: dict[str, list[Any]] = {
            "feature": [-1],
            "threshold": [0.0],
            "missing_left": [False],
            "left": [0],
            "right": [0],
            "value": [0.0],
            "leaf": [True],
        }
        big = {key: values * 256 for key, values in leaf.items()}
        self.assert_rejects(self.mutate(["scorer", "trees"], [big]), "256 nodes")
        tree = {key: values * 21 for key, values in leaf.items()}
        self.assert_rejects(
            self.mutate(["scorer", "trees"], [tree] * 1000), "21000 nodes"
        )
        self.assert_rejects(
            self.mutate(["scorer", "trees"], [leaf] * 1001), "1001 trees"
        )
        self.assert_rejects(self.mutate(["scorer", "trees"], []), "0 trees")

    def test_non_finite_and_bool_numbers(self) -> None:
        for bad in (math.nan, math.inf, True, "1.0", None):
            self.assert_rejects(
                self.mutate(["scorer", "trees", 0, "threshold", 0], bad),
                "invalid tree: 0",
            )
            self.assert_rejects(
                self.mutate(["scorer", "baseline"], bad), "invalid scorer"
            )
            self.assert_rejects(self.mutate(["platt", "coef"], bad), "invalid platt")
        self.assert_rejects(
            self.mutate(["scorer", "trees", 0, "left", 0], True), "invalid tree: 0"
        )
        self.assert_rejects(
            self.mutate(["scorer", "trees", 0, "leaf", 0], 0), "invalid tree: 0"
        )

    def test_platt_clip(self) -> None:
        for clip in (0.0, 0.01, -1e-7):
            self.assert_rejects(self.mutate(["platt", "clip"], clip), "clip")

    def test_bad_risk_disables_only_risk(self) -> None:
        bad_blocks: list[tuple[list[Any], Any]] = [
            (["risk", "scale", 0], 0.0),
            (["risk", "quantiles"], [0.5, 0.1]),
            (["risk", "quantiles"], [0.5]),
            (["risk", "quantiles"], [0.1, 1.5]),
            (["risk", "share_min"], 0.3),
            (["risk", "features"], ["log_files", "nope"]),
            (["risk", "coef"], [0.7]),
            (["risk", "intercept"], math.nan),
            (["risk"], "x"),
        ]
        for path, value in bad_blocks:
            model = parse(self.mutate(path, value))
            self.assertIsNone(model.risk_model, path)
            self.assertIsNotNone(model.risk_error, path)
            self.assertIsNone(model.risk(RISK_VALUES))
            self.assertEqual(len(model.predict(fixture.ROWS)), len(fixture.ROWS))

    def test_absent_risk(self) -> None:
        doc = copy.deepcopy(fixture.DOC)
        del doc["risk"]
        model = parse(doc)
        self.assertEqual(model.risk_error, "absent")
        self.assertIsNone(model.risk(RISK_VALUES))

    def assert_bad_paths(self, paths: Any, reason: str) -> None:
        doc = copy.deepcopy(fixture.DOC)
        if paths is None:
            del doc["risk"]["paths"]
        else:
            doc["risk"]["paths"] = paths
        model = parse(doc)
        self.assertIsNone(model.risk_model, reason)
        self.assertIn(reason, str(model.risk_error))
        self.assertEqual(len(model.predict(fixture.ROWS)), len(fixture.ROWS))

    def test_paths_required_by_a_path_feature(self) -> None:
        self.assert_bad_paths(None, "no paths")
        doc = copy.deepcopy(fixture.DOC)
        del doc["risk"]["paths"]
        for key in ("fill", "mean", "scale", "coef"):
            doc["risk"][key] = doc["risk"][key][:1]
        doc["risk"]["features"] = ["log_files"]
        risk_model = parse(doc).risk_model
        if risk_model is None:
            self.fail("a risk block without path features needs no paths")
        self.assertIsNone(risk_model.paths)

    def test_bad_paths_disable_only_risk(self) -> None:
        good = fixture.DOC["risk"]["paths"]
        nodes = good["nodes"]
        cases: list[tuple[Any, str]] = [
            ("x", "not an object"),
            ({"nodes": nodes}, "base: not a number"),
            ({"base": True, "nodes": nodes}, "base: not a number"),
            ({"base": 1.5, "nodes": nodes}, "base: not in [0, 1]"),
            ({"base": -0.1, "nodes": nodes}, "base: not in [0, 1]"),
            ({"base": 0.1}, "nodes: not an object"),
            ({"base": 0.1, "nodes": [["a/", 0.1]]}, "nodes: not an object"),
            ({"base": 0.1, "nodes": {"a/": 2.0}}, "node: not in [0, 1]"),
            ({"base": 0.1, "nodes": {"a/": "0.1"}}, "node: not a number"),
            ({"base": 0.1, "nodes": {"a/": None}}, "node: not a number"),
            ({"base": 0.1, "nodes": {"": 0.1}}, "invalid node key"),
            ({"base": 0.1, "nodes": {"a" * 513: 0.1}}, "invalid node key"),
            ({"base": 0.1, "nodes": {"a/\tb": 0.1}}, "invalid node key"),
            ({"base": 0.1, "nodes": {"a/\x1fb": 0.1}}, "invalid node key"),
            ({"base": 0.1, "nodes": {"a/\x7fb": 0.1}}, "invalid node key"),
            (
                {"base": 0.1, "nodes": {f"d{i}/": 0.1 for i in range(20001)}},
                "20001 nodes",
            ),
        ]
        for paths, reason in cases:
            self.assert_bad_paths(paths, reason)

    def test_paths_limits_accepted(self) -> None:
        doc = copy.deepcopy(fixture.DOC)
        nodes = {f"d{i}/": 0.0 for i in range(19999)}
        nodes["a" * 512] = 1.0
        doc["risk"]["paths"] = {"base": 0, "nodes": nodes, "extra": 1}
        risk_model = parse(doc).risk_model
        if risk_model is None or risk_model.paths is None:
            self.fail("paths at the limits rejected")
        self.assertEqual(len(risk_model.paths.nodes), 20000)
        self.assertEqual(risk_model.paths.base, 0.0)


class TestLoad(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "td_scorer.json"

    def tearDown(self) -> None:
        self.dir.cleanup()

    def load(self) -> scorer_model.Model:
        return scorer_model.load_model(self.path, KNOWN, KNOWN_RISK)

    def assert_reason(self, reason: str) -> None:
        with self.assertRaises(scorer_model.ModelError) as ctx:
            self.load()
        self.assertEqual(ctx.exception.reason, reason)

    def test_loads(self) -> None:
        self.path.write_text(json.dumps(fixture.DOC))
        self.assertEqual(self.load().features, ("fail_n", "test_seconds", "is_pr"))

    def test_missing(self) -> None:
        self.assert_reason("missing")

    def test_unreadable(self) -> None:
        self.path.write_text("{not json")
        self.assert_reason("unreadable")
        self.path.write_bytes(b"\xff\xfe")
        self.assert_reason("unreadable")

    def test_too_large_is_checked_before_parsing(self) -> None:
        with open(self.path, "w") as f:
            f.write(" " * (scorer_model.MAX_FILE_BYTES + 1))
        self.assertGreater(os.path.getsize(self.path), scorer_model.MAX_FILE_BYTES)
        self.assert_reason("too large")

    def test_nan_token_is_rejected(self) -> None:
        self.path.write_text(
            json.dumps(fixture.DOC).replace('"baseline": -3.0', '"baseline": NaN')
        )
        with self.assertRaises(scorer_model.ModelError):
            self.load()


if __name__ == "__main__":
    unittest.main()
