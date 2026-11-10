"""The TD job's scorer step: p per test from the learned model, written to
td_results.json next to TD's own scores. Any problem gives a fallback block
and leaves TD's results as they are."""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any, NamedTuple, TYPE_CHECKING

from tools.stats.import_test_stats import (
    ADDITIONAL_CI_FILES_FOLDER,
    TEST_FILE_RATINGS_FILE,
)
from tools.testing.target_determination.scorer.features import (
    build_rows,
    LLM_FEATURE,
    MODEL_FEATURES,
    RISK_FEATURES,
    risk_values,
    served_scores,
    test_seconds,
    Unit,
    UnregisteredHeuristic,
)
from tools.testing.target_determination.scorer.model import load_model, ModelError


if TYPE_CHECKING:
    import os
    from collections.abc import Collection, Mapping, Sequence

    from tools.testing.target_determination.heuristics.interface import (
        AggregatedHeuristics,
    )
    from tools.testing.target_determination.scorer.model import Model, RiskScore


REPO_ROOT = Path(__file__).resolve().parents[4]
RATINGS_PATH = REPO_ROOT / ADDITIONAL_CI_FILES_FOLDER / TEST_FILE_RATINGS_FILE
MAX_REASON_CHARS = 300


class DegradedInputs(Exception):
    """An input the model reads is missing: training never saw it zeroed."""


class ScorerResult(NamedTuple):
    block: dict[str, Any]  # td_results.json's "scorer"
    metric: dict[str, Any]  # the td_scorer metric


def _registered_names() -> list[str]:
    from tools.testing.target_determination.heuristics import HEURISTICS

    return [heuristic.name for heuristic in HEURISTICS]


def _risk(
    model: Model, unit: Unit, p: Sequence[float], llm: Sequence[float] | None
) -> RiskScore | None:
    risk_model = model.risk_model
    if risk_model is None:
        print(f"TD scorer: no risk score ({model.risk_error})")
        return None
    try:
        score = model.risk(risk_values(risk_model.features, unit, p, llm))
    except Exception:
        print(f"TD scorer: no risk score:\n{traceback.format_exc()}")
        return None
    if score is None:
        print("TD scorer: no risk score (an infinite input)")
    return score


def _llm_scores(
    aggregated: AggregatedHeuristics,
    tests: Sequence[str],
    registered: Collection[str],
) -> list[float] | None:
    try:
        served = served_scores(aggregated, [LLM_FEATURE], registered)
    except UnregisteredHeuristic:
        return None
    return [served[LLM_FEATURE].get(test, 0.0) for test in tests]


def _has_ratings(path: Path) -> bool:
    try:
        with open(path) as f:
            ratings = json.load(f)
    except (OSError, ValueError):
        return False
    return isinstance(ratings, dict) and bool(ratings)


def check_inputs(
    features: Collection[str],
    tests: Sequence[str],
    aggregated: AggregatedHeuristics,
    test_times: Mapping[str, Mapping[str, Mapping[str, float]]],
    ratings_path: Path,
    registered: Collection[str],
) -> None:
    """Raises DegradedInputs when an input behind one of the model's
    `features` is missing for the whole job."""
    if "test_seconds" in features:
        seconds = test_seconds(test_times)
        if not any(test in seconds for test in tests):
            raise DegradedInputs("no test times")
    if LLM_FEATURE in features:
        llm = served_scores(aggregated, [LLM_FEATURE], registered)[LLM_FEATURE]
        if not any(llm.get(test, 0.0) != 0.0 for test in tests):
            raise DegradedInputs("no LLM scores")
    if "fail_n" in features and not _has_ratings(ratings_path):
        raise DegradedInputs("no failure ratings")


def compute_scorer(
    tests: Sequence[str],
    aggregated: AggregatedHeuristics,
    test_times: Mapping[str, Mapping[str, Mapping[str, float]]],
    model_path: str | os.PathLike[str],
    unit: Unit | None = None,
    registered: Collection[str] | None = None,
    ratings_path: Path | None = None,
) -> ScorerResult:
    """The scorer block for td_results.json and the td_scorer metric. Never
    raises: any problem gives {"fallback": reason}."""
    metric: dict[str, Any] = {
        "trained_at": None,
        "fallback": None,
        "heuristics_raised": None,
        "risk": None,
        "risk_percentile": None,
        "budget_share": None,
        "n_tests": len(tests),
    }
    try:
        if registered is None:
            registered = _registered_names()
        if unit is None:
            unit = Unit()
        metric["heuristics_raised"] = sorted(
            heuristic
            for heuristic in set(registered)
            if aggregated.get_heuristic_results(heuristic) is None
        )
        model = load_model(model_path, MODEL_FEATURES, RISK_FEATURES)
        metric["trained_at"] = model.trained_at
        check_inputs(
            model.features,
            tests,
            aggregated,
            test_times,
            RATINGS_PATH if ratings_path is None else ratings_path,
            registered,
        )
        rows = build_rows(
            model.features, tests, aggregated, test_times, unit, registered
        )
        p = model.predict(rows)
        score = _risk(model, unit, p, _llm_scores(aggregated, tests, registered))
        block = {
            "trained_at": model.trained_at,
            "p": dict(zip(tests, p)),
            "risk": None if score is None else score.risk,
            "risk_percentile": None if score is None else score.percentile,
            "budget_share": None if score is None else score.share,
        }
        json.dumps(block, allow_nan=False)
    except ModelError as e:
        reason = e.reason
    except DegradedInputs as e:
        reason = str(e)
    except Exception as e:
        print(f"TD scorer failed:\n{traceback.format_exc()}")
        reason = f"{type(e).__name__}: {e}"
    else:
        metric.update(
            risk=block["risk"],
            risk_percentile=block["risk_percentile"],
            budget_share=block["budget_share"],
        )
        print(
            f"TD scorer: model {model.trained_at}, {len(p)} tests, "
            f"risk {block['risk']}, budget share {block['budget_share']}"
        )
        return ScorerResult(block, metric)
    reason = reason[:MAX_REASON_CHARS]
    print(f"TD scorer: falling back to TD's ranking ({reason})")
    metric["fallback"] = reason
    return ScorerResult({"fallback": reason}, metric)
