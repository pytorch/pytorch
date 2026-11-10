"""The TD scorer model file: validation and evaluation.

Standard library only and free of repo imports: the TD job has only boto3, and
pytorch/test-infra loads this file by path to validate a model and check its
evaluator against the trainer before publishing.
"""

from __future__ import annotations

import json
import math
from bisect import bisect_right
from itertools import pairwise
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping


MAX_MODEL_BYTES = 1 << 20
MAX_NAME_CHARS = 200
MAX_HEURISTICS = 64
MAX_BUILD_ENVS = 2000
MAX_TEST_CONFIGS = 500
MAX_RISK_FEATURES = 32
MAX_BREAKPOINTS = 512
MAX_OVERHEAD_SECONDS = 600.0
N_QUANTILES = 101
Y_MIN = 1e-6
Y_MAX = 1 - 1e-6

TOP_KEYS = frozenset(
    {"schema", "version", "enabled", "overhead_seconds", "scorer", "risk"}
)
SCORER_KEYS = frozenset(
    {"intercept", "cost_weight", "heuristics", "build_env", "test_config"}
)
HEURISTIC_KEYS = frozenset({"weight", "x", "y"})
RISK_KEYS = frozenset({"intercept", "features", "quantiles"})
RISK_FEATURE_KEYS = frozenset({"weight", "mean", "scale"})
DOCS_SUFFIXES = (".md", ".rst", ".txt")
NON_SOURCE_DIRS = ("test/", "docs/", "third_party/")


def _is_docs(path: str) -> bool:
    return path.startswith("docs/") or path.endswith(DOCS_SUFFIXES)


def _is_source(path: str) -> bool:
    return not path.startswith(NON_SOURCE_DIRS) and not path.endswith(DOCS_SUFFIXES)


RISK_FEATURES: dict[str, Callable[[list[str]], int]] = {
    "n_files": len,
    "n_test_files": lambda paths: sum(p.startswith("test/") for p in paths),
    "n_source_files": lambda paths: sum(_is_source(p) for p in paths),
    "n_areas": lambda paths: len({p.split("/", 1)[0] for p in paths}),
    "docs_only": lambda paths: int(all(_is_docs(p) for p in paths)),
}


class ModelError(ValueError):
    pass


def _reject_constant(name: str) -> None:
    raise ModelError(f"non-finite number {name}")


def parse_model(raw: bytes) -> dict[str, Any]:
    """Size-check, parse and validate a td-scorer.json file."""
    if len(raw) > MAX_MODEL_BYTES:
        raise ModelError(f"file larger than {MAX_MODEL_BYTES} bytes")
    try:
        obj = json.loads(raw, parse_constant=_reject_constant)
    except ModelError:
        raise
    except ValueError as e:
        raise ModelError(f"not JSON: {e}") from e
    return validate_model(obj)


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelError(f"{where} is not a number")
    try:
        number = float(value)
    except OverflowError:
        number = math.inf
    if not math.isfinite(number):
        raise ModelError(f"{where} is not finite")
    return number


def _dict(value: Any, where: str, keys: frozenset[str] | None = None) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ModelError(f"{where} is not an object")
    if keys is not None and set(value) != keys:
        raise ModelError(f"{where} keys {sorted(value)} != {sorted(keys)}")
    return value


def _names(value: Any, where: str, limit: int) -> dict[str, Any]:
    mapping = _dict(value, where)
    if len(mapping) > limit:
        raise ModelError(f"{where} has more than {limit} entries")
    for name in mapping:
        if not name or len(name) > MAX_NAME_CHARS:
            raise ModelError(f"{where} name {name[:MAX_NAME_CHARS]!r} is invalid")
    return mapping


def _number_list(value: Any, where: str) -> list[float]:
    if not isinstance(value, list):
        raise ModelError(f"{where} is not a list")
    return [_number(v, f"{where}[{i}]") for i, v in enumerate(value)]


def _validate_heuristic(name: str, h: Any) -> None:
    where = f"heuristic {name!r}"
    _dict(h, where, HEURISTIC_KEYS)
    if _number(h["weight"], f"{where} weight") < 0:
        raise ModelError(f"{where} weight is negative")
    x = _number_list(h["x"], f"{where} x")
    y = _number_list(h["y"], f"{where} y")
    if not 1 <= len(x) <= MAX_BREAKPOINTS or len(x) != len(y):
        raise ModelError(f"{where} needs 1..{MAX_BREAKPOINTS} (x, y) pairs")
    if x[0] != 0.0:
        raise ModelError(f"{where} x[0] is not 0")
    if any(a >= b for a, b in pairwise(x)):
        raise ModelError(f"{where} x is not strictly increasing")
    if any(a > b for a, b in pairwise(y)):
        raise ModelError(f"{where} y is not non-decreasing")
    if any(not Y_MIN <= v <= Y_MAX for v in y):
        raise ModelError(f"{where} y is outside [{Y_MIN}, {Y_MAX}]")


def validate_model(value: Any) -> dict[str, Any]:
    """Return value if it is a valid model; raise ModelError otherwise."""
    obj = _dict(value, "model", TOP_KEYS)
    schema = obj["schema"]
    if isinstance(schema, bool) or schema != 1:
        raise ModelError(f"schema {schema!r} is not 1")
    version = obj["version"]
    if not isinstance(version, str) or len(version) > MAX_NAME_CHARS:
        raise ModelError(f"version is not a string of at most {MAX_NAME_CHARS} chars")
    if not isinstance(obj["enabled"], bool):
        raise ModelError("enabled is not a bool")
    overhead = _number(obj["overhead_seconds"], "overhead_seconds")
    if not 0 < overhead <= MAX_OVERHEAD_SECONDS:
        raise ModelError(f"overhead_seconds is outside (0, {MAX_OVERHEAD_SECONDS}]")

    scorer = _dict(obj["scorer"], "scorer", SCORER_KEYS)
    _number(scorer["intercept"], "scorer intercept")
    _number(scorer["cost_weight"], "scorer cost_weight")
    for name, h in _names(scorer["heuristics"], "heuristics", MAX_HEURISTICS).items():
        _validate_heuristic(name, h)
    for key, limit in (
        ("build_env", MAX_BUILD_ENVS),
        ("test_config", MAX_TEST_CONFIGS),
    ):
        for name, value in _names(scorer[key], key, limit).items():
            _number(value, f"{key} {name!r}")

    risk = _dict(obj["risk"], "risk", RISK_KEYS)
    _number(risk["intercept"], "risk intercept")
    features = _names(risk["features"], "risk features", MAX_RISK_FEATURES)
    for name, f in features.items():
        where = f"risk feature {name!r}"
        if name not in RISK_FEATURES:
            raise ModelError(f"{where} is unknown")
        _dict(f, where, RISK_FEATURE_KEYS)
        _number(f["weight"], f"{where} weight")
        _number(f["mean"], f"{where} mean")
        if _number(f["scale"], f"{where} scale") <= 0:
            raise ModelError(f"{where} scale is not positive")
    q = _number_list(risk["quantiles"], "risk quantiles")
    if len(q) != N_QUANTILES:
        raise ModelError(f"risk quantiles length {len(q)} != {N_QUANTILES}")
    if any(a > b for a, b in pairwise(q)):
        raise ModelError("risk quantiles are not non-decreasing")
    return obj


def interp(xs: list[float], ys: list[float], s: float) -> float:
    """Piecewise linear through (xs, ys), clamped at both ends."""
    if s <= xs[0]:
        return ys[0]
    if s >= xs[-1]:
        return ys[-1]
    i = bisect_right(xs, s) - 1
    t = (s - xs[i]) / (xs[i + 1] - xs[i])
    return ys[i] + t * (ys[i + 1] - ys[i])


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def file_probability(
    model: Mapping[str, Any],
    scores: Mapping[str, float],
    cost: float,
    build_env_key: str,
    test_config: str,
    raised: Iterable[str] = (),
) -> float:
    """p that one test file fails in one job.

    scores maps heuristic name to this file's score (missing means 0.0); cost is
    the file's seconds plus overhead_seconds. build_env_key is the job's
    test-times key: run_test.py's get_job_base_name(JOB_NAME), or
    BUILD_ENVIRONMENT when JOB_NAME is empty.
    """
    s = model["scorer"]
    z = (
        s["intercept"]
        + s["cost_weight"] * math.log(cost)
        + s["build_env"].get(build_env_key, 0.0)
        + s["test_config"].get(test_config, 0.0)
    )
    skip = set(raised)
    for name, h in s["heuristics"].items():
        if name in skip:
            continue
        c = interp(h["x"], h["y"], scores.get(name, 0.0))
        z += h["weight"] * math.log(c / (1 - c))
    p = _sigmoid(z)
    if not math.isfinite(p):
        raise ValueError(f"p is not finite (z={z})")
    return p


def change_features(paths: list[str]) -> dict[str, int]:
    """Risk features of a change, as raw counts, from its changed file paths."""
    return {name: feature(paths) for name, feature in RISK_FEATURES.items()}


def risk_score(model: Mapping[str, Any], features: Mapping[str, int]) -> float:
    """r in [-1, 1]: the change's raw risk placed in the model's quantile table."""
    risk = model["risk"]
    raw = risk["intercept"]
    for name, f in risk["features"].items():
        raw += f["weight"] * (math.log(1 + features[name]) - f["mean"]) / f["scale"]
    if not math.isfinite(raw):
        raise ValueError(f"raw risk is not finite ({raw})")
    q = risk["quantiles"]
    if raw < q[0]:
        frac = 0.0
    elif raw >= q[-1]:
        frac = 1.0
    else:
        i = bisect_right(q, raw) - 1
        frac = (i + (raw - q[i]) / (q[i + 1] - q[i])) / (N_QUANTILES - 1)
        frac = min(max(frac, 0.0), 1.0)
    return 2 * frac - 1
