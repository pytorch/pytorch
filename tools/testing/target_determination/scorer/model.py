"""The learned TD scorer published by pytorch/test-infra (td_scorer.json,
format 1): loading, validation and evaluation, stdlib only, in float64 and in
the same order of operations as test-infra's trainer."""

from __future__ import annotations

import json
import math
import os
from bisect import bisect_right
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, TYPE_CHECKING

from tools.testing.target_determination.scorer.features import (
    parse_paths,
    PATH_FEATURES,
)


if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

    from tools.testing.target_determination.scorer.features import Paths


FORMAT = 1
MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_FEATURES = 200
MAX_TREES = 1000
MAX_NODES_PER_TREE = 255
MAX_DEPTH = 8
MAX_TOTAL_NODES = 20000
MIN_QUANTILES = 2
MAX_QUANTILES = 2001
MAX_CLIP = 0.01
MAX_TRAINED_AT_CHARS = 32
SHARE_FLOOR = 0.10
SHARE_CEILING = 0.40
# td_scorer.json format 1 takes the logistic of anything below this as 0.0, so
# both repos agree and exp() never overflows.
MIN_LOGIT = -700.0

_TREE_KEYS = ("feature", "threshold", "missing_left", "left", "right", "value", "leaf")
_RISK_LISTS = ("fill", "mean", "scale", "coef")


class ModelError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class Tree:
    feature: tuple[int, ...]
    threshold: tuple[float, ...]
    missing_left: tuple[bool, ...]
    left: tuple[int, ...]
    right: tuple[int, ...]
    value: tuple[float, ...]
    leaf: tuple[bool, ...]


@dataclass(frozen=True)
class Risk:
    features: tuple[str, ...]
    fill: tuple[float, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    coef: tuple[float, ...]
    intercept: float
    quantiles: tuple[float, ...]
    share_min: float
    share_max: float
    paths: Paths | None


@dataclass(frozen=True)
class RiskScore:
    risk: float
    percentile: float
    share: float


def _logistic(x: float) -> float:
    return 0.0 if x < MIN_LOGIT else 1.0 / (1.0 + math.exp(-x))


@dataclass(frozen=True)
class Model:
    trained_at: str | None
    features: tuple[str, ...]
    baseline: float
    trees: tuple[Tree, ...]
    platt_coef: float
    platt_intercept: float
    platt_clip: float
    risk_model: Risk | None
    risk_error: str | None

    def _raw(self, row: Sequence[float]) -> float:
        raw = self.baseline
        for tree in self.trees:
            n = 0
            while not tree.leaf[n]:
                x = row[tree.feature[n]]
                if x != x:
                    n = tree.left[n] if tree.missing_left[n] else tree.right[n]
                elif x <= tree.threshold[n]:
                    n = tree.left[n]
                else:
                    n = tree.right[n]
            raw += tree.value[n]
        return raw

    def predict(self, rows: Sequence[Sequence[float]]) -> list[float]:
        """p per row, each row holding the values of `features` in order.
        Raises ModelError unless every p is finite and > 0."""
        clip = self.platt_clip
        out = []
        for row in rows:
            if len(row) != len(self.features):
                raise ModelError(
                    f"row has {len(row)} values, model has {len(self.features)} features"
                )
            try:
                p_raw = _logistic(self._raw(row))
                p_raw = min(max(p_raw, clip), 1 - clip)
                z = math.log(p_raw / (1 - p_raw))
                p = 1.0 / (
                    1.0 + math.exp(-(self.platt_coef * z + self.platt_intercept))
                )
            except (OverflowError, ValueError) as e:
                raise ModelError(f"non-finite p: {e}") from e
            if not (math.isfinite(p) and p > 0):
                raise ModelError(f"non-finite or zero p: {p}")
            out.append(p)
        return out

    def risk(self, values: Mapping[str, float]) -> RiskScore | None:
        """The change's risk score, or None when the risk block is off or a
        value is infinite."""
        r = self.risk_model
        if r is None:
            return None
        s = r.intercept
        for name, fill, mean, scale, coef in zip(
            r.features, r.fill, r.mean, r.scale, r.coef
        ):
            x = values.get(name, math.nan)
            if x != x:
                x = fill
            s += coef * (x - mean) / scale
        if not math.isfinite(s):
            return None
        risk = _logistic(s)
        f = bisect_right(r.quantiles, risk) / len(r.quantiles)
        return RiskScore(risk, f, r.share_min + (r.share_max - r.share_min) * f)


def _number(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelError(f"{what}: not a number")
    try:
        number = float(value)
    except OverflowError as e:
        raise ModelError(f"{what}: not finite") from e
    if not math.isfinite(number):
        raise ModelError(f"{what}: not finite")
    return number


def _index(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ModelError(f"{what}: not an integer")
    return value


def _flag(value: object, what: str) -> bool:
    if not isinstance(value, bool):
        raise ModelError(f"{what}: not a bool")
    return value


def _list(value: object, what: str) -> list[Any]:
    if not isinstance(value, list):
        raise ModelError(f"{what}: not a list")
    return value


def _dict(value: object, what: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ModelError(f"{what}: not an object")
    return value


def _features(value: object, known: Collection[str]) -> tuple[str, ...]:
    names = _list(value, "invalid features")
    if not 1 <= len(names) <= MAX_FEATURES:
        raise ModelError(f"invalid features: {len(names)} names")
    for name in names:
        if not isinstance(name, str):
            raise ModelError("invalid features: not a string")
        if name not in known:
            raise ModelError(f"unknown feature: {name}")
    if len(set(names)) != len(names):
        raise ModelError("invalid features: duplicates")
    return tuple(names)


def _tree(index: int, value: object, n_features: int) -> Tree:
    what = f"invalid tree: {index}"
    doc = _dict(value, what)
    lists = {key: _list(doc.get(key), f"{what} {key}") for key in _TREE_KEYS}
    n = len(lists["leaf"])
    if not 1 <= n <= MAX_NODES_PER_TREE:
        raise ModelError(f"{what}: {n} nodes")
    if any(len(nodes) != n for nodes in lists.values()):
        raise ModelError(f"{what}: lists of different lengths")
    tree = Tree(
        feature=tuple(_index(x, what) for x in lists["feature"]),
        threshold=tuple(_number(x, what) for x in lists["threshold"]),
        missing_left=tuple(_flag(x, what) for x in lists["missing_left"]),
        left=tuple(_index(x, what) for x in lists["left"]),
        right=tuple(_index(x, what) for x in lists["right"]),
        value=tuple(_number(x, what) for x in lists["value"]),
        leaf=tuple(_flag(x, what) for x in lists["leaf"]),
    )
    depth: list[int | None] = [None] * n
    depth[0] = 0
    for node in range(n):
        if tree.leaf[node]:
            continue
        if not 0 <= tree.feature[node] < n_features:
            raise ModelError(f"{what}: node {node} feature out of range")
        node_depth = depth[node]
        for child in (tree.left[node], tree.right[node]):
            if not node < child < n:
                raise ModelError(f"{what}: node {node} child {child} invalid")
            if node_depth is None:
                continue
            if node_depth + 1 > MAX_DEPTH:
                raise ModelError(f"{what}: deeper than {MAX_DEPTH}")
            child_depth = depth[child]
            if child_depth is None or child_depth < node_depth + 1:
                depth[child] = node_depth + 1
    return tree


def _risk(value: object, known: Collection[str]) -> Risk:
    doc = _dict(value, "invalid risk")
    names = _list(doc.get("features"), "invalid risk features")
    for name in names:
        if not isinstance(name, str) or name not in known:
            raise ModelError(f"unknown risk feature: {name}")
    if len(set(names)) != len(names):
        raise ModelError("invalid risk features: duplicates")
    lists = {
        key: tuple(
            _number(x, f"invalid risk {key}")
            for x in _list(doc.get(key), f"invalid risk {key}")
        )
        for key in _RISK_LISTS
    }
    if any(len(numbers) != len(names) for numbers in lists.values()):
        raise ModelError("invalid risk: lists of different lengths")
    if any(scale <= 0 for scale in lists["scale"]):
        raise ModelError("invalid risk: scale <= 0")
    quantiles = tuple(
        _number(x, "invalid risk quantiles")
        for x in _list(doc.get("quantiles"), "invalid risk quantiles")
    )
    if not MIN_QUANTILES <= len(quantiles) <= MAX_QUANTILES:
        raise ModelError(f"invalid risk: {len(quantiles)} quantiles")
    if any(not 0 <= q <= 1 for q in quantiles) or any(
        a > b for a, b in pairwise(quantiles)
    ):
        raise ModelError("invalid risk: quantiles not ascending in [0, 1]")
    share_min = _number(doc.get("share_min"), "invalid risk share_min")
    share_max = _number(doc.get("share_max"), "invalid risk share_max")
    if not SHARE_FLOOR <= share_min <= share_max <= SHARE_CEILING:
        raise ModelError(
            f"invalid risk: shares not in order within [{SHARE_FLOOR}, {SHARE_CEILING}]"
        )
    paths = None
    if "paths" in doc:
        try:
            paths = parse_paths(doc["paths"])
        except ValueError as e:
            raise ModelError(f"invalid risk paths: {e}") from e
    elif set(names) & set(PATH_FEATURES):
        raise ModelError("invalid risk: no paths")
    return Risk(
        features=tuple(names),
        fill=lists["fill"],
        mean=lists["mean"],
        scale=lists["scale"],
        coef=lists["coef"],
        intercept=_number(doc.get("intercept"), "invalid risk intercept"),
        quantiles=quantiles,
        share_min=share_min,
        share_max=share_max,
        paths=paths,
    )


def parse_model(
    value: object,
    known_features: Collection[str],
    known_risk_features: Collection[str],
) -> Model:
    """Validates a decoded td_scorer.json; raises ModelError with a short
    reason. A bad or absent risk block only turns the risk score off."""
    doc = _dict(value, "format")
    if doc.get("format") != FORMAT or isinstance(doc.get("format"), bool):
        raise ModelError("format")
    if doc.get("enabled") is not True:
        raise ModelError("disabled")
    features = _features(doc.get("features"), known_features)
    scorer = _dict(doc.get("scorer"), "invalid scorer")
    trees_doc = _list(scorer.get("trees"), "invalid scorer trees")
    if not 1 <= len(trees_doc) <= MAX_TREES:
        raise ModelError(f"invalid scorer: {len(trees_doc)} trees")
    trees = tuple(_tree(i, t, len(features)) for i, t in enumerate(trees_doc))
    total = sum(len(tree.leaf) for tree in trees)
    if total > MAX_TOTAL_NODES:
        raise ModelError(f"invalid scorer: {total} nodes")
    platt = _dict(doc.get("platt"), "invalid platt")
    clip = _number(platt.get("clip"), "invalid platt clip")
    if not 0 < clip < MAX_CLIP:
        raise ModelError("invalid platt: clip out of range")

    risk_model: Risk | None = None
    risk_error: str | None = "absent"
    if "risk" in doc:
        try:
            risk_model, risk_error = _risk(doc["risk"], known_risk_features), None
        except ModelError as e:
            risk_error = e.reason

    trained_at = doc.get("trained_at")
    if isinstance(trained_at, str) and len(trained_at) > MAX_TRAINED_AT_CHARS:
        raise ModelError("format")
    return Model(
        trained_at=trained_at if isinstance(trained_at, str) else None,
        features=features,
        baseline=_number(scorer.get("baseline"), "invalid scorer baseline"),
        trees=trees,
        platt_coef=_number(platt.get("coef"), "invalid platt coef"),
        platt_intercept=_number(platt.get("intercept"), "invalid platt intercept"),
        platt_clip=clip,
        risk_model=risk_model,
        risk_error=risk_error,
    )


def load_model(
    path: str | os.PathLike[str],
    known_features: Collection[str],
    known_risk_features: Collection[str],
) -> Model:
    """Reads and validates td_scorer.json at `path` (at most MAX_FILE_BYTES);
    raises ModelError with a short reason."""
    try:
        size = os.path.getsize(path)
    except FileNotFoundError as e:
        raise ModelError("missing") from e
    except OSError as e:
        raise ModelError("unreadable") from e
    if size > MAX_FILE_BYTES:
        raise ModelError("too large")
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError, RecursionError) as e:
        raise ModelError("unreadable") from e
    return parse_model(doc, known_features, known_risk_features)
