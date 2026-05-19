"""Observe ``torch.compile`` recompiles and suggest a ``ShapesSpec``.

Recompile signals come from two sources:

- ``torch._dynamo.utils._compilation_metrics`` (deque of ``CompilationMetrics``):
  provides ``n_compiles`` and ``entire_frame_compile_time_s``.
- ``torch._dynamo.utils.guard_failures`` (defaultdict keyed by code obj):
  every guard-fail reason string Dynamo composed, regardless of whether the
  compile path classified the event as a "recompile". This is the reliable
  source for parsing — entries like::

      1/1: tensor 'L['x']' size mismatch at index 0. expected 4, actual 8
      0/3: L['n'] == 10

  exist here even when ``CompilationMetrics.recompile_reason`` is ``None``
  (e.g. under ``automatic_dynamic_shapes=True``).

The bot is useful when its proposed spec beats Dynamo's default
``automatic_dynamic_shapes`` behavior — primarily for scalar ints, branching
on size, and 0/1 specialization. For workloads where auto-dynamic already
settles in 1-2 compiles, the suggester should produce an empty spec and the
caller is told to stick with the default.

This module is opt-in and side-effect free at import time.
"""

from __future__ import annotations

import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from torch.fx.experimental.dynamic_spec import (
    IntVar,
    ParamsSpec,
    ShapesSpec,
    ShapeVar,
    TensorSpec,
)


if TYPE_CHECKING:
    from collections.abc import Iterator


__all__ = [
    "CompileSnapshot",
    "SourceObservation",
    "dynamic_spec_observer",
    "suggest_spec_for",
]


# Reason-string formats observed in torch._dynamo.utils.guard_failures
# (verified against actual output, not Source.name() conventions):
#
#   "0/0: tensor 'x' size mismatch at index 0. expected 3, actual 10"
#   "0/2: tensor 'x' size mismatch at index 0. expected 0, actual 1"   (0/1 specialization)
#   "0/1: 6 <= x.size()[0]  # ..."                                     (sympy branch guard)
#   "0/3: L['n'] == 10"                                                (scalar specialization, when it happens)
#
# Source is the bare argname ('x'), not the full Source.name() chain.
_TENSOR_SIZE_MISMATCH = re.compile(
    r"tensor\s+'(?P<src>[^']+)'\s+"
    r"size mismatch at index (?P<dim>\d+)\.\s+"
    r"expected\s+(?P<expected>-?\d+),\s+actual\s+(?P<actual>-?\d+)\b"
)

# Sympy-style guards that reference a dim. We don't need the comparator —
# the presence of the guard tells us the dim was branched/specialized on.
# Used to mark a dim as "guarded" even when no values are directly extractable.
_SYMPY_SIZE_REF = re.compile(r"\b(?P<src>[A-Za-z_]\w*)\.size\(\)\[(?P<dim>\d+)\]")

# Scalar-int equals guards on a local. Format from the legacy L[...] form;
# kept for backward compatibility in case dynamo emits it.
_LOCAL_EQ_INT = re.compile(r"L\['(?P<src>[^']+)'\]\s*==\s*(?P<value>-?\d+)\b")


@dataclass
class SourceObservation:
    """Per-source aggregation extracted from recompile reasons.

    Either ``dim_values`` is populated (tensor-dim guard) or
    ``scalar_values`` is (scalar-int guard) — never both.
    """

    source: str
    dim_values: dict[int, set[int]] = field(default_factory=dict)
    scalar_values: set[int] = field(default_factory=set)

    def is_tensor_dim(self) -> bool:
        return bool(self.dim_values)

    def is_scalar(self) -> bool:
        return bool(self.scalar_values)


@dataclass
class CompileSnapshot:
    """Result of one ``dynamic_spec_observer()`` window."""

    co_name: str | None
    n_compiles: int
    cold_time_s: float
    warm_time_s: float
    total_compile_time_s: float
    observations: dict[str, SourceObservation]
    reasons: list[str]
    observed_co_names: list[str | None] = field(default_factory=list)

    def __repr__(self) -> str:
        lines = [
            f"CompileSnapshot(co_name={self.co_name!r}):",
            f"  n_compiles            = {self.n_compiles}",
            f"  cold_time_s           = {self.cold_time_s:.4f}",
            f"  warm_time_s           = {self.warm_time_s:.4f}",
            f"  total_compile_time_s  = {self.total_compile_time_s:.4f}",
            f"  observations          = {len(self.observations)} source(s)",
        ]
        for src, obs in self.observations.items():
            if obs.is_tensor_dim():
                dims = ", ".join(
                    f"dim {d}: {sorted(vs)}" for d, vs in obs.dim_values.items()
                )
                lines.append(f"    {src}: {dims}")
            elif obs.is_scalar():
                lines.append(f"    {src}: scalar values {sorted(obs.scalar_values)}")
        return "\n".join(lines)


def _parse_reason_into(reason: str, observations: dict[str, SourceObservation]) -> None:
    """Walk one guard-failure reason string and update ``observations``.

    A reason may contain multiple ``;``-joined sub-reasons (see
    ``get_guard_fail_reason_helper`` in ``torch/_dynamo/guards.py``).
    """
    for part in reason.split(";"):
        m = _TENSOR_SIZE_MISMATCH.search(part)
        if m:
            src = m.group("src")
            dim = int(m.group("dim"))
            obs = observations.setdefault(src, SourceObservation(src))
            obs.dim_values.setdefault(dim, set()).update(
                {int(m.group("expected")), int(m.group("actual"))}
            )
            continue
        m = _LOCAL_EQ_INT.search(part)
        if m:
            src = m.group("src")
            obs = observations.setdefault(src, SourceObservation(src))
            obs.scalar_values.add(int(m.group("value")))
            continue
        # Sympy guards like "6 <= x.size()[0]" don't give us a value pair,
        # but they tell us the dim was guarded — register it with an empty
        # value set so the suggester sees the source/dim pair.
        for sym in _SYMPY_SIZE_REF.finditer(part):
            src = sym.group("src")
            dim = int(sym.group("dim"))
            obs = observations.setdefault(src, SourceObservation(src))
            obs.dim_values.setdefault(dim, set())


def _build_spec(
    observations: dict[str, SourceObservation],
    threshold: int,
    arg_ranks: dict[str, int],
) -> ShapesSpec:
    """Combine per-source observations into a single ``ShapesSpec``.

    ``arg_ranks`` lets the caller pad ``TensorSpec`` entries with trailing
    ``None`` to match the actual tensor rank. Without it the spec would be
    rejected at compile time (rank mismatch).
    """
    arg_to_tensor_dims: dict[str, dict[int, ShapeVar]] = {}
    arg_to_scalar: dict[str, IntVar] = {}

    for src, obs in observations.items():
        argname = _extract_argname(src)
        if argname is None:
            continue
        if obs.is_tensor_dim():
            for dim, values in obs.dim_values.items():
                # A guarded dim with no extractable values (e.g. caught only
                # by a sympy reference) still means "Dynamo guarded on this
                # dim and saw it change" — promote it. The threshold only
                # filters out dims with a known small set of values.
                if values and len(values) < threshold:
                    continue
                arg_to_tensor_dims.setdefault(argname, {})[dim] = ShapeVar(
                    f"{argname}_d{dim}"
                )
        elif obs.is_scalar():
            if len(obs.scalar_values) >= threshold:
                arg_to_scalar[argname] = IntVar(argname)

    params: dict[str, Any] = {}
    for argname, dim_map in arg_to_tensor_dims.items():
        rank = max(arg_ranks.get(argname, 0), max(dim_map) + 1)
        dims: list[ShapeVar | None] = [None] * rank
        for d, sv in dim_map.items():
            dims[d] = sv
        params[argname] = TensorSpec(dims)
    params.update(arg_to_scalar)

    return ShapesSpec(params=ParamsSpec(params))


def _extract_argname(source: str) -> str | None:
    """Convert an observed source string back into an argname for the spec.

    Accepts ``L['x']`` (legacy) and bare ``x`` (current guard-failure format).
    """
    m = re.fullmatch(r"L\[['\"]([^'\"]+)['\"]\]", source)
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Za-z_]\w*", source):
        return source
    return None


def _read_metrics_deque() -> list[Any]:
    # Imported lazily to avoid a hard import cycle (this module is in
    # torch.fx, dynamo lives at torch._dynamo).
    from torch._dynamo.utils import _compilation_metrics

    return list(_compilation_metrics)


def _read_guard_failures_for(co_name: str | None) -> list[str]:
    """Pull every guard-fail reason string captured during the workload.

    ``guard_failures`` is keyed by the original user code object (via
    ``orig_code_map``). We can't easily snapshot it before the window since
    the code object isn't known yet, so callers should either filter by
    ``co_name`` or snapshot ``len()`` before the workload — the observer
    does the latter.
    """
    from torch._dynamo.utils import guard_failures

    out: list[str] = []
    for code, reasons in guard_failures.items():
        if co_name is not None and getattr(code, "co_name", None) != co_name:
            continue
        for r in reasons:
            if isinstance(r, str):
                out.append(r)
    return out


@contextmanager
def dynamic_spec_observer(
    co_name: str | None = None,
    *,
    threshold: int | None = None,
) -> Iterator[_ObserverHandle]:
    """Capture recompile reasons and compile timings produced inside this block.

    ``co_name`` filters captured entries to a specific function; pass ``None``
    to capture everything emitted during the window (rare; useful when the
    caller drives a single workload).

    Example::

        with dynamic_spec_observer("forward") as obs:
            for x in inputs:
                compiled_fn(x)
        print(obs.snapshot())
        print(obs.suggest_spec())
    """
    before = _read_metrics_deque()
    before_len = len(before)
    # Snapshot per-code reason-list lengths so we can compute a delta even if
    # the workload runs after earlier compiles populated guard_failures.
    from torch._dynamo.utils import guard_failures as _gf

    gf_before = {code: len(reasons) for code, reasons in _gf.items()}
    handle = _ObserverHandle(
        co_name=co_name,
        threshold=threshold,
        _start_idx=before_len,
        _gf_before=gf_before,
    )
    t0 = time.perf_counter()
    try:
        yield handle
    finally:
        handle._wall_time_s = time.perf_counter() - t0
        handle._finalize()


@dataclass
class _ObserverHandle:
    co_name: str | None
    threshold: int | None
    _start_idx: int
    _gf_before: dict[Any, int] = field(default_factory=dict)
    _snapshot: CompileSnapshot | None = None
    _wall_time_s: float = 0.0

    def _finalize(self) -> None:
        all_entries = _read_metrics_deque()
        # The deque is bounded; if it rolled over while the workload ran we
        # silently skip overwritten entries — those compiles still happened
        # but the deque can't show them.
        new_entries = all_entries[self._start_idx :]
        # co_name filtering is only useful when multiple compiled functions
        # are exercised in the same window. When the caller passes co_name
        # but no entry matches, keep the entries unfiltered and let the
        # caller see the mismatch via snapshot.observed_co_names.
        observed_co_names = [getattr(e, "co_name", None) for e in new_entries]
        if self.co_name is not None:
            filtered = [
                e for e in new_entries if getattr(e, "co_name", None) == self.co_name
            ]
            if filtered:
                new_entries = filtered

        observations: dict[str, SourceObservation] = {}
        reasons: list[str] = []
        compile_times: list[float] = []
        for entry in new_entries:
            reason = getattr(entry, "recompile_reason", None)
            if reason:
                reasons.append(reason)
                _parse_reason_into(reason, observations)
            t = getattr(entry, "entire_frame_compile_time_s", None)
            if t is not None:
                compile_times.append(t)

        # Supplement with guard_failures (delta from before the window).
        # This captures every guard-fail string Dynamo composed during the
        # workload, including ones the CompilationMetrics path skipped.
        from torch._dynamo.utils import guard_failures as _gf

        for code, reasons_list in _gf.items():
            if (
                self.co_name is not None
                and getattr(code, "co_name", None) != self.co_name
            ):
                continue
            start = self._gf_before.get(code, 0)
            for r in reasons_list[start:]:
                if isinstance(r, str):
                    reasons.append(r)
                    _parse_reason_into(r, observations)

        n_compiles = len(new_entries)
        cold = compile_times[0] if compile_times else 0.0
        warm = (
            sum(compile_times[1:]) / max(1, len(compile_times) - 1)
            if len(compile_times) > 1
            else 0.0
        )
        self._snapshot = CompileSnapshot(
            co_name=self.co_name,
            n_compiles=n_compiles,
            cold_time_s=cold,
            warm_time_s=warm,
            total_compile_time_s=sum(compile_times),
            observations=observations,
            reasons=reasons,
            observed_co_names=observed_co_names,
        )

    def snapshot(self) -> CompileSnapshot:
        if self._snapshot is None:
            raise RuntimeError(
                "snapshot() called before observer's context manager exited"
            )
        return self._snapshot

    def suggest_spec(
        self,
        arg_ranks: dict[str, int] | None = None,
        arg_shapes: dict[str, dict[int, set[int]]] | None = None,
    ) -> ShapesSpec:
        """Build a ``ShapesSpec`` from the captured observations.

        ``arg_ranks`` maps argname -> tensor rank. The resulting
        ``TensorSpec`` for each tensor arg is padded with trailing ``None``s
        to match its rank, so applying the spec doesn't raise a rank
        mismatch.

        ``arg_shapes`` provides a per-arg, per-dim set of sizes observed
        across calls (typically computed by the caller iterating inputs).
        It supplements guard-failure parsing: ``guard_failures`` only
        surfaces dims that triggered a guard mismatch in isolation, so dims
        that auto-promoted alongside other dims (no isolated guard fail) get
        missed. Direct shape tracking has full visibility.

        Returns an empty ``ShapesSpec`` if nothing in the window crossed the
        promotion threshold.
        """
        threshold = self.threshold
        if threshold is None:
            # Pull the live config value so users tuning at runtime see it.
            from torch._dynamo import config as _dynamo_config

            threshold = _dynamo_config.dynamic_spec_suggest_threshold

        observations = dict(self.snapshot().observations)
        if arg_shapes:
            for argname, dim_to_sizes in arg_shapes.items():
                obs = observations.setdefault(argname, SourceObservation(argname))
                for dim, sizes in dim_to_sizes.items():
                    obs.dim_values.setdefault(dim, set()).update(sizes)
        return _build_spec(observations, threshold, arg_ranks or {})


def suggest_spec_for(
    co_name: str,
    *,
    threshold: int | None = None,
    arg_ranks: dict[str, int] | None = None,
) -> ShapesSpec:
    """Scan the full ``CompilationMetrics`` deque for ``co_name`` and emit a spec.

    Use this when ``torch._dynamo.config.dynamic_spec_suggest`` is on and the
    workload has run normally — no surrounding context manager.
    """
    observations: dict[str, SourceObservation] = {}
    for entry in _read_metrics_deque():
        if getattr(entry, "co_name", None) != co_name:
            continue
        reason = getattr(entry, "recompile_reason", None)
        if reason:
            _parse_reason_into(reason, observations)
    if threshold is None:
        from torch._dynamo import config as _dynamo_config

        threshold = _dynamo_config.dynamic_spec_suggest_threshold
    return _build_spec(observations, threshold, arg_ranks or {})
