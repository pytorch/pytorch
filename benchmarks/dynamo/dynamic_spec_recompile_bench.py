"""Driver for ``torch.fx.experimental.dynamic_spec_optimizer``.

Observes ``torch.compile`` recompiles and suggests a ``ShapesSpec``.

For each scenario, runs three modes in fresh subprocesses (so backend caches
don't leak across cells):

- ``pessimistic``  — ``automatic_dynamic_shapes=False``. Recompiles per shape.
- ``default``      — ``automatic_dynamic_shapes=True``. Auto-promotes.
- ``applied``      — ``shapes_spec`` from the default run's suggester.

The advisor's value is the gap between ``default`` and ``applied``: cases
where pinpointing dynamic dims upfront avoids the chain of one-by-one
auto-discovery recompiles, or skips 0/1 specialization.

Run with::

    python benchmarks/dynamo/dynamic_spec_recompile_bench.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any

import torch


_CHILD_FLAG = "--child"


# ---- scenarios -------------------------------------------------------------


@dataclass
class Scenario:
    name: str
    fn_src: str  # function source, evaluated in child
    inputs_src: str  # expression producing list[tuple] of inputs
    expected_pessimistic_compiles: int


def _scenarios() -> list[Scenario]:
    """Scenarios where a spec can demonstrably beat auto-dynamic.

    Excluded: branching cases — the bot's unbacked ShapeVar can't resolve a
    branch on size without bounds, so it gives no improvement over auto-dyn.

    Included: cases where (a) auto-dyn discovers each shifting dim one
    compile at a time, and (b) 0/1 specialization keeps biting under
    auto-dyn's backed dims but is skipped under the spec's unbacked dims.
    """
    return [
        Scenario(
            # Auto-dyn promotes dim 0 on call 2, then dim 1 on call 3 — a
            # chain of recompiles. Spec marks both dims dynamic from start.
            name="multi_dim_shifting",
            fn_src=("def fn(x):\n    return x * 2 + 1\n"),
            inputs_src=(
                "[(torch.randn(s),) for s in "
                "[(4, 3), (8, 3), (8, 5), (16, 7), (32, 11)]]"
            ),
            expected_pessimistic_compiles=5,
        ),
        Scenario(
            # 0 and 1 appear in dim 0. Auto-dyn promotes to backed dynamic
            # but still 0/1-specializes. Spec's unbacked ShapeVar skips it.
            name="zero_one_specialization",
            fn_src=("def fn(x):\n    return x.sum(0)\n"),
            inputs_src="[(torch.randn(n, 3),) for n in [4, 8, 0, 1, 16]]",
            expected_pessimistic_compiles=5,
        ),
        Scenario(
            # End-to-end: tiny transformer encoder block. Batch and seq both
            # vary across calls — auto-dyn discovers each independently.
            name="tiny_transformer",
            fn_src=(
                "import torch.nn as nn\n"
                "torch.manual_seed(0)\n"
                "_model = nn.Sequential(\n"
                "    nn.Linear(64, 64),\n"
                "    nn.GELU(),\n"
                "    nn.Linear(64, 64),\n"
                ").eval()\n"
                "def fn(x):  # x: (batch, seq, 64)\n"
                "    return _model(x).mean(dim=1)\n"
            ),
            inputs_src=(
                "[(torch.randn(*s, 64),) for s in "
                "[(1, 32), (4, 32), (4, 64), (8, 64), (8, 128)]]"
            ),
            expected_pessimistic_compiles=5,
        ),
        Scenario(
            # End-to-end: variable-resolution image classifier. Batch + H + W
            # all vary. Auto-dyn discovers each dim one compile at a time.
            name="variable_resolution_cnn",
            fn_src=(
                "import torch.nn as nn\n"
                "import torch.nn.functional as F\n"
                "torch.manual_seed(0)\n"
                "_model = nn.Sequential(\n"
                "    nn.Conv2d(3, 8, kernel_size=3, padding=1),\n"
                "    nn.ReLU(),\n"
                "    nn.Conv2d(8, 8, kernel_size=3, padding=1),\n"
                ").eval()\n"
                "def fn(x):  # x: (batch, 3, H, W)\n"
                "    return F.adaptive_avg_pool2d(_model(x), 1).flatten(1)\n"
            ),
            inputs_src=(
                "[(torch.randn(b, 3, h, w),) for b, h, w in "
                "[(1, 16, 16), (2, 16, 16), (2, 24, 32), (4, 32, 32), (4, 48, 64)]]"
            ),
            expected_pessimistic_compiles=5,
        ),
    ]


# ---- spec JSON round-trip --------------------------------------------------


def _spec_from_jsonable(data: dict[str, Any]) -> Any:
    """Rebuild a ``ShapesSpec`` from ``to_jsonable()`` output."""
    from torch.fx.experimental.dynamic_spec import (
        IntVar,
        ParamsSpec,
        ShapesSpec,
        ShapeVar,
        TensorSpec,
    )

    def _leaf(entry: Any) -> Any:
        if entry is None or isinstance(entry, (int, float, str)):
            return entry
        kind = entry["type"]
        if kind == "ShapeVar":
            return ShapeVar(entry["name"], max=entry.get("max"))
        if kind == "IntVar":
            return IntVar(entry["name"])
        if kind == "TensorSpec":
            return TensorSpec([_leaf(d) for d in entry["dims"]])
        raise ValueError(f"Unknown spec node: {kind}")

    params = ParamsSpec({k: _leaf(v) for k, v in data["params"]["named_args"].items()})
    return ShapesSpec(params=params)


# ---- child execution -------------------------------------------------------


def _child(scenario_name: str, mode: str, spec_json: str | None) -> None:
    from torch.fx.experimental.dynamic_spec_optimizer import dynamic_spec_observer

    scenario = next(s for s in _scenarios() if s.name == scenario_name)

    # Evaluate fn source and inputs.
    ns: dict[str, Any] = {"torch": torch}
    exec(scenario.fn_src, ns)
    fn = ns["fn"]
    inputs = eval(scenario.inputs_src, {"torch": torch})

    auto_dyn = mode != "pessimistic"
    kwargs: dict[str, Any] = {"backend": "eager"}
    if spec_json:
        kwargs["shapes_spec"] = _spec_from_jsonable(json.loads(spec_json))

    compiled = torch.compile(fn, **kwargs)

    # Capture tensor arg ranks and per-dim shape history across calls. The
    # suggester needs ranks to pad TensorSpec correctly, and shape history
    # to catch dims that auto-dyn promoted silently (no isolated guard fail).
    import inspect
    from collections import defaultdict

    sig = inspect.signature(fn)
    param_names = list(sig.parameters)
    arg_ranks: dict[str, int] = {}
    arg_shapes: dict[str, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    for i, arg in enumerate(inputs[0]):
        if i < len(param_names) and isinstance(arg, torch.Tensor):
            arg_ranks[param_names[i]] = arg.ndim
    for args in inputs:
        for i, arg in enumerate(args):
            if i < len(param_names) and isinstance(arg, torch.Tensor):
                for dim, sz in enumerate(arg.shape):
                    arg_shapes[param_names[i]][dim].add(int(sz))

    with torch._dynamo.config.patch(automatic_dynamic_shapes=auto_dyn):
        with dynamic_spec_observer(co_name=None) as obs:
            t0 = time.perf_counter()
            for args in inputs:
                compiled(*args)
            wall_s = time.perf_counter() - t0

    snap = obs.snapshot()
    suggested = obs.suggest_spec(
        arg_ranks=arg_ranks,
        arg_shapes={k: dict(v) for k, v in arg_shapes.items()},
    )
    suggested_json = (
        suggested.to_jsonable()
        if suggested.to_jsonable().get("params")
        and suggested.to_jsonable()["params"].get("named_args")
        else None
    )

    out = {
        "scenario": scenario_name,
        "mode": mode,
        "n_compiles": snap.n_compiles,
        "wall_s": wall_s,
        "total_compile_s": snap.total_compile_time_s,
        "reasons": snap.reasons,
        "suggested": suggested_json,
        "observed_co_names": snap.observed_co_names,
    }
    sys.stdout.write(json.dumps(out) + "\n")


# ---- parent orchestration --------------------------------------------------


def _run_child(
    scenario_name: str, mode: str, spec_json: str | None = None
) -> dict[str, Any]:
    argv = [sys.executable, __file__, _CHILD_FLAG, scenario_name, mode]
    if spec_json is not None:
        argv.append(spec_json)
    res = subprocess.run(argv, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        sys.stderr.write(
            f"\n[child failed: {scenario_name}/{mode}]\nSTDOUT:\n{res.stdout}\n"
            f"STDERR:\n{res.stderr}\n"
        )
        raise SystemExit(res.returncode)
    # The child may emit warnings/logging before the JSON line; take the
    # last non-empty line.
    last = [ln for ln in res.stdout.strip().splitlines() if ln.strip()][-1]
    return json.loads(last)


def _row(label: str, data: dict[str, Any]) -> str:
    return (
        f"  {label:<22}n_compiles={data['n_compiles']:<3}"
        f"wall={data['wall_s'] * 1000:>7.1f}ms  "
        f"(reported compile {data['total_compile_s'] * 1000:>7.1f}ms)"
    )


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == _CHILD_FLAG:
        _child(
            scenario_name=sys.argv[2],
            mode=sys.argv[3],
            spec_json=sys.argv[4] if len(sys.argv) > 4 else None,
        )
        return

    for scenario in _scenarios():
        print(f"\n=== Scenario: {scenario.name} ===")

        pessimistic = _run_child(scenario.name, "pessimistic")
        default = _run_child(scenario.name, "default")

        ok = pessimistic["n_compiles"] == scenario.expected_pessimistic_compiles
        check = (
            "OK"
            if ok
            else f"unexpected (expected {scenario.expected_pessimistic_compiles})"
        )
        print(f"  pessimistic n_compiles check: {check}")

        suggested = default["suggested"]
        if suggested is None:
            print("  suggested spec: (none — bot couldn't improve on auto-dynamic)")
        else:
            print(f"  suggested spec: {json.dumps(suggested, indent=2)}")

        if not default["reasons"] and not pessimistic["reasons"]:
            print(
                "  [diagnostic] no reasons captured from either run — "
                "guard_failures was empty"
            )
        elif not default["reasons"]:
            print(
                "  [diagnostic] default-run captured 0 reasons; "
                f"pessimistic captured {len(pessimistic['reasons'])}"
            )
            for i, r in enumerate(pessimistic["reasons"][:6]):
                print(f"    pessimistic.reason[{i}] = {r!r}")
        else:
            print(
                f"  [diagnostic] default captured {len(default['reasons'])} reason(s)"
            )
            for i, r in enumerate(default["reasons"][:6]):
                print(f"    default.reason[{i}] = {r!r}")

        print(_row("pessimistic:", pessimistic))
        print(_row("default (auto-dyn):", default))

        if suggested is not None:
            applied = _run_child(
                scenario.name, "applied", spec_json=json.dumps(suggested)
            )
            print(_row("suggested + applied:", applied))
        else:
            print("  applied: (skipped — no suggestion)")


if __name__ == "__main__":
    main()
