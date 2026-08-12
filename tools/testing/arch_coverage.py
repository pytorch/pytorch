#!/usr/bin/env python3
"""Work out which GPU each arch-gated test needs, by reading its skip decorator.

Reports the tests that require an H100 or B200, grouped into the .ci/pytorch/test.sh
function that runs there, plus an inventory of gates it could not classify.

Only two decorator shapes are treated as a requirement:

    @skipIf(not PRED, ...)      @skipUnless(PRED, ...)      @skipCUDAIf(not PRED, ...)

Anything else -- a compound condition, or a bare `skipIf(PRED)` which excludes
rather than requires -- is declined and listed, never guessed at. Guessing is how
an earlier version decided test__int_mm "requires" sm90 when its gate is
`skipIf(SM90OrLater, "Expected failure on sm90")`.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path
from typing import NamedTuple


REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = REPO_ROOT / "test"
COMMON_CUDA = REPO_ROOT / "torch/testing/_internal/common_cuda.py"

SKIP_DECORATORS = {"skipIf", "skipUnless", "skipCUDAIf"}

# Capability of the highest runner reached without a curated list. Anything a test
# can run on at or below this already runs in CI, so it needs no entry.
AUTO_DISCOVERED_MAX = (8, 9)

# (label, runner capability, test.sh function). The labels are independent, so a
# test that runs on both belongs in both lists.
TARGETS = [
    ("ciflow/h100", (9, 0), "test_python_smoke"),
    ("ciflow/b200", (10, 0), "test_python_smoke_b200"),
]


class Gate(NamedTuple):
    file: str  # run_test.py style, e.g. "inductor/test_user_streams"
    test: str
    line: int
    predicate: str


class Unclassified(NamedTuple):
    file: str
    line: int
    reason: str
    source: str


def load_tables() -> tuple[dict, set[str]]:
    """(GATE_ARCH_RANGES, NOT_ARCH_GATED), read as source so this needs no torch."""
    found: dict[str, object] = {}
    for node in ast.parse(COMMON_CUDA.read_text(encoding="utf-8")).body:
        target = node.target if isinstance(node, ast.AnnAssign) else None
        name = getattr(target, "id", "")
        if name in ("GATE_ARCH_RANGES", "NOT_ARCH_GATED"):
            found[name] = ast.literal_eval(node.value)
    missing = {"GATE_ARCH_RANGES", "NOT_ARCH_GATED"} - set(found)
    if missing:
        raise SystemExit(f"arch_coverage: {missing} not found in {COMMON_CUDA}")
    return found["GATE_ARCH_RANGES"], set(found["NOT_ARCH_GATED"])


def requires(node: ast.Call) -> ast.expr | None:
    """The predicate a decorator requires, or None if it isn't a requirement."""
    name = ast.unparse(node.func).split(".")[-1]
    if name not in SKIP_DECORATORS or not node.args:
        return None
    cond = node.args[0]
    negated = isinstance(cond, ast.UnaryOp) and isinstance(cond.op, ast.Not)
    inner = cond.operand if negated else cond
    if isinstance(inner, ast.BoolOp):
        return None
    if negated or name == "skipUnless":
        return inner
    return None


def predicate_name(node: ast.expr) -> str:
    """`SM90OrLater` / `has_triton_tma_device()` -> the bare name."""
    if isinstance(node, ast.Call):
        node = node.func
    return ast.unparse(node).split(".")[-1]


def scan(
    path: Path, rel: str, ranges: dict, exempt: set[str]
) -> tuple[list[Gate], list[Unclassified]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return [], []
    gates: list[Gate] = []
    unclassified: list[Unclassified] = []

    def visit(node: ast.AST, cls: str | None) -> None:
        for child in ast.iter_child_nodes(node):
            if not isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                continue
            is_class = isinstance(child, ast.ClassDef)
            named = child.name.startswith("Test" if is_class else "test")
            for dec in child.decorator_list:
                if not isinstance(dec, ast.Call):
                    continue
                src = ast.unparse(dec)
                if not ARCH_HINT.search(src):
                    continue
                pred = requires(dec)
                if pred is None:
                    if named:
                        unclassified.append(
                            Unclassified(
                                rel, dec.lineno, "not a requirement", src[:100]
                            )
                        )
                    continue
                name = predicate_name(pred)
                if name in exempt:
                    continue
                if name not in ranges:
                    if named:
                        unclassified.append(
                            Unclassified(
                                rel,
                                dec.lineno,
                                f"{name} has no declared range",
                                src[:100],
                            )
                        )
                    continue
                if named:
                    gates.append(Gate(rel, child.name, child.lineno, name))
            if is_class:
                visit(child, child.name)

    visit(tree, None)
    return gates, unclassified


def rel_name(path: Path) -> str | None:
    if not path.name.startswith("test_") or path.suffix != ".py":
        return None
    return str(path.relative_to(TEST_ROOT).with_suffix(""))


def collect(ranges: dict, exempt: set[str]) -> tuple[list[Gate], list[Unclassified]]:
    gates: list[Gate] = []
    unclassified: list[Unclassified] = []
    for root, dirs, files in os.walk(TEST_ROOT):
        # Vendored CPython tests and fbcode-only trees are not ours to schedule.
        dirs[:] = [d for d in dirs if d not in ("fb", "cpython")]
        for name in sorted(files):
            path = Path(root) / name
            rel = rel_name(path)
            if rel:
                g, u = scan(path, rel, ranges, exempt)
                gates.extend(g)
                unclassified.extend(u)
    return gates, unclassified


def needed(gates: list[Gate], ranges: dict) -> dict[str, dict[str, set[str]]]:
    """{test.sh function: {file: test names}} for tests a curated list must carry."""
    out: dict[str, dict[str, set[str]]] = {fn: {} for _, _, fn in TARGETS}
    for gate in gates:
        lo, hi = ranges[gate.predicate]
        if lo <= AUTO_DISCOVERED_MAX and (hi is None or AUTO_DISCOVERED_MAX < hi):
            continue
        for _, cap, fn in TARGETS:
            if cap >= lo and (hi is None or cap < hi):
                out[fn].setdefault(gate.file, set()).add(gate.test)
    return out


ARCH_HINT = re.compile(
    r"\b(SM\d+OrLater|IS_SM\w+|has_\w*tma\w*|has_datacenter\w*|PLATFORM_SUPPORTS_\w+)\b"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument(
        "--unclassified", action="store_true", help="list gates that were declined"
    )
    args = parser.parse_args()

    ranges, exempt = load_tables()
    gates, unclassified = collect(ranges, exempt)
    plan = needed(gates, ranges)

    if args.json:
        json.dump(
            {
                "needed": {
                    fn: {f: sorted(t) for f, t in files.items()}
                    for fn, files in plan.items()
                },
                "unclassified": [u._asdict() for u in unclassified],
            },
            sys.stdout,
            indent=2,
        )
        print()
        return 0

    if args.unclassified:
        print(f"{len(unclassified)} arch gate(s) could not be classified:\n")
        for u in unclassified:
            print(f"  test/{u.file}.py:{u.line}  {u.reason}")
            print(f"      {u.source}")
        return 0

    for _, _, fn in TARGETS:
        files = plan[fn]
        total = sum(len(t) for t in files.values())
        print(f"{fn}(): {len(files)} file(s), {total} test(s)")
        for f in sorted(files):
            print(f'  --include {f} -k "{" or ".join(sorted(files[f]))}"')
        print()
    print(f"{len(unclassified)} gate(s) unclassified (see --unclassified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
