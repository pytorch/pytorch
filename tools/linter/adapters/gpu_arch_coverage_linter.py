#!/usr/bin/env python3
"""Keep the GPU smoke lists in sync with the arch gates in test files.

A test gated on a GPU architecture (`SM90OrLater`, `has_triton_tma_device()`, ...)
only runs if `.ci/pytorch/test.sh` selects it in the function the ciflow/h100 or
ciflow/b200 job dispatches to. Nothing connects that list to the
`@skipUnless(SM90OrLater)` that created the need, so gated tests silently skip on
the a10g/A100 runners that serve most of CI.

This derives the list from the gates, so adding a gated test also adds it to the
smoke run. Triggering stays manual: applying the ciflow/h100 label is a
deliberate, human decision, because H100/B200 capacity is scarce. This linter
only guarantees that when the job *is* run, it covers the gated tests.

Wiring these files into .github/labeler.yml would make the trigger automatic
too, but that would fire the H100 job on every PR touching an arch-gated test
file (~9 commits/day). Deliberately not done.

Only *named* predicates are routed. Hand-rolled `get_device_capability()`
comparisons get advice to switch to a named predicate rather than a fix, because
their direction is not reliably recoverable -- see test/inductor/test_pad_mm.py,
which skips *on* sm90+.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from enum import Enum
from typing import NamedTuple


LINTER_CODE = "GPU_ARCH_COVERAGE"

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TEST_ROOT = os.path.join(REPO_ROOT, "test")
LABELER = os.path.join(REPO_ROOT, ".github", "labeler.yml")
TEST_SH = os.path.join(REPO_ROOT, ".ci", "pytorch", "test.sh")
BASELINE = os.path.join(os.path.dirname(__file__), "gpu_arch_coverage-baseline.json")

# (min capability inclusive, max capability exclusive or None).
#
# The has_* entries are compound -- they also require a Triton feature -- but the
# capability bound is a necessary condition, which is all that is needed to decide
# whether a runner can possibly satisfy the gate.
ARCH_PREDICATES: dict[str, tuple[tuple[int, int], tuple[int, int] | None]] = {
    "SM89OrLater": ((8, 9), None),
    "SM90OrLater": ((9, 0), None),
    "SM100OrLater": ((10, 0), None),
    "SM120OrLater": ((12, 0), None),
    "IS_SM89": ((8, 9), (9, 0)),
    "IS_SM90": ((9, 0), (9, 1)),
    "IS_SM100": ((10, 0), (10, 1)),
    "IS_SM103": ((10, 3), (10, 4)),
    "IS_SM10X": ((10, 0), (11, 0)),
    "IS_SM12X": ((12, 0), (13, 0)),
    "has_triton_tma_device": ((9, 0), None),
    "has_triton_tensor_descriptor_host_tma": ((9, 0), None),
    "has_triton_experimental_host_tma": ((9, 0), None),
    "has_triton_stable_tma_api": ((9, 0), None),
    "has_datacenter_blackwell_tma_device": ((10, 0), (11, 0)),
}

# Highest capability each ciflow label's runner provides, cheapest first.
CIFLOW_RUNNERS = [
    ("ciflow/h100", (9, 0)),
    ("ciflow/b200", (10, 0)),
]

# test.sh functions reachable from an H100/B200 runner, and the label that gets
# there. New entries are appended to the first function listed for each label;
# the others only count towards existing coverage.
SMOKE_FUNCTIONS = {
    "test_python_smoke": "ciflow/h100",
    "test_h100_cutlass_backend": "ciflow/h100",
    "test_h100_distributed": "ciflow/h100",
    "test_python_smoke_b200": "ciflow/b200",
    # Called by both test_h100_symm_mem and test_b200_symm_mem.
    "_run_symm_mem_tests": None,
}
INSERT_INTO = {
    "ciflow/h100": "test_python_smoke",
    "ciflow/b200": "test_python_smoke_b200",
}

# Past this many gated symbols a -k expression stops being useful; run the file.
MAX_K_TERMS = 12

# a10g (8.6) serves most of CI; anything it satisfies needs no GPU-specific label.
BASELINE_CAPABILITY = (8, 6)

OPT_OUT = re.compile(r"#\s*gpu-coverage:\s*skip\b")
HAND_ROLLED = re.compile(r"get_device_capability|get_device_properties")
# @patch("torch.cuda.get_device_capability", ...) mocks a gate, it is not one.
MOCKING = re.compile(r"\b(patch|mock)\b")


class LintSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"


class LintMessage(NamedTuple):
    path: str | None
    line: int | None
    char: int | None
    code: str
    severity: LintSeverity
    name: str
    original: str | None
    replacement: str | None
    description: str | None


class Gate(NamedTuple):
    file: str  # run_test.py-style, e.g. "inductor/test_user_streams"
    symbol: str
    line: int
    lo: tuple[int, int]
    hi: tuple[int, int] | None


def label_for(lo: tuple[int, int], hi: tuple[int, int] | None) -> str | None:
    """Cheapest ciflow label whose runner capability falls inside [lo, hi)."""
    if lo <= BASELINE_CAPABILITY and hi is None:
        return None
    for label, cap in CIFLOW_RUNNERS:
        if cap >= lo and (hi is None or cap < hi):
            return label
    return None


def scan_file(path: str, rel: str) -> tuple[list[Gate], list[tuple[str, int]]]:
    """Returns (routable gates, hand-rolled gates needing advice)."""
    try:
        with open(path, encoding="utf-8") as f:
            src = f.read()
    except (OSError, UnicodeDecodeError):
        return [], []
    if not (any(p in src for p in ARCH_PREDICATES) or HAND_ROLLED.search(src)):
        return [], []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return [], []
    lines = src.splitlines()

    gates: list[Gate] = []
    handrolled: list[tuple[str, int]] = []
    gated_classes: set[str] = set()

    def visit(node: ast.AST, enclosing: str | None) -> None:
        for child in ast.iter_child_nodes(node):
            if not isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                continue
            is_class = isinstance(child, ast.ClassDef)
            named = child.name.startswith("Test" if is_class else "test")
            lo: tuple[int, int] | None = None
            hi: tuple[int, int] | None = None
            opted_out = False
            for dec in child.decorator_list:
                text = "\n".join(lines[dec.lineno - 1 : dec.end_lineno])
                if OPT_OUT.search(text):
                    opted_out = True
                    break
                matched = False
                for name, (plo, phi) in ARCH_PREDICATES.items():
                    if not re.search(rf"\b{name}\b", text):
                        continue
                    matched = True
                    lo = plo if lo is None else max(lo, plo)
                    if phi is not None:
                        hi = phi if hi is None else min(hi, phi)
                if (
                    not matched
                    and named
                    and HAND_ROLLED.search(text)
                    and not MOCKING.search(text)
                ):
                    handrolled.append((child.name, dec.lineno))
            # A gated class covers its methods; do not emit both.
            if (
                named
                and not opted_out
                and lo is not None
                and enclosing not in gated_classes
            ):
                gates.append(Gate(rel, child.name, child.lineno, lo, hi))
                if is_class:
                    gated_classes.add(child.name)
            if is_class:
                visit(child, child.name)

    visit(tree, None)
    return gates, handrolled


def rel_test_name(path: str) -> str | None:
    """Absolute path -> run_test.py-style name, or None if not a test file."""
    path = os.path.abspath(path)
    if not path.startswith(TEST_ROOT + os.sep) or not path.endswith(".py"):
        return None
    base = os.path.basename(path)
    if not base.startswith("test_"):
        return None
    return os.path.relpath(path, TEST_ROOT)[: -len(".py")]


def all_test_files() -> list[tuple[str, str]]:
    out = []
    for root, dirs, files in os.walk(TEST_ROOT):
        dirs[:] = [d for d in dirs if d not in ("fb", "cpython")]
        for name in sorted(files):
            path = os.path.join(root, name)
            rel = rel_test_name(path)
            if rel:
                out.append((path, rel))
    return out


def parse_smoke_includes(text: str) -> dict[str, dict[str, list[str | None]]]:
    """{ciflow label -> {run_test.py file -> [-k expr, ...]}}.

    A file can be included by several functions reachable from the same label
    (e.g. test_python_smoke and test_h100_distributed both run
    test_fully_shard_comm with different -k), so coverage is the union of them.
    A None in the list means some include runs the whole file.
    """
    out: dict[str, dict[str, list[str | None]]] = {l: {} for l, _ in CIFLOW_RUNNERS}
    for fn, label in SMOKE_FUNCTIONS.items():
        body = re.search(rf"^{fn}\(\) \{{\n(.*?)^\}}", text, re.DOTALL | re.MULTILINE)
        if not body:
            raise SystemExit(f"{LINTER_CODE}: could not locate {fn}() in test.sh")
        targets = [label] if label else [l for l, _ in CIFLOW_RUNNERS]
        for line in re.sub(r"\\\n\s*", " ", body.group(1)).splitlines():
            m = re.search(r"--include\s+(.+?)(?:\s+\$|\s*$)", line)
            if not m:
                continue
            rest, kexpr = m.group(1), None
            if " -k " in rest:
                rest, _, kexpr = rest.partition(" -k ")
                kexpr = kexpr.strip().strip('"')
            for f in rest.split():
                f = f.removesuffix(".py")
                for t in targets:
                    out[t].setdefault(f, []).append(kexpr)
    return out


def parse_labeler(text: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    current = None
    for line in text.splitlines():
        m = re.match(r'^"?(ciflow/[\w.-]+)"?:\s*$', line)
        if m:
            current = m.group(1)
            out[current] = []
        elif current and line.startswith("- "):
            out[current].append(line[2:].strip())
        elif line.strip() and not line.startswith(("-", "#", " ")):
            current = None
    return out


def k_expression(symbols: list[Gate]) -> str | None:
    """None means run the whole file -- see MAX_K_TERMS."""
    # Substring matching means a base name also selects @parametrize variants,
    # and a gated class subsumes its methods, so both collapse for free.
    names = sorted({g.symbol for g in symbols})
    return None if len(names) > MAX_K_TERMS else " or ".join(names)


def compute_gaps(
    gates: list[Gate], smoke: dict
) -> list[tuple[str, str, str | None, Gate]]:
    """-> [(label, file, kexpr, first_gate)] for gates the smoke list misses"""
    by_label: dict[str, dict[str, list[Gate]]] = {}
    for g in gates:
        label = label_for(g.lo, g.hi)
        if label:
            by_label.setdefault(label, {}).setdefault(g.file, []).append(g)

    gaps = []
    for label, files in sorted(by_label.items()):
        for test_file, syms in sorted(files.items()):
            exprs = smoke[label].get(test_file, [])
            # Covered if every gated symbol is selected by at least one include.
            in_sh = bool(exprs) and all(
                any(
                    e is None or any(p.strip() in g.symbol for p in e.split(" or "))
                    for e in exprs
                )
                for g in syms
            )
            if in_sh:
                continue
            gaps.append(
                (label, test_file, k_expression(syms), min(syms, key=lambda g: g.line))
            )
    return gaps


def find_stale(smoke: dict) -> list[tuple[str, str]]:
    """Smoke-list includes pointing at files that no longer exist.

    -> [(label, listed path)]
    """
    stale = []
    for label, _ in CIFLOW_RUNNERS:
        for rel in smoke.get(label, {}):
            if not os.path.exists(os.path.join(TEST_ROOT, rel + ".py")):
                stale.append((label, f"test/{rel}.py"))
    return stale


def apply_test_sh(text: str, additions: dict[str, list[tuple[str, str | None]]]) -> str:
    for label, entries in additions.items():
        fn = INSERT_INTO[label]
        m = re.search(rf"^({fn}\(\) \{{\n)(.*?)(^\}})", text, re.DOTALL | re.MULTILINE)
        if not m:
            continue
        body = m.group(2)
        new_lines = ""
        for test_file, kexpr in entries:
            sel = f' -k "{kexpr}"' if kexpr else ""
            new_lines += (
                f"  time python test/run_test.py --include {test_file}{sel} "
                f"$PYTHON_TEST_EXTRA_OPTION --upload-artifacts-while-running\n"
            )
        # Keep assert_git_not_dirty last if the function has one.
        anchor = "  assert_git_not_dirty\n"
        if anchor in body:
            body = body.replace(anchor, new_lines + anchor, 1)
        else:
            body = body + new_lines
        text = text[: m.start(2)] + body + text[m.end(2) :]
    return text


def load_baseline() -> set[str]:
    if not os.path.exists(BASELINE):
        return set()
    with open(BASELINE, encoding="utf-8") as f:
        return set(json.load(f).get("known_uncovered", []))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check GPU arch test coverage.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--show-backfill", action="store_true")
    parser.add_argument("--show-handrolled", action="store_true")
    parser.add_argument("filenames", nargs="*")
    args = parser.parse_args()

    gates: list[Gate] = []
    handrolled: dict[str, list[tuple[str, int]]] = {}
    for path, rel in all_test_files():
        g, h = scan_file(path, rel)
        gates.extend(g)
        if h:
            handrolled[rel] = h

    with open(TEST_SH, encoding="utf-8") as f:
        test_sh_text = f.read()

    smoke = parse_smoke_includes(test_sh_text)
    gaps = compute_gaps(gates, smoke)

    if args.write_baseline:
        keys = sorted(f"{label}::{f}" for label, f, _, _ in gaps)
        with open(BASELINE, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "_comment": (
                        "Files with arch-gated tests that predate GPU_ARCH_COVERAGE. "
                        "Remove entries as they are added to the smoke lists; do not add new ones."
                    ),
                    "known_uncovered": keys,
                },
                f,
                indent=2,
            )
            f.write("\n")
        print(f"wrote {len(keys)} baseline entries to {BASELINE}")
        return

    if args.show_handrolled:
        for rel, hits in sorted(handrolled.items()):
            for symbol, line in hits:
                print(f"test/{rel}.py:{line}\t{symbol}")
        return

    if args.show_backfill:
        for label, test_file, kexpr, _ in gaps:
            sel = f' -k "{kexpr}"' if kexpr else ""
            print(f"{label}\ttest/{test_file}.py\t{sel}")
        return

    baseline = load_baseline()
    touched = {os.path.abspath(p) for p in args.filenames}
    changed_tests = {rel_test_name(p) for p in args.filenames} - {None}
    # Editing test.sh can invalidate any entry, so widen coverage checking to the
    # whole tree; otherwise only report on files lintrunner handed us.
    sh_changed = TEST_SH in touched
    in_scope = {f for _, f, _, _ in gaps} if sh_changed else changed_tests

    messages: list[LintMessage] = []
    add_sh: dict[str, list[tuple[str, str | None]]] = {}

    for label, test_file, kexpr, first in gaps:
        if test_file not in in_scope or f"{label}::{test_file}" in baseline:
            continue
        add_sh.setdefault(label, []).append((test_file, kexpr))
        messages.append(
            LintMessage(
                path=os.path.join(TEST_ROOT, test_file + ".py"),
                line=first.line,
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.ERROR,
                name="test not run on required GPU",
                original=None,
                replacement=None,
                description=(
                    f"Tests here need compute capability >= {first.lo[0]}.{first.lo[1]}, "
                    f"so they skip on the a10g/A100 runners that serve most of CI, and "
                    f"{INSERT_INTO[label]}() in .ci/pytorch/test.sh does not select them. "
                    f"Run `lintrunner -a` to add them, or annotate with "
                    f"`# gpu-coverage: skip <reason>`. Note this only makes the {label} "
                    f"job cover the test; applying the {label} label to run it stays manual."
                ),
            )
        )

    if sh_changed:
        for label, entry in find_stale(smoke):
            messages.append(
                LintMessage(
                    path=TEST_SH,
                    line=None,
                    char=None,
                    code=LINTER_CODE,
                    severity=LintSeverity.ERROR,
                    name="stale smoke-list entry",
                    original=None,
                    replacement=None,
                    description=(
                        f"{entry} is included by the {label} smoke run but no longer "
                        f"exists. Remove the entry."
                    ),
                )
            )

    # Hand-rolled `get_device_capability()` gates cannot be routed automatically
    # (see test/inductor/test_pad_mm.py, which skips *on* sm90+). lintrunner treats
    # advice as a failure, so surface them via --show-handrolled rather than here.

    if add_sh:
        new = apply_test_sh(test_sh_text, add_sh)
        if new != test_sh_text:
            messages.append(
                LintMessage(
                    path=TEST_SH,
                    line=None,
                    char=None,
                    code=LINTER_CODE,
                    severity=LintSeverity.ERROR,
                    name="missing smoke-test entries",
                    original=test_sh_text,
                    replacement=new,
                    description="Add arch-gated tests so the GPU job covers them.",
                )
            )

    for m in messages:
        print(json.dumps(m._asdict()), flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
