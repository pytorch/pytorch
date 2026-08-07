#!/usr/bin/env python3
"""Check that tests requiring an H100/B200 are selected by the smoke runs.

A test gated on a GPU architecture only runs if `.ci/pytorch/test.sh` selects it
in the function the ciflow/h100 or ciflow/b200 job dispatches to. Nothing links
that list to the gate, so it drifts and the tests silently skip on the a10g/A100
runners that serve most of CI.

Which tests need which capability is *measured*, not inferred from decorator
text: tools/testing/introspection imports each file under a simulated capability
and records what ran. A test that runs at sm90 but not at sm86 needs an H100; one
that merely fails on sm90 does not, and a decorator scan cannot tell those apart.

Running the GPU jobs stays a manual ciflow label. This only decides what those
jobs cover, and only ever edits .ci/pytorch/test.sh.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = REPO_ROOT / "test"
TEST_SH = REPO_ROOT / ".ci" / "pytorch" / "test.sh"

sys.path.insert(0, str(REPO_ROOT))

# The capability a10g provides; most of CI runs here, so anything that already
# runs at this level needs no GPU-specific job.
BASELINE_JOB = "linux-cuda-sm86/default"

# (label, job simulating that label's runner, test.sh function it dispatches to)
TARGETS = [
    ("ciflow/h100", "linux-cuda-sm90/default", "test_python_smoke"),
    ("ciflow/b200", "linux-cuda-sm100/default", "test_python_smoke_b200"),
]

# Functions reachable from each label's job, for deciding existing coverage. Read
# off the test-matrix configs in .github/workflows/test-{h100,b200}.yml. Only the
# `smoke` configs are keyed to the bare ciflow/h100 and ciflow/b200 tags -- the
# cutlass/distributed/symm-mem jobs have their own ciflow labels and do not run
# when those are applied.
COVERING_FUNCTIONS = {
    "ciflow/h100": ["test_python_smoke"],
    "ciflow/b200": ["test_python_smoke_b200"],
}


def relpath(path: str) -> str | None:
    """Absolute or repo-relative path -> path relative to the repo root."""
    p = Path(path).resolve()
    try:
        rel = p.relative_to(REPO_ROOT)
    except ValueError:
        return None
    return str(rel) if p.is_file() else None


def parse_smoke_includes(text: str, function: str) -> dict[str, list[str | None]]:
    """{run_test.py file: [-k expr, ...]} for one test.sh function.

    A None entry means some include runs the whole file.
    """
    body = re.search(rf"^{function}\(\) \{{\n(.*?)^\}}", text, re.DOTALL | re.MULTILINE)
    if not body:
        raise SystemExit(f"gpu_coverage: could not find {function}() in {TEST_SH}")
    out: dict[str, list[str | None]] = {}
    for line in re.sub(r"\\\n\s*", " ", body.group(1)).splitlines():
        if line.lstrip().startswith("#"):
            continue
        m = re.search(r"--include\s+(.+?)(?:\s+\$|\s*$)", line)
        if not m:
            continue
        rest, kexpr = m.group(1), None
        if " -k " in rest:
            rest, _, kexpr = rest.partition(" -k ")
            kexpr = kexpr.strip().strip("\"'")
        for f in rest.split():
            out.setdefault(f.removesuffix(".py"), []).append(kexpr)
    return out


def base_names(test_file: Path, tests: set[str], locations: dict) -> set[str]:
    """Concrete test ids -> the `def` names a -k expression should carry.

    Parametrized variants share their template function's def line, so grouping
    by location collapses them to the one name that -k substring-matches.
    """
    try:
        tree = ast.parse(test_file.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return {t.split("::")[-1] for t in tests}
    by_line = {
        n.lineno: n.name
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    names = set()
    for t in sorted(tests):
        loc = locations.get(t)
        # _locate points at the decorator-stripped def; scan down a few lines to
        # tolerate the offset rather than guessing from the mangled test id.
        name = None
        if loc:
            for probe in range(loc[1], loc[1] + 8):
                if probe in by_line:
                    name = by_line[probe]
                    break
        names.add(name or t.split("::")[-1])
    return names


def measure(rel: str) -> dict[str, set[str]]:
    """{label: concrete tests in `rel` that need that label's runner}."""
    from tools.testing.introspection import collector, platforms

    ran = {}
    for job in [BASELINE_JOB] + [j for _, j, _ in TARGETS]:
        ran[job] = set(collector.status(rel, platforms.get_job(job))["ran"])

    baseline = ran[BASELINE_JOB]
    needs: dict[str, set[str]] = {}
    seen = set(baseline)
    for label, job, _ in TARGETS:
        # Runs here but not at any cheaper capability we have already accounted for.
        needs[label] = ran[job] - seen
        seen |= ran[job]
    return needs


def locations_for(rel: str) -> dict:
    from tools.testing.introspection import collector, platforms

    job = platforms.get_job(TARGETS[-1][1])
    return collector.collect(job, "enumloc", [rel])[rel].get("locations", {})


def uncovered(rel: str) -> list[tuple[str, str, set[str]]]:
    """-> [(label, test.sh function, base names needing to be added)]"""
    smoke_text = TEST_SH.read_text(encoding="utf-8")
    include_name = rel[len("test/") : -len(".py")]
    test_file = REPO_ROOT / rel

    needs = measure(rel)
    if not any(needs.values()):
        return []
    locations = locations_for(rel)

    gaps = []
    for label, _, insert_into in TARGETS:
        if not needs[label]:
            continue
        selected: list[str | None] = []
        for fn in COVERING_FUNCTIONS[label]:
            selected += parse_smoke_includes(smoke_text, fn).get(include_name, [])
        names = base_names(test_file, needs[label], locations)
        missing = {
            n
            for n in names
            if not any(
                e is None or any(p.strip() in n for p in e.split(" or "))
                for e in selected
            )
        }
        if missing:
            gaps.append((label, insert_into, missing))
    return gaps


def smoke_line(include_name: str, names: set[str]) -> str:
    kexpr = " or ".join(sorted(names))
    return (
        f'  time python test/run_test.py --include {include_name} -k "{kexpr}" '
        f"$PYTHON_TEST_EXTRA_OPTION --upload-artifacts-while-running\n"
    )


def add_lines(text: str, function: str, line: str) -> str:
    m = re.search(
        rf"^({function}\(\) \{{\n)(.*?)(^\}})", text, re.DOTALL | re.MULTILINE
    )
    if not m:
        raise SystemExit(f"gpu_coverage: could not find {function}() in {TEST_SH}")
    body = m.group(2)
    anchor = "  assert_git_not_dirty\n"
    body = body.replace(anchor, line + anchor, 1) if anchor in body else body + line
    return text[: m.start(2)] + body + text[m.end(2) :]


def changed_test_files() -> list[str]:
    # ghstack PRs are based on gh/<user>/<n>/base, not main.
    base_ref = os.environ.get("BASE_REF", "main")
    base = subprocess.run(
        ["git", "merge-base", "HEAD", f"origin/{base_ref}"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    ).stdout.strip()
    if not base:
        return []
    out = subprocess.run(
        ["git", "diff", "--name-only", base, "HEAD"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    ).stdout
    return [
        f
        for f in out.splitlines()
        if f.startswith("test/")
        and f.endswith(".py")
        and os.path.basename(f).startswith("test_")
        and (REPO_ROOT / f).is_file()
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="apply the fix to test.sh")
    ap.add_argument("--changed", action="store_true", help="use the PR's changed files")
    ap.add_argument("files", nargs="*")
    args = ap.parse_args()

    files = args.files or (changed_test_files() if args.changed else [])
    files = [r for r in (relpath(f) for f in files) if r and r.startswith("test/")]
    if not files:
        print("gpu_coverage: no test files to check")
        return 0

    text = TEST_SH.read_text(encoding="utf-8")
    failures = 0
    for rel in files:
        for label, insert_into, names in uncovered(rel):
            failures += 1
            include_name = rel[len("test/") : -len(".py")]
            line = smoke_line(include_name, names)
            if args.write:
                text = add_lines(text, insert_into, line)
                print(f"{rel}: added {len(names)} test(s) to {insert_into}()")
                continue
            print(f"\nGPU coverage gap: {rel}")
            for n in sorted(names):
                print(f"    {n}  -- runs on {label.split('/')[1]}, not on a10g")
            print(f"  not selected by {insert_into}() in .ci/pytorch/test.sh")
            print(f"\n  Add to .ci/pytorch/test.sh:\n{line}")
            print(f"  Or run:  python tools/testing/gpu_coverage.py --write {rel}\n")

    if args.write and failures:
        TEST_SH.write_text(text, encoding="utf-8")
        return 0
    if failures:
        print(
            f"{failures} gap(s). These tests will silently skip wherever CI runs them.\n"
            f"Note this only makes the job cover them; running it still needs the\n"
            f"ciflow label applied manually."
        )
        return 1
    print(f"gpu_coverage: {len(files)} file(s) OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
