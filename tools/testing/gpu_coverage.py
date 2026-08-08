#!/usr/bin/env python3
"""Check that tests requiring an H100/B200 are selected by the smoke runs.

Reports gaps in .ci/pytorch/test.sh; --write inserts the missing run_test.py lines.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_SH = REPO_ROOT / ".ci" / "pytorch" / "test.sh"

sys.path.insert(0, str(REPO_ROOT))

# a10g capability: anything that already runs here needs no GPU-specific job.
BASELINE_JOB = "linux-cuda-sm86/default"

# (ciflow label, job simulating its runner, test.sh function it dispatches to).
# Only the `smoke` configs are keyed to the bare ciflow/h100 and ciflow/b200 tags;
# the cutlass/distributed/symm-mem jobs carry their own labels.
TARGETS = [
    ("ciflow/h100", "linux-cuda-sm90/default", "test_python_smoke"),
    ("ciflow/b200", "linux-cuda-sm100/default", "test_python_smoke_b200"),
]


def function_body(text: str, function: str) -> re.Match[str]:
    m = re.search(
        rf"^({function}\(\) \{{\n)(.*?)(^\}})", text, re.DOTALL | re.MULTILINE
    )
    if not m:
        raise SystemExit(f"gpu_coverage: could not find {function}() in {TEST_SH}")
    return m


def smoke_patterns(text: str, function: str) -> dict[str, list[str | None]]:
    """{run_test.py file: [-k expr, ...]} for one test.sh function.

    A None entry means some include runs the whole file.
    """
    out: dict[str, list[str | None]] = {}
    body = re.sub(r"\\\n\s*", " ", function_body(text, function).group(2))
    for line in body.split("\n"):
        try:
            toks = shlex.split(line, comments=True)
        except ValueError:
            continue
        if "--include" not in toks:
            continue
        files = []
        for tok in toks[toks.index("--include") + 1 :]:
            if tok.startswith("-"):
                break
            files.append(tok.removesuffix(".py"))
        kexpr = toks[toks.index("-k") + 1] if "-k" in toks[:-1] else None
        for f in files:
            out.setdefault(f, []).append(kexpr)
    return out


def selects(patterns: list[str | None], test_id: str) -> bool:
    """Would any of these -k expressions select this Class::method id?"""
    for expr in patterns:
        if expr is None:
            return True
        # Only ` or ` is understood. A pattern using `and`/`not` is opaque and
        # assumed not to select, so we over-report rather than silently skip.
        if any(p and p.strip() in test_id for p in expr.split(" or ")):
            return True
    return False


def measure(files: list[str]) -> tuple[dict[str, dict[str, set[str]]], dict[str, str]]:
    """-> ({file: {label: concrete tests needing that label}}, {file: error})"""
    from tools.testing.introspection import collector, platforms

    ran: dict[str, dict[str, set[str]]] = {}
    errors: dict[str, str] = {}
    for job in [BASELINE_JOB] + [j for _, j, _ in TARGETS]:
        results = collector.collect(platforms.get_job(job), "status", files)
        for rel, payload in results.items():
            if "error" in payload:
                errors.setdefault(rel, str(payload["error"]))
            else:
                ran.setdefault(rel, {})[job] = set(payload["ran"])

    needs: dict[str, dict[str, set[str]]] = {}
    for rel, by_job in ran.items():
        if len(by_job) != len(TARGETS) + 1:
            errors.setdefault(rel, "incomplete measurement across capabilities")
            continue
        # ciflow/h100 and ciflow/b200 are independent labels, so a test needing
        # sm90+ belongs in both smoke lists: a b200-labelled PR runs only the b200
        # list and would otherwise skip it.
        needs[rel] = {
            label: by_job[job] - by_job[BASELINE_JOB] for label, job, _ in TARGETS
        }
    return needs, errors


def locations(files: list[str]) -> dict[str, dict[str, list]]:
    """{file: {Class::method: [defining file, first decorator line]}}"""
    from tools.testing.introspection import collector, platforms

    job = platforms.get_job(TARGETS[-1][1])  # widest capability: most tests exist
    out = {}
    for rel, payload in collector.collect(job, "enumloc", files).items():
        # A file whose status measured fine can still fail enumloc (different
        # import path); fall back to concrete ids rather than aborting.
        out[rel] = {} if "error" in payload else payload.get("locations", {})
    return out


def decorator_starts(path: Path) -> dict[int, str]:
    """{line a function's decorator stack starts on: function name}.

    inspect.getsourcelines, which the engine uses for locations, reports that
    line rather than the `def`.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return {}
    return {
        min([d.lineno for d in n.decorator_list] + [n.lineno]): n.name
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def base_names(tests: set[str], locs: dict[str, list]) -> set[str]:
    """Concrete test ids -> the `def` names a -k expression should carry.

    Parametrized variants share a def, so grouping by location collapses them.
    """
    cache: dict[str, dict[int, str]] = {}
    names = set()
    for t in tests:
        loc = locs.get(t)
        name = None
        if loc:
            defs = cache.setdefault(loc[0], decorator_starts(REPO_ROOT / loc[0]))
            name = defs.get(loc[1])
        # Falling back to the concrete id happens for tests defined outside the
        # test tree; it is a valid -k token but pins the parametrize values.
        names.add(name or t.split("::")[-1])
    return names


def smoke_line(rel: str, names: set[str]) -> str:
    include = rel[len("test/") : -len(".py")]
    kexpr = " or ".join(sorted(names))
    return (
        f"  time python test/run_test.py --include {include} "
        f'-k "{kexpr}" $PYTHON_TEST_EXTRA_OPTION --upload-artifacts-while-running\n'
    )


def insert(text: str, function: str, line: str) -> str:
    m = function_body(text, function)
    anchor = "  assert_git_not_dirty\n"
    body = m.group(2)
    if anchor not in body:
        raise SystemExit(f"gpu_coverage: no insertion anchor in {function}()")
    return (
        text[: m.start(2)] + body.replace(anchor, line + anchor, 1) + text[m.end(2) :]
    )


def is_test_file(rel: str) -> bool:
    return (
        rel.startswith("test/")
        and rel.endswith(".py")
        and os.path.basename(rel).startswith("test_")
        and (REPO_ROOT / rel).is_file()
    )


def changed_test_files() -> list[str]:
    # ghstack PRs are based on gh/<user>/<n>/base, not main.
    base_ref = os.environ.get("BASE_REF", "main")
    merge_base = subprocess.run(
        ["git", "merge-base", "HEAD", f"origin/{base_ref}"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    if merge_base.returncode or not merge_base.stdout.strip():
        # Returning nothing here would silently make the whole check a no-op, so
        # it is fatal. A shallow fetch of the base branch is the usual cause.
        raise SystemExit(
            f"gpu_coverage: no merge-base with origin/{base_ref} "
            f"({merge_base.stderr.strip() or 'unknown error'}); "
            f"the base branch must be fetched unshallowed"
        )
    diff = subprocess.run(
        ["git", "diff", "--name-only", merge_base.stdout.strip(), "HEAD"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=True,
    )
    return [f for f in diff.stdout.split() if is_test_file(f)]


def resolve(paths: list[str]) -> list[str]:
    out = []
    for p in paths:
        try:
            rel = str(Path(p).resolve().relative_to(REPO_ROOT))
        except ValueError:
            raise SystemExit(f"gpu_coverage: {p} is not inside {REPO_ROOT}") from None
        if not is_test_file(rel):
            raise SystemExit(f"gpu_coverage: {p} is not an existing test/**/test_*.py")
        out.append(rel)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="apply the fix to test.sh")
    ap.add_argument("--changed", action="store_true", help="use the PR's changed files")
    ap.add_argument("files", nargs="*")
    args = ap.parse_args()

    files = set(resolve(args.files))
    if args.changed:
        files |= set(changed_test_files())
    if not files:
        print("gpu_coverage: no test files to check")
        return 0

    text = TEST_SH.read_text(encoding="utf-8")
    selected = {fn: smoke_patterns(text, fn) for _, _, fn in TARGETS}
    needs, errors = measure(sorted(files))
    locs = locations(sorted(needs)) if needs else {}

    gaps = 0
    for rel in sorted(needs):
        include = rel[len("test/") : -len(".py")]
        for label, _, function in TARGETS:
            missing = {
                t
                for t in needs[rel][label]
                if not selects(selected[function].get(include, []), t)
            }
            if not missing:
                continue
            gaps += 1
            names = base_names(missing, locs.get(rel, {}))
            line = smoke_line(rel, names)
            if args.write:
                text = insert(text, function, line)
                print(f"{rel}: added {len(names)} test(s) to {function}()")
                continue
            runner = label.split("/")[1]
            print(
                f"::error file={rel},title=GPU coverage gap::"
                f"{', '.join(sorted(names))} run on {runner} but not on a10g, and "
                f"{function}() in .ci/pytorch/test.sh does not select them. Run "
                f"`python tools/testing/gpu_coverage.py --write {rel}` to fix."
            )
            print(f"\nGPU coverage gap: {rel}")
            for n in sorted(names):
                print(f"    {n}  -- runs on {runner}, not on a10g")
            print(f"  not selected by {function}() in .ci/pytorch/test.sh")
            print(f"\n  Add to .ci/pytorch/test.sh:\n{line}")
            print(f"  Or run:  python tools/testing/gpu_coverage.py --write {rel}\n")

    for rel, err in sorted(errors.items()):
        print(f"::warning file={rel},title=GPU coverage not measured::{err}")
        print(f"gpu_coverage: could not measure {rel}: {err}")

    if args.write:
        TEST_SH.write_text(text, encoding="utf-8")
        return 0
    if gaps:
        print(
            f"\n{gaps} gap(s). These tests skip silently wherever CI runs them.\n"
            f"This only makes the job cover them; running it still needs the "
            f"ciflow label applied manually."
        )
        return 1
    print(f"gpu_coverage: {len(files)} file(s) OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
