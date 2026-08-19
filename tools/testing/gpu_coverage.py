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
from typing import NamedTuple


REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_SH_REL = ".ci/pytorch/test.sh"
TEST_SH = REPO_ROOT / TEST_SH_REL

sys.path.insert(0, str(REPO_ROOT))

# a10g capability: anything that already runs here needs no GPU-specific job.
BASELINE_JOB = "linux-cuda-sm86/default"

# (ciflow label, job simulating its runner, test.sh function it dispatches to).
# Only the `smoke` configs are keyed to the bare ciflow/h100 and ciflow/b200 tags;
# the cutlass / distributed / symm-mem jobs carry their own labels.
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
    """{run_test.py file: [-k expr, ...]}; a None entry runs the whole file."""
    out: dict[str, list[str | None]] = {}
    joined = re.sub(r"\\\n\s*", " ", function_body(text, function).group(2))
    for line in joined.split("\n"):
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
    """Would any of these -k expressions select this Class::method id?

    Matching the whole id, rather than the method name, is what makes a
    class-scoped filter such as `-k TestForeachMM` count as coverage.
    """
    for expr in patterns:
        if expr is None:
            return True
        # Only ` or ` is understood. A pattern using `and`/`not` is treated as
        # opaque, so we over-report a gap rather than miss one.
        if any(p and p.strip() in test_id for p in expr.split(" or ")):
            return True
    return False


class Measured(NamedTuple):
    needs: dict[str, dict[str, set[str]]]  # {file: {label: concrete tests}}
    errors: dict[str, str]  # {file: why it was not measured}
    gates: dict[str, dict[str, list[str]]]  # {file: {test: names its skip reads}}
    probes: dict[str, dict[str, dict[str, bool]]]  # {file: {job: {name: value}}}
    undecidable: dict[str, list[str]]  # {file: descriptor-forced predicates it uses}


def measure(files: list[str]) -> Measured:
    from tools.testing.introspection import collector, platforms

    ran: dict[str, dict[str, set[str]]] = {}
    errors: dict[str, str] = {}
    gates: dict[str, dict[str, list[str]]] = {}
    probes: dict[str, dict[str, dict[str, bool]]] = {}
    for job in [BASELINE_JOB] + [j for _, j, _ in TARGETS]:
        for rel, payload in collector.collect(
            platforms.get_job(job), "status", files
        ).items():
            if "error" in payload:
                errors.setdefault(rel, str(payload["error"]))
                continue
            seen = len(payload["ran"]) + len(payload["skipped"])
            if seen < payload["loadable"]:
                # The suite was abandoned partway, so the tests after the abort are
                # in neither bucket and would silently read as needing nothing.
                errors.setdefault(
                    rel,
                    f"{job}: suite stopped after {seen} of {payload['loadable']} "
                    f"tests, so the rest could not be measured",
                )
                continue
            ran.setdefault(rel, {})[job] = set(payload["ran"])
            gates.setdefault(rel, {}).update(payload["hidden_gates"])
            # Per file, not merged: two files can gate on the same name with
            # different module-local values, and merging picks whichever ran last.
            probes.setdefault(rel, {})[job] = payload["probes"]

    needs = {}
    for rel, by_job in ran.items():
        if rel in errors:
            continue
        # h100 and b200 are independent labels, so a test needing sm90+ belongs in
        # both lists: a b200-labelled PR runs only the b200 list.
        needs[rel] = {
            label: by_job[job] - by_job[BASELINE_JOB] for label, job, _ in TARGETS
        }

    # A predicate the descriptor pins to one value cannot distinguish the runners,
    # so a decorator gating on it hides the test from the difference entirely --
    # no gap, no warning. Derived from the descriptor rather than listed here.
    forced = forced_predicates()
    undecidable = {}
    for rel in needs:
        src = (REPO_ROOT / rel).read_text(encoding="utf-8", errors="ignore")
        if hits := sorted(n for n in forced if n in src):
            undecidable[rel] = hits
    return Measured(
        needs=needs, errors=errors, gates=gates, probes=probes, undecidable=undecidable
    )


def forced_predicates() -> list[str]:
    """Capability overrides the descriptor pins to the same value everywhere."""
    from tools.testing.introspection import platforms

    caps = {c: platforms._cuda_caps(c, rocm=False) for c in [(8, 6), (9, 0), (10, 0)]}
    return [n for n in caps[(9, 0)] if len({caps[c][n] for c in caps}) == 1]


def unjudged(rel: str, m: Measured) -> list[tuple[str, str, str]]:
    """(test, cause, action) for tests in this file whose gating was not observed.

    A skip inside setUp or a test body never runs here, so the test is recorded as
    having executed at every capability and drops out of the difference. The causes
    want different things from different people, so they are reported apart.
    """
    out = []
    by_job = m.probes.get(rel, {})
    judged = {t for tests in m.needs.get(rel, {}).values() for t in tests}

    if forced := m.undecidable.get(rel):
        out.append(
            (
                rel,
                f"{', '.join(forced)} is pinned to one value by the descriptor at "
                f"every capability, so a test gating on it shows no difference",
                "if such a test needs an H100 or B200, list it by hand",
            )
        )

    for test, names in sorted(m.gates.get(rel, {}).items()):
        if test in judged:
            continue  # already reported, as a gap or above
        seen = [
            (n, [by_job.get(j, {}).get(n) for j in by_job])
            for n in names
            if any(n in p for p in by_job.values())
        ]
        dark = [n for n, vals in seen if not any(vals)]
        if dark:
            out.append(
                (
                    test,
                    f"{', '.join(dark)} reads False at every simulated capability "
                    f"in this image, so nothing distinguishes the runners",
                    "not yours to fix: the image lacks what the predicate probes",
                )
            )
        else:
            out.append(
                (
                    test,
                    "its skip is inside setUp or the test body, which is not run here",
                    "declare the requirement in a decorator if the shape allows it, "
                    "otherwise add the --include/-k line by hand",
                )
            )
    return out


def locations(files: list[str]) -> dict[str, dict[str, list]]:
    """{file: {Class::method: [defining file, line its decorators start on]}}"""
    from tools.testing.introspection import collector, platforms

    job = platforms.get_job(TARGETS[-1][1])  # widest capability: most tests exist
    out = {}
    for rel, payload in collector.collect(job, "enumloc", files).items():
        if "error" in payload:
            raise SystemExit(f"gpu_coverage: cannot locate tests in {rel}: {payload}")
        out[rel] = payload.get("locations", {})
    return out


def decorator_starts(path: Path) -> dict[int, str]:
    """{line a function's decorator stack starts on: function name}.

    inspect.getsourcelines, which the engine uses, reports that line and not the
    `def`, so this is the key its locations are expressed in.
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
        # Without a location the test is defined outside the test tree; its
        # concrete id is still a valid -k token, just a brittle one.
        names.add(name or t.split("::")[-1])
    return names


def smoke_line(rel: str, names: set[str]) -> str:
    include = rel[len("test/") : -len(".py")]
    kexpr = " or ".join(sorted(names))
    return (
        f'  time python test/run_test.py --include {include} -k "{kexpr}" '
        f"$PYTHON_TEST_EXTRA_OPTION --upload-artifacts-while-running\n"
    )


def insert(text: str, function: str, line: str) -> str:
    m = function_body(text, function)
    anchor = "  assert_git_not_dirty\n"
    if anchor not in m.group(2):
        raise SystemExit(f"gpu_coverage: no insertion anchor in {function}()")
    body = m.group(2).replace(anchor, line + anchor, 1)
    return text[: m.start(2)] + body + text[m.end(2) :]


def is_test_file(rel: str) -> bool:
    return (
        rel.startswith("test/")
        and rel.endswith(".py")
        and os.path.basename(rel).startswith("test_")
        and (REPO_ROOT / rel).is_file()
    )


def git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=REPO_ROOT)


def merge_base() -> str:
    # ghstack PRs are based on gh/<user>/<n>/base, not main.
    base_ref = os.environ.get("BASE_REF", "main")
    r = git("merge-base", "HEAD", f"origin/{base_ref}")
    if r.returncode or not r.stdout.strip():
        # Returning nothing here would make the whole check a silent no-op. A
        # shallow fetch of the base branch is the usual cause.
        raise SystemExit(
            f"gpu_coverage: no merge-base with origin/{base_ref}: "
            f"{r.stderr.strip() or 'unknown'}"
        )
    return r.stdout.strip()


def show(rev: str, rel: str) -> str:
    r = git("show", f"{rev}:{rel}")
    if r.returncode:
        raise SystemExit(
            f"gpu_coverage: cannot read {rel} at {rev}: {r.stderr.strip()}"
        )
    return r.stdout


def smoke_files(text: str) -> set[str]:
    """The test files named by either smoke function."""
    named = {f for _, _, fn in TARGETS for f in smoke_patterns(text, fn)}
    return {f"test/{f}.py" for f in named if is_test_file(f"test/{f}.py")}


def resolve(paths: list[str]) -> list[str]:
    out = []
    for p in paths:
        try:
            rel = str(Path(p).resolve().relative_to(REPO_ROOT))
        except ValueError:
            raise SystemExit(f"gpu_coverage: {p} is outside {REPO_ROOT}") from None
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

    text = TEST_SH.read_text(encoding="utf-8")
    files = resolve(args.files)
    base_text = None
    if args.changed:
        base = merge_base()
        diff = git("diff", "--name-only", base, "HEAD")
        if diff.returncode:
            # An empty list here would make the whole check a silent no-op.
            raise SystemExit(f"gpu_coverage: git diff failed: {diff.stderr.strip()}")
        changed = diff.stdout.split()
        files += [f for f in changed if is_test_file(f)]
        if TEST_SH_REL in changed:
            base_text = show(base, TEST_SH_REL)

    # Editing test.sh can drop coverage without touching a test file, so re-check
    # everything it lists. Only regressions are reported for those: holding a PR
    # to account for pre-existing gaps in files it never touched is what the
    # changed-file scoping exists to prevent.
    direct = sorted(set(files))
    regress_only = (
        (smoke_files(base_text) | smoke_files(text)) - set(direct)
        if base_text is not None
        else set()
    )
    files = direct + sorted(regress_only)
    if not files:
        print("gpu_coverage: no test files to check")
        return 0

    patterns = {fn: smoke_patterns(text, fn) for _, _, fn in TARGETS}
    was_patterns = (
        {fn: smoke_patterns(base_text, fn) for _, _, fn in TARGETS}
        if base_text is not None
        else {}
    )
    m = measure(files)
    needs, errors = m.needs, m.errors
    locs = locations(sorted(needs)) if needs else {}

    gaps = 0
    for rel in sorted(needs):
        include = rel[len("test/") : -len(".py")]
        dropped = rel in regress_only
        for label, _, function in TARGETS:
            required = needs[rel][label]
            if dropped:
                was = was_patterns[function].get(include, [])
                required = {t for t in required if selects(was, t)}
            missing = {
                t
                for t in required
                if not selects(patterns[function].get(include, []), t)
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
            title = "GPU coverage removed" if dropped else "GPU coverage gap"
            verb = "no longer selected" if dropped else "not selected"
            print(
                f"::error file={rel},title={title}::{', '.join(sorted(names))} "
                f"run on {runner} but not on a10g, and are {verb} by {function}() "
                f"in .ci/pytorch/test.sh. Fix: python "
                f"tools/testing/gpu_coverage.py --write {rel}"
            )
            print(f"\n{title}: {rel}")
            for n in sorted(names):
                print(f"    {n}  -- runs on {runner}, not on a10g")
            print(f"  {verb} by {function}() in .ci/pytorch/test.sh")
            print(f"\n  Add to .ci/pytorch/test.sh:\n{line}")
            print(f"  Or run:  python tools/testing/gpu_coverage.py --write {rel}\n")

    for rel, err in sorted(errors.items()):
        print(f"::warning file={rel},title=GPU coverage not measured::{err}")
        print(f"gpu_coverage: could not measure {rel}: {err}")

    # Only for files this PR touched. A file pulled in because test.sh changed is
    # not the author's to answer for, and an unactionable warning is noise that
    # teaches people to ignore the actionable ones.
    incomplete = 0
    for rel in direct:
        for test, cause, action in unjudged(rel, m):
            incomplete += 1
            print(
                f"::warning file={rel},title=GPU coverage incomplete::"
                f"{test} could not be classified: {cause}. {action}."
            )
            label = rel if test == rel else f"{rel}::{test}"
            print(f"\n{label} not classified")
            print(f"    why:  {cause}")
            print(f"    fix:  {action}")

    if args.write:
        TEST_SH.write_text(text, encoding="utf-8")
        return 0
    if gaps:
        print(
            f"\n{gaps} gap(s). These tests skip silently wherever CI runs them.\n"
            f"This only makes the job cover them; running it still needs the "
            f"ciflow label applied by hand."
        )
        return 1
    # "OK" must mean measured-and-clean. A file that could not be measured is
    # counted apart, so the summary never credits a file the check never read.
    measured = len(files) - len(errors)
    suffix = f", {incomplete} incomplete" if incomplete else ""
    if errors:
        suffix += f", {len(errors)} NOT MEASURED"
    print(f"gpu_coverage: {measured} of {len(files)} file(s) OK{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
