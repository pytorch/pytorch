"""Behavior 1: tests added / removed between two git refs.

Collects the concrete test set for the affected files at each ref and set-diffs them.
Each ref's test files are read from a throwaway git worktree (via TESTINTRO_ROOT)
while torch comes from the current build -- so a single build serves both refs for
the common test-only / OpInfo-unchanged diff. Changes that touch native/codegen or
the test infra (common_*, run_test, conftest) are treated as broad and widen the
scope to the whole selected file set (with a warning), since they can change
generation anywhere.
"""

from __future__ import annotations

import ast
import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from typing import TYPE_CHECKING

from tools.testing.introspection import collector


if TYPE_CHECKING:
    from tools.testing.introspection.platforms import Job


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=str(collector.REPO),
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _gh(*args: str) -> str:
    return subprocess.run(
        ["gh", *args],
        cwd=str(collector.REPO),
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _sha_present(sha: str) -> bool:
    try:
        _git("cat-file", "-e", f"{sha}^{{commit}}")
        return True
    except subprocess.CalledProcessError:
        return False


def resolve_pr(pr: int) -> tuple[str, str]:
    """Resolve a GitHub PR to (base_sha, head_sha), fetching objects only if missing.

    base is the fork point (merge-base of the PR head and its base branch) so the
    diff reflects only the PR's own changes, not unrelated base-branch movement.
    """
    info = json.loads(_gh("pr", "view", str(pr), "--json", "headRefOid,baseRefName"))
    head, base_branch = info["headRefOid"], info["baseRefName"]
    if not _sha_present(head):
        _git("fetch", "origin", f"pull/{pr}/head")
    try:
        _git("rev-parse", "--verify", f"origin/{base_branch}")
    except subprocess.CalledProcessError:
        _git("fetch", "origin", base_branch)
    base = _git("merge-base", head, f"origin/{base_branch}").strip()
    return base, head


def _changed_files(a: str, b: str) -> list[str]:
    return [ln for ln in _git("diff", "--name-only", a, b).splitlines() if ln]


def _is_test_py(path: str) -> bool:
    return path.startswith("test/") and path.endswith(".py") and "/test_" in f"/{path}"


# Generation/selection surface: changes here can alter which tests EXIST (the only
# thing B1 reports), anywhere. Other torch/ source changes only affect behavior/skips,
# not the test set, so they are not broad for an added/removed diff.
_GEN_PREFIXES = ("torch/testing/_internal/", "tools/testing/")


def _is_broad(path: str) -> bool:
    if path.startswith("test/"):
        # Non-.py files under test/ (expected-failure lists, json, data) affect
        # xfail/skip at runtime, not which tests exist -> not broad.
        if not path.endswith(".py"):
            return False
        base = path.rsplit("/", 1)[-1]
        # A .py helper/base (not a test_*.py file) can change generation broadly.
        return base in ("run_test.py", "conftest.py") or not base.startswith("test_")
    if path.startswith(_GEN_PREFIXES):
        return True
    # Codegen inputs that synthesize ops/tests.
    return path.endswith((".yaml", ".yml")) and (
        "native_functions" in path or "deriv" in path
    )


def _module_ids(relpath: str) -> tuple[str, str]:
    rel = relpath[len("test/") : -len(".py")]
    return rel.replace("/", "."), rel.split("/")[-1]


def _import_graph(b: str, files: list[str], root) -> dict[str, set[str]]:
    """Import graph for ref `b`, disk-cached by the resolved sha (contents are fixed
    per sha), so repeat diffs of the same head skip the ~15s AST pass."""
    try:
        bsha = _git("rev-parse", b).strip()
    except subprocess.CalledProcessError:
        bsha = b
    import hashlib

    key = hashlib.sha256(
        f"graph|{collector.CACHE_VERSION}|{bsha}|{len(files)}".encode()
    ).hexdigest()
    path = collector.CACHE_DIR / f"{key}.json"
    if path.exists():
        try:
            return {k: set(v) for k, v in json.loads(path.read_text()).items()}
        except Exception:
            pass
    g = _build_import_graph(files, root)
    try:
        collector.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({k: sorted(v) for k, v in g.items()}))
    except Exception:
        pass
    return g


def _build_import_graph(files: list[str], root) -> dict[str, set[str]]:
    """{relpath: set of module tokens it imports}, by AST (no torch). Depends only on
    file contents, so it can be built once and reused across jobs/platforms."""
    imports_of: dict[str, set[str]] = {}
    for f in files:
        try:
            tree = ast.parse((root / f).read_text())
        except Exception:
            continue
        toks: set[str] = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.module:
                toks.add(n.module)
            elif isinstance(n, ast.Import):
                for a in n.names:
                    toks.add(a.name)
        imports_of[f] = toks
    return imports_of


def _scope(
    changed_tests: list[str], selected: list[str], imports_of: dict[str, set[str]]
) -> set[str]:
    """Affected = changed test files + any selected file that (transitively) imports a
    changed test module (catches synthetic subclassers, e.g. test_jit_legacy).
    `imports_of` is a prebuilt import graph (see _build_import_graph)."""
    targets: set[str] = set()
    for f in changed_tests:
        targets.update(_module_ids(f))
    affected = set(changed_tests)
    grew = True
    while grew:
        grew = False
        for f, toks in imports_of.items():
            if f in affected:
                continue
            if toks & targets:
                affected.add(f)
                targets.update(_module_ids(f))
                grew = True
    return (affected & set(selected)) | set(changed_tests)


def _worktree(ref: str) -> str:
    d = tempfile.mkdtemp(prefix="testintro_wt_")
    subprocess.run(
        ["git", "worktree", "add", "--detach", "-f", d, ref],
        cwd=str(collector.REPO),
        capture_output=True,
        text=True,
        check=True,
    )
    return d


def _remove_worktree(d: str) -> None:
    subprocess.run(
        ["git", "worktree", "remove", "--force", d],
        cwd=str(collector.REPO),
        capture_output=True,
        text=True,
    )
    shutil.rmtree(d, ignore_errors=True)


@contextlib.contextmanager
def _root_env(path: str):
    prev = os.environ.get("TESTINTRO_ROOT")
    os.environ["TESTINTRO_ROOT"] = path
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("TESTINTRO_ROOT", None)
        else:
            os.environ["TESTINTRO_ROOT"] = prev


def _collect_job_files(
    job: Job, files: list[str], max_workers: int, op: str = "enumerate"
) -> tuple[dict[str, set[str]], set[str], dict[str, dict]]:
    """Collect `files` for one job at the currently-set TESTINTRO_ROOT. Returns
    ({relpath: {Class::method,...}}, errored, {relpath: {Class::method: [file, line]}}).
    Locations are present only when op == 'enumloc'."""
    present = [f for f in files if (collector._root() / f).exists()]
    raw = collector.collect(job, op, present, max_workers=max_workers)
    out: dict[str, set[str]] = {}
    errored: set[str] = set()
    locs: dict[str, dict] = {}
    for f in files:
        r = raw.get(f)
        if r and "error" not in r:
            out[f] = {f"{c}::{m}" for c, ms in r["classes"].items() for m in ms}
            if "locations" in r:
                locs[f] = r["locations"]
        elif r and "error" in r:
            out[f] = set()
            errored.add(f)
        else:
            out[f] = set()  # absent at this root
    return out, errored, locs


def _collect_phase(
    root: str,
    jobs: list[Job],
    affected_by_job: dict[str, list[str]],
    label: str,
    op: str = "enumerate",
) -> dict[str, tuple]:
    """Collect all jobs at one ref `root`, in parallel. The root is identical for
    every job in the phase, so it is set once (no per-thread env mutation), and the
    worker budget is split across jobs to avoid oversubscription."""
    from concurrent.futures import as_completed, ThreadPoolExecutor

    cap = collector._default_workers()
    per_job_workers = max(1, cap // max(1, len(jobs)))
    results: dict[str, tuple] = {}
    with _root_env(root):
        with ThreadPoolExecutor(max_workers=max(1, len(jobs))) as ex:
            futs = {
                ex.submit(
                    _collect_job_files,
                    job,
                    affected_by_job[job.name],
                    per_job_workers,
                    op,
                ): job
                for job in jobs
            }
            for k, fut in enumerate(as_completed(futs), 1):
                results[futs[fut].name] = fut.result()
                print(
                    f"\r  {label}: {k}/{len(jobs)} platforms   ",
                    end="",
                    file=sys.stderr,
                    flush=True,
                )
    print(file=sys.stderr)
    return results


def diff(
    jobs: list[Job],
    a: str,
    b: str,
    files_filter: str | None = None,
    full: bool = False,
    locations: bool = False,
) -> dict:
    """Tests added/removed between refs a and b, for each job. Both refs are
    materialized once as git worktrees and reused across all jobs. Selection (varies
    only by config+rocm) and the import graph (content-only) are computed once and
    reused; collection fans out across jobs in parallel per ref."""
    changed = _changed_files(a, b)
    changed_tests = [f for f in changed if _is_test_py(f)]
    broad = full or any(_is_broad(f) for f in changed)
    scope_reason = (
        "broad change (native/codegen/test-infra) -> full scope"
        if broad
        else "test-only change -> scoped to changed files + importers"
    )

    wt_a, wt_b = _worktree(a), _worktree(b)
    try:
        # Selection + import graph depend on the head tree, not the platform (modulo
        # the rocm blocklist), so compute each once and share across jobs.
        sel_by_key: dict[tuple, list[str]] = {}
        with _root_env(wt_b):
            for job in jobs:
                key = (job.config.name, job.platform.rocm)
                if key not in sel_by_key:
                    print(f"selecting files {key}...", file=sys.stderr, flush=True)
                    sel_by_key[key] = collector.select_files(job)["files"]
            imports_of: dict[str, set[str]] = {}
            if not broad:
                union = sorted(set().union(*sel_by_key.values()))
                print(
                    f"building import graph ({len(union)} files)...",
                    file=sys.stderr,
                    flush=True,
                )
                imports_of = _import_graph(b, union, collector._root())

        affected_by_job: dict[str, list[str]] = {}
        for job in jobs:
            selected = sel_by_key[(job.config.name, job.platform.rocm)]
            if broad:
                affected = sorted(set(selected) | set(changed_tests))
            else:
                affected = sorted(_scope(changed_tests, selected, imports_of))
            if files_filter:
                affected = [f for f in affected if files_filter in f]
            affected_by_job[job.name] = affected

        op = "enumloc" if locations else "enumerate"
        at_a = _collect_phase(wt_a, jobs, affected_by_job, "collecting base", op)
        at_b = _collect_phase(wt_b, jobs, affected_by_job, "collecting head", op)

        per_job = {}
        # Locations key on the comment's test id "file::Class::method": added tests are
        # located at the head (b), removed at the base (a), since each exists only there.
        added_loc: dict[str, list] = {}
        removed_loc: dict[str, list] = {}
        for job in jobs:
            affected = affected_by_job[job.name]
            sa_map, err_a, loc_a = at_a[job.name]
            sb_map, err_b, loc_b = at_b[job.name]
            errored = err_a | err_b
            per_file = {}
            for f in affected:
                if f in errored:
                    continue  # unreliable at a ref -> don't report a fake diff
                added = sorted(sb_map[f] - sa_map[f])
                removed = sorted(sa_map[f] - sb_map[f])
                if added or removed:
                    per_file[f] = {"added": added, "removed": removed}
                for t in added:
                    loc = loc_b.get(f, {}).get(t)
                    if loc:
                        added_loc[f"{f}::{t}"] = loc
                for t in removed:
                    loc = loc_a.get(f, {}).get(t)
                    if loc:
                        removed_loc[f"{f}::{t}"] = loc
            per_job[job.name] = {
                "scope_reason": scope_reason,
                "n_affected": len(affected),
                "uncomparable": sorted(errored),
                "per_file": per_file,
            }
    finally:
        _remove_worktree(wt_a)
        _remove_worktree(wt_b)

    result = {"from": a, "to": b, "per_job": per_job}
    if locations:
        result["added_loc"] = added_loc
        result["removed_loc"] = removed_loc
    return result


# --------------------------------------------------------------------------- #
# Result shaping shared by the CLI text view and the PR-comment renderer
# --------------------------------------------------------------------------- #
def invert_per_job(
    res: dict,
) -> tuple[dict[str, set[str]], dict[str, set[str]], list[str]]:
    """From a diff() result, return (added, removed, job_names) where added/removed map
    a concrete test id "file::Class::method" -> the set of jobs it was added/removed on."""
    job_names = list(res["per_job"])
    added: dict[str, set[str]] = {}
    removed: dict[str, set[str]] = {}
    for job_name, jr in res["per_job"].items():
        for f, v in jr["per_file"].items():
            for t in v["added"]:
                added.setdefault(f"{f}::{t}", set()).add(job_name)
            for t in v["removed"]:
                removed.setdefault(f"{f}::{t}", set()).add(job_name)
    return added, removed, job_names


def group_by_platform_set(
    m: dict[str, set[str]], all_jobs: set[str]
) -> list[tuple[frozenset[str], list[str]]]:
    """Group test ids by the set of jobs they apply to; all-platforms group first."""
    groups: dict[frozenset[str], list[str]] = {}
    for test, plats in m.items():
        groups.setdefault(frozenset(plats), []).append(test)
    return [
        (fs, sorted(groups[fs]))
        for fs in sorted(groups, key=lambda fs: (fs != all_jobs, sorted(fs)))
    ]
