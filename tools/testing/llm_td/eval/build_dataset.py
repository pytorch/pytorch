"""Build the labelled eval set from CI history.

    python -m tools.testing.llm_td.eval.build_dataset --data DIR --start 2026-09-20 --end 2026-10-26

Four kinds of sample, each a JSON file under DIR/samples:

  pr_<N>      a PR whose own CI failed tests that main did not fail. The
              ground truth is every (test file, CI config) that failed on the
              earliest failing head, in a job that runs TD and did not pass on
              any attempt on that head. These runs are censored: pull/trunk
              jobs already skip 75% of files through TD, so truth is a lower
              bound.
  head_<N>_<sha>
              every head of a pr_ sample's PR with a non-flaky test failure in a
              job that runs TD; the truth is those failures. A failure is flaky
              by Dr.CI's signals, with hindsight (see flaky_flags).
  rev_<sha>   a commit the autorevert bot reverted between --start and --end
              because it broke a test on main, where every test runs. The
              truth is the test files of the reverting signal that failed on
              main between landing and the revert, on the commits that
              contain it, in every job that runs TD they failed in there. A
              revert with no such failure is skipped.
  neu_<sha>   an ordinary main commit from the 25 days before --end, with no
              truth, used only to measure how much a selector runs on typical
              traffic. Reverted commits and those whose title starts with
              Revert or Reland are left out.

Each truth file's `truth_jobs` lists the [workflow, job] it failed in.
Samples are diffed as select_tests diffs them: against `merge_base`, the merge
base of the head with `base`. For PR and head samples `base` is the PR base CI
passed to select_tests for that head (the PR below it for a ghstack PR), read
from the head's pull_request runs (see pr_bases), or the merge base with main
when no run recorded one (`base_source` says which). For revert and neutral samples it
is the parent commit. Candidates approximate what select_tests offers: CI reads
TESTS from its main checkout when it runs, which this replaces with TESTS at
the head's merge base with main, plus the test files the sample adds. PR
samples carry the raw PR body.
Samples are only written once, so rebuilding into an existing DIR keeps them.
PR and head samples carry the PR's title and body as of the build, which may
have been edited after the failure; revert and neutral samples carry the
landed commit message.

A PR failure is attributed to the PR only if, before the PR landed, the same
test did not fail on another author's PR, or on main within four days of the
failure or of the PR's merge base. PR
truths that reproduced (same test failing in two build environments, or on
two heads of the PR) are kept in `truth_strong`. A one-off is not necessarily a flake:
often only one environment runs the test, or the author fixed it in the next
push.
"""

from __future__ import annotations

import argparse
import ast
import collections
import concurrent.futures as cf
import datetime as dt
import json
import random
import re
import subprocess
import time
from pathlib import Path

from tools.testing.llm_td.eval import REPO, write_json
from tools.testing.llm_td.eval.clickhouse import query
from tools.testing.llm_td.eval.score import job_runs_td
from tools.testing.llm_td.select_tests import changed_files, diffstat_and_diff


KINDS = ("pr", "head", "revert", "neutral")
MAIN_WINDOW = dt.timedelta(days=4)
# Main failures this far before --start are loaded, so PRs whose merge base is
# older than --start still get the main-breakage check.
MAIN_LOOKBACK = dt.timedelta(days=30)
FLAKY_WINDOW = dt.timedelta(days=14)
NEVER = dt.datetime.max.replace(tzinfo=dt.timezone.utc)
SKIP_JOBS = ("rerun_disabled_tests", "mem_leak_check")


def git(*args: str, check: bool = True) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=check
    ).stdout


def ts(s: str) -> dt.datetime:
    """ClickHouse DateTimes are UTC."""
    return dt.datetime.fromisoformat(s[:19]).replace(tzinfo=dt.timezone.utc)


def commit_time(sha: str) -> dt.datetime:
    return dt.datetime.fromtimestamp(
        int(git("log", "-1", "--format=%ct", sha)), tz=dt.timezone.utc
    )


def fetch(shas: list[str]) -> None:
    for i in range(0, len(shas), 100):
        p = subprocess.run(
            ["git", "fetch", "-q", "origin", *shas[i : i + 100]],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        if p.returncode:
            print(
                f"git fetch of {len(shas[i : i + 100])} shas failed: {p.stderr.strip()[-500:]}",
                flush=True,
            )


def has_commit(sha: str) -> bool:
    if subprocess.run(
        ["git", "cat-file", "-e", f"{sha}^{{commit}}"], cwd=REPO, capture_output=True
    ).returncode:
        print(f"skipping {sha}: not fetched", flush=True)
        return False
    return True


def is_main(branch: str) -> bool:
    """pull_failures prefixes a fork's branches with its repository, so a fork's main is not main."""
    return branch == "main" or branch.startswith("trunk/")


def pull_failures(start: dt.date, end: dt.date) -> list[dict]:
    """Every failing (job, test) in pytorch/pytorch CI, two days per query to stay under ClickHouse limits."""
    rows: list[dict] = []
    d = start
    while d < end:
        n = d + dt.timedelta(days=2)
        rows += query(f"""
            WITH f AS (
              SELECT job_id, replaceRegexpOne(file, '\\.py$', '') tf, classname, name, min(time_inserted) t
              FROM tests.all_test_runs
              WHERE time_inserted >= '{d}' AND time_inserted < '{n}' AND (failure_count > 0 OR error_count > 0)
              GROUP BY job_id, tf, classname, name)
            , j AS (
              SELECT id, run_id, head_sha, head_branch, workflow_name, name FROM default.workflow_job
              WHERE id IN (SELECT job_id FROM f) AND repository_full_name = 'pytorch/pytorch')
            SELECT f.job_id job_id, tf, classname, f.name test, toString(t) t, j.head_sha sha,
                   if(r.hr IN ('', 'pytorch/pytorch'), j.head_branch, concat(r.hr, ':', j.head_branch)) br,
                   j.workflow_name wf, j.name job
            FROM f JOIN j ON j.id = f.job_id
            LEFT JOIN (
              SELECT id, any(head_repository.full_name) hr FROM default.workflow_run
              WHERE id IN (SELECT run_id FROM j) GROUP BY id) r
            ON r.id = j.run_id""")
        print(d, len(rows), flush=True)
        d = n
    return [r for r in rows if not any(s in r["job"] for s in SKIP_JOBS)]


def pr_of_shas(shas: list[str]) -> dict[str, int]:
    out = {}
    for i in range(0, len(shas), 1000):
        chunk = ",".join(f"'{s}'" for s in shas[i : i + 1000])
        for r in query(
            f"SELECT head_sha, any(number) pr FROM materialized_views.pr_by_sha WHERE head_sha IN ({chunk}) GROUP BY head_sha"
        ):
            out[r["head_sha"]] = int(r["pr"])
    return out


def pr_authors(prs: list[int]) -> dict[int, str]:
    out = {}
    for i in range(0, len(prs), 500):
        keys = ",".join(f"'pytorch/pytorch/{p}'" for p in prs[i : i + 500])
        for r in query(
            f"SELECT number, any(user.login) author FROM default.pull_request FINAL WHERE dynamoKey IN ({keys}) GROUP BY number"
        ):
            out[int(r["number"])] = r["author"]
    return out


def pr_caused_failures(
    rows: list[dict], start: dt.date, end: dt.date, landed: dict[int, dt.datetime]
) -> tuple[list[dict], dict]:
    """Split failures into main and PR runs and keep the PR failures in [start, end), in TD jobs, attributable to the PR.

    `rows` may extend past the window on both sides, so failures near its edges
    are still checked against main and other authors, within MAIN_WINDOW.
    Failures after the PR landed (`landed`, see landed_times) may be its own
    breakage and are not held against it, so its own failures from then on
    cannot be checked and are dropped. Unstable jobs, and jobs that passed on
    any attempt on the same head, are not truth, as in flaky_flags. Failures
    on branches with no known PR count against PR failures here like main's,
    but only main's are returned."""
    sha2pr = pr_of_shas(sorted({r["sha"] for r in rows if not is_main(r["br"])}))
    main, unmapped = collections.defaultdict(list), collections.defaultdict(list)
    pr_rows: list[dict] = []
    for r in rows:
        k = (r["tf"], r["classname"], r["test"])
        if is_main(r["br"]) or r["sha"] not in sha2pr:
            (main if is_main(r["br"]) else unmapped)[k].append(ts(r["t"]))
        else:
            pr_rows.append({**r, "pr": sha2pr[r["sha"]]})
    author = pr_authors(sorted({r["pr"] for r in pr_rows}))
    authors = collections.defaultdict(list)
    for r in pr_rows:
        authors[(r["tf"], r["classname"], r["test"])].append((ts(r["t"]), author.get(r["pr"], r["pr"])))  # fmt: skip
    lo, hi = ts(f"{start} 00:00:00"), ts(f"{end} 00:00:00")
    keep = []
    for r in pr_rows:
        k, t, cutoff = (r["tf"], r["classname"], r["test"]), ts(r["t"]), landed.get(r["pr"], NEVER)  # fmt: skip
        others = {who for wt, who in authors[k] if abs(wt - t) < MAIN_WINDOW and wt < cutoff} - {author.get(r["pr"], r["pr"])}  # fmt: skip
        if lo <= t < min(hi, cutoff) and "unstable" not in r["job"] and job_runs_td(r["job"], r["wf"]) and not others and not any(abs(t - m) < MAIN_WINDOW and m < cutoff for m in main.get(k, []) + unmapped.get(k, [])):  # fmt: skip
            keep.append(r)
    green = green_jobs(sorted({r["sha"] for r in keep}))
    return [r for r in keep if (r["sha"], r["wf"], r["job"]) not in green], main


def green_jobs(shas: list[str]) -> set[tuple[str, str, str]]:
    """The (head sha, workflow, job) that succeeded on any attempt."""
    out = set()
    for i in range(0, len(shas), 200):
        chunk = ",".join(f"'{s}'" for s in shas[i : i + 200])
        out |= {(j["head_sha"], j["workflow_name"], j["name"]) for j in query(f"""
            SELECT DISTINCT head_sha, workflow_name, name FROM default.workflow_job FINAL
            WHERE head_sha IN ({chunk}) AND repository_full_name = 'pytorch/pytorch' AND conclusion = 'success'""")}  # fmt: skip
    return out


def ci_bases(heads: dict[str, int]) -> dict[str, list[str]]:
    """Head sha -> the base shas of PR `heads[sha]` that the head's pull_request(_target) runs recorded.

    llm-td.yml passes the base.sha of the head's push event to select_tests.
    A run records the base as of its latest webhook, which can be a base
    pushed after the head; see pr_bases. Heads with no such run, such as a
    fork's, are left out."""
    shas, out = sorted(heads), {}
    for i in range(0, len(shas), 200):
        chunk = ",".join(f"'{s}'" for s in shas[i : i + 200])
        sql = f"""
            SELECT head_sha, tupleElement(p, 1) pr, groupUniqArray(tupleElement(p, 2)) bases
            FROM default.workflow_run ARRAY JOIN arrayZip(pull_requests.number, pull_requests.base.sha) AS p
            WHERE head_sha IN ({chunk}) AND event IN ('pull_request', 'pull_request_target')
              AND repository.full_name = 'pytorch/pytorch'
            GROUP BY head_sha, pr"""
        for r in query(sql):
            if int(r["pr"]) == heads[r["head_sha"]]:
                out[r["head_sha"]] = sorted(r["bases"])
    return out


def pr_bases(sha: str, bases: list[str]) -> dict[str, str] | None:
    """The `base`, `merge_base` and `main_base` (the merge base with main) of head `sha`, given the CI bases ci_bases recorded.

    The base pushed with the head is in its history, and a base pushed later
    has a merge base with it no newer than that, so the base with the newest
    merge base is the one CI saw. Falls back to diffing against main_base,
    as for a fork's PR, when no base is recorded or fetched; None if `sha`
    has no merge base with main."""
    main_base = git("merge-base", "origin/main", sha, check=False).strip()
    if not main_base:
        return None
    by_mb: dict[str, str] = {}
    for b in bases:
        if mb := git("merge-base", b, sha, check=False).strip():
            by_mb.setdefault(mb, b)
    if not by_mb:
        return {"base": main_base, "merge_base": main_base, "main_base": main_base, "base_source": "main"}  # fmt: skip
    mb = git("merge-base", "--independent", *by_mb).split()[0] if len(by_mb) > 1 else next(iter(by_mb))  # fmt: skip
    return {"base": by_mb[mb], "merge_base": mb, "main_base": main_base, "base_source": "ci"}  # fmt: skip


def candidates_at(
    main: str, head: str | None = None, base: str | None = None
) -> list[str]:
    """The candidates select_tests offers for `head` diffed against `base` (default `main`) when main is at `main`:
    discover_tests.TESTS at `main`, without importing that tree, plus the test
    files `head` adds that `main`'s BLOCKLISTED_PATTERNS do not exclude.

    TESTS's arguments are list literals, or names of module-level lists such
    as BLOCKLISTED_PATTERNS (older trees inline it)."""
    tree = ast.parse(git("show", f"{main}:tools/testing/discover_tests.py"))
    assigned = {
        n.targets[0].id: n.value
        for n in tree.body
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
    }
    kw: dict[str | None, list] = {}
    for k in assigned["TESTS"].keywords:  # type: ignore[attr-defined]
        v = assigned[k.value.id] if isinstance(k.value, ast.Name) else k.value
        if isinstance(v, ast.List):
            kw[k.arg] = [e.value for e in v.elts if isinstance(e, ast.Constant)]
    patterns = tuple(kw.get("blocklisted_patterns", []))
    out = set(kw.get("extra_tests", []))
    for f in git("ls-tree", "-r", "--name-only", main, "test/").splitlines():
        t = f.removeprefix("test/").removesuffix(".py")
        if f.rsplit("/", 1)[-1].startswith("test_") and f.endswith(".py"):
            if not t.startswith(patterns) and t not in kw.get("blocklisted_tests", []):
                out.add(t)
    if head is not None:
        # As select_tests.select does.
        added = git(
            "diff", "--name-only", "--no-renames", "--diff-filter=A", base or main, head
        )
        for f in added.splitlines():
            t = f.removeprefix("test/").removesuffix(".py")
            if re.match(r"test/([\w/]+/)?test_\w+\.py$", f) and not t.startswith(
                patterns
            ):
                out.add(t)
    return sorted(out)


def strip_trailers(msg: str) -> str:
    pat = r"^(Pull Request resolved|Approved by|ghstack-source-id|Pull-Request|Differential Revision|Co-authored-by|Co-Authored-By|Fixes|X-link):"
    return "\n".join(
        line for line in msg.splitlines() if not re.match(pat, line)
    ).strip()


def pr_text(pr: int, tries: int = 4) -> tuple[str, str]:
    """The PR's title and raw body, as select_tests reads them from the event; raises CalledProcessError if gh keeps failing (e.g. rate limited)."""
    cmd = ["gh", "api", f"repos/pytorch/pytorch/pulls/{pr}", "--jq", "{title:.title, body:.body}"]  # fmt: skip
    for i in range(tries):
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode == 0:
            try:
                d = json.loads(p.stdout)
                return d["title"], d["body"] or ""
            except (ValueError, KeyError, TypeError):
                pass
        if i + 1 == tries:
            raise subprocess.CalledProcessError(p.returncode, cmd, p.stdout, p.stderr)
        time.sleep(10 * 2**i)
    raise ValueError(f"tries must be positive, got {tries}")


def write_sample(
    out: Path,
    sid: str,
    kind: str,
    sha: str,
    base: str,
    title: str,
    body: str,
    merge_base: str | None = None,
    main_base: str | None = None,
    **extra,
) -> None:
    """Write sample `sid` for `sha` diffed against `merge_base` (default `base`), with main at `main_base` (default `merge_base`)."""
    path = out / f"{sid}.json"
    if path.exists():
        return
    merge_base = merge_base or base
    diffstat, diff = diffstat_and_diff(merge_base, sha)
    write_json(path, {
        "id": sid, "kind": kind, "sha": sha, "base": base, "merge_base": merge_base, "title": title, "body": body,
        "diff": diff, "diffstat": diffstat, "changed_files": changed_files(merge_base, sha),
        "candidates": candidates_at(main_base or merge_base, sha, merge_base),
        **extra,
    })  # fmt: skip


def pr_samples(
    keep: list[dict],
    main: dict,
    out: Path,
    jobs: int,
    start: dt.date,
    landed: dict[int, dt.datetime],
) -> None:
    envs, heads = collections.defaultdict(set), collections.defaultdict(set)
    first: dict[int, dict] = {}
    for r in keep:
        k = (r["pr"], r["tf"], r["classname"], r["test"])
        envs[k].add(r["job"].split(" / ")[0])
        heads[k].add(r["sha"])
        if r["pr"] not in first or r["t"] < first[r["pr"]]["t"]:
            first[r["pr"]] = r
    reproduced = {k for k in envs if len(envs[k]) >= 2 or len(heads[k]) >= 2}
    first_rows = collections.defaultdict(list)
    for r in keep:
        if r["sha"] == first[r["pr"]]["sha"]:
            first_rows[r["pr"]].append(r)
    fetch(sorted({f["sha"] for f in first.values()}))
    ci = ci_bases({f["sha"]: pr for pr, f in first.items()})
    sources: list[str] = []

    def go(pr: int) -> None:
        sha = first[pr]["sha"]
        if not has_commit(sha):
            return
        bases = pr_bases(sha, ci.get(sha, []))
        if bases is None:
            return
        base_t = commit_time(bases["main_base"])
        if base_t - MAIN_WINDOW < ts(f"{start - MAIN_LOOKBACK} 00:00:00"):
            print(f"skipping PR {pr}: merge base before the loaded main failures", flush=True)  # fmt: skip
            return
        # A test broken on main around the merge base, before this PR landed, is not its fault.
        cutoff = landed.get(pr, NEVER)
        by, strong = collections.defaultdict(set), set()
        for r in first_rows[pr]:
            k = (r["tf"], r["classname"], r["test"])
            if not any(
                abs(base_t - m) < MAIN_WINDOW and m < cutoff for m in main.get(k, [])
            ):
                by[r["tf"]].add((r["wf"], r["job"]))
                if (pr, *k) in reproduced:
                    strong.add(r["tf"])
        truth = sorted(by)
        if truth:
            try:
                title, body = pr_text(pr)
                write_sample(out, f"pr_{pr}", "pr", sha, title=title, body=body, **bases, pr=pr, date=first[pr]["t"][:19], truth=truth,
                             truth_strong=[tf for tf in truth if tf in strong],
                             truth_jobs={tf: sorted(by[tf]) for tf in truth})  # fmt: skip
                sources.append(bases["base_source"])
            except subprocess.CalledProcessError as e:
                print(f"skipping PR {pr}: {e}", flush=True)

    with cf.ThreadPoolExecutor(jobs) as ex:
        list(ex.map(go, sorted(first)))
    print(f"{sources.count('main')} of {len(sources)} PR samples have no CI base and are diffed against main", flush=True)  # fmt: skip


# fmt: off


def landed_times(since: dt.date) -> dict[int, dt.datetime]:
    """PR number -> when it first landed on main, from the merge bot's `Pull Request resolved:` line, the last one of a body that quotes others."""
    out: dict[int, dt.datetime] = {}
    for c in git("log", "origin/main", f"--since={since} 00:00 +0000", "--format=%ct %b%x00").split("\0"):
        if (m := re.match(r"\s*(\d+) ", c)) and (prs := re.findall(r"Pull Request resolved: \S+/pull/(\d+)", c)):
            t, pr = dt.datetime.fromtimestamp(int(m.group(1)), tz=dt.timezone.utc), int(prs[-1])
            out[pr] = min(t, out.get(pr, t))
    return out


def flaky_flags(rows: list[dict], start: dt.date, end: dt.date) -> list[dict]:
    """Dr.CI's flaky signals for each failing (job, test) of `rows`, with hindsight on both sides.

    A failure is flaky if its job passed (on this attempt, after run_test.py's
    rerun of the file, or on a rerun of the job), the job is marked unstable,
    the test was disabled, or the same test failed on another author's PR, or
    on main, before the PR landed and within FLAKY_WINDOW. A PR's own breakage
    that landed and failed on main or on PRs based on it until its revert does
    not make it flaky, so a failure after the PR landed cannot be checked and
    counts as flaky."""
    # `rows` start at start - FLAKY_WINDOW, so look back a full window from there.
    w, lo = FLAKY_WINDOW, start - 2 * FLAKY_WINDOW
    others = pull_failures(lo, end + w)
    sha2pr = pr_of_shas(sorted({r["sha"] for r in others if not is_main(r["br"])}))
    author = pr_authors(sorted(set(sha2pr.values()) | {r["pr"] for r in rows}))
    landed = landed_times(lo)
    seen = collections.defaultdict(list)
    for r in others:
        pr = sha2pr.get(r["sha"])
        who = "main" if is_main(r["br"]) or pr is None else author.get(pr, pr)
        seen[(r["tf"], r["classname"], r["test"])].append((ts(r["t"]), who))
    disabled = collections.defaultdict(list)
    sql = f"""
        SELECT toString(day) d, name FROM misc.disabled_tests_historical
        WHERE day >= '{lo}' AND day < '{end + w}'"""
    for d in query(sql):
        # "test_foo (__main__.TestBar)"
        test, _, cls = d["name"].removesuffix(")").partition(" (")
        disabled[(cls.rsplit(".", 1)[-1], test)].append(ts(d["d"] + " 00:00:00"))
    for r in rows:
        t, me, cutoff = ts(r["t"]), author.get(r["pr"], r["pr"]), landed.get(r["pr"], NEVER)
        near = {who for wt, who in seen[(r["tf"], r["classname"], r["test"])] if abs(wt - t) < w and wt < cutoff}
        was_disabled = any(abs(d - t) < w for d in disabled[(r["classname"], r["test"])])
        r["flaky"] = r["job_green"] or "unstable" in r["job"] or bool(near - {me}) or was_disabled or t >= cutoff
    return rows


def head_failures(prs: list[int], start: dt.date, end: dt.date) -> list[dict]:
    """Every failing (job, test) in a job that runs TD, on every head of `prs`, before `end`.

    Later heads are left out because flaky_flags only has other failures up to
    `end + FLAKY_WINDOW`."""
    numbers = ",".join(map(str, prs))
    sql = f"SELECT DISTINCT number, head_sha FROM materialized_views.pr_by_sha WHERE number IN ({numbers})"
    sha2pr = {r["head_sha"]: int(r["number"]) for r in query(sql)}
    shas = sorted(sha2pr)
    jobs = []
    for i in range(0, len(shas), 200):
        chunk = ",".join(f"'{s}'" for s in shas[i : i + 200])
        jobs += query(f"""
            SELECT id, head_sha, workflow_name, name, run_attempt, conclusion FROM default.workflow_job FINAL
            WHERE head_sha IN ({chunk}) AND repository_full_name = 'pytorch/pytorch'""")
    green = {(j["head_sha"], j["workflow_name"], j["name"]) for j in jobs if j["conclusion"] == "success"}
    by_id = {j["id"]: j for j in jobs if job_runs_td(j["name"], j["workflow_name"]) and not any(s in j["name"] for s in SKIP_JOBS)}
    ids, rows = [str(i) for i in by_id], []
    for i in range(0, len(ids), 2000):
        sql = f"""
            SELECT job_id, replaceRegexpOne(file, '\\.py$', '') tf, classname, name test, toString(min(time_inserted)) t
            FROM tests.all_test_runs
            WHERE time_inserted >= '{start - FLAKY_WINDOW}' AND time_inserted < '{end}' AND job_id IN ({",".join(ids[i : i + 2000])})
              AND (failure_count > 0 OR error_count > 0)
            GROUP BY job_id, tf, classname, test"""
        for r in query(sql):
            j = by_id[int(r["job_id"])]
            sha, wf, name = j["head_sha"], j["workflow_name"], j["name"]
            passed = (sha, wf, name) in green
            rows.append({**r, "sha": sha, "pr": sha2pr[sha], "wf": wf, "job": name, "job_green": passed})
    return rows


def head_samples(out: Path, jobs: int, start: dt.date, end: dt.date) -> None:
    """One sample per failing head of each pr_ sample's PR, truth = its non-flaky failures in TD jobs."""
    prs = {}
    for p in out.glob("pr_*.json"):
        s = json.loads(p.read_text())
        prs[s["pr"]] = s
    if not prs:
        raise SystemExit(f"--kinds head needs the pr_ samples in {out}; build --kinds pr first")
    by = collections.defaultdict(lambda: collections.defaultdict(set))
    for r in flaky_flags(head_failures(sorted(prs), start, end), start, end):
        if not r["flaky"]:
            by[(r["pr"], r["sha"])][r["tf"]].add((r["wf"], r["job"]))
    fetch(sorted({sha for _, sha in by}))
    ci = ci_bases({sha: pr for pr, sha in by})
    sources: list[str] = []

    def go(k: tuple[int, str]) -> None:
        (pr, sha), s = k, prs[k[0]]
        if not has_commit(sha):
            return
        if bases := pr_bases(sha, ci.get(sha, [])):
            truth_jobs = {tf: sorted(js) for tf, js in by[k].items()}
            try:
                write_sample(out, f"head_{pr}_{sha[:12]}", "head", sha, title=s["title"], body=s["body"], **bases, pr=pr,
                             truth=sorted(truth_jobs), truth_jobs=truth_jobs)
                sources.append(bases["base_source"])
            except subprocess.CalledProcessError as e:
                print(f"skipping head {sha}: {e}", flush=True)

    with cf.ThreadPoolExecutor(jobs) as ex:
        list(ex.map(go, by))
    print(f"{sources.count('main')} of {len(sources)} head samples have no CI base and are diffed against main", flush=True)
# fmt: on


def revert_truth_jobs(
    signal: set[str], sha: str, landed: dt.datetime, reverted: str
) -> dict[str, list[tuple[str, str]]]:
    """The main-branch (workflow, job)s that run TD each signal test file failed in between landing and the revert, on the main commits from `sha` up to the revert.

    `reverted` is when autorevert decided to revert, so the main commits that
    landed between that and the revert commit are left out: truth is only
    lost, never added."""
    files = ",".join(f"'{t}.py'" for t in signal)
    t0 = landed.strftime("%Y-%m-%d %H:%M:%S")
    with_change = git("rev-list", "--first-parent", "--ancestry-path", f"--until={reverted[:19]} +0000", "--invert-grep",
                      f"--grep=This reverts commit {sha}", f"{sha}^..origin/main").split()  # fmt: skip
    shas = ",".join(f"'{c}'" for c in with_change or [sha])
    failed = query(f"""
        SELECT DISTINCT replaceRegexpOne(file, '\\.py$', '') tf, j.workflow_name wf, j.name job FROM tests.all_test_runs t
        JOIN (SELECT id, workflow_name, name FROM default.workflow_job
              WHERE created_at >= toDateTime('{t0}') - INTERVAL 1 HOUR
                AND created_at <= parseDateTimeBestEffort('{reverted}') + INTERVAL 1 HOUR
                AND repository_full_name = 'pytorch/pytorch' AND (head_branch = 'main' OR head_branch LIKE 'trunk/%')
                AND head_sha IN ({shas})
                AND run_id IN (SELECT id FROM default.workflow_run WHERE head_repository.full_name = 'pytorch/pytorch'
                               AND created_at >= toDateTime('{t0}') - INTERVAL 1 HOUR)
                AND {" AND ".join(f"name NOT LIKE '%{s}%'" for s in SKIP_JOBS)}) j ON j.id = t.job_id
        WHERE time_inserted >= toDateTime('{t0}') AND time_inserted <= parseDateTimeBestEffort('{reverted}') + INTERVAL 2 HOUR
          AND file IN ({files}) AND (failure_count > 0 OR error_count > 0)""")
    truth_jobs: dict[str, set] = collections.defaultdict(set)
    for f in failed:
        if job_runs_td(f["job"], f["wf"]):
            truth_jobs[f["tf"]].add((f["wf"], f["job"]))
    return {k: sorted(v) for k, v in truth_jobs.items()}


def revert_samples(out: Path, jobs: int, start: dt.date, end: dt.date) -> None:
    sql = f"""
        SELECT toString(ts) reverted_at, commit_sha, source_signal_keys FROM misc.autorevert_events_v2
        WHERE repo = 'pytorch/pytorch' AND action = 'revert' AND dry_run = 0
          AND ts >= toDateTime('{start}') AND ts < toDateTime('{end}')
          AND arrayExists(k -> position(k, '::') > 0, source_signal_keys)
        ORDER BY ts"""
    rows: dict[str, dict] = {}
    for r in query(sql):
        rows.setdefault(r["commit_sha"], r)
    fetch(sorted(rows))

    def go(r: dict) -> None:
        sha = r["commit_sha"]
        if not has_commit(sha):
            return
        signal = {
            k.split("::")[0].removesuffix(".py")
            for k in r["source_signal_keys"]
            if "::" in k
        }
        try:
            truth_jobs = revert_truth_jobs(signal, sha, commit_time(sha), r["reverted_at"])  # fmt: skip
            if not truth_jobs:
                print(f"skipping revert {sha}: no main job that runs TD failed on its signal tests", flush=True)  # fmt: skip
                return
            title, _, body = git("log", "-1", "--format=%B", sha).partition("\n")
            write_sample(out, f"rev_{sha[:12]}", "revert", sha, sha + "^", title, strip_trailers(body), date=r["reverted_at"],
                         truth=sorted(truth_jobs), truth_jobs=truth_jobs)  # fmt: skip
        except (subprocess.CalledProcessError, RuntimeError) as e:
            print(f"skipping revert {sha}: {e}", flush=True)

    with cf.ThreadPoolExecutor(jobs) as ex:
        list(ex.map(go, rows.values()))


def neutral_samples(out: Path, start: dt.date, end: dt.date, n: int) -> None:
    log = git(
        "log",
        "--first-parent",
        "--format=%H %s",
        f"--since={start} 00:00 +0000",
        f"--until={end} 00:00 +0000",
        "origin/main",
    ).splitlines()
    shas = [
        sha
        for sha, _, title in (line.partition(" ") for line in log)
        if not re.match(r"(Revert|Reland)", title)
    ]
    # Reverts (by message, by the autorevert bot's record, or with a rev_
    # sample) and titles naming a revert or reland anywhere filter the sample
    # rather than the population, so a later revert only drops its id from the
    # draw instead of reshuffling it.
    reverted = set(
        re.findall(
            r"This reverts commit ([0-9a-f]{40})",
            git("log", "--format=%B", f"--since={start} 00:00 +0000", "origin/main"),
        )
    )
    reverted |= {
        r["commit_sha"]
        for r in query(f"""
            SELECT DISTINCT commit_sha FROM misc.autorevert_events_v2
            WHERE repo = 'pytorch/pytorch' AND action = 'revert' AND dry_run = 0 AND ts >= toDateTime('{start}')""")
    }
    reverted_ids = {p.stem.removeprefix("rev_") for p in out.glob("rev_*.json")}
    for sha in random.Random(0).sample(shas, min(n, len(shas))):
        title, _, body = git("log", "-1", "--format=%B", sha).partition("\n")
        if sha in reverted or sha[:12] in reverted_ids or re.search(r"\b(revert|reland)\b", title, re.IGNORECASE):  # fmt: skip
            continue
        try:
            write_sample(out, f"neu_{sha[:12]}", "neutral", sha, sha + "^", title, strip_trailers(body), truth=[])  # fmt: skip
        except subprocess.CalledProcessError as e:
            print(f"skipping neutral {sha}: {e}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--start", type=dt.date.fromisoformat, required=True)
    p.add_argument("--end", type=dt.date.fromisoformat, required=True)
    p.add_argument("--neutral", type=int, default=200)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--kinds", nargs="+", choices=KINDS, default=KINDS)
    args = p.parse_args()
    out = args.data / "samples"
    out.mkdir(parents=True, exist_ok=True)
    git("fetch", "-q", "origin", "+refs/heads/main:refs/remotes/origin/main")
    if "pr" in args.kinds:
        rows = pull_failures(args.start - MAIN_LOOKBACK, args.end + MAIN_WINDOW)
        landed = landed_times(args.start - MAIN_LOOKBACK)
        keep, main_fail = pr_caused_failures(rows, args.start, args.end, landed)
        pr_samples(keep, main_fail, out, args.jobs, args.start, landed)
    if "head" in args.kinds:
        head_samples(out, args.jobs, args.start, args.end)
    if "revert" in args.kinds:
        revert_samples(out, args.jobs, args.start, args.end)
    if "neutral" in args.kinds:
        neutral_samples(out, args.end - dt.timedelta(days=25), args.end, args.neutral)


if __name__ == "__main__":
    main()
