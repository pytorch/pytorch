"""Build the labelled eval set from CI history.

    python -m tools.testing.llm_td.eval.build_dataset --data DIR --start 2026-09-20 --end 2026-10-26

Four kinds of sample, each a JSON file under DIR/samples:

  pr_<N>      a PR whose own CI failed tests that main did not fail: its
              earliest head with a failure between --start and --end that the
              rules below, short of the two on the head's bases, leave as the
              PR's own. The ground truth is every (test file, CI config) of
              that head's own failures. These runs are censored: pull/trunk
              jobs already skip 75% of files through TD, so truth is a lower
              bound.
  head_<N>_<sha12>
              every head of a pr_ sample's PR with a failure that is the PR's
              own; the truth is those failures, so the pr_ sample's head gets
              the same truth here.
              `previous_failures` are the test files that failed, flaky or
              not, in a job that runs TD on another head of the PR before this
              head's first job was created: what TD's PreviouslyFailedInPR
              reads from the PR's pytest cache. It overstates that set, as a
              test that passed again in the same job leaves the cache.
  rev_<sha12> a commit the autorevert bot reverted between --start and --end
              because it broke a test on main, where every test runs. The
              truth is the test files of the reverting signal that failed on
              main between landing and the revert, on the commits that
              contain it, in every job that runs TD they failed in there. A
              revert with no such failure is skipped.
  neu_<sha12> an ordinary main commit from the 25 days before --end (--start
              is ignored), with no truth, used only to measure how much a
              selector runs on typical traffic. --neutral commits are drawn
              from those whose title does not start with Revert or Reland, and
              then the draw drops reverted commits (by a revert's message or
              the autorevert bot), those with a rev_ sample and titles naming
              a revert or reland anywhere, so fewer than --neutral come out.

<sha12> is the first 12 hex digits of the sample's commit.

Each truth file's `truth_jobs` lists the [workflow, job] it failed in.
Samples are diffed as select_tests diffs them: against `merge_base`, the merge
base of the head with `base`. For PR and head samples `base` is the PR base CI
passed to select_tests for that head (the PR below it for a ghstack PR), read
from the head's pull_request runs (see pr_bases), or the merge base with main
when no run recorded one (`base_source` says which). For revert and neutral samples it
is the parent commit. Candidates approximate what select_tests offers: CI reads
TESTS from its main checkout when it runs, which this replaces with TESTS at
the head's merge base with main (`main_base`), plus the test files the sample
adds. PR samples carry the raw PR body.
Samples are only written once, so rebuilding into an existing DIR keeps them.
PR and head samples carry the PR's title and body as of the build, which may
have been edited after the failure; revert and neutral samples carry the
landed commit message.

PR and head samples judge a failure on a PR head with one rule set,
History.why_dropped, with hindsight on both sides as Dr.CI does. A failure is
not the PR's own, and is dropped, if
  - it is before --start - WINDOW or from --end on, or after the PR landed,
    when its own breakage on main and on PRs based on it can no longer be told
    apart from someone else's;
  - its job does not run TD, is unstable, or passed on any attempt on the head;
  - the test was disabled within WINDOW of the failure;
  - the same test failed on main, on a branch with no known PR, or on another
    author's PR within WINDOW of the failure, before the PR landed;
  - the same test failed on main within WINDOW of the head's `main_base`,
    before the PR landed;
  - the same test failed on another PR's head that is below this head in a
    stack (see below), which is the breakage of a PR below, and the diff
    against the CI base does not contain it. In a GitHub-native stack that
    head is the CI base or one of the head's ancestors. ghstack bases each
    PR on a synthetic gh/USER/N/base commit with the tree of the PR below,
    and the lower head is not an ancestor, so there it is a head with the CI
    base's tree.
WINDOW is 14 days, Dr.CI's flaky window: truth is a lower bound anyway, so a
shorter window would buy a few more truths at the price of counting more
flakes and parallel breakages as the PR's. The builder prints how many
failures each rule dropped. PR truths that reproduced (same test failing in
two build environments, or on two heads of the PR) are kept in
`truth_strong`. A one-off is not necessarily a flake: often only one
environment runs the test, or the author fixed it in the next push.
"""

from __future__ import annotations

import argparse
import ast
import collections
import concurrent.futures as cf
import dataclasses
import datetime as dt
import json
import random
import re
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any, TYPE_CHECKING

from tools.testing.llm_td.eval import REPO, write_json
from tools.testing.llm_td.eval.clickhouse import query
from tools.testing.llm_td.eval.score import job_runs_td
from tools.testing.llm_td.select_tests import changed_files, diffstat_and_diff


if TYPE_CHECKING:
    from collections.abc import Callable


KINDS = ("pr", "head", "revert", "neutral")
WINDOW = dt.timedelta(days=14)
# Failures this far before --start are loaded: heads are judged from
# --start - WINDOW, against failures up to WINDOW earlier, and merge bases may be
# older still.
LOOKBACK = dt.timedelta(days=30)
NEVER = dt.datetime.max.replace(tzinfo=dt.timezone.utc)
SKIP_JOBS = ("rerun_disabled_tests", "mem_leak_check")
Disabled = dict[tuple[str, str], list[dt.datetime]]
Key = tuple[str, str, str]


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


def disabled_tests(lo: dt.date, hi: dt.date) -> Disabled:
    """(class, test) -> the days in [lo, hi) it was on the disabled list."""
    out: Disabled = collections.defaultdict(list)
    sql = f"""
        SELECT toString(day) d, name FROM misc.disabled_tests_historical
        WHERE day >= '{lo}' AND day < '{hi}'"""
    for d in query(sql):
        # "test_foo (__main__.TestBar)"
        test, _, cls = d["name"].removesuffix(")").partition(" (")
        out[(cls.rsplit(".", 1)[-1], test)].append(ts(d["d"] + " 00:00:00"))
    return out


def was_disabled(disabled: Disabled, r: dict) -> bool:
    t, days = ts(r["t"]), disabled.get((r["classname"], r["test"]), [])
    return any(abs(d - t) < WINDOW for d in days)


def is_ancestor(a: str, b: str) -> bool:
    """False also when `a` is not fetched, which an ancestor of a fetched `b` always is."""
    return not subprocess.run(["git", "merge-base", "--is-ancestor", a, b], cwd=REPO, capture_output=True).returncode  # fmt: skip


def tree_of(sha: str) -> str:
    """'' when `sha` is not fetched."""
    return git("rev-parse", "-q", "--verify", f"{sha}^{{tree}}", check=False).strip()


def below(other: str, base: str, head: str) -> bool:
    """Whether PR head `other` is below PR head `head`, whose CI base is `base`, in a stack (see the module docstring)."""
    if other == base or is_ancestor(other, head):
        return True
    tree = tree_of(base)
    return bool(tree) and tree_of(other) == tree


def key(r: dict) -> Key:
    return r["tf"], r["classname"], r["test"]


@dataclasses.dataclass
class History:
    """The test failures CI saw from --start - LOOKBACK to --end + WINDOW, and what tells whether a PR head's failure is the PR's own."""

    start: dt.date
    end: dt.date
    # PR head sha -> its failures, each with its "pr".
    heads: dict[str, list[dict]]
    # Every failure of a test: time, author ("main" for main and branches with no known PR), sha, PR.
    seen: dict[Key, list[tuple[dt.datetime, str, str, int | None]]]
    main: dict[Key, list[dt.datetime]]
    author: dict[int, str]
    landed: dict[int, dt.datetime]
    disabled: Disabled
    # Head sha -> the (workflow, job)s that passed on it; see load_green.
    green: dict[str, set[tuple[str, str]]] = dataclasses.field(default_factory=dict)

    @classmethod
    def load(cls, start: dt.date, end: dt.date) -> History:
        rows = pull_failures(start - LOOKBACK, end + WINDOW)
        sha2pr = pr_of_shas(sorted({r["sha"] for r in rows if not is_main(r["br"])}))
        author = pr_authors(sorted(set(sha2pr.values())))
        seen, main, heads = (
            collections.defaultdict(list),
            collections.defaultdict(list),
            collections.defaultdict(list),
        )
        for r in rows:
            pr = None if is_main(r["br"]) else sha2pr.get(r["sha"])
            who = "main" if pr is None else author.get(pr, str(pr))
            seen[key(r)].append((ts(r["t"]), who, r["sha"], pr))
            if is_main(r["br"]):
                main[key(r)].append(ts(r["t"]))
            elif pr is not None:
                heads[r["sha"]].append({**r, "pr": pr})
        return cls(start, end, dict(heads), seen, main, author, landed_times(start - LOOKBACK), disabled_tests(start - LOOKBACK, end + WINDOW))  # fmt: skip

    def load_green(self, shas: set[str]) -> None:
        todo = sorted(shas - self.green.keys())
        for sha in todo:
            self.green[sha] = set()
        for sha, wf, job in green_jobs(todo):
            self.green[sha].add((wf, job))

    def why_dropped(self, r: dict, base: str | None = None, base_t: dt.datetime | None = None) -> str | None:  # fmt: skip
        """The first rule (see the module docstring) that drops PR failure `r`, or None if it is the PR's own.

        The rules on the head's CI `base` and the time `base_t` of its merge base
        with main are skipped without them, and the job_green rule for a head
        load_green has not loaded."""
        t, cutoff = ts(r["t"]), self.landed.get(r["pr"], NEVER)
        me, k = self.author.get(r["pr"], str(r["pr"])), key(r)
        if not ts(f"{self.start - WINDOW} 00:00:00") <= t < ts(f"{self.end} 00:00:00"):
            return "window"
        if t >= cutoff:
            return "after_landing"
        if "unstable" in r["job"] or not job_runs_td(r["job"], r["wf"]):
            return "job"
        if (r["wf"], r["job"]) in self.green.get(r["sha"], ()):
            return "job_green"
        if was_disabled(self.disabled, r):
            return "disabled"
        if any(who != me and abs(wt - t) < WINDOW and wt < cutoff for wt, who, _, _ in self.seen.get(k, [])):  # fmt: skip
            return "other_failure"
        if base_t is not None and any(abs(base_t - m) < WINDOW and m < cutoff for m in self.main.get(k, [])):  # fmt: skip
            return "main_near_base"
        if base is not None and any(pr not in (None, r["pr"]) and below(sha, base, r["sha"]) for _, _, sha, pr in self.seen.get(k, [])):  # fmt: skip
            return "stacked"
        return None

    def other_heads(self, shas: set[str]) -> set[str]:
        """The other PRs' heads that failed a test one of PR heads `shas` failed, which the stacked rule compares to them."""
        return {o for sha in shas for r in self.heads.get(sha, []) for _, _, o, pr in self.seen.get(key(r), []) if pr not in (None, r["pr"])}  # fmt: skip

    def truth(self, sha: str, bases: dict[str, str], dropped: collections.Counter) -> list[dict] | None:  # fmt: skip
        """The failures on PR head `sha` that are the PR's own, counting the others in `dropped` by rule; None if its main_base is older than the loaded failures."""
        if sha not in self.green:
            raise AssertionError(f"load_green({sha!r}) first")
        base_t = commit_time(bases["main_base"])
        if base_t - WINDOW < ts(f"{self.start - LOOKBACK} 00:00:00"):
            print(f"skipping {sha}: merge base before the loaded failures", flush=True)
            return None
        kept = []
        for r in self.heads[sha]:
            if why := self.why_dropped(r, bases["base"], base_t):
                dropped[why] += 1
            else:
                kept.append(r)
        return kept


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
    """Head sha -> the base shas of PR `heads[sha]` that the head's pull_request runs recorded.

    llm-td.yml passes the base.sha of the head's push event to select_tests.
    pull_request_target runs record the base branch as head_sha, so they never
    match a head. A run records the base as of its latest webhook, which can be
    a base pushed after the head; see pr_bases. Heads with no such run, such
    as a fork's, are left out."""
    shas, out = sorted(heads), {}
    for i in range(0, len(shas), 200):
        chunk = ",".join(f"'{s}'" for s in shas[i : i + 200])
        sql = f"""
            SELECT head_sha, tupleElement(p, 1) pr, groupUniqArray(tupleElement(p, 2)) bases
            FROM default.workflow_run ARRAY JOIN arrayZip(pull_requests.number, pull_requests.base.sha) AS p
            WHERE head_sha IN ({chunk}) AND event = 'pull_request'
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
        # A later ghstack base B2 is a child of the pushed B1, so both merge-base at B1: B1 is the one CI saw.
        if (mb := git("merge-base", b, sha, check=False).strip()) and (mb not in by_mb or b == mb):  # fmt: skip
            by_mb[mb] = b
    if not by_mb:
        return {"base": main_base, "merge_base": main_base, "main_base": main_base, "base_source": "main"}  # fmt: skip
    # When the merge bases are truly independent (rare), --independent keeps them all and this picks one.
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
    """Write sample `sid` for `sha` diffed against `merge_base` (default `base`), with main at `main_base` (default `merge_base`), whose TESTS give the candidates."""
    path = out / f"{sid}.json"
    if path.exists():
        return
    merge_base = merge_base or base
    main_base = main_base or merge_base
    diffstat, diff = diffstat_and_diff(merge_base, sha)
    write_json(path, {
        "id": sid, "kind": kind, "sha": sha, "base": base, "merge_base": merge_base, "main_base": main_base, "title": title, "body": body,
        "diff": diff, "diffstat": diffstat, "changed_files": changed_files(merge_base, sha),
        "candidates": candidates_at(main_base, sha, merge_base),
        **extra,
    })  # fmt: skip


def run_each(kind: str, go: Callable[[Any], None], items: list, jobs: int) -> None:
    """Run `go` on every item in `jobs` threads, printing, counting and skipping the items on which it raises."""
    failed = []

    def guarded(x: Any) -> None:
        try:
            go(x)
        except Exception:
            print(f"skipping {kind} {x}: {traceback.format_exc()}", flush=True)
            failed.append(x)

    with cf.ThreadPoolExecutor(jobs) as ex:
        list(ex.map(guarded, items))
    if failed:
        print(f"{len(failed)} of {len(items)} {kind} samples raised and were skipped", flush=True)  # fmt: skip


def by_file(rows: list[dict]) -> dict[str, list[tuple[str, str]]]:
    out = collections.defaultdict(set)
    for r in rows:
        out[r["tf"]].add((r["wf"], r["job"]))
    return {tf: sorted(js) for tf, js in sorted(out.items())}


def print_dropped(kind: str, dropped: list[collections.Counter]) -> None:
    total = sum(dropped, collections.Counter())
    by_count = dict(sorted(total.items(), key=lambda x: (-x[1], x[0])))
    print(
        f"{kind} truth: dropped by rule over {len(dropped)} heads: {by_count}",
        flush=True,
    )


def pr_samples(h: History, out: Path, jobs: int) -> None:
    lo = ts(f"{h.start} 00:00:00")
    own = [r for rows in h.heads.values() for r in rows if ts(r["t"]) >= lo and h.why_dropped(r) is None]  # fmt: skip
    h.load_green({r["sha"] for r in own})
    own = [r for r in own if h.why_dropped(r) is None]
    envs, heads = collections.defaultdict(set), collections.defaultdict(set)
    first: dict[int, dict] = {}
    for r in own:
        k = (r["pr"], *key(r))
        envs[k].add(r["job"].split(" / ")[0])
        heads[k].add(r["sha"])
        if r["pr"] not in first or r["t"] < first[r["pr"]]["t"]:
            first[r["pr"]] = r
    reproduced = {k for k in envs if len(envs[k]) >= 2 or len(heads[k]) >= 2}
    fetch(sorted({f["sha"] for f in first.values()}))
    fetch(sorted(h.other_heads({f["sha"] for f in first.values()})))
    ci = ci_bases({f["sha"]: pr for pr, f in first.items()})
    sources: list[str] = []
    dropped: list[collections.Counter] = []

    def go(pr: int) -> None:
        sha = first[pr]["sha"]
        if not has_commit(sha):
            return
        bases = pr_bases(sha, ci.get(sha, []))
        if bases is None:
            return
        dropped.append(c := collections.Counter())
        if kept := h.truth(sha, bases, c):
            truth_jobs = by_file(kept)
            strong = {r["tf"] for r in kept if (pr, *key(r)) in reproduced}
            try:
                title, body = pr_text(pr)
                write_sample(out, f"pr_{pr}", "pr", sha, title=title, body=body, **bases, pr=pr, date=first[pr]["t"][:19], truth=sorted(truth_jobs),
                             truth_strong=sorted(strong), truth_jobs=truth_jobs)  # fmt: skip
                sources.append(bases["base_source"])
            except subprocess.CalledProcessError as e:
                print(f"skipping PR {pr}: {e}", flush=True)

    run_each("PR", go, sorted(first), jobs)
    print_dropped("pr", dropped)
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


def head_created(shas: list[str]) -> dict[str, str]:
    """Each head's first workflow job creation time, roughly when CI's target-determination job reads the PR's previous failures."""
    out = {}
    for i in range(0, len(shas), 200):
        chunk = ",".join(f"'{s}'" for s in shas[i : i + 200])
        out |= {r["head_sha"]: r["t"] for r in query(f"""
            SELECT head_sha, toString(min(created_at)) t FROM default.workflow_job FINAL
            WHERE head_sha IN ({chunk}) AND repository_full_name = 'pytorch/pytorch' GROUP BY head_sha""")}
    return out


def head_samples(h: History, out: Path, jobs: int) -> None:
    """One sample per head of each pr_ sample's PR with failures that are the PR's own, truth = those failures."""
    prs = {}
    for p in out.glob("pr_*.json"):
        s = json.loads(p.read_text())
        prs[s["pr"]] = s
    if not prs:
        print(f"no head samples: there are no pr_ samples in {out}", flush=True)
        return
    shas = {sha for sha, rows in h.heads.items() if rows[0]["pr"] in prs}
    h.load_green(shas)
    shas = {sha for sha in shas if any(h.why_dropped(r) is None for r in h.heads[sha])}
    fetch(sorted(shas))
    fetch(sorted(h.other_heads(shas)))
    ci = ci_bases({sha: h.heads[sha][0]["pr"] for sha in shas})
    created = {sha: ts(t) for sha, t in head_created(sorted(shas)).items()}
    failed = collections.defaultdict(list)
    for other, rows in h.heads.items():
        for r in rows:
            if r["pr"] in prs and job_runs_td(r["job"], r["wf"]):
                failed[r["pr"]].append((ts(r["t"]), other, r["tf"]))
    sources: list[str] = []
    dropped: list[collections.Counter] = []

    def go(sha: str) -> None:
        pr = h.heads[sha][0]["pr"]
        if not has_commit(sha) or (bases := pr_bases(sha, ci.get(sha, []))) is None:
            return
        dropped.append(c := collections.Counter())
        if kept := h.truth(sha, bases, c):
            truth_jobs = by_file(kept)
            previous = sorted({tf for t, other, tf in failed[pr] if other != sha and sha in created and t < created[sha]})
            try:
                write_sample(out, f"head_{pr}_{sha[:12]}", "head", sha, title=prs[pr]["title"], body=prs[pr]["body"], **bases, pr=pr,
                             truth=sorted(truth_jobs), truth_jobs=truth_jobs, previous_failures=previous)
                sources.append(bases["base_source"])
            except subprocess.CalledProcessError as e:
                print(f"skipping head {sha}: {e}", flush=True)

    run_each("head", go, sorted(shas), jobs)
    print_dropped("head", dropped)
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

    def go(sha: str) -> None:
        r = rows[sha]
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

    run_each("revert", go, list(rows), jobs)


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

    def go(sha: str) -> None:
        title, _, body = git("log", "-1", "--format=%B", sha).partition("\n")
        if sha in reverted or sha[:12] in reverted_ids or re.search(r"\b(revert|reland)\b", title, re.IGNORECASE):  # fmt: skip
            return
        try:
            write_sample(out, f"neu_{sha[:12]}", "neutral", sha, sha + "^", title, strip_trailers(body), truth=[])  # fmt: skip
        except subprocess.CalledProcessError as e:
            print(f"skipping neutral {sha}: {e}", flush=True)

    run_each("neutral", go, random.Random(0).sample(shas, min(n, len(shas))), 1)


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
    if {"pr", "head"} & set(args.kinds):
        h = History.load(args.start, args.end)
        if "pr" in args.kinds:
            pr_samples(h, out, args.jobs)
        if "head" in args.kinds:
            head_samples(h, out, args.jobs)
    if "revert" in args.kinds:
        revert_samples(out, args.jobs, args.start, args.end)
    if "neutral" in args.kinds:
        neutral_samples(out, args.end - dt.timedelta(days=25), args.end, args.neutral)


if __name__ == "__main__":
    main()
