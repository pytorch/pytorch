"""Build the labelled eval set from CI history.

    python -m tools.testing.llm_selector.eval.build_dataset --data DIR --start 2026-10-01 --end 2026-10-26 --kinds pr neutral
    python -m tools.testing.llm_selector.eval.build_dataset --data DIR --start 2026-10-01 --end 2026-10-26 --kinds head
    python -m tools.testing.llm_selector.eval.build_dataset --data DIR --start 2026-04-01 --end 2026-10-26 --kinds revert

eval/ids holds sorted sample ids (file names under DIR/samples without .json).
The report scores eval.txt, head.txt and their subsets; dev.txt and pilot.txt
hold samples read while writing the prompt and are not scored:
  eval.txt    the pr_ and rev_ samples dated (a PR's failure, a revert's
              revert time) inside the pr and revert builds above, less dev.txt's
              PRs, and neu_ samples drawn from main commits of that pr window
  neutral.txt the neu_ ids of eval.txt
  head.txt    the head_ samples with truth of eval.txt's PRs
  agentic.txt the eval.txt samples the agentic rerun was made on, recorded with
              that run, not cut by date
  dev.txt     pr_ samples dated 2026-09-20 to 2026-09-30 and rev_ samples
              reverted 2025-11-11 to 2026-03-31; test_checked_in_ids checks that
              no id is in both dev.txt and eval.txt
  pilot.txt   the 10 neu_ samples of the pilot prompt run, not cut by date; 48 of
              its 50 pr_ and rev_ samples are in dev.txt, the other 2 in no list
The pr_, rev_ and head_ ids come from one rebuild with the builder at tag
llmtd-r17-pre-r39, which rebuilt only the PRs and reverts already in eval.txt;
the neu_, dev.txt and pilot.txt samples are carried over from earlier builds.
So a rebuild gives similar, not identical, lists.

Four kinds of sample, each a JSON file under DIR/samples:

  pr_<N>      a PR whose own CI failed tests that main did not fail: its
              earliest head with a failure between --start and --end that the
              rules below, short of the two on the head's bases, leave as the
              PR's own. The ground truth is every (test file, CI config) of
              that head's own failures under all the rules, and `date` is the
              earliest of them from --start on. When the rules on the head's
              bases drop every one from --start on, or its main_base is older
              than the loaded failures, the PR has no pr_ sample and so no
              head samples; no later head stands in. These runs are censored:
              pull/trunk jobs already skip 75% of files through TD, so truth
              is a lower bound.
  head_<N>_<sha12>
              every head of a pr_ sample's PR with a failure that is the PR's
              own; the truth is those failures, so the pr_ sample's head gets
              the same `truth` here. Only pr_ samples carry `truth_strong`,
              which score.py uses by default for pr_ samples alone, so pr_ and
              head_ metrics are comparable only under --all-truth.
              `previous_failures` are the test files with a test that failed,
              and did not pass again in the same job, in a job that runs TD on
              another head of the PR before this head's first job was created,
              whatever the other rules below say: what TD's
              PreviouslyFailedInPR reads from the PR's pytest cache, which a
              test that passes again leaves.
  rev_<sha12> a commit the autorevert bot reverted between --start and --end
              because it broke a test on main, where every test runs. The
              truth is the test files whose signal tests (the reverting
              signal's file::test keys) failed on main between landing and
              the revert, on the commits before the revert that contain it
              (up to when autorevert decided, if earlier), with every job that
              runs TD they failed in there, short of the job, job_green and
              passed_in_job rules below. A file::class::test key matches
              only that class's test; a file::test key, as autorevert writes
              them, matches the name in any class of the file. A revert with
              no such failure is skipped.
  neu_<sha12> an ordinary main commit from the 25 days before --end (--start
              is ignored), with no truth, used only to measure how much a
              selector runs on typical traffic. --neutral commits are drawn
              from those whose title does not start with Revert or Reland, and
              then the draw drops reverted commits (by a revert's message or
              the autorevert bot) and titles naming a revert or reland
              anywhere, so fewer than --neutral come out.

<sha12> is the first 12 hex digits of the sample's commit.

Each truth file's `truth_jobs` lists the [workflow, job] it failed in.
Samples are diffed as select_tests diffs them: against `merge_base`, the merge
base of the head with `base`. For PR and head samples `base` is the PR base CI
passed to select_tests for that head (the PR below it for a ghstack PR), read
from the head's pull_request runs (see pr_bases; of several main tips with
the same merge base, which one is arbitrary, but `merge_base` is not), or the merge base with main
when no run recorded one (`base_source` says which). For revert and neutral samples it
is the parent commit. Candidates approximate what select_tests offers: CI reads
TESTS from its main checkout when it runs, which this replaces with TESTS at
the head's merge base with main (`main_base`), plus the test files the sample
adds. PR samples carry the raw PR body.
Samples are only written once, so rebuilding into an existing DIR keeps them.
Every sample records the BUILDER that wrote it, the window it was drawn
from (--start and --end; for neutral samples the 25 days before --end) and a
hash of the job classification score.ci_jobs reads from this checkout's
.github/workflows, which the job rule, previous_failures and revert truth
follow, as `builder`, `window` and `workflows`, and a kind refuses to run on a
DIR whose samples of that kind (and, for head, its pr_ samples) have another
builder, window or workflows, or none. PR truth reads failures and disables up to --end + WINDOW, so pr
and head refuse to run before that has passed.
PR and head samples carry the PR's title and body as of the build, which may
have been edited after the failure; revert and neutral samples carry the
landed commit message without what the merge bot added (see commit_text).

PR and head samples judge a failure on a PR head with one rule set,
History.why_dropped, with hindsight on both sides as Dr.CI does. A failure is
not the PR's own, and is dropped, if
  - it is before --start - WINDOW or from --end on, or after the PR landed,
    when its own breakage on main and on PRs based on it can no longer be told
    apart from someone else's;
  - its job does not run TD, is unstable, or passed on any attempt on the head;
  - the test passed again in the same job, as when run_test.py reruns it in
    a new process;
  - the test was disabled within WINDOW of the failure, matched by exact
    (class, test): a parametrized or device-suffixed variant that
    check_if_enable would also skip does not count;
  - the same test failed on main, on a branch with no known PR, or on another
    author's PR within WINDOW of the failure, before the PR landed;
  - the same test failed on main within WINDOW of the head's `main_base`,
    before the PR landed;
  - the same test failed, without passing again in its job, on another PR's
    head whose change `merge_base` already has, so the head's diff leaves it
    out and the failure is the breakage of a PR below it in a stack (or of
    one that landed before it). The author's own PRs count until they land,
    back to the loaded failures (--start - LOOKBACK), since a stack's lower PRs
    can sit for weeks; another author's count only within WINDOW, so a
    long-landed PR that once broke the test does not drop the failure. Within WINDOW and before the PR landed,
    other_failure already drops it, so for another author this rule only adds
    lower heads that failed after the PR landed, which other_failure leaves
    out because there it cannot tell this PR's breakage from theirs. The
    lower head's failure must not have passed again in its job and must pass
    the job and job_green rules; the other rules are not applied to it.
    In a GitHub-native stack that head is an ancestor of `merge_base`.
    ghstack bases each PR on a synthetic gh/USER/N/base commit with the tree
    of the PR below, so no lower head is an ancestor; there the lower head's
    merge base with main is in `merge_base`'s history, so a lower change that
    deletes what main added later does not count, and its diff against that
    merge base reverse-applies to `merge_base`
    with one line of context, for every PR down the stack, and none of its
    hunks that only delete lines applies forward with full context (the
    reverse of such a hunk is just context, which matches almost anywhere).
    A lower PR whose
    changes a PR between them rewrote is missed, and keeps its failures. So
    does a lower head the diff contains, as for a PR that targets main but
    branches off another PR, or one diffed against main for want of a CI
    base.
WINDOW is 14 days, Dr.CI's flaky window: truth is a lower bound anyway, so a
shorter window would buy a few more truths at the price of counting more
flakes and parallel breakages as the PR's. The builder prints how many
failures each rule dropped. PR truths that reproduced (same test failing in
two build environments, the job names less their last ` / ` part, or on
two heads of the PR, also heads from before --start that the window rule keeps, each failure the PR's own under all the
rules with its own head's bases) are kept in `truth_strong`. A one-off is not necessarily a flake: often only one
environment runs the test, or the author fixed it in the next push.
"""

from __future__ import annotations

import argparse
import ast
import collections
import concurrent.futures as cf
import dataclasses
import datetime as dt
import functools
import hashlib
import json
import os
import random
import re
import subprocess
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, TYPE_CHECKING

from tools.testing.llm_selector.eval import REPO, score, write_json
from tools.testing.llm_selector.eval.clickhouse import query
from tools.testing.llm_selector.eval.score import has_head, job_runs_td
from tools.testing.llm_selector.select_tests import (
    changed_files,
    diffstat_and_diff,
    git_paths,
)


if TYPE_CHECKING:
    from collections.abc import Callable


KINDS = ("pr", "head", "revert", "neutral")
# Recorded on every sample; bump it when a change alters what a sample holds.
BUILDER = 2
WINDOW = dt.timedelta(days=14)
# Failures this far before --start are loaded: heads are judged from
# --start - WINDOW, against failures up to WINDOW earlier, merge bases may be
# older still, and History.stacked sees the author's own lower heads only this
# far back.
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
    """Fetch `shas` 100 at a time; a batch that fails, as one sha the remote no longer has makes it, is retried one sha at a time."""

    def run(batch: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "fetch", "-q", "origin", *batch], cwd=REPO, capture_output=True, text=True)  # fmt: skip

    for i in range(0, len(shas), 100):
        if not run(batch := shas[i : i + 100]).returncode:
            continue
        for sha in batch:
            if (p := run([sha])).returncode:
                print(
                    f"git fetch of {sha} failed: {p.stderr.strip()[-300:]}", flush=True
                )


def has_commit(sha: str) -> bool:
    if not has_head(sha):
        print(f"skipping {sha}: not fetched", flush=True)
        return False
    return True


def sql_str(s: str) -> str:
    return "'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'"


def query_in(sql: str, values: list[str], size: int) -> list[dict]:
    """The rows of `sql` with its {} filled by each `size` of `values`, quoted and comma-joined."""
    rows: list[dict] = []
    for i in range(0, len(values), size):
        rows += query(sql.format(",".join(sql_str(v) for v in values[i : i + size])))
    return rows


def is_main(branch: str) -> bool:
    """pull_failures prefixes a fork's branches with its repository, so a fork's main is not main."""
    return branch == "main" or branch.startswith("trunk/")


def pull_failures(start: dt.date, end: dt.date) -> list[dict]:
    """Every failing (job, test) in pytorch/pytorch CI, two days per query to stay under ClickHouse limits.

    `flaky` marks a test that also passed in the same job, as when
    run_test.py's rerun of it in a new process passes.

    These read tests.all_test_runs, which keeps about a year of runs, rather
    than ci.runs, which keeps 14 days and so would not reach the backtest
    window. The ci-metrics skill says not to use the legacy tests database;
    the rebuild on or after 2026-11-09 should check whether ci.runs now
    reaches the window and move both these and revert_truth_jobs to it, and
    an empty chunk, as a store no longer written would give, raises."""
    rows: list[dict] = []
    d = start
    while d < end:
        n = min(d + dt.timedelta(days=2), end)
        chunk = query(f"""
            WITH f AS (
              SELECT job_id, replaceRegexpOne(file, '\\\\.py$', '') tf, classname, name, min(time_inserted) t
              FROM tests.all_test_runs
              WHERE time_inserted >= '{d}' AND time_inserted < '{n}' AND (failure_count > 0 OR error_count > 0)
              GROUP BY job_id, tf, classname, name)
            , passed AS (
              SELECT DISTINCT job_id, replaceRegexpOne(file, '\\\\.py$', '') tf, classname, name FROM tests.all_test_runs
              WHERE time_inserted >= toDate('{d}') - 1 AND time_inserted < toDate('{n}') + 1 AND job_id IN (SELECT job_id FROM f)
                AND failure_count = 0 AND error_count = 0 AND skipped_count = 0)
            , j AS (
              SELECT id, run_id, head_sha, head_branch, workflow_name, name FROM default.workflow_job FINAL
              WHERE id IN (SELECT job_id FROM f) AND repository_full_name = 'pytorch/pytorch'
                AND {" AND ".join(f"name NOT LIKE '%{s}%'" for s in SKIP_JOBS)})
            SELECT f.job_id job_id, tf, classname, f.name test, toString(t) t, j.head_sha sha,
                   if(r.hr IN ('', 'pytorch/pytorch'), j.head_branch, concat(r.hr, ':', j.head_branch)) br,
                   j.workflow_name wf, j.name job, (f.job_id, f.tf, f.classname, f.name) IN (SELECT * FROM passed) flaky
            FROM f JOIN j ON j.id = f.job_id
            LEFT JOIN (
              SELECT id, any(head_repository.full_name) hr FROM default.workflow_run
              WHERE id IN (SELECT run_id FROM j) GROUP BY id) r
            ON r.id = j.run_id""")
        if not chunk:
            raise RuntimeError(f"tests.all_test_runs has no failing test from {d} to {n}; check that it is still written (see pull_failures)")  # fmt: skip
        rows += chunk
        print(d, len(rows), flush=True)
        d = n
    return rows


def check_written(days: list[dt.date]) -> None:
    """Raise unless tests.all_test_runs has a failing test on each of `days`, so a store no longer written, or one whose retention no longer reaches them, is not read as no truth."""
    sql = " UNION ALL ".join(f"SELECT '{d}' d FROM (SELECT 1 FROM tests.all_test_runs WHERE time_inserted >= '{d}' AND time_inserted < '{d + dt.timedelta(days=1)}' AND (failure_count > 0 OR error_count > 0) LIMIT 1)" for d in days)  # fmt: skip
    if missing := sorted({str(d) for d in days} - {r["d"] for r in query(sql)}):
        raise RuntimeError(f"tests.all_test_runs has no failing test on {', '.join(missing)}; check that it is still written and kept that long (see pull_failures)")  # fmt: skip


def pr_of_shas(shas: list[str]) -> dict[str, int]:
    sql = "SELECT head_sha, any(number) pr FROM materialized_views.pr_by_sha WHERE head_sha IN ({}) GROUP BY head_sha"  # fmt: skip
    return {r["head_sha"]: int(r["pr"]) for r in query_in(sql, shas, 1000)}


def pr_authors(prs: list[int]) -> dict[int, str]:
    sql = "SELECT number, any(user.login) author FROM default.pull_request FINAL WHERE dynamoKey IN ({}) GROUP BY number"  # fmt: skip
    keys = [f"pytorch/pytorch/{p}" for p in prs]
    return {int(r["number"]): r["author"] for r in query_in(sql, keys, 500)}


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


@functools.cache
def is_ancestor(a: str, b: str) -> bool:
    """False also when `a` is not fetched, which an ancestor of a fetched `b` always is."""
    return not subprocess.run(["git", "merge-base", "--is-ancestor", a, b], cwd=REPO, capture_output=True).returncode  # fmt: skip


@functools.cache
def own_change(sha: str) -> bytes:
    """PR head `sha`'s diff against its merge base with main, as bytes since files need not be UTF-8; empty when `sha` is not fetched."""
    if not (main_base := git("merge-base", "origin/main", sha, check=False).strip()):
        return b""
    # Plumbing-stable output for `git apply`, whatever the user's diff config; GIT_DIFF_OPTS would override -U.
    cmd = ["git", "diff", "--no-ext-diff", "--no-textconv", "--no-color", "--src-prefix=a/", "--dst-prefix=b/", "--binary", "--full-index", "--no-renames", "-U3", "--inter-hunk-context=0", "--submodule=short", main_base, sha]  # fmt: skip
    env = {k: v for k, v in os.environ.items() if k != "GIT_DIFF_OPTS"}
    return subprocess.run(cmd, cwd=REPO, env=env, capture_output=True).stdout


@functools.cache
def below(other: str, merge_base: str) -> bool:
    """Whether `merge_base` already has PR head `other`'s change, so a diff against it leaves that change out (see the module docstring).

    That is when `other` is an ancestor, or its merge base with main is in
    `merge_base`'s history (else a deletion of what main added since
    `merge_base` would reverse-apply too) and its change reverse-applies to
    `merge_base` with one line of context, so edits two lines away still match,
    and none of its deletion-only hunks applies forward with full context."""
    if is_ancestor(other, merge_base):
        return True
    lower_base = git("merge-base", "origin/main", other, check=False).strip()
    if not lower_base or not is_ancestor(lower_base, merge_base) or not (patch := own_change(other)):  # fmt: skip
        return False
    with tempfile.TemporaryDirectory() as d:
        env = {**os.environ, "GIT_INDEX_FILE": os.path.join(d, "index")}

        def fails(*args: str, stdin: bytes | None = None) -> int:
            return subprocess.run(["git", *args], cwd=REPO, env=env, input=stdin, capture_output=True).returncode  # fmt: skip

        # The user's apply.* config, such as apply.ignoreWhitespace=change, would let a re-indent reverse-apply.
        apply = ("-c", "apply.ignoreWhitespace=no", "-c", "apply.whitespace=nowarn", "apply", "--cached", "--check", "--whitespace=nowarn")  # fmt: skip
        if fails("read-tree", merge_base) or fails(*apply, "-R", "-C1", stdin=patch):
            return False
        return all(fails(*apply, stdin=h) for h in deletion_hunks(patch))


def deletion_hunks(patch: bytes) -> list[bytes]:
    """Each hunk of `patch` that only deletes lines from a file that stays, as a patch of its own."""
    out = []
    for f in re.split(rb"(?m)^(?=diff --git )", patch):
        header, *hunks = re.split(rb"(?m)^(?=@@ )", f)
        if re.search(rb"(?m)^(new|deleted) file mode ", header):
            continue
        for h in hunks:
            lines = h.splitlines()[1:]
            if any(x.startswith(b"-") for x in lines) and not any(x.startswith(b"+") for x in lines):  # fmt: skip
                out.append(header + h)
    return out


def key(r: dict) -> Key:
    return r["tf"], r["classname"], r["test"]


def job_dropped(r: dict, green: set[tuple[str, str]]) -> str | None:
    """The job or job_green rule, if one drops failure `r`, given the (workflow, job)s that passed on its head."""
    if "unstable" in r["job"] or not job_runs_td(r["job"], r["wf"]):
        return "job"
    if (r["wf"], r["job"]) in green:
        return "job_green"
    return None


@dataclasses.dataclass
class History:
    """The test failures CI saw from --start - LOOKBACK to --end + WINDOW, and what tells whether a PR head's failure is the PR's own."""

    start: dt.date
    end: dt.date
    # PR head sha -> its failures, each with its "pr".
    heads: dict[str, list[dict]]
    # (PR head sha, test) -> the head's failures of that test.
    head_fails: dict[tuple[str, Key], list[dict]]
    # Every failure of a test: time, author ("main" for main and branches with no known PR), sha, PR, flaky.
    seen: dict[Key, list[tuple[dt.datetime, str, str, int | None, bool]]]
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
        seen, main, heads, head_fails = (
            collections.defaultdict(list),
            collections.defaultdict(list),
            collections.defaultdict(list),
            collections.defaultdict(list),
        )
        for r in rows:
            # ClickHouse returns the flag as 0/1; a cache may hold it as a string.
            r["flaky"] = bool(int(r["flaky"]))
            pr = None if is_main(r["br"]) else sha2pr.get(r["sha"])
            who = "main" if pr is None else author.get(pr, str(pr))
            seen[key(r)].append((ts(r["t"]), who, r["sha"], pr, r["flaky"]))
            if is_main(r["br"]):
                main[key(r)].append(ts(r["t"]))
            elif pr is not None:
                heads[r["sha"]].append({**r, "pr": pr})
                head_fails[r["sha"], key(r)].append(heads[r["sha"]][-1])
        return cls(start, end, dict(heads), dict(head_fails), seen, main, author, landed_times(start - LOOKBACK), disabled_tests(start - LOOKBACK, end + WINDOW))  # fmt: skip

    def load_green(self, shas: set[str]) -> None:
        todo = sorted(shas - self.green.keys())
        for sha in todo:
            self.green[sha] = set()
        for sha, wf, job in green_jobs(todo):
            self.green[sha].add((wf, job))

    def why_dropped(self, r: dict, bases: dict[str, str] | None = None, base_t: dt.datetime | None = None) -> str | None:  # fmt: skip
        """The first rule (see the module docstring) that drops PR failure `r`, or None if it is the PR's own.

        The rules on the head's `bases` (see pr_bases) and the time `base_t` of its merge base
        with main are skipped without them, and the job_green rule for a head
        load_green has not loaded."""
        t, cutoff = ts(r["t"]), self.landed.get(r["pr"], NEVER)
        me, k = self.author.get(r["pr"], str(r["pr"])), key(r)
        if not ts(f"{self.start - WINDOW} 00:00:00") <= t < ts(f"{self.end} 00:00:00"):
            return "window"
        if t >= cutoff:
            return "after_landing"
        if why := self.job_dropped(r):
            return why
        if r["flaky"]:
            return "passed_in_job"
        if was_disabled(self.disabled, r):
            return "disabled"
        if any(who != me and abs(wt - t) < WINDOW and wt < cutoff for wt, who, _, _, _ in self.seen.get(k, [])):  # fmt: skip
            return "other_failure"
        if base_t is not None and any(abs(base_t - m) < WINDOW and m < cutoff for m in self.main.get(k, [])):  # fmt: skip
            return "main_near_base"
        if bases is not None and self.stacked(r, bases["merge_base"]):
            return "stacked"
        return None

    def job_dropped(self, r: dict) -> str | None:
        return job_dropped(r, self.green.get(r["sha"], set()))

    def stacked(self, r: dict, merge_base: str) -> bool:
        """The stacked rule: whether another PR's head below `merge_base` failed `r`'s test in a way that counts (see the module docstring)."""
        t, me, k = ts(r["t"]), self.author.get(r["pr"], str(r["pr"])), key(r)
        others = sorted({sha for _, _, sha, pr, _ in self.seen.get(k, []) if pr not in (None, r["pr"])})  # fmt: skip
        for o in (o for sha in others for o in self.head_fails[sha, k]):
            mine = self.author.get(o["pr"], str(o["pr"])) == me
            near = t < self.landed.get(o["pr"], NEVER) if mine else abs(ts(o["t"]) - t) < WINDOW  # fmt: skip
            if near and not o["flaky"] and self.job_dropped(o) is None and below(o["sha"], merge_base):  # fmt: skip
                return True
        return False

    def other_heads(self, shas: set[str]) -> set[str]:
        """The other PRs' heads that failed a test one of PR heads `shas` failed, which the stacked rule compares to them."""
        return {o for sha in shas for r in self.heads.get(sha, []) for _, _, o, pr, _ in self.seen.get(key(r), []) if pr not in (None, r["pr"])}  # fmt: skip

    def truth(self, sha: str, bases: dict[str, str], dropped: collections.Counter) -> list[dict] | None:  # fmt: skip
        """The failures on PR head `sha` that are the PR's own, counting the others in `dropped` by rule; None if its main_base is older than the loaded failures."""
        if missing := ({sha} | self.other_heads({sha})) - self.green.keys():
            raise AssertionError(f"load_green({sorted(missing)!r}) first")
        base_t = commit_time(bases["main_base"])
        if base_t - WINDOW < ts(f"{self.start - LOOKBACK} 00:00:00"):
            print(f"skipping {sha}: merge base before the loaded failures", flush=True)
            return None
        kept = []
        for r in self.heads[sha]:
            if why := self.why_dropped(r, bases, base_t):
                dropped[why] += 1
            else:
                kept.append(r)
        return kept


def green_jobs(shas: list[str]) -> set[tuple[str, str, str]]:
    """The (head sha, workflow, job) that succeeded on any attempt."""
    sql = """
            SELECT DISTINCT head_sha, workflow_name, name FROM default.workflow_job FINAL
            WHERE head_sha IN ({}) AND repository_full_name = 'pytorch/pytorch' AND conclusion = 'success'"""
    return {(j["head_sha"], j["workflow_name"], j["name"]) for j in query_in(sql, shas, 200)}  # fmt: skip


def ci_bases(heads: dict[str, int]) -> dict[str, list[str]]:
    """Head sha -> the base shas of PR `heads[sha]` that the head's pull_request runs recorded.

    llm-selector.yml passes the base.sha of the head's push event to select_tests.
    pull_request_target runs record the base branch as head_sha, so they never
    match a head. A run records the base as of its latest webhook, which can be
    a base pushed after the head; see pr_bases. Heads with no such run, such
    as a fork's, are left out."""
    sql = """
            SELECT head_sha, tupleElement(p, 1) pr, groupUniqArray(tupleElement(p, 2)) bases
            FROM default.workflow_run ARRAY JOIN arrayZip(pull_requests.number, pull_requests.base.sha) AS p
            WHERE head_sha IN ({}) AND event = 'pull_request'
              AND repository.full_name = 'pytorch/pytorch'
            GROUP BY head_sha, pr"""
    return {r["head_sha"]: sorted(r["bases"]) for r in query_in(sql, sorted(heads), 200) if int(r["pr"]) == heads[r["head_sha"]]}  # fmt: skip


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
    for f in git_paths("ls-tree", "-r", "--name-only", main, "test/"):
        t = f.removeprefix("test/").removesuffix(".py")
        if f.rsplit("/", 1)[-1].startswith("test_") and f.endswith(".py"):
            if not t.startswith(patterns) and t not in kw.get("blocklisted_tests", []):
                out.add(t)
    if head is not None:
        # As select_tests.select does.
        added = git_paths(
            "diff", "--name-only", "--no-renames", "--diff-filter=A", base or main, head
        )
        for f in added:
            t = f.removeprefix("test/").removesuffix(".py")
            if re.match(r"test/([\w/]+/)?test_\w+\.py$", f) and not t.startswith(
                patterns
            ):
                out.add(t)
    return sorted(out)


def strip_trailers(msg: str) -> str:
    pat = r"^(Pull Request resolved|Pull-Request-resolved|Approved by|ghstack dependencies|Stack dependencies|ghstack-source-id|Pull-Request|Differential Revision|Co-authored-by|Co-Authored-By|Fixes|X-link):"
    return "\n".join(
        line for line in msg.splitlines() if not re.match(pat, line)
    ).strip()


def commit_text(sha: str) -> tuple[str, str]:
    """A landed commit's title and body, near its PR's event: without the " (#N)"
    the merge bot appends to the title, and without strip_trailers' lines,
    which are the trailers the bot adds (trymerge.py's gen_commit_message) and
    also author trailers such as Co-authored-by, Differential Revision and
    Fixes. The cc line and ghstack stack list the bot removes are not
    restored. pr_ and head_ samples keep the raw PR body, trailers and all, so
    their prompts differ slightly from revert and neutral ones."""
    title, _, body = git("log", "-1", "--format=%B", sha).partition("\n")
    return re.sub(r" \(#\d+\)$", "", title.rstrip()), strip_trailers(body)


def pr_text(pr: int, tries: int = 4) -> tuple[str, str]:
    """The PR's title and raw body, as select_tests reads them from the event.

    Raises CalledProcessError when gh fails permanently (not found, bad
    credentials) or keeps failing for `tries` attempts (e.g. rate limited),
    and ValueError when it succeeds with output that is not a PR."""
    if tries < 1:
        raise ValueError(f"tries must be positive, got {tries}")
    cmd = ["gh", "api", f"repos/pytorch/pytorch/pulls/{pr}", "--jq", "{title:.title, body:.body}"]  # fmt: skip
    for i in range(tries):
        if i:
            time.sleep(10 * 2 ** (i - 1))
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode == 0:
            try:
                d = json.loads(p.stdout)
                return d["title"], d["body"] or ""
            except (ValueError, KeyError, TypeError) as e:
                raise ValueError(
                    f"gh api returned no PR {pr}: {p.stdout[:200]!r}"
                ) from e
        if re.search(r"HTTP (401|404|410)\b", p.stderr):
            break
    raise subprocess.CalledProcessError(p.returncode, cmd, p.stdout, p.stderr)


def deleted_files(base: str, head: str) -> list[str]:
    """The files `head` deletes against `base`, counting a moved file's old path, as select_tests passes them to from_model_output."""
    return git_paths("diff", "--name-only", "--no-renames", "--diff-filter=D", base, head)  # fmt: skip


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
    *,
    window: tuple[dt.date, dt.date],
    **extra,
) -> bool:
    """Write sample `sid` for `sha` diffed against `merge_base` (default `base`), with main at `main_base` (default `merge_base`), whose TESTS give the candidates, drawn from `window`; False if it already exists."""
    path = out / f"{sid}.json"
    if path.exists():
        return False
    merge_base = merge_base or base
    main_base = main_base or merge_base
    diffstat, diff = diffstat_and_diff(merge_base, sha)
    write_json(path, {
        "id": sid, "kind": kind, "sha": sha, "base": base, "merge_base": merge_base, "main_base": main_base, "title": title, "body": body,
        "diff": diff, "diffstat": diffstat, "changed_files": changed_files(merge_base, sha), "deleted": deleted_files(merge_base, sha),
        "candidates": candidates_at(main_base, sha, merge_base), "builder": BUILDER, "window": [str(d) for d in window], "workflows": workflows(),
        **extra,
    })  # fmt: skip
    return True


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


def workflows() -> str:
    """A hash of score.ci_jobs, the job classification the job rule, revert truth and previous_failures read from this checkout's .github/workflows."""
    jobs = {str(td): sorted(js) for td, js in sorted(score.ci_jobs().items())}
    return hashlib.sha256(json.dumps(jobs).encode()).hexdigest()[:12]


def other_build(out: Path, prefix: str, window: tuple[dt.date, dt.date]) -> str | None:
    """Why the `prefix` samples in `out` cannot be extended by this BUILDER from `window` with this checkout's workflows, if they cannot."""
    marks = {p.stem: json.loads(p.read_text()) for p in sorted(out.glob(f"{prefix}*.json"))}  # fmt: skip
    marks = {k: (m.get("builder"), m.get("window"), m.get("workflows")) for k, m in marks.items()}  # fmt: skip
    if legacy := [k for k, (b, _, _) in marks.items() if b is None]:
        return f"{prefix} samples in {out}, e.g. {legacy[0]}, predate the builder field; build into a fresh --data"  # fmt: skip
    if other := [k for k, (b, _, _) in marks.items() if b != BUILDER]:
        return f"{prefix} samples in {out}, e.g. {other[0]}, were written by builder {marks[other[0]][0]}, not {BUILDER}; build into a fresh --data"  # fmt: skip
    if other := [k for k, (_, w, _) in marks.items() if w != [str(d) for d in window]]:
        return f"{prefix} samples in {out}, e.g. {other[0]}, were drawn from {marks[other[0]][1]}, not {window[0]}/{window[1]}"  # fmt: skip
    here = workflows()
    if other := [k for k, (_, _, w) in marks.items() if w != here]:
        return f"{prefix} samples in {out}, e.g. {other[0]}, judged jobs by workflows {marks[other[0]][2]}, not this checkout's {here}; extend it from the checkout that built it or build into a fresh --data"  # fmt: skip
    return None


def pr_samples(h: History, out: Path, jobs: int) -> None:
    lo = ts(f"{h.start} 00:00:00")
    own = [r for rows in h.heads.values() for r in rows if h.why_dropped(r) is None]
    h.load_green({r["sha"] for r in own})
    own = [r for r in own if h.why_dropped(r) is None]
    first: dict[int, dict] = {}
    for r in own:
        if ts(r["t"]) >= lo and (r["pr"] not in first or r["t"] < first[r["pr"]]["t"]):
            first[r["pr"]] = r
    # A reproduction counts on any of the PR's heads the window rule keeps, also one from before --start.
    shas = {r["sha"] for r in own if r["pr"] in first}
    fetch(sorted(shas))
    fetch(sorted(lower := h.other_heads(shas)))
    h.load_green(lower)
    ci = ci_bases({sha: h.heads[sha][0]["pr"] for sha in shas})
    # A base CI recorded after this head is not in its history, so fetch it as head_samples' fetch of the later head would.
    fetch(sorted({b for bs in ci.values() for b in bs}))
    truths: dict[str, tuple[dict[str, str], list[dict], collections.Counter]] = {}

    def judge(sha: str) -> None:
        if not has_commit(sha) or (bases := pr_bases(sha, ci.get(sha, []))) is None:
            return
        if (kept := h.truth(sha, bases, c := collections.Counter())) is not None:
            truths[sha] = (bases, kept, c)

    run_each("PR head", judge, sorted(shas), jobs)
    envs, heads = collections.defaultdict(set), collections.defaultdict(set)
    for _, kept, _ in truths.values():
        for r in kept:
            envs[k := (r["pr"], *key(r))].add(r["job"].rsplit(" / ", 1)[0])
            heads[k].add(r["sha"])
    reproduced = {k for k in envs if len(envs[k]) >= 2 or len(heads[k]) >= 2}
    sources: list[str] = []
    dropped: list[collections.Counter] = []

    def go(pr: int) -> None:
        if (sha := first[pr]["sha"]) not in truths:
            return
        bases, kept, c = truths[sha]
        dropped.append(c)
        if dates := [r["t"] for r in kept if ts(r["t"]) >= lo]:
            truth_jobs = by_file(kept)
            strong = {r["tf"] for r in kept if (pr, *key(r)) in reproduced}
            if (out / f"pr_{pr}.json").exists():
                return
            try:
                title, body = pr_text(pr)
                if write_sample(out, f"pr_{pr}", "pr", sha, title=title, body=body, **bases, pr=pr, date=min(dates)[:19], truth=sorted(truth_jobs),
                                window=(h.start, h.end), truth_strong=sorted(strong), truth_jobs=truth_jobs):  # fmt: skip
                    sources.append(bases["base_source"])
            except subprocess.CalledProcessError as e:
                print(f"skipping PR {pr}: {e} {e.stderr or ''}", flush=True)

    run_each("PR", go, sorted(first), jobs)
    print_dropped("pr", dropped)
    print(f"{sources.count('main')} of {len(sources)} PR samples written have no CI base and are diffed against main", flush=True)  # fmt: skip


def landed_times(since: dt.date) -> dict[int, dt.datetime]:
    """PR number -> when it first landed on main since `since`, from the merge bot's `Pull Request resolved:` line, the last one of a body that quotes others.

    A PR that first landed before `since` gets the time of its first reland, or none.
    """
    out: dict[int, dt.datetime] = {}
    for c in git("log", "origin/main", f"--since={since} 00:00 +0000", "--format=%ct %b%x00").split("\0"):  # fmt: skip
        if (m := re.match(r"\s*(\d+) ", c)) and (prs := re.findall(r"Pull Request resolved: \S+/pull/(\d+)", c)):  # fmt: skip
            t, pr = dt.datetime.fromtimestamp(int(m.group(1)), tz=dt.timezone.utc), int(prs[-1])  # fmt: skip
            out[pr] = min(t, out.get(pr, t))
    return out


def head_created(shas: list[str]) -> dict[str, str]:
    """Each head's first workflow job creation time, roughly when CI's target-determination job reads the PR's previous failures."""
    sql = """
            SELECT head_sha, toString(min(created_at)) t FROM default.workflow_job FINAL
            WHERE head_sha IN ({}) AND repository_full_name = 'pytorch/pytorch' GROUP BY head_sha"""
    return {r["head_sha"]: r["t"] for r in query_in(sql, shas, 200)}


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
    fetch(sorted(lower := h.other_heads(shas)))
    h.load_green(lower)
    ci = ci_bases({sha: h.heads[sha][0]["pr"] for sha in shas})
    fetch(sorted({b for bs in ci.values() for b in bs}))
    created = {sha: ts(t) for sha, t in head_created(sorted(shas)).items()}
    failed = collections.defaultdict(list)
    for other, rows in h.heads.items():
        for r in rows:
            if r["pr"] in prs and not r["flaky"] and job_runs_td(r["job"], r["wf"]):
                failed[r["pr"]].append((ts(r["t"]), other, r["tf"]))
    sources: list[str] = []
    dropped: list[collections.Counter] = []

    def go(sha: str) -> None:
        pr = h.heads[sha][0]["pr"]
        if not has_commit(sha) or (bases := pr_bases(sha, ci.get(sha, []))) is None:
            return
        if (kept := h.truth(sha, bases, c := collections.Counter())) is None:
            return
        dropped.append(c)
        if kept:
            truth_jobs = by_file(kept)
            previous = sorted({tf for t, other, tf in failed[pr] if other != sha and sha in created and t < created[sha]})  # fmt: skip
            try:
                if write_sample(out, f"head_{pr}_{sha[:12]}", "head", sha, title=prs[pr]["title"], body=prs[pr]["body"], **bases, pr=pr,
                                window=(h.start, h.end), truth=sorted(truth_jobs), truth_jobs=truth_jobs, previous_failures=previous):  # fmt: skip
                    sources.append(bases["base_source"])
            except subprocess.CalledProcessError as e:
                print(f"skipping head {sha}: {e} {e.stderr or ''}", flush=True)

    run_each("head", go, sorted(shas), jobs)
    print_dropped("head", dropped)
    print(f"{sources.count('main')} of {len(sources)} head samples written have no CI base and are diffed against main", flush=True)  # fmt: skip


def revert_truth_jobs(
    signal: set[tuple[str, str, str]], sha: str, landed: dt.datetime, reverted: str
) -> dict[str, list[tuple[str, str]]]:
    """The main-branch (workflow, job)s that run TD each signal test file failed in between landing and the revert, on the main commits from `sha` up to the revert.

    `signal` holds the (test file without .py, class, test name) of the
    autorevert signal's tests, class "" when its key names none, which then
    matches the name in any class; other tests of those files do not count.

    A failure is dropped as PR truth drops it when its job is unstable or
    passed on any attempt on the commit, or the test passed again in the job.

    `reverted` is when autorevert decided to revert. The range ends before the
    first commit that reverts `sha` and at `reverted`, whichever is earlier, so
    the main commits that landed between the two are left out: truth is only
    lost, never added."""
    tests = ",".join(sorted({f"({sql_str(f + '.py')}, {sql_str(n)})" for f, _, n in signal}))  # fmt: skip
    t0 = landed.strftime("%Y-%m-%d %H:%M:%S")
    reverts = git("rev-list", "--first-parent", f"--grep=This reverts commit {sha}", f"{sha}..origin/main").split()  # fmt: skip
    tip = f"{reverts[-1]}^" if reverts else "origin/main"
    with_change = git("rev-list", "--first-parent", "--ancestry-path", f"--until={reverted[:19]} +0000", f"{sha}^..{tip}").split()  # fmt: skip
    shas = ",".join(f"'{c}'" for c in with_change or [sha])
    failed = query(f"""
        WITH j AS (
          SELECT id, head_sha, workflow_name, name FROM default.workflow_job FINAL
          WHERE created_at >= toDateTime('{t0}') - INTERVAL 1 HOUR
            AND created_at <= parseDateTimeBestEffort('{reverted}') + INTERVAL 1 HOUR
            AND repository_full_name = 'pytorch/pytorch' AND (head_branch = 'main' OR head_branch LIKE 'trunk/%')
            AND head_sha IN ({shas})
            AND run_id IN (SELECT id FROM default.workflow_run WHERE head_repository.full_name = 'pytorch/pytorch'
                           AND created_at >= toDateTime('{t0}') - INTERVAL 1 HOUR
                           AND created_at <= parseDateTimeBestEffort('{reverted}') + INTERVAL 1 HOUR)
            AND {" AND ".join(f"name NOT LIKE '%{s}%'" for s in SKIP_JOBS)})
        SELECT DISTINCT tf, classname, test, sha, wf, job FROM (
          SELECT t.job_id, replaceRegexpOne(file, '\\\\.py$', '') tf, classname, t.name test, j.head_sha sha, j.workflow_name wf, j.name job
          FROM tests.all_test_runs t JOIN j ON j.id = t.job_id
          WHERE t.job_id IN (SELECT id FROM j) AND (file, t.name) IN ({tests})
            AND time_inserted >= toDateTime('{t0}') AND time_inserted <= parseDateTimeBestEffort('{reverted}') + INTERVAL 2 HOUR
          GROUP BY t.job_id, tf, classname, test, sha, wf, job
          -- A test that also passed in the job, as run_test.py's rerun of it in a new process does, is flaky there.
          HAVING countIf(failure_count > 0 OR error_count > 0) > 0 AND countIf(failure_count = 0 AND error_count = 0 AND skipped_count = 0) = 0)""")
    failed = [f for f in failed if {(f["tf"], "", f["test"]), (f["tf"], f["classname"], f["test"])} & signal]  # fmt: skip
    green: dict[str, set[tuple[str, str]]] = collections.defaultdict(set)
    for c, wf, job in green_jobs(sorted({f["sha"] for f in failed})):
        green[c].add((wf, job))
    truth_jobs: dict[str, set] = collections.defaultdict(set)
    for f in failed:
        if job_dropped(f, green[f["sha"]]) is None:
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
    if days := sorted({dt.date.fromisoformat(r["reverted_at"][:10]) for r in rows.values()}):  # fmt: skip
        # Revert truth reads tests.all_test_runs around each revert, so its first and last must still be there.
        check_written(sorted({days[0], days[-1]}))
    fetch(sorted(rows))
    found: list[str] = []
    empty: list[str] = []

    def go(sha: str) -> None:
        r = rows[sha]
        if not has_commit(sha):
            return
        keys = [k.split("::") for k in r["source_signal_keys"] if "::" in k]
        signal = {(k[0].removesuffix(".py"), k[1] if len(k) == 3 else "", k[-1]) for k in keys}  # fmt: skip
        try:
            truth_jobs = revert_truth_jobs(signal, sha, commit_time(sha), r["reverted_at"])  # fmt: skip
            (found if truth_jobs else empty).append(sha)
            if not truth_jobs:
                print(f"skipping revert {sha}: no failure of its signal tests in a main job that runs TD survives the job, job_green and passed_in_job rules", flush=True)  # fmt: skip
                return
            title, body = commit_text(sha)
            write_sample(out, f"rev_{sha[:12]}", "revert", sha, sha + "^", title, body, window=(start, end), date=r["reverted_at"],
                         truth=sorted(truth_jobs), truth_jobs=truth_jobs)  # fmt: skip
        except subprocess.CalledProcessError as e:
            print(f"skipping revert {sha}: {e} {e.stderr or ''}", flush=True)

    run_each("revert", go, list(rows), jobs)
    if empty and not found:
        raise RuntimeError(f"none of the {len(empty)} reverts has a failure of its signal tests in tests.all_test_runs; check that it is still written (see pull_failures)")  # fmt: skip


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
    # Reverts (by message or by the autorevert bot's record, which has every
    # commit a rev_ sample can be of) and titles naming a revert or reland
    # anywhere filter the sample rather than the population, so a later revert
    # only drops its id from the draw instead of reshuffling it.
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

    def go(sha: str) -> None:
        title, body = commit_text(sha)
        if sha in reverted or re.search(r"\b(revert|reland)\b", title, re.IGNORECASE):  # fmt: skip
            return
        try:
            write_sample(out, f"neu_{sha[:12]}", "neutral", sha, sha + "^", title, body, window=(start, end), truth=[])  # fmt: skip
        except subprocess.CalledProcessError as e:
            print(f"skipping neutral {sha}: {e} {e.stderr or ''}", flush=True)

    run_each("neutral", go, random.Random(0).sample(shas, min(n, len(shas))), 1)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data", type=Path, required=True, help="samples are written to DATA/samples")  # fmt: skip
    p.add_argument("--start", type=dt.date.fromisoformat, required=True, help="first day (UTC); neutral ignores it")  # fmt: skip
    p.add_argument("--end", type=dt.date.fromisoformat, required=True, help="day after the last (UTC); neutral samples the 25 days before it")  # fmt: skip
    p.add_argument("--neutral", type=int, default=200, help="neutral commits to sample")
    p.add_argument("--jobs", type=int, default=8, help="worker threads; neutral always uses 1")  # fmt: skip
    p.add_argument("--kinds", nargs="+", choices=KINDS, default=KINDS)
    args = p.parse_args()
    if {"pr", "head"} & set(args.kinds) and (closes := args.end + WINDOW) > dt.datetime.now(dt.timezone.utc).date():  # fmt: skip
        p.error(f"PR truth looks for other authors' failures and disables up to --end + {WINDOW.days} days; rerun on or after {closes}")  # fmt: skip
    out = args.data / "samples"
    out.mkdir(parents=True, exist_ok=True)
    neutral = (args.end - dt.timedelta(days=25), args.end)
    builds = {"pr": [("pr_", (args.start, args.end))], "head": [("pr_", (args.start, args.end)), ("head_", (args.start, args.end))],
              "revert": [("rev_", (args.start, args.end))], "neutral": [("neu_", neutral)]}  # fmt: skip
    for kind in args.kinds:
        for prefix, window in builds[kind]:
            if why := other_build(out, prefix, window):
                raise SystemExit(f"refusing {kind} samples: {why}")
    if "head" in args.kinds and "pr" not in args.kinds and not any(out.glob("pr_*.json")):  # fmt: skip
        print(f"no head samples: there are no pr_ samples in {out}; build them with --kinds pr first", flush=True)  # fmt: skip
        args.kinds = [k for k in args.kinds if k != "head"]
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
        neutral_samples(out, *neutral, args.neutral)


if __name__ == "__main__":
    main()
