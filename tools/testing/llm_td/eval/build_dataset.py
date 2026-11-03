"""Build the labelled eval set from CI history.

    python -m tools.testing.llm_td.eval.build_dataset --data DIR --start 2026-09-20 --end 2026-10-26

Three kinds of sample, each a JSON file under DIR/samples:

  pr_<N>      a PR whose own CI failed tests that main did not fail. The
              ground truth is every (test file, CI config) that failed on the
              earliest failing head. These runs are censored: pull/trunk jobs
              already skip 75% of files through TD, so truth is a lower bound.
  rev_<sha>   a commit that landed and was auto-reverted between --start and
              --end because it broke a test on main, where every test runs.
              The truth is the test files of the reverting signal, in every
              config they failed in on main between landing and the revert.
  neu_<sha>   an ordinary main commit from the 25 days before --end, with no
              truth, used only to measure how much a selector runs on typical
              traffic. Reverts and relands are left out.

A PR failure is attributed to the PR only if the same test did not fail on
main within four days and did not fail on another author's PR. PR truths that
reproduced (same test failing in two build environments, or on two heads of
the PR) are kept in `truth_strong`; single-job one-offs are mostly flakes.
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
from pathlib import Path

from tools.testing.llm_td.eval import REPO
from tools.testing.llm_td.eval.clickhouse import query


MAIN_WINDOW = dt.timedelta(days=4)


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


KINDS = ("pr", "revert", "neutral")


def is_main(branch: str) -> bool:
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
            SELECT f.job_id job_id, tf, classname, f.name test, toString(t) t, j.head_sha sha,
                   j.head_branch br, j.workflow_name wf, j.name job
            FROM f JOIN (
              SELECT id, head_sha, head_branch, workflow_name, name FROM default.workflow_job
              WHERE id IN (SELECT job_id FROM f) AND repository_full_name = 'pytorch/pytorch') j
            ON j.id = f.job_id""")
        print(d, len(rows), flush=True)
        d = n
    return [
        r
        for r in rows
        if "rerun_disabled_tests" not in r["job"] and "mem_leak_check" not in r["job"]
    ]


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


def pr_caused_failures(rows: list[dict]) -> tuple[list[dict], dict]:
    """Split failures into main and PR runs and keep the PR failures attributable to the PR."""
    sha2pr = pr_of_shas(sorted({r["sha"] for r in rows if not is_main(r["br"])}))
    main = collections.defaultdict(list)
    pr_rows = []
    for r in rows:
        if is_main(r["br"]) or r["sha"] not in sha2pr:
            main[(r["tf"], r["test"])].append(ts(r["t"]))
        else:
            pr_rows.append({**r, "pr": sha2pr[r["sha"]]})
    author = pr_authors(sorted({r["pr"] for r in pr_rows}))
    authors = collections.defaultdict(set)
    for r in pr_rows:
        authors[(r["tf"], r["test"])].add(author.get(r["pr"], r["pr"]))
    keep = [
        r
        for r in pr_rows
        if len(authors[(r["tf"], r["test"])]) < 2
        and not any(
            abs(ts(str(r["t"])) - m) < MAIN_WINDOW
            for m in main.get((r["tf"], r["test"]), [])
        )
    ]
    return keep, main


def candidates_at(sha: str) -> list[str]:
    """The test files run_test.py would consider at `sha`: discover_tests.TESTS, without importing that tree."""
    tree = ast.parse(git("show", f"{sha}:tools/testing/discover_tests.py"))
    kw: dict[str | None, list] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and getattr(node.targets[0], "id", None) == "TESTS"
        ):
            for k in node.value.keywords:  # type: ignore[attr-defined]
                if isinstance(k.value, ast.List):
                    kw[k.arg] = [
                        e.value for e in k.value.elts if isinstance(e, ast.Constant)
                    ]
    out = set(kw.get("extra_tests", []))
    for f in git("ls-tree", "-r", "--name-only", sha, "test/").splitlines():
        t = f.removeprefix("test/").removesuffix(".py")
        if f.rsplit("/", 1)[-1].startswith("test_") and f.endswith(".py"):
            if not any(
                t.startswith(p) for p in kw.get("blocklisted_patterns", [])
            ) and t not in kw.get("blocklisted_tests", []):
                out.add(t)
    return sorted(out)


def strip_trailers(msg: str) -> str:
    pat = r"^(Pull Request resolved|Approved by|ghstack-source-id|Pull-Request|Differential Revision|Co-authored-by|Co-Authored-By|Fixes|X-link):"
    return "\n".join(
        line for line in msg.splitlines() if not re.match(pat, line)
    ).strip()


def pr_text(pr: int) -> tuple[str, str]:
    p = subprocess.run(
        [
            "gh",
            "api",
            f"repos/pytorch/pytorch/pulls/{pr}",
            "--jq",
            "{title:.title, body:.body}",
        ],
        capture_output=True,
        text=True,
    )
    try:
        d = json.loads(p.stdout)
    except json.JSONDecodeError:
        return "", ""
    return d["title"], re.sub(
        r"(?s)Stack from \[ghstack\].*?(\n\n|$)", "", d["body"] or ""
    )


def write_sample(
    out: Path, sid: str, kind: str, sha: str, base: str, title: str, body: str, **extra
) -> None:
    path = out / f"{sid}.json"
    if path.exists():
        return
    path.write_text(json.dumps({
        "id": sid, "kind": kind, "sha": sha, "base": base, "title": title, "body": body,
        "diff": git("diff", "--no-color", "--no-ext-diff", base, sha, check=False),
        "diffstat": git("diff", "--stat=160,120", "--stat-count=400", base, sha),
        "changed_files": git("diff", "--name-only", base, sha).split(),
        "candidates": candidates_at(sha),
        **extra,
    }))  # fmt: skip


def pr_samples(keep: list[dict], main: dict, out: Path, jobs: int) -> None:
    main_by_file = collections.defaultdict(list)
    for (tf, _), times in main.items():
        main_by_file[tf] += times
    envs, heads = collections.defaultdict(set), collections.defaultdict(set)
    first: dict[int, dict] = {}
    for r in keep:
        k = (r["pr"], r["tf"], r["classname"], r["test"])
        envs[k].add(r["job"].split(" / ")[0])
        heads[k].add(r["sha"])
        if r["pr"] not in first or r["t"] < first[r["pr"]]["t"]:
            first[r["pr"]] = r
    reproduced = {(k[0], k[1]) for k in envs if len(envs[k]) >= 2 or len(heads[k]) >= 2}
    by = collections.defaultdict(lambda: collections.defaultdict(set))
    for r in keep:
        if r["sha"] == first[r["pr"]]["sha"]:
            by[r["pr"]][r["tf"]].add(r["job"])
    fetch(sorted({f["sha"] for f in first.values()}))

    def go(pr: int) -> None:
        sha = first[pr]["sha"]
        if not has_commit(sha):
            return
        base = git("merge-base", "origin/main", sha, check=False).strip()
        if not base:
            return
        base_t = commit_time(base)
        # A file broken on main around the merge base is not this PR's fault.
        truth = sorted(
            tf
            for tf in by[pr]
            if not any(abs(base_t - m) < MAIN_WINDOW for m in main_by_file.get(tf, []))
        )
        if truth:
            title, body = pr_text(pr)
            write_sample(out, f"pr_{pr}", "pr", sha, base, title, body, pr=pr, date=first[pr]["t"][:19], truth=truth,
                         truth_strong=[tf for tf in truth if (pr, tf) in reproduced],
                         truth_jobs={tf: sorted(by[pr][tf]) for tf in truth})  # fmt: skip

    with cf.ThreadPoolExecutor(jobs) as ex:
        list(ex.map(go, sorted(first)))


def revert_truth_jobs(
    signal: set[str], landed: dt.datetime, reverted: str
) -> dict[str, list[str]]:
    """The main-branch jobs each signal test file failed in between landing and the revert."""
    files = ",".join(f"'{t}.py'" for t in signal)
    t0 = landed.strftime("%Y-%m-%d %H:%M:%S")
    failed = query(f"""
        SELECT DISTINCT replaceRegexpOne(file, '\\.py$', '') tf, j.name job FROM tests.all_test_runs t
        JOIN (SELECT id, name FROM default.workflow_job
              WHERE created_at >= toDateTime('{t0}') - INTERVAL 1 HOUR
                AND created_at <= parseDateTimeBestEffort('{reverted}') + INTERVAL 1 HOUR
                AND repository_full_name = 'pytorch/pytorch' AND (head_branch = 'main' OR head_branch LIKE 'trunk/%')
                AND name NOT LIKE '%rerun_disabled_tests%' AND name NOT LIKE '%mem_leak_check%') j ON j.id = t.job_id
        WHERE time_inserted >= toDateTime('{t0}') AND time_inserted <= parseDateTimeBestEffort('{reverted}') + INTERVAL 2 HOUR
          AND file IN ({files}) AND (failure_count > 0 OR error_count > 0)""")
    truth_jobs: dict[str, set] = collections.defaultdict(set)
    for f in failed:
        truth_jobs[f["tf"]].add(f["job"])
    return {k: sorted(v) for k, v in truth_jobs.items()}


def revert_samples(out: Path, jobs: int, start: dt.date, end: dt.date) -> None:
    rows = query(f"""
        SELECT toString(ts) reverted_at, commit_sha, source_signal_keys FROM misc.autorevert_events_v2
        WHERE repo = 'pytorch/pytorch' AND action = 'revert' AND dry_run = 0
          AND ts >= toDateTime('{start}') AND ts < toDateTime('{end}')
          AND arrayExists(k -> position(k, '::') > 0, source_signal_keys)""")
    fetch(sorted({r["commit_sha"] for r in rows}))

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
            truth_jobs = revert_truth_jobs(signal, commit_time(sha), r["reverted_at"])
            title, _, body = git("log", "-1", "--format=%B", sha).partition("\n")
            write_sample(out, f"rev_{sha[:12]}", "revert", sha, sha + "^", title, strip_trailers(body), date=r["reverted_at"],
                         truth=sorted(signal), truth_jobs=truth_jobs)  # fmt: skip
        except (subprocess.CalledProcessError, RuntimeError) as e:
            print(f"skipping revert {sha}: {e}", flush=True)

    with cf.ThreadPoolExecutor(jobs) as ex:
        list(ex.map(go, rows))


def neutral_samples(out: Path, start: dt.date, end: dt.date, n: int) -> None:
    log = git(
        "log",
        "--first-parent",
        "--format=%H %s",
        f"--since={start}",
        f"--until={end}",
        "origin/main",
    ).splitlines()
    reverted = set(
        re.findall(
            r"This reverts commit ([0-9a-f]{40})",
            git("log", "--format=%B", f"--since={start}", "origin/main"),
        )
    )
    shas = [
        sha
        for sha, _, title in (line.partition(" ") for line in log)
        if not re.match(r"(Revert|Reland)", title) and sha not in reverted
    ]
    for sha in random.Random(0).sample(shas, min(n, len(shas))):
        msg = git("log", "-1", "--format=%B", sha)
        title, _, body = msg.partition("\n")
        write_sample(
            out,
            f"neu_{sha[:12]}",
            "neutral",
            sha,
            sha + "^",
            title,
            strip_trailers(body),
            truth=[],
        )


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
    if "pr" in args.kinds:
        keep, main_fail = pr_caused_failures(pull_failures(args.start, args.end))
        pr_samples(keep, main_fail, out, args.jobs)
    if "revert" in args.kinds:
        revert_samples(out, args.jobs, args.start, args.end)
    if "neutral" in args.kinds:
        neutral_samples(out, args.end - dt.timedelta(days=25), args.end, args.neutral)


if __name__ == "__main__":
    main()
