#!/usr/bin/env python3

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from signal import SIG_DFL, signal, SIGPIPE
from typing import Dict, Iterator, List, Optional, Set, Tuple

from tools.stats.s3_stat_parser import get_cases, get_test_stats_summaries, Report


def get_git_commit_history(*, path: str, ref: str) -> List[Tuple[str, datetime]]:
    rc = subprocess.check_output(
        ["git", "-C", path, "log", "--pretty=format:%H %ct", ref],
    ).decode("latin-1")
    return [
        (x[0], datetime.fromtimestamp(int(x[1]), tz=timezone.utc))
        for x in [line.split(" ") for line in rc.split("\n")]
    ]


def make_column(
    *,
    data: Optional[Report],
    filename: Optional[str],
    suite_name: Optional[str],
    test_name: str,
    digits: int,
) -> Tuple[str, int]:
    decimals = 3
    num_length = digits + 1 + decimals
    if data:
        cases = get_cases(
            data=data, filename=filename, suite_name=suite_name, test_name=test_name
        )
        if cases:
            case = cases[0]
            status = case["status"]
            omitted = len(cases) - 1
            if status:
                return f"{status.rjust(num_length)} ", omitted
            else:
                return f'{case["seconds"]:{num_length}.{decimals}f}s', omitted
        else:
            return f'{"absent".rjust(num_length)} ', 0
    else:
        return " " * (num_length + 1), 0


def make_columns(
    *,
    jobs: List[str],
    jsons: Dict[str, Report],
    omitted: Dict[str, int],
    filename: Optional[str],
    suite_name: Optional[str],
    test_name: str,
    digits: int,
) -> str:
    columns = []
    total_omitted = 0
    total_suites = 0
    for job in jobs:
        data = jsons.get(job)
        column, omitted_suites = make_column(
            data=data,
            filename=filename,
            suite_name=suite_name,
            test_name=test_name,
            digits=digits,
        )
        columns.append(column)
        total_suites += omitted_suites
        if job in omitted:
            total_omitted += omitted[job]
    if total_omitted > 0:
        columns.append(f"({total_omitted} job re-runs omitted)")
    if total_suites > 0:
        columns.append(f"({total_suites} matching suites omitted)")
    return " ".join(columns)


def make_lines(
    *,
    jobs: Set[str],
    jsons: Dict[str, List[Report]],
    filename: Optional[str],
    suite_name: Optional[str],
    test_name: str,
) -> List[str]:
    lines = []
    for job, reports in jsons.items():
        for data in reports:
            cases = get_cases(
                data=data,
                filename=filename,
                suite_name=suite_name,
                test_name=test_name,
            )
            if cases:
                case = cases[0]
                status = case["status"]
                line = f'{job} {case["seconds"]}s{f" {status}" if status else ""}'
                if len(cases) > 1:
                    line += f" ({len(cases) - 1} matching suites omitted)"
                lines.append(line)
            elif job in jobs:
                lines.append(f"{job} (test not found)")
    if lines:
        return lines
    else:
        return ["(no reports in S3)"]


def history_lines(
    *,
    commits: List[Tuple[str, datetime]],
    jobs: Optional[List[str]],
    filename: Optional[str],
    suite_name: Optional[str],
    test_name: str,
    delta: int,
    sha_length: int,
    mode: str,
    digits: int,
) -> Iterator[str]:
    prev_time = datetime.now(tz=timezone.utc)
    for sha, time in commits:
        if (prev_time - time).total_seconds() < delta * 3600:
            continue
        prev_time = time
        if jobs is None:
            summaries = get_test_stats_summaries(sha=sha)
        else:
            summaries = get_test_stats_summaries(sha=sha, jobs=jobs)
        if mode == "columns":
            assert jobs is not None
            # we assume that get_test_stats_summaries here doesn't
            # return empty lists
            omitted = {job: len(l) - 1 for job, l in summaries.items() if len(l) > 1}
            lines = [
                make_columns(
                    jobs=jobs,
                    jsons={job: l[0] for job, l in summaries.items()},
                    omitted=omitted,
                    filename=filename,
                    suite_name=suite_name,
                    test_name=test_name,
                    digits=digits,
                )
            ]
        else:
            assert mode == "multiline"
            lines = make_lines(
                jobs=set(jobs or []),
                jsons=summaries,
                filename=filename,
                suite_name=suite_name,
                test_name=test_name,
            )
        for line in lines:
            yield f"{time:%Y-%m-%d %H:%M:%S}Z {sha[:sha_length]} {line}".rstrip()


class HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    pass


def description() -> str:
    return r"""
Display the history of a test.

Each line of (non-error) output starts with the timestamp and SHA1 hash
of the commit it refers to, in this format:

    YYYY-MM-DD hh:mm:ss 0123456789abcdef0123456789abcdef01234567

In multiline mode, each line next includes the name of a CircleCI job,
followed by the time of the specified test in that job at that commit.
Example:

    $ tools/stats/test_history.py --mode=multiline --ref=86a961af879 --sha-length=8 \
      --test=test_composite_compliance_dot_cpu_float32 \
      --job linux-xenial-py3.7-gcc5.4-test-default1 --job linux-xenial-py3.7-gcc7-test-default1
    2022-02-18 15:47:37Z 86a961af linux-xenial-py3.7-gcc5.4-test-default1 0.001s
    2022-02-18 15:47:37Z 86a961af linux-xenial-py3.7-gcc7-test-default1 0.001s
    2022-02-18 15:12:34Z f5e201e4 linux-xenial-py3.7-gcc5.4-test-default1 0.001s
    2022-02-18 15:12:34Z f5e201e4 linux-xenial-py3.7-gcc7-test-default1 0.001s
    2022-02-18 13:14:56Z 1c0df265 linux-xenial-py3.7-gcc5.4-test-default1 0.001s
    2022-02-18 13:14:56Z 1c0df265 linux-xenial-py3.7-gcc7-test-default1 0.001s
    2022-02-18 13:14:56Z e73eaffd (no reports in S3)
    2022-02-18 06:29:12Z 710f12f5 linux-xenial-py3.7-gcc5.4-test-default1 0.001s

Another multiline example, this time with the --all flag:

    $ tools/stats/test_history.py --mode=multiline --all --ref=86a961af879 --delta=12 --sha-length=8 \
      --test=test_composite_compliance_dot_cuda_float32
    2022-02-18 03:49:46Z 69389fb5 linux-bionic-cuda10.2-py3.9-gcc7-test-default1 0.001s skipped
    2022-02-18 03:49:46Z 69389fb5 linux-bionic-cuda10.2-py3.9-gcc7-test-slow1 0.001s skipped
    2022-02-18 03:49:46Z 69389fb5 linux-xenial-cuda11.3-py3.7-gcc7-test-default1 0.001s skipped
    2022-02-18 03:49:46Z 69389fb5 periodic-linux-bionic-cuda11.5-py3.7-gcc7-test-default1 0.001s skipped
    2022-02-18 03:49:46Z 69389fb5 periodic-linux-xenial-cuda10.2-py3-gcc7-slow-gradcheck-test-default1 0.001s skipped
    2022-02-18 03:49:46Z 69389fb5 periodic-linux-xenial-cuda11.1-py3.7-gcc7-debug-test-default1 0.001s skipped

In columns mode, the name of the job isn't printed, but the order of the
columns is guaranteed to match the order of the jobs passed on the
command line. Example:

    $ tools/stats/test_history.py --mode=columns --ref=86a961af879 --sha-length=8 \
      --test=test_composite_compliance_dot_cpu_float32 \
      --job linux-xenial-py3.7-gcc5.4-test-default1 --job linux-xenial-py3.7-gcc7-test-default1
    2022-02-18 15:47:37Z 86a961af    0.001s    0.001s
    2022-02-18 15:12:34Z f5e201e4    0.001s    0.001s
    2022-02-18 13:14:56Z 1c0df265    0.001s    0.001s
    2022-02-18 13:14:56Z e73eaffd
    2022-02-18 06:29:12Z 710f12f5    0.001s    0.001s
    2022-02-18 05:20:30Z 51b04f27    0.001s    0.001s
    2022-02-18 03:49:46Z 69389fb5    0.001s    0.001s
    2022-02-18 00:19:12Z 056b6260    0.001s    0.001s
    2022-02-17 23:58:32Z 39fb7714    0.001s    0.001s

Minor note: in columns mode, a blank cell means that no report was found
in S3, while the word "absent" means that a report was found but the
indicated test was not found in that report.
"""


def parse_args(raw: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        __file__,
        description=description(),
        formatter_class=HelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["columns", "multiline"],
        help="output format",
        default="columns",
    )
    parser.add_argument(
        "--pytorch",
        help="path to local PyTorch clone",
        default=".",
    )
    parser.add_argument(
        "--ref",
        help="starting point (most recent Git ref) to display history for",
        default="master",
    )
    parser.add_argument(
        "--delta",
        type=int,
        help="minimum number of hours between commits",
        default=0,
    )
    parser.add_argument(
        "--sha-length",
        type=int,
        help="length of the prefix of the SHA1 hash to show",
        default=40,
    )
    parser.add_argument(
        "--digits",
        type=int,
        help="(columns) number of digits to display before the decimal point",
        default=4,
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="(multiline) ignore listed jobs, show all jobs for each commit",
    )
    parser.add_argument(
        "--file",
        help="name of the file containing the test",
    )
    parser.add_argument(
        "--suite",
        help="name of the suite containing the test",
    )
    parser.add_argument("--test", help="name of the test", required=True)
    parser.add_argument(
        "--job",
        help="names of jobs to display columns for, in order",
        action="append",
        default=[],
    )
    args = parser.parse_args(raw)

    args.jobs = None if args.all else args.job
    # We dont allow implicit or empty "--jobs", unless "--all" is specified.
    if args.jobs == []:
        parser.error("No jobs specified.")

    return args


def run(raw: List[str]) -> Iterator[str]:
    args = parse_args(raw)

    commits = get_git_commit_history(path=args.pytorch, ref=args.ref)

    return history_lines(
        commits=commits,
        jobs=args.jobs,
        filename=args.file,
        suite_name=args.suite,
        test_name=args.test,
        delta=args.delta,
        mode=args.mode,
        sha_length=args.sha_length,
        digits=args.digits,
    )


def main() -> None:
    for line in run(sys.argv[1:]):
        print(line, flush=True)


if __name__ == "__main__":
    signal(SIGPIPE, SIG_DFL)  # https://stackoverflow.com/a/30091579
    try:
        main()
    except KeyboardInterrupt:
        pass
