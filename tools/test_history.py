#!/usr/bin/env python3

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from typing import Dict, Iterator, List, Optional, Set, Tuple

from tools.stats_utils.s3_stat_parser import (Report, get_cases,
                                              get_test_stats_summaries)


def get_git_commit_history(
    *,
    path: str,
    ref: str
) -> List[Tuple[str, datetime]]:
    rc = subprocess.check_output(
        ['git', '-C', path, 'log', '--pretty=format:%H %ct', ref],
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
            data=data,
            filename=filename,
            suite_name=suite_name,
            test_name=test_name
        )
        if cases:
            case = cases[0]
            status = case['status']
            omitted = len(cases) - 1
            if status:
                return f'{status.rjust(num_length)} ', omitted
            else:
                return f'{case["seconds"]:{num_length}.{decimals}f}s', omitted
        else:
            return f'{"absent".rjust(num_length)} ', 0
    else:
        return ' ' * (num_length + 1), 0


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
        columns.append(f'({total_omitted} job re-runs omitted)')
    if total_suites > 0:
        columns.append(f'({total_suites} matching suites omitted)')
    return ' '.join(columns)


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
                status = case['status']
                line = f'{job} {case["seconds"]}s{f" {status}" if status else ""}'
                if len(cases) > 1:
                    line += f' ({len(cases) - 1} matching suites omitted)'
                lines.append(line)
            elif job in jobs:
                lines.append(f'{job} (test not found)')
    if lines:
        return lines
    else:
        return ['(no reports in S3)']


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
        if mode == 'columns':
            assert jobs is not None
            # we assume that get_test_stats_summaries here doesn't
            # return empty lists
            omitted = {
                job: len(l) - 1
                for job, l in summaries.items()
                if len(l) > 1
            }
            lines = [make_columns(
                jobs=jobs,
                jsons={job: l[0] for job, l in summaries.items()},
                omitted=omitted,
                filename=filename,
                suite_name=suite_name,
                test_name=test_name,
                digits=digits,
            )]
        else:
            assert mode == 'multiline'
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
    return r'''
Display the history of a test.

Each line of (non-error) output starts with the timestamp and SHA1 hash
of the commit it refers to, in this format:

    YYYY-MM-DD hh:mm:ss 0123456789abcdef0123456789abcdef01234567

In multiline mode, each line next includes the name of a CircleCI job,
followed by the time of the specified test in that job at that commit.
Example:

    $ tools/test_history.py --mode=multiline --ref=594a66 --sha-length=8 --test=test_set_dir \
      --job pytorch_linux_xenial_py3_6_gcc5_4_test --job pytorch_linux_xenial_py3_6_gcc7_test
    2021-02-10 11:13:34Z 594a66d7 pytorch_linux_xenial_py3_6_gcc5_4_test 0.36s
    2021-02-10 11:13:34Z 594a66d7 pytorch_linux_xenial_py3_6_gcc7_test 0.573s errored
    2021-02-10 10:13:25Z 9c0caf03 pytorch_linux_xenial_py3_6_gcc5_4_test 0.819s
    2021-02-10 10:13:25Z 9c0caf03 pytorch_linux_xenial_py3_6_gcc7_test 0.449s
    2021-02-10 10:09:14Z 602434bc pytorch_linux_xenial_py3_6_gcc5_4_test 0.361s
    2021-02-10 10:09:14Z 602434bc pytorch_linux_xenial_py3_6_gcc7_test 0.454s
    2021-02-10 10:09:10Z 2e35fe95 (no reports in S3)
    2021-02-10 10:09:07Z ff73be7e (no reports in S3)
    2021-02-10 10:05:39Z 74082f0d (no reports in S3)
    2021-02-10 07:42:29Z 0620c96f pytorch_linux_xenial_py3_6_gcc5_4_test 0.414s
    2021-02-10 07:42:29Z 0620c96f pytorch_linux_xenial_py3_6_gcc5_4_test 0.476s
    2021-02-10 07:42:29Z 0620c96f pytorch_linux_xenial_py3_6_gcc7_test 0.377s
    2021-02-10 07:42:29Z 0620c96f pytorch_linux_xenial_py3_6_gcc7_test 0.326s

Another multiline example, this time with the --all flag:

    $ tools/test_history.py --mode=multiline --all --ref=321b9 --delta=12 --sha-length=8 \
      --test=test_qr_square_many_batched_complex_cuda
    2021-01-07 10:04:56Z 321b9883 pytorch_linux_xenial_cuda10_2_cudnn7_py3_gcc7_test2 424.284s
    2021-01-07 10:04:56Z 321b9883 pytorch_linux_xenial_cuda10_2_cudnn7_py3_slow_test 0.006s skipped
    2021-01-07 10:04:56Z 321b9883 pytorch_linux_xenial_cuda11_1_cudnn8_py3_gcc7_test 402.572s
    2021-01-07 10:04:56Z 321b9883 pytorch_linux_xenial_cuda9_2_cudnn7_py3_gcc7_test 287.164s
    2021-01-06 20:58:28Z fcb69d2e pytorch_linux_xenial_cuda10_2_cudnn7_py3_gcc7_test2 436.732s
    2021-01-06 20:58:28Z fcb69d2e pytorch_linux_xenial_cuda10_2_cudnn7_py3_slow_test 0.006s skipped
    2021-01-06 20:58:28Z fcb69d2e pytorch_linux_xenial_cuda11_1_cudnn8_py3_gcc7_test 407.616s
    2021-01-06 20:58:28Z fcb69d2e pytorch_linux_xenial_cuda9_2_cudnn7_py3_gcc7_test 287.044s

In columns mode, the name of the job isn't printed, but the order of the
columns is guaranteed to match the order of the jobs passed on the
command line. Example:

    $ tools/test_history.py --mode=columns --ref=3cf783 --sha-length=8 --test=test_set_dir \
      --job pytorch_linux_xenial_py3_6_gcc5_4_test --job pytorch_linux_xenial_py3_6_gcc7_test
    2021-02-10 12:18:50Z 3cf78395    0.644s    0.312s
    2021-02-10 11:13:34Z 594a66d7    0.360s  errored
    2021-02-10 10:13:25Z 9c0caf03    0.819s    0.449s
    2021-02-10 10:09:14Z 602434bc    0.361s    0.454s
    2021-02-10 10:09:10Z 2e35fe95
    2021-02-10 10:09:07Z ff73be7e
    2021-02-10 10:05:39Z 74082f0d
    2021-02-10 07:42:29Z 0620c96f    0.414s    0.377s (2 job re-runs omitted)
    2021-02-10 07:27:53Z 33afb5f1    0.381s    0.294s

Minor note: in columns mode, a blank cell means that no report was found
in S3, while the word "absent" means that a report was found but the
indicated test was not found in that report.
'''


def parse_args(raw: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        __file__,
        description=description(),
        formatter_class=HelpFormatter,
    )
    parser.add_argument(
        '--mode',
        choices=['columns', 'multiline'],
        help='output format',
        default='columns',
    )
    parser.add_argument(
        '--pytorch',
        help='path to local PyTorch clone',
        default='.',
    )
    parser.add_argument(
        '--ref',
        help='starting point (most recent Git ref) to display history for',
        default='master',
    )
    parser.add_argument(
        '--delta',
        type=int,
        help='minimum number of hours between commits',
        default=0,
    )
    parser.add_argument(
        '--sha-length',
        type=int,
        help='length of the prefix of the SHA1 hash to show',
        default=40,
    )
    parser.add_argument(
        '--digits',
        type=int,
        help='(columns) number of digits to display before the decimal point',
        default=4,
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='(multiline) ignore listed jobs, show all jobs for each commit',
    )
    parser.add_argument(
        '--file',
        help='name of the file containing the test',
    )
    parser.add_argument(
        '--suite',
        help='name of the suite containing the test',
    )
    parser.add_argument(
        '--test',
        help='name of the test',
        required=True
    )
    parser.add_argument(
        '--job',
        help='names of jobs to display columns for, in order',
        action='append',
        default=[],
    )
    args = parser.parse_args(raw)

    args.jobs = None if args.all else args.job
    # We dont allow implicit or empty "--jobs", unless "--all" is specified.
    if args.jobs == []:
        parser.error('No jobs specified.')

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
        print(line)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
