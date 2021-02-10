#!/usr/bin/env python3

import argparse
import bz2
import json
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import boto3  # type: ignore[import]


def get_git_commit_history(
    *,
    path: str,
    ref: str
) -> List[Tuple[str, datetime]]:
    rc = subprocess.check_output(
        ['git', '-C', path, 'log', '--pretty=format:%H %ct', ref],
    ).decode("latin-1")
    return [
        (x[0], datetime.fromtimestamp(int(x[1])))
        for x in [line.split(" ") for line in rc.split("\n")]
    ]


def get_ossci_json(
    *,
    bucket: Any,
    sha: str,
    job: str
) -> Any:
    objs = list(bucket.objects.filter(Prefix=f"test_time/{sha}/{job}"))
    if len(objs) == 0:
        return {}
    return json.loads(bz2.decompress(objs[0].get()['Body'].read()))


def case_status(case: Dict[str, Any]) -> Optional[str]:
    for k in {'errored', 'failed', 'skipped'}:
        if case[k]:
            return k
    return None


def make_column(
    *,
    bucket: Any,
    sha: str,
    job: str,
    suite_name: str,
    test_name: str,
    digits: int,
) -> str:
    decimals = 3
    num_length = digits + 1 + decimals

    data = get_ossci_json(bucket=bucket, sha=sha, job=job)
    suite = data.get('suites', {}).get(suite_name)
    if suite:
        testcase_times = {
            case['name']: case
            for case in suite['cases']
        }
        case = testcase_times.get(test_name)
        if case:
            status = case_status(case)
            if status:
                return f'{status.rjust(num_length)} '
            else:
                return f'{case["seconds"]:{num_length}.{decimals}f}s'
    return ' ' * (num_length + 1)


def display_history(
    *,
    bucket: Any,
    commits: List[Tuple[str, datetime]],
    jobs: List[str],
    suite_name: str,
    test_name: str,
    delta: int,
    digits: int,
) -> None:
    prev_time = datetime.now()
    for sha, time in commits:
        if (prev_time - time).total_seconds() < delta * 3600:
            continue
        prev_time = time
        columns = [
            make_column(
                bucket=bucket,
                sha=sha,
                job=job,
                suite_name=suite_name,
                test_name=test_name,
                digits=digits,
            )
            for job in jobs
        ]
        print(f"{time} {sha} {' '.join(columns)}".rstrip())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        __file__,
        description='Display the history of a test.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        help='minimum number of hours between rows',
        default=12,
    )
    parser.add_argument(
        '--digits',
        type=int,
        help='number of digits to display before the decimal point',
        default=4,
    )
    parser.add_argument(
        'suite',
        help='name of the suite containing the test',
    )
    parser.add_argument(
        'test',
        help='name of the test',
    )
    parser.add_argument(
        'job',
        nargs='*',
        help='names of jobs to display columns for, in order',
    )
    args = parser.parse_args()

    commits = get_git_commit_history(path=args.pytorch, ref=args.ref)

    s3 = boto3.resource("s3")
    bucket = s3.Bucket('ossci-metrics')

    display_history(
        bucket=bucket,
        commits=commits,
        jobs=args.job,
        suite_name=args.suite,
        test_name=args.test,
        delta=args.delta,
        digits=args.digits,
    )
