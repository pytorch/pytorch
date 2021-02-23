#!/usr/bin/env python3

import argparse
import bz2
import json
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import boto3  # type: ignore[import]
import botocore  # type: ignore[import]


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


def get_ossci_jsons(
    *,
    bucket: Any,
    sha: str,
    jobs: Optional[List[str]]
) -> Dict[str, Any]:
    prefix = f"test_time/{sha}/"
    objs: List[Any]
    if jobs is None:
        objs = list(bucket.objects.filter(Prefix=prefix))
    else:
        objs = []
        for job in jobs:
            objs.extend(list(bucket.objects.filter(Prefix=f"{prefix}{job}/")))
    # initial pass to avoid downloading more than necessary
    # in the case where there are multiple reports for a single sha+job
    uniqueified = {obj.key.split('/')[2]: obj for obj in objs}
    return {
        job: json.loads(bz2.decompress(obj.get()['Body'].read()))
        for job, obj in uniqueified.items()
    }


def get_case(
    *,
    data: Any,
    suite_name: str,
    test_name: str,
) -> Optional[Dict[str, Any]]:
    suite = data.get('suites', {}).get(suite_name)
    if suite:
        testcase_times = {
            case['name']: case
            for case in suite['cases']
        }
        return testcase_times.get(test_name)
    return None


def case_status(case: Dict[str, Any]) -> Optional[str]:
    for k in {'errored', 'failed', 'skipped'}:
        if case[k]:
            return k
    return None


def make_column(
    *,
    data: Any,
    suite_name: str,
    test_name: str,
    digits: int,
) -> str:
    decimals = 3
    num_length = digits + 1 + decimals
    case = get_case(data=data, suite_name=suite_name, test_name=test_name)
    if case:
        status = case_status(case)
        if status:
            return f'{status.rjust(num_length)} '
        else:
            return f'{case["seconds"]:{num_length}.{decimals}f}s'
    return ' ' * (num_length + 1)


def make_columns(
    *,
    bucket: Any,
    sha: str,
    jobs: List[str],
    suite_name: str,
    test_name: str,
    digits: int,
) -> str:
    jsons = get_ossci_jsons(bucket=bucket, sha=sha, jobs=jobs)
    return ' '.join(
        make_column(
            data=jsons.get(job, {}),
            suite_name=suite_name,
            test_name=test_name,
            digits=digits,
        )
        for job in jobs
    )


def make_lines(
    *,
    bucket: Any,
    sha: str,
    jobs: Optional[List[str]],
    suite_name: str,
    test_name: str,
) -> List[str]:
    jsons = get_ossci_jsons(bucket=bucket, sha=sha, jobs=jobs)
    lines = []
    for job, data in jsons.items():
        case = get_case(data=data, suite_name=suite_name, test_name=test_name)
        if case:
            status = case_status(case)
            lines.append(f'{job} {case["seconds"]} {status or ""}')
    return lines


def display_history(
    *,
    bucket: Any,
    commits: List[Tuple[str, datetime]],
    jobs: Optional[List[str]],
    suite_name: str,
    test_name: str,
    delta: int,
    mode: str,
    digits: int,
) -> None:
    any_yet = False
    prev_time = datetime.now()
    for sha, time in commits:
        if (prev_time - time).total_seconds() < delta * 3600:
            continue
        prev_time = time
        lines: List[str]
        if mode == 'columns':
            assert jobs is not None
            lines = [make_columns(
                bucket=bucket,
                sha=sha,
                jobs=jobs,
                suite_name=suite_name,
                test_name=test_name,
                digits=digits,
            )]
        else:
            assert mode == 'multiline'
            lines = make_lines(
                bucket=bucket,
                sha=sha,
                jobs=jobs,
                suite_name=suite_name,
                test_name=test_name,
            )
        if lines:
            any_yet = True
        elif not any_yet:
            lines = [f'no jobs found with test {suite_name}.{test_name}']
        for line in lines:
            print(f"{time} {sha} {line}".rstrip())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        __file__,
        description='Display the history of a test.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'mode',
        choices=['columns', 'multiline'],
        help='output format',
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
        default=12,
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
        default=[],
    )
    args = parser.parse_args()

    jobs = None if args.all else args.job
    if jobs == []:  # no jobs, and not None (which would mean all jobs)
        parser.error('No jobs specified.')

    commits = get_git_commit_history(path=args.pytorch, ref=args.ref)

    s3 = boto3.resource("s3", config=botocore.config.Config(signature_version=botocore.UNSIGNED))
    bucket = s3.Bucket('ossci-metrics')

    display_history(
        bucket=bucket,
        commits=commits,
        jobs=jobs,
        suite_name=args.suite,
        test_name=args.test,
        delta=args.delta,
        mode=args.mode,
        digits=args.digits,
    )
