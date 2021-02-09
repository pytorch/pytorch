#!/usr/bin/env python3

import argparse
import bz2
import json
import subprocess
from datetime import datetime
from typing import Any, List, Optional, Tuple

import boto3  # type: ignore[import]


def get_git_commit_history(
    path: str,
    branch: str = "master"
) -> List[Tuple[str, datetime]]:
    rc = subprocess.check_output(['git', '-C', path, 'log', '--pretty=format:%H %ct', branch]).decode("latin-1")
    return [(x[0], datetime.fromtimestamp(int(x[1]))) for x in [line.split(" ") for line in rc.split("\n")]]


def get_ossci_json(
    bucket: Any,
    sha: str,
    config: str = 'pytorch_linux_xenial_cuda10_2_cudnn7_py3_gcc7_test2'
) -> Any:
    objs = list(bucket.objects.filter(Prefix=f"test_time/{sha}/{config}"))
    if len(objs) == 0:
        return {}
    return json.loads(bz2.decompress(objs[0].get()['Body'].read()))


def search_for_commit(
    bucket: Any,
    commits: List[Tuple[str, datetime]],
    suite_name: str,
    test_name: str,
    delta: int,
) -> Optional[str]:
    last_sha = None
    prev_time = datetime.now()
    for sha, time in commits:
        if (prev_time - time).total_seconds() < delta * 3600:
            continue
        data = get_ossci_json(bucket, sha)
        suites = data['suites'] if 'suites' in data else {}
        if suite_name not in suites:
            data = get_ossci_json(bucket, sha, 'pytorch_linux_xenial_cuda10_2_cudnn7_py3_gcc7_test')
            suites = data['suites'] if 'suites' in data else {}
            if suite_name not in suites:
                print(f"{time} {sha}: Can't find {suite_name} in {suites.keys()}")
                continue
        testcase_times = {case['name']: case['seconds'] for case in suites[suite_name]['cases']}
        if test_name not in testcase_times:
            break
        last_sha = sha
        prev_time = time
        print(f"{time} {sha} {testcase_times[test_name]}")
    return last_sha


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
    return parsed


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
        '--delta',
        type=positive_integer,
        default=12,
        help='minimum number of hours between rows',
    )
    parser.add_argument(
        '--digits',
        type=positive_integer,
        default=3,
        help='number of digits to display before the decimal point',
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

    pytorch_git_path = args.pytorch or '.'
    commits = get_git_commit_history(pytorch_git_path)

    s3 = boto3.resource("s3")
    bucket = s3.Bucket('ossci-metrics')

    last_sha = search_for_commit(bucket, commits, args.suite, args.test, delta=args.delta)
