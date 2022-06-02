from typing import Any
from datetime import datetime, timedelta
from gitutils import _check_output

from rockset import Client, ParamDict  # type: ignore[import]
import os

rs = Client(api_key=os.getenv("ROCKSET_API_KEY", None))
qlambda = rs.QueryLambda.retrieve(
    'commit_jobs_query',
    version='c2a4dbce081d0144',
    workspace='commons')

def parse_args() -> Any:
    from argparse import ArgumentParser
    parser = ArgumentParser("Print latest commits")
    parser.add_argument("--minutes", type=int, default=30, help="duration in minutes of last commits")
    return parser.parse_args()

def print_latest_commits(minutes: int = 30) -> None:
    current_time = datetime.now()
    time_since = current_time - timedelta(minutes=minutes)
    timestamp_since = datetime.timestamp(time_since)
    commits = _check_output(
        [
            "git",
            "rev-list",
            f"--max-age={timestamp_since}",
            "--remotes=*origin/master",
        ],
        encoding="ascii",
    ).splitlines()

    for commit in commits:
        print(commit)
        print_commit_status(commit)

def print_commit_status(sha: str) -> None:
    params = ParamDict()
    params['sha'] = sha
    results = qlambda.execute(parameters=params)
    for check in results['results']:
        print(f"\t{check['conclusion']:>10}: {check['name']}")

def main() -> None:
    args = parse_args()
    print_latest_commits(args.minutes)

if __name__ == "__main__":
    main()
