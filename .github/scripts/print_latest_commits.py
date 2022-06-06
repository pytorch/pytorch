from typing import Any, Dict, List, NamedTuple
from datetime import datetime, timedelta
from gitutils import _check_output

from rockset import Client, ParamDict  # type: ignore[import]
import os

class workflowCheck(NamedTuple):
    workflowName: str
    name: str
    jobName: str
    conclusion: str

rs = Client(api_key=os.getenv("ROCKSET_API_KEY", None))
qlambda = rs.QueryLambda.retrieve(
    'commit_jobs_batch_query',
    version='15aba20837ae9d75',
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

    params = ParamDict()
    params['shas'] = ",".join(commits)
    results = qlambda.execute(parameters=params)

    for commit in commits:
        print_commit_status(commit, results)

def print_commit_status(commit: str, results: Dict[str, Any]) -> None:
    print(commit)
    for check in results['results']:
        if check['sha'] == commit:
            print(f"\t{check['conclusion']:>10}: {check['name']}")

def get_commit_results(commit: str, results: Dict[str, Any]) -> List[Dict[str, Any]]:
    workflow_checks = []
    for check in results['results']:
        if check['sha'] == commit:
            workflow_checks.append(workflowCheck(
                workflowName=check['workflowName'],
                name=check['name'],
                jobName=check['jobName'],
                conclusion=check['conclusion'],
            )._asdict())
    return workflow_checks

def isGreen(results: List[workflowCheck]) -> bool:
    return True

def main() -> None:
    args = parse_args()
    print_latest_commits(args.minutes)

if __name__ == "__main__":
    main()
