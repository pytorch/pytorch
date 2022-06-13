from typing import Any, Dict, List, NamedTuple
from gitutils import _check_output

import rockset  # type: ignore[import]
import os
import re

regex = [
    "^pull+",
    "^trunk+",
    "^lint+",
    "^linux-binary+",
    "^android-tests+",
    "^windows-binary+"
]

class WorkflowCheck(NamedTuple):
    workflowName: str
    name: str
    jobName: str
    conclusion: str

def get_latest_commits() -> List[str]:
    latest_viable_commit = _check_output(
        [
            "git",
            "log",
            "-n",
            "1",
            "--pretty=format:%H",
            "origin/viable/strict",
        ],
        encoding="ascii",
    )
    commits = _check_output(
        [
            "git",
            "rev-list",
            f"{latest_viable_commit}^..HEAD",
            "--remotes=*origin/master",
        ],
        encoding="ascii",
    ).splitlines()

    return commits

def query_commits(commits: List[str], qlambda: Any) -> Any:
    params = rockset.ParamDict()
    params['shas'] = ",".join(commits)
    results = qlambda.execute(parameters=params)

    return results

def print_commit_status(commit: str, results: Dict[str, Any]) -> None:
    print(commit)
    for check in results['results']:
        if check['sha'] == commit:
            print(f"\t{check['conclusion']:>10}: {check['name']}")

def get_commit_results(commit: str, results: Dict[str, Any]) -> List[Dict[str, Any]]:
    workflow_checks = []
    for check in results['results']:
        if check['sha'] == commit:
            workflow_checks.append(WorkflowCheck(
                workflowName=check['workflowName'],
                name=check['name'],
                jobName=check['jobName'],
                conclusion=check['conclusion'],
            )._asdict())
    return workflow_checks

def isGreen(commit: str, results: Dict[str, Any]) -> bool:
    workflow_checks = get_commit_results(commit, results)

    for check in workflow_checks:
        workflowName = check['workflowName']
        conclusion = check['conclusion']
        if re.search("|".join(regex), workflowName, flags=re.IGNORECASE) and conclusion != 'success':
            if check['name'] == "pull / win-vs2019-cuda11.3-py3" and conclusion == 'skipped':
                pass
                # there are trunk checks that run the same tests, so this pull workflow check can be skipped
            else:
                return False
        elif workflowName in ["periodic", "docker-release-builds"] and conclusion not in ["success", "skipped"]:
            return False
    return True

def get_latest_green_commit(commits: List[str], results: Dict[str, Any]) -> Any:
    for commit in commits:
        if isGreen(commit, results):
            return commit
    return None

def main() -> None:
    rs = rockset.Client(
        api_server="api.rs2.usw2.rockset.com", api_key=os.environ["ROCKSET_API_KEY"]
    )
    qlambda = rs.QueryLambda.retrieve(
        'commit_jobs_batch_query',
        version='15aba20837ae9d75',
        workspace='commons')

    commits = get_latest_commits()
    results = query_commits(commits, qlambda)

    latest_viable_commit = get_latest_green_commit(commits, results)
    print(latest_viable_commit)

if __name__ == "__main__":
    main()
