import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, cast, Optional

import requests
from clickhouse import query_clickhouse  # type: ignore[import]


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
QUERY = """
WITH most_recent_strict_commits AS (
    SELECT
        distinct push.head_commit.'id' as sha
    FROM
        -- not bothering with final
        default.push
    WHERE
        push.ref = 'refs/heads/viable/strict'
        AND push.repository.'full_name' = 'pytorch/pytorch'
    ORDER BY
        push.head_commit.'timestamp' desc
    LIMIT
        3
), workflows AS (
    SELECT
        id
    FROM
        default.workflow_run w final
    WHERE
        w.id in (select id from materialized_views.workflow_run_by_head_sha
            where head_sha in (select sha from most_recent_strict_commits)
        )
        and w.name != 'periodic'
),
job AS (
    SELECT
        j.id as id
    FROM
        default.workflow_job j final
    WHERE
        j.run_id in (select id from workflows)
        and j.name NOT LIKE '%asan%'
),
duration_per_job AS (
    SELECT
        test_run.classname as classname,
        test_run.name as name,
        job.id as id,
        SUM(test_run.time) as time
    FROM
        default.test_run_s3 test_run
        INNER JOIN job ON test_run.job_id = job.id
    WHERE
        /* cpp tests do not populate `file` for some reason. */
        /* Exclude them as we don't include them in our slow test infra */
        test_run.file != ''
        /* do some more filtering to cut down on the test_run size */
        AND empty(test_run.skipped)
        AND empty(test_run.failure)
        AND empty(test_run.error)
        and test_run.job_id in (select id from job)
    GROUP BY
        test_run.classname,
        test_run.name,
        job.id
)
SELECT
    CONCAT(
        name,
        ' (__main__.',
        classname,
        ')'
    ) as test_name,
    AVG(time) as avg_duration_sec
FROM
    duration_per_job
GROUP BY
    CONCAT(
        name,
        ' (__main__.',
        classname,
        ')'
    )
HAVING
    AVG(time) > 60.0
ORDER BY
    test_name
"""


UPDATEBOT_TOKEN = os.environ["UPDATEBOT_TOKEN"]
PYTORCHBOT_TOKEN = os.environ["PYTORCHBOT_TOKEN"]


def git_api(
    url: str, params: dict[str, Any], type: str = "get", token: str = UPDATEBOT_TOKEN
) -> Any:
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {token}",
    }
    if type == "post":
        return requests.post(
            f"https://api.github.com{url}",
            data=json.dumps(params),
            headers=headers,
        ).json()
    elif type == "patch":
        return requests.patch(
            f"https://api.github.com{url}",
            data=json.dumps(params),
            headers=headers,
        ).json()
    else:
        return requests.get(
            f"https://api.github.com{url}",
            params=params,
            headers=headers,
        ).json()


def make_pr(source_repo: str, params: dict[str, Any]) -> int:
    response = git_api(f"/repos/{source_repo}/pulls", params, type="post")
    print(f"made pr {response['html_url']}")
    return cast(int, response["number"])


def approve_pr(source_repo: str, pr_number: int) -> None:
    params = {"event": "APPROVE"}
    # use pytorchbot to approve the pr
    git_api(
        f"/repos/{source_repo}/pulls/{pr_number}/reviews",
        params,
        type="post",
        token=PYTORCHBOT_TOKEN,
    )


def make_comment(source_repo: str, pr_number: int, msg: str) -> None:
    params = {"body": msg}
    # comment with pytorchbot because pytorchmergebot gets ignored
    git_api(
        f"/repos/{source_repo}/issues/{pr_number}/comments",
        params,
        type="post",
        token=PYTORCHBOT_TOKEN,
    )


def add_labels(source_repo: str, pr_number: int, labels: list[str]) -> None:
    params = {"labels": labels}
    git_api(
        f"/repos/{source_repo}/issues/{pr_number}/labels",
        params,
        type="post",
    )


def search_for_open_pr(
    source_repo: str, search_string: str
) -> Optional[tuple[int, str]]:
    params = {
        "q": f"is:pr is:open in:title author:pytorchupdatebot repo:{source_repo} {search_string}",
        "sort": "created",
    }
    response = git_api("/search/issues", params)
    if response["total_count"] != 0:
        # pr does exist
        pr_num = response["items"][0]["number"]
        link = response["items"][0]["html_url"]
        response = git_api(f"/repos/{source_repo}/pulls/{pr_num}", {})
        branch_name = response["head"]["ref"]
        print(
            f"pr does exist, number is {pr_num}, branch name is {branch_name}, link is {link}"
        )
        return pr_num, branch_name
    return None


if __name__ == "__main__":
    results = query_clickhouse(QUERY, {})
    slow_tests = {row["test_name"]: row["avg_duration_sec"] for row in results}

    with open(REPO_ROOT / "test" / "slow_tests.json", "w") as f:
        json.dump(slow_tests, f, indent=2)

    branch_name = f"update_slow_tests_{int(time.time())}"
    pr_num = None

    open_pr = search_for_open_pr("pytorch/pytorch", "Update slow tests")
    if open_pr is not None:
        pr_num, branch_name = open_pr

    subprocess.run(["git", "checkout", "-b", branch_name], cwd=REPO_ROOT)
    subprocess.run(["git", "add", "test/slow_tests.json"], cwd=REPO_ROOT)
    subprocess.run(["git", "commit", "-m", "Update slow tests"], cwd=REPO_ROOT)
    subprocess.run(
        f"git push --set-upstream origin {branch_name} -f".split(), cwd=REPO_ROOT
    )

    params = {
        "title": "Update slow tests",
        "head": branch_name,
        "base": "main",
        "body": "This PR is auto-generated weekly by [this action](https://github.com/pytorch/pytorch/blob/main/"
        + ".github/workflows/weekly.yml).\nUpdate the list of slow tests.",
    }
    if pr_num is None:
        # no existing pr, so make a new one and approve it
        pr_num = make_pr("pytorch/pytorch", params)
        time.sleep(5)
        add_labels("pytorch/pytorch", pr_num, ["ciflow/slow", "ci-no-td"])
        approve_pr("pytorch/pytorch", pr_num)
    make_comment("pytorch/pytorch", pr_num, "@pytorchbot merge")
