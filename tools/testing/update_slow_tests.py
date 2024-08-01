import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests
import rockset  # type: ignore[import]


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
QUERY = """
WITH most_recent_strict_commits AS (
    SELECT
        push.head_commit.id as sha,
    FROM
        commons.push
    WHERE
        push.ref = 'refs/heads/viable/strict'
        AND push.repository.full_name = 'pytorch/pytorch'
    ORDER BY
        push._event_time DESC
    LIMIT
        3
), workflows AS (
    SELECT
        id
    FROM
        commons.workflow_run w
        INNER JOIN most_recent_strict_commits c on w.head_sha = c.sha
    WHERE
        w.name != 'periodic'
),
job AS (
    SELECT
        j.id
    FROM
        commons.workflow_job j
        INNER JOIN workflows w on w.id = j.run_id
    WHERE
        j.name NOT LIKE '%asan%'
),
duration_per_job AS (
    SELECT
        test_run.classname,
        test_run.name,
        job.id,
        SUM(time) as time
    FROM
        commons.test_run_s3 test_run
        /* `test_run` is ginormous and `job` is small, so lookup join is essential */
        INNER JOIN job ON test_run.job_id = job.id HINT(join_strategy = lookup)
    WHERE
        /* cpp tests do not populate `file` for some reason. */
        /* Exclude them as we don't include them in our slow test infra */
        test_run.file IS NOT NULL
        /* do some more filtering to cut down on the test_run size */
        AND test_run.skipped IS NULL
        AND test_run.failure IS NULL
        AND test_run.error IS NULL
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

if __name__ == "__main__":
    if (
        "ROCKSET_API_KEY" not in os.environ
        or "PYTORCHBOT_TOKEN" not in os.environ
        or "UPDATEBOT_TOKEN" not in os.environ
    ):
        print("env keys are not set")
        sys.exit(1)

    rs_client = rockset.RocksetClient(
        host="api.usw2a1.rockset.com", api_key=os.environ["ROCKSET_API_KEY"]
    )

    results = rs_client.sql(QUERY).results
    slow_tests = {row["test_name"]: row["avg_duration_sec"] for row in results}

    with open(REPO_ROOT / "test" / "slow_tests.json", "w") as f:
        json.dump(slow_tests, f, indent=2)

    branch_name = f"update_slow_tests_{int(time.time())}"

    subprocess.run(["git", "checkout", "-b", branch_name], cwd=REPO_ROOT)
    subprocess.run(["git", "add", "test/slow_tests.json"], cwd=REPO_ROOT)
    subprocess.run(["git", "commit", "-m", "Update slow tests"], cwd=REPO_ROOT)
    subprocess.run(["git", "push", "origin", branch_name], cwd=REPO_ROOT)

    params = {
        "title": "Update slow tests",
        "head": branch_name,
        "base": "main",
        "body": "This PR is auto-generated weekly by [this action](https://github.com/pytorch/pytorch/blob/main/"
        + ".github/workflows/weeekly.yml).\nUpdate the list of slow tests.",
    }
    result = requests.post(
        "https://api.github.com/repos/pytorch/pytorch/pulls",
        data=json.dumps(params),
        headers={
            "Authorization": f"token {os.environ['UPDATEBOT_TOKEN']}",
            "Accept": "application/vnd.github.v3+json",
        },
    ).json()
    print(result)
    requests.post(
        f"https://api.github.com/repos/pytorch/pytorch/pulls/{result['number']}/reviews",
        data=json.dumps({"event": "APPROVE"}),
        headers={
            "Authorization": f"token {os.environ['PYTORCHBOT_TOKEN']}",
            "Accept": "application/vnd.github.v3+json",
        },
    )
    time.sleep(5)
    requests.post(
        f"https://api.github.com/repos/pytorch/pytorch/issues/{result['number']}/comments",
        data=json.dumps({"body": "@pytorchbot merge"}),
        headers={
            "Authorization": f"token {os.environ['PYTORCHBOT_TOKEN']}",
            "Accept": "application/vnd.github.v3+json",
        },
    )
