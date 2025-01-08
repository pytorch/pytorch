import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
from gitutils import retries_decorator


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from tools.testing.clickhouse import query_clickhouse


sys.path.pop(0)


LOGS_QUERY = """
with
    shas as (
        SELECT
            distinct
            push.head_commit.id as sha
        FROM
            -- Not bothering with final here
            default.push
        WHERE
            push.ref = 'refs/heads/viable/strict'
            AND push.repository.'full_name' = 'pytorch/pytorch'
        ORDER BY
            push.head_commit.'timestamp' desc
        LIMIT
            5
    )
select
    id,
    name
from
    default.workflow_job j final
    join shas on shas.sha = j.head_sha
where
    j.id in (select id from materialized_views.workflow_job_by_head_sha where head_sha in (select sha from shas))
    and j.name like '% / test%'
    and j.name not like '%rerun_disabled_tests%'
    and j.name not like '%mem_leak_check%'
"""

TEST_EXISTS_QUERY = """
select
    name
from
    default.test_run_s3
where
    name::String like {name: String}
    and classname like {classname: String}
    and time_inserted > CURRENT_TIMESTAMP() - INTERVAL 7 DAY
limit 1
"""

CLOSING_COMMENT = (
    "I cannot find any mention of this test in the database for the past 7 days "
    "or in the logs for the past 5 commits on viable/strict.  Closing this "
    "issue as it is highly likely that this test has either been renamed or "
    "removed.  If you think this is a false positive, please feel free to "
    "re-open this issue."
)

DISABLED_TESTS_JSON = (
    "https://ossci-metrics.s3.amazonaws.com/disabled-tests-condensed.json"
)


@retries_decorator()
def query_db(query: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    return query_clickhouse(query, params)


def parse_args() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list the tests.",
    )
    return parser.parse_args()


def download_log_worker(temp_dir: str, id: int, name: str) -> None:
    url = f"https://ossci-raw-job-status.s3.amazonaws.com/log/{id}"
    data = requests.get(url).text
    with open(f"{temp_dir}/{name.replace('/', '_')} {id}.txt", "x") as f:
        f.write(data)


def printer(item: Tuple[str, Tuple[int, str, List[Any]]], extra: str) -> None:
    test, (_, link, _) = item
    print(f"{link:<55} {test:<120} {extra}")


def close_issue(num: int) -> None:
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {os.environ['GITHUB_TOKEN']}",
    }
    requests.post(
        f"https://api.github.com/repos/pytorch/pytorch/issues/{num}/comments",
        data=json.dumps({"body": CLOSING_COMMENT}),
        headers=headers,
    )
    requests.patch(
        f"https://api.github.com/repos/pytorch/pytorch/issues/{num}",
        data=json.dumps({"state": "closed"}),
        headers=headers,
    )


def check_if_exists(
    item: Tuple[str, Tuple[int, str, List[str]]], all_logs: List[str]
) -> Tuple[bool, str]:
    test, (_, link, _) = item
    # Test names should look like `test_a (module.path.classname)`
    reg = re.match(r"(\S+) \((\S*)\)", test)
    if reg is None:
        return False, "poorly formed"

    name = reg[1]
    classname = reg[2].split(".")[-1]

    # Check if there is any mention of the link or the test name in the logs.
    # The link usually shows up in the skip reason.
    present = False
    for log in all_logs:
        if link in log:
            present = True
            break
        if f"{classname}::{name}" in log:
            present = True
            break
    if present:
        return True, "found in logs"

    # Query DB to see if the test is there
    count = query_db(
        TEST_EXISTS_QUERY, {"name": f"{name}%", "classname": f"{classname}%"}
    )
    if len(count) == 0:
        return False, "not found"
    return True, "found in DB"


if __name__ == "__main__":
    args = parse_args()
    disabled_tests_json = json.loads(requests.get(DISABLED_TESTS_JSON).text)

    all_logs = []
    jobs = query_db(LOGS_QUERY, {})
    with tempfile.TemporaryDirectory() as temp_dir:
        pool = mp.Pool(20)
        for job in jobs:
            id = job["id"]
            name = job["name"]
            pool.apply_async(download_log_worker, args=(temp_dir, id, name))
        pool.close()
        pool.join()

        for filename in os.listdir(temp_dir):
            with open(f"{temp_dir}/{filename}") as f:
                all_logs.append(f.read())

    # If its less than 200 something definitely went wrong.
    assert len(all_logs) > 200
    assert len(all_logs) == len(jobs)

    to_be_closed = []
    for item in disabled_tests_json.items():
        exists, reason = check_if_exists(item, all_logs)
        printer(item, reason)
        if not exists:
            to_be_closed.append(item)

    print(f"There are {len(to_be_closed)} issues that will be closed:")
    for item in to_be_closed:
        printer(item, "")

    if args.dry_run:
        print("dry run, not actually closing")
    else:
        for item in to_be_closed:
            _, (num, _, _) = item
            close_issue(num)
