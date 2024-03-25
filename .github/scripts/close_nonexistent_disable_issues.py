import argparse
import json
import multiprocessing as mp
import os
import re
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import requests
import rockset  # type: ignore[import]
from gitutils import retries_decorator

LOGS_QUERY = """
with
    shas as (
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
            5
    )
select
    id,
    name
from
    workflow_job j
    join shas on shas.sha = j.head_sha
where
    j.name like '% / test%'
    and j.name not like '%rerun_disabled_tests%'
    and j.name not like '%mem_leak_check%'
"""

TEST_EXISTS_QUERY = """
select
    count(*) as c
from
    test_run_s3
where
    cast(name as string) like :name
    and classname like :classname
    and _event_time > CURRENT_TIMESTAMP() - DAYS(7)
"""

CLOSING_COMMENT = (
    "I cannot find any mention of this test in rockset for the past 7 days "
    "or in the logs for the past 5 commits on viable/strict.  Closing this "
    "issue as it is highly likely that this test has either been renamed or "
    "removed.  If you think this is a false positive, please feel free to "
    "re-open this issue."
)

DISABLED_TESTS_JSON = (
    "https://ossci-metrics.s3.amazonaws.com/disabled-tests-condensed.json"
)


def parse_args() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list the tests.",
    )
    return parser.parse_args()


@retries_decorator()
def query_rockset(
    query: str, params: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    res = rockset.RocksetClient(
        host="api.rs2.usw2.rockset.com", api_key=os.environ["ROCKSET_API_KEY"]
    ).sql(query, params)
    results: List[Dict[str, Any]] = res.results
    return results


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

    # Query rockset to see if the test is there
    count = query_rockset(
        TEST_EXISTS_QUERY, {"name": f"{name}%", "classname": f"{classname}%"}
    )
    if count[0]["c"] == 0:
        return False, "not found"
    return True, "found in rockset"


if __name__ == "__main__":
    args = parse_args()
    disabled_tests_json = json.loads(requests.get(DISABLED_TESTS_JSON).text)

    all_logs = []
    jobs = query_rockset(LOGS_QUERY)
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
