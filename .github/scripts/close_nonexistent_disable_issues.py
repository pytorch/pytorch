import argparse
import json
import multiprocessing as mp
import os
import re
import tempfile
import time
from typing import Any, Dict, List, Optional

import requests
import rockset

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
    "issue as it is highly likely that this test has either bee renamed or removed."
)


def parse_args() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list the tests.",
    )
    return parser.parse_args()


def query_rockset(
    query: str, params: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    res: List[Dict[str, Any]] = rockset.RocksetClient(
        host="api.rs2.usw2.rockset.com", api_key=os.environ["ROCKSET_API_KEY"]
    ).sql(query, params)
    return res.results


def _download_log_worker(temp_dir: str, id: int, name: str):
    url = f"https://ossci-raw-job-status.s3.amazonaws.com/log/{id}"
    data = requests.get(url).text
    with open(f"{temp_dir}/{name.replace('/', '_')} {id}.txt", "x") as f:
        f.write(data)


def printer(item, extra):
    test, (_, link, _) = item
    print(f"{link:<55} {test:<120} {extra}")


def close_issue(num: int):
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


if __name__ == "__main__":
    args = parse_args()
    disabled_tests_json = json.loads(
        requests.get(
            "https://raw.githubusercontent.com/pytorch/test-infra/generated-stats/stats/disabled-tests-condensed.json"
        ).text
    )

    al = []
    with tempfile.TemporaryDirectory() as temp_dir:
        jobs = query_rockset(LOGS_QUERY)
        os.mkdir("tempdir")
        print(len(jobs))
        pool = mp.Pool(20)
        for r in jobs:
            id = r["id"]
            name = r["name"]
            pool.apply_async(_download_log_worker, args=(temp_dir, id, name))
        pool.close()
        pool.join()
        assert len(os.listdir(temp_dir)) == len(jobs)

        for filename in os.listdir(temp_dir):
            with open(f"{temp_dir}/{filename}") as f:
                al.append(f.read())

    to_be_closed = []
    for item in disabled_tests_json.items():
        test, (num, link, _) = item
        reg = re.match("(\S+) \((\S*)\)", test)
        if reg is None:
            printer(item, "poorly formed")
            to_be_closed.append(item)
            continue
        name = reg[1]
        classname = reg[2].split(".")[-1]
        present = False
        for a in al:
            if link in a:
                present = True
                break
            if f"{classname}::{name}" in a:
                present = True
                break
        if present:
            printer(item, "found in logs")
            continue

        r = query_rockset(
            TEST_EXISTS_QUERY, {"name": f"{name}%", "classname": f"{classname}%"}
        )
        if r[0]["c"] == 0:
            printer(item, "not found")
            to_be_closed.append(item)
        else:
            printer(item, "found in rockset")

    print("The following issues will be closed:")
    for item in to_be_closed:
        printer(item, "")

    if args.dry_run:
        print("dry run, not actually closing")
    else:
        print("uh oh")
        exit(0)
        for item in to_be_closed:
            _, (num, _, _) = item
            close_issue(num)
