"""
Update commited CSV files used as reference points by dynamo/inductor CI.

Currently only cares about graph breaks, so only saves those columns.

Hardcodes a list of job names and artifacts per job, but builds the lookup
by querying github sha and finding associated github actions workflow ID and CI jobs,
downloading artifact zips, extracting CSVs and filtering them.

Usage:

python benchmarks/dynamo/ci_expected_accuracy.py <sha of pytorch commit that has completed inductor benchmark jobs>

Known limitations:
- doesn't handle 'retry' jobs in CI, if the same hash has more than one set of artifacts, gets the first one
"""

import argparse
import json
import os
import subprocess
import sys
import urllib
from io import BytesIO
from itertools import product
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd
import requests


"""
WITH job as (
    SELECT
        job.created_at as time,
        job.name as job_name,
        workflow.name as workflow_name,
        job.id as id,
        job.run_attempt as run_attempt,
        workflow.id as workflow_id
    FROM
        default.workflow_job job final
        INNER JOIN default.workflow_run workflow final on workflow.id = job.run_id
    WHERE
        job.name != 'ciflow_should_run'
        AND job.name != 'generate-test-matrix'
        -- Filter out workflow_run-triggered jobs, which have nothing to do with the SHA
        AND workflow.event != 'workflow_run'
        -- Filter out repository_dispatch-triggered jobs, which have nothing to do with the SHA
        AND workflow.event != 'repository_dispatch'
        AND workflow.head_sha = {sha: String}
        AND job.head_sha = {sha: String}
        AND workflow.repository.'full_name' = {repo: String}
)
SELECT
    workflow_name as workflowName,
    job_name as jobName,
    CAST(id as String) as id,
    run_attempt as runAttempt,
    CAST(workflow_id as String) as workflowId,
    time
from
    job
ORDER BY
    workflowName, jobName
"""
ARTIFACTS_QUERY_URL = (
    "https://console-api.clickhouse.cloud/.api/query-endpoints/"
    "c1cdfadc-6bb2-4a91-bbf9-3d19e1981cd4/run?format=JSON"
)
CSV_LINTER = str(
    Path(__file__).absolute().parent.parent.parent.parent
    / "tools/linter/adapters/no_merge_conflict_csv_linter.py"
)


def query_job_sha(repo, sha):
    params = {
        "queryVariables": {"sha": sha, "repo": repo},
    }
    # If you are a Meta employee, go to P1679979893 to get the id and secret.
    # Otherwise, ask a Meta employee give you the id and secret.
    KEY_ID = os.environ["CH_KEY_ID"]
    KEY_SECRET = os.environ["CH_KEY_SECRET"]

    r = requests.post(
        url=ARTIFACTS_QUERY_URL,
        data=json.dumps(params),
        headers={"Content-Type": "application/json"},
        auth=(KEY_ID, KEY_SECRET),
    )
    return r.json()["data"]


def parse_job_name(job_str):
    return (part.strip() for part in job_str.split("/"))


def parse_test_str(test_str):
    return (part.strip() for part in test_str[6:].strip(")").split(","))


S3_BASE_URL = "https://gha-artifacts.s3.amazonaws.com"


def get_artifacts_urls(results, suites):
    urls = {}
    for r in results:
        if (
            r["workflowName"] in ("inductor", "inductor-periodic")
            and "test" in r["jobName"]
        ):
            config_str, test_str = parse_job_name(r["jobName"])
            suite, shard_id, num_shards, machine, *_ = parse_test_str(test_str)
            workflowId = r["workflowId"]
            id = r["id"]
            runAttempt = r["runAttempt"]

            if suite in suites:
                artifact_filename = f"test-reports-test-{suite}-{shard_id}-{num_shards}-{machine}_{id}.zip"
                s3_url = f"{S3_BASE_URL}/{repo}/{workflowId}/{runAttempt}/artifact/{artifact_filename}"
                urls[(suite, int(shard_id))] = s3_url
                print(f"{suite} {shard_id}, {num_shards}: {s3_url}")
    return urls


def normalize_suite_filename(suite_name):
    strs = suite_name.split("_")
    subsuite = strs[-1]
    if "timm" in subsuite:
        subsuite = subsuite.replace("timm", "timm_models")

    return subsuite


def download_artifacts_and_extract_csvs(urls):
    dataframes = {}
    for (suite, shard), url in urls.items():
        try:
            resp = urlopen(url)
            subsuite = normalize_suite_filename(suite)
            artifact = ZipFile(BytesIO(resp.read()))
            for phase in ("training", "inference"):
                name = f"test/test-reports/{phase}_{subsuite}.csv"
                try:
                    df = pd.read_csv(artifact.open(name))
                    df["graph_breaks"] = df["graph_breaks"].fillna(0).astype(int)
                    prev_df = dataframes.get((suite, phase), None)
                    dataframes[(suite, phase)] = (
                        pd.concat([prev_df, df]) if prev_df is not None else df
                    )
                except KeyError:
                    print(
                        f"Warning: Unable to find {name} in artifacts file from {url}, continuing"
                    )
        except urllib.error.HTTPError:
            print(f"Unable to download {url}, perhaps the CI job isn't finished?")

    return dataframes


def write_filtered_csvs(root_path, dataframes):
    for (suite, phase), df in dataframes.items():
        out_fn = os.path.join(root_path, f"{suite}_{phase}.csv")
        df.to_csv(out_fn, index=False, columns=["name", "accuracy", "graph_breaks"])
        apply_lints(out_fn)


def apply_lints(filename):
    patch = json.loads(subprocess.check_output([sys.executable, CSV_LINTER, filename]))
    if patch.get("replacement"):
        with open(filename) as fd:
            data = fd.read().replace(patch["original"], patch["replacement"])
        with open(filename, "w") as fd:
            fd.write(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("sha")
    args = parser.parse_args()

    repo = "pytorch/pytorch"

    suites = {
        f"{a}_{b}"
        for a, b in product(
            [
                "aot_eager",
                "aot_inductor",
                "cpu_aot_inductor",
                "cpu_aot_inductor_amp_freezing",
                "cpu_aot_inductor_freezing",
                "cpu_inductor",
                "cpu_inductor_amp_freezing",
                "cpu_inductor_freezing",
                "dynamic_aot_eager",
                "dynamic_cpu_aot_inductor",
                "dynamic_cpu_aot_inductor_amp_freezing",
                "dynamic_cpu_aot_inductor_freezing",
                "dynamic_cpu_inductor",
                "dynamic_inductor",
                "dynamo_eager",
                "inductor",
            ],
            ["huggingface", "timm", "torchbench"],
        )
    }

    root_path = "benchmarks/dynamo/ci_expected_accuracy/"
    assert os.path.exists(root_path), f"cd <pytorch root> and ensure {root_path} exists"

    results = query_job_sha(repo, args.sha)
    urls = get_artifacts_urls(results, suites)
    dataframes = download_artifacts_and_extract_csvs(urls)
    write_filtered_csvs(root_path, dataframes)
    print("Success. Now, confirm the changes to .csvs and `git add` them if satisfied.")
