"""
Update commited CSV files used as reference points by dynamo/inductor CI.

Currently only cares about graph breaks, so only saves those columns.

Hardcodes a list of job names and artifacts per job, but builds the lookup
by querying github sha and finding associated github actions workflow ID and CI jobs,
downloading artifact zips, extracting CSVs and filtering them.

Usage:

1) get a rocksdp API key and save it in ~/rocks_api_key
2) python benchmarks/dynamo/ci_expected_accuracy.py <sha of pytorch commit that has completed inductor benchmark jobs>

Known limitations:
- doesn't handle 'retry' jobs in CI, if the same hash has more than one set of artifacts, gets the first one
- needs rocks API key (plan to use a public endpoint and remove this limitation)
"""

import argparse
import os
import urllib
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd

from rockset import RocksetClient


def query_job_sha(repo, sha, api_key):
    rs = RocksetClient(api_key=api_key, host="https://api.usw2a1.rockset.com")

    params = list()
    params.append({"name": "repo", "type": "string", "value": repo})
    params.append({"name": "sha", "type": "string", "value": sha})

    response = rs.QueryLambdas.execute_query_lambda(
        query_lambda="commit_jobs_query",
        version="cc524c5036e78794",
        workspace="commons",
        parameters=params,
    )
    return response.results


def parse_job_name(job_str):
    return (part.strip() for part in job_str.split("/"))


def parse_test_str(test_str):
    return (part.strip() for part in test_str[6:].strip(")k").split(","))


S3_BASE_URL = "https://gha-artifacts.s3.amazonaws.com"


def get_artifacts_urls(results, suites):
    urls = {}
    for r in results:
        if "inductor" == r["workflowName"] and f"test" in r["jobName"]:
            config_str, test_str = parse_job_name(r["jobName"])
            suite, shard_id, num_shards, machine = parse_test_str(test_str)
            workflowId = r["workflowId"]
            id = r["id"]
            runattempt = 1  # ? guessing here

            if suite in suites:
                artifact_filename = f"test-reports-test-{suite}-{shard_id}-{num_shards}-{machine}_{id}.zip"
                s3_url = f"{S3_BASE_URL}/{repo}/{workflowId}/{runattempt}/artifact/{artifact_filename}"
                urls[(suite, int(shard_id))] = s3_url
                print(f"{suite} {shard_id}, {num_shards}: {s3_url}")
    return urls


def normalize_suite_filename(suite_name):
    assert suite_name.find("inductor_") == 0
    subsuite = suite_name.split("_")[1]
    if "timm" in subsuite:
        subsuite = f"{subsuite}_models"
    return subsuite


def download_artifacts_and_extract_csvs(urls):
    dataframes = {}
    try:
        for (suite, shard), url in urls.items():
            resp = urlopen(url)
            subsuite = normalize_suite_filename(suite)
            artifact = ZipFile(BytesIO(resp.read()))
            for phase in ("training", "inference"):
                name = f"test/test-reports/{phase}_{subsuite}.csv"
                df = pd.read_csv(artifact.open(name))
                dataframes[(suite, phase, shard)] = df
    except urllib.error.HTTPError:
        print(f"Unable to download {url}, perhaps the CI job isn't finished?")
    return dataframes


def write_filtered_csvs(root_path, dataframes):
    for (suite, phase, shard), df in dataframes.items():
        suite_fn = normalize_suite_filename(suite)
        if "timm" in suite:
            out_fn = os.path.join(root_path, f"{phase}_{suite_fn}{shard - 1}.csv")
        else:
            out_fn = os.path.join(root_path, f"{phase}_{suite_fn}.csv")
        df.to_csv(out_fn, index=False, columns=["name", "graph_breaks"])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("sha")
    args = parser.parse_args()

    repo = "pytorch/pytorch"
    suites = {
        "inductor_huggingface",
        "inductor_timm",
        "inductor_torchbench",
        # "inductor_huggingface_dynamic",
        # "inductor_timm_dynamic",
        # "inductor_torchbench_dynamic",
    }

    # TODO check path and warn to run from pytorch root
    root_path = "benchmarks/dynamo/ci_expected_accuracy/"

    # TODO open public rocksdb endpoint, no apikey
    with open(os.path.join(os.path.expanduser("~"), "rocks_api_key"), "r") as f:
        api_key = f.read()

    results = query_job_sha(repo, args.sha, api_key)
    urls = get_artifacts_urls(results, suites)
    dataframes = download_artifacts_and_extract_csvs(urls)
    write_filtered_csvs(root_path, dataframes)
