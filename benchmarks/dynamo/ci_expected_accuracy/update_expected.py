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
import os
import urllib
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd
import requests

# Note: the public query url targets this rockset lambda:
# https://console.rockset.com/lambdas/details/commons.artifacts
ARTIFACTS_QUERY_URL = "https://api.usw2a1.rockset.com/v1/public/shared_lambdas/4ca0033e-0117-41f5-b043-59cde19eff35"


def query_job_sha(repo, sha):
    params = {
        "parameters": [
            {"name": "sha", "type": "string", "value": sha},
            {"name": "repo", "type": "string", "value": repo},
        ]
    }

    r = requests.post(url=ARTIFACTS_QUERY_URL, json=params)
    data = r.json()
    return data["results"]


def parse_job_name(job_str):
    return (part.strip() for part in job_str.split("/"))


def parse_test_str(test_str):
    return (part.strip() for part in test_str[6:].strip(")").split(","))


S3_BASE_URL = "https://gha-artifacts.s3.amazonaws.com"


def get_artifacts_urls(results, suites):
    urls = {}
    for r in results:
        if "inductor" == r["workflowName"] and "test" in r["jobName"]:
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
    subsuite = strs[2] if strs[0] == "aot" else strs[1]
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("sha")
    args = parser.parse_args()

    repo = "pytorch/pytorch"
    suites = {
        "aot_inductor_huggingface",
        "inductor_huggingface",
        "inductor_huggingface_dynamic",
        "aot_inductor_timm",
        "inductor_timm",
        "inductor_timm_dynamic",
        "aot_inductor_torchbench",
        "inductor_torchbench",
        "inductor_torchbench_dynamic",
    }

    root_path = "benchmarks/dynamo/ci_expected_accuracy/"
    assert os.path.exists(root_path), f"cd <pytorch root> and ensure {root_path} exists"

    results = query_job_sha(repo, args.sha)
    urls = get_artifacts_urls(results, suites)
    dataframes = download_artifacts_and_extract_csvs(urls)
    write_filtered_csvs(root_path, dataframes)
    print("Success. Now, confirm the changes to .csvs and `git add` them if satisfied.")
