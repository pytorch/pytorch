"""
Update committed CSV files used as reference points by dynamo/inductor CI.

Currently only cares about graph breaks, so only saves those columns.

Hardcodes a list of job names and artifacts per job, but builds the lookup
by querying github sha and finding associated github actions workflow ID and CI jobs,
downloading artifact zips, extracting CSVs and filtering them.

Usage:

python benchmarks/dynamo/ci_expected_accuracy.py <sha of pytorch commit that has completed inductor benchmark jobs>

Known limitations:
- doesn't handle 'retry' jobs in CI, if the same job has multiple run attempts, it may pick the wrong one
"""

import argparse
import json
import os
import subprocess
import sys
import urllib
from concurrent.futures import as_completed, ThreadPoolExecutor
from contextlib import closing
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
    "https://console-api.clickhouse.cloud/.api/query-endpoints/"  # @lint-ignore
    "c1cdfadc-6bb2-4a91-bbf9-3d19e1981cd4/run?format=JSON"
)
GITHUB_API_URL = "https://api.github.com"
TARGET_WORKFLOWS = ("inductor", "inductor-periodic")
SKIPPED_WORKFLOW_EVENTS = {"workflow_run", "repository_dispatch"}
SKIPPED_JOB_NAMES = {"ciflow_should_run", "generate-test-matrix"}
CSV_LINTER = str(
    Path(__file__).absolute().parents[3]
    / "tools/linter/adapters/no_merge_conflict_csv_linter.py"
)


def query_job_sha_from_clickhouse(repo, sha, key_id, key_secret):
    params = {
        "queryVariables": {"sha": sha, "repo": repo},
    }

    r = requests.post(
        url=ARTIFACTS_QUERY_URL,
        data=json.dumps(params),
        headers={"Content-Type": "application/json"},
        auth=(key_id, key_secret),
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["data"]


def github_request_headers():
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    github_token = os.environ.get("GITHUB_TOKEN")
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
    return headers


def github_pages(url, params):
    headers = github_request_headers()
    while url:
        with closing(
            requests.get(url, params=params, headers=headers, timeout=30)
        ) as r:
            try:
                r.raise_for_status()
            except requests.HTTPError as e:
                if (
                    r.status_code == 403
                    and r.headers.get("x-ratelimit-remaining") == "0"
                ):
                    raise RuntimeError(
                        "GitHub API rate limit exceeded while querying workflow jobs. "
                        "Set GITHUB_TOKEN to raise the limit."
                    ) from e
                raise
            page = r.json()
            next_url = r.links.get("next", {}).get("url")
        yield page
        url = next_url
        params = None


def query_job_sha_from_github(repo, sha):
    results = []
    runs_url = f"{GITHUB_API_URL}/repos/{repo}/actions/runs"
    for runs_page in github_pages(runs_url, {"head_sha": sha, "per_page": 100}):
        for workflow in runs_page.get("workflow_runs", []):
            workflow_name = workflow.get("name")
            if workflow_name not in TARGET_WORKFLOWS:
                continue
            if workflow.get("event") in SKIPPED_WORKFLOW_EVENTS:
                continue
            if workflow.get("head_sha") != sha:
                continue

            workflow_id = workflow["id"]
            jobs_url = f"{GITHUB_API_URL}/repos/{repo}/actions/runs/{workflow_id}/jobs"
            for jobs_page in github_pages(jobs_url, {"filter": "all", "per_page": 100}):
                for job in jobs_page.get("jobs", []):
                    job_name = job.get("name")
                    if job_name in SKIPPED_JOB_NAMES:
                        continue
                    if job.get("head_sha") != sha:
                        continue
                    results.append(
                        {
                            "workflowName": workflow_name,
                            "jobName": job_name,
                            "id": str(job["id"]),
                            "runAttempt": job.get(
                                "run_attempt", workflow.get("run_attempt", 1)
                            ),
                            "workflowId": str(workflow_id),
                            "time": job.get("created_at")
                            or workflow.get("created_at", ""),
                        }
                    )

    return sorted(results, key=lambda r: (r["workflowName"], r["jobName"]))


def query_job_sha(repo, sha):
    # Prefer the CI ClickHouse endpoint when credentials are available, but
    # local contributors can get the same public job metadata from GitHub.
    key_id = os.environ.get("CH_KEY_ID")
    key_secret = os.environ.get("CH_KEY_SECRET")
    if key_id and key_secret:
        try:
            return query_job_sha_from_clickhouse(repo, sha, key_id, key_secret)
        except requests.RequestException as e:
            print(
                "Warning: ClickHouse query failed; falling back to GitHub "
                f"Actions API. Error: {e}",
                file=sys.stderr,
            )
    elif key_id or key_secret:
        print(
            "Warning: only one of CH_KEY_ID and CH_KEY_SECRET is set; "
            "falling back to GitHub Actions API.",
            file=sys.stderr,
        )
    else:
        print(
            "ClickHouse credentials not found; querying GitHub Actions API.",
            file=sys.stderr,
        )

    return query_job_sha_from_github(repo, sha)


def parse_job_name(job_str):
    return (part.strip() for part in job_str.split("/"))


def parse_test_str(test_str):
    return (part.strip() for part in test_str[6:].strip(")").split(","))


S3_BASE_URL = "https://gha-artifacts.s3.amazonaws.com"


def get_artifacts_urls(repo, results, suites, is_rocm=False):
    urls = {}
    # Sort by time (oldest first) to prefer earlier completed workflow runs
    # over potentially still-running newer ones
    sorted_results = sorted(results, key=lambda x: x.get("time", ""))
    for r in sorted_results:
        if (
            r["workflowName"] in TARGET_WORKFLOWS
            and "test" in r["jobName"]
            and "build" not in r["jobName"]
            and "runner-determinator" not in r["jobName"]
            and "unit-test" not in r["jobName"]
        ):
            # Filter out CUDA-13 jobs so it won't override CUDA-12 results.
            # The result files should be shared between CUDA-12 and CUDA-13, but
            # CUDA-13 skips more tests at the moment.
            if "cuda13" in r["jobName"]:
                continue

            # Filter based on whether this is a ROCm or CUDA job
            job_is_rocm = "rocm" in r["jobName"].lower()
            if job_is_rocm != is_rocm:
                continue

            *_, test_str = parse_job_name(r["jobName"])
            suite, shard_id, num_shards, machine, *_ = parse_test_str(test_str)
            workflowId = r["workflowId"]
            id = r["id"]
            runAttempt = r["runAttempt"]

            if suite in suites:
                artifact_filename = f"test-reports-test-{suite}-{shard_id}-{num_shards}-{machine}_{id}.zip"
                s3_url = f"{S3_BASE_URL}/{repo}/{workflowId}/{runAttempt}/artifact/{artifact_filename}"
                # Collect all candidate URLs per (suite, shard), ordered oldest first
                key = (suite, int(shard_id))
                if key not in urls:
                    urls[key] = []
                if s3_url not in urls[key]:
                    urls[key].append(s3_url)
    return urls


def normalize_suite_filename(suite_name):
    strs = suite_name.split("_")
    subsuite = strs[-1]
    if "timm" in subsuite:
        subsuite = subsuite.replace("timm", "timm_models")

    return subsuite


def download_single_artifact(suite, shard, url_candidates):
    """Download a single artifact, trying each URL candidate until one succeeds.

    Returns a tuple of (suite, shard, result_dict) where result_dict maps
    (suite, phase) -> DataFrame, or None if download failed.
    """
    subsuite = normalize_suite_filename(suite)
    for url in url_candidates:
        try:
            resp = urlopen(url)
            artifact = ZipFile(BytesIO(resp.read()))
            result = {}
            for phase in ("training", "inference"):
                # Try both paths - CUDA uses test/test-reports/, ROCm uses test-reports/
                possible_names = [
                    f"test/test-reports/{phase}_{subsuite}.csv",
                    f"test-reports/{phase}_{subsuite}.csv",
                ]
                found = False
                for name in possible_names:
                    try:
                        df = pd.read_csv(artifact.open(name))
                        df["graph_breaks"] = df["graph_breaks"].fillna(0).astype(int)
                        result[(suite, phase)] = df
                        found = True
                        break
                    except KeyError:
                        continue
                if not found and phase == "inference":
                    # No warning for training, since it's expected to be missing for some tests
                    print(
                        f"Warning: Unable to find {phase}_{subsuite}.csv in artifacts file from {url}, continuing"
                    )
            return (suite, shard, result)
        except urllib.error.HTTPError:
            continue  # Try next candidate URL
    return (suite, shard, None)


def download_artifacts_and_extract_csvs(urls):
    dataframes = {}
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {
            executor.submit(download_single_artifact, suite, shard, url_candidates): (
                suite,
                shard,
                url_candidates,
            )
            for (suite, shard), url_candidates in urls.items()
        }
        for future in as_completed(futures):
            suite, shard, url_candidates = futures[future]
            suite_result, shard_result, result = future.result()
            if result is None:
                print(
                    f"Unable to download any artifact for {suite} shard {shard}, tried {len(url_candidates)} URLs"
                )
            else:
                for (s, phase), df in result.items():
                    prev_df = dataframes.get((s, phase), None)
                    dataframes[(s, phase)] = (
                        pd.concat([prev_df, df]) if prev_df is not None else df
                    )

    return dataframes


def write_filtered_csvs(root_path, dataframes):
    for (suite, phase), df in dataframes.items():
        out_fn = os.path.join(root_path, f"{suite}_{phase}.csv")
        # Read existing CSV and merge with new data to preserve entries
        # from shards that failed to download
        if os.path.exists(out_fn):
            existing_df = pd.read_csv(out_fn)
            # Use new data where available, keep old data for missing entries
            # Set 'name' as index for both, update existing with new, then reset
            existing_df = existing_df.set_index("name")
            df = df.set_index("name")
            existing_df.update(df)
            # Add any new entries from df that weren't in existing
            df = existing_df.combine_first(df).reset_index()
        df = df.sort_values(by="name")
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
    if not os.path.exists(root_path):
        raise AssertionError(f"cd <pytorch root> and ensure {root_path} exists")
    rocm_path = "benchmarks/dynamo/ci_expected_accuracy/rocm/"
    if not os.path.exists(rocm_path):
        raise AssertionError(f"cd <pytorch root> and ensure {rocm_path} exists")

    results = query_job_sha(repo, args.sha)

    # Get URLs for both CUDA and ROCm
    cuda_urls = get_artifacts_urls(repo, results, suites, is_rocm=False)
    rocm_urls = get_artifacts_urls(repo, results, suites, is_rocm=True)

    # Download CUDA and ROCm artifacts in parallel
    print("Downloading CUDA and ROCm artifacts in parallel...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        cuda_future = executor.submit(download_artifacts_and_extract_csvs, cuda_urls)
        rocm_future = executor.submit(download_artifacts_and_extract_csvs, rocm_urls)
        cuda_dataframes = cuda_future.result()
        rocm_dataframes = rocm_future.result()

    print("Writing CUDA CSVs...")
    write_filtered_csvs(root_path, cuda_dataframes)

    print("Writing ROCm CSVs...")
    write_filtered_csvs(rocm_path, rocm_dataframes)

    print("Success. Now, confirm the changes to .csvs and `git add` them if satisfied.")
