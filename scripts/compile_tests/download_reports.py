import json
import os
import re
import subprocess

import requests

JOBS = {
    "linux-focal-py3.8-clang10 / test (dynamo, 1, 3, linux.2xlarge)": "dynamo38",
    "linux-focal-py3.8-clang10 / test (dynamo, 2, 3, linux.2xlarge)": "dynamo38",
    "linux-focal-py3.8-clang10 / test (dynamo, 3, 3, linux.2xlarge)": "dynamo38",
    "linux-focal-py3.11-clang10 / test (dynamo, 1, 3, linux.2xlarge)": "dynamo311",
    "linux-focal-py3.11-clang10 / test (dynamo, 2, 3, linux.2xlarge)": "dynamo311",
    "linux-focal-py3.11-clang10 / test (dynamo, 3, 3, linux.2xlarge)": "dynamo311",
    "linux-focal-py3.11-clang10 / test (default, 1, 3, linux.2xlarge)": "eager311",
    "linux-focal-py3.11-clang10 / test (default, 2, 3, linux.2xlarge)": "eager311",
    "linux-focal-py3.11-clang10 / test (default, 3, 3, linux.2xlarge)": "eager311",
}


def download_reports(commit_sha):
    log_dir = "tmp_test_reports_" + commit_sha
    subdirs = ["dynamo38", "dynamo311", "eager311"]
    subdir_paths = []
    for subdir in subdirs:
        subdir_path = f"{log_dir}/{subdir}"
        subdir_paths.append(subdir_path)

    if os.path.exists(log_dir):
        # Assume we already downloaded it
        print(f"{log_dir}/ output directory already exists, not downloading again")
        return subdir_paths
    output = subprocess.check_output(
        ["gh", "run", "list", "-c", commit_sha, "-w", "pull", "--json", "databaseId"]
    ).decode()
    workflow_run_id = str(json.loads(output)[0]["databaseId"])
    output = subprocess.check_output(["gh", "run", "view", workflow_run_id])
    workflow_jobs = parse_workflow_jobs(output)

    for key in JOBS:
        assert key in workflow_jobs, key

    # This page lists all artifacts
    listings = requests.get(
        f"https://hud.pytorch.org/api/artifacts/s3/{workflow_run_id}"
    ).json()

    os.mkdir(log_dir)
    for subdir in subdirs:
        subdir_path = f"{log_dir}/{subdir}"
        os.mkdir(subdir_path)

    def download_report(job_name):
        subdir = f"{log_dir}/{JOBS[job_name]}"
        job_id = workflow_jobs[job_name]
        for listing in listings:
            name = listing["name"]
            if not name.startswith("test-reports-"):
                continue
            if name.endswith(f"_{job_id}.zip"):
                url = listing["url"]
                subprocess.run(["wget", "-P", subdir, url], check=True)
                path_to_zip = f"{subdir}/{name}"
                dir_name = path_to_zip[:-4]
                subprocess.run(["unzip", path_to_zip, "-d", dir_name], check=True)
                return
        raise AssertionError("should not be hit")

    for job_name in JOBS.keys():
        download_report(job_name)

    return subdir_paths


def parse_workflow_jobs(output):
    result = {}
    lines = output.decode().split("\n")
    for line in lines:
        match = re.search(r"(\S+ / .*) in .* \(ID (\d+)\)", line)
        if match is None:
            continue
        result[match.group(1)] = match.group(2)
    return result
