from functools import lru_cache
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional
import argparse
import zipfile

import requests

@lru_cache
def get_merge_base() -> str:
    merge_base = subprocess.check_output(
        ["git", "merge-base", "HEAD", "origin/main"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
    return merge_base

def ok_changed_file(file: str) -> bool:
    if file.startswith("torch/") and file.endswith(".py") and not file.startswith("torch/csrc/"):
        return True
    if file.startswith("test/") and file.endswith(".py"):
        return True
    return False

def check_changed_files():
    merge_base = get_merge_base()
    changed_files = subprocess.check_output(
        ["git", "diff", "--name-only", merge_base, "HEAD"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip().split()

    for file in changed_files:
        if not ok_changed_file(file):
            print(f"File {file} is not allowed to be changed.")
            return False
        else:
            print(f"File {file} is allowed to be changed.")
    return True

def query_github_api(url: str) -> dict:
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
    }
    response = requests.get(url, headers=headers)
    return response.json()

def find_old_whl(workflow_id: str, build_environment: str) -> bool:
    if build_environment is None:
        print("BUILD_ENVIRONMENT is not set.")
        return False
    merge_base = get_merge_base()

    workflow_runs = query_github_api(
        f"https://api.github.com/repos/pytorch/pytorch/actions/workflows/{workflow_id}/runs?head_sha={merge_base}&branch=main&status=completed&per_page=100"
    )
    if workflow_runs.get("total_count", 0) == 0:
        print("No workflow runs found.")
        return False
    for run in workflow_runs.get("workflow_runs", []):
        # Look in s3 for the old whl
        run_id = run["id"]
        try:
            url = f"https://gha-artifacts.s3.amazonaws.com/pytorch/pytorch/{run_id}/{build_environment}/artifacts.zip"
            response = requests.get(
                url,
            )
            if response.status_code == 200:
                os.makedirs("/tmp", exist_ok=True)
                with open("/tmp/artifacts.zip", "wb") as f:
                    f.write(response.content)
                    print(f"Found old whl file from s3: {url}")
                    return True
        except requests.RequestException as e:
            print(f"Error checking for old whl: {e}")
            continue
    return False


def unzip_artifact_and_replace_files():
    # Unzip the artifact and replace files
    with zipfile.ZipFile("/tmp/artifacts.zip", "r") as zip_ref:
        zip_ref.extractall("/tmp/artifacts")

    # Rename wheel into zip
    wheel_path = Path("/tmp/artifacts/dist").glob("*.whl")
    print(wheel_path)
    for path in wheel_path:
        new_path = path.with_suffix(".zip")
        os.rename(path, new_path)
        print(f"Renamed {path} to {new_path}")
        # Unzip the wheel
        with zipfile.ZipFile(new_path, "r") as zip_ref:
            print(f"Extracting {new_path} to /tmp/artifacts/dist/{new_path.stem}")
            zip_ref.extractall(f"/tmp/artifacts/dist/{new_path.stem}")

        # Copy python files into the artifact
        subprocess.check_output(
            ["rsync", "-avz", "torch", f"/tmp/artifacts/dist/{new_path.stem}/torch"],
        )

        # Zip the wheel back
        with zipfile.ZipFile(new_path, "w") as zip_ref:
            for root, _, files in os.walk(f"/tmp/artifacts/dist/{new_path.stem}"):
                for file in files:
                    file_path = os.path.join(root, file)
                    zip_ref.write(file_path, os.path.relpath(file_path, f"/tmp/artifacts/dist/{new_path.stem}"))

        # Reame back to whl
        os.rename(new_path, path)

        # Remove the extracted folder
        subprocess.check_output(
            ["rm", "-rf", f"/tmp/artifacts/dist/{new_path.stem}"],
        )

    # Rezip the artifact
    with zipfile.ZipFile("/tmp/artifacts.zip", "w") as zip_ref:
        for root, _, files in os.walk("/tmp/artifacts"):
            for file in files:
                file_path = os.path.join(root, file)
                zip_ref.write(file_path, os.path.relpath(file_path, "/tmp/artifacts"))
    # move artifact to the current directory
    os.rename("/tmp/artifacts.zip", "artifacts.zip")

    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Check for old whl files.")
    parser.add_argument("--workflow-id", type=str, required=True, help="Workflow ID")
    parser.add_argument("--build-environment", type=str, required=True, help="Build environment")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    can_use_old_whl = check_changed_files()
    if not find_old_whl(args.workflow_id, args.build_environment):
        exit(1)
    unzip_artifact_and_replace_files()
