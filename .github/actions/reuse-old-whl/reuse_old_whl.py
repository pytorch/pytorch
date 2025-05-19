import argparse
import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any, cast, Optional

import requests


FORCE_REBUILD_LABEL = "ci-force-rebuild"


@lru_cache
def get_merge_base() -> str:
    merge_base = subprocess.check_output(
        ["git", "merge-base", "HEAD", "origin/main"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
    # Remove this when we turn this off for the main branch
    if merge_base == get_head_sha():
        print("Merge base is the same as HEAD, using HEAD^")
        merge_base = subprocess.check_output(
            ["git", "rev-parse", "HEAD^"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    print(f"Merge base: {merge_base}")
    return merge_base


@lru_cache
def get_head_sha() -> str:
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
    return sha


def is_main_branch() -> bool:
    return False
    # Testing on main branch for now
    # print(
    #     f"Checking if we are on main branch: merge base {get_merge_base()}, head {get_head_sha()}"
    # )
    # return get_merge_base() == get_head_sha()


def query_github_api(url: str) -> Any:
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
    }
    response = requests.get(url, headers=headers)
    return response.json()


@lru_cache
def check_labels_for_pr() -> bool:
    # Check if the current commit is part of a PR and if it has the
    # FORCE_REBUILD_LABEL
    head_sha = get_head_sha()
    url = f"https://api.github.com/repos/pytorch/pytorch/commits/{head_sha}/pulls"
    response = query_github_api(url)

    print(
        f"Found {len(response)} PRs for commit {head_sha}: {[pr['number'] for pr in response]}"
    )
    for pr in response:
        labels = pr.get("labels", [])
        for label in labels:
            if label["name"] == FORCE_REBUILD_LABEL:
                print(f"Found label {FORCE_REBUILD_LABEL} in PR {pr['number']}.")
                return True
    return False


def check_issue_open() -> bool:
    # Check if issue #153759 is open.  This is the config issue for quickly
    # forcing everyone to build
    url = "https://api.github.com/repos/pytorch/pytorch/issues/153759"
    response = query_github_api(url)
    if response.get("state") == "open":
        print("Issue #153759 is open.")
        return True
    else:
        print("Issue #153759 is not open.")
        return False


def get_workflow_id(run_id: str) -> Optional[str]:
    # Get the workflow ID that corresponds to the file for the run ID
    url = f"https://api.github.com/repos/pytorch/pytorch/actions/runs/{run_id}"
    response = query_github_api(url)
    if "workflow_id" in response:
        print(f"Found workflow ID for run ID {run_id}: {response['workflow_id']}")
        return cast(str, response["workflow_id"])
    else:
        print("No workflow ID found.")
        return None


def ok_changed_file(file: str) -> bool:
    # Return true if the file is in the list of allowed files to be changed to
    # reuse the old whl
    if (
        file.startswith("torch/")
        and file.endswith(".py")
        and not file.startswith("torch/csrc/")
    ):
        return True
    if file.startswith("test/") and file.endswith(".py"):
        return True
    return False


def check_changed_files(sha: str) -> bool:
    # Return true if all the changed files are in the list of allowed files to
    # be changed to reuse the old whl
    changed_files = (
        subprocess.check_output(
            ["git", "diff", "--name-only", sha, "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        .strip()
        .split()
    )
    print(f"Checking changed files between {sha} and HEAD:")
    for file in changed_files:
        if not ok_changed_file(file):
            print(f"  File {file} is not allowed to be changed.")
            return False
        else:
            print(f"  File {file} is allowed to be changed.")
    return True


def find_old_whl(workflow_id: str, build_environment: str, sha: str) -> bool:
    # Find the old whl on s3 and download it to artifacts.zip
    if build_environment is None:
        print("BUILD_ENVIRONMENT is not set.")
        return False
    print(f"SHA: {sha}, workflow_id: {workflow_id}")

    workflow_runs = query_github_api(
        f"https://api.github.com/repos/pytorch/pytorch/actions/workflows/{workflow_id}/runs?head_sha={sha}&branch=main&per_page=100"
    )
    if workflow_runs.get("total_count", 0) == 0:
        print("No workflow runs found.")
        return False
    for run in workflow_runs.get("workflow_runs", []):
        # Look in s3 for the old whl
        run_id = run["id"]
        try:
            url = f"https://gha-artifacts.s3.amazonaws.com/pytorch/pytorch/{run_id}/{build_environment}/artifacts.zip"
            print(f"Checking for old whl at {url}")
            response = requests.get(
                url,
            )
            if response.status_code == 200:
                with open("artifacts.zip", "wb") as f:
                    f.write(response.content)
                    print(f"Found old whl file from s3: {url}")
                    return True
        except requests.RequestException as e:
            print(f"Error checking for old whl: {e}")
            continue
    return False


def unzip_artifact_and_replace_files() -> None:
    # Unzip the artifact and replace files
    subprocess.check_output(
        ["unzip", "-o", "artifacts.zip", "-d", "artifacts"],
    )
    os.remove("artifacts.zip")

    # Rename wheel into zip
    wheel_path = Path("artifacts/dist").glob("*.whl")
    for path in wheel_path:
        new_path = path.with_suffix(".zip")
        os.rename(path, new_path)
        print(f"Renamed {path} to {new_path}")
        print(new_path.stem)
        # Unzip the wheel
        subprocess.check_output(
            ["unzip", "-o", new_path, "-d", f"artifacts/dist/{new_path.stem}"],
        )
        # Copy python files into the artifact
        subprocess.check_output(
            ["rsync", "-avz", "torch", f"artifacts/dist/{new_path.stem}"],
        )

        # Zip the wheel back
        subprocess.check_output(
            ["zip", "-r", f"{new_path.stem}.zip", "."],
            cwd=f"artifacts/dist/{new_path.stem}",
        )
        subprocess.check_output(
            [
                "mv",
                f"artifacts/dist/{new_path.stem}/{new_path.stem}.zip",
                f"artifacts/dist/{new_path.stem}.whl",
            ],
        )

        # Remove the extracted folder
        subprocess.check_output(
            ["rm", "-rf", f"artifacts/dist/{new_path.stem}"],
        )

    # Rezip the artifact
    subprocess.check_output(["zip", "-r", "artifacts.zip", "."], cwd="artifacts")
    subprocess.check_output(
        ["mv", "artifacts/artifacts.zip", "."],
    )
    return None


def set_output() -> None:
    # Disable for now so we can monitor first
    # pass
    if os.getenv("GITHUB_OUTPUT"):
        with open(str(os.getenv("GITHUB_OUTPUT")), "a") as env:
            print("reuse=true", file=env)
    else:
        print("::set-output name=reuse::true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check for old whl files.")
    parser.add_argument("--run-id", type=str, required=True, help="Workflow ID")
    parser.add_argument(
        "--build-environment", type=str, required=True, help="Build environment"
    )
    parser.add_argument(
        "--github-ref",
        type=str,
    )
    return parser.parse_args()


def can_reuse_whl(args: argparse.Namespace) -> bool:
    # if is_main_branch() or (
    #     args.github_ref
    #     and any(
    #         args.github_ref.startswith(x)
    #         for x in ["refs/heads/release", "refs/tags/v", "refs/heads/main"]
    #     )
    # ):
    #     print("On main branch or release branch, rebuild whl")
    #     return False

    if check_labels_for_pr():
        print(f"Found {FORCE_REBUILD_LABEL} label on PR, rebuild whl")
        return False

    if check_issue_open():
        print("Issue #153759 is open, rebuild whl")
        return False

    if not check_changed_files(get_merge_base()):
        print("Cannot use old whl due to the changed files, rebuild whl")
        return False

    workflow_id = get_workflow_id(args.run_id)
    if workflow_id is None:
        print("No workflow ID found, rebuild whl")
        return False

    if not find_old_whl(workflow_id, args.build_environment, get_merge_base()):
        print("No old whl found, rebuild whl")
        # TODO: go backwards from merge base to find more runs
        return False

    return True


if __name__ == "__main__":
    args = parse_args()

    if can_reuse_whl(args):
        print("Reusing old whl")
        unzip_artifact_and_replace_files()
        set_output()
