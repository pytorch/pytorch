import json
import os
import subprocess
import requests
from typing import Any, Dict
from argparse import ArgumentParser

MERGEBOT_TOKEN = os.environ["MERGEBOT_TOKEN"]
PYTORCHBOT_TOKEN = os.environ["PYTORCHBOT_TOKEN"]
OWNER, REPO = "pytorch", "pytorch"


def git_api(
    url: str, params: Dict[str, str], post: bool = False, token: str = MERGEBOT_TOKEN
) -> Any:
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {token}",
    }
    if post:
        return requests.post(
            f"https://api.github.com{url}",
            data=json.dumps(params),
            headers=headers,
        ).json()
    else:
        return requests.get(
            f"https://api.github.com{url}",
            params=params,
            headers=headers,
        ).json()


def parse_args() -> Any:
    parser = ArgumentParser("Rebase PR into branch")
    parser.add_argument("--repo-name", type=str)
    parser.add_argument("--branch", type=str)
    return parser.parse_args()


def make_pr(repo_name: str, branch_name: str) -> Any:
    params = {
        "title": f"[{repo_name} hash update] update the pinned {repo_name} hash",
        "head": branch_name,
        "base": "master",
        "body": "This PR is auto-generated nightly by [this action](https://github.com/pytorch/pytorch/blob/master/"
        + f".github/workflows/_update-commit-hash.yml).\nUpdate the pinned {repo_name} hash.",
    }
    response = git_api(f"/repos/{OWNER}/{REPO}/pulls", params, post=True)
    print(f"made pr {response['html_url']}")
    return response["number"]


def approve_pr(pr_number: str) -> None:
    params = {"event": "APPROVE"}
    # use pytorchbot to approve the pr
    git_api(
        f"/repos/{OWNER}/{REPO}/pulls/{pr_number}/reviews",
        params,
        post=True,
        token=PYTORCHBOT_TOKEN,
    )


def make_comment(pr_number: str) -> None:
    params = {"body": "@pytorchbot merge -g"}
    # comment with pytorchbot because pytorchmergebot gets ignored
    git_api(
        f"/repos/{OWNER}/{REPO}/issues/{pr_number}/comments",
        params,
        post=True,
        token=PYTORCHBOT_TOKEN,
    )


def main() -> None:
    args = parse_args()

    branch_name = os.environ["NEW_BRANCH_NAME"]
    pr_num = None

    # query to see if a pr already exists
    params = {
        "q": f"is:pr is:open in:title author:pytorchmergebot repo:{OWNER}/{REPO} {args.repo_name} hash update"
    }
    response = git_api("/search/issues", params)
    if response["total_count"] != 0:
        # pr does exist
        pr_num = response["items"][0]["number"]
        response = git_api(f"/repos/{OWNER}/{REPO}/pulls/{pr_num}", {})
        branch_name = response["head"]["ref"]
        print(f"pr does exist, number is {pr_num}, branch name is {branch_name}")

    # update file
    hash = subprocess.run(
        f"git rev-parse {args.branch}".split(),
        capture_output=True,
        cwd=f"{args.repo_name}",
    ).stdout.decode("utf-8")
    with open(f".github/ci_commit_pins/{args.repo_name}.txt", "w") as f:
        f.write(hash)
    git_diff = subprocess.run(
        f"git diff --exit-code .github/ci_commit_pins/{args.repo_name}.txt".split()
    )
    if git_diff.returncode == 1:
        # if there was an update, push to branch
        subprocess.run(f"git checkout -b {branch_name}".split())
        subprocess.run(f"git add .github/ci_commit_pins/{args.repo_name}.txt".split())
        subprocess.run(
            "git commit -m".split() + [f"update {args.repo_name} commit hash"]
        )
        subprocess.run(f"git push --set-upstream origin {branch_name} -f".split())
        print(f"changes pushed to branch {branch_name}")
        if pr_num is None:
            # no existing pr, so make a new one and approve it
            pr_num = make_pr(args.repo_name, branch_name)
            approve_pr(pr_num)
    if pr_num is not None:
        # comment to merge if all checks are green
        make_comment(pr_num)


if __name__ == "__main__":
    main()
