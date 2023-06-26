import json
import os
import subprocess
from argparse import ArgumentParser
from typing import Any, Dict

import requests

UPDATEBOT_TOKEN = os.environ["UPDATEBOT_TOKEN"]
PYTORCHBOT_TOKEN = os.environ["PYTORCHBOT_TOKEN"]
OWNER, REPO = "pytorch", "pytorch"


def git_api(
    url: str, params: Dict[str, str], type: str = "get", token: str = UPDATEBOT_TOKEN
) -> Any:
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {token}",
    }
    if type == "post":
        return requests.post(
            f"https://api.github.com{url}",
            data=json.dumps(params),
            headers=headers,
        ).json()
    elif type == "patch":
        return requests.patch(
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
    parser.add_argument("--pin-folder", type=str)
    return parser.parse_args()


def make_pr(repo_name: str, branch_name: str) -> Any:
    params = {
        "title": f"[{repo_name} hash update] update the pinned {repo_name} hash",
        "head": branch_name,
        "base": "main",
        "body": "This PR is auto-generated nightly by [this action](https://github.com/pytorch/pytorch/blob/main/"
        + f".github/workflows/_update-commit-hash.yml).\nUpdate the pinned {repo_name} hash.",
    }
    response = git_api(f"/repos/{OWNER}/{REPO}/pulls", params, type="post")
    print(f"made pr {response['html_url']}")
    return response["number"]


def approve_pr(pr_number: str) -> None:
    params = {"event": "APPROVE"}
    # use pytorchbot to approve the pr
    git_api(
        f"/repos/{OWNER}/{REPO}/pulls/{pr_number}/reviews",
        params,
        type="post",
        token=PYTORCHBOT_TOKEN,
    )


def make_comment(pr_number: str, msg: str) -> None:
    params = {"body": msg}
    # comment with pytorchbot because pytorchmergebot gets ignored
    git_api(
        f"/repos/{OWNER}/{REPO}/issues/{pr_number}/comments",
        params,
        type="post",
        token=PYTORCHBOT_TOKEN,
    )


def close_pr(pr_number: str) -> None:
    params = {"state": "closed"}
    git_api(
        f"/repos/{OWNER}/{REPO}/pulls/{pr_number}",
        params,
        type="patch",
    )


def is_newer_hash(new_hash: str, old_hash: str, repo_name: str) -> bool:
    def _get_date(hash: str) -> int:
        # this git command prints the unix timestamp of the hash
        return int(
            subprocess.run(
                f"git show --no-patch --no-notes --pretty=%ct {hash}".split(),
                capture_output=True,
                cwd=f"{repo_name}",
            )
            .stdout.decode("utf-8")
            .strip()
        )

    return _get_date(new_hash) > _get_date(old_hash)


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
        link = response["items"][0]["html_url"]
        response = git_api(f"/repos/{OWNER}/{REPO}/pulls/{pr_num}", {})
        branch_name = response["head"]["ref"]
        print(
            f"pr does exist, number is {pr_num}, branch name is {branch_name}, link is {link}"
        )

    hash = (
        subprocess.run(
            f"git rev-parse {args.branch}".split(),
            capture_output=True,
            cwd=f"{args.repo_name}",
        )
        .stdout.decode("utf-8")
        .strip()
    )
    with open(f"{args.pin_folder}/{args.repo_name}.txt", "r+") as f:
        old_hash = f.read().strip()
        subprocess.run(f"git checkout {old_hash}".split(), cwd=args.repo_name)
        f.seek(0)
        f.truncate()
        f.write(f"{hash}\n")
    if is_newer_hash(hash, old_hash, args.repo_name):
        # if there was an update, push to branch
        subprocess.run(f"git checkout -b {branch_name}".split())
        subprocess.run(f"git add {args.pin_folder}/{args.repo_name}.txt".split())
        subprocess.run(
            "git commit -m".split() + [f"update {args.repo_name} commit hash"]
        )
        subprocess.run(f"git push --set-upstream origin {branch_name} -f".split())
        print(f"changes pushed to branch {branch_name}")
        if pr_num is None:
            # no existing pr, so make a new one and approve it
            pr_num = make_pr(args.repo_name, branch_name)
            approve_pr(pr_num)
        make_comment(pr_num, "@pytorchbot merge")
    else:
        print(
            f"tried to update from old hash: {old_hash} to new hash: {hash} but the old hash seems to be newer, not creating pr"
        )
        if pr_num is not None:
            make_comment(pr_num, "closing pr as the current hash seems up to date")
            close_pr(pr_num)
            print(f"closing PR {pr_num}")


if __name__ == "__main__":
    main()
