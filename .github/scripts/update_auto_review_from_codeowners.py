"""
Updates the auto_request_review.yml `files` section from the CODEOWNERS file.

Usage:
    pip install requests ruamel.yaml
    export GITHUB_TOKEN=your_token
    python .github/scripts/update_auto_review_from_codeowners.py
    # commit the changes
"""

import os
from typing import Any, Dict, List, Optional

import requests
from ruamel.yaml import YAML


user_permissions_cache: Dict[str, bool] = {}

token = os.environ.get("GITHUB_TOKEN")
if not token:
    raise ValueError("GITHUB_TOKEN not found in environment variables")

headers = {
    "Authorization": f"token {token}",
    "Accept": "application/vnd.github.v3+json",
}


def get_repo_teams(repo: str) -> Dict[str, Dict[str, Any]]:
    """
    Get all the teams in a repository.
    """
    response = requests.get(
        f"https://api.github.com/repos/{repo}/teams", headers=headers
    )
    if response.status_code == 200:
        dict = {}
        for team in response.json():
            dict[team["slug"]] = team
        return dict
    else:
        raise ValueError(f"Error fetching teams for repo {repo}: {response.json()}")


def user_can_review(repo: str, user: str) -> bool:
    """
    Check if a user has triage or greater permission in a repository.
    """
    cache_key = f"{repo}/{user}"
    if cache_key in user_permissions_cache:
        return user_permissions_cache[f"{repo}/{user}"]

    response = requests.get(
        f"https://api.github.com/repos/{repo}/collaborators/{user}/permission",
        headers=headers,
    )
    result = False
    if response.status_code == 200:
        # true if user has triage or greater permission
        required_perms = ["admin", "maintain", "write", "triage"]
        permissions = response.json()["user"]["permissions"]
        for perm in required_perms:
            if permissions.get(perm):
                print(f"{user} is a valid reviewer for {repo}.")
                user_permissions_cache[cache_key] = True
                return True
        print(f"WARN: `{user}` does not have triage or greater permission in {repo}")
    else:
        print(f"ERROR fetching permission for user {user}: {response.json()}")

    user_permissions_cache[cache_key] = False
    return False


def convert_codeowners_to_dict(codeowners_path: str) -> Dict[str, List[str]]:
    """
    Read CODEOWNERS file and convert it to the dictionary of files and their owners.
    """
    result = {}
    with open(codeowners_path, "r") as file:
        for line in file:
            if line.startswith("#") or not line.strip():
                continue

            parts = line.strip().split()
            path, owners = parts[0], parts[1:]
            if path.endswith("/"):
                path += "**"
            if path.startswith("/"):
                path = path[1:]
            else:
                path = f"**/{path}"
            owners = [owner.lstrip("@") for owner in owners]
            result[path] = owners

    return result


def filter_owners(files_with_owners: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Filter the owners to only include users that have triage or greater permission.
    Resolves teams to their members.
    """
    known_teams = get_repo_teams("pytorch/pytorch")

    def get_team(team: str) -> Optional[Dict[str, Any]]:
        team = team.removeprefix("pytorch/")  # type: ignore[attr-defined]
        return known_teams.get(team)

    print(f"Known teams: {list(known_teams.keys())}")

    filtered_files_with_owners = {}
    for file, owners in files_with_owners.items():
        # partition into teams and users
        teams = []
        users = []
        for owner in owners:
            team = get_team(owner)
            if team is not None:
                teams.append(team["slug"])
            else:
                users.append(owner)

        filtered_owners = [u for u in users if user_can_review("pytorch/pytorch", u)]
        filtered_owners += teams

        if filtered_owners:
            filtered_files_with_owners[file] = filtered_owners

    return filtered_files_with_owners


def update_auto_request_review_file(
    codeowners_path: str, auto_request_review_path: str
) -> None:
    """
    Update the auto_request_review.yml file with the new content from the CODEOWNERS file.
    """
    new_files_content = convert_codeowners_to_dict(codeowners_path)
    new_files_content = filter_owners(new_files_content)

    yaml = YAML()
    yaml.preserve_quotes = True  # type: ignore[assignment]
    yaml.indent(mapping=2, sequence=4, offset=2)

    with open(auto_request_review_path, "r") as file:
        current_yaml = yaml.load(file)

    current_yaml["files"] = new_files_content

    with open(auto_request_review_path, "w") as file:
        yaml.dump(current_yaml, file)


def main() -> None:
    from os.path import dirname, realpath

    script_path = realpath(__file__)
    root_dir = dirname(dirname(dirname(script_path)))

    codeowners_path = f"{root_dir}/CODEOWNERS"
    auto_request_review_path = f"{root_dir}/.github/auto_request_review.yml"
    update_auto_request_review_file(codeowners_path, auto_request_review_path)


if __name__ == "__main__":
    main()
