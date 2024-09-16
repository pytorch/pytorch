
import requests
import json
import os

def get_workflow_id(head_sha) -> None:
    headers = {
            "Accept": "application/vnd.github.v3+json",
            f"Authorization" :"Bearer {OS.environ['GITHUB_TOKEN']}",
            "X-GitHub-Api-Version": "2022-11-28"
            }

    response = requests.get(
            f"https://api.github.com/repos/pytorch/pytorch/actions/runs?head_sha={head_sha}",
        headers=headers,
    )

    if response.status_code  != 200:
        print(f"Error: {response.text}")
        return
    data = response.json()

    for entry in data.get("workflow_runs", []):
        if entry.get("name") == 'pull':
            print(entry.get("id"))
            return

    print(response)


if __name__ == "__main__":

    get_workflow_id("a1a57a424dc992f4dc2d44bdc1e4e7e500881a9c")
