import requests
import torch._logging.scribe as scribe

def log_failure(reason):
    scribe.open_source_signpost(
        subsystem="pr_time_benchmarks",
        name="compare_test_failure",
        parameters=json.dumps(reason),
    )


def get_workflow_id(head_sha) -> None:
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": "Bearer {OS.environ['GITHUB_TOKEN']}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    response = requests.get(
        f"https://api.github.com/repos/pytorch/pytorch/actions/runs?head_sha={head_sha}",
        headers=headers,
    )

    if response.status_code != 200:
        print(f"Error: {response.text}")
        log_failure(f"Failed to get workflow id : response.status_code is{response.status_code }")
        return

    data = response.json()

    for entry in data.get("workflow_runs", []):
        if entry.get("name") == "pull":
            print(entry.get("id"))
            return

    log_failure(f"did not find any pull workflow with push event")

if __name__ == "__main__":
    get_workflow_id("a1a57a424dc992f4dc2d44bdc1e4e7e500881a9c")
