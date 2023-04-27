import os
import requests

PYTORCH_REPO = os.environ.get("GITHUB_REPOSITORY", "pytorch/pytorch")
PYTORCH_GITHUB_API = f"https://api.github.com/repos/{PYTORCH_REPO}"
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
REQUEST_HEADERS = {
    "Accept": "application/vnd.github.v3+json",
    "Authorization": "token " + GITHUB_TOKEN,
}

# the same works from the browser
# https://api.github.com/repos/pytorch/pytorch/actions/runs/3761062705/jobs?per_page=100
workflow_run_id = 3761062705
response = requests.get(
    f"{PYTORCH_GITHUB_API}/actions/runs/{workflow_run_id}/jobs?per_page=100",
    headers=REQUEST_HEADERS,
)

print(response)
