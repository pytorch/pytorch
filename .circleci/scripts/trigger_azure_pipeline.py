import os
import re
import sys
import requests
import time

AZURE_PIPELINE_BASE_URL = "https://aiinfra.visualstudio.com/PyTorch/"
AZURE_DEVOPS_PAT_BASE64 = "LW46YnZoNWhwM2xnZ3MzdWVqM2VteDR0ZnV4cHJ6djVwc2xuNWg0emZqaDdsa2Y0ZjdzN2lzcQ=="
PIPELINE_ID = "911"
PROJECT_ID = "0628bce4-2d33-499e-bac5-530e12db160f"

s = requests.Session()
s.headers.update({"Authorization": "Basic " + AZURE_DEVOPS_PAT_BASE64})

build_base_url = AZURE_PIPELINE_BASE_URL + "_apis/build/builds?api-version=6.0"

TARGET_BRANCH = os.environ.get("CIRCLE_BRANCH", "master")

# Convert the branch name for Azure DevOps
if TARGET_BRANCH.startswith('pull/'):
    match = re.search(r'pull/(\d+)', TARGET_BRANCH)
    pr_num = match.group(1)
    SOURCE_BRANCH = 'refs/pull/' + str(pr_num) + '/head'
else:
    SOURCE_BRANCH = "refs/heads/" + TARGET_BRANCH

print("Submitting build for branch: " + SOURCE_BRANCH)

run_build_raw = s.post(build_base_url, json={
    "definition": {"id": PIPELINE_ID},
    "project": {"id": PROJECT_ID},
    "sourceBranch": SOURCE_BRANCH
})

run_build_json = run_build_raw.json()
build_id = run_build_json['id']

print("Submitted Build: " + str(build_id))
print("Build URL: " + run_build_json['url'])


def get_build(_id):
    get_build_url = AZURE_PIPELINE_BASE_URL + "/_apis/build/builds/{}?api-version=6.0".format(_id)
    get_build_raw = s.get(get_build_url)
    return get_build_raw.json()


def get_build_logs(_id):
    get_build_logs_url = AZURE_PIPELINE_BASE_URL + "/_apis/build/builds/{}/logs?api-version=6.0".format(_id)
    get_build_logs_raw = s.get(get_build_logs_url)
    return get_build_logs_raw.json()


def get_log_content(url):
    resp = s.get(url)
    return resp.text


build_detail = get_build(build_id)
build_status = build_detail['status']

while build_status == 'notStarted':
    print('Waiting for run to start: ' + str(build_id))
    sys.stdout.flush()
    try:
        build_detail = get_build(build_id)
        build_status = build_detail['status']
    except requests.exceptions.ConnectionError:
        pass
    time.sleep(60)

print("Build started: ", str(build_id))

handled_logs = set()
while build_status == 'inProgress':
    try:
        logs = get_build_logs(build_id)
        for log in logs['value']:
            log_id = log['id']
            if log_id in handled_logs:
                continue
            handled_logs.add(log_id)
            print('Fetching log: \n' + log['url'])
            log_content = get_log_content(log['url'])
            print(log_content)
            sys.stdout.flush()
        build_detail = get_build(build_id)
        build_status = build_detail['status']
    except requests.exceptions.ConnectionError:
        pass
    time.sleep(30)

build_result = build_detail['result']

print("Build status: " + build_status)
print("Build result: " + build_result)

if build_result != 'succeeded':
    sys.exit(-1)
