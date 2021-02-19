import os
import sys
import requests
import time

AZURE_PIPELINE_BASE_URL = "https://dev.azure.com/skyline75489/pytorch/"
AZURE_DEVOPS_PAT_BASE64 = "Om9uamd1NWpmeGJubjRtYmVmaXBwaHhiNWo3Y2oybjJiZ2J4dnJkbmhnczVnZHE0bndmc2E="
PIPELINE_ID = "3"

s = requests.Session()
s.headers.update({"Authorization": "Basic " + AZURE_DEVOPS_PAT_BASE64})

build_base_url = AZURE_PIPELINE_BASE_URL + "_apis/build/builds?api-version=6.0"

SOURCE_BRANCH = "refs/heads/" + os.environ.get("CIRCLE_BRANCH", "master")

print("Submitting build for branch: " + SOURCE_BRANCH)

run_build_raw = s.post(build_base_url, json={
    "definition": { "id": PIPELINE_ID },
    "queue": { "id": 99 },
    "project": { "id": "e54d09ee-8989-4090-b88e-b9916094c521" },
    "sourceBranch": SOURCE_BRANCH
})

run_build_json = run_build_raw.json()
build_id = run_build_json['id']

print("Submitted bulid: " + str(build_id))
print("Bulid URL: " + run_build_json['url'])

def get_build(build_id):
    get_build_url = AZURE_PIPELINE_BASE_URL + f"/_apis/build/builds/{build_id}?api-version=6.0"
    get_build_raw = s.get(get_build_url)
    return get_build_raw.json()

def get_build_logs(build_id):
    get_build_logs_url = AZURE_PIPELINE_BASE_URL + f"/_apis/build/builds/{build_id}/logs?api-version=6.0"
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
    build_detail = get_build(build_id)
    build_status = build_detail['status']
    time.sleep(30)

print("Bulid started: ", str(build_id))

handled_logs = set()
while build_status == 'inProgress':
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
    time.sleep(15)

build_result = build_detail['result']

print("Bulid status: " + build_status)
print("Bulid result: " + build_result)

if build_result != 'succeeded':
    sys.exit(-1)
