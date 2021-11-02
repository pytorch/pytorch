# Documentation: https://docs.microsoft.com/en-us/rest/api/azure/devops/build/?view=azure-devops-rest-6.0

import re
import json
import os
import sys
import requests
import time

AZURE_PIPELINE_BASE_URL = "https://aiinfra.visualstudio.com/PyTorch/"
AZURE_DEVOPS_PAT_BASE64 = os.environ.get("AZURE_DEVOPS_PAT_BASE64_SECRET", "")
PIPELINE_ID = "911"
PROJECT_ID = "0628bce4-2d33-499e-bac5-530e12db160f"
TARGET_BRANCH = os.environ.get("CIRCLE_BRANCH", "master")
TARGET_COMMIT = os.environ.get("CIRCLE_SHA1", "")

build_base_url = AZURE_PIPELINE_BASE_URL + "_apis/build/builds?api-version=6.0"

s = requests.Session()
s.headers.update({"Authorization": "Basic " + AZURE_DEVOPS_PAT_BASE64})

def submit_build(pipeline_id, project_id, source_branch, source_version):
    print("Submitting build for branch: " + source_branch)
    print("Commit SHA1: ", source_version)

    run_build_raw = s.post(build_base_url, json={
        "definition": {"id": pipeline_id},
        "project": {"id": project_id},
        "sourceBranch": source_branch,
        "sourceVersion": source_version
    })

    try:
        run_build_json = run_build_raw.json()
    except json.decoder.JSONDecodeError as e:
        print(e)
        print("Failed to parse the response. Check if the Azure DevOps PAT is incorrect or expired.")
        sys.exit(-1)

    build_id = run_build_json['id']

    print("Submitted bulid: " + str(build_id))
    print("Bulid URL: " + run_build_json['url'])
    return build_id

def get_build(_id):
    get_build_url = AZURE_PIPELINE_BASE_URL + f"/_apis/build/builds/{_id}?api-version=6.0"
    get_build_raw = s.get(get_build_url)
    return get_build_raw.json()

def get_build_logs(_id):
    get_build_logs_url = AZURE_PIPELINE_BASE_URL + f"/_apis/build/builds/{_id}/logs?api-version=6.0"
    get_build_logs_raw = s.get(get_build_logs_url)
    return get_build_logs_raw.json()

def get_log_content(url):
    resp = s.get(url)
    return resp.text

def wait_for_build(_id):
    build_detail = get_build(_id)
    build_status = build_detail['status']

    while build_status == 'notStarted':
        print('Waiting for run to start: ' + str(_id))
        sys.stdout.flush()
        try:
            build_detail = get_build(_id)
            build_status = build_detail['status']
        except Exception as e:
            print("Error getting build")
            print(e)

        time.sleep(30)

    print("Bulid started: ", str(_id))

    handled_logs = set()
    while build_status == 'inProgress':
        try:
            print("Waiting for log: " + str(_id))
            logs = get_build_logs(_id)
        except Exception as e:
            print("Error fetching logs")
            print(e)
            time.sleep(30)
            continue

        for log in logs['value']:
            log_id = log['id']
            if log_id in handled_logs:
                continue
            handled_logs.add(log_id)
            print('Fetching log: \n' + log['url'])
            try:
                log_content = get_log_content(log['url'])
                print(log_content)
            except Exception as e:
                print("Error getting log content")
                print(e)
            sys.stdout.flush()
        build_detail = get_build(_id)
        build_status = build_detail['status']
        time.sleep(30)

    build_result = build_detail['result']

    print("Bulid status: " + build_status)
    print("Bulid result: " + build_result)

    return build_status, build_result

if __name__ == '__main__':
    # Convert the branch name for Azure DevOps
    match = re.search(r'pull/(\d+)', TARGET_BRANCH)
    if match is not None:
        pr_num = match.group(1)
        SOURCE_BRANCH = f'refs/pull/{pr_num}/head'
    else:
        SOURCE_BRANCH = f'refs/heads/{TARGET_BRANCH}'

    MAX_RETRY = 2
    retry = MAX_RETRY

    while retry > 0:
        build_id = submit_build(PIPELINE_ID, PROJECT_ID, SOURCE_BRANCH, TARGET_COMMIT)
        build_status, build_result = wait_for_build(build_id)

        if build_result != 'succeeded':
            retry = retry - 1
            if retry > 0:
                print("Retrying... remaining attempt: " + str(retry))
                # Wait a bit before retrying
                time.sleep((MAX_RETRY - retry) * 120)
                continue
            else:
                print("No more chance to retry. Giving up.")
                sys.exit(-1)
        else:
            break
