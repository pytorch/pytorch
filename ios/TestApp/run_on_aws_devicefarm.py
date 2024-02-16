#!/usr/bin/env python3

import datetime
import os
import random
import string
import sys
import time
import warnings
from typing import Any

import boto3
import requests

POLLING_DELAY_IN_SECOND = 5
MAX_UPLOAD_WAIT_IN_SECOND = 600

# NB: This is the curated top devices from AWS. We could create our own device
# pool if we want to
DEFAULT_DEVICE_POOL_ARN = (
    "arn:aws:devicefarm:us-west-2::devicepool:082d10e5-d7d7-48a5-ba5c-b33d66efa1f5"
)


def parse_args() -> Any:
    from argparse import ArgumentParser

    parser = ArgumentParser("Run iOS tests on AWS Device Farm")
    parser.add_argument(
        "--project-arn", type=str, required=True, help="the ARN of the project on AWS"
    )
    parser.add_argument(
        "--app-file", type=str, required=True, help="the iOS ipa app archive"
    )
    parser.add_argument(
        "--xctest-file",
        type=str,
        required=True,
        help="the XCTest suite to run",
    )
    parser.add_argument(
        "--name-prefix",
        type=str,
        required=True,
        help="the name prefix of this test run",
    )
    parser.add_argument(
        "--device-pool-arn",
        type=str,
        default=DEFAULT_DEVICE_POOL_ARN,
        help="the name of the device pool to test on",
    )

    return parser.parse_args()


def upload_file(
    client: Any,
    project_arn: str,
    prefix: str,
    filename: str,
    filetype: str,
    mime: str = "application/octet-stream",
):
    """
    Upload the app file and XCTest suite to AWS
    """
    r = client.create_upload(
        projectArn=project_arn,
        name=f"{prefix}_{os.path.basename(filename)}",
        type=filetype,
        contentType=mime,
    )
    upload_name = r["upload"]["name"]
    upload_arn = r["upload"]["arn"]
    upload_url = r["upload"]["url"]

    with open(filename, "rb") as file_stream:
        print(f"Uploading {filename} to Device Farm as {upload_name}...")
        r = requests.put(upload_url, data=file_stream, headers={"content-type": mime})
        if not r.ok:
            raise Exception(f"Couldn't upload {filename}: {r.reason}")

    start_time = datetime.datetime.now()
    # Polling AWS till the uploaded file is ready
    while True:
        waiting_time = datetime.datetime.now() - start_time
        if waiting_time > datetime.timedelta(seconds=MAX_UPLOAD_WAIT_IN_SECOND):
            raise Exception(
                f"Uploading {filename} is taking longer than {MAX_UPLOAD_WAIT_IN_SECOND} seconds, terminating..."
            )

        r = client.get_upload(arn=upload_arn)
        status = r["upload"].get("status", "")

        print(f"{filename} is in state {status} after {waiting_time}")

        if status == "FAILED":
            raise Exception(f"Couldn't upload {filename}: {r}")
        if status == "SUCCEEDED":
            break

        time.sleep(POLLING_DELAY_IN_SECOND)

    return upload_arn


def main() -> None:
    args = parse_args()

    client = boto3.client("devicefarm")
    unique_prefix = f"{args.name_prefix}-{datetime.date.today().isoformat()}-{''.join(random.sample(string.ascii_letters, 8))}"

    # Upload the test app
    appfile_arn = upload_file(
        client=client,
        project_arn=args.project_arn,
        prefix=unique_prefix,
        filename=args.app_file,
        filetype="IOS_APP",
    )
    print(f"Uploaded app: {appfile_arn}")
    # Upload the XCTest suite
    xctest_arn = upload_file(
        client=client,
        project_arn=args.project_arn,
        prefix=unique_prefix,
        filename=args.xctest_file,
        filetype="XCTEST_TEST_PACKAGE",
    )
    print(f"Uploaded XCTest: {xctest_arn}")

    # Schedule the test
    r = client.schedule_run(
        projectArn=args.project_arn,
        name=unique_prefix,
        appArn=appfile_arn,
        devicePoolArn=args.device_pool_arn,
        test={"type": "XCTEST", "testPackageArn": xctest_arn},
    )
    run_arn = r["run"]["arn"]

    start_time = datetime.datetime.now()
    print(f"Run {unique_prefix} is scheduled as {run_arn}:")

    state = "UNKNOWN"
    result = ""
    try:
        while True:
            r = client.get_run(arn=run_arn)
            state = r["run"]["status"]

            if state == "COMPLETED":
                result = r["run"]["result"]
                break

            waiting_time = datetime.datetime.now() - start_time
            print(
                f"Run {unique_prefix} in state {state} after {datetime.datetime.now() - start_time}"
            )
            time.sleep(30)
    except Exception as error:
        warnings.warn(f"Failed to run {unique_prefix}: {error}")
        sys.exit(1)

    if not result or result == "FAILED":
        print(f"Run {unique_prefix} failed, exiting...")
        sys.exit(1)


if __name__ == "__main__":
    main()
