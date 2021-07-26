import base64
import bz2
import os
import json


def send_to_scribe(logs: str) -> str:
    access_token = os.environ.get("SCRIBE_GRAPHQL_ACCESS_TOKEN", "")

    # boto3 can be used when the runner has IAM roles setup
    # currently it's used as a fallback when SCRIBE_GRAPHQL_ACCESS_TOKEN is empty
    if access_token == "":
        return _send_to_scribe_via_boto3(logs)

    return _send_to_scribe_via_http(access_token, logs)


def _send_to_scribe_via_boto3(logs: str) -> str:
    # lazy import so that we don't need to introduce extra dependencies
    import boto3  # type: ignore[import]

    print("Scribe access token not provided, sending report via boto3...")
    event = {"base64_bz2_logs": base64.b64encode(bz2.compress(logs.encode())).decode()}
    client = boto3.client("lambda")
    res = client.invoke(FunctionName='gh-ci-scribe-proxy', Payload=json.dumps(event).encode())
    payload = str(res['Payload'].read().decode())
    if res.get('FunctionError'):
        raise Exception(payload)
    return payload


def _send_to_scribe_via_http(access_token: str, logs: str) -> str:
    # lazy import so that we don't need to introduce extra dependencies
    import requests  # type: ignore[import]

    print("Scribe access token provided, sending report via http...")
    r = requests.post(
        "https://graph.facebook.com/scribe_logs",
        data={"access_token": access_token, "logs": logs},
    )
    r.raise_for_status()
    return str(r.text)
