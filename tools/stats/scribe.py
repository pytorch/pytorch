import base64
import bz2
import json
import os
from typing import Any


_lambda_client = None


def sprint(*args: Any) -> None:
    print("[scribe]", *args)


def aws_lambda() -> Any:
    global _lambda_client
    # lazy import so that we don't need to introduce extra dependencies
    import boto3  # type: ignore[import]

    if _lambda_client is None:
        _lambda_client = boto3.client("lambda")

    return _lambda_client


def invoke_lambda(name: str, payload: Any) -> Any:
    res = aws_lambda().invoke(FunctionName=name, Payload=json.dumps(payload).encode())
    payload = str(res["Payload"].read().decode())
    if res.get("FunctionError"):
        raise Exception(payload)
    return payload


def send_to_scribe(logs: str) -> str:
    access_token = os.environ.get("SCRIBE_GRAPHQL_ACCESS_TOKEN", "")

    # boto3 can be used when the runner has IAM roles setup
    # currently it's used as a fallback when SCRIBE_GRAPHQL_ACCESS_TOKEN is empty
    if access_token == "":
        return _send_to_scribe_via_boto3(logs)

    return _send_to_scribe_via_http(access_token, logs)


def _send_to_scribe_via_boto3(logs: str) -> str:
    sprint("Scribe access token not provided, sending report via boto3...")
    event = {"base64_bz2_logs": base64.b64encode(bz2.compress(logs.encode())).decode()}
    return str(invoke_lambda("gh-ci-scribe-proxy", event))


def _send_to_scribe_via_http(access_token: str, logs: str) -> str:
    # lazy import so that we don't need to introduce extra dependencies
    import requests  # type: ignore[import]

    sprint("Scribe access token provided, sending report via http...")
    r = requests.post(
        "https://graph.facebook.com/scribe_logs",
        data={"access_token": access_token, "logs": logs},
    )
    r.raise_for_status()
    return str(r.text)
