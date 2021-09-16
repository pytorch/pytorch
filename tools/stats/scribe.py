import base64
import bz2
import os
import json
from typing import Dict, Any, List, Union


_lambda_client = None


IS_GHA = os.getenv("IS_GHA", "0") == "1"


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


def invoke_rds(events: List[Dict[str, Any]]) -> Any:
    if not IS_GHA:
        sprint(f"Not invoking RDS lambda outside GitHub Actions:\n{events}")
        return

    return invoke_lambda("rds-proxy", events)


def register_rds_schema(table_name: str, schema: Dict[str, str]) -> None:
    base = {
        "pr": "string",
        "ref": "string",
        "branch": "string",
        "workflow_id": "string",
    }

    event = [{"create_table": {"table_name": table_name, "fields": {**schema, **base}}}]

    invoke_rds(event)


def schema_from_sample(data: Dict[str, Any]) -> Dict[str, str]:
    schema = {}
    for key, value in data.items():
        if isinstance(value, str):
            schema[key] = "string"
        elif isinstance(value, int):
            schema[key] = "int"
        else:
            raise RuntimeError(f"Unsupported value type: {key}: {value}")
    return schema


Query = Dict[str, Any]


def rds_query(queries: Union[Query, List[Query]]) -> Any:
    if not isinstance(queries, list):
        queries = [queries]

    events = []
    for query in queries:
        events.append({"read": {**query}})

    return invoke_rds(events)


def rds_saved_query(query_names: Union[str, List[str]]) -> Any:
    if not isinstance(query_names, list):
        query_names = [query_names]

    events = []
    for name in query_names:
        events.append({"read": {"saved_query_name": name}})

    return invoke_rds(events)


def rds_write(
    table_name: str, values_list: List[Dict[str, Any]], only_on_master: bool = True
) -> None:
    sprint("Writing for ", os.getenv("CIRCLE_PR_NUMBER"))
    is_master = os.getenv("CIRCLE_PR_NUMBER", "").strip() == ""
    if not only_on_master and not is_master:
        sprint("Skipping RDS write on PR")
        return

    pr = os.getenv("CIRCLE_PR_NUMBER", None)
    if pr.strip() == "":
        pr = None

    base = {
        "pr": pr,
        "ref": os.getenv("CIRCLE_SHA1"),
        "branch": os.getenv("CIRCLE_BRANCH"),
        "workflow_id": os.getenv("GITHUB_WORKFLOW_RUN_ID"),
    }

    events = []
    for values in values_list:
        events.append(
            {"write": {"table_name": table_name, "values": {**values, **base}}}
        )

    sprint("Wrote stats for", table_name)
    invoke_rds(events)
