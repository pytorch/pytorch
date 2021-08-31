import base64
import bz2
import os
import json
from typing import Dict, Any, List, Union


_lambda_client = None


def aws_lambda() -> Any:
    global _lambda_client
    # lazy import so that we don't need to introduce extra dependencies
    import boto3  # type: ignore[import]

    if _lambda_client is None:
        _lambda_client = boto3.client("lambda")

    return _lambda_client


def send_to_scribe(logs: str) -> str:
    access_token = os.environ.get("SCRIBE_GRAPHQL_ACCESS_TOKEN", "")

    # boto3 can be used when the runner has IAM roles setup
    # currently it's used as a fallback when SCRIBE_GRAPHQL_ACCESS_TOKEN is empty
    if access_token == "":
        return _send_to_scribe_via_boto3(logs)

    return _send_to_scribe_via_http(access_token, logs)


def _send_to_scribe_via_boto3(logs: str) -> str:

    print("Scribe access token not provided, sending report via boto3...")
    event = {"base64_bz2_logs": base64.b64encode(bz2.compress(logs.encode())).decode()}
    res = aws_lambda().invoke(
        FunctionName="gh-ci-scribe-proxy", Payload=json.dumps(event).encode()
    )
    payload = str(res["Payload"].read().decode())
    if res.get("FunctionError"):
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


def _rds_invoke(events: List[Dict[str, Any]]) -> Any:
    res = aws_lambda().invoke(
        FunctionName="rds-proxy", Payload=json.dumps(events).encode()
    )
    payload = str(res["Payload"].read().decode())
    if res.get("FunctionError"):
        raise Exception(payload)

    return json.loads(payload)


def register_rds_table(table_name: str, schema: Dict[str, str]) -> None:
    base = {
        "pr": "string",
        "ref": "string",
        "branch": "string",
        "workflow_id": "string",
    }

    event = [
        {
            "create_table": {
                "table_name": table_name,
                "fields": {**schema, **base},
            }
        }
    ]

    _rds_invoke(event)


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

    return _rds_invoke(events)


def rds_saved_query(query_names: Union[str, List[str]]) -> Any:
    if not isinstance(query_names, list):
        query_names = [query_names]

    events = []
    for name in query_names:
        events.append(
            {
                "read": {
                    "saved_query_name": name,
                }
            }
        )

    return _rds_invoke(events)


def rds_write(table_name: str, values_list: List[Dict[str, Any]]) -> None:
    base = {
        "pr": os.environ.get("CIRCLE_PR_NUMBER"),
        "ref": os.environ.get("CIRCLE_SHA1"),
        "branch": os.environ.get("CIRCLE_BRANCH"),
        "workflow_id": os.environ.get("CIRCLE_WORKFLOW_ID"),
    }

    events = []
    for values in values_list:
        events.append(
            {
                "write": {
                    "table_name": table_name,
                    "values": {**values, **base},
                }
            }
        )

    _rds_invoke(events)


if __name__ == "__main__":
    # Test read
    print(rds_saved_query(["sample"]))
    print(rds_query([{"table_name": "workflow_run", "fields": ["name"], "limit": 10}]))
