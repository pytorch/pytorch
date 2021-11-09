import os
import json
from typing import Dict, Any, List, Union, Optional


_lambda_client = None


IS_GHA = os.getenv("IS_GHA", "0") == "1"


def _sprint(*args: Any) -> None:
    print("[scribe]", *args)


# This is copy-pasted from scribe.py for easier importing in other projects (so
# this file is entirely self contained)
def _aws_lambda() -> Any:
    global _lambda_client
    # lazy import so that we don't need to introduce extra dependencies
    import boto3  # type: ignore[import]

    if _lambda_client is None:
        _lambda_client = boto3.client("lambda")

    return _lambda_client


def _invoke_lambda(name: str, payload: Any) -> Any:
    res = _aws_lambda().invoke(FunctionName=name, Payload=json.dumps(payload).encode())
    payload = str(res["Payload"].read().decode())
    if res.get("FunctionError"):
        raise Exception(payload)
    return payload


def _invoke_rds(events: List[Dict[str, Any]]) -> Any:
    if not IS_GHA:
        _sprint(f"Not invoking RDS lambda outside GitHub Actions:\n{events}")
        return

    return _invoke_lambda("rds-proxy", events)


def register_rds_schema(table_name: str, schema: Dict[str, str]) -> None:
    """
    Register a table in RDS so it can be written to later on with 'rds_write'.
    'schema' should be a mapping of field names -> types, where supported types
    are 'int' and 'string'.

    Metadata fields such as pr, ref, branch, workflow_id, and build_environment
    will be added automatically.
    """
    base = {
        "pr": "string",
        "ref": "string",
        "branch": "string",
        "workflow_id": "string",
        "build_environment": "string",
    }

    event = [{"create_table": {"table_name": table_name, "fields": {**schema, **base}}}]

    _invoke_rds(event)


def schema_from_sample(data: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract a schema compatible with 'register_rds_schema' from data.
    """
    schema = {}
    for key, value in data.items():
        if isinstance(value, str):
            schema[key] = "string"
        elif isinstance(value, int):
            schema[key] = "int"
        elif isinstance(value, float):
            schema[key] = "float"
        else:
            raise RuntimeError(f"Unsupported value type: {key}: {value}")
    return schema


Query = Dict[str, Any]


def rds_query(queries: Union[Query, List[Query]]) -> Any:
    """
    Execute a simple read query on RDS. Queries should be of the form below,
    where everything except 'table_name' and 'fields' is optional.

    {
        "table_name": "my_table",
        "fields": ["something", "something_else"],
        "where": [
            {
                "field": "something",
                "value": 10
            }
        ],
        "group_by": ["something"],
        "order_by": ["something"],
        "limit": 5,
    }
    """
    if not isinstance(queries, list):
        queries = [queries]

    events = []
    for query in queries:
        events.append({"read": {**query}})

    return _invoke_rds(events)


def rds_saved_query(query_names: Union[str, List[str]]) -> Any:
    """
    Execute a hardcoded RDS query by name. See
    https://github.com/pytorch/test-infra/blob/main/aws/lambda/rds-proxy/lambda_function.py#L52
    for available queries or submit a PR there to add a new one.
    """
    if not isinstance(query_names, list):
        query_names = [query_names]

    events = []
    for name in query_names:
        events.append({"read": {"saved_query_name": name}})

    return _invoke_rds(events)


def rds_write(
    table_name: str,
    values_list: List[Dict[str, Any]],
    only_on_master: bool = True,
    only_on_jobs: Optional[List[str]] = None,
) -> None:
    """
    Note: Only works from GitHub Actions CI runners

    Write a set of entries to a particular RDS table. 'table_name' should be
    a table registered via 'register_rds_schema' prior to calling rds_write.
    'values_list' should be a list of dictionaries that map field names to
    values.
    """
    _sprint("Writing for", os.getenv("CIRCLE_PR_NUMBER"))
    is_master = os.getenv("CIRCLE_PR_NUMBER", "").strip() == ""
    if only_on_master and not is_master:
        _sprint("Skipping RDS write on PR")
        return

    pr = os.getenv("CIRCLE_PR_NUMBER", None)
    if pr is not None and pr.strip() == "":
        pr = None

    build_environment = os.environ.get("BUILD_ENVIRONMENT", "").split()[0]
    if only_on_jobs is not None and build_environment not in only_on_jobs:
        _sprint(f"Skipping write since {build_environment} is not in {only_on_jobs}")
        return

    base = {
        "pr": pr,
        "ref": os.getenv("CIRCLE_SHA1"),
        "branch": os.getenv("CIRCLE_BRANCH"),
        "workflow_id": os.getenv("GITHUB_RUN_ID"),
        "build_environment": build_environment,
    }

    events = []
    for values in values_list:
        events.append(
            {"write": {"table_name": table_name, "values": {**values, **base}}}
        )

    _sprint("Wrote stats for", table_name)
    _invoke_rds(events)
