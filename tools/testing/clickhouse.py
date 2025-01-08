import json
import os
from functools import lru_cache
from typing import Any

import clickhouse_connect  # type: ignore[import]


@lru_cache(maxsize=1)
def get_clickhouse_client() -> Any:
    endpoint = os.environ["CLICKHOUSE_ENDPOINT"]
    # I cannot figure out why these values aren't being handled automatically
    # when it is fine in the lambda
    if endpoint.startswith("https://"):
        endpoint = endpoint[len("https://") :]
    if endpoint.endswith(":8443"):
        endpoint = endpoint[: -len(":8443")]
    return clickhouse_connect.get_client(
        host=endpoint,
        user=os.environ["CLICKHOUSE_USERNAME"],
        password=os.environ["CLICKHOUSE_PASSWORD"],
        secure=True,
        interface="https",
        port=8443,
    )


def query_clickhouse(query: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Queries ClickHouse.  Returns datetime in YYYY-MM-DD HH:MM:SS format.
    """

    def convert_to_json_list(res: bytes) -> list[dict[str, Any]]:
        rows = []
        for row in res.decode().split("\n"):
            if row:
                rows.append(json.loads(row))
        return rows

    res = get_clickhouse_client().raw_query(query, params, fmt="JSONEachRow")
    return convert_to_json_list(res)
