import os
from typing import Any, List, Union

import rockset  # type: ignore[import]


def run_rockset_query(
    collection_name: str,
    lambda_function_name: str,
    version: str,
    query_parameters: Union[List[Any], None] = None,
) -> Any:
    rs = rockset.RocksetClient(
        host="api.usw2a1.rockset.com", api_key=os.environ["ROCKSET_API_KEY"]
    )
    api_response = rs.QueryLambdas.execute_query_lambda(
        workspace=collection_name,
        query_lambda=lambda_function_name,
        version=version,
        parameters=query_parameters,
    )
    return api_response
