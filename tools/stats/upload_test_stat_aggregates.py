import argparse
import ast
import datetime
import json
import os
import re
from typing import Any, List, Union

import rockset  # type: ignore[import]

from tools.stats.upload_stats_lib import upload_to_s3


def get_oncall_from_testfile(testfile: str) -> Union[List[str], None]:
    path = f"test/{testfile}"
    if not path.endswith(".py"):
        path += ".py"
    # get oncall on test file
    try:
        with open(path) as f:
            for line in f:
                if line.startswith("# Owner(s): "):
                    possible_lists = re.findall(r"\[.*\]", line)
                    if len(possible_lists) > 1:
                        raise Exception("More than one list found")
                    elif len(possible_lists) == 0:
                        raise Exception("No oncalls found or file is badly formatted")
                    oncalls = ast.literal_eval(possible_lists[0])
                    return list(oncalls)
    except Exception as e:
        if "." in testfile:
            return [f"module: {testfile.split('.')[0]}"]
        else:
            return ["module: unmarked"]
    return None


def get_test_stat_aggregates(date: datetime.date) -> Any:
    # Initialize the Rockset client with your API key
    rockset_api_key = os.environ["ROCKSET_API_KEY"]
    rockset_api_server = "api.rs2.usw2.rockset.com"
    iso_date = date.isoformat()
    rs = rockset.RocksetClient(
        host="api.usw2a1.rockset.com", api_key=rockset_api_key
    )

    # Define the name of the Rockset collection and lambda function
    collection_name = "commons"
    lambda_function_name = "test_insights_per_daily_upload"
    query_parameters = [
        rockset.models.QueryParameter(name="startTime", type="string", value=iso_date)
    ]
    api_response = rs.QueryLambdas.execute_query_lambda(
        query_lambda=lambda_function_name,
        version="692684fa5b37177f",
        parameters=query_parameters,
    )
    for i in range(len(api_response["results"])):
        oncalls = get_oncall_from_testfile(api_response["results"][i]["test_file"])
        api_response["results"][i]["oncalls"] = oncalls
    return json.loads(
        json.dumps(api_response["results"], indent=4, sort_keys=True, default=str)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload test stat aggregates to Rockset."
    )
    parser.add_argument(
        "--date",
        type=datetime.date.fromisoformat,
        help="Date to upload test stat aggregates for (YYYY-MM-DD). Must be in the last 30 days",
        required=True,
    )
    args = parser.parse_args()
    if args.date < datetime.datetime.now().date() - datetime.timedelta(days=30):
        raise ValueError("date must be in the last 30 days")
    data = get_test_stat_aggregates(date=args.date)
    upload_to_s3(
        bucket_name="torchci-aggregated-stats",
        key=f"test_data_aggregates/{str(args.date)}",
        docs=data,
    )
