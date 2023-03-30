import rockset
import json
import os 
import argparse
import datetime

def get_test_stat_aggregates(date: str):
    # Initialize the Rockset client with your API key
    rockset_api_key = os.environ["ROCKSET_API_KEY"]
    rockset_api_server = "api.rs2.usw2.rockset.com"
    iso_date = date.isoformat()
    rs = rockset.RocksetClient(
            host="api.usw2a1.rockset.com", api_key=os.environ["ROCKSET_API_KEY"]
        )

    # Define the name of the Rockset collection and lambda function
    collection_name = "commons"
    lambda_function_name = "test_insights_per_daily_upload"
    query_parameters = [
        rockset.models.QueryParameter(name="startTime", type="string", value=iso_date)
    ]
    api_response = rs.QueryLambdas.execute_query_lambda(
    query_lambda=lambda_function_name,
    version="865e3748f31e9b59",
    parameters = query_parameters,
    )
    return api_response["results"]
    # Define the Rockset lambda query parameters
    # params = rockset.ParamDict(query_params)

    # Run the Rockset lambda query and get the result
    # result = rs.sql(lambda_function_name, sql_params=query_params, collection=collection_name)

    # Print the result as a JSON string
    # print(json.dumps(result, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload test stat aggregates to Rockset."
    )
    parser.add_argument(
        "--date", type=datetime.date.fromisoformat, help="Date to upload test stat aggregates for (YYYY-MM-DD). Must be in the last 30 days", required=True
    )
    args = parser.parse_args()
    print(args.date)
    if args.date < datetime.datetime.now().date() - datetime.timedelta(days=30):
        raise ValueError("date must be in the last 30 days")
    data = upload_test_stat_aggregates(date=args.date)
    upload_to_s3(
        bucket_name="torchci-testing-aggregate-data",
        key=f"test_data_aggregates/{str(args.date)}",
        docs=data,
    )
    
        