import os
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

from rockset import RocksetClient
import argparse
import pandas as pd

def query_job_sha(repo, sha, api_key):
    rs = RocksetClient(api_key=api_key, host="https://api.usw2a1.rockset.com")

    params = list()
    params.append({"name": "repo", "type": "string", "value": repo})
    params.append({"name": "sha", "type": "string", "value": sha})

    response = rs.QueryLambdas.execute_query_lambda(
        query_lambda="commit_jobs_query",
        version="cc524c5036e78794",
        workspace="commons",
        parameters=params,
    )
    return response.results


def parse_job_name(job_str):
    return (part.strip() for part in job_str.split("/"))


def parse_test_str(test_str):
    return (part.strip() for part in test_str[6:].strip(")k").split(","))


S3_BASE_URL = "https://gha-artifacts.s3.amazonaws.com"


def get_artifacts_urls(results, suites):
    urls = {}
    for r in results:
        if "inductor" == r["workflowName"] and f"test" in r["jobName"]:
            config_str, test_str = parse_job_name(r["jobName"])
            suite, shard_id, num_shards, machine = parse_test_str(test_str)
            workflowId = r["workflowId"]
            id = r["id"]
            runattempt = 1  # ? guessing here

            if suite in suites:
                artifact_filename = f"test-reports-test-{suite}-{shard_id}-{num_shards}-{machine}_{id}.zip"
                s3_url = f"{S3_BASE_URL}/{repo}/{workflowId}/{runattempt}/artifact/{artifact_filename}"
                urls[(suite, int(shard_id))] = s3_url
                print(f"{suite} {shard_id}, {num_shards}: {s3_url}")
    return urls

def normalize_suite_filename(suite_name):
    assert suite_name.find("inductor_") == 0
    subsuite = suite_name.split("_")[1]
    if "timm" in subsuite:
      subsuite = f"{subsuite}_models"
    return subsuite

def download_artifacts_and_extract_csvs(urls):
  dataframes = {}
  for (suite, shard), url in urls.items():
      resp = urlopen(url)
      subsuite = normalize_suite_filename(suite)
      artifact = ZipFile(BytesIO(resp.read()))
      for phase in ("training", "inference"):
        name = f"test/test-reports/{phase}_{subsuite}.csv"
        df = pd.read_csv(artifact.open(name))
        dataframes[(suite, phase, shard)] = df

  return dataframes

def write_filtered_csvs(root_path, dataframes):
  for (suite, phase, shard), df in dataframes.items():
    if "timm" in suite:
      out_fn = os.path.join(root_path, f"{phase}_{suite}{shard - 1}.csv")
    else:
      out_fn = os.path.join(root_path, f"{phase}_{suite}.csv")
    df.to_csv(out_fn, index=False, columns=["name", "graph_breaks"])


parser = argparse.ArgumentParser()
parser.add_argument("sha")
args = parser.parse_args()

repo = "pytorch/pytorch"

root_path = "benchmarks/dynamo/ci_expected_accuracy/"
# TODO open public rocksdb endpoint, no apikey
with open(os.path.join(os.path.expanduser("~"), "rocks_api_key"), "r") as f:
    api_key = f.read()

results = query_job_sha(repo, args.sha, api_key)
suites = {
    "inductor_huggingface",
    # "inductor_huggingface_dynamic",
    "inductor_timm",
    # "inductor_timm_dynamic",
    "inductor_torchbench",
    # "inductor_torchbench_dynamic",
}
urls = get_artifacts_urls(results, suites)
dataframes = download_artifacts_and_extract_csvs(urls)
write_filtered_csvs(root_path, dataframes)
