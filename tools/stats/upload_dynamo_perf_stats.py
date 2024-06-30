from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from tools.stats.upload_stats_lib import download_s3_artifacts, unzip, upload_to_rockset


ARTIFACTS = [
    "test-reports",
]
ARTIFACT_REGEX = re.compile(
    r"test-reports-test-(?P<name>[\w\-]+)-\d+-\d+-(?P<runner>[\w\.]+)_(?P<job>\d+).zip"
)


def upload_dynamo_perf_stats_to_rockset(
    repo: str,
    workflow_run_id: int,
    workflow_run_attempt: int,
    head_branch: str,
    match_filename: str,
) -> list[dict[str, Any]]:
    match_filename_regex = re.compile(match_filename)
    perf_stats = []
    with TemporaryDirectory() as temp_dir:
        print("Using temporary directory:", temp_dir)
        os.chdir(temp_dir)

        for artifact in ARTIFACTS:
            artifact_paths = download_s3_artifacts(
                artifact, workflow_run_id, workflow_run_attempt
            )

            # Unzip to get perf stats csv files
            for path in artifact_paths:
                m = ARTIFACT_REGEX.match(str(path))
                if not m:
                    print(f"Test report {path} has an invalid name. Skipping")
                    continue

                test_name = m.group("name")
                runner = m.group("runner")
                job_id = m.group("job")

                # Extract all files
                unzip(path)

                for csv_file in Path(".").glob("**/*.csv"):
                    filename = os.path.splitext(os.path.basename(csv_file))[0]
                    if not re.match(match_filename_regex, filename):
                        continue
                    print(f"Processing {filename} from {path}")

                    with open(csv_file) as csvfile:
                        reader = csv.DictReader(csvfile, delimiter=",")

                        for row in reader:
                            row.update(
                                {
                                    "workflow_id": workflow_run_id,  # type: ignore[dict-item]
                                    "run_attempt": workflow_run_attempt,  # type: ignore[dict-item]
                                    "test_name": test_name,
                                    "runner": runner,
                                    "job_id": job_id,
                                    "filename": filename,
                                    "head_branch": head_branch,
                                }
                            )
                            perf_stats.append(row)

                    # Done processing the file, removing it
                    os.remove(csv_file)

    return perf_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload dynamo perf stats from S3 to Rockset"
    )
    parser.add_argument(
        "--workflow-run-id",
        type=int,
        required=True,
        help="id of the workflow to get perf stats from",
    )
    parser.add_argument(
        "--workflow-run-attempt",
        type=int,
        required=True,
        help="which retry of the workflow this is",
    )
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="which GitHub repo this workflow run belongs to",
    )
    parser.add_argument(
        "--head-branch",
        type=str,
        required=True,
        help="head branch of the workflow",
    )
    parser.add_argument(
        "--rockset-collection",
        type=str,
        required=True,
        help="the name of the Rockset collection to store the stats",
    )
    parser.add_argument(
        "--rockset-workspace",
        type=str,
        default="commons",
        help="the name of the Rockset workspace to store the stats",
    )
    parser.add_argument(
        "--match-filename",
        type=str,
        default="",
        help="the regex to filter the list of CSV files containing the records to upload",
    )
    args = parser.parse_args()
    perf_stats = upload_dynamo_perf_stats_to_rockset(
        args.repo,
        args.workflow_run_id,
        args.workflow_run_attempt,
        args.head_branch,
        args.match_filename,
    )
    upload_to_rockset(
        collection=args.rockset_collection,
        docs=perf_stats,
        workspace=args.rockset_workspace,
    )
