from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from tools.stats.upload_stats_lib import (
    download_s3_artifacts,
    upload_workflow_stats_to_s3,
)


def get_sccache_stats(
    workflow_run_id: int, workflow_run_attempt: int
) -> list[dict[str, Any]]:
    with TemporaryDirectory() as temp_dir:
        print("Using temporary directory:", temp_dir)
        os.chdir(temp_dir)

        # Download and extract all the reports (both GHA and S3)
        download_s3_artifacts("sccache-stats", workflow_run_id, workflow_run_attempt)

        stats_jsons = []
        for json_file in Path(".").glob("**/*.json"):
            with open(json_file) as f:
                stats_jsons.append(json.load(f))
        return stats_jsons


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload test stats to Rockset")
    parser.add_argument(
        "--workflow-run-id",
        type=int,
        required=True,
        help="id of the workflow to get artifacts from",
    )
    parser.add_argument(
        "--workflow-run-attempt",
        type=int,
        required=True,
        help="which retry of the workflow this is",
    )
    args = parser.parse_args()
    stats = get_sccache_stats(args.workflow_run_id, args.workflow_run_attempt)
    upload_workflow_stats_to_s3(
        args.workflow_run_id, args.workflow_run_attempt, "sccache_stats", stats
    )
