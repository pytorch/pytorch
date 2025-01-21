from __future__ import annotations
# python script is mainly for uploading test stats to s3 for a test job

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from multiprocessing import cpu_count, Pool
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Optional
import zipfile

from tools.stats.utilization_stats_lib import UtilizationMetadata, UtilizationRecord, getDataModelVersion

from tools.stats.test_dashboard import upload_additional_info
from tools.stats.upload_stats_lib import (
    download_s3_artifacts,
    get_job_id,
    remove_nan_inf,
    unzip,
    upload_workflow_stats_to_s3,
)

USAGE_LOG_FILENAME = "usage_log.txt"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload test stats to s3")
    parser.add_argument(
        "--workflow-run-id",
        required=True,
        help="id of the workflow to get artifacts from",
    )
    parser.add_argument(
        "--workflow-run-attempt",
        type=int,
        required=True,
        help="which retry of the workflow this is",
    )
    parser.add_argument(
        "--workflow-name",
        required=True,
        help="id of the workflow to get artifacts from",
    )
    parser.add_argument(
        "--job-id",
        required=True,
        help="id of the workflow to get artifacts from",
    )
    parser.add_argument(
        "--job-name",
        required=True,
        help="id of the workflow to get artifacts from",
    )
    parser.add_argument(
        "--head-branch",
        required=True,
        help="Head branch of the workflow",
    )
    parser.add_argument(
        "--head-repository",
        required=True,
        help="Head repository of the workflow",
    )

    args = parser.parse_args()

    print(f"Workflow id is: {args.workflow_run_id}")

    # Flush stdout so that any errors in the upload show up last in the logs.
    sys.stdout.flush()


def unzip_file(path: Path,file_name: str) -> str:
    try:
        with zipfile.ZipFile(path) as zip_file:
            # Read the desired file from the zip archive
            return zip_file.read(name=file_name).decode()
    except Exception as e:
        print(f"::warning trying to download test log {object} failed by: {e}")
        return ""

def getTestLog(workflow_run_id: int,job_id: int, workflow_run_attempt: int):
    artifact_paths = download_s3_artifacts("logs-test",workflow_run_id,workflow_run_attempt,job_id)
    if len(artifact_paths) == 0:
        print(f"Failed to download artifacts for workflow {workflow_run_id} and job {job_id}")
        return
    elif len(artifact_paths) > 1:
        print(f"Found more than one artifact for workflow {workflow_run_id} and job {job_id}, {artifact_paths}")
        return

    p = artifact_paths[0]
    test_log_content = unzip_file(p,USAGE_LOG_FILENAME)
    metadata, records = convertToDataModel(test_log_content)
    if metadata == None:
        return
    print(f"metadata: {metadata}")


def prepareDbBatchInsertion(metadata: UtilizationMetadata, records: list[UtilizationRecord]):

    pass


def process_line(line: str) -> tuple[Optional[UtilizationRecord], bool]:
    try:
        record = UtilizationRecord.from_json(line)
        if record.error:
            return record, False
        return record, True
    except Exception as e:
        print(f"Failed to parse JSON line: {e}")
        return None, False

def convertToDataModel(content: str) -> tuple[Optional[UtilizationMetadata], list[UtilizationRecord]]:
    if not content:
        return None , []
    lines = content.splitlines()
    metadata = None
    if len(lines) < 2:
        print("Expected at least two records from log file")
        return None, []
    print(f"peek metadata json: {lines[0]}")
    print(f"peek log record json: {lines[1]}")

    try:
        metadata = UtilizationMetadata.from_json(lines[0])
    except Exception as e:
        print(f"Failed to parse metadata: {e} for data: {lines[0]}")
        return None, []

    if metadata.data_model_version != getDataModelVersion():
        print(f"Data model version mismatch: {metadata.data_model_version} != {getDataModelVersion()}")
        return None, []

    result_logs, _ = process_utilization_records(lines)

    if result_logs:
        return metadata,result_logs
    return metadata, []

def process_utilization_records(lines: list[str]) -> tuple[list[UtilizationRecord], list[UtilizationRecord]]:
    results = [process_line(line) for line in lines[1:]]
    valid_records = [record for record, valid in results if valid and record is not None]
    invalid_records = [record for record, valid in results if not valid and record is not None]
    return valid_records, invalid_records
