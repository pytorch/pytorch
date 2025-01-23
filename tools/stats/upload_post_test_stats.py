from __future__ import annotations

import argparse
from asyncio import run
import os
import sys
from typing import Any, Optional
import pandas as pd
import datetime
import json
from dataclasses import asdict, dataclass

# python script is mainly for uploading test stats to s3 for a test job
# adding sys.path makes the monitor script able to import path tools.stats.utilization_stats_lib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import zipfile
from pathlib import Path
from tools.stats.upload_stats_lib import upload_to_s3
from tools.stats.upload_stats_lib import download_s3_artifacts
from tools.stats.utilization_stats_lib import (
    getDataModelVersion,
    UtilizationMetadata,
    UtilizationRecord,
    OssCiUtilizationMetadataV1,
    OssCiUtilizationTimeSeriesV1,
    oss_ci_utilization_segment_v1,
)
USAGE_LOG_FILENAME = "usage_log.txt"
CMD_PYTHON_LEVEL = "CMD_PYTHON"
UTILIZATION_BUCKET = "ossci-utilization"

class SegmentGenerator():
    def generate(self,logs:list[UtilizationRecord]):
        df = self.get_pytest_cmd_timestamp(logs)
        cmd_list = self._get_unique_pytest_cmd(df)
        segments = self._generate_segments(df,'cmd',cmd_list)
        print(f"[Db Segments] detected pytest cmd: {len(cmd_list)}, generated segments: {len(segments)}")
        return segments

    def _get_unique_pytest_cmd(self,df) -> list[str]:
        unique_cmds_df = pd.DataFrame(df['cmd'].unique(), columns=['cmd'])
        result = unique_cmds_df[unique_cmds_df['cmd'].str.startswith("python")]['cmd'].tolist()
        return result

    def _find_continuous_windows(self,dataframe, threshold, time_column_name='time') -> list[dict[str, Any]]:
        time_threshold = pd.Timedelta(seconds= threshold)
        dataframe = dataframe.sort_values(by=time_column_name).reset_index(drop=True)
        dataframe['time_diff'] = dataframe[time_column_name].diff()
        dataframe['segment'] = (dataframe['time_diff'] > time_threshold).cumsum()
        segments = dataframe.groupby('segment').agg(
            start_time=(time_column_name, 'first'),
            end_time=(time_column_name, 'last')
        ).reset_index(drop=True)
        return segments[['start_time', 'end_time']].to_dict(orient="records")

    def _generate_segments(self,data_frame_with_time, column_name, column_values, delta_time_threshold = 60) -> list[oss_ci_utilization_segment_v1]:
        segments:list[oss_ci_utilization_segment_v1] = []
        for value in column_values:
            subset = data_frame_with_time[data_frame_with_time[column_name] == value].copy()
            continuous_segments = self._find_continuous_windows(subset, delta_time_threshold)
            for row in continuous_segments:
                segment = oss_ci_utilization_segment_v1(
                    level= CMD_PYTHON_LEVEL,
                    name=value,
                    start_at = get_datetime_string(row['start_time'].timestamp()),
                    end_at=get_datetime_string(row['end_time'].timestamp()),
                    extra_info = {},
                )
                segments.append(segment)
        return segments

    def get_pytest_cmd_timestamp(self,records:list[UtilizationRecord]) -> pd.DataFrame:
        # Flatten logs with processes into a single list of dictionaries
        flattened_cmd_and_time = [
            {'time': record.timestamp , 'cmd': process}
            for record in records
            for process in (record.cmd_names or [])
        ]
        # Create a Pandas DataFrame from the flattened list
        df = pd.DataFrame(flattened_cmd_and_time)
        df["time"] = pd.to_datetime(df['time'], unit='s')
        return df



def get_datetime_string(timestamp: float) -> str:
        dt = datetime.datetime.fromtimestamp(timestamp)
        dt_str = dt.strftime("%Y-%m-%d %H:%M:%S.%f")
        return dt_str

class UtilizationDbConverter():
    def __init__(self,
    workflow_run_id: int,
    workflow_name:str,
    job_id:int,
    run_attempt:int,
    job_name:str,
    metadata: UtilizationMetadata,
    records: list[UtilizationRecord],
    segments: list[oss_ci_utilization_segment_v1],
    repo = "pytorch/pytorch"):
        self.repo = repo
        self.metadata = metadata
        self.records = records
        self.segments = segments
        dt = datetime.datetime.now().timestamp()
        dt_str = get_datetime_string(dt)
        self.created_at = dt_str
        self.run_attempt = run_attempt
        self.workflow_run_id = workflow_run_id
        self.workflow_name = workflow_name
        self.job_id = job_id
        self.job_name = job_name
        end_time_stamp =  max([record.timestamp for record in records])
        self.end_at = get_datetime_string(end_time_stamp)

    def convert(self) -> tuple[OssCiUtilizationMetadataV1, list[OssCiUtilizationTimeSeriesV1]]:
        db_metadata = self._to_oss_ci_metadata()
        timeseries = self._to_oss_ci_timeseries_list()
        return db_metadata, timeseries

    def _to_oss_ci_metadata(self) -> OssCiUtilizationMetadataV1:
        return OssCiUtilizationMetadataV1(
            repo= self.repo,
            workflow_id= self.workflow_run_id,
            run_attempt= self.run_attempt,
            job_id= self.job_id,
            workflow_name= self.workflow_name,
            job_name= self.job_name,
            usage_collect_interval= self.metadata.usage_collect_interval,
            data_model_version= str(self.metadata.data_model_version),
            created_at= self.created_at,
            gpu_count= self.metadata.gpu_count if self.metadata.gpu_count else 0,
            cpu_count= self.metadata.cpu_count if self.metadata.cpu_count else 0,
            gpu_type= self.metadata.gpu_type if self.metadata.gpu_type else "",
            start_at= get_datetime_string(self.metadata.start_at),
            end_at= self.end_at,
            segments= self.segments,
        )
    def _to_oss_ci_timeseries_list(self) -> list[OssCiUtilizationTimeSeriesV1]:
        return [self._to_oss_ci_time_series(record,type= "utilization",tags=["record"]) for record in self.records]

    def _to_oss_ci_time_series(self, record: UtilizationRecord, type:str, tags: list[str]) -> OssCiUtilizationTimeSeriesV1:
        return OssCiUtilizationTimeSeriesV1(
            created_at= self.created_at,
            type= type,
            tags = tags,
            time_stamp= get_datetime_string(record.timestamp),
            repo= self.repo,
            workflow_id= self.workflow_run_id,
            run_attempt= self.run_attempt,
            job_id= self.job_id,
            workflow_name= self.workflow_name,
            job_name= self.job_name,
            json_data = str(asdict(record.data) if record.data else {}),
        )

def unzip_file(path: Path, file_name: str):
    try:
        with zipfile.ZipFile(path) as zip_file:
            # Read the desired file from the zip archive
            return zip_file.read(name=file_name).decode()
    except Exception as e:
        print(f"::warning trying to download test log {object} failed by: {e}")
        return ""

def process_line(line: str):
    try:
        record = UtilizationRecord.from_json(line)
        if record.error:
            return record, False
        return record, True
    except Exception as e:
        print(f"Failed to parse JSON line: {e}")
        return None, False

def process_utilization_records(
    lines: list[str],
) -> tuple[list[UtilizationRecord], list[UtilizationRecord]]:
    results = [process_line(line) for line in lines[1:]]
    valid_records = [
        record for record, valid in results if valid and record is not None
    ]
    invalid_records = [
        record for record, valid in results if not valid and record is not None
    ]
    return valid_records, invalid_records

def convert_to_log_models(content: str) -> tuple[Optional[UtilizationMetadata], list[UtilizationRecord], list[UtilizationRecord]]:
    if not content:
        return None, [],[]
    lines = content.splitlines()
    metadata = None
    if len(lines) < 2:
        print("Expected at least two records from log file")
        return None, [],[]
    print(f"[Raw Log] Peek raw metadata json: {lines[0]} \n")
    print(f"[Raw Log] Peek raw record json: {lines[1]} \n")

    try:
        metadata = UtilizationMetadata.from_json(lines[0])
    except Exception as e:
        print(f"Failed to parse metadata: {e} for data: {lines[0]}")
        return None, [],[]

    if metadata.data_model_version != getDataModelVersion():
        print(
            f"Data model version mismatch: {metadata.data_model_version} != {getDataModelVersion()}"
        )
        return None, [],[]

    result_logs, error_logs = process_utilization_records(lines)
    return metadata, result_logs,error_logs

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Upload test stats to s3")
    parser.add_argument(
        "--workflow-run-id",
        type=int,
        required=False,
        help="id of the workflow to get artifacts from",
    )
    parser.add_argument(
        "--workflow-run-attempt",
        type=int,
        required=False,
        help="which retry of the workflow this is",
    )
    parser.add_argument(
        "--workflow-name",
        type=str,
        required=False,
        help="id of the workflow to get artifacts from",
    )
    parser.add_argument(
        "--job-id",
        type=int,
        required=False,
        help="id of the workflow to get artifacts from",
    )
    parser.add_argument(
        "--job-name",
        type=str,
        required=False,
        help="id of the workflow to get artifacts from",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode")

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Enable dry-run mode")

    return parser.parse_args()


class UploadUtilizationData:
    def __init__(self, workflow_run_id: int, job_id: int, workflow_run_attempt: int, workflow_name: str, job_name: str, dry_run: bool = False, debug: bool = False):
        self.workflow_run_id = workflow_run_id
        self.job_id = job_id
        self.workflow_run_attempt = workflow_run_attempt
        self.workflow_name = workflow_name
        self.job_name = job_name
        self.segment_generator = SegmentGenerator()
        self.debug_mode = debug
        self.dry_run = dry_run

    def _process_test_log(self, workflow_run_id: int, job_id: int, workflow_run_attempt: int) -> tuple[Optional[UtilizationMetadata], list[UtilizationRecord], list[UtilizationRecord]]:
        artifact_paths = download_s3_artifacts(
        "logs-test", workflow_run_id, workflow_run_attempt, job_id
    )
        if len(artifact_paths) == 0:
            print(
                f"Failed to download artifacts for workflow {workflow_run_id} and job {job_id}"
            )
            return None, [], []
        elif len(artifact_paths) > 1:
            print(
                f"Found more than one artifact for workflow {workflow_run_id} and job {job_id}, {artifact_paths}"
            )
            return None, [], []

        p = artifact_paths[0]
        test_log_content = unzip_file(p, USAGE_LOG_FILENAME)
        metadata, records,error_records = convert_to_log_models(test_log_content)
        if metadata == None:
            return None, [], []

        print(f"Converted Log Model: UtilizationMetadata:\n {metadata}")
        return metadata, records, error_records

    def _upload_utilization_data_to_s3(
        self,
        collection: str,
        workflow_run_id: int,
        workflow_run_attempt: int,
        job_id: int,
        name: str,
        docs: list[dict[str, Any]],
    ) -> None:
        bucket_name = UTILIZATION_BUCKET
        key = f"{collection}/{workflow_run_id}/{workflow_run_attempt}/{job_id}/{name}"
        upload_to_s3(bucket_name, key, docs)

    def start(self):
        metadata,valid_records,_ = self._process_test_log(self.workflow_run_id,self.job_id, self.workflow_run_attempt)
        if metadata == None:
            print("[Log Model] Failed to process test log, metadata is None")
            return
        segments = self.segment_generator.generate(valid_records)
        db_metadata, db_records= UtilizationDbConverter(
            workflow_run_id,
            workflow_name,
            job_id,
            workflow_run_attempt,
            job_name,
            metadata,
            valid_records,
            segments,
        ).convert()
        print(f"[db model] Peek db metadatga \n: {json.dumps(asdict(db_metadata),indent=4)}")

        if len(db_records) > 0:
            print(f"[db model] Peek db timeseries \n:{json.dumps(asdict(db_records[0]),indent=4)}")

        if self.dry_run:
            print("[dry-run-mode]: no upload in dry run mode")
            return

        self._upload_utilization_data_to_s3(f"utilization_v{db_metadata.data_model_version}", self.workflow_run_id, self.workflow_run_attempt, self.job_id, "metadata", [asdict(db_metadata)])
        self._upload_utilization_data_to_s3(f"utilization_v{db_metadata.data_model_version}", self.workflow_run_id, self.workflow_run_attempt, self.job_id, "timeseries", [asdict(record) for record in db_records])

if __name__ == "__main__":
    workflow_run_id = 12838344646
    job_id = 35805158112
    workflow_name = "inductor-unittest"
    workflow_run_attempt = 1
    job_name = (
        "cuda12.4-py3.12-gcc9-sm86 / test (inductor, 2, 2, linux.g5.4xlarge.nvidia.gpu)"
    )
    args = parse_args()
    ud= UploadUtilizationData(
        workflow_run_id,
        job_id,
        workflow_run_attempt,
        workflow_name,
        job_name,
        dry_run = args.dry_run,
        debug =args.debug)
    ud.start()
    # Flush stdout so that any errors in the upload show up last in the logs.
    sys.stdout.flush()
