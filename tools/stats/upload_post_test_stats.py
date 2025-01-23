from __future__ import annotations

import argparse
from asyncio import run
import os
import sys
from typing import Optional
import pandas as pd
import datetime

# python script is mainly for uploading test stats to s3 for a test job
# adding sys.path makes the monitor script able to import path tools.stats.utilization_stats_lib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import zipfile
from pathlib import Path

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

class SegmentGenerator():
    def generate(self,logs:list[UtilizationRecord]):
        df = self.get_pytest_cmd_timestamp(logs)
        cmd_list = self._get_unique_pytest_cmd(df)
        segments = self._generate_segments(df,'cmd',cmd_list)
        print(f"detected pytest cmd: {len(cmd_list)}, generated segments: {len(segments)}")
        return segments

    def _get_unique_pytest_cmd(self,df):
        unique_cmds_df = pd.DataFrame(df['cmd'].unique(), columns=['cmd'])
        result = unique_cmds_df[unique_cmds_df['cmd'].str.startswith("python")]['cmd'].tolist()
        return result

    def _find_continuous_windows(self,dataframe, threshold, time_column_name='time'):
        time_threshold = pd.Timedelta(seconds= threshold)
        dataframe = dataframe.sort_values(by=time_column_name).reset_index(drop=True)
        dataframe['time_diff'] = dataframe[time_column_name].diff()
        dataframe['segment'] = (dataframe['time_diff'] > time_threshold).cumsum()
        segments = dataframe.groupby('segment').agg(
            start_time=(time_column_name, 'first'),
            end_time=(time_column_name, 'last')
        ).reset_index(drop=True)
        return segments[['start_time', 'end_time']].to_dict(orient="records")

    def _generate_segments(self,data_frame_with_time, column_name, column_values, delta_time_threshold = 60):
        segments:list[oss_ci_utilization_segment_v1] = []
        for value in column_values:
            subset = data_frame_with_time[data_frame_with_time[column_name] == value].copy()
            continuous_segments = self._find_continuous_windows(subset, delta_time_threshold)
            for row in continuous_segments:
                print("test segments",value,row['start_time'],row['end_time'])
                segment = oss_ci_utilization_segment_v1(
                    level="pytest",
                    name=value,
                    start_at=row['start_time'],
                    end_at=row['end_time'],
                    extra_info = {},
                )
                segments.append(segment)
        return segments

    def get_pytest_cmd_timestamp(self,records:list[UtilizationRecord]):
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

def process_test_log(workflow_run_id: int, job_id: int, workflow_run_attempt: int) -> tuple[Optional[UtilizationMetadata], list[UtilizationRecord], list[UtilizationRecord]]:
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
    metadata, records,error_records = convert_to_data_model(test_log_content)
    if metadata == None:
        return None, [], []
    print(f"metadata: {metadata}")
    return metadata, records, error_records


class Converter():
    def __init__(self,
    workflow_run_id: str,
    workflow_name:str,
    job_id:str,
    run_attempt:int,
    job_name:str,
    metadata: UtilizationMetadata,
    records: list[UtilizationRecord],
    segments: list[oss_ci_utilization_segment_v1]):
        self.metadata = metadata
        self.records = records
        self.segments = segments
        dt = datetime.datetime.now(datetime.UTC)
        dt_str = dt.strftime("%Y-%m-%d %H:%M:%S.%f")
        self.created_at = dt_str
        self.workflow_run_id = workflow_run_id
        self.workflow_name = workflow_name
        self.job_id = job_id
        self.job_name = job_name
        self.end_at = max((record.timestamp for record in records if record.timestamp is not None), default=None)
    def _get_datetime_string(self,timestamp: float):
        dt = datetime.datetime.fromtimestamp(timestamp,datetime.UTC)
        dt_str = dt.strftime("%Y-%m-%d %H:%M:%S.%f")
        return dt_str

    def convert(self):
        return
    def _to_oss_ci_metadata(self, metadata: UtilizationMetadata) -> OssCiUtilizationMetadataV1:
         OssCiUtilizationMetadataV1(
            created_at = self.created_at,
            repo= "pytorch/pytorch",
            workflow_id= self.workflow_run_id,
            run_attempt= self.run_attempt,
            job_id= self.job_id,
            workflow_name= self.workflow_name,
            job_name= self.job_name,
            usage_collect_interval= metadata.usage_collect_interval,
            data_model_version=self.metadata.data_model_version,
            gpu_count= self.metadata.gpu_count,
            cpu_count= self.metadata.cpu_count,
            gpu_type= self.metadata.gpu_type,
            start_at= self.metadata.start_at,
            end_at= ,
            head_branch=self.metadata.head_branch,
            head_repository=self.metadata.head_repository,
            segments=self.segments,
            utilization_records=self.records,

        )

        return OssCiUtilizationMetadataV1()



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

def convert_to_data_model(content: str) -> tuple[Optional[UtilizationMetadata], list[UtilizationRecord], list[UtilizationRecord]]:
    if not content:
        return None, [],[]
    lines = content.splitlines()
    metadata = None
    if len(lines) < 2:
        print("Expected at least two records from log file")
        return None, [],[]
    print(f"peek metadata json: {lines[0]}")
    print(f"peek log record json: {lines[1]}")

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



def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
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
    return parser.parse_args()

def prepareDbBatchInsertion(
    metadata: UtilizationMetadata, records: list[UtilizationRecord], segments: list[oss_ci_utilization_segment_v1]
):




def upload_logs_stats_to_s3(
    workflow_run_id: int,
    workflow_run_attempt: int,
    collection: str,
    docs: list[dict[str, Any]],
) -> None:
    bucket_name = "ossci-raw-job-status"
    key = f"{collection}/{workflow_run_id}/{workflow_run_attempt}"
    upload_to_s3(bucket_name, key, docs)



if __name__ == "__main__":
    # args = parse_args()
    # print(f"Workflow id is: {args.workflow_run_id}")
    workflow_run_id = 12838344646
    job_id = 35805158112
    workflow_name = "inductor-unittest"
    workflow_run_attempt = 1
    job_name = (
        "cuda12.4-py3.12-gcc9-sm86 / test (inductor, 2, 2, linux.g5.4xlarge.nvidia.gpu)"
    )
    metadata,records,error_records = process_test_log(workflow_run_id, job_id, workflow_run_attempt)
    generator = SegmentGenerator()
    segments = generator.generate(records)

    # Flush stdout so that any errors in the upload show up last in the logs.
    sys.stdout.flush()

@dataclass
class oss_ci_utilization_segment_v1(DataClassJsonMixin):
    level: str
    name: str
    start_at: str
    end_at: str
    extra_info: dict[str, str]

@dataclass
class OssCiUtilizationMetadataV1():
    created_at: str
    repo: str
    workflow_id:int
    run_attempt: int
    job_id: int
    workflow_name: str
    job_name: str
    usage_collect_interval: float
    data_model_version: float
    gpu_count: int
    cpu_count: int
    gpu_type: str
    start_at: str
    end_at: str
    error: str
    segments: list[oss_ci_utilization_segment_v1]

@dataclass
class OssCiUtilizationTimeSeriesV1():
    created_at: str
    type: str
    tags: list[str]
    time_stamp: str
    repo: str
    workflow_id: int
    run_attempt: int
    job_id: int
    workflow_name: str
    job_name: str
    json_data: str
    _meta: tuple[str,str]
