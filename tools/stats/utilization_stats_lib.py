from dataclasses import dataclass
from typing import Optional

from dataclasses_json import DataClassJsonMixin


_DATA_MODEL_VERSION = 1.0


# data model for test log usage
@dataclass
class UtilizationStats:
    avg: Optional[float] = None
    max: Optional[float] = None


@dataclass
class UtilizationMetadata(DataClassJsonMixin):
    level: str
    workflow_id: str
    job_id: str
    workflow_name: str
    job_name: str
    usage_collect_interval: float
    data_model_version: float
    start_at: float
    gpu_count: Optional[int] = None
    cpu_count: Optional[int] = None
    gpu_type: Optional[str] = None
    error: Optional[str] = None


@dataclass
class GpuUsage(DataClassJsonMixin):
    uuid: Optional[str] = None
    util_percent: Optional[UtilizationStats] = None
    mem_util_percent: Optional[UtilizationStats] = None


@dataclass
class RecordData(DataClassJsonMixin):
    cpu: Optional[UtilizationStats] = None
    memory: Optional[UtilizationStats] = None
    gpu_usage: Optional[list[GpuUsage]] = None


@dataclass
class UtilizationRecord(DataClassJsonMixin):
    level: str
    timestamp: float
    data: Optional[RecordData] = None
    cmd_names: Optional[list[str]] = None
    error: Optional[str] = None
    log_duration: Optional[str] = None


@dataclass
class OssCiSegmentV1(DataClassJsonMixin):
    level: str
    name: str
    start_at: float
    end_at: float
    extra_info: dict[str, str]


# the db schema related to this is:
# https://github.com/pytorch/test-infra/blob/main/clickhouse_db_schema/oss_ci_utilization/oss_ci_utilization_metadata_schema.sql
@dataclass
class OssCiUtilizationMetadataV1:
    created_at: float
    repo: str
    workflow_id: int
    run_attempt: int
    job_id: int
    workflow_name: str
    job_name: str
    usage_collect_interval: float
    data_model_version: str
    gpu_count: int
    cpu_count: int
    gpu_type: str
    start_at: float
    end_at: float
    segments: list[OssCiSegmentV1]
    tags: list[str] = []


# this data model is for the time series data:
# https://github.com/pytorch/test-infra/blob/main/clickhouse_db_schema/oss_ci_utilization/oss_ci_utilization_time_series_schema.sql
@dataclass
class OssCiUtilizationTimeSeriesV1:
    created_at: float
    type: str
    tags: list[str]
    time_stamp: float
    repo: str
    workflow_id: int
    run_attempt: int
    job_id: int
    workflow_name: str
    job_name: str
    json_data: str


def getDataModelVersion() -> float:
    return _DATA_MODEL_VERSION


@dataclass
class WorkflowInfo:
    workflow_run_id: int
    workflow_name: str
    job_id: int
    run_attempt: int
    job_name: str
    repo: str = "pytorch/pytorch"
