from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from dataclasses_json import DataClassJsonMixin


_DATA_MODEL_VERSION = 1.5


# data model for test log usage
@dataclass
class UtilizationStats:
    avg: Optional[float] = None
    max: Optional[float] = None
    raw: Optional[list[float]] = None


@dataclass
class UtilizationMetadata(DataClassJsonMixin):
    level: str
    workflow_id: str
    job_id: str
    workflow_name: str
    job_name: str
    usage_collect_interval: float
    data_model_version: float
    start_at: int
    gpu_count: Optional[int] = None
    cpu_count: Optional[int] = None
    gpu_type: Optional[str] = None
    error: Optional[str] = None


@dataclass
class GpuUsage(DataClassJsonMixin):
    uuid: Optional[str] = None
    util_percent: Optional[UtilizationStats] = None
    mem_util_percent: Optional[UtilizationStats] = None
    allocated_mem_percent: Optional[UtilizationStats] = None
    allocated_mem_value: Optional[UtilizationStats] = None
    total_mem_value: Optional[float] = None


@dataclass
class RecordData(DataClassJsonMixin):
    cpu: Optional[UtilizationStats] = None
    memory: Optional[UtilizationStats] = None
    gpu_usage: Optional[list[GpuUsage]] = None


@dataclass
class UtilizationRecord(DataClassJsonMixin):
    level: str
    timestamp: int
    data: Optional[RecordData] = None
    cmd_names: Optional[list[str]] = None
    error: Optional[str] = None
    log_duration: Optional[str] = None
    logs: Optional[list[str]] = None


# the db schema related to this is:
# https://github.com/pytorch/test-infra/blob/main/clickhouse_db_schema/oss_ci_utilization/oss_ci_utilization_metadata_schema.sql
@dataclass
class OssCiSegmentV1(DataClassJsonMixin):
    level: str
    name: str
    start_at: int
    end_at: int
    extra_info: dict[str, str]


@dataclass
class OssCiUtilizationMetadataV1:
    created_at: int
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
    start_at: int
    end_at: int
    segments: list[OssCiSegmentV1]
    tags: list[str] = field(default_factory=list)


# this data model is for the time series data:
# https://github.com/pytorch/test-infra/blob/main/clickhouse_db_schema/oss_ci_utilization/oss_ci_time_series_schema.sql
@dataclass
class OssCiUtilizationTimeSeriesV1:
    created_at: int
    type: str
    tags: list[str]
    time_stamp: int
    repo: str
    workflow_id: int
    run_attempt: int
    job_id: int
    workflow_name: str
    job_name: str
    json_data: str


def getDataModelVersion() -> float:
    return _DATA_MODEL_VERSION


def getTsNow() -> int:
    ts = datetime.now().timestamp()
    return int(ts)


@dataclass
class WorkflowInfo:
    workflow_run_id: int
    workflow_name: str
    job_id: int
    run_attempt: int
    job_name: str
    repo: str = "pytorch/pytorch"
