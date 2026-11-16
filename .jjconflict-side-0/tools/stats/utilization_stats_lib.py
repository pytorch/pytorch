from dataclasses import dataclass, field
from datetime import datetime

#  pyrefly: ignore [missing-import]
from dataclasses_json import DataClassJsonMixin  # type: ignore[import-not-found]


_DATA_MODEL_VERSION = 1.5


# data model for test log usage
@dataclass
class UtilizationStats:
    avg: float | None = None
    max: float | None = None
    raw: list[float] | None = None


@dataclass
class UtilizationMetadata(DataClassJsonMixin):  # type: ignore[misc, no-any-unimported]
    level: str
    workflow_id: str
    job_id: str
    workflow_name: str
    job_name: str
    usage_collect_interval: float
    data_model_version: float
    start_at: int
    gpu_count: int | None = None
    cpu_count: int | None = None
    gpu_type: str | None = None
    error: str | None = None


@dataclass
class GpuUsage(DataClassJsonMixin):  # type: ignore[misc, no-any-unimported]
    uuid: str | None = None
    util_percent: UtilizationStats | None = None
    mem_util_percent: UtilizationStats | None = None
    allocated_mem_percent: UtilizationStats | None = None
    allocated_mem_value: UtilizationStats | None = None
    total_mem_value: float | None = None


@dataclass
class RecordData(DataClassJsonMixin):  # type: ignore[misc, no-any-unimported]
    cpu: UtilizationStats | None = None
    memory: UtilizationStats | None = None
    gpu_usage: list[GpuUsage] | None = None


@dataclass
class UtilizationRecord(DataClassJsonMixin):  # type: ignore[misc, no-any-unimported]
    level: str
    timestamp: int
    data: RecordData | None = None
    cmd_names: list[str] | None = None
    error: str | None = None
    log_duration: str | None = None
    logs: list[str] | None = None


# the db schema related to this is:
# https://github.com/pytorch/test-infra/blob/main/clickhouse_db_schema/oss_ci_utilization/oss_ci_utilization_metadata_schema.sql
@dataclass
class OssCiSegmentV1(DataClassJsonMixin):  # type: ignore[misc, no-any-unimported]
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
