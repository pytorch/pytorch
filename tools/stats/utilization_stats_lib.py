from dataclasses import dataclass
from typing import Optional

from dataclasses_json import DataClassJsonMixin


_DATA_MODEL_VERSION = 1.0


# the db schema related to this is:
# https://github.com/pytorch/test-infra/blob/main/clickhouse_db_schema/oss_ci_utilization/oss_ci_utilization_metadata_schema.sql
# data model for test log usage
@dataclass
class UtilizationStats:
    avg: Optional[float] = None
    max: Optional[float] = None


@dataclass
class UtilizationMetadata(DataClassJsonMixin):
    level: Optional[str] = None
    workflow_id: Optional[str] = None
    job_id: Optional[str] = None
    workflow_name: Optional[str] = None
    job_name: Optional[str] = None
    usage_collect_interval: Optional[float] = None
    data_model_version: Optional[float] = None
    gpu_count: Optional[int] = None
    cpu_count: Optional[int] = None
    gpu_type: Optional[str] = None
    start_at: Optional[float] = None
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
    level: Optional[str] = None
    timestamp: Optional[float] = None
    data: Optional[RecordData] = None
    cmd_names: Optional[list[str]] = None
    error: Optional[str] = None
    log_duration: Optional[str] = None


def getDataModelVersion() -> float:
    return _DATA_MODEL_VERSION
