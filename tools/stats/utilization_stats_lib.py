from dataclasses import dataclass
from typing import Optional
from dataclasses_json import dataclass_json

_DATA_MODEL_VERSION = 1.0


# data model for test log usage
@dataclass
class UtilizationStats:
    avg: Optional[float] = None
    max: Optional[float] = None

@dataclass_json
@dataclass
class UtilizationMetadata:
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

@dataclass_json
@dataclass
class GpuUsage:
    uuid: Optional[str] = None
    util_percent: Optional[UtilizationStats] = None
    mem_util_percent: Optional[UtilizationStats] = None

@dataclass_json
@dataclass
class RecordData:
    cpu: Optional[UtilizationStats] = None
    memory: Optional[UtilizationStats] = None
    gpu_usage: Optional[list[GpuUsage]] = None

@dataclass_json
@dataclass
class UtilizationRecord:
    level: Optional[str] = None
    timestamp: Optional[float] = None
    data: Optional[RecordData] = None
    cmd_names: Optional[list[str]] = None
    error: Optional[str] = None
    log_duration: Optional[str] = None


def getDataModelVersion() -> float:
    return _DATA_MODEL_VERSION
