from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional

from torch._export.serde.dynamic_shapes import DynamicShapesSpec
from torch._export.serde.union import _Union


class FailureType(IntEnum):
    MISSING_FAKE_KERNEL = 1
    DATA_DEPENDENT_ERROR = 2
    CONSTRAINT_VIOLATION_ERROR = 3


@dataclass
class OperatorProfiles:
    op: str


@dataclass
class DataDependentErrorInfo:
    expr: str
    result: str
    stack: str


@dataclass
class ConstraintViolationErrorInfo:
    expr: str
    symbol_to_sources: Dict[str, str]
    stack: str
    new_dynamic_shapes: Optional[DynamicShapesSpec] = None


@dataclass(repr=False)
class FailureData(_Union):
    for_missing_fake_kernel: OperatorProfiles
    for_data_dependent_error: DataDependentErrorInfo
    for_constraint_violation_error: ConstraintViolationErrorInfo


@dataclass
class FailureReport:
    failure_type: FailureType
    data: FailureData
    xfail: bool


@dataclass
class DraftExportReport:
    failures: List[FailureReport]
    str_to_filename: Dict[str, str]
