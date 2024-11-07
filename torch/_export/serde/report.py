from typing import Any, Dict, List

import torch._export.serde.report_schema as schema
from torch._export.serde.serialize import SerializeError
from torch.export._draft_export import DraftExportReport, FailureReport, FailureType


def serialize_failure_type(failure_type: FailureType) -> schema.FailureType:
    return schema.FailureType(failure_type)


def deserialize_failure_type(failure_type: schema.FailureType) -> FailureType:
    return FailureType(failure_type)


def serialize_failure_data(failure: FailureReport) -> schema.FailureData:
    if failure.failure_type == FailureType.MISSING_FAKE_KERNEL:
        return schema.FailureData.create(
            for_missing_fake_kernel=schema.OperatorProfiles(
                op=failure.data["op"],
            )
        )
    elif failure.failure_type == FailureType.DATA_DEPENDENT_ERROR:
        return schema.FailureData.create(
            for_data_dependent_error=schema.DataDependentErrorInfo(
                expr=failure.data["expr"],
                result=failure.data["result"],
                stack=failure.data["stack"],
            )
        )
    elif failure.failure_type == FailureType.CONSTRAINT_VIOLATION_ERROR:
        return schema.FailureData.create(
            for_constraint_violation_error=schema.ConstraintViolationErrorInfo(
                expr=failure.data["expr"],
                symbol_to_sources=failure.data["symbol_to_sources"],
                stack=failure.data["stack"],
                # TODO: _dump_dynamic_shapes(failure.data["new_dynamic_shapes"])
            )
        )
    else:
        raise SerializeError(f"Serializing {failure} is not supported")


def deserialize_failure_data(failure: schema.FailureData) -> Dict[str, Any]:
    typ_ = failure.type
    if typ_ == "for_missing_fake_kernel":
        operator_profiles: schema.OperatorProfiles = failure.for_missing_fake_kernel
        assert isinstance(operator_profiles, schema.OperatorProfiles)
        return {
            "op": operator_profiles.op,
        }
    elif typ_ == "for_data_dependent_error":
        serialized_data_dep_error: schema.DataDependentErrorInfo = (
            failure.for_data_dependent_error
        )
        assert isinstance(serialized_data_dep_error, schema.DataDependentErrorInfo)
        return {
            "expr": serialized_data_dep_error.expr,
            "result": serialized_data_dep_error.result,
            "stack": serialized_data_dep_error.stack,
        }
    elif typ_ == "for_constraint_violation_error":
        serialized_cv_error: schema.ConstraintViolationErrorInfo = (
            failure.for_constraint_violation_error
        )
        assert isinstance(serialized_cv_error, schema.ConstraintViolationErrorInfo)
        return {
            "expr": serialized_cv_error.expr,
            "symbol_to_sources": serialized_cv_error.symbol_to_sources,
            "stack": serialized_cv_error.stack,
            "new_dynamic_shapes": None,  # TODO
        }
    else:
        raise SerializeError(f"Unhandled argument {failure}")


def serialize_report(report: DraftExportReport) -> schema.DraftExportReport:
    serialized_failures = []

    for failure in report.failures:
        serialized_failure = schema.FailureReport(
            failure_type=serialize_failure_type(failure.failure_type),
            data=serialize_failure_data(failure),
            xfail=failure.xfail,
        )
        serialized_failures.append(serialized_failure)

    return schema.DraftExportReport(
        failures=serialized_failures,
        str_to_filename=report.str_to_filename,
    )


def deserialize_report(
    serialized_report: schema.DraftExportReport,
) -> DraftExportReport:
    failures: List[FailureReport] = []
    for serialized_failure in serialized_report.failures:
        failures.append(
            FailureReport(
                failure_type=deserialize_failure_type(serialized_failure.failure_type),
                data=deserialize_failure_data(serialized_failure.data),
                xfail=serialized_failure.xfail,
            )
        )
    return DraftExportReport(failures, serialized_report.str_to_filename)
