import getpass
import json
import logging
import os
import re
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Optional, Union

import torch
import torch._logging._internal
import torch.utils._pytree as pytree
from torch._dynamo.exc import UserError, UserErrorType
from torch._export.passes.insert_custom_op_guards import (
    get_op_profiles,
    insert_custom_op_guards,
    OpProfile,
)
from torch._utils_internal import log_draft_export_usage

from ._trace import _export, get_ep_stats
from .dynamic_shapes import _DimHint, _DimHintType, Dim
from .exported_program import ExportedProgram


log = logging.getLogger(__name__)


class FailureType(IntEnum):
    MISSING_FAKE_KERNEL = 1
    DATA_DEPENDENT_ERROR = 2
    GUARD_ADDED = 3
    MISMATCHED_FAKE_KERNEL = 4

    def __str__(self) -> str:
        return self.name


def prettify_stack(stack: list[dict[str, str]], str_to_filename: dict[int, str]) -> str:
    res = ""
    for frame in stack:
        if frame["filename"] not in str_to_filename:
            continue

        res += f"""
        File {str_to_filename[frame["filename"]]}, lineno {frame["line"]}, in {frame["name"]}"""  # type: ignore[index]

    res += f"\n            {stack[-1]['loc']}"
    return res


def prettify_frame_locals(
    loc: str, locals: dict[str, Any], symbols: dict[str, Any]
) -> str:
    local_str = "\n".join(f"            {k}: {v}" for k, v in locals.items())
    res = f"""
        Locals:
{local_str}
"""
    if any(v is not None for v in symbols.values()):
        symbol_str = "\n".join(
            f"           {k}: {v}" for k, v in symbols.items() if v is not None
        )
        res += f"""
        Symbols:
{symbol_str}
"""
    return res


def get_loc(filename: str, lineno: int) -> Optional[str]:
    try:
        with open(filename) as f:
            for i, line in enumerate(f):
                if i == lineno - 1:
                    return line.strip()
    except FileNotFoundError:
        pass
    return None


class FailureReport:
    def __init__(
        self, failure_type: FailureType, data: dict[str, Any], xfail: bool = False
    ) -> None:
        self.failure_type: FailureType = failure_type
        self.data: dict[str, Any] = data
        self.xfail: bool = xfail

    def __repr__(self) -> str:
        return f"FailureReport(failure_type={self.failure_type}, xfail={self.xfail}, data={self.data})"

    def print(self, str_to_filename: dict[int, str]) -> str:
        if self.failure_type == FailureType.MISSING_FAKE_KERNEL:
            op = self.data["op"]

            return f"""Missing fake kernel.
    torch.ops.{op} is missing a fake kernel implementation.

    Please refer to https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ahugy69p2jmz for more detailed instructions on how to write a meta implementation.
"""  # noqa: B950

        elif self.failure_type == FailureType.GUARD_ADDED:
            locals_info = (
                prettify_frame_locals(**self.data["frame_locals"])
                if self.data["frame_locals"]
                else ""
            )
            return f"""Guard Added.
    A guard was added during tracing, which might've resulted in some incorrect
    tracing or constraint violation error.
    Specifically, this guard was added: {self.data["expr"]}, where {self.data["symbol_to_sources"]}.
    This occurred at the following stacktrace: {prettify_stack(self.data["user_stack"], str_to_filename)}:
        {locals_info}
    And the following framework stacktrace: {prettify_stack(self.data["stack"], str_to_filename)}\n
    Because of this, we have modified the dynamic shapes structure to be the
    following. You can also use torch.export.Dim.AUTO instead to specify your
    dynamic shapes, and we will automatically infer the dynamism for you.
    ```
    dynamic_shapes = {self.data["new_dynamic_shapes"]}
    ```
"""

        elif self.failure_type == FailureType.DATA_DEPENDENT_ERROR:
            locals_info = (
                prettify_frame_locals(**self.data["frame_locals"])
                if self.data["frame_locals"]
                else ""
            )
            return f"""Data dependent error.
    When exporting, we were unable to evaluate the value of `{self.data["expr"]}`.
    This was encountered {self.data["occurrences"]} times.
    This occurred at the following user stacktrace: {prettify_stack(self.data["user_stack"], str_to_filename)}
        {locals_info}
    And the following framework stacktrace: {prettify_stack(self.data["stack"], str_to_filename)}\n
    As a result, it was specialized to a constant (e.g. `{self.data["result"]}` in the 1st occurrence), and asserts were inserted into the graph.

    Please add `torch._check(...)` to the original code to assert this data-dependent assumption.
    Please refer to https://docs.google.com/document/d/1kZ_BbB3JnoLbUZleDT6635dHs88ZVYId8jT-yTFgf3A/edit#heading=h.boi2xurpqa0o for more details.
"""  # noqa: B950

        elif self.failure_type == FailureType.MISMATCHED_FAKE_KERNEL:
            op = self.data["op"]
            reason = self.data["reason"]
            return f"""Mismatched fake kernel.
    torch.ops.{op} has a fake kernel implementation, but it has incorrect behavior, based on the real kernel.
    The reason for the mismatch is: {reason}.

    Please refer to https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ahugy69p2jmz for more detailed instructions on how to write a fake implementation.
"""  # noqa: B950

        else:
            raise ValueError(f"Unknown failure type: {self.failure_type}")


class DraftExportReport:
    def __init__(
        self,
        failures: list[FailureReport],
        str_to_filename: dict[int, str],
        expressions_created: dict[int, dict[str, Any]],
        op_profiles: dict[str, set[OpProfile]],
    ):
        self.failures: list[FailureReport] = failures
        self.str_to_filename = str_to_filename
        self.expressions_created: dict[int, dict[str, Any]] = expressions_created
        self.op_profiles = op_profiles

    def successful(self) -> bool:
        return len(self.failures) == 0 or all(
            failure.xfail for failure in self.failures
        )

    def __repr__(self) -> str:
        return f"DraftExportReport({self.failures})"

    def __str__(self) -> str:
        WARNING_COLOR = "\033[93m"
        GREEN_COLOR = "\033[92m"
        END_COLOR = "\033[0m"

        if self.successful():
            return f"""{GREEN_COLOR}
##############################################################################################
Congratuations: No issues are found during export, and it was able to soundly produce a graph.
You can now change back to torch.export.export()
##############################################################################################
{END_COLOR}"""

        error = f"""{WARNING_COLOR}
###################################################################################################
WARNING: {len(self.failures)} issue(s) found during export, and it was not able to soundly produce a graph.
Please follow the instructions to fix the errors.
###################################################################################################

"""

        for i, failure in enumerate(self.failures):
            error += f"{i + 1}. {failure.print(self.str_to_filename)}\n"
        error += END_COLOR
        return error

    def apply_suggested_fixes(self) -> None:
        raise NotImplementedError("Not implemented yet")


@dataclass
class ExpressionCreatedNode:
    result_id: int
    argument_ids: list[int]
    record: dict[str, object]
    visited: bool = False


class LogRecord:
    def __init__(self) -> None:
        self.log_count: dict[int, int] = {}
        self.logs: list[tuple[str, dict[str, Any]]] = []

    def _hash(self, element: tuple[str, dict[str, Any]]) -> int:
        key, data = element

        if key == "missing_fake_kernel":
            return hash((key, data["op"]))
        elif key == "mismatched_fake_kernel":
            return hash((key, data["op"], data["reason"]))
        elif key == "propagate_real_tensors_provenance":
            return hash((key, json.dumps(data["user_stack"])))
        elif key == "guard_added":
            return hash((key, json.dumps(data["user_stack"])))
        elif key == "create_unbacked_symbol":
            return hash((key, json.dumps(data["user_stack"])))

        return hash((key, json.dumps(data)))

    def try_add(self, element: tuple[str, dict[str, str]]) -> bool:
        hash_value = self._hash(element)
        if hash_value in self.log_count:
            self.log_count[hash_value] += 1
            return False

        self.log_count[hash_value] = 1
        self.logs.append(element)
        return True

    def get_log_count(self, element: tuple[str, dict[str, Any]]) -> int:
        return self.log_count[self._hash(element)]


class CaptureStructuredTrace(torch._logging._internal.LazyTraceHandler):
    def __init__(self) -> None:
        self.specific_log_keys = [
            "str",
            "exported_program",
            "propagate_real_tensors_provenance",
            "guard_added",
            "missing_fake_kernel",
            "mismatched_fake_kernel",
            "expression_created",
            "create_unbacked_symbol",
        ]
        self.log_record: LogRecord = LogRecord()
        self.expression_created_logs: dict[int, ExpressionCreatedNode] = {}
        self.symbol_to_expressions: dict[str, list[dict[str, Any]]] = {}
        self.logger = logging.getLogger("torch.__trace")
        self.prev_get_dtrace = False

        if root_dir := os.environ.get(torch._logging._internal.DTRACE_ENV_VAR):
            super().__init__(root_dir)
        else:
            sanitized_username = re.sub(r'[\\/:*?"<>|]', "_", getpass.getuser())
            root_dir = os.path.join(
                tempfile.gettempdir(),
                "export_" + sanitized_username,
            )
            super().__init__(root_dir)

        self.setFormatter(torch._logging._internal.TorchLogsFormatter(trace=True))

    def __enter__(self) -> "CaptureStructuredTrace":
        self.log_record = LogRecord()
        self.expression_created_logs = {}

        # Remove the lazy trace handler if it exists
        possible_lazy_trace_handlers = [
            handler
            for handler in self.logger.handlers
            if isinstance(handler, torch._logging._internal.LazyTraceHandler)
        ]
        for handler in possible_lazy_trace_handlers:
            self.logger.removeHandler(handler)

        self.logger.addHandler(self)
        self.prev_get_dtrace = torch._logging._internal.GET_DTRACE_STRUCTURED
        # pyrefly: ignore [bad-assignment]
        torch._logging._internal.GET_DTRACE_STRUCTURED = True
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # type: ignore[no-untyped-def]
        self.log_record = LogRecord()
        self.expression_created_logs = {}
        self.logger.removeHandler(self)
        # pyrefly: ignore [bad-assignment]
        torch._logging._internal.GET_DTRACE_STRUCTURED = self.prev_get_dtrace
        self.prev_get_dtrace = False

    def emit(self, record: Any) -> None:
        def _log_expression_created(
            emit_func: Callable[[Any], None], sym_node_id: int
        ) -> None:
            # Log all the relevant expression_created logs
            if sym_node_id is None:
                return
            if res := self.expression_created_logs.get(sym_node_id, None):
                # Don't log the expression if we have already
                # printed it beforehand
                if not res.visited:
                    res.visited = True
                    for arg in res.argument_ids:
                        _log_expression_created(emit_func, arg)

                emit_func(res.record)

        metadata = record.metadata
        for key in self.specific_log_keys:
            if key in metadata:
                if self.log_record.try_add((key, metadata[key])):
                    if key == "expression_created":
                        # We don't want to log all expression_created logs, only
                        # the ones that are relevant to the
                        # guards/propagate_real_tensor
                        self.expression_created_logs[metadata[key]["result_id"]] = (
                            ExpressionCreatedNode(
                                metadata[key]["result_id"],
                                metadata[key].get("argument_ids", []),
                                record,
                            )
                        )
                        return

                    elif key == "propagate_real_tensors_provenance":
                        _log_expression_created(
                            super().emit, metadata[key].get("expr_node_id")
                        )

                    elif key == "guard_added":
                        if len(metadata[key]["symbol_to_sources"]) == 0:
                            # We only want to include guards added that are relevant to
                            # the symbolic shapes corresponding to the inputs which were
                            # specified in the dynamic_shapes arg. These have a source.
                            return
                        elif metadata[key]["prefix"] == "runtime_assert":
                            # This should've been captured by a
                            # propagate_real_tensors log
                            return

                        _log_expression_created(
                            super().emit, metadata[key].get("expr_node_id")
                        )

                    super().emit(record)


def draft_export(
    mod: torch.nn.Module,
    args: tuple[Any, ...],
    kwargs: Optional[Mapping[str, Any]] = None,
    *,
    dynamic_shapes: Optional[Union[dict[str, Any], tuple[Any], list[Any]]] = None,
    preserve_module_call_signature: tuple[str, ...] = (),
    strict: bool = False,
    pre_dispatch: bool = True,
    prefer_deferred_runtime_asserts_over_guards: bool = False,
) -> ExportedProgram:
    start_time = time.time()
    kwargs = kwargs or {}
    dynamic_shapes = dynamic_shapes or {}

    constraint_violation_msg = None
    capture_structured_log = CaptureStructuredTrace()

    with (
        torch._functorch.config.patch(
            fake_tensor_propagate_real_tensors=True,
            generate_fake_kernels_from_real_mismatches=True,
        ),
        capture_structured_log,
    ):
        try:
            new_shapes = None
            ep = _export(
                mod,
                args,
                kwargs,
                dynamic_shapes=dynamic_shapes,
                strict=strict,
                pre_dispatch=pre_dispatch,
                preserve_module_call_signature=preserve_module_call_signature,
                prefer_deferred_runtime_asserts_over_guards=prefer_deferred_runtime_asserts_over_guards,
            )
        except Exception as exc:
            if (
                isinstance(exc, UserError)
                and exc.error_type == UserErrorType.CONSTRAINT_VIOLATION
            ):
                constraint_violation_msg = exc.msg

                def convert_dim_to_auto(dim: Any) -> Any:
                    if isinstance(dim, Dim):
                        return Dim.AUTO(min=dim.min, max=dim.max)
                    elif isinstance(dim, _DimHint) and dim.type == _DimHintType.DYNAMIC:
                        return Dim.AUTO(min=dim.min, max=dim.max)
                    return dim

                new_shapes = pytree.tree_map(convert_dim_to_auto, dynamic_shapes)
                ep = _export(
                    mod,
                    args,
                    kwargs,
                    dynamic_shapes=new_shapes,
                    strict=strict,
                    pre_dispatch=pre_dispatch,
                    preserve_module_call_signature=preserve_module_call_signature,
                    prefer_deferred_runtime_asserts_over_guards=prefer_deferred_runtime_asserts_over_guards,
                )
            else:
                log_draft_export_usage(
                    error=True,
                    export_time=time.time() - start_time,
                    strict=strict,
                    message=str(exc),
                    type=f"{type(exc).__name__}.{type(exc).__qualname__}",
                )
                raise exc

        torch._logging.dtrace_structured("exported_program", payload_fn=lambda: str(ep))

        str_to_filename: dict[int, str] = {}
        failures: list[FailureReport] = []
        incorrect_custom_ops: set[str] = set()
        expressions_created: dict[int, dict[str, Any]] = {}

        for log_name, log_contents in capture_structured_log.log_record.logs:
            failure_type = None

            if log_name == "str":
                str_to_filename[log_contents[1]] = log_contents[0]  # type: ignore[index]
                continue

            elif log_name == "propagate_real_tensors_provenance":
                log_contents["occurrences"] = (
                    capture_structured_log.log_record.get_log_count(
                        (log_name, log_contents)
                    )
                )

                failure_type = FailureType.DATA_DEPENDENT_ERROR

            elif log_name == "guard_added":
                if new_shapes is None:
                    continue

                failure_type = FailureType.GUARD_ADDED
                log_contents["new_dynamic_shapes"] = new_shapes
            elif log_name == "missing_fake_kernel":
                failure_type = FailureType.MISSING_FAKE_KERNEL
                incorrect_custom_ops.add(log_contents["op"])

            elif log_name == "mismatched_fake_kernel":
                failure_type = FailureType.MISMATCHED_FAKE_KERNEL
                incorrect_custom_ops.add(log_contents["op"])

            else:
                continue

            assert failure_type is not None
            failures.append(
                FailureReport(
                    failure_type,
                    log_contents,
                )
            )

        for k, v in capture_structured_log.expression_created_logs.items():
            if v.visited:
                expressions_created[k] = v.record

        op_profiles = get_op_profiles(ep.graph_module, incorrect_custom_ops)
        report = DraftExportReport(
            failures, str_to_filename, expressions_created, op_profiles
        )

        # Add asserts around custom ops
        insert_custom_op_guards(ep.graph_module, incorrect_custom_ops)

    ep._report = report
    if not report.successful():
        log_filename = capture_structured_log.stream.name

        warning_msg = f"""
###################################################################################################
WARNING: {len(report.failures)} issue(s) found during export, and it was not able to soundly produce a graph.
To view the report of failures in an html page, please run the command:
    `tlparse {log_filename} --export`
Or, you can view the errors in python by inspecting `print(ep._report)`.
"""

        if len(report.op_profiles) > 0:
            warning_msg += f"""
While tracing we found {len(report.op_profiles)} operator(s) which do not have a fake kernel registered.
If you intend to retrace the exported graph or run it with fake tensors, please run it under the
following context manager, which will register a fake kernel for those operators.
```
with torch._library.fake_profile.unsafe_generate_fake_kernels(ep._report.op_profiles):
    # run with fake tensors
```
"""

        warning_msg += """#################################################################################################"""

        log.warning(warning_msg)

    else:
        log.info(
            """
##############################################################################################
Congratuations: No issues are found during export, and it was able to soundly produce a graph.
You can now change back to torch.export.export()
##############################################################################################
    """
        )

    log_draft_export_usage(
        error=False,
        export_time=time.time() - start_time,
        strict=strict,
        constraint_violations=constraint_violation_msg,
        report=ep._report,
        **get_ep_stats(ep),
    )
    return ep
