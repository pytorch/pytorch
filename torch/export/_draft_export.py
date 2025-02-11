import getpass
import inspect
import json
import logging
import os
import re
import tempfile
from enum import IntEnum
from typing import Any, Optional, Union

import torch
import torch._logging._internal
import torch._logging.structured
from torch._export.passes.insert_custom_op_guards import insert_custom_op_guards
from torch.export import ExportedProgram
from torch.export._trace import _export
from torch.export.dynamic_shapes import refine_dynamic_shapes_from_suggested_fixes


log = logging.getLogger(__name__)


class FailureType(IntEnum):
    MISSING_FAKE_KERNEL = 1
    DATA_DEPENDENT_ERROR = 2
    CONSTRAINT_VIOLATION_ERROR = 3
    MISMATCHED_FAKE_KERNEL = 4

    def __str__(self) -> str:
        return self.name


def prettify_stack(stack: list[dict[str, str]], str_to_filename: dict[str, str]) -> str:
    res = ""
    for frame in stack:
        if frame["filename"] not in str_to_filename:
            continue

        res += f"""
        File {str_to_filename[frame['filename']]}, lineno {frame['line']}, in {frame['name']}"""
    return res


def prettify_frame_locals(
    loc: str, locals: dict[str, Any], symbols: dict[str, Any]
) -> str:
    res = f"    {loc}\n"
    local_str = "\n".join(f"            {k}: {v}" for k, v in locals.items())
    res += f"""
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


def filter_stack(
    stack: list[dict[str, str]], str_to_filename: dict[str, str]
) -> list[dict[str, str]]:
    for i, s in enumerate(reversed(stack)):
        s["filename"] = str(s["filename"])
        if s["filename"] not in str_to_filename:
            continue
        torch_filepath = os.path.dirname(inspect.getfile(torch)) + os.path.sep
        if torch_filepath not in str_to_filename[s["filename"]]:
            return stack[len(stack) - i - 3 : len(stack) - i]
    return stack[-3:]


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

    def print(self, str_to_filename: dict[str, str]) -> str:
        if self.failure_type == FailureType.MISSING_FAKE_KERNEL:
            op = self.data["op"]

            return f"""Missing fake kernel.
    torch.ops.{op} is missing a fake kernel implementation.

    Please refer to https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ahugy69p2jmz for more detailed instructions on how to write a meta implementation.
"""  # noqa: B950

        elif self.failure_type == FailureType.CONSTRAINT_VIOLATION_ERROR:
            locals_info = (
                prettify_frame_locals(**self.data["frame_locals"])
                if self.data["frame_locals"]
                else ""
            )
            return f"""Constraint violation error.
    The specified input dynamic_shapes spec was found to be incorrect during tracing.
    Specifically, this guard was added: {self.data["expr"]}, where {self.data["symbol_to_sources"]}.
    This occured at the following stacktrace: {prettify_stack(self.data["stack"], str_to_filename)}:
        {locals_info}
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
    This occurred at the following stacktrace: {prettify_stack(self.data["stack"], str_to_filename)}:
        {locals_info}
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
    def __init__(self, failures: list[FailureReport], str_to_filename: dict[str, str]):
        self.failures: list[FailureReport] = failures
        self.str_to_filename = str_to_filename

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
            return hash((key, json.dumps(data["stack"])))
        elif key == "create_unbacked_symbol":
            return hash((key, json.dumps(data["stack"])))

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
    def __init__(self, specific_log_keys: list[str]):
        self.specific_log_keys = specific_log_keys
        self.log_record: LogRecord = LogRecord()
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
        torch._logging._internal.GET_DTRACE_STRUCTURED = True
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # type: ignore[no-untyped-def]
        self.log_record = LogRecord()
        self.logger.removeHandler(self)
        torch._logging._internal.GET_DTRACE_STRUCTURED = self.prev_get_dtrace
        self.prev_get_dtrace = False

    def emit(self, record: Any) -> None:
        metadata = record.metadata
        for key in self.specific_log_keys:
            if key in metadata:
                if self.log_record.try_add((key, metadata[key])):
                    super().emit(record)


def draft_export(
    mod: torch.nn.Module,
    args: tuple[Any, ...],
    kwargs: Optional[dict[str, Any]] = None,
    *,
    dynamic_shapes: Optional[Union[dict[str, Any], tuple[Any], list[Any]]] = None,
    preserve_module_call_signature: tuple[str, ...] = (),
    strict: bool = False,
    pre_dispatch: bool = False,
) -> tuple[ExportedProgram, DraftExportReport]:
    kwargs = kwargs or {}
    dynamic_shapes = dynamic_shapes or {}

    capture_structured_log = CaptureStructuredTrace(
        [
            "propagate_real_tensors_provenance",
            "str",
            "exported_program",
            "guard_added",
            "missing_fake_kernel",
            "mismatched_fake_kernel",
            "expression_created",
            "create_unbacked_symbol",
        ]
    )

    with torch._functorch.config.patch(
        fake_tensor_propagate_real_tensors=True,
        generate_fake_kernels_from_real_mismatches=True,
    ), capture_structured_log:
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
            )
        except torch._dynamo.exc.UserError as exc:
            new_shapes = refine_dynamic_shapes_from_suggested_fixes(
                exc.msg, dynamic_shapes
            )
            ep = _export(
                mod,
                args,
                kwargs,
                dynamic_shapes=new_shapes,
                strict=strict,
                pre_dispatch=pre_dispatch,
                preserve_module_call_signature=preserve_module_call_signature,
            )

        torch._logging.dtrace_structured("exported_program", payload_fn=lambda: str(ep))

        str_to_filename: dict[str, str] = {
            str(v): k for (k, v) in torch._logging.structured.INTERN_TABLE.items()
        }
        failures: list[FailureReport] = []
        custom_ops_logs: dict[
            Any, tuple[dict[str, Any], FailureType]
        ] = {}  # For adding in assertions before custom ops

        for log_name, log_contents in capture_structured_log.log_record.logs:
            failure_type = None

            if log_name == "propagate_real_tensors_provenance":
                log_contents[
                    "occurrences"
                ] = capture_structured_log.log_record.get_log_count(
                    (log_name, log_contents)
                )
                log_contents["stack"] = filter_stack(
                    log_contents["stack"], str_to_filename
                )

                failure_type = FailureType.DATA_DEPENDENT_ERROR

            elif log_name == "guard_added":
                if new_shapes is None:
                    continue

                failure_type = FailureType.CONSTRAINT_VIOLATION_ERROR
                if len(log_contents["symbol_to_sources"]) == 0:
                    # We only want to include guards added that are relevant to
                    # the symbolic shapes corresponding to the inputs which were
                    # specified in the dynamic_shapes arg. These have a source.
                    continue

                log_contents["stack"] = filter_stack(
                    log_contents["stack"], str_to_filename
                )
                log_contents["new_dynamic_shapes"] = new_shapes
            elif log_name == "missing_fake_kernel":
                failure_type = FailureType.MISSING_FAKE_KERNEL
                custom_ops_logs[log_contents["op"]] = (log_contents, failure_type)
            elif log_name == "mismatched_fake_kernel":
                failure_type = FailureType.MISMATCHED_FAKE_KERNEL
                custom_ops_logs[(log_contents["op"], log_contents["reason"])] = (
                    log_contents,
                    failure_type,
                )
            else:
                continue

            assert failure_type is not None
            failures.append(
                FailureReport(
                    failure_type,
                    log_contents,
                )
            )

        report = DraftExportReport(failures, str_to_filename)

        # Add asserts around custom ops
        insert_custom_op_guards(ep.graph_module, list(custom_ops_logs.keys()))

    ep._report = report
    if not report.successful():
        log_filename = capture_structured_log.stream.name

        log.warning(
            """
###################################################################################################
WARNING: %s issue(s) found during export, and it was not able to soundly produce a graph.
To view the report of failures in an html page, please run the command:
    `tlparse %s --export`
Or, you can view the errors in python by inspecting `print(ep._report)`.
###################################################################################################
        """,
            len(report.failures),
            log_filename,
        )
    else:
        log.info(
            """
##############################################################################################
Congratuations: No issues are found during export, and it was able to soundly produce a graph.
You can now change back to torch.export.export()
##############################################################################################
    """
        )

    return ep, report
