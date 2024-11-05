import inspect
import logging
import sys
from enum import IntEnum
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch._logging._internal
from torch.export import ExportedProgram
from torch.export._trace import _export
from torch.export.dynamic_shapes import refine_dynamic_shapes_from_suggested_fixes


log = logging.getLogger(__name__)


class FailureType(IntEnum):
    MISSING_FAKE_KERNEL = 1
    DATA_DEPENDENT_ERROR = 2
    CONSTRAINT_VIOLATION_ERROR = 3

    def __str__(self) -> str:
        return self.name


@lru_cache
def uninteresting_files() -> Set[str]:
    import torch._inductor.sizevars
    import torch._subclasses.fake_tensor
    import torch._subclasses.meta_utils

    mods = [
        sys.modules[__name__],
        torch.fx.experimental.recording,
        torch.fx.experimental.sym_node,
        torch.fx.experimental.symbolic_shapes,
        torch.fx.interpreter,
        torch,
        torch._inductor.sizevars,
        torch._logging._internal,
        torch._subclasses.meta_utils,
        torch._subclasses.fake_tensor,
        torch._subclasses.functional_tensor,
    ]
    return {inspect.getfile(m) for m in mods}


def prettify_stack(
    stack: List[Dict["str", "str"]], str_to_filename: Dict[str, str]
) -> str:
    res = ""
    for frame in stack:
        if frame["filename"] not in str_to_filename:
            continue

        res += f"""
        File {str_to_filename[frame['filename']]}, lineno {frame['line']}, in {frame['name']}"""
    return res


def filter_stack(
    stack: List[Dict[str, str]], str_to_filename: Dict[str, str]
) -> List[Dict[str, str]]:
    for i, s in enumerate(reversed(stack)):
        s["filename"] = str(s["filename"])
        if s["filename"] not in str_to_filename:
            continue
        if str_to_filename[s["filename"]] not in uninteresting_files():
            return stack[len(stack) - i - 3 : len(stack) - i]
    return stack[-3:]


def hash_stack(stack: List[Dict[str, str]]) -> str:
    return ";".join(f'line: {s["line"]} filename: {s["filename"]}' for s in stack)


class FailureReport:
    def __init__(
        self, failure_type: FailureType, data: Dict[str, Any], xfail: bool = False
    ) -> None:
        self.failure_type: FailureType = failure_type
        self.data: Dict[str, Any] = data
        self.xfail: bool = xfail

    def __repr__(self) -> str:
        return f"FailureReport(failure_type={self.failure_type}, xfail={self.xfail}, data={self.data})"

    def print(self, str_to_filename: Dict[str, str]) -> str:
        if self.failure_type == FailureType.MISSING_FAKE_KERNEL:
            op = self.data["op"]

            return f"""Missing fake kernel.
    torch.ops.{op} is missing a fake kernel implementation.

    Please refer to https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ahugy69p2jmz for more detailed instructions on how to write a meta implementation.
"""  # noqa: B950

        elif self.failure_type == FailureType.CONSTRAINT_VIOLATION_ERROR:
            return f"""Constraint violation error.
    The specified input dynamic_shapes spec was found to be incorrect during tracing.
    Specifically, this guard was added: {self.data["expr"]}, where {self.data["symbol_to_sources"]}.
    This occured at the following stacktrace: {prettify_stack(self.data["stack"], str_to_filename)}.
    Because of this, we have modified the dynamic shapes structure to be the following:
    ```
    dynamic_shapes = {self.data["new_dynamic_shapes"]}
    ```
"""

        elif self.failure_type == FailureType.DATA_DEPENDENT_ERROR:
            return f"""Data dependent error.
    When exporting, we were unable to figure out if the expression `{self.data["expr"]}` always holds.
    This occurred at the following stacktrace: {prettify_stack(self.data["stack"], str_to_filename)}.
    As a result, it was specialized to evaluate to `{self.data["result"]}`, and asserts were inserted into the graph.

    Please add `torch._check(...)` to the original code to assert this data-dependent assumption.
    Please refer to https://docs.google.com/document/d/1kZ_BbB3JnoLbUZleDT6635dHs88ZVYId8jT-yTFgf3A/edit#heading=h.boi2xurpqa0o for more details.
"""  # noqa: B950

        else:
            raise ValueError(f"Unknown failure type: {self.failure_type}")


class DraftExportReport:
    def __init__(self, failures: List[FailureReport], str_to_filename: Dict[str, str]):
        self.failures: List[FailureReport] = failures
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


class CaptureStructuredTrace(logging.Handler):
    def __init__(self, specific_log_keys: List[str]):
        super().__init__()
        self.specific_log_keys = specific_log_keys
        self.logs: List[Tuple[str, Dict[str, Any]]] = []
        self.logger = logging.getLogger("torch.__trace")
        self.prev_get_dtrace = False

    def __enter__(self) -> "CaptureStructuredTrace":
        self.logs = []
        self.logger.addHandler(self)
        self.prev_get_dtrace = torch._logging._internal.GET_DTRACE_STRUCTURED
        torch._logging._internal.GET_DTRACE_STRUCTURED = True
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # type: ignore[no-untyped-def]
        self.logs = []
        self.logger.removeHandler(self)
        torch._logging._internal.GET_DTRACE_STRUCTURED = self.prev_get_dtrace
        self.prev_get_dtrace = False

    def emit(self, record: Any) -> None:
        metadata = record.metadata
        for key in self.specific_log_keys:
            if key in metadata:
                self.logs.append((key, metadata[key]))


def draft_export(
    mod: torch.nn.Module,
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
    *,
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any], List[Any]]] = None,
    preserve_module_call_signature: Tuple[str, ...] = (),
    strict: bool = False,
    pre_dispatch: bool = False,
) -> Tuple[ExportedProgram, DraftExportReport]:
    kwargs = kwargs or {}
    dynamic_shapes = dynamic_shapes or {}

    capture_structured_log = CaptureStructuredTrace(
        ["str", "propagate_real_tensors", "guard_added", "generated_fake_kernel"]
    )

    with torch._functorch.config.patch(
        fake_tensor_propagate_real_tensors=True
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

        str_to_filename: Dict[str, str] = {}
        failures: List[FailureReport] = []
        custom_ops_logs: Dict[str, Dict[str, Any]] = {}  # Dedup custom ops
        data_dependent_logs: Dict[
            str, Dict[str, Any]
        ] = {}  # Dedup data dependent errors based on stacktrace

        for log_name, log_contents in capture_structured_log.logs:
            failure_type = None

            if log_name == "propagate_real_tensors":
                log_contents["stack"] = filter_stack(
                    log_contents["stack"], str_to_filename
                )
                if hash_stack(log_contents["stack"]) in data_dependent_logs:
                    continue

                data_dependent_logs[hash_stack(log_contents["stack"])] = log_contents
                failure_type = FailureType.DATA_DEPENDENT_ERROR

            elif log_name == "str":
                filename, idx = log_contents
                str_to_filename[str(idx)] = filename
                continue

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

            elif log_name == "generated_fake_kernel":
                if log_contents["op"] in custom_ops_logs:
                    continue

                failure_type = FailureType.MISSING_FAKE_KERNEL
                custom_ops_logs[log_contents["op"]] = log_contents

            else:
                raise RuntimeError(f"Unknown log name: {log_name}")

            assert failure_type is not None
            failures.append(
                FailureReport(
                    failure_type,
                    log_contents,
                )
            )

        report = DraftExportReport(failures, str_to_filename)

    ep._report = report
    if not report.successful():
        log.warning(report)
    return ep, report
