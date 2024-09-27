from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from op import OperatorProfilingMode

import torch
from torch.export import ExportedProgram
from torch.export.dynamic_shapes import refine_dynamic_shapes_from_suggested_fixes
from torch.fx.experimental.symbolic_shapes import RealTensorLoggingMode


class FailureType(Enum):
    MISSING_FAKE_KERNEL = 1
    DATA_DEPENDENT_ERROR = 2
    INPUT_SHAPE_MISMATCH = 3

    def __str__(self):
        return self.name


class FailureReport:
    def __init__(self, failure_type, data, xfail=False):
        self.failure_type = failure_type
        self.data = data
        self.xfail = xfail

    def __str__(self):
        if self.failure_type == FailureType.MISSING_FAKE_KERNEL:
            op, profiles = self.data
            example_output_metadata = profiles[0][-1]

            return f"""Missing fake kernel.
    torch.ops.{op} is missing a fake kernel implementation.

    Here's a template for registering a fake kernel implementation. Please refer to https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ahugy69p2jmz for more detailed instructions.

    Here is an example of a fake kernel for your op, but it might not be correct for all use cases:
    ```
    @torch.library.register_fake("{op.name()}")
    def {str(op).replace(".", "_")}_fake(*args, **kwargs):
        ctx = torch.library.get_ctx()
        fake_shape = [ctx.new_dynamic_size() for _ in range(2)]
        return torch.empty(fake_shape, dtype={example_output_metadata.dtype}, device="{example_output_metadata.device}")
    ```
"""

        elif self.failure_type == FailureType.INPUT_SHAPE_MISMATCH:
            return f"""Input shape mismatch.
    The specified input dynamic_shapes spec was found to be incorrect during tracing.
    Instead, we have modified the dynamic shapes structure to be the following:
    ```
    dynamic_shapes = {self.data}
    ```
"""

        elif self.failure_type == FailureType.DATA_DEPENDENT_ERROR:
            return f"""Data dependent error.
    When exporting, we were unable to figure out if the expression `{self.data["expr"]}` always holds.
    This occurred on the following line: {self.data["stack"]}.
    As a result, it was specialized to evaluate to `{self.data["result"]}`, and asserts were inserted into the graph.

    Please add `torch._check(...)` to the original code to assert this data-dependent assumption.
    Please refer to https://docs.google.com/document/d/1kZ_BbB3JnoLbUZleDT6635dHs88ZVYId8jT-yTFgf3A/edit#heading=h.boi2xurpqa0o for more details.
"""

        else:
            raise ValueError(f"Unknown failure type: {self.failure_type}")


class DraftExportReport:
    def __init__(self, failures: List[FailureReport]):
        self.failures: List[FailureReport] = failures

    def __str__(self):
        if len(self.failures) == 0:
            return """
##############################################################################################
Congratuations: No issues are found during export, and it was able to soundly produce a graph.
You can now change back to torch.export.export()
##############################################################################################
"""

        error = f"""
###################################################################################################
WARNING: {len(self.failures)} issue(s) found during export, and it was not able to soundly produce a graph.
Please follow the instructions to fix the errors.
Issues are compiled in hive table: TODO
###################################################################################################

"""

        for i, failure in enumerate(self.failures):
            error += f"{i + 1}. {str(failure)}\n"

        return error

    def serialize(self):
        pass

    def apply_suggested_fixes(self):
        pass


def draft_export(
    mod: torch.nn.Module,
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
    *,
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any], List[Any]]] = None,
    preserve_module_call_signature: Tuple[str, ...] = (),
    fake_tensor_propagate_real_tensors: Optional[bool] = True,
) -> ExportedProgram:
    kwargs = kwargs or {}

    with OperatorProfilingMode() as mode:
        mod(*args, **kwargs)
    custom_op_profile = mode.reports()

    torch._dynamo.config.custom_ops_profile = custom_op_profile

    with torch._functorch.config.patch(
        fake_tensor_propagate_real_tensors=fake_tensor_propagate_real_tensors
    ), RealTensorLoggingMode():
        try:
            new_shapes = None
            ep = torch.export.export(
                mod,
                args,
                kwargs,
                dynamic_shapes=dynamic_shapes,
                strict=False,
                preserve_module_call_signature=preserve_module_call_signature,
            )
        except torch._dynamo.exc.UserError as exc:
            new_shapes = refine_dynamic_shapes_from_suggested_fixes(
                exc.msg, dynamic_shapes
            )
            ep = torch.export.export(
                mod,
                args,
                kwargs,
                dynamic_shapes=new_shapes,
                strict=False,
                preserve_module_call_signature=preserve_module_call_signature,
            )

        from torch.fx.experimental.symbolic_shapes import REAL_TENSOR_LOGGING

        failures = []
        for op, profiles in custom_op_profile.data.items():
            failures.append(
                FailureReport(
                    FailureType.MISSING_FAKE_KERNEL,
                    (op, profiles),
                )
            )

        if new_shapes is not None:
            failures.append(
                FailureReport(
                    FailureType.INPUT_SHAPE_MISMATCH,
                    new_shapes,
                )
            )

        for log in REAL_TENSOR_LOGGING:
            failures.append(
                FailureReport(
                    FailureType.DATA_DEPENDENT_ERROR,
                    log,
                )
            )

        report = DraftExportReport(failures)
    ep._report = report

    return ep, report
