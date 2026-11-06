"""
Dynamo and user-defined CuTeDSL kernels (``@cute.jit`` host functions): Dynamo does
not capture a call of a wrapper that ``torch._library.cutedsl.wrap_cutedsl`` returns,
and refuses it with a graph break at the call.
"""

from collections.abc import Sequence
from typing import cast, TYPE_CHECKING

from torch._higher_order_ops.cutedsl_kernel_wrap import cutedsl_kernel_name

from ..exc import unimplemented
from .user_defined import UserDefinedObjectVariable


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase
    from torch._higher_order_ops.cutedsl_kernel_wrap import (
        TraceableCuteDSLKernelWrapper,
    )

    from .base import VariableTracker


class CuteDSLKernelWrapperVariable(UserDefinedObjectVariable):
    """What wrap_cutedsl returns (a TraceableCuteDSLKernelWrapper), or an instance of
    a subclass of it. A call of it graph-breaks at the call, rather than where
    VariableBuilder makes this variable: to make the call outside the graph, Dynamo
    rebuilds the wrapper from its source, and it then resumes after the call,
    compiling the rest of the frame. Any other use of the wrapper, such as reading an
    attribute, is traced as for any object."""

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: Sequence["VariableTracker"],
        kwargs: dict[str, "VariableTracker"],
    ) -> "VariableTracker":
        wrapper = cast("TraceableCuteDSLKernelWrapper", self.value)
        name = cutedsl_kernel_name(wrapper.kernel)
        unimplemented(
            gb_type="CuTeDSL kernel wrapper not supported",
            context=f"{type(wrapper).__name__}({name}, mutates_args={wrapper.mutates_args!r})",
            explanation=f"CuTeDSL kernel {name}: Dynamo does not capture a call of a "
            "CuTeDSL kernel wrapper (what torch._library.cutedsl.wrap_cutedsl returns): "
            "the call graph-breaks and runs outside the graph, so "
            "torch.compile(fullgraph=True) and strict torch.export refuse it. make_fx "
            "and non-strict torch.export capture the call, but not in the body of a "
            "higher-order op that Dynamo traces even eagerly and under make_fx and "
            "non-strict torch.export, such as torch.cond, while_loop, scan, map and "
            "associative_scan, or in a torch.compiler.nested_compile_region function "
            "under non-strict torch.export.",
            hints=[
                "Call it outside the compiled region, or drop fullgraph=True to run it between the compiled graphs.",
                "To capture the launch in a graph, use make_fx or non-strict torch.export, with the call outside the body of any higher-order op that Dynamo traces (such as torch.cond, while_loop, scan, map and associative_scan) and, under non-strict torch.export, outside any torch.compiler.nested_compile_region function.",
            ],
        )
