import dataclasses
import weakref
from typing import Any, Callable, List, Tuple, Optional, Dict, Union

import sympy

import torch
import torch._dynamo
import torch.fx
from .exported_program import (
    CallSpec,
    ExportedProgram,
    ExportGraphSignature,
    _set_constraints,
)
from torch._decomp import core_aten_decompositions
from torch._dynamo.eval_frame import Constraint

import torch.utils._pytree as pytree
from torch.fx.experimental.symbolic_shapes import (
    ConstraintViolationError,
    GuardOnDataDependentSymNode,
    StrictMinMaxConstraint,
)
from torch._dynamo.exc import UserError, UserErrorType
from torch.utils._sympy.value_ranges import ValueRanges, ValueRangeError



# Note - [On Export Dynamic Dimension UX]
#
# After a lot of discussion, we have settled on a dynamic marking API
# for export that meets the following constraints:
# 1) Stateless
# 2) Safe for numerous .export calls within a single process
# 3) Simple to use
# 4) Can be extended to constraints easily
#
# While the underlying API is still torch._dynamo.mark_dynamic, we offer a higher
# level API that meets the constraints above.
#
# This API produces an object that is meant to be passed into torch._dynamo.export
# constraints field. See docs on torch._dynamo.export for more details.
#
# Note - The output type and structure here is NOT BC and NOT A CONTRACT, we reserve
# the right to change the output here at any time, and will do so as we extend the API.
#
# result = torch._dynamo.export(
#     my_model,
#     *sixtyfour_tensors,
#     constraints=[
#         # if you do only dynamic_dim, this is sugar for
#         # -Inf <= dynamic_dim(blah, 0) <= Inf; we don’t otherwise
#         # permit direct int->bool conversion
#         dynamic_dim(blah, 0),
#         # operator overloading because it makes it clear whether
#         # or not you’re inclusive-exclusive range or not
#         0 <= dynamic_dim(blah, 1) <= 100,
#         # NB: But we actually truncate ranges to be >= 2, because of
#         # 0/1 specialization
#     ]
# )
def dynamic_dim(t: torch.Tensor, index: int):
    return Constraint(
        weakref.ref(t),
        id(t),
        index,
        StrictMinMaxConstraint(
            vr=ValueRanges(lower=2, upper=sympy.oo), warn_only=False
        ),
    )


@dataclasses.dataclass
class ExportDynamoConfig:
    """
    Manage Export-specific configurations of Dynamo.
    TODO add tests to make sure the flags are not outdated
    """
    capture_scalar_outputs: bool = True
    capture_dynamic_output_shape_ops: bool = True
    guard_nn_modules: bool = True
    dynamic_shapes: bool = True
    specialize_int: bool = True
    allow_rnn: bool = True


DECOMP_TABLE = core_aten_decompositions()


def export(
    f: Callable,
    args: Tuple[Any],
    constraints: Optional[List[Constraint]] = None,
) -> ExportedProgram:
    """
    Traces either an nn.Module's forward function or just a callable with PyTorch
    operations inside and produce a ExportedProgram.

    Args:
        m: the `nn.Module` or callable to trace.

        args: Tracing example inputs.

        constraints: A list of constraints on the dynamic arguments specifying
            their possible range of their shapes

    Returns:
        An ExportedProgram containing the traced method.
    """
    if constraints is None:
        constraints = []

    with torch._dynamo.config.patch(dataclasses.asdict(ExportDynamoConfig())):  # type: ignore[attr-defined]
        try:
            gm, _ = torch._dynamo.export(
                f,
                *args,
                aten_graph=True,
                tracing_mode="symbolic",
                decomposition_table=DECOMP_TABLE,
                constraints=constraints,
                assume_static_by_default=True,
                functionalize=True,
            )
        except (ConstraintViolationError, ValueRangeError) as e:
            raise UserError(UserErrorType.CONSTRAIN_VIOLATION, str(e))
        except GuardOnDataDependentSymNode as e:
            raise UserError(
                UserErrorType.ANTI_PATTERN,
                f"Consider annotating your code using constrain_as_*(). {str(e)}")

    flat_args, in_spec = pytree.tree_flatten(args)
    out_spec = (
        gm.graph._codegen.pytree_info.out_spec or pytree.tree_flatten(f(*args))[1]  # type: ignore[attr-defined]
    )

    # TODO(tugsuu): Fill out signature/state_dict
    graph_signature = ExportGraphSignature(
        parameters=[],
        buffers=[],
        user_inputs=[],
        user_outputs=[],
        inputs_to_parameters={},
        inputs_to_buffers={},
        buffers_to_mutate={},
        backward_signature=None,
    )

    exported_program = ExportedProgram(
        gm,
        gm.graph,
        graph_signature,
        CallSpec(in_spec, out_spec),
        {},
    )
    _set_constraints(
        exported_program,
        gm.meta.get("input_shape_constraints", []),
        gm.meta.get("inline_constraints", []),
        flat_args,
    )

    return exported_program
