import dataclasses
import weakref
import re
from collections import OrderedDict
from typing import Any, Callable, List, Tuple, Optional, Dict, Union

import sympy

import torch
import torch._dynamo
import torch.fx
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo
from .exported_program import (
    CallSpec,
    ExportedProgram,
    ExportBackwardSignature,
    ExportGraphSignature,
    _process_constraints,
)
from torch._decomp import core_aten_decompositions
from torch._dynamo.eval_frame import Constraint
from torch._functorch.aot_autograd import aot_export_module
from torch._guards import detect_fake_mode

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
    *,
    _add_runtime_assertions=True,
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
            gm_torch_level, _ = torch._dynamo.export(
                f,
                *args,
                constraints=constraints,
                assume_static_by_default=True,
            )

            params_buffers: "OrderedDict[str, Union[torch.Tensor, torch.nn.Parameter]]" = OrderedDict()
            for name, param in gm_torch_level.named_parameters(recurse=True, remove_duplicate=False):
                params_buffers[name] = param

            for name, buffer in gm_torch_level.named_buffers(recurse=True, remove_duplicate=False):
                params_buffers[name] = buffer

            fake_inps = []
            for node in gm_torch_level.graph.nodes:
                if node.op == "placeholder" and "val" in node.meta:
                    fake_val = node.meta["val"]
                    fake_inps.append(fake_val)

            fake_mode = detect_fake_mode(fake_inps)

            fake_args = pytree.tree_map_only(torch.Tensor, fake_mode.from_tensor, args)

            # Fix the graph output signature to be tuple if scalar
            # because aot_export expects a tuple as return type
            return_val = f(*args)
            flat_args, in_spec = pytree.tree_flatten(args)
            out_spec = orig_out_spec = gm_torch_level._out_spec
            # this means it is scalar return value, so will make it tuple
            if not isinstance(return_val, (list, tuple, dict)):
                out_spec = pytree.tree_flatten((return_val,))[1]

            orig_args = gm_torch_level.graph._codegen.pytree_info.orig_args  # type: ignore[attr-defined]

            gm_torch_level.graph._codegen = _PyTreeCodeGen(
                _PyTreeInfo(
                    orig_args,
                    gm_torch_level._in_spec,
                    out_spec,
                )
            )

            gm_torch_level.recompile()
            gm, graph_signature = aot_export_module(gm_torch_level, fake_args, decompositions=DECOMP_TABLE, trace_joint=False)

            export_backward_signature = ExportBackwardSignature(
                gradients_to_parameters=graph_signature.backward_signature.gradients_to_parameters,
                gradients_to_user_inputs=graph_signature.backward_signature.gradients_to_user_inputs,
                loss_output=graph_signature.backward_signature.loss_output
            ) if graph_signature.backward_signature is not None else None

            export_graph_signature = ExportGraphSignature(
                parameters=graph_signature.parameters,
                buffers=graph_signature.buffers,
                user_inputs=graph_signature.user_inputs,
                user_outputs=graph_signature.user_outputs,
                inputs_to_parameters=graph_signature.inputs_to_parameters,
                inputs_to_buffers=graph_signature.inputs_to_buffers,
                buffers_to_mutate=graph_signature.buffers_to_mutate,
                backward_signature=export_backward_signature
            )

            # TODO unfortunately preserving meta at graph level is not
            # working well with aot_export. So we manually copy it.
            # The node level meta is preserved.
            for key, val in gm_torch_level.meta.items():
                gm.meta[key] = val

            # The unbacked symint symbols are updated in aot_export
            # so we serialize them here instead of inside dynamo
            gm.meta["inline_constraints"] = {
                k: v
                for k, v in fake_mode.shape_env.var_to_range.items()
                if re.match(r"^[if]\d+$", str(k))
            }

            range_constraints, equality_constraints = _process_constraints(
                gm,
                export_graph_signature,
                flat_args,
            )
            exported_program = ExportedProgram(
                gm,
                gm.graph,
                export_graph_signature,
                CallSpec(in_spec, orig_out_spec),
                params_buffers,
                range_constraints,
                equality_constraints,
            )

            if _add_runtime_assertions:
                exported_program = exported_program._add_runtime_assertions()

            return exported_program

        except (ConstraintViolationError, ValueRangeError) as e:
            raise UserError(UserErrorType.CONSTRAIN_VIOLATION, str(e))
        except GuardOnDataDependentSymNode as e:
            raise UserError(
                UserErrorType.ANTI_PATTERN,
                f"Consider annotating your code using constrain_as_*(). {str(e)}")
