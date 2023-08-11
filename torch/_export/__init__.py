import dataclasses
import inspect
import re
import weakref
from collections import OrderedDict
from unittest.mock import patch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import sympy

import torch
import torch._dynamo
import torch.fx
import torch.fx._pytree as fx_pytree
from torch.fx._compatibility import compatibility

import torch.utils._pytree as pytree
from torch._decomp import core_aten_decompositions, get_decompositions
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.eval_frame import Constraint
from torch._dynamo.exc import UserError, UserErrorType
from torch._functorch.aot_autograd import aot_export_module
from torch._functorch.eager_transforms import functionalize
from torch._guards import detect_fake_mode
from torch._ops import OpOverload
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx import traceback as fx_traceback
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import (
    ConstraintViolationError,
    GuardOnDataDependentSymNode,
    ShapeEnv,
    StrictMinMaxConstraint,
)
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo
from torch.utils._sympy.value_ranges import ValueRangeError, ValueRanges

from .exported_program import (
    _process_constraints,
    CallSpec,
    combine_args_kwargs,
    ExportBackwardSignature,
    ExportedProgram,
    ExportGraphSignature,
)
from .passes.add_runtime_assertions_for_constraints_pass import (
    _AddRuntimeAssertionsForInlineConstraintsPass,
)
from .passes.replace_sym_size_ops_pass import _ReplaceSymSizeOpPass
from .passes.replace_view_ops_with_view_copy_ops_pass import (
    ReplaceViewOpsWithViewCopyOpsPass,
)

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
# )(
#     *sixtyfour_tensors,
# )
def dynamic_dim(t: torch.Tensor, index: int):
    if not isinstance(t, torch.Tensor):
        raise UserError(
            UserErrorType.DYNAMIC_DIM,
            f"Expected tensor as input to dynamic_dim but got {type(t)}"
        )

    if t.dim() < 1:
        raise UserError(
            UserErrorType.DYNAMIC_DIM,
            "Cannot mark 0-dimension tensors to be dynamic"
        )

    if index >= t.dim():
        raise UserError(
            UserErrorType.DYNAMIC_DIM,
            f"Expected the dimension passed to dynamic_dim to be in the range [0:{t.dim()-1}]"
            f" but got {index}, which is out of bounds for the given tensor."
        )

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
    """
    allow_rnn: bool = True

DEFAULT_EXPORT_DYNAMO_CONFIG = ExportDynamoConfig()


DECOMP_TABLE = core_aten_decompositions()


# FIXME: actually migrate it to pre_autograd tracing
@compatibility(is_backward_compatible=False)
def capture_pre_autograd_graph(
    f: Callable,
    args: Tuple[Any],
    kwargs: Optional[Dict[str, Any]] = None,
    constraints: Optional[List[Constraint]] = None,
    decomp_table: Dict[OpOverload, Callable] = core_aten_decompositions(),
) -> torch.nn.Module:
    """
    A helper function that is intended to trace a module before any pre-autograd
    decomposition is run. The produced module will be "non-functional" and
    composed of aten operators. You can manually specify decomp_table to control
    decomposition rule. Later this API will be deleted in favor of more general
    torch.export API.

    Args:
      f: A callable to be traced

      args: example positional inputs.

      kwargs: optional example keyword inputs.

      constraints: A optional list of constraints on the dynamic arguments specifying
            their possible range of their shapes

      decomp_table: A optional table of specifying how to decompose certain aten op.
    Returns:
        An nn.Module containing the traced method.

    """

    with patch("torch._export.DECOMP_TABLE", decomp_table):
        ep = export(f, args, kwargs, constraints=constraints)
    return ep.transform(ReplaceViewOpsWithViewCopyOpsPass()).module()


def export(
    f: Callable,
    args: Tuple[Any],
    kwargs: Optional[Dict[str, Any]] = None,
    constraints: Optional[List[Constraint]] = None,
) -> ExportedProgram:
    """
    Traces either an nn.Module's forward function or just a callable with PyTorch
    operations inside and produce a ExportedProgram.

    Args:
        m: the `nn.Module` or callable to trace.

        args: example positional inputs.

        kwargs: optional example keyword inputs.

        constraints: A optional list of constraints on the dynamic arguments specifying
            their possible range of their shapes

    Returns:
        An ExportedProgram containing the traced method.
    """
    constraints = constraints or []
    kwargs = kwargs or {}

    if not isinstance(args, tuple):
        raise UserError(UserErrorType.INVALID_INPUT,
                        f"Expecting `args` to be a tuple of example positional inputs, got {type(args)}")

    with torch._dynamo.config.patch(dataclasses.asdict(DEFAULT_EXPORT_DYNAMO_CONFIG)):  # type: ignore[attr-defined]
        try:
            # TODO horrible hack to skip dynamo when retracing

            def _safe_to_skip(gm: torch.fx.GraphModule):
                for node in gm.graph.nodes:
                    if "is_torch_exported" in node.meta:
                        return True
                return False

            if isinstance(f, torch.fx.GraphModule) and _safe_to_skip(f):
                gm_torch_level = f
            else:
                gm_torch_level, _ = torch._dynamo.export(
                    f,
                    constraints=constraints,
                    assume_static_by_default=True,
                    tracing_mode="symbolic",
                )(
                    *args,
                    **kwargs,
                )

            params_buffers: OrderedDict[str, Union[torch.Tensor, torch.nn.Parameter]] = OrderedDict()
            for name, param in gm_torch_level.named_parameters(recurse=True, remove_duplicate=False):
                params_buffers[name] = param

            for name, buffer in gm_torch_level.named_buffers(recurse=True, remove_duplicate=False):
                params_buffers[name] = buffer

            fake_inps: List[torch.Tensor] = []
            fake_mode = FakeTensorMode(
                allow_fallback_kernels=False,
                allow_non_fake_inputs=True,
                shape_env=ShapeEnv(
                    assume_static_by_default=True,
                ),
            )

            for node in gm_torch_level.graph.nodes:
                if node.op == "placeholder" and "val" in node.meta:
                    fake_val = node.meta["val"]
                    if fake_val is not None:
                        assert isinstance(fake_val, torch.Tensor)
                        fake_inps.append(fake_val)

            if detected_fake_mode := detect_fake_mode(fake_inps):
                fake_mode = detected_fake_mode

            count = 0

            def convert_to_fake(x):
                nonlocal count
                val = fake_inps[count]
                count += 1
                return val

            fake_args = pytree.tree_map_only(torch.Tensor, convert_to_fake, args)
            # TODO properly use the cached fake tensor
            fake_kwargs = pytree.tree_map_only(torch.Tensor, fake_mode.from_tensor, kwargs)

            # First, we want to pass through the graph to try populating
            # val field for getattr if there is anything missing.
            # THis can happen when quantization adds extra params and forgets
            # to update "val"
            for node in gm_torch_level.graph.nodes:
                if node.op == "get_attr" and "val" not in node.meta:
                    attr = getattr(gm_torch_level, node.target)
                    # Checks if it is not a HigherOrderOp branch or a module
                    if not isinstance(attr, torch.nn.Module):
                        node.meta["val"] = fake_mode.from_tensor(attr, static_shapes=True)

            # When aot_export lifts the params, we lose the nn_module_stack
            # and source_fn from the param nodes as they are treated as fresh inputs
            # Therefore, we manually extract them before calling into aot_export
            params_buffers_to_node_meta = OrderedDict()
            for node in gm_torch_level.graph.nodes:
                target = node.target
                meta = node.meta
                if node.op == "call_module":
                    submodule = getattr(gm_torch_level, target)
                    if isinstance(submodule, torch.nn.Module):
                        for name, _ in submodule.named_parameters(recurse=True, remove_duplicate=False):
                            params_buffers_to_node_meta[target + "." + name] = meta

                        for name, _ in submodule.named_buffers(recurse=True, remove_duplicate=False):
                            params_buffers_to_node_meta[target + "." + name] = meta

                if node.op == "get_attr":
                    submodule = getattr(gm_torch_level, target)
                    if not isinstance(submodule, torch.fx.GraphModule):
                        params_buffers_to_node_meta[target] = meta

                # If the call_function uses param as input, we also need to capture the meta for it
                # This is basically the same flow as torch.fx.traceback.preserve_meta()
                if node.op == "call_function" and not isinstance(node.target, torch._ops.HigherOrderOperator):
                    for arg in node._input_nodes:
                        if arg.op == "get_attr":
                            for entry in torch.fx.proxy._COPY_META_FIELDS:
                                if entry in meta:
                                    params_buffers_to_node_meta[arg.target][entry] = meta[entry]

            # Fix the graph output signature to be tuple if scalar
            # because aot_export expects a tuple as return type
            return_val = f(*args, **kwargs)
            out_spec = orig_out_spec = gm_torch_level._out_spec
            # this means it is scalar return value, so will make it tuple
            if not isinstance(return_val, (list, tuple)):
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

            # Note: aot_export_module doesn't accept kwargs, we'd like to reorder the kwargs as an OrderedDict
            # to follow the order in orig_args and correctly call gm_torch_level
            gm, graph_signature = aot_export_module(
                gm_torch_level,
                (*fake_args, *_reorder_kwargs_by_names(orig_args, fake_args, fake_kwargs).values()),
                decompositions=DECOMP_TABLE,
                trace_joint=False
            )

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

            # After aot_export, set the param/buffer metadata back into placeholders
            # Technically, users can still construct this data from param names
            # without relying on this metadata
            for node in gm.graph.nodes:
                if node.op == "placeholder":
                    if node.target in export_graph_signature.inputs_to_parameters:
                        param_name = export_graph_signature.inputs_to_parameters[node.target]
                        if param_name in params_buffers_to_node_meta:
                            for k, v in params_buffers_to_node_meta[param_name].items():
                                node.meta[k] = v
                    if node.target in export_graph_signature.inputs_to_buffers:
                        buffer_name = export_graph_signature.inputs_to_buffers[node.target]
                        if buffer_name in params_buffers_to_node_meta:
                            for k, v in params_buffers_to_node_meta[buffer_name].items():
                                node.meta[k] = v

                node.meta["is_torch_exported"] = True

            flat_args, in_spec = pytree.tree_flatten(combine_args_kwargs(args, kwargs))
            range_constraints, equality_constraints = _process_constraints(
                gm,
                export_graph_signature,
                flat_args,
            )
            assert orig_out_spec is not None
            exported_program = ExportedProgram(
                gm,
                gm.graph,
                export_graph_signature,
                CallSpec(in_spec, orig_out_spec),
                params_buffers,
                range_constraints,
                equality_constraints,
            )

            exported_program = exported_program.transform(
                _AddRuntimeAssertionsForInlineConstraintsPass(range_constraints, equality_constraints)
            )
            return exported_program.transform(_ReplaceSymSizeOpPass())

        except (ConstraintViolationError, ValueRangeError) as e:
            raise UserError(UserErrorType.CONSTRAIN_VIOLATION, str(e))
        except GuardOnDataDependentSymNode as e:
            raise UserError(
                UserErrorType.ANTI_PATTERN,
                f"Consider annotating your code using constrain_as_*(). {str(e)}")

def _reorder_kwargs_by_names(arg_names: List[str], args: Tuple[Any], kwargs: Dict[str, Any]):
    assert len(arg_names) == len(args) + len(kwargs), (
        f"Total number of arg names is expected to be {len(arg_names)} "
        f"but got {len(args)} positional args, {len(kwargs)} kwargs."
    )
    return OrderedDict({kw_name: kwargs[kw_name] for kw_name in arg_names[len(args):]})


def aot_compile(
    f: Callable,
    args: Tuple[Any],
    kwargs: Optional[Dict[str, Any]] = None,
    constraints: Optional[List[Constraint]] = None,
    options: Optional[Dict[str, Any]] = None,
) -> Tuple[str, ExportedProgram]:
    """
    Note: this function is not stable yet

    Traces either an nn.Module's forward function or just a callable with PyTorch
    operations inside, generates executable cpp code from the program, and returns
    the path to the generated shared library

    Args:
        f: the `nn.Module` or callable to trace.

        args: example positional inputs.

        kwargs: optional example keyword inputs.

        constraints: A optional list of constraints on the dynamic arguments specifying
            their possible range of their shapes

        options: A dictionary of options to control inductor

    Returns:
        Path to the generated shared library, and the exported program
    """
    from torch._inductor.compile_fx import compile_fx_aot
    from torch._inductor.decomposition import select_decomp_table

    global DECOMP_TABLE
    DECOMP_TABLE = select_decomp_table()
    ep = export(f, args, kwargs, constraints)
    # Reset the global value
    DECOMP_TABLE = core_aten_decompositions()

    param_buffer_values = list(ep.state_dict.values())
    flat_example_inputs = fx_pytree.tree_flatten_spec(
        combine_args_kwargs(args, kwargs), ep.call_spec.in_spec  # type: ignore[arg-type]
    )
    all_args = (*param_buffer_values, *flat_example_inputs)

    so_path = compile_fx_aot(
        ep.graph_module,
        all_args,  # type: ignore[arg-type]
        config_patches=options,
    )
    return so_path, ep
