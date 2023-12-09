import copy
import dataclasses
import functools
import io
import json
import pathlib
import re
import sys
import os
import types
import warnings
import weakref
import zipfile
from collections import OrderedDict
from contextlib import contextmanager

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from unittest.mock import patch

import sympy

import torch
import torch._dynamo
import torch.fx
import torch.fx._pytree as fx_pytree

import torch.utils._pytree as pytree
from torch._decomp import core_aten_decompositions, get_decompositions
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.exc import UserError, UserErrorType
from torch._dynamo.source import ConstantSource
from torch._export.passes.collect_tracepoints_pass import CollectTracepointsPass
from torch._functorch.aot_autograd import aot_export_module, GraphSignature
from torch._functorch.eager_transforms import functionalize
from torch._guards import detect_fake_mode
from torch._ops import OpOverload
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.export.exported_program import (
    ExportedProgram,
    ModuleCallEntry,
    ModuleCallSignature,
    _disable_prexisiting_fake_mode,
)
from torch.export.graph_signature import (
    _sig_to_specs,
    ArgumentSpec,
    ConstantArgument,
    ExportGraphSignature,
    InputKind,
    InputSpec,
    OutputKind,
    OutputSpec,
    SymIntArgument,
    TensorArgument,
)
from torch.export.dynamic_shapes import (
    Constraint,
    dynamic_dim,
    _process_constraints,
    _process_dynamic_shapes,
)
from torch.fx import traceback as fx_traceback
from torch.fx._compatibility import compatibility
from torch.fx.experimental.proxy_tensor import make_fx, maybe_disable_fake_tensor_mode
from torch.fx.experimental.symbolic_shapes import (
    ConstraintViolationError,
    GuardOnDataDependentSymNode,
    ShapeEnv,
    StrictMinMaxConstraint,
)
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo
from torch.utils._sympy.value_ranges import ValueRangeError, ValueRanges

from .exported_program import (
    _create_stateful_graph_module,
    CallSpec,
)
from .passes.add_runtime_assertions_for_constraints_pass import (
    _AddRuntimeAssertionsForInlineConstraintsPass,
)
from .passes.lift_constant_tensor_pass import lift_constant_tensor_pass
from .passes.remove_runtime_assertions import _RemoveRuntimeAssertionsPass
from .passes.replace_sym_size_ops_pass import _replace_sym_size_ops_pass
from .passes.replace_view_ops_with_view_copy_ops_pass import (
    ReplaceViewOpsWithViewCopyOpsPass,
)
from .wrappers import _wrap_submodules
from torch._inductor import config


def export__RC__(
    f: Callable,
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
    *,
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
    strict: bool = True,
    preserve_module_call_signature: Tuple[str, ...] = (),
) -> ExportedProgram:
    """
    API for exporting with dynamic shape specifications instead of constraints.
    It should be considered "release candidate" (RC), meant to replace `export`.

    Here, `dynamic_shapes` is expected to be a dict from
    argument names of `f` to dynamic shape specifications OR a tuple where each element
    corresponds to the original order of the arguments defined in the function signature
    ,as follows:
    - The dynamic shape of a tensor argument can be specified as:
      - Either a dict from dynamic dimension indices to Dim types. It is not
        required to include static dimension indices in this dict, but when
        they are, they should be mapped to None.
      - Or a tuple of Dim types or None. The Dim types correspond to dynamic
        dimensions, whereas static dimensions are denoted by None.
    - Arguments that are dicts or tuples of tensors are recursively specified
      by using mappings or sequences of contained specifications.

    See `export` for documentation of `f`, `args`, `kwargs` and return.
    """
    from torch.export._trace import _export

    constraints = _process_dynamic_shapes(f, args, kwargs, dynamic_shapes)
    return _export(
        f,
        args,
        kwargs,
        constraints=constraints,
        strict=strict,
        preserve_module_call_signature=preserve_module_call_signature
    )


@dataclasses.dataclass
class ExportDynamoConfig:
    """
    Manage Export-specific configurations of Dynamo.
    """
    allow_rnn: bool = True


DECOMP_TABLE = core_aten_decompositions()


@compatibility(is_backward_compatible=False)
def capture_pre_autograd_graph(
    f: Callable,
    args: Tuple[Any],
    kwargs: Optional[Dict[str, Any]] = None,
    constraints: Optional[List[Constraint]] = None,
    _functional_pre_dispatch_IR: bool = False,
) -> torch.nn.Module:
    """
    A helper function that is intended to trace a module before any pre-autograd
    decomposition is run. The produced module will be "non-functional" and
    composed of aten operators. Later this API will be deleted in favor of more general
    torch.export API.

    Args:
      f: A callable to be traced

      args: example positional inputs.

      kwargs: optional example keyword inputs.

      constraints: A optional list of constraints on the dynamic arguments specifying
            their possible range of their shapes

    Returns:
        An nn.Module containing the traced method.

    """
    from torch.export._trace import _convert_input_to_fake, DEFAULT_EXPORT_DYNAMO_CONFIG

    decomp_table = {
        torch.ops.aten.dropout.default: torch.ops.aten.dropout.default.decompose,
        torch.ops.aten.batch_norm.default: torch.ops.aten.batch_norm.default.decompose,
        torch.ops.aten._batch_norm_impl_index.default: torch.ops.aten._batch_norm_impl_index.default.decompose,
        torch.ops.aten.native_batch_norm.default: torch.ops.aten.native_batch_norm.default.decompose,
    }

    if not _functional_pre_dispatch_IR:
        if kwargs is None:
            kwargs = {}

        with torch._dynamo.config.patch(dataclasses.asdict(DEFAULT_EXPORT_DYNAMO_CONFIG)):
            m = torch._dynamo.export(
                f,
                constraints=constraints,
                assume_static_by_default=True,
                tracing_mode="symbolic",
                decomposition_table=decomp_table,
                pre_dispatch=True,
                aten_graph=True,
            )(
                *args,
                **kwargs,
            )[0]

            _, _, _, fake_mode = _convert_input_to_fake(m, args, kwargs)

            m.meta["inline_constraints"] = {
                k: v
                for k, v in fake_mode.shape_env.runtime_var_to_range.items()
                if re.match(r"^[if]\d+$", str(k))
            }

            flat_args, _ = pytree.tree_flatten((args, kwargs or {}))
            range_constraints, equality_constraints = _process_constraints(m, 0, flat_args)
            module = _create_stateful_graph_module(
                m,
                range_constraints=range_constraints,
                equality_constraints=equality_constraints,
            )
    else:
        module = _export(f, args, kwargs, constraints=constraints, pre_dispatch=True, decomp_table=decomp_table).module()

    def _train(self, mode: bool = True):
        raise NotImplementedError("Calling train() is not supported yet.")

    def _eval(self, mode: bool = True):
        raise NotImplementedError("Calling eval() is not supported yet.")

    module.train = types.MethodType(_train, module)  # type: ignore[method-assign]
    module.eval = types.MethodType(_eval, module)  # type: ignore[method-assign]
    return module


def _export_to_torch_ir(
    f: Callable,
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
    constraints: Optional[List[Constraint]] = None,
    *,
    preserve_module_call_signature: Tuple[str, ...] = (),
    disable_constraint_solver: bool = False,
) -> torch.fx.GraphModule:
    from torch.export._trace import _export_to_torch_ir
    return _export_to_torch_ir(
        f,
        args,
        kwargs,
        constraints,
        preserve_module_call_signature=preserve_module_call_signature,
        disable_constraint_solver=disable_constraint_solver
    )


def export(
    f: Callable,
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
    constraints: Optional[List[Constraint]] = None,
    *,
    strict: bool = True,
    preserve_module_call_signature: Tuple[str, ...] = (),
) -> ExportedProgram:
    from torch.export._trace import _export

    if constraints is not None:
        warnings.warn(
            "Using `constraints` to specify dynamic shapes for export is DEPRECATED "
            "and will not be supported in the future. "
            "Please use `dynamic_shapes` instead (see docs on `torch.export.export`).",
            DeprecationWarning,
            stacklevel=2,
        )
    return _export(
        f,
        args,
        kwargs,
        constraints,
        strict=strict,
        preserve_module_call_signature=preserve_module_call_signature,
    )


def _export_non_strict(
    mod,
    fake_args,
    fake_kwargs,
    fake_params_buffers,
    *,
    pre_dispatch=False,
    decomp_table=None,
    transform=lambda x: x  # TODO(zhxchen17) Revisit if this is needed later.
):
    # This _reparametrize_module makes sure inputs and module.params/buffers have the same fake_mode,
    # otherwise aot_export_module will error out because it sees a mix of fake_modes.
    # And we want aot_export_module to use the fake_tensor mode in dynamo to keep the pipeline easy to reason about.
    with torch.nn.utils.stateless._reparametrize_module(mod, fake_params_buffers):
        gm, graph_signature = transform(aot_export_module)(
            mod,
            (*fake_args, *fake_kwargs.values()),
            trace_joint=False,
            pre_dispatch=pre_dispatch,
            decompositions=decomp_table,
        )

    # NOTE: aot_export adds symint metadata for placeholders with int values;
    # since these become specialized, we replace such metadata with the original values
    flat_args = pytree.tree_leaves((fake_args, fake_kwargs))
    index = 0
    total_param_buffers = len(graph_signature.parameters) + len(graph_signature.buffers)
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            if index >= total_param_buffers:
                user_arg = flat_args[index - total_param_buffers]
                if not isinstance(user_arg, torch.Tensor):
                    node.meta["val"] = user_arg
            index += 1

    is_joint = graph_signature.backward_signature is not None

    def make_argument_spec(node) -> ArgumentSpec:
        assert "val" in node.meta, f"{node} has no 'val' metadata field"
        val = node.meta["val"]
        if isinstance(val, FakeTensor):
            return TensorArgument(name=node.name)
        elif isinstance(val, torch.SymInt):
            return SymIntArgument(name=node.name)
        else:
            return ConstantArgument(value=val)

    input_specs, output_specs = _sig_to_specs(
        user_inputs=set(graph_signature.user_inputs),
        inputs_to_parameters=graph_signature.inputs_to_parameters,  # type: ignore[arg-type]
        inputs_to_buffers=graph_signature.inputs_to_buffers,  # type: ignore[arg-type]
        user_outputs=set(graph_signature.user_outputs),  # type: ignore[arg-type]
        buffer_mutations=graph_signature.buffers_to_mutate,  # type: ignore[arg-type]
        user_input_mutations=gm.meta.get("user_inputs_to_mutate", {}),  # type: ignore[arg-type]
        grad_params=graph_signature.backward_signature.gradients_to_parameters if is_joint else {},  # type: ignore[arg-type, union-attr]
        grad_user_inputs=graph_signature.backward_signature.gradients_to_user_inputs if is_joint else {},  # type: ignore[arg-type, union-attr]
        loss_output=graph_signature.backward_signature.loss_output if is_joint else None,  # type: ignore[arg-type, union-attr]
        inputs=[make_argument_spec(node) for node in gm.graph.nodes if node.op == "placeholder"],
        outputs=[make_argument_spec(node) for node in pytree.tree_leaves(next(iter(reversed(gm.graph.nodes))).args)],
    )
    export_graph_signature = ExportGraphSignature(input_specs=input_specs, output_specs=output_specs)

    tensor_constants = lift_constant_tensor_pass(gm, export_graph_signature)

    @dataclasses.dataclass
    class _ExportedProgramNonStrict:
        gm: torch.fx.GraphModule
        sig: ExportGraphSignature
        tensor_constants: Dict[str, torch.Tensor]

    return _ExportedProgramNonStrict(
        gm,
        export_graph_signature,
        tensor_constants,
    )


def _get_params_buffers(mod: torch.nn.Module) -> Dict[str, torch.Tensor]:
    params_buffers: Dict[str, torch.Tensor] = {}
    for name, param in mod.named_parameters(remove_duplicate=False):
        params_buffers[name] = param

    for name, buffer in mod.named_buffers(remove_duplicate=False):
        params_buffers[name] = buffer
    return params_buffers


@_disable_prexisiting_fake_mode
def _export(
    f: Callable,
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
    constraints: Optional[List[Constraint]] = None,
    *,
    strict: bool = True,
    decomp_table: Optional[Dict[str, Callable]] = None,
    pre_dispatch: bool = False,
    preserve_module_call_signature: Tuple[str, ...] = (),
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

        preserve_module_call_signature: A list of submodule paths for which the original
            calling conventions are preserved as metadata.

    Returns:
        An ExportedProgram containing the traced method.
    """
    from torch.export._trace import _export

    return _export(
        f,
        args,
        kwargs,
        constraints,
        strict=strict,
        preserve_module_call_signature=preserve_module_call_signature,
    )


def save(
    ep: ExportedProgram,
    f: Union[str, pathlib.Path, io.BytesIO],
    *,
    extra_files: Optional[Dict[str, Any]] = None,
    opset_version: Optional[Dict[str, int]] = None,
) -> None:
    from .serde.serialize import serialize, SerializedArtifact
    from .serde.schema import SCHEMA_VERSION
    artifact: SerializedArtifact = serialize(ep, opset_version)

    if isinstance(f, (str, pathlib.Path)):
        f = str(f)

    with zipfile.ZipFile(f, 'w') as zipf:
        # Save every field the SerializedArtifact to a file
        for field in dataclasses.fields(artifact):
            field_name = field.name
            serialized_field = getattr(artifact, field_name)
            zipf.writestr(f"serialized_{field_name}.json", serialized_field)

        zipf.writestr('version', str(SCHEMA_VERSION))

        # Add extra files if provided
        if extra_files:
            for extra_file_name, content in extra_files.items():
                encoded_content = content.encode('utf-8')
                zipf.writestr(f"extra_files/{extra_file_name}", encoded_content)


def load(
    f: Union[str, pathlib.Path, io.BytesIO],
    *,
    extra_files: Optional[Dict[str, Any]] = None,
    expected_opset_version: Optional[Dict[str, int]] = None,
) -> ExportedProgram:
    if isinstance(f, (str, pathlib.Path)):
        f = str(f)

    with zipfile.ZipFile(f, 'r') as zipf:
        # Check the version
        version = int(zipf.read('version'))
        from .serde.schema import SCHEMA_VERSION

        if version != SCHEMA_VERSION:
            raise RuntimeError(
                f"Serialized version {version} does not match our current "
                f"schema version {SCHEMA_VERSION}."
            )

        from .serde.serialize import deserialize, SerializedArtifact

        # Load serialized_ep and serialized_state_dict from the zip file
        artifact: SerializedArtifact = SerializedArtifact(
            **{
                field.name: zipf.read(f"serialized_{field.name}.json")
                for field in dataclasses.fields(SerializedArtifact)
            }
        )

        # Deserialize ExportedProgram
        ep = deserialize(artifact)

        # Populate extra_files map
        if extra_files is not None:
            for filename in extra_files.keys():
                extra_files[filename] = zipf.read(f"extra_files/{filename}").decode('utf-8')

        return ep


def aot_compile(
    f: Callable,
    args: Tuple[Any],
    kwargs: Optional[Dict[str, Any]] = None,
    *,
    constraints: Optional[List[Constraint]] = None,
    dynamic_shapes: Optional[Dict[str, Any]] = None,
    options: Optional[Dict[str, Any]] = None,
    remove_runtime_assertions: bool = False,
    disable_constraint_solver: bool = False,
) -> str:
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

        dynamic_shapes: An experimental new feature designed to subsume ``constraints``.
            A dict mapping argument names of ``f`` to their dynamic shape
            specifications, as follows. Dynamic shape specifications can be a
            dict from dynamic dimensions to ``Dim`` types, or a tuple/list of
            ``Optional[Dim]`` corresponding to each input dimension.

        options: A dictionary of options to control inductor

        disable_constraint_solver: Whether the dim constraint solver must be disabled.

    Returns:
        Path to the generated shared library
    """
    if constraints is not None:
        warnings.warn(
            "The constraints field is deprecated. "
            "Please use dynamic_shapes instead."
        )

    from torch._inductor.decomposition import select_decomp_table

    if constraints is None:
        constraints = _process_dynamic_shapes(f, args, kwargs, dynamic_shapes)

    if config.is_predispatch:
        gm = capture_pre_autograd_graph(f, args, kwargs, constraints)
    else:
        # We want to export to Torch IR here to utilize the pre_grad passes in
        # inductor, which run on Torch IR.
        gm = _export_to_torch_ir(
            f,
            args,
            kwargs,
            constraints,
            disable_constraint_solver=disable_constraint_solver
        )
    flat_example_inputs = pytree.arg_tree_leaves(*args, **(kwargs or {}))

    with torch.no_grad():
        so_path = torch._inductor.aot_compile(gm, flat_example_inputs, options)  # type: ignore[arg-type]

    return so_path
