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
from torch._subclasses.functional_tensor import FunctionalTensor
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
from torch.export._unlift import _create_stateful_graph_module
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


@dataclasses.dataclass
class ExportDynamoConfig:
    """
    Manage Export-specific configurations of Dynamo.
    """
    allow_rnn: bool = True


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

    if _functional_pre_dispatch_IR:
        from torch.export._trace import _export
        module = _export(f, args, kwargs, constraints=constraints, pre_dispatch=True).module()
    else:
        if kwargs is None:
            kwargs = {}

        decomp_table = {op: op.decompose for op in FunctionalTensor.fake_functional_ops}
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

            if isinstance(f, torch.nn.Module):
                from torch.export._trace import _restore_state_dict
                _restore_state_dict(f, m)

            flat_args, _ = pytree.tree_flatten((args, kwargs or {}))
            range_constraints, equality_constraints = _process_constraints(m, 0, flat_args)
            module = _create_stateful_graph_module(
                m,
                range_constraints=range_constraints,
                equality_constraints=equality_constraints,
            )

    def _train(self, mode: bool = True):
        raise NotImplementedError("Calling train() is not supported yet.")

    def _eval(self, mode: bool = True):
        raise NotImplementedError("Calling eval() is not supported yet.")

    module.train = types.MethodType(_train, module)  # type: ignore[method-assign]
    module.eval = types.MethodType(_eval, module)  # type: ignore[method-assign]
    return module


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
    warnings.warn("This function is deprecated. Please use torch.export.export instead.")

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

    from torch.export._trace import _export_to_torch_ir
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
