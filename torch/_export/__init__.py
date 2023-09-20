import builtins
import copy
import dataclasses
import io
import re
import pathlib
import types
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
from torch.export import Constraint, _create_constraint
from torch._dynamo.exc import UserError, UserErrorType
from torch._dynamo.source import ConstantSource
from torch._export.exported_program import ModuleCallEntry, ModuleCallSignature
from torch._export.passes.collect_tracepoints_pass import CollectTracepointsPass
from torch._functorch.aot_autograd import aot_export_module
from torch._functorch.eager_transforms import functionalize
from torch._guards import detect_fake_mode
from torch._ops import OpOverload
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx import traceback as fx_traceback
from torch.fx._compatibility import compatibility
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
    DynamoExportedProgram,
    ExportGraphSignature,
)
from .passes.add_runtime_assertions_for_constraints_pass import (
    _AddRuntimeAssertionsForInlineConstraintsPass,
)
from .passes.lift_constant_tensor_pass import lift_constant_tensor_pass
from .passes.replace_sym_size_ops_pass import _ReplaceSymSizeOpPass
from .passes.replace_view_ops_with_view_copy_ops_pass import (
    ReplaceViewOpsWithViewCopyOpsPass,
)
from .wrappers import _wrap_submodules


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

    return _create_constraint(
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


# TODO(zhxchen17) This is not needed if we output pre_dispatch graph upfront from export().
@contextmanager
def _disable_decomp_table():
    global DECOMP_TABLE
    prev, DECOMP_TABLE = DECOMP_TABLE, {}
    try:
        yield
    finally:
        DECOMP_TABLE = prev


@compatibility(is_backward_compatible=False)
def capture_pre_autograd_graph(
    f: Callable,
    args: Tuple[Any],
    kwargs: Optional[Dict[str, Any]] = None,
    constraints: Optional[List[Constraint]] = None,
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

    decomp_table = {
        torch.ops.aten.dropout.default: torch.ops.aten.dropout.default.decompose,
        torch.ops.aten.batch_norm.default: torch.ops.aten.batch_norm.default.decompose,
        torch.ops.aten._batch_norm_impl_index.default: torch.ops.aten._batch_norm_impl_index.default.decompose,
        torch.ops.aten.native_batch_norm.default: torch.ops.aten.native_batch_norm.default.decompose,
    }

    if kwargs is None:
        kwargs = {}

    with torch._dynamo.config.patch(dataclasses.asdict(DEFAULT_EXPORT_DYNAMO_CONFIG)):  # type: ignore[attr-defined]
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

        for n in m.graph.nodes:
            n.meta["is_torch_exported"] = True

        def _train(self, mode: bool = True):
            raise NotImplementedError("Calling train() is not supported yet.")

        def _eval(self, mode: bool = True):
            raise NotImplementedError("Calling eval() is not supported yet.")

        m.train = types.MethodType(_train, m)  # type: ignore[method-assign]
        m.eval = types.MethodType(_eval, m)  # type: ignore[method-assign]
        return m


def export(
    f: Callable,
    f_args: Tuple[Any, ...],
    f_kwargs: Optional[Dict[str, Any]] = None,
    *,
    backend: Union[str, Callable] = "dynamo",
    options: Optional[Dict[str, Any]] = None,
) -> DynamoExportedProgram:
    """
    Traces either an nn.Module's forward function or just a callable with PyTorch
    operations inside and produce a DynamoExportedProgram.

    Args:
        f: the `nn.Module` or callable to trace.
        f_args: example positional inputs.
        f_kwargs: optional example keyword inputs.
        backend: the backend to use for exporting.  See `torch._export.list_backends()`
        options: A dictionary of options to control ``backend``

    Returns:
        An DynamoExportedProgram containing the traced method.
    """

    if not options:
        options = {}
    export_fn = torch.export.backends.registry._lookup_backend(backend)
    return export_fn(f, f_args, f_kwargs, **options)


def save(
    ep: DynamoExportedProgram,
    f: Union[str, pathlib.Path, io.BytesIO],
    *,
    extra_files: Optional[Dict[str, Any]] = None,
    opset_version: Optional[Dict[str, int]] = None,
) -> None:
    from .serde.serialize import serialize
    from .serde.schema import SCHEMA_VERSION
    serialized_program, serialized_state_dict = serialize(ep, opset_version)

    if isinstance(f, (str, pathlib.Path)):
        f = str(f)

    with zipfile.ZipFile(f, 'w') as zipf:
        # Save serialized_ep and serialized_state_dict to the zip file
        zipf.writestr('serialized_exported_program.json', serialized_program)
        zipf.writestr('serialized_state_dict.json', serialized_state_dict)
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
) -> DynamoExportedProgram:
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

        # Load serialized_ep and serialized_state_dict from the zip file
        serialized_ep = zipf.read('serialized_exported_program.json')
        serialized_state_dict = zipf.read('serialized_state_dict.json')

        # Deserialize DynamoExportedProgram
        from .serde.serialize import deserialize
        ep = deserialize(serialized_ep, serialized_state_dict, expected_opset_version)

        # Populate extra_files map
        if extra_files is not None:
            for filename in extra_files.keys():
                extra_files[filename] = zipf.read(f"extra_files/{filename}").decode('utf-8')

        return ep


def aot_compile(
    f: Callable,
    args: Tuple[Any],
    kwargs: Optional[Dict[str, Any]] = None,
    constraints: Optional[List[Constraint]] = None,
    options: Optional[Dict[str, Any]] = None,
) -> Tuple[str, DynamoExportedProgram]:
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
    from torch._inductor.decomposition import select_decomp_table

    global DECOMP_TABLE
    DECOMP_TABLE = select_decomp_table()
    ep = export(f, args, kwargs, options={"constraints": constraints})
    # Reset the global value
    DECOMP_TABLE = core_aten_decompositions()

    flat_example_inputs = fx_pytree.tree_flatten_spec(
        combine_args_kwargs(args, kwargs), ep.call_spec.in_spec  # type: ignore[arg-type]
    )

    unlifted_module = ep.module()
    unlifted_module.graph.set_codegen(torch.fx.CodeGen())  # type: ignore[attr-defined]
    unlifted_module.recompile()
    aot_compile_options = (
        {"from_export": True}
        if options is None
        else {**options, "from_export": True}
    )
    so_path = torch._inductor.aot_compile(unlifted_module, flat_example_inputs, aot_compile_options)  # type: ignore[arg-type]

    user_inputs = []
    user_outputs = []
    for node in unlifted_module.graph.nodes:
        if node.op == "placeholder":
            user_inputs.append(node.name)
        elif node.op == "output":
            user_outputs = [arg.name for arg in node.args[0]]

    unlifted_ep = DynamoExportedProgram(
        unlifted_module,
        unlifted_module.graph,
        ExportGraphSignature(
            parameters=[],
            buffers=[],
            user_inputs=user_inputs,
            user_outputs=user_outputs,
            inputs_to_parameters={},
            inputs_to_buffers={},
            buffers_to_mutate={},
            backward_signature=None,
        ),
        call_spec=copy.deepcopy(ep.call_spec),
        state_dict={},
        range_constraints=copy.deepcopy(ep.range_constraints),
        equality_constraints=copy.deepcopy(ep.equality_constraints),
        module_call_graph=ep.module_call_graph,
    )

    return so_path, unlifted_ep
