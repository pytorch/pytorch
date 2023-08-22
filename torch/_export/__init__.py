import dataclasses
import inspect
import io
import re
import pathlib
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
from torch.export import Constraint
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
    constraints: Optional[List[_Constraint]] = None,
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


def _convert_input_to_fake(gm, args, kwargs):
    fake_inps: List[torch.Tensor] = []
    fake_mode = FakeTensorMode(
        allow_fallback_kernels=False,
        allow_non_fake_inputs=True,
        shape_env=ShapeEnv(
            assume_static_by_default=True,
        ),
    )

    for node in gm.graph.nodes:
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
    return fake_args, fake_kwargs, fake_mode


def _safe_to_skip_dynamo(gm: torch.fx.GraphModule):
    for node in gm.graph.nodes:
        if "is_torch_exported" in node.meta:
            return True
    return False


def export(
    f: Callable,
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
    constraints: Optional[List[_Constraint]] = None,
    *,
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
    constraints = constraints or []
    kwargs = kwargs or {}

    if not isinstance(args, tuple):
        raise UserError(UserErrorType.INVALID_INPUT,
                        f"Expecting `args` to be a tuple of example positional inputs, got {type(args)}")

    # We convert to nn.Module because __call__ of ExportedProgram
    # is untracable right now.
    if isinstance(f, ExportedProgram):
        if len(constraints) > 0:
            raise UserError(
                UserErrorType.INVALID_INPUT,
                "Cannot provide constraints for already exported program."
            )
        f = f.module()

    with torch._dynamo.config.patch(dataclasses.asdict(DEFAULT_EXPORT_DYNAMO_CONFIG)):  # type: ignore[attr-defined]
        try:
            module_call_signatures: Dict[str, ModuleCallSignature] = {}
            # TODO Horrible hack to skip dynamo
            if isinstance(f, torch.fx.GraphModule) and _safe_to_skip_dynamo(f):
                if len(constraints) > 0:
                    raise UserError(
                        UserErrorType.INVALID_INPUT,
                        "Cannot provide constraints for already exported program."
                    )
                gm_torch_level = f
            else:
                with _wrap_submodules(f, preserve_module_call_signature, module_call_signatures):
                    gm_torch_level, _ = torch._dynamo.export(
                        f,
                        constraints=constraints,
                        assume_static_by_default=True,
                        tracing_mode="symbolic",
                    )(
                        *args,
                        **kwargs,
                    )
        except (ConstraintViolationError, ValueRangeError) as e:
            raise UserError(UserErrorType.CONSTRAIN_VIOLATION, str(e))
        except GuardOnDataDependentSymNode as e:
            raise UserError(
                UserErrorType.ANTI_PATTERN,
                f"Consider annotating your code using constrain_as_*(). {str(e)}")

    params_buffers: OrderedDict[str, Union[torch.Tensor, torch.nn.Parameter]] = OrderedDict()
    for name, param in gm_torch_level.named_parameters(recurse=True, remove_duplicate=False):
        params_buffers[name] = param

    for name, buffer in gm_torch_level.named_buffers(recurse=True, remove_duplicate=False):
        params_buffers[name] = buffer

    fake_args, fake_kwargs, fake_mode = _convert_input_to_fake(gm_torch_level, args, kwargs)

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

        # If the call_function uses param as input, we also need to update params' meta
        # with this call_function node's meta.
        # This is basically the same flow as torch.fx.traceback.preserve_meta()
        if node.op == "call_function" and not isinstance(node.target, torch._ops.HigherOrderOperator):
            for arg in node._input_nodes:
                if arg.op == "get_attr":
                    for entry in torch.fx.proxy._COPY_META_FIELDS:
                        if entry in meta:
                            params_buffers_to_node_meta[arg.target][entry] = meta[entry]

    # Fix the graph output signature to be tuple if scalar
    out_spec = orig_out_spec = gm_torch_level._out_spec
    # aot_export expect the return type to always be a tuple.
    if out_spec.type not in (list, tuple):
        out_spec = pytree.TreeSpec(tuple, None, [out_spec])

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

    # NOTE: aot_export adds symint metadata for placeholders with int values;
    # since these become specialized, we replace such metadata with the original values
    # TODO: we should add runtime assertions for them
    for node in gm.graph.nodes:
        if node.op == "placeholder" and "val" in node.meta:
            s = node.meta['val']
            if (
                isinstance(s, torch.SymInt) and
                isinstance(fake_mode.shape_env.var_to_sources[s.node.expr][0], ConstantSource)
            ):
                node.meta['val'] = s.node.hint

    # TODO unfortunately preserving graph-level metadata is not
    # working well with aot_export. So we manually copy it.
    # (The node-level meta is addressed above.)
    gm.meta.update(gm_torch_level.meta)

    # The unbacked symint symbols are updated in aot_export
    # so we serialize them here instead of inside dynamo
    gm.meta["inline_constraints"] = {
        k: v
        for k, v in fake_mode.shape_env.runtime_var_to_range.items()
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
        [ModuleCallEntry(fqn, sig) for fqn, sig in module_call_signatures.items()],
    )

    exported_program = exported_program.transform(
        _AddRuntimeAssertionsForInlineConstraintsPass(range_constraints, equality_constraints)
    )
    if len(preserve_module_call_signature) > 0:
        exported_program = exported_program.transform(CollectTracepointsPass(module_call_signatures))
    return exported_program.transform(_ReplaceSymSizeOpPass())


def _reorder_kwargs_by_names(arg_names: List[str], args: Tuple[Any], kwargs: Dict[str, Any]):
    assert len(arg_names) == len(args) + len(kwargs), (
        f"Total number of arg names is expected to be {len(arg_names)} "
        f"but got {len(args)} positional args, {len(kwargs)} kwargs."
    )
    return OrderedDict({kw_name: kwargs[kw_name] for kw_name in arg_names[len(args):]})


def save(
    ep: ExportedProgram,
    f: Union[str, pathlib.Path, io.BytesIO],
    *,
    extra_files: Optional[Dict[str, Any]] = None,
    opset_version: Optional[Dict[str, int]] = None,
) -> None:
    """
    Saves an :class:`ExportedProgram` to a file-like object. It can then be
    loaded using the Python API :func:`torch._export.load <torch._export.load>`.

    Args:
        ep (ExportedProgram): The exported program to save.

        f (Union[str, pathlib.Path, io.BytesIO): A file-like object (has to
            implement write and flush) or a string containing a file name.

        extra_files (Optional[Dict[str, Any]]): Map from filename to contents
            which will be stored as part of f.

        opset_version (Optional[Dict[str, int]]): A map of opset names
            to the version of this opset


    Example:

    .. testcode::

        import torch
        import torch._export
        import io

        class MyModule(torch.nn.Module):
            def forward(self, x):
                return x + 10

        ep = torch._export.export(MyModule(), torch.randn(5))

        # Save to file
        torch._export.save(ep, 'exported_program.pt2')

        # Save to io.BytesIO buffer
        buffer = io.BytesIO()
        torch._export.save(ep, buffer)

        # Save with extra files
        extra_files = {'foo.txt': b'bar'}
        torch._export.save(ep, 'exported_program.pt2', extra_files=extra_files)
    """
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
) -> ExportedProgram:
    """
    Loads an :class:`ExportedProgram` previously saved with
    :func:`torch._export.save <torch._export.save>`.

    Args:
        ep (ExportedProgram): The exported program to save.

        f (Union[str, pathlib.Path, io.BytesIO): A file-like object (has to
            implement write and flush) or a string containing a file name.

        extra_files (Optional[Dict[str, Any]]): The extra filenames given in
            this map would be loaded and their content would be stored in the
            provided map.

        expected_opset_version (Optional[Dict[str, int]]): A map of opset names
            to expected opset versions

    Returns:
        An :class:`ExportedProgram` object

    Example:

    .. testcode::

        import torch
        import torch._export
        import io

        # Load ExportedProgram from file
        ep = torch._export.load('exported_program.pt2')

        # Load ExportedProgram from io.BytesIO object
        with open('exported_program.pt2', 'rb') as f:
            buffer = io.BytesIO(f.read())
        buffer.seek(0)
        ep = torch._export.load(buffer)

        # Load with extra files.
        extra_files = {'foo.txt': ''}  # values will be replaced with data
        ep = torch._export.load('exported_program.pt2', extra_files=extra_files)
        print(extra_files['foo.txt'])
    """
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

        # Deserialize ExportedProgram
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
    constraints: Optional[List[_Constraint]] = None,
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

    so_path = torch._inductor.aot_compile(ep.graph_module, list(all_args), options)
    return so_path, ep
