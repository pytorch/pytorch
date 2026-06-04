from __future__ import annotations

import inspect
import logging
import sys
import traceback
import types
from collections import namedtuple
from contextlib import nullcontext
from typing import Any, cast, TYPE_CHECKING, TypeVar

import sympy

import torch
import torch.fx
import torch.utils._pytree as pytree
from torch._dynamo.convert_frame import CaptureOutput, fullgraph_capture, get_traced_fn
from torch._dynamo.decorators import disable as dynamo_disable
from torch._dynamo.eval_frame import argument_names, check_user_input_output
from torch._dynamo.exc import UserErrorType
from torch._dynamo.source import GetItemSource
from torch._dynamo.utils import dynamo_timed, get_metrics_context
from torch._export.utils import _compiling_state_context
from torch._guards import detect_fake_mode, TracingContext
from torch.export.dynamic_shapes import _IntWrapper, _RelaxedConstraint, Constraint
from torch.fx.experimental.symbolic_shapes import (
    ConstraintViolationError,
    DimDynamic,
    StatelessSymbolicContext,
)
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from torch._dynamo.output_graph import OutputReturnInfo
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.export._trace import _DynamicShapesInput
    from torch.fx import Node
    from torch.fx.node import Argument, Target

from torch.fx.experimental.dynamic_spec import (
    DictSpec,
    IntermediateSpec,
    IntVar,
    ObjectSpec,
    ParamsSpec,
    SeqSpec,
    ShapesSpec,
    TensorSpec,
)


T = TypeVar("T")
log = logging.getLogger(__name__)


def post_process_error_msg(
    constraint_violation_error: ConstraintViolationError,
    func: Callable[..., Any],
    args: Any,
    kwargs: Any,
) -> ConstraintViolationError:
    """
    Because we trace a different callable, the sources are all messed up.
    Manually patch them so the error message looks correct.
    """
    from torch.export._unlift import _get_input_paths, _replace_sources

    orig_sig = inspect.signature(func)
    flat_input_paths = _get_input_paths((args, kwargs), orig_sig)
    if constraint_violation_error.args:
        constraint_violation_error.args = (
            _replace_sources(constraint_violation_error.args[0], flat_input_paths),
        )
    return constraint_violation_error


EXPORT_ROOT_REPLACEMENTS = [
    ("__export_root_", "_"),
    ("_export_root.", ""),
    ("._export_root", ""),
]


def clean_export_root_string(text: str) -> str:
    """Generic utility to clean export_root patterns from strings."""
    result = text
    for pattern, replacement in EXPORT_ROOT_REPLACEMENTS:
        result = result.replace(pattern, replacement)
    return result


def clean_nn_module_stack_and_source_fn(
    graph_module: torch.fx.GraphModule, is_inline_builtin: bool = False
) -> torch.fx.GraphModule:
    """
    Clean up nn_module_stack metadata by removing export_root references.

    Removes the _export_root module references from nn_module_stack metadata
    in graph nodes, which are artifacts from the export process. Fixes two patterns:

    1. Keys: Removes "__export_root_" and "__modules['_export_root']_" prefixes
       - Normal case: "L__self____export_root_child" -> "L__self__child"
       - inline_builtin case: Uses numeric ID strings like "140468831433840"

    2. Values: Removes "._export_root" and "._modules['_export_root']" from child names
       e.g., "L['self']._export_root.child" -> "L['self'].child"
       e.g., "L['self']._modules['_export_root'].child" -> "L['self'].child"

    Also removes the root export entry "L__self____export_root" entirely.

    Args:
        graph_module: The GraphModule to clean up
        is_inline_builtin: If True, keys are numeric ID strings and self references
                          (L['self']) are filtered out

    Returns:
        The cleaned GraphModule (modified in-place)
    """

    def _process_nn_module_stack(
        nn_module_stack: dict[str, tuple[str, T]],
    ) -> dict[str, tuple[str, T]]:
        if "L__self____export_root" in nn_module_stack:
            del nn_module_stack["L__self____export_root"]

        # Clean up remaining entries
        cleaned_stack = {}
        for key, (child_name, child_class) in nn_module_stack.items():
            # Clean key by removing export_root patterns
            clean_key = clean_export_root_string(key)

            # Clean child_name by removing export_root patterns
            clean_name = clean_export_root_string(child_name)

            # Skip self reference for inline builtin case
            if is_inline_builtin and clean_name == "L['self']":
                continue

            cleaned_stack[clean_key] = (clean_name, child_class)
        return cleaned_stack

    def _process_source_fn(source_fn_stack: Iterable[T]) -> Iterable[T]:
        cleaned_stack = []
        for item in source_fn_stack:
            if isinstance(item, tuple) and len(item) == 2:
                name, cls = item
                if isinstance(name, str):
                    clean_name = clean_export_root_string(name)
                    cleaned_stack.append((clean_name, cls))
                else:
                    cleaned_stack.append(item)
            else:
                # pyrefly: ignore [bad-argument-type]
                cleaned_stack.append(item)
        # pyrefly: ignore [bad-return]
        return cleaned_stack

    for node in graph_module.graph.nodes:
        if "nn_module_stack" in node.meta:
            node.meta["nn_module_stack"] = _process_nn_module_stack(
                node.meta["nn_module_stack"].copy()
            )

        source_fn_stack = node.meta.get("source_fn_stack", None)
        if source_fn_stack:
            node.meta["source_fn_stack"] = _process_source_fn(source_fn_stack.copy())

    if "dynamo_flat_name_to_original_fqn" in graph_module.meta:
        # Clean up flat name to original fqn mapping
        clean_name_to_original_fqn = {}
        for flat_name, original_fqn in graph_module.meta[
            "dynamo_flat_name_to_original_fqn"
        ].items():
            clean_name_to_original_fqn[clean_export_root_string(flat_name)] = (
                clean_export_root_string(original_fqn)
            )
        graph_module.meta["dynamo_flat_name_to_original_fqn"] = (
            clean_name_to_original_fqn
        )

    return graph_module


def clean_export_root(graph_module: torch.fx.GraphModule) -> None:
    """Remove export_root artifacts from FX graph in-place"""

    # Unlike getattr node, call_module can be invoked multiple times
    # In those cases, we should fix all invocations of call_module
    clean_named_module_map: dict[str, str] = {}

    # Update get_attr nodes in-place
    for node in graph_module.graph.nodes:
        if node.op == "get_attr":
            old_target = node.target
            new_target = clean_export_root_string(old_target)
            if new_target != old_target:
                node.target = new_target
                if not hasattr(graph_module, old_target):
                    raise AssertionError(
                        f"graph_module missing attribute {old_target!r}"
                    )
                # Move the parameter to the new name
                param = torch.fx.graph_module._get_attr(graph_module, old_target)
                torch.fx.graph_module._assign_attr(param, graph_module, new_target)
                torch.fx.graph_module._del_attr(graph_module, old_target)
        # Dynamo will only have one nested level
        if node.op == "call_module":
            old_target = node.target
            if not isinstance(old_target, str):
                raise AssertionError(
                    f"expected old_target to be str, got {type(old_target)}"
                )
            new_target = clean_export_root_string(old_target)
            if not isinstance(new_target, str):
                raise AssertionError(
                    f"expected new_target to be str, got {type(new_target)}"
                )
            new_name = clean_export_root_string(node.name)
            if new_target == old_target:
                continue

            # if this module has already been cleaned before, just lookup from map.
            if old_target in clean_named_module_map:
                node.target = clean_named_module_map[old_target]
                node.name = new_name
                continue
            target = graph_module.get_submodule(old_target)
            graph_module.delete_submodule(old_target)
            graph_module.add_submodule(new_target, target)
            node.target = new_target
            node.name = new_name
            clean_named_module_map[old_target] = new_target


class ModuleToTrace(torch.nn.Module):
    def __init__(self, foo: Any, in_spec: Any) -> None:
        super().__init__()
        self._export_root = foo
        self.in_spec = in_spec

    def forward(self, *flat_args: Any) -> ExportTracerOutput:
        args, kwargs = pytree.tree_unflatten(flat_args, self.in_spec)
        res = self._export_root(*args, **kwargs)
        out_flat, out_spec = pytree.tree_flatten(res)
        return ExportTracerOutput(out_flat, out_spec)


ExportTracerOutput = namedtuple("ExportTracerOutput", ["flat_args", "out_spec"])


# mypy: disable-error-code="no-untyped-def,var-annotated,assignment,index,operator"
class DynamoGraphTransformer(torch.fx.Transformer):
    """Graph transformer for dynamo export that flattens inputs/outputs without complex matching."""

    def __init__(
        self,
        module: torch.fx.GraphModule,
        flat_inputs: list[Any],
        flat_args_dynamic_dims: list[set[int]],
        graph_input_order: dict[int, int],
        graph_output_map: dict[int, OutputReturnInfo],
        fake_mode: Any | None = None,
        graph_inputs: dict[int, Any] | None = None,
    ) -> None:
        super().__init__(module)

        if len(flat_args_dynamic_dims) != len(flat_inputs):
            raise AssertionError(
                f"flat_args_dynamic_dims length ({len(flat_args_dynamic_dims)}) "
                f"must match flat_inputs length ({len(flat_inputs)})"
            )

        self.flat_inputs = flat_inputs
        self.flat_args_dynamic_dims = flat_args_dynamic_dims
        self.graph_input_order = graph_input_order
        self.graph_output_map = graph_output_map
        self.fake_mode = fake_mode
        self.graph_inputs = graph_inputs or {}

        # Get original placeholders and output
        self.placeholders = [n for n in module.graph.nodes if n.op == "placeholder"]
        self.output_node = next(n for n in module.graph.nodes if n.op == "output")

        # Create new flattened input placeholders
        self.new_input_nodes: dict[int, torch.fx.Node] = {}
        self._create_flattened_inputs()

        # Iterator for replacing old placeholders
        self.old_to_new_mapping = {}
        self._create_placeholder_mapping()

    def _create_flattened_inputs(self) -> None:
        """Create new placeholder nodes for flattened inputs with proper fake tensors."""
        for i in range(len(self.flat_inputs)):
            placeholder = super().placeholder(f"arg_{i}", (), {})

            # Check if this user input (index i) maps to a graph placeholder
            if i in self.graph_input_order:
                # graph_input_order[i] gives us which graph placeholder this user input corresponds to
                graph_placeholder_idx = self.graph_input_order[i]
                if graph_placeholder_idx < len(self.placeholders):
                    orig_placeholder = self.placeholders[graph_placeholder_idx]
                    # Copy other metadata but not "val" yet
                    for key, value in orig_placeholder.meta.items():
                        if key != "val":
                            placeholder.node.meta[key] = value

            # Always ensure we have proper "val" metadata from fake tensor
            if self.fake_mode is not None and isinstance(
                self.flat_inputs[i], torch.Tensor
            ):
                placeholder.node.meta["val"] = self.fake_mode.from_tensor(
                    self.flat_inputs[i],
                    symbolic_context=StatelessSymbolicContext(
                        dynamic_sizes=[
                            (
                                DimDynamic.DYNAMIC
                                if d in self.flat_args_dynamic_dims[i]
                                else DimDynamic.STATIC
                            )
                            for d in range(len(self.flat_inputs[i].shape))
                        ],
                        constraint_sizes=[None] * len(self.flat_inputs[i].shape),
                    ),
                )
            elif hasattr(self.flat_inputs[i], "val"):  # _IntWrapper case
                placeholder.node.meta["val"] = self.flat_inputs[i].val
            else:
                placeholder.node.meta["val"] = self.flat_inputs[i]

            # pyrefly: ignore [unsupported-operation]
            self.new_input_nodes[i] = placeholder

    def _create_placeholder_mapping(self) -> None:
        """Create mapping from old placeholders to new ones."""
        # graph_input_order maps: user_input_index -> graph_placeholder_index
        # We need to create: old_graph_placeholder -> new_user_input_placeholder
        for user_input_idx, graph_placeholder_idx in self.graph_input_order.items():
            if graph_placeholder_idx < len(self.placeholders):
                old_placeholder = self.placeholders[graph_placeholder_idx]
                new_placeholder = self.new_input_nodes[user_input_idx]
                self.old_to_new_mapping[old_placeholder] = new_placeholder

    def placeholder(
        self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Any:
        """Replace old placeholders with new flattened ones."""
        # Return the corresponding new placeholder
        if self.current_node in self.old_to_new_mapping:
            new_arg = self.old_to_new_mapping[self.current_node]

            # Copy over additional metadata from current node, but don't overwrite "val"
            for key in ["tensor_dict", "example_value", "unbacked_bindings"]:
                if key in self.current_node.meta:
                    new_arg.node.meta[key] = self.current_node.meta[key]

            # Only copy "val" if we don't already have a good one
            if "val" in self.current_node.meta and "val" not in new_arg.node.meta:
                new_arg.node.meta["val"] = self.current_node.meta["val"]

            return new_arg
        else:
            # Convert captured objects (e.g., opaque objects from closures) to
            # get_attr nodes
            placeholder_idx = self.placeholders.index(self.current_node)
            if placeholder_idx in self.graph_inputs:
                source = self.graph_inputs[placeholder_idx]
                if not isinstance(source, GetItemSource):
                    example_val = self.current_node.meta.get(
                        "val"
                    ) or self.current_node.meta.get("example_value")
                    if example_val is not None:
                        attr_name = f"_captured_{placeholder_idx}"
                        if isinstance(example_val, torch.Tensor):
                            self.module.register_buffer(attr_name, example_val)
                        else:
                            setattr(self.module, attr_name, example_val)
                        result = self.tracer.create_proxy("get_attr", attr_name, (), {})
                        result.node.meta = self.current_node.meta.copy()
                        result.node.meta["val"] = example_val
                        return result
            return super().placeholder(target, args, kwargs)

    def output(
        self, target: Target, args: Sequence[Any], kwargs: dict[str, Any]
    ) -> Any:
        """Transform output according to graph_output_map."""
        original_outputs = args[0]

        # Build new output list based on graph_output_map
        new_outputs = []
        for i in sorted(self.graph_output_map.keys()):
            output_type, val = self.graph_output_map[i]

            if output_type == "graph_out":
                new_outputs.append(original_outputs[cast(int, val)])
            elif output_type == "input":
                input_idx = cast(GetItemSource, val).index
                new_outputs.append(self.new_input_nodes[input_idx])
            elif output_type == "constant":
                new_outputs.append(val)

        return super().output(target, (tuple(new_outputs),), {})

    def run_node(self, n: Node) -> Any:
        """Run node transformation and preserve metadata."""
        self.current_node = n
        result = super().run_node(n)

        # Copy important metadata
        if hasattr(result, "node") and result.node is not n:
            for key in ["val", "example_value", "unbacked_bindings"]:
                if key in n.meta:
                    result.node.meta[key] = n.meta[key]

            # Preserve node names (except output)
            if n.op != "output" and hasattr(n, "name"):
                result.node._rename(n.name)

        return result

    def transform(self) -> torch.fx.GraphModule:
        """Perform the graph transformation and copy module metadata."""
        result_gm = super().transform()

        # Copy module metadata like the original implementation
        if hasattr(self.module, "meta"):
            # pyrefly: ignore [unsupported-operation]
            if "dynamo_flat_name_to_original_fqn" in self.module.meta:
                # pyrefly: ignore [bad-index]
                result_gm.meta["dynamo_flat_name_to_original_fqn"] = self.module.meta[
                    # pyrefly: ignore [bad-index]
                    "dynamo_flat_name_to_original_fqn"
                ]
            # pyrefly: ignore [unsupported-operation]
            if "dynamo_compile_id" in self.module.meta:
                # pyrefly: ignore [bad-index]
                result_gm.meta["dynamo_compile_id"] = self.module.meta[
                    # pyrefly: ignore [bad-index]
                    "dynamo_compile_id"
                ]

        return result_gm


def _suggest_or_raise_constraint_violation(
    module_to_trace: torch.nn.Module,
    orig_callable: Callable[..., Any],
    fake_mode: FakeTensorMode | None,
    graph_capture_output: CaptureOutput,
    args: Any,
    kwargs: Any,
    dynamic_shapes: dict[str, Any] | tuple[Any] | list[Any] | None,
) -> None:
    constraint_violation_error = None
    try:
        # Check if we have any constraint violations
        fn, _ = get_traced_fn(module_to_trace)
        graph_capture_output.graph_capture_output.build_guards(fn.__code__)
    except ConstraintViolationError as e:
        constraint_violation_error = e

    if (
        (shape_env := getattr(fake_mode, "shape_env", None)) is not None
        and (dim_constraints := shape_env.dim_constraints) is not None
        and not isinstance(
            module_to_trace.forward,
            torch._ops.OpOverloadPacket | torch._ops.OpOverload,
        )
    ):
        dim_constraints.solve()

        forced_specializations = dim_constraints.forced_specializations()

        msg = dim_constraints.prettify_results(
            inspect.signature(orig_callable),  # type: ignore[attr-defined]
            dynamic_shapes,
            constraint_violation_error,
            forced_specializations,
        )
        if constraint_violation_error:
            if constraint_violation_error.args:
                constraint_violation_error.args = (
                    constraint_violation_error.args[0] + msg,
                )
            else:
                constraint_violation_error.args = (msg,)
        else:
            if forced_specializations:
                constraint_violation_error = ConstraintViolationError(msg)
            else:
                log.info(
                    "Summary of dimension constraints:%s",
                    msg,
                )

        # Error if we have any constraints on static values

        for k in shape_env.var_to_range:
            if isinstance(k, sympy.Integer):
                constraint_violation_error = ConstraintViolationError(
                    f"{''.join(traceback.format_list(shape_env.var_to_stack[k]))}\n"
                    "It appears that you're trying to set a constraint on a "
                    f"value which we evaluated to have a static value of {k}. "
                    'Set TORCH_LOGS="+export" for more information.'
                )
    if constraint_violation_error:
        constraint_violation_error = post_process_error_msg(
            constraint_violation_error, orig_callable, args, kwargs
        )
        raise constraint_violation_error


def _normalize_shuffle_graph(shuffle_gm: torch.fx.GraphModule) -> None:
    shuffle_gm.graph.eliminate_dead_code()
    shuffle_gm.recompile()
    for name, buffer in list(shuffle_gm.named_buffers()):
        delattr(shuffle_gm, name)
        setattr(shuffle_gm, name, buffer)


def normalize_graph_module(gm: torch.fx.GraphModule) -> None:
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            node.meta["val"] = node.meta["example_value"]


class InputProcessor:
    def __init__(
        self,
        root: object,
        num_args: int,
        kwarg_names: list[str],
    ) -> None:
        self.root = root
        self.num_args = num_args
        self.kwarg_names = kwarg_names

    def __call__(
        self, inputs: tuple[object, ...]
    ) -> tuple[tuple[object, ...], dict[str, object]]:
        args = inputs
        # pyrefly: ignore [implicit-any]
        kwargs = {}
        if len(args) > self.num_args:
            kwargs = dict(zip(self.kwarg_names, args[self.num_args :]))
            args = args[: self.num_args]
        if self.root is not None:
            if isinstance(self.root, torch.fx.GraphModule):
                if not isinstance(self.root.graph._codegen, _DynamoBytecodeCodeGen):
                    raise AssertionError(
                        f"expected _DynamoBytecodeCodeGen, got {type(self.root.graph._codegen)}"
                    )
                if not hasattr(
                    self.root.graph._codegen.dynamo_bytecode_flatten, "input_processor"
                ):
                    raise AssertionError(
                        "dynamo_bytecode_flatten missing input_processor attribute"
                    )
                if (
                    self.root.graph._codegen.dynamo_bytecode_flatten.input_processor
                    is not self
                ):
                    raise AssertionError(
                        "dynamo_bytecode_flatten.input_processor is not self"
                    )
            args = (self.root, *args)
        return args, kwargs


class Yield(Exception):
    pass


class DynamoBytecodeFlatten:
    def __init__(
        self,
        input_processor: InputProcessor,
        out: CaptureOutput,
        f_globals: dict[str, object],
    ) -> None:
        self.input_processor = input_processor
        self.out = out
        self.f_globals = f_globals
        self.gm_inputs: tuple[Any, ...] | None = None

    @dynamo_disable(reason="do not trace internal dynamo graph capture")  # type: ignore[misc]
    def __call__(self, *inputs: object) -> object:
        def backend_dummy(*example_inputs: object) -> None:
            self.gm_inputs = example_inputs
            raise Yield

        args, kwargs = self.input_processor(inputs)
        try:
            self.out.forward_callable(
                compiled_fn=backend_dummy, extra_globals=self.f_globals
            )(*args, **kwargs)
        except Yield:
            if self.gm_inputs is None:
                raise AssertionError("gm_inputs must not be None after Yield") from None
            return self.gm_inputs
        raise RuntimeError


class DynamoBytecodeUnflatten:
    def __init__(
        self,
        input_processor: InputProcessor,
        out: CaptureOutput,
        f_globals: dict[str, object],
    ) -> None:
        self.input_processor = input_processor
        self.out = out
        self.f_globals = f_globals

    @dynamo_disable(reason="do not trace internal dynamo graph capture")  # type: ignore[misc]
    def __call__(
        self, flat_outs: Sequence[object], inputs: tuple[object, ...]
    ) -> object:
        def backend_dummy(*example_inputs: object) -> Sequence[object]:
            return flat_outs

        args, kwargs = self.input_processor(inputs)
        with torch._C._DisableTorchDispatch():
            results = self.out.forward_callable(
                compiled_fn=backend_dummy, extra_globals=self.f_globals
            )(*args, **kwargs)
        return results


def create_fx_graph_from_captured_output(
    out: CaptureOutput, mod: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> torch.fx.GraphModule:
    if out.backend_input is None:
        raise AssertionError("out.backend_input must not be None")
    backend_input = out.backend_input

    _, root = torch._dynamo.convert_frame.get_traced_fn(mod)

    flat_real_args = pytree.tree_leaves((args, kwargs))
    torch._dynamo.eval_frame.check_user_input_output(
        flat_real_args, UserErrorType.INVALID_INPUT
    )
    f_globals = out.graph_capture_output.f_globals

    graph_module = backend_input.graph_module
    if isinstance(root, torch.nn.Module):
        graph_module._parameters = root._parameters
        graph_module._buffers = root._buffers
        if not all(not hasattr(graph_module, m) for m in root._modules):
            raise AssertionError(
                "graph_module already has attributes conflicting with root._modules"
            )
        graph_module._modules.update(root._modules)
        graph_module._non_persistent_buffers_set = root._non_persistent_buffers_set
        if sys.version_info >= (3, 14):
            import annotationlib  # added in 3.14

            annotations = annotationlib.get_annotations(torch.nn.Module)
        else:
            annotations = getattr(torch.nn.Module, "__annotations__", None)
        for name, value in root.__dict__.items():
            if annotations and name not in annotations:
                graph_module.__dict__[name] = value
        graph_module._forward_hooks = root._forward_hooks.copy()
        graph_module._forward_pre_hooks = root._forward_pre_hooks.copy()
        graph_module._backward_hooks = root._backward_hooks.copy()
        graph_module._backward_pre_hooks = root._backward_pre_hooks.copy()
        if graph_module._forward_hooks or graph_module._forward_pre_hooks:
            # Even forward hooks are traced through, they still capture a bunch
            # of state through closure. We need to make sure these data are
            # accessible through the captured module (but the hooks should be
            # disabled).
            if getattr(graph_module, "_wrapped_call", None) is None:
                raise AssertionError("graph_module._wrapped_call must not be None")
            if not isinstance(
                graph_module._wrapped_call, torch.fx.graph_module._WrappedCall
            ):
                raise AssertionError(
                    f"expected _WrappedCall, got {type(graph_module._wrapped_call)}"
                )
            if graph_module._wrapped_call.cls_call is not None:
                raise AssertionError("graph_module._wrapped_call.cls_call must be None")

            def dynamo_wrapped_call(self, *args: object, **kwargs: object) -> object:
                if "forward" in self.__dict__:
                    raise AssertionError("'forward' should not be in self.__dict__")

                fwd_hooks = self._forward_hooks
                fwd_pre_hooks = self._forward_pre_hooks
                original_forward = type(self).forward

                def patched_forward(self, *args: object, **kwargs: object) -> object:
                    self._forward_hooks = fwd_hooks
                    self._forward_pre_hooks = fwd_pre_hooks
                    return original_forward(self, *args, **kwargs)

                try:
                    self.forward = types.MethodType(patched_forward, self)
                    # pyrefly: ignore [implicit-any]
                    self._forward_hooks = {}
                    # pyrefly: ignore [implicit-any]
                    self._forward_pre_hooks = {}
                    # pyrefly: ignore [invalid-argument]
                    return super(type(self), self).__call__(*args, **kwargs)
                finally:
                    self.__dict__.pop("forward")
                    self._forward_hooks = fwd_hooks
                    self._forward_pre_hooks = fwd_pre_hooks

            # pyrefly: ignore [bad-assignment]
            graph_module._wrapped_call.cls_call = dynamo_wrapped_call

    root = graph_module if isinstance(root, torch.nn.Module) else root
    input_processor = InputProcessor(root, len(args), list(kwargs.keys()))
    dynamo_bytecode_flatten = DynamoBytecodeFlatten(input_processor, out, f_globals)
    dynamo_bytecode_unflatten = DynamoBytecodeUnflatten(input_processor, out, f_globals)

    graph_module.graph._codegen = _DynamoBytecodeCodeGen(
        argument_names(inspect.signature(mod), args, kwargs),
        dynamo_bytecode_flatten,
        dynamo_bytecode_unflatten,
    )  # type: ignore[attr-defined]
    normalize_graph_module(graph_module)
    if hasattr(graph_module, "_dynamo_bytecode_flatten"):
        raise AssertionError(
            "graph_module already has _dynamo_bytecode_flatten attribute"
        )
    if hasattr(graph_module, "_dynamo_bytecode_unflatten"):
        raise AssertionError(
            "graph_module already has _dynamo_bytecode_unflatten attribute"
        )
    # pyrefly: ignore [bad-argument-type]
    graph_module._dynamo_bytecode_flatten = dynamo_bytecode_flatten
    # pyrefly: ignore [bad-argument-type]
    graph_module._dynamo_bytecode_unflatten = dynamo_bytecode_unflatten
    delattr(graph_module, "_param_name_to_source")
    graph_module.recompile()
    graph_module.meta["module_call_specs"] = (
        out.graph_capture_output.output_graph.export_metadata.module_call_spec
    )
    if out.backend_input is None:
        raise AssertionError("out.backend_input must not be None for fake_mode")
    graph_module.meta["fake_mode"] = out.backend_input.fake_mode  # type: ignore[attr-defined]
    graph_module.meta["fake_mode"].allow_non_fake_inputs = True
    tracing_context = TracingContext(graph_module.meta["fake_mode"])
    tracing_context.tensor_to_context = out.backend_input.tensor_to_context  # type: ignore[attr-defined]
    graph_module.meta["tracing_context"] = tracing_context
    return graph_module


class _DynamoBytecodeCodeGen(torch.fx.graph.CodeGen):
    def __init__(
        self,
        orig_arg_names: list[str],
        # pyrefly: ignore [implicit-any]
        dynamo_bytecode_flatten: Callable,
        # pyrefly: ignore [implicit-any]
        dynamo_bytecode_unflatten: Callable,
    ) -> None:
        super().__init__()
        self.orig_arg_names = orig_arg_names
        self.dynamo_bytecode_flatten = dynamo_bytecode_flatten
        self.dynamo_bytecode_unflatten = dynamo_bytecode_unflatten
        self.wrap_tuple = False
        self._inputs: tuple[Any, ...] | None = None

    def process_inputs(self, *inputs: Any) -> Any:
        self._inputs = inputs
        results = self.dynamo_bytecode_flatten(*inputs)
        fake_mode = detect_fake_mode()
        if fake_mode is not None and pytree.tree_any(
            lambda x: isinstance(x, torch.Tensor) and not fake_mode.is_our_fake(x),
            results,
        ):
            # Bytecode replay can recover tensors captured from module
            # attributes as extra FX placeholder values. These tensors are not
            # visible in AOTAutograd's user-facing call signature, so allow the
            # active FakeTensorMode to convert them at dispatch time. This must
            # remain enabled for AOT's backward trace too, since autograd can
            # save the recovered tensors from the forward.
            fake_mode.allow_non_fake_inputs = True
        return results

    def process_outputs(self, outputs: Any) -> Any:
        results = self.dynamo_bytecode_unflatten(outputs, self._inputs)
        if self.wrap_tuple:
            results = (results,)
        self._inputs = None
        return results

    def gen_fn_def(
        self,
        free_vars: list[str],
        maybe_return_annotation: str,
        *,
        expanded_def: bool = False,
    ) -> str:
        fn_args = self.orig_arg_names
        has_orig_self = (fn_args[0] == "self") if len(fn_args) > 0 else False
        if has_orig_self:
            free_vars.insert(0, "self")
        # Rename any non-first `self` in fn_args to a unique name. The base
        # CodeGen.gen_fn_def prepends `"self"` for the GraphModule's bound-
        # method receiver whenever fn_args[0] != "self", which collides with
        # a schema param literally named `self` (e.g. `aten.where.self(Tensor
        # cond, Tensor self, Tensor other)` -> `def forward(self, cond, self,
        # other)` -> `SyntaxError: duplicate argument 'self' in function
        # definition`). Both the function-def arg list (via super().gen_fn_def)
        # and the body binding (via gen_var_bindings) reference fn_args, so
        # the rename is consistent end-to-end.
        fn_args = list(fn_args)
        first_pos = 1 if has_orig_self else 0
        for i in range(first_pos, len(fn_args)):
            if fn_args[i] == "self":
                new_name = "self_"
                while new_name in fn_args:
                    new_name += "_"
                fn_args[i] = new_name
        fn_definition = super().gen_fn_def(
            fn_args[:], maybe_return_annotation, expanded_def=expanded_def
        )

        if len(free_vars) > 0:  # pytree has placeholders in it
            fn_definition += self.gen_var_bindings(fn_args, free_vars, expanded_def)
        return fn_definition

    def gen_var_bindings(
        self, fn_args: list[str], free_vars: list[str], expanded_def: bool
    ) -> str:
        without_annotation = [x.split(":")[0].split("#")[0] for x in free_vars]
        if len(fn_args) == 0:
            fn_signature = ""
        elif len(fn_args) == 1:
            fn_signature = f"{fn_args[0]}, "
        else:
            fn_signature = f"{', '.join(fn_args)}"
        return f"""
    _fn_args = ({fn_signature})
    {", ".join(without_annotation)}, = self._dynamo_bytecode_flatten(*_fn_args)"""

    def generate_output(
        self,
        output_args: torch.fx.node.Argument,
        *,
        descs: object | None = None,
        repr_fn: Any | None = None,
    ) -> str:
        if repr_fn is None:
            repr_fn = repr
        # pyrefly: ignore [not-iterable]
        returned = f"self._dynamo_bytecode_unflatten(({', '.join([repr_fn(a) for a in output_args])},), _fn_args)"
        if self.wrap_tuple:
            returned = f"({returned},)"
        return f"return {returned}"


def dynamo_graph_capture_for_export(
    fn: Callable[..., Any],
    constraints: list[Constraint] | None = None,
) -> Callable[..., Any]:
    if isinstance(fn, torch._ops.OpOverload):

        def default_annotation(arg: torch.Argument) -> str:
            if arg.has_default_value():
                return f"={arg.default_value!r}"
            return ""

        has_kwarg_only = False
        arg_list = []
        for arg in fn._schema.arguments:
            if arg.kwarg_only and not has_kwarg_only:
                has_kwarg_only = True
                arg_list.append("*")
            arg_list.append(arg.name + default_annotation(arg))
        func_str = f"""
def op_overload_wrapper({", ".join(arg_list)}):
    return op({", ".join([f"{arg.name}={arg.name}" for arg in fn._schema.arguments])})
"""
        out = {}
        exec(func_str, {"op": fn}, out)
        fn = out["op_overload_wrapper"]  # type: ignore[assignment]

    def inner(*args: Any, **kwargs: Any) -> Any:
        if torch._dynamo.config.install_free_tensors:
            raise AssertionError("install_free_tensors must be False")
        with (
            _compiling_state_context(),
            torch._dynamo.config.patch(
                replay_side_effects=False, side_effect_replay_policy="warn"
            ),
            get_metrics_context(),
            dynamo_timed("fullgraph_capture"),
        ):
            out = fullgraph_capture(
                fn,
                args,
                kwargs,
                constraints=constraints,
            )
        graph_module = create_fx_graph_from_captured_output(out, fn, args, kwargs)
        return graph_module

    return inner


def _walk_spec(
    user_spec: IntermediateSpec | None,
    arg_value: Any,
    out_leaf_specs: list[IntermediateSpec | None],
    flat_idx: int,
    where: str,
) -> int:
    """Walk ``(user_spec, arg_value)`` pairwise, writing leaf specs into
    ``out_leaf_specs`` starting at ``flat_idx``, and return the **count
    of flat leaves consumed** by this subtree.

    Container specs (``SeqSpec`` / ``DictSpec`` / ``ObjectSpec``) walk in
    parallel with the runtime value's pytree.

    Flat-index alignment with the export tracer
    -------------------------------------------
    Each recursive case visits children in exactly the same order
    ``pytree.tree_flatten`` does, guaranteed structurally:

    - **list / tuple** (``SeqSpec`` or no-spec subtree): both this
      walker and pytree's ``_list_flatten`` / ``_tuple_flatten``
      iterate via plain ``enumerate(seq)``.
    - **dict** (``DictSpec`` or no-spec subtree): this walker
      iterates ``arg_value.items()`` (insertion order); pytree's
      ``_dict_flatten`` does ``list(d.values())``, same order.
      *Note*: we walk the runtime dict (not the spec dict), so spec
      keys may be listed in any order — they're looked up by name.
    - **pytree-registered objects** (``ObjectSpec``): this walker calls
      the type's own registered ``flatten_with_keys_fn`` directly
      (same function pytree itself dispatches to). Order is identical
      by construction.
    - **leaf spec**: contributes exactly one slot.
    - **no spec (None)**: contributes ``len(pytree.tree_leaves(value))``
      — same count pytree will emit for that subtree.

    ``where`` is a human-readable path string used solely for error
    messages; it accumulates as the walker descends (e.g.
    ``"ParamsSpec entry for forward param 'd'['a'][0]"``).
    """
    # No spec for this subtree (top-level None or unspecified slot) ⇒
    # everything below it is static; count its flat leaves without
    # writing anything (slots are pre-initialized to None).
    if user_spec is None:
        return len(pytree.tree_leaves(arg_value))

    if isinstance(user_spec, SeqSpec):
        if not isinstance(arg_value, (list, tuple)):
            raise ValueError(
                f"{where}: SeqSpec expected list/tuple, got {type(arg_value).__name__}"
            )
        entries = user_spec._entries
        if len(entries) > len(arg_value):
            raise ValueError(
                f"{where}: SeqSpec has {len(entries)} entries beyond runtime "
                f"sequence length {len(arg_value)}"
            )
        consumed = 0
        for i, value in enumerate(arg_value):
            sub_spec = entries[i] if i < len(entries) else None
            consumed += _walk_spec(
                sub_spec,
                value,
                out_leaf_specs,
                flat_idx + consumed,
                where=f"{where}[{i}]",
            )
        return consumed

    if isinstance(user_spec, DictSpec):
        if not isinstance(arg_value, dict):
            raise ValueError(
                f"{where}: DictSpec expected dict, got {type(arg_value).__name__}"
            )
 
        unmatched = set(user_spec) - set(arg_value)
        if unmatched:
            raise ValueError(
                f"{where}: DictSpec has entries {sorted(unmatched)!r} "
                f"that do not match any key in the runtime dict. "
                f"Runtime keys: {sorted(arg_value.keys())!r}"
            )
        # Walk runtime ordering so flat positions align with
        # pytree.tree_flatten (insertion order for plain dicts).
        consumed = 0
        for key, value in arg_value.items():
            sub_spec = user_spec._entries[key] if key in user_spec else None
            consumed += _walk_spec(
                sub_spec,
                value,
                out_leaf_specs,
                flat_idx + consumed,
                where=f"{where}[{key!r}]",
            )
        return consumed

    if isinstance(user_spec, ObjectSpec):
        # By the time we get here, export has already validated that
        # arg_value is pytree-flattenable (its top-level input check
        # rejects unregistered types). The only ObjectSpec-specific
        # requirement is that the registered handler also expose a
        # `flatten_with_keys_fn` so we can address children by
        # attribute name.
        node_type = pytree._get_node_type(arg_value)
        handler = pytree.SUPPORTED_NODES.get(node_type)
        if handler is None:
            raise ValueError(
                f"{where}: ObjectSpec requires the runtime value's type "
                f"to be pytree-registered, but {type(arg_value).__name__} "
                f"is not registered. Register it via "
                f"`torch.export.register_dataclass(<cls>)` (for dataclasses) "
                f"or `pytree.register_pytree_node(...)`."
            )
        if handler.flatten_with_keys_fn is None:
            # Note: this requirement is not ObjectSpec-specific — plain
            # export() and the legacy dynamic_shapes API also fail on
            # types registered without a `flatten_with_keys_fn` (their
            # input-path construction uses `tree_flatten_with_path`).
            # We just catch it earlier here with a clearer message.
            raise ValueError(
                f"{where}: export requires "
                f"`flatten_with_keys_fn` to be registered for type "
                f"{type(arg_value).__name__}, but none was found. "
                f"Re-register via "
                f"`pytree.register_pytree_node(..., flatten_with_keys_fn=...)` "
                f"or use `torch.export.register_dataclass` for dataclasses."
            )
        key_children, _ = handler.flatten_with_keys_fn(arg_value)
        # Fail-fast on unmatched attrs before recursing. Non-attribute
        # keys (SequenceKey / MappingKey) contribute no matchable names.
        available_names = {
            ke.name
            for ke, _ in key_children
            if isinstance(ke, pytree.GetAttrKey)
        }
        unmatched = set(user_spec) - available_names
        if unmatched:
            raise ValueError(
                f"{where}: ObjectSpec has entries {sorted(unmatched)!r} "
                f"that do not match any attribute on the runtime object "
                f"of type {type(arg_value).__name__}. Available "
                f"attributes: {sorted(available_names)!r}"
            )
        consumed = 0
        for key_entry, child in key_children:
            # Only ``GetAttrKey`` entries can match an ObjectSpec entry
            # (ObjectSpec addresses by attribute name); any other key
            # shape contributes a static subtree.
            if (
                isinstance(key_entry, pytree.GetAttrKey)
                and key_entry.name in user_spec
            ):
                consumed += _walk_spec(
                    user_spec._fields[key_entry.name],
                    child,
                    out_leaf_specs,
                    flat_idx + consumed,
                    where=f"{where}.{key_entry.name}",
                )
            else:
                # No spec for this attribute — count its leaves as static.
                consumed += len(pytree.tree_leaves(child))
        return consumed

    # Leaf spec — single flat slot. Type-check the leaf spec against
    # the runtime value, then write it.
    if isinstance(user_spec, TensorSpec):
        if not isinstance(arg_value, torch.Tensor):
            raise ValueError(
                f"{where}: spec is TensorSpec but the actual arg is "
                f"{type(arg_value).__name__}, not a Tensor."
            )
    elif isinstance(user_spec, (IntVar, int)):
        # Scalar spec — arg must be a Python int, a SymInt, or the
        # export-internal `_IntWrapper` (export wraps user ints in
        # `_IntWrapper` upstream via `pytree.tree_map_only(int, ...)`).
        if not isinstance(arg_value, (int, torch.SymInt, _IntWrapper)):
            raise ValueError(
                f"{where}: spec is {type(user_spec).__name__} "
                f"(scalar spec) but the actual arg is "
                f"{type(arg_value).__name__}, not int/SymInt."
            )
    else:
        raise AssertionError(
            f"{where}: unexpected leaf spec type {type(user_spec).__name__}"
        )
    out_leaf_specs[flat_idx] = user_spec
    return 1


def _flatten_shapes_spec(
    f: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None,
    shapes_spec: ShapesSpec,
) -> ShapesSpec:
    """Re-key a user-provided ``ShapesSpec`` for export's intermediate
    trace module.

    Export traces an intermediate module (``ModuleToTrace``) instead of
    the user's original module. The intermediate module's forward
    signature is ``forward(*flat_args)`` — a single varargs that holds
    all original inputs flattened. Concretely (see ``ModuleToTrace``)::

        def forward(self, *flat_args):
            # Reconstructs original (args, kwargs) from the flat list.
            args, kwargs = pytree.tree_unflatten(flat_args, self.in_spec)
            # User's module is called with its ORIGINAL signature.
            return self._export_root(*args, **kwargs)

    The *entry point* dynamo traces is the wrapper above — so every input
    source dynamo tracks is rooted at a positional ``flat_args[i]``. The
    user, however, wrote their spec against the *original* module's
    parameter names. This function rewrites the spec so it targets the
    intermediate module's flat layout ``flat_args``.

    """
    params_spec = shapes_spec._params
    kwargs = kwargs or {}
    if params_spec is None:
        # nothing to do all static no spec needed.
        return shapes_spec

    params_spec_named_args = params_spec._named_args
    params_spec_varargs = params_spec._varargs  # may be None

    assert isinstance(f, torch.nn.Module), (  # noqa: S101
        "_flatten_shapes_spec only supports nn.Module (the only thing "
        "torch.export.export accepts)."
    )
    sig = inspect.signature(f.forward)
    pos_params = list(sig.parameters.values())

    # The user's signature has up to four logical regions of parameters
    # (in this order). "Bound from" = where each signature param gets its
    # runtime value when the user makes the call.
    #   1) named-positional: POSITIONAL_ONLY or POSITIONAL_OR_KEYWORD
    #      params before `*args`. Bound from user's positional `args[i]`
    #      `args[i]` where i < varargs_idx. Spec lookup:
    #      `user._named_args[name]` using `pos_params[i].name`.
    #   2) varargs: VAR_POSITIONAL (`*args`). Bound from user's positional
    #      `args[i]` where i >= varargs_idx. Spec lookup:
    #      `user._varargs[i - varargs_idx]`.
    #   3) kwargs: KEYWORD_ONLY params (also POSITIONAL_OR_KEYWORD ones
    #      passed by name) and VAR_KEYWORD (`**kwargs`). All arrive in
    #      user's `kwargs[name]` and are handled by one loop. Spec lookup:
    #      `user._named_args[name]` first; falls back to
    #      `user._varkw[name]` for names not declared as named params.
    # Python's call grammar guarantees the user's call layout is always
    # `[positionals][kwargs]` — never interleaved (`func(a, b=c, d)` is a
    # SyntaxError). So we walk `args` first (regions 1 + 2 below), then
    # `kwargs.items()` (regions 3 + 4, handled in the kwargs loop further
    # down).
    #
    # `varargs_idx` is the position of `*args` in the signature's parameter
    # list (or `len(pos_params)` if the signature has no `*args` — sentinel
    # meaning "no varargs region; everything the user passes positionally is
    # named"). We use it as a slicing boundary on `args`:
    #   args[:varargs_idx]  → named-positional region
    #   args[varargs_idx:]  → varargs region
    #
    # Example: signature `def forward(self, x, y, *args, **kwargs)`, user
    # calls `mod(T1, T2, T3, T4, foo=T5, bar=T6)`:
    #   pos_params = [x, y, *args, **kwargs]    varargs_idx = 2
    #   args   = (T1, T2, T3, T4)
    #   kwargs = {"foo": T5, "bar": T6}
    #   args[:2] = (T1, T2)        → named-positional loop (x, y)
    #   args[2:] = (T3, T4)        → varargs loop (*args[0], *args[1])
    #   kwargs.items()             → kwargs loop (foo, bar) handled separately
    varargs_idx = len(pos_params)  # default: no `*args` in the signature
    for i, p in enumerate(pos_params):
        if p.kind is inspect.Parameter.VAR_POSITIONAL:
            varargs_idx = i
            break

    # Walk the user's actual call structure.
    _, in_spec = pytree.tree_flatten((args, kwargs))
    total_leaves = in_spec.num_leaves
    # out_leaf_specs[i] is the leaf-spec for flat_args[i] (or None for static).
    # Length matches pytree.tree_flatten((args, kwargs))[0]. It's keyed under
    # "*args" in the returned ParamsSpec because the intermediate
    # ModuleToTrace.forward(*flat_args) only has a varargs signature.
    out_leaf_specs: list[IntermediateSpec | None] = [None] * total_leaves

    flat_idx = 0

    # Track which named / **kwargs spec entries actually get bound to an
    # input. Anything left over at the end is an entry that matched no
    # passed argument (typo or a spec for an omitted defaulted param), which
    # we reject below rather than silently ignore.
    matched_named_keys: set[str] = set()
    matched_varkw_keys: set[str] = set()

    # Loop 1: named-positional region. Look up each arg's spec by the
    # corresponding signature param name in `params_spec_named_args`.
    for i, arg_value in enumerate(args[:varargs_idx]):
        arg_name = pos_params[i].name
        # Note: distinguish "key absent" from "key present with value None"
        # (the latter means the user explicitly marked this arg static).
        # Both skip spec binding, but the explicit-None form still counts
        # as matched so it isn't reported in `unmatched` below.
        if arg_name in params_spec_named_args:
            matched_named_keys.add(arg_name)
            user_spec = params_spec_named_args[arg_name]
        else:
            user_spec = None
        consumed = _walk_spec(
            user_spec,
            arg_value,
            out_leaf_specs,
            flat_idx,
            where=f"shapes_spec[{arg_name!r}]",
        )
        flat_idx += consumed

    # Pad params_spec_varargs to match the actual `*args` count, filling
    # missing tail entries with None ("static"). Lets Loop 2 below index
    # uniformly without a bounds check.
    n_actual_varargs = len(args) - varargs_idx
    params_spec_varargs = list(params_spec_varargs or [])
    params_spec_varargs += [None] * max(0, n_actual_varargs - len(params_spec_varargs))

    # Loop 2: varargs region. Look up each arg's spec by its position
    # within `*args` in `params_spec_varargs`.
    for user_idx, arg_value in enumerate(args[varargs_idx:]):
        user_spec = params_spec_varargs[user_idx]
        consumed = _walk_spec(
            user_spec,
            arg_value,
            out_leaf_specs,
            flat_idx,
            where=f"shapes_spec['*args'][{user_idx}]",
        )
        flat_idx += consumed

    # Loop 3: kwargs region. Each kwarg's spec can come from either:
    #   (a) `params_spec_named_args[name]` — when the kwarg matches a named param
    #       in the signature (e.g. user passed `mod(foo=T1)` to a forward
    #       that has `def forward(self, foo)`).
    #   (b) `params_spec_varkw[name]` — when the kwarg flows through the
    #       signature's `**kwargs` slot (e.g. `def forward(self, **kwargs)`
    #       called with `mod(foo=T1)`, user spec `{"**kwargs": {"foo": ...}}`).
    params_spec_varkw = params_spec._varkw  # may be None
    for arg_name, arg_value in kwargs.items():
        if arg_name in params_spec_named_args:
            user_spec = params_spec_named_args[arg_name]
            matched_named_keys.add(arg_name)
        elif params_spec_varkw is not None and arg_name in params_spec_varkw:
            user_spec = params_spec_varkw[arg_name]
            matched_varkw_keys.add(arg_name)
        else:
            user_spec = None
        consumed = _walk_spec(
            user_spec,
            arg_value,
            out_leaf_specs,
            flat_idx,
            where=f"shapes_spec[{arg_name!r}]",
        )
        flat_idx += consumed

    # Every named / **kwargs spec entry must bind to an argument that was
    # actually passed to export(). A leftover entry is almost always a
    # mistake (a misspelled parameter name, or a spec for a defaulted param
    # the caller omitted), so error out instead of silently dropping it.
    unmatched = set(params_spec_named_args) - matched_named_keys
    if params_spec_varkw is not None:
        unmatched |= set(params_spec_varkw) - matched_varkw_keys
    if unmatched:
        n_named_positional = len(args[:varargs_idx])
        passed = [p.name for p in pos_params[:n_named_positional]] + list(kwargs)
        raise ValueError(
            f"ParamsSpec has entries {sorted(unmatched)!r} that do not match "
            f"any argument passed to export(). Spec keys must be forward "
            f"parameter names that were actually passed. Inputs received: "
            f"{passed!r}."
        )

    # Sanity check: summing leaves per-arg must equal pytree.tree_flatten
    # on (args, kwargs) as a whole. A mismatch means our per-arg walk
    # drifted from the tracer's flatten layout.
    if flat_idx != total_leaves:
        raise AssertionError(
            f"_flatten_shapes_spec leaf-count drift: walked {flat_idx} leaves "
            f"but pytree.tree_flatten((args, kwargs)) yields {total_leaves}. "
            f"This means the translator and the export tracer disagree on the "
            f"flat input layout."
        )

    # Carry assumptions / derived expressions through unchanged: this
    # function only re-keys *which positional slot* each TensorSpec lands
    # in; the ShapeVar/IntVar symbols inside the assumption expressions
    # are the same Python objects, so the assumptions remain valid.
    return ShapesSpec(
        ParamsSpec({"*args": out_leaf_specs}),
        assumptions=shapes_spec._assumptions or None,
    )


def _dynamo_graph_capture_for_export(
    mod: Callable[..., Any],
    *,
    constraints: list[Constraint] | None = None,
    dynamic_shapes: _DynamicShapesInput = None,
) -> Callable[..., torch.fx.GraphModule]:
    """
    Improved dynamo graph capture using transformer approach with proper fake tensor handling.

    This function creates a capture instance that handles:
    1. PyTree flattening/unflattening with proper input ordering
    2. Dynamo graph capture with export-specific context
    3. FX graph transformation for export compatibility
    4. Proper fake tensor metadata preservation
    5. Dynamic dimension constraint handling

    Notable improvements over manual approach:
    - Uses FX Transformer for cleaner graph manipulation
    - Properly handles fake tensor metadata and dynamic dimensions
    - Preserves all necessary metadata for export
    - More robust error handling and edge case management

    TODO:
    1. Are we actually gonna run the bytecode?
    2. Need to attach guards
    """

    _constraints = constraints

    def inner(*args: Any, **kwargs: Any) -> torch.fx.GraphModule:
        # This sets the is_exporting flag when building guards.
        with _compiling_state_context():
            flat_inputs, in_spec = pytree.tree_flatten((args, kwargs))
            check_user_input_output(flat_inputs, UserErrorType.INVALID_INPUT)
            module_to_trace = ModuleToTrace(mod, in_spec)
            orig_callable = mod.forward if isinstance(mod, torch.nn.Module) else mod

            constraints: list[Constraint] | None = _constraints

            from . import reset  # type: ignore[attr-defined]

            reset()

            dynamo_config_ctx = torch._dynamo.config.patch(
                specialize_int=True,
                specialize_float=True,
                assume_static_by_default=True,
                automatic_dynamic_shapes=False,
                capture_dynamic_output_shape_ops=True,
                capture_scalar_outputs=True,
                constant_fold_autograd_profiler_enabled=True,
                log_graph_in_out_metadata=True,
                # install_free_tensors ensures that params and buffers are still
                # added as graph attributes, and makes Dynamo emits graphs that
                # follow export pytree-able input requirements In future, if we
                # fully rely on bytecode for the runtime, we can turn this flag
                # off.
                install_free_tensors=torch._dynamo.config.install_free_tensors_for_export,
            )

            # If `dynamic_shapes` is a ShapesSpec/ParamsSpec, auto-wrap
            # ParamsSpec → ShapesSpec, flatten into the (args, kwargs) layout
            # the tracer builds above, and expose it via
            # `torch._dynamo.config._shapes_spec` for the variable builder.
            shapes_spec_in_use = False
            shapes_spec_ctx = nullcontext()
            if isinstance(dynamic_shapes, (ShapesSpec, ParamsSpec)):
                shapes_spec_in_use = True
                user_spec = (
                    ShapesSpec(dynamic_shapes)
                    if isinstance(dynamic_shapes, ParamsSpec)
                    else dynamic_shapes
                )
                flattened_spec = _flatten_shapes_spec(mod, args, kwargs, user_spec)
                shapes_spec_ctx = torch._dynamo.config.patch(
                    _shapes_spec=flattened_spec
                )

            with (
                get_metrics_context(),
                dynamo_timed("fullgraph_capture"),
                dynamo_config_ctx,
                shapes_spec_ctx,
            ):
                out = fullgraph_capture(
                    module_to_trace,
                    tuple(flat_inputs),
                    constraints=_constraints,
                    _is_export_deprecated_do_not_use=True,
                )

                if out.graph_capture_output.output_graph is None:
                    raise AssertionError(
                        "output_graph must not be None after fullgraph_capture"
                    )

                example_inputs: list[Any] = []
                if out.backend_input is not None:
                    graph = out.backend_input.graph_module
                    fake_mode = out.backend_input.fake_mode
                    example_inputs = out.backend_input.example_inputs
                else:
                    graph = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
                    graph.graph.output(None)
                    graph.recompile()
                    fake_mode = None

                # ShapesSpec has its own export-time soundness check and uses
                # unbacked SymInts, so the legacy guard-based violation
                # detection / prettifier is not applicable on this path.
                if not shapes_spec_in_use:
                    _suggest_or_raise_constraint_violation(
                        module_to_trace,
                        orig_callable,
                        fake_mode,
                        out,
                        args,
                        kwargs,
                        dynamic_shapes,  # type: ignore[arg-type]
                    )

                # Extract export metadata from the new location
                export_metadata = out.graph_capture_output.output_graph.export_metadata
                graph_inputs = export_metadata.graph_input_idx_to_local_source
                graph_output_map = export_metadata.output_return_type
                out_spec = export_metadata.out_spec
                module_call_spec = export_metadata.module_call_spec

            # Compute dynamic dimensions for each input based on constraints
            flat_args_dynamic_dims = [
                {
                    c.dim
                    for c in (constraints or ())
                    if (
                        c.t_id == id(x)
                        and not isinstance(c, _RelaxedConstraint)
                        and c.constraint_range.vr.lower != c.constraint_range.vr.upper
                    )
                }
                for x in flat_inputs
            ]

            # Create input order mapping from dynamo's internal order to user order
            # Only process inputs that come from function arguments (GetItemSource).
            # Skip inputs that come from other sources like closures (e.g., captured
            # opaque objects like DeviceMesh).
            graph_input_order: dict[int, int] = {}
            for inp in graph_inputs:
                source = graph_inputs[inp]
                if isinstance(source, torch._dynamo.source.GetItemSource):
                    graph_input_order[source.index] = len(graph_input_order)

            for real_idx, graph_idx in graph_input_order.items():
                flat_inputs[real_idx] = example_inputs[graph_idx]

            # Use FX transformer to rebuild the graph cleanly
            transformed_graph = DynamoGraphTransformer(
                graph,
                flat_inputs,
                flat_args_dynamic_dims,
                graph_input_order,
                graph_output_map,
                fake_mode,
                graph_inputs,
            ).transform()

            # Set up PyTree codegen for proper input/output handling
            transformed_graph.graph._codegen = _PyTreeCodeGen(
                _PyTreeInfo(
                    argument_names(inspect.signature(orig_callable), args, kwargs),  # type: ignore[attr-defined, arg-type]
                    in_spec,
                    out_spec,
                )
            )
            transformed_graph.recompile()

            clean_nn_module_stack_and_source_fn(transformed_graph, True)
            clean_export_root(transformed_graph)

            transformed_graph.meta["module_call_specs"] = module_call_spec
            transformed_graph.meta["fake_mode"] = fake_mode

            return transformed_graph

    return inner
