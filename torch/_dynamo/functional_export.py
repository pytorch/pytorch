import inspect
import logging
import traceback
from collections import namedtuple
from typing import Any, Callable, Optional, TYPE_CHECKING, Union

import sympy

import torch
import torch.fx
import torch.utils._pytree as pytree
from torch._dynamo.convert_frame import CaptureOutput, fullgraph_capture, get_traced_fn
from torch._dynamo.eval_frame import argument_names
from torch._dynamo.utils import dynamo_timed, get_metrics_context
from torch._export.utils import _compiling_state_context
from torch.export.dynamic_shapes import _RelaxedConstraint, Constraint
from torch.fx import Node
from torch.fx.experimental.symbolic_shapes import (
    ConstraintViolationError,
    DimDynamic,
    StatelessSymbolicContext,
)
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo


if TYPE_CHECKING:
    from torch._subclasses.fake_tensor import FakeTensorMode


log = logging.getLogger(__name__)


def post_process_error_msg(
    constraint_violation_error: ConstraintViolationError,
    func: Callable[..., Any],
    args: Any,
    kwargs: Any,
):
    """
    Because we trace a different callable, the sources are all messed up.
    Manually patch them so the error message looks correct.
    """
    from torch.export._unlift import _get_input_paths, _replace_sources

    orig_sig = inspect.signature(func)
    flat_input_paths = _get_input_paths((args, kwargs), orig_sig)
    constraint_violation_error.args = (
        _replace_sources(constraint_violation_error.args[0], flat_input_paths),
    )
    return constraint_violation_error


def clean_nn_module_stack(
    graph_module: torch.fx.GraphModule, is_inline_builtin=False
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
    for node in graph_module.graph.nodes:
        if "nn_module_stack" not in node.meta:
            continue

        nn_module_stack = node.meta["nn_module_stack"].copy()

        if "L__self____export_root" in nn_module_stack:
            del nn_module_stack["L__self____export_root"]

        # Clean up remaining entries
        cleaned_stack = {}
        for key, (child_name, child_class) in nn_module_stack.items():
            # Clean key by removing export_root patterns
            clean_key = key.replace("__modules['_export_root']_", "").replace(
                "__export_root_", ""
            )

            # Clean child_name by removing export_root patterns
            clean_name = child_name.replace("._modules['_export_root']", "").replace(
                "._export_root", ""
            )

            # Skip self reference for inline builtin case
            if is_inline_builtin and clean_name == "L['self']":
                continue

            cleaned_stack[clean_key] = (clean_name, child_class)

        node.meta["nn_module_stack"] = cleaned_stack

    return graph_module


def clean_export_root(graph_module: torch.fx.GraphModule) -> None:
    """Remove export_root artifacts from FX graph in-place"""

    # Clean parameter names: L__self____export_root_param -> L__self___param
    def clean_name(name) -> str:
        if "____modules___export_root_" in name:
            return name.replace("____modules___export_root_", "_")
        if "__export_root_" in name:
            return name.replace("__export_root_", "_")
        return name

    # Unlike getattr node, call_module can be invoked multiple times
    # In those cases, we should fix all invocations of call_module
    clean_named_module_map: dict[str, str] = {}

    # Update get_attr nodes in-place
    for node in graph_module.graph.nodes:
        if node.op == "get_attr":
            old_target = node.target
            new_target = clean_name(old_target)
            if new_target != old_target:
                node.target = new_target
                assert hasattr(graph_module, old_target)
                # Move the parameter to the new name
                param = torch.fx.graph_module._get_attr(graph_module, old_target)
                torch.fx.graph_module._assign_attr(param, graph_module, new_target)
                torch.fx.graph_module._del_attr(graph_module, old_target)
        # Dynamo will only have one nested level
        if node.op == "call_module":
            old_target = node.target
            new_target = clean_name(old_target)
            new_name = clean_name(node.name)
            if new_target == old_target:
                continue

            # if this module has already been cleaned before, just lookup from map.
            if old_target in clean_named_module_map:
                node.target = clean_named_module_map[old_target]
                node.name = new_name
                continue
            assert isinstance(old_target, str)
            assert isinstance(new_target, str)
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

    def forward(self, *flat_args: Any) -> "ExportTracerOutput":
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
        graph_output_map: dict[int, tuple[str, Any]],
        fake_mode: Optional[Any] = None,
    ) -> None:
        super().__init__(module)

        assert len(flat_args_dynamic_dims) == len(flat_inputs)

        self.flat_inputs = flat_inputs
        self.flat_args_dynamic_dims = flat_args_dynamic_dims
        self.graph_input_order = graph_input_order
        self.graph_output_map = graph_output_map
        self.fake_mode = fake_mode

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

            # pyrefly: ignore  # unsupported-operation
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

    def placeholder(self, target, args, kwargs) -> Any:
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
            # Shouldn't happen if mapping is correct, but fallback
            return super().placeholder(target, args, kwargs)

    def output(self, target, args, kwargs) -> Any:
        """Transform output according to graph_output_map."""
        original_outputs = args[0]

        # Build new output list based on graph_output_map
        new_outputs = []
        for i in sorted(self.graph_output_map.keys()):
            output_type, val = self.graph_output_map[i]

            if output_type == "graph_out":
                new_outputs.append(original_outputs[val])
            elif output_type == "input":
                input_idx = val.index
                new_outputs.append(self.new_input_nodes[input_idx])
            elif output_type == "constant":
                new_outputs.append(val)

        return super().output(target, (tuple(new_outputs),), {})

    def run_node(self, node: Node) -> Any:
        """Run node transformation and preserve metadata."""
        self.current_node = node
        result = super().run_node(node)

        # Copy important metadata
        if hasattr(result, "node") and result.node is not node:
            for key in ["val", "example_value", "unbacked_bindings"]:
                if key in node.meta:
                    result.node.meta[key] = node.meta[key]

            # Preserve node names (except output)
            if node.op != "output" and hasattr(node, "name"):
                result.node._rename(node.name)

        return result

    def transform(self) -> torch.fx.GraphModule:
        """Perform the graph transformation and copy module metadata."""
        result_gm = super().transform()

        # Copy module metadata like the original implementation
        if hasattr(self.module, "meta"):
            # pyrefly: ignore  # unsupported-operation
            if "dynamo_flat_name_to_original_fqn" in self.module.meta:
                # pyrefly: ignore  # index-error
                result_gm.meta["dynamo_flat_name_to_original_fqn"] = self.module.meta[
                    # pyrefly: ignore  # index-error
                    "dynamo_flat_name_to_original_fqn"
                ]
            # pyrefly: ignore  # unsupported-operation
            if "dynamo_compile_id" in self.module.meta:
                # pyrefly: ignore  # index-error
                result_gm.meta["dynamo_compile_id"] = self.module.meta[
                    # pyrefly: ignore  # index-error
                    "dynamo_compile_id"
                ]

        return result_gm


def _suggest_or_raise_constraint_violation(
    module_to_trace: torch.nn.Module,
    orig_callable: Callable,  # type: ignore[type-arg]
    fake_mode: Optional["FakeTensorMode"],
    graph_capture_output: CaptureOutput,
    args: Any,
    kwargs: Any,
    dynamic_shapes: Optional[Union[dict[str, Any], tuple[Any], list[Any]]],
):
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
        # pyrefly: ignore  # unbound-name
        dim_constraints.solve()
        # pyrefly: ignore  # unbound-name
        forced_specializations = dim_constraints.forced_specializations()
        # pyrefly: ignore  # unbound-name
        msg = dim_constraints.prettify_results(
            inspect.signature(orig_callable),  # type: ignore[attr-defined]
            dynamic_shapes,
            constraint_violation_error,
            forced_specializations,
        )
        if constraint_violation_error:
            constraint_violation_error.args = (
                constraint_violation_error.args[0] + msg,
            )
        else:
            if forced_specializations:
                constraint_violation_error = ConstraintViolationError(msg)
            else:
                log.info(
                    "Summary of dimension constraints:%s",
                    msg,
                )

        # Error if we have any constraints on static values
        # pyrefly: ignore  # unbound-name
        for k in shape_env.var_to_range.keys():
            if isinstance(k, sympy.Integer):
                constraint_violation_error = ConstraintViolationError(
                    # pyrefly: ignore  # unbound-name
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


def _dynamo_graph_capture_for_export(
    mod: Callable[..., Any],
    *,
    constraints: Optional[list[Constraint]] = None,
    dynamic_shapes: Optional[Union[dict[str, Any], tuple[Any], list[Any]]] = None,
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

    _dynamic_shapes = dynamic_shapes
    _constraints = constraints

    def inner(*args: Any, **kwargs: Any) -> torch.fx.GraphModule:
        # This sets the is_exporting flag when building guards.
        with _compiling_state_context():
            flat_inputs, in_spec = pytree.tree_flatten((args, kwargs))
            module_to_trace = ModuleToTrace(mod, in_spec)
            orig_callable = mod.forward if isinstance(mod, torch.nn.Module) else mod

            constraints: Optional[list[Constraint]] = _constraints
            dynamic_shapes: Optional[Union[dict[str, Any], tuple[Any], list[Any]]] = (
                _dynamic_shapes
            )

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
            )

            with (
                get_metrics_context(),
                dynamo_timed("fullgraph_capture"),
                dynamo_config_ctx,
            ):
                out = fullgraph_capture(
                    module_to_trace,
                    tuple(flat_inputs),
                    constraints=_constraints,
                    _is_export_deprecated_do_not_use=True,
                )

                assert out.graph_capture_output.output_graph is not None

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

                _suggest_or_raise_constraint_violation(
                    module_to_trace,
                    orig_callable,
                    fake_mode,
                    out,
                    args,
                    kwargs,
                    dynamic_shapes,
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
            graph_input_order: dict[int, int] = {}
            for inp in graph_inputs:
                source = graph_inputs[inp]
                assert isinstance(source, torch._dynamo.source.GetItemSource)
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

            clean_nn_module_stack(
                transformed_graph, torch._dynamo.config.inline_inbuilt_nn_modules
            )
            clean_export_root(transformed_graph)

            transformed_graph.meta["module_call_specs"] = module_call_spec
            transformed_graph.meta["fake_mode"] = fake_mode

            return transformed_graph

    return inner
