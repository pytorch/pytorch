"""
This module contains classes and utilities for handling higher-order operators in Dynamo.
It provides functionality for tracing and transforming control flow constructs like
conditions (torch.cond), loops (torch.while_loop), maps (torch.ops.higher_order.map),
and other higher-order operations.

The module includes specialized VariableTracker classes for different types of
higher-order operations, along with utilities for:
- Speculating and capturing subgraphs
- Managing control flow
- Handling autograd function applications
- Supporting function transformations
- Processing activation checkpoints

These classes work together to enable Dynamo to correctly trace and compile code
containing complex control flow patterns and higher-order functions while preserving
their semantic behavior.
"""

import contextlib
import copy
import functools
import inspect
import itertools
import logging
import traceback
import types
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast, Literal, Optional, TYPE_CHECKING, Union

import torch._C
import torch.fx
import torch.nn
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.utils import _make_inlined, get_fake_value
from torch._dynamo.variables.constant import CONSTANT_VARIABLE_NONE, ConstantVariable
from torch._dynamo.variables.ctx_manager import RepararametrizeModuleContextVariable
from torch._dynamo.variables.functions import UserFunctionVariable
from torch._dynamo.variables.nn_module import UnspecializedNNModuleVariable
from torch._dynamo.variables.script_object import TorchScriptObjectVariable
from torch._dynamo.variables.tensor import SymNodeVariable, TensorVariable
from torch._guards import Source
from torch._higher_order_ops.invoke_subgraph import NestedCompileRegionOptions
from torch._ops import HigherOrderOperator
from torch.fx.graph_module import GraphModule
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.fx.proxy import Proxy
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils import _pytree as pytree
from torch.utils._ordered_set import OrderedSet

from ... import graph_break_hints, variables
from ...exc import (
    ObservedException,
    UncapturedHigherOrderOpError,
    unimplemented,
    Unsupported,
)
from ...source import AttrSource, DictGetItemSource
from ...utils import proxy_args_kwargs, set_example_value
from ..base import VariableTracker
from ..dicts import ConstDictVariable
from ..lazy import LazyVariableTracker
from ..lists import ListVariable, TupleVariable


if TYPE_CHECKING:
    from torch._dynamo.output_graph import SubgraphTracer
    from torch._dynamo.symbolic_convert import InstructionTranslator
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
    from .control_flow import (
        WhileLoopHigherOrderVariable,
        WhileLoopStackOutputHigherOrderVariable,
    )

from collections.abc import Generator, Iterable
from typing import ParamSpec, TypeVar


P = ParamSpec("P")
R = TypeVar("R")
HOP_VT_Alias = TypeVar("HOP_VT_Alias", bound="TorchHigherOrderOperatorVariable")

log = logging.getLogger(__name__)
hc_log = torch._logging.getArtifactLogger(__name__, "hierarchical_compile")

__all__ = [
    "Any",
    "AttrSource",
    "CONSTANT_VARIABLE_NONE",
    "ConstantVariable",
    "ConstDictVariable",
    "DictGetItemSource",
    "Generator",
    "GraphModule",
    "HOP_VT_Alias",
    "HigherOrderOperator",
    "Iterable",
    "LazyVariableTracker",
    "ListVariable",
    "Literal",
    "NestedCompileRegionOptions",
    "Optional",
    "OrderedSet",
    "OutputSpec",
    "Proxy",
    "RepararametrizeModuleContextVariable",
    "Sequence",
    "Source",
    "StorageAliasingTracker",
    "SubgraphTracingInfo",
    "SymNodeVariable",
    "TensorVariable",
    "TorchHigherOrderOperatorVariable",
    "TorchScriptObjectVariable",
    "TupleVariable",
    "UncapturedHigherOrderOpError",
    "Union",
    "UnspecializedNNModuleVariable",
    "UserFunctionVariable",
    "VariableTracker",
    "_assert_tensors_nonaliasing",
    "_call_function_and_unflatten_output",
    "_call_function_with_auto_output_flattening",
    "_call_while_loop",
    "_check_all_tensorvariable",
    "_check_supported_callable_arg",
    "_extract_tensor_metadata",
    "_get_fake_value",
    "_hop_name_to_variable_class",
    "_make_inlined",
    "_merge_graph_inputs",
    "add_call_function",
    "add_hop_context",
    "are_same_graph_modules",
    "cast",
    "check_aliasing_and_input_mutation",
    "check_meta_consistency_vt",
    "collect_intermediate_outputs",
    "contextlib",
    "copy",
    "discard_graph_changes",
    "enable_python_dispatcher",
    "field",
    "find_mismatched_vars",
    "functools",
    "get_fake_value",
    "get_hop_args",
    "get_tensor_storages",
    "graph_break_hints",
    "inspect",
    "itertools",
    "log",
    "make_attr",
    "maybe_positional_arg_names",
    "move_lifted_freevars_phs_to_end",
    "only_consist_of",
    "overwrite_tensor_vt_proxy",
    "overwrite_tensor_vt_requires_grad",
    "proxy_args_kwargs",
    "pytree",
    "set_example_value",
    "speculate_subgraph",
    "speculate_subgraph_with_auto_output_flattening",
    "torch",
    "trace_hop_function",
    "trace_hop_function_with_auto_output_flattening",
    "traceback",
    "types",
    "unimplemented",
    "validate_args_and_maybe_create_graph_inputs",
    "validate_subgraph_output_types",
    "variables",
    "warnings",
]


@dataclass
class OutputSpec:
    """
    Contains the treespec of the output of the speculated subgraph, and the
    information to mask out the constant values from the output during
    flattening and inserting them back during unflattening. Cleaning up
    constants from the graph makes the graph simpler for AOTDispatcher and
    Inductor.
    """

    treespec: pytree.TreeSpec
    # list of True/False to identify the locations of const values in the
    # subgraph output. True means that value at that index is a constant.
    masks_to_filter_const_values: list[bool] | None = None
    # The actual constant values that were present in the subgraph output. Note
    # that this is the same length as the mask, we just look at the indices
    # where mask is True.
    const_values: list[Any] | None = None
    # Number of intermediate nodes that are also made subgraph outputs.
    num_intermediate_nodes_as_outputs: int = 0

    def __post_init__(self) -> None:
        if (
            self.masks_to_filter_const_values is not None
            and self.const_values is not None
        ):
            assert len(self.masks_to_filter_const_values) == len(self.const_values)


@dataclass(frozen=True)
class SubgraphTracingInfo:
    """Properties observed during subgraph tracing.

    Expandable container for metadata collected while tracing a HOP subgraph.
    Future additions may include aliasing info, mutation info, etc.
    """

    # User code stack at the point where the first externally-visible side
    # effect was detected.  None means no side effect was observed.
    side_effect_stack: traceback.StackSummary | None = None
    # All sources accessed via VariableBuilder during the subgraph trace.
    traced_sources: OrderedSet[Any] = field(default_factory=OrderedSet)


@contextlib.contextmanager
def discard_graph_changes(tx: "InstructionTranslator") -> Generator[None, None, None]:
    ctx = tx.output.subtracer("subgraph_wrapper", None)
    try:
        ctx.__enter__()
        yield
    finally:
        ctx.__exit__(None, None, None)


def check_meta_consistency_vt(
    vars1: list[VariableTracker],
    vars2: list[VariableTracker],
    lhs_name: str,
    rhs_name: str,
    include_contiguity: bool = True,
) -> None:
    from torch._higher_order_ops.utils import check_meta_consistency

    def _unwrap_var(var: VariableTracker) -> Any:
        if var.is_tensor():
            # pyrefly: ignore[missing-attribute]
            return var.proxy.node.meta["example_value"]
        elif isinstance(var, SymNodeVariable):
            return var.sym_num
        elif var.is_python_constant():
            return var.as_python_constant()
        else:
            unimplemented(
                gb_type="cannot unwrap variable for check_meta_consistency",
                context=str(var),
                explanation=f"Expected {var} to be TensorVariable, SymNodeVariable, or ConstantVariable",
                hints=[],
            )

    unwrapped1 = [_unwrap_var(var) for var in vars1]
    unwrapped2 = [_unwrap_var(var) for var in vars2]

    return check_meta_consistency(
        unwrapped1,
        unwrapped2,
        lhs_name,
        rhs_name,
        include_contiguity=include_contiguity,
    )


@contextlib.contextmanager
def dynamo_enable_grad(
    tx: "InstructionTranslator", enable: bool = True
) -> Generator[None, None, None]:
    from .. import GradModeVariable

    org_value = torch.is_grad_enabled()
    try:
        GradModeVariable.create(tx, enable, initialized=True)
        yield
    finally:
        GradModeVariable.create(tx, org_value, initialized=True)


@contextlib.contextmanager
def dynamo_allow_side_effects_in_hop(
    tx: "InstructionTranslator",
) -> Generator[None, None, None]:
    orig_val = tx.output.current_tracer.allow_side_effects_in_hop
    try:
        tx.output.current_tracer.allow_side_effects_in_hop = True
        yield
    finally:
        tx.output.current_tracer.allow_side_effects_in_hop = orig_val


def find_mismatched_vars(
    var: Any, types: type | tuple[type, ...], allow_none: bool = False
) -> set[VariableTracker]:
    """
    Recursively finds variables whose type is not an instance of the specified types.
    Args:
        var: The variable to check.
        types: A tuple of allowed types.
        allow_none (bool): Whether to allow None values. Defaults to False.
    Returns:
        A set of variables whose type is not an instance of the specified types.
    """
    mismatched_vars = set()
    if isinstance(var, (list, tuple)):
        for item in var:
            mismatched_vars.update(find_mismatched_vars(item, types, allow_none))
    elif isinstance(var, (TupleVariable, ListVariable)):
        for item in var.items:
            mismatched_vars.update(find_mismatched_vars(item, types, allow_none))
    elif isinstance(var, ConstDictVariable):
        for value in var.items.values():
            mismatched_vars.update(find_mismatched_vars(value, types, allow_none))
    else:
        if not isinstance(var, types) and not (allow_none and var.is_constant_none()):
            mismatched_vars.add(var)
    return mismatched_vars


def only_consist_of(
    var: Any, types: tuple[type, ...], allow_none: bool = False
) -> bool:
    mismatch_vars = find_mismatched_vars(var, types, allow_none=allow_none)
    return len(mismatch_vars) == 0


def add_call_function(
    tx: "InstructionTranslator",
    fn: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    flat_example_value: Any,
    config: NestedCompileRegionOptions | None = None,
) -> VariableTracker:
    from ..builder import wrap_fx_proxy

    proxy = tx.output.create_proxy(
        "call_function",
        fn,
        args=args,
        kwargs=kwargs,
    )

    # Set backend metadata if provided
    if config is not None:
        if "custom" not in proxy.node.meta:
            # pyrefly: ignore [implicit-any]
            proxy.node.meta["custom"] = {}
        proxy.node.meta["custom"]["nested_region_config"] = config
        assert proxy.node.target == torch._higher_order_ops.invoke_subgraph

    # Store the invocation as a call
    flat_variable = wrap_fx_proxy(
        tx=tx,
        proxy=proxy,
        example_value=flat_example_value,
    )
    return flat_variable


def overwrite_tensor_vt_requires_grad(
    graph_output_vts: Iterable[VariableTracker], flat_variable: VariableTracker
) -> None:
    # All outputs of autograd.Function have requires_grad=True. We turn off
    # grad_mode in autograd.Function, so our outputs naively have
    # requires_grad=False. So we hackily force them back on here. A better
    # solution would be to write python code that Dynamo could trace but we
    # decided that it wasn't worth it.
    # pyrefly: ignore[missing-attribute]
    for orig_vt, subgraph_vt in zip(graph_output_vts, flat_variable.items):
        if isinstance(orig_vt, variables.TensorVariable):
            assert isinstance(subgraph_vt, variables.TensorVariable)
            orig_vt.requires_grad = subgraph_vt.requires_grad
            if orig_vt.requires_grad:
                orig_vt.has_grad_fn = True


def overwrite_tensor_vt_proxy(
    graph_output_vts: Iterable[VariableTracker], flat_variable: VariableTracker
) -> None:
    # wrap_fx_proxy creates fresh variable trackers. However, the main program
    # after the speculate subgraph can still use the original tensor vts that
    # are still pointing to the nodes present in the subgraph. So, we reproxify
    # the original tensor vts with the subgraph outputs. This way, whenever the
    # outer graph uses an original vt, it uses the subgraph output.
    #
    # This is critical for maintaining the separation between:
    # - `body_r`: The output VT structure that Dynamo continues tracing (may
    #   contain non-proxyable objects, nested structures, etc.)
    # - `graph_output_vts`: Only the tensor/symint VTs that were actual graph
    #   outputs from speculate_subgraph
    #
    # By overwriting the proxies of VTs in `body_r` with the proxies from the
    # HOP call, we ensure the outer graph correctly references the HOP outputs
    # while still allowing `body_r` to contain arbitrary Python objects.
    # pyrefly: ignore[missing-attribute]
    for orig_vt, subgraph_vt in zip(graph_output_vts, flat_variable.items):
        if isinstance(
            orig_vt,
            (
                variables.SymNodeVariable,
                variables.TensorVariable,
                TorchScriptObjectVariable,
            ),
        ):
            assert subgraph_vt.is_tensor() or isinstance(
                subgraph_vt, (SymNodeVariable, TorchScriptObjectVariable)
            )
            orig_vt.proxy = subgraph_vt.proxy


def _call_function_with_auto_output_flattening(
    tx: "InstructionTranslator",
    fn: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    flat_example_value: Any,
    body_r: VariableTracker | None,
    graph_output_vts: VariableTracker | tuple[VariableTracker, ...],
    config: NestedCompileRegionOptions | None = None,
) -> VariableTracker | None:
    """
    Create HOP call node and reproxify output VTs for HOPs with auto output semantics.

    This function is used by HOPs with auto output semantics (see speculate_subgraph_with_auto_output_flattening)
    to create the actual HOP call in the FX graph and properly handle the output variable trackers.

    The key operation is "reproxifying" - updating the proxies of the original tensor VTs
    (from body_r) to point to the HOP call outputs, ensuring the outer graph correctly
    references the HOP outputs while allowing body_r to contain arbitrary Python objects.

    Args:
        tx: The instruction translator
        fn: The HOP function to call
        args: Arguments for the HOP call (typically includes the subgraph node)
        kwargs: Keyword arguments for the HOP call
        flat_example_value: Example value for the HOP output
        body_r: The output VT structure that Dynamo continues tracing with (may be None)
        graph_output_vts: Tensor/symint VTs that were actual graph outputs

    Returns:
        The body_r VT (unchanged), which Dynamo will continue tracing with
    """

    flat_variable = add_call_function(tx, fn, args, kwargs, flat_example_value, config)
    if body_r is not None:
        # pyrefly: ignore[bad-argument-type]
        overwrite_tensor_vt_proxy(graph_output_vts, flat_variable)
    return body_r


def _call_function_and_unflatten_output(
    tx: "InstructionTranslator",
    fn: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    flat_example_value: Any,
    ret_spec: OutputSpec,
    body_r: VariableTracker | None,
) -> VariableTracker:
    from ..builder import SourcelessBuilder, wrap_fx_proxy

    # Store the invocation as a call
    flat_variable = wrap_fx_proxy(
        tx=tx,
        proxy=tx.output.create_proxy(
            "call_function",
            fn,
            args=args,
            kwargs=kwargs,
        ),
        example_value=flat_example_value,
    )

    # wrap_fx_proxy creates fresh variable trackers. However, the main program
    # after the speculate subgraph can still use the original tensor vts that
    # are still pointing to the nodes present in the subgraph. So, we reproxify
    # the original tensor vts with the subgraph outputs. This way, whenever the
    # outer graph uses an original vt, it uses the subgraph output.
    if body_r is not None:
        # mypy: ignore[attr-defined]
        for orig_vt, subgraph_vt in zip(body_r.items, flat_variable.items):
            if orig_vt.is_tensor() or isinstance(orig_vt, SymNodeVariable):
                assert subgraph_vt.is_tensor() or isinstance(
                    subgraph_vt, SymNodeVariable
                )
                orig_vt.proxy = subgraph_vt.proxy

    if ret_spec.num_intermediate_nodes_as_outputs:
        # The treespec was computed w/o any extra intermediate outputs. At this
        # point, it is safe to just get rid of the extra outputs
        flat_variable = SourcelessBuilder.create(
            tx,
            flat_variable.items[  # mypy: ignore[attr-defined]
                : -ret_spec.num_intermediate_nodes_as_outputs
            ],
        )

    if ret_spec.masks_to_filter_const_values:
        from torch._dynamo.external_utils import insert_const_values_with_mask

        # During flattening, we removed the constant values. To ensure Dynamo
        # can trace correctly, insert back the constant values in the output.
        flat_variable = _make_inlined(tx, insert_const_values_with_mask)(
            flat_variable, ret_spec.masks_to_filter_const_values, ret_spec.const_values
        )

    # Transform variable back into a list (previously made into a tuple by
    # speculate_subgraph function) so as to respect the pytree API typing.
    flat_list_variable = SourcelessBuilder.create(tx, list).call_function(
        tx, [flat_variable], {}
    )
    return (
        _make_inlined(tx, pytree.tree_unflatten)(flat_list_variable, ret_spec.treespec)
        if ret_spec.treespec
        else flat_variable
    )


def _assert_tensors_nonaliasing(inputs: Any, outputs: Any) -> None:
    input_tensor_ids = {
        id(t) for t in pytree.tree_leaves(inputs) if isinstance(t, torch.Tensor)
    }
    output_tensor_ids = {
        id(t) for t in pytree.tree_leaves(outputs) if isinstance(t, torch.Tensor)
    }
    assert input_tensor_ids.isdisjoint(output_tensor_ids), (
        "inputs to function body cannot alias outputs"
    )


def get_tensor_storages(tensor: torch.Tensor) -> set[StorageWeakRef]:
    """
    Get storage references from a tensor.

    Handles regular tensors. Raises NotImplementedError for sparse tensors
    and traceable wrapper subclasses.

    Args:
        tensor: The tensor to extract storages from

    Returns:
        Set of StorageWeakRef objects for the tensor's storage(s)
    """
    from torch.multiprocessing.reductions import StorageWeakRef
    from torch.utils._python_dispatch import is_traceable_wrapper_subclass

    storages: set[StorageWeakRef] = set()

    if not isinstance(tensor, torch.Tensor):
        return storages

    if tensor.is_sparse or tensor.is_sparse_csr:
        raise NotImplementedError("get_tensor_storages does not support sparse tensors")

    if is_traceable_wrapper_subclass(tensor):
        raise NotImplementedError(
            "get_tensor_storages does not support traceable wrapper subclasses"
        )
    else:
        storages.add(StorageWeakRef(tensor._typed_storage()))

    return storages


class StorageAliasingTracker:
    """
    Tracks storage references to detect aliasing between tensors.

    This class encapsulates the logic for collecting storages from tensors
    and checking for aliasing conflicts. Used to filter intermediate outputs
    that would create input-output or output-output aliasing.
    """

    def __init__(self) -> None:
        self.excluded_storages: set[StorageWeakRef] = set()

    def _collect_storages_from_tensor(self, example_value: torch.Tensor) -> None:
        self.excluded_storages.update(get_tensor_storages(example_value))

    def collect_from_inputs(self, tx: "InstructionTranslator") -> None:
        """Collect storages from graph input placeholders."""
        from torch._higher_order_ops.utils import _collect_fake_inputs

        for node in tx.output.graph.nodes:
            if node.op == "placeholder":
                example_value = _collect_fake_inputs([node])[0]
                if isinstance(example_value, torch.Tensor):
                    self._collect_storages_from_tensor(example_value)
            else:
                break

    def collect_from_outputs(self, graph_output_vts: Sequence[VariableTracker]) -> None:
        """Collect storages from existing graph outputs."""
        from torch._higher_order_ops.utils import _collect_fake_inputs

        for vt in graph_output_vts:
            proxy = vt.as_proxy()
            example_value = _collect_fake_inputs([proxy.node])[0]
            if isinstance(example_value, torch.Tensor):
                self._collect_storages_from_tensor(example_value)

    def check_and_track(self, proxy_node: Proxy) -> bool:
        """
        Check if a tensor can be added as a subgraph output without causing aliasing issues.

        Given a proxy node, extracts its example tensor value and checks if its storage
        aliases with any previously tracked storages (from inputs or other outputs).
        If there's no aliasing conflict, the tensor's storage is added to the tracked set.

        Args:
            proxy_node: An FX proxy node whose example_value is the tensor to check.

        Returns:
            True if the tensor doesn't alias with tracked storages (safe to add as output),
            False if it aliases (should be filtered out).
        """
        from torch._higher_order_ops.utils import _collect_fake_inputs
        from torch.multiprocessing.reductions import StorageWeakRef
        from torch.utils._python_dispatch import is_traceable_wrapper_subclass

        example_value = _collect_fake_inputs([proxy_node])[0]

        # Non-tensor outputs (e.g., symints) don't have aliasing concerns
        if not isinstance(example_value, torch.Tensor):
            return True

        # Check if any storage aliases with existing inputs/outputs
        tensor_storages = get_tensor_storages(example_value)
        if tensor_storages & self.excluded_storages:
            return False

        # Track this tensor's storage (for wrapper subclasses, inner storages were already checked)
        if not is_traceable_wrapper_subclass(example_value):
            if not (example_value.is_sparse or example_value.is_sparse_csr):
                self.excluded_storages.add(
                    StorageWeakRef(example_value._typed_storage())
                )

        return True


def collect_intermediate_outputs(
    tx: "InstructionTranslator",
    subtracer: "SubgraphTracer",
    graph_output_vts: Sequence[VariableTracker],
    filter_aliased_intermediates: bool = False,
) -> list[VariableTracker]:
    extra_outputs = []
    existing_out_proxies = {vt.as_proxy() for vt in graph_output_vts}

    # Build the aliasing tracker if we're filtering
    tracker = None
    if filter_aliased_intermediates:
        tracker = StorageAliasingTracker()
        tracker.collect_from_inputs(tx)
        tracker.collect_from_outputs(graph_output_vts)

    for out in subtracer.tracked_proxyable_vt:
        proxy = out.as_proxy()

        # Skip if already in output
        if proxy in existing_out_proxies:
            continue

        # TODO floats are not supported in HOP input/output
        if isinstance(out, SymNodeVariable) and out.python_type() is float:
            continue

        if not filter_aliased_intermediates:
            extra_outputs.append(out)
        else:
            # Filter out intermediates that alias with inputs or outputs.
            # This is needed for HOPs like invoke_subgraph that don't support aliasing.
            # TODO: If a filtered intermediate is captured by side effects (e.g., appended
            # to a list), it will fail later with "does not belong to this Graph" error
            # when the outer graph tries to use it. See test_side_effect_with_aliased_intermediate.
            assert tracker is not None
            if tracker.check_and_track(proxy.node):
                extra_outputs.append(out)

    return extra_outputs


def _check_all_tensorvariable(args: Sequence[VariableTracker]) -> None:
    if not all(type(a.realize()) is TensorVariable for a in args):
        unimplemented(
            gb_type="HOP: non torch.Tensor leaf",
            context=f"args types: {[type(a.realize()) for a in args]}",
            explanation="Expected all leaves to be of torch.Tensor type.",
            hints=[],
        )


def _check_supported_callable_arg(
    tx: "InstructionTranslator", func_var: VariableTracker, arg_name: str
) -> None:
    from ..builder import SourcelessBuilder

    is_callable = (
        SourcelessBuilder.create(tx, callable)
        .call_function(tx, [func_var], {})
        .as_python_constant()
    )
    if not is_callable:
        unimplemented(
            gb_type="HOP: non-callable variable",
            context=f"arg name: {arg_name}, func_var type: {str(func_var)}",
            explanation=f"{arg_name} should be a callable but is of type {str(func_var)}.",
            hints=[],
        )


def validate_subgraph_output_types(
    output: VariableTracker | Sequence[VariableTracker],
) -> None:
    """Verify that that the output of the subgraph is a tensor,
    int, bool, SymBool, or SymInt.
    """
    from .. import TensorVariable

    if non_tensor_output := find_mismatched_vars(
        output, TensorVariable, allow_none=True
    ):
        for out in non_tensor_output:
            if (
                (isinstance(out, SymNodeVariable) and out.python_type() in (int, bool))
                or (
                    out.is_python_constant()
                    and isinstance(out.as_python_constant(), (int, bool))
                )
                or isinstance(out, TorchScriptObjectVariable)
            ):
                continue
            unimplemented(
                gb_type="HOP body output unsupported",
                context=f"non-tensor outputs: {non_tensor_output}",
                explanation="HigherOrderOperator body's output must consist of tensors or ints/bools only "
                f"but got {out.python_type()}.",
                hints=[
                    *graph_break_hints.USER_ERROR,
                ],
            )


def _call_while_loop(
    self: Union[
        "WhileLoopHigherOrderVariable", "WhileLoopStackOutputHigherOrderVariable"
    ],
    tx: "InstructionTranslator",
    args: Sequence[VariableTracker],
    kwargs: dict[str, VariableTracker],
    stack_output: bool,
    hop_name: str,
) -> VariableTracker:
    from torch._higher_order_ops.while_loop import _create_unbacked_symint

    args, kwargs = LazyVariableTracker.realize_all((args, kwargs))
    cond_fn, body_fn, operands, additional_inputs = args

    # Input checks
    for i, k in enumerate(["cond_fn", "body_fn", "operands"]):
        if v := kwargs.pop(k, None):
            assert i == len(args), (
                "did not provide the right number of non-keyword args"
            )
            args.append(v)

    if kwargs or len(args) != 4:
        unimplemented(
            gb_type="torch.while_loop: improper args/kwargs",
            context=f"args: {args}, kwargs: {kwargs}",
            explanation=f"torch.while_loop expects 4 positional arguments (got {len(args)}) "
            f"and no keyword arguments (got {len(kwargs)}) "
            "Usage: while_loop(cond_fn, body_fn, operands)",
            hints=[
                *graph_break_hints.USER_ERROR,
            ],
        )

    # cond_fn and body_fn input check
    _check_supported_callable_arg(tx, cond_fn, "cond_fn")
    _check_supported_callable_arg(tx, body_fn, "body_fn")

    # operands input check
    operands_seq = operands.unpack_var_sequence(tx)

    # additional_inputs input check
    if not isinstance(additional_inputs, (ListVariable, TupleVariable)):
        unimplemented(
            gb_type="torch.while_loop: improper additional_inputs",
            context=str(additional_inputs),
            explanation=f"Expected additional_inputs to be a list/tuple but got {additional_inputs.python_type()}",
            hints=[
                *graph_break_hints.DYNAMO_BUG,
            ],
        )
    additional_inputs_seq = additional_inputs.unpack_var_sequence(tx)

    with discard_graph_changes(tx):
        # Note: this must be run under discard graph changes.
        def unspecialize_carried_inputs(
            tx: "InstructionTranslator", carry: VariableTracker
        ) -> VariableTracker:
            # See NOTE [unspecialize int carry with unbacked symints]
            if (
                carry.is_python_constant()
                and isinstance(carry.as_python_constant(), int)
            ) or isinstance(carry, SymNodeVariable):
                example_value = _create_unbacked_symint(
                    tx.output.fake_mode, ignore_fresh_unbacked_symbols=True
                )
                proxy = tx.output.current_tracer.create_graph_input(
                    "unbacked_symint", type(example_value), example_value
                )
                return SymNodeVariable.create(tx, proxy, example_value)
            else:
                # See NOTE [unspecialize constant tensor carry]
                assert carry.is_tensor()
                cloned_carry = carry.clone()
                # type: ignore[attr-defined]
                cloned_carry.proxy.node.meta["example_value"].constant = None
                return cloned_carry

        # clone inputs across subgraphs, to avoid unbacked memoization in fake prop
        cond_operands_seq = [
            unspecialize_carried_inputs(
                tx,
                (
                    carry.call_method(tx, "clone", args=(), kwargs={})
                    if carry.is_tensor()
                    else carry
                ),
            )
            for carry in operands_seq
        ]
        body_operands_seq = [
            unspecialize_carried_inputs(
                tx,
                (
                    carry.call_method(tx, "clone", args=(), kwargs={})
                    if carry.is_tensor()
                    else carry
                ),
            )
            for carry in operands_seq
        ]

    # create cond subgrpahs
    (
        (cond_r, _cond_treespec),
        cond_graph,
        cond_lifted_freevars,
    ) = speculate_subgraph(
        tx,
        cond_fn,
        cond_operands_seq + additional_inputs_seq,
        {},
        hop_name,
        source_target=self.value,
        # NOTE [why we cannot use "automatic" for while_loop]:
        # The reason is that we want to enforce
        # the ordering of inputs and outputs to be consistent and the ordering
        # of cond_fn and body_fn to the consistent.
        # e.g. suppose we use "automatic" and we have:
        #
        # def body_fn(ph1, ph2):
        #   new_a, new_b = ph2.cos(), ph1.sin()
        #   return new_a, new_b
        #
        # a, b = torch.randn(3), torch.randn(3)
        # new_a, new_b = body_fn(a, b)
        #
        # Using automatic, the ordering of arguments will be the order that they're
        # used. In this example, the capture graph looks like:
        #
        # def captured_body(ph1, ph2):
        #   new_a, new_b = ph1.cos(), ph2.add_(1)
        #   return new_a, new_b
        #
        # This is fine when we change the calling convention of captured_body to be
        # new_a, new_b = captured_body(b, a).
        # But for while_loop, the next iteration's input is previous iteration output
        # we'll end up feeding captured_body(new_a, new_b) instead.
        # So it's best we always enforce the ordering of carried_inputs the same as outputs
        # with "flatten_manual".
        set_subgraph_inputs="flatten_manual",
        supports_input_mutation=self.supports_input_mutation,
        supports_aliasing=self.supports_aliasing,
        remove_consts_from_outputs=False,
    )
    cond_nn_modules = dict(tx.output.nn_modules)
    validate_subgraph_output_types(cond_r)
    if cond_r.is_tensor():
        cond_r_meta = _extract_tensor_metadata(
            # type: ignore[attr-defined]
            cond_r.proxy.node.meta["example_value"],
            include_contiguity=False,
        )
        if cond_r_meta.dtype != torch.bool or cond_r_meta.shape != torch.Size([]):
            unimplemented(
                gb_type="torch.while_loop: unsupported cond_fn return type",
                context=str(cond_r),
                explanation=f"Expected cond_fn to return a scalar tensor or a bool but got {cond_r_meta.shape}.",
                hints=[
                    *graph_break_hints.USER_ERROR,
                ],
            )
    elif cond_r.is_python_constant():
        # short-circuiting while_loop when cond_fn returns a constant such as 0, 1 True or False
        pred = cond_r.as_python_constant()
        if pred:
            unimplemented(
                gb_type="torch.while_loop: infinite loop detected",
                context=str(cond_r),
                explanation=f"Infinite loop detected because while_loop's cond_fn always returns the same value {pred}.",
                hints=[
                    *graph_break_hints.USER_ERROR,
                ],
            )
        else:
            return operands

    # create body subgraph
    (
        (body_r, body_treespec),
        body_graph,
        body_lifted_freevars,
    ) = speculate_subgraph(
        tx,
        body_fn,
        body_operands_seq + additional_inputs_seq,
        {},
        hop_name,
        source_target=self.value,
        set_subgraph_inputs="flatten_manual",
        should_flatten_outputs=True,
        supports_input_mutation=False,
        supports_aliasing=False,
        remove_consts_from_outputs=False,
    )
    validate_subgraph_output_types(body_r)

    # We set include contiguity=False because we have vmap x HOP tests, where if
    # include_contiguity=True will call t.is_contiguous inside of vmap and get an error
    # "querying is_contiguous inside of vmap for memory_format other than
    # torch.contiguous_format is not yet implemented". This is okay because stride
    # is still checked.
    check_meta_consistency_vt(
        body_r.unpack_var_sequence(tx),
        operands_seq,
        "body_fn_output",
        "carried_inputs",
        include_contiguity=False,
    )

    (
        cond_graph,
        body_graph,
        cond_shared,
        _body_shared,
        cond_unique,
        body_unique,
    ) = _merge_graph_inputs(
        cond_graph,
        cond_lifted_freevars,
        "cond_fn",
        body_graph,
        body_lifted_freevars,
        "body_fn",
    )

    # Note: cond_shared and body_shared refer to the same proxy in parent graph
    # so using either of them is OK. Use cond_shared as it doesn't matter.
    additional_lifted_inputs = cond_shared + cond_unique + body_unique

    body_nn_modules = dict(tx.output.nn_modules)

    cond_gm = torch.fx.GraphModule(cond_nn_modules, cond_graph)
    body_gm = torch.fx.GraphModule(body_nn_modules, body_graph)
    cond_name = tx.output.install_subgraph("cond_fn", cond_gm)
    body_name = tx.output.install_subgraph("body_fn", body_gm)

    cond_node = make_attr(tx, cond_name)
    body_node = make_attr(tx, body_name)

    operands_proxy = tuple(operand.as_proxy() for operand in operands_seq)
    additional_inputs_proxy = tuple(
        [inp.as_proxy() for inp in additional_inputs_seq] + additional_lifted_inputs
    )
    p_args = (
        cond_node,
        body_node,
        operands_proxy,
        additional_inputs_proxy,
    )
    return _call_function_and_unflatten_output(
        tx,
        self.value,
        p_args,
        {},
        None,
        body_treespec,
        body_r,
    )


def are_same_graph_modules(
    fn_name: str, a_mod: GraphModule, b_mod: GraphModule, fake_mode: "FakeTensorMode"
) -> bool:
    from torch._subclasses._fake_tensor_utils import _CacheKeyState
    from torch._subclasses.fake_tensor import extract_tensor_metadata

    # Maps the equivalent nodes from a to b
    node_map = {}

    def check_all_args(a_nodes: Iterable[Any], b_nodes: Iterable[Any]) -> bool:
        for arg_a, arg_b in zip(a_nodes, b_nodes):
            if isinstance(arg_a, torch.fx.Node):
                if node_map[arg_a] != arg_b:
                    return False
            elif isinstance(arg_a, slice):
                if not isinstance(arg_b, slice):
                    return False
                if not check_all_args(
                    (arg_a.start, arg_a.stop, arg_a.step),
                    (arg_b.start, arg_b.stop, arg_b.step),
                ):
                    return False
            elif arg_a != arg_b:
                # This is a catch-all for everything else. `slice` was a
                # surprise but can there be other data structures that can
                # contain fx.Nodes in them?
                return False
        return True

    for a_node, b_node in zip(a_mod.graph.nodes, b_mod.graph.nodes):
        if a_node.op != b_node.op:
            return False

        if a_node.op == "placeholder":
            a_value = a_node.meta["example_value"]
            b_value = b_node.meta["example_value"]

            if isinstance(a_value, torch.Tensor):
                if not isinstance(b_value, torch.Tensor):
                    return False
                # Extract fake tensor metadata for a and b and then compare
                # pyrefly: ignore [implicit-any]
                a_result = []
                state = _CacheKeyState(fake_mode.shape_env)
                a_metadata = extract_tensor_metadata(a_value)
                a_metadata._flatten_into(a_result, fake_mode, state)

                b_result = []
                state = _CacheKeyState(fake_mode.shape_env)
                b_metadata = extract_tensor_metadata(b_value)
                b_metadata._flatten_into(b_result, fake_mode, state)
                if a_result != b_result:
                    return False
            elif isinstance(a_value, torch.SymInt):
                if not isinstance(b_value, torch.SymInt):
                    return False
                if a_value is not b_value:
                    return False
        elif a_node.op == "call_function":
            if a_node.target is not b_node.target:
                return False
            a_flat, _ = pytree.tree_flatten((a_node.args, a_node.kwargs))
            b_flat, _ = pytree.tree_flatten((b_node.args, b_node.kwargs))
            if not check_all_args(a_flat, b_flat):
                hc_log.debug(
                    "%s: Graph comparison failed at node (call_function): %s",
                    fn_name,
                    a_node,
                )
                return False
        elif a_node.op == "call_method":
            if a_node.target != b_node.target:
                return False
            a_flat, _ = pytree.tree_flatten((a_node.args, a_node.kwargs))
            b_flat, _ = pytree.tree_flatten((b_node.args, b_node.kwargs))
            if not check_all_args(a_flat, b_flat):
                hc_log.debug(
                    "%s: Graph comparison failed at node (call_method) : %s",
                    fn_name,
                    a_node,
                )
                return False
        elif a_node.op == "output":
            a_flat, _ = pytree.tree_flatten((a_node.args, a_node.kwargs))
            b_flat, _ = pytree.tree_flatten((b_node.args, b_node.kwargs))
            if not check_all_args(a_flat, b_flat):
                hc_log.debug("%s: Graph comparison failed at the output node", fn_name)
                return False
        elif a_node.op == "get_attr":
            a_attr = getattr(a_mod, a_node.target)
            b_attr = getattr(b_mod, b_node.target)
            if isinstance(a_attr, torch.fx.GraphModule):
                if not isinstance(b_attr, torch.fx.GraphModule):
                    return False
                # This is an example of a HOP inside a HOP
                if not are_same_graph_modules(fn_name, a_attr, b_attr, fake_mode):
                    return False
            else:
                # TODO - write an example with tensor as a graph attribute in
                # the Fx graph
                raise NotImplementedError(f"get_attr with {type(a_attr)}")
        else:
            # TODO - call_module is not supported because Dynamo Fx graph does
            # not install a call_module
            raise NotImplementedError(f"Graph equivalence check saw a {a_node.op}")

        # Two nodes are equal - add them to them map
        node_map[a_node] = b_node

    return True


def validate_args_and_maybe_create_graph_inputs(
    sub_args: list[VariableTracker],
    tracer: "SubgraphTracer",
    tx: "InstructionTranslator",
    set_subgraph_inputs: str,
    description: str,
    sub_args_names: Sequence[str] | None = None,
) -> list[Any]:
    from .. import AutogradFunctionContextVariable
    from ..builder import SourcelessBuilder, wrap_fx_proxy_cls

    assert tracer.parent is not None

    if set_subgraph_inputs == "flatten_manual":
        flat_args, tree_spec = _make_inlined(tx, pytree.tree_flatten)(
            SourcelessBuilder.create(tx, sub_args)
        ).unpack_var_sequence(tx)

        flat_inputs = validate_args_and_maybe_create_graph_inputs(
            flat_args.unpack_var_sequence(tx),
            tracer,
            tx,
            set_subgraph_inputs="manual",
            description=description,
        )

        return _make_inlined(tx, pytree.tree_unflatten)(
            SourcelessBuilder.create(tx, list(flat_inputs)), tree_spec
        ).unpack_var_sequence(tx)
    else:
        if sub_args_names is not None:
            # Can be greater if user passes some args as kwargs
            assert len(sub_args_names) >= len(sub_args)
        args = []
        for idx, a in enumerate(sub_args):
            assert isinstance(a, VariableTracker)
            new_arg = None
            if set_subgraph_inputs == "automatic":
                args.append(a)
                continue
            elif set_subgraph_inputs == "automatic_with_forced_inputs":
                if isinstance(a, variables.TensorVariable):
                    node = a.maybe_fx_node()
                    assert node is not None
                    example_value = node.meta["example_value"]
                    arg_name = (
                        a.as_proxy().node.name
                        if sub_args_names is None
                        else sub_args_names[idx]
                    )
                    new_proxy = tracer.create_graph_input(
                        arg_name, a.python_type(), example_value
                    )
                    example_value = node.meta.get("example_value", None)
                    a = wrap_fx_proxy_cls(
                        target_cls=type(a),
                        tx=tx,
                        proxy=new_proxy,
                        example_value=example_value,
                    )
                args.append(a)
                continue

            if a.is_python_constant():
                # This arg is not used in the body of the higher order op.
                # Currently, this new input is added to make the calls
                # happy, which expect a fixed number of arguments. In
                # future, we can clean this up.
                arg_name = (
                    "const_unused"
                    if sub_args_names is None
                    else f"const_unused_{sub_args_names[idx]}"
                )
                tracer.create_graph_input(
                    arg_name, a.python_type(), a.as_python_constant()
                )
                new_arg = a
            # Weird special case, we probably want to delete it or fold it
            # into the next case (of `a` being placeable into a graph)
            elif isinstance(a, AutogradFunctionContextVariable):
                example_value = a.as_proxy().node.meta["example_value"]
                arg_name = (
                    a.as_proxy().node.name
                    if sub_args_names is None
                    else sub_args_names[idx]
                )
                tracer.create_graph_input(arg_name, a.python_type(), example_value)
                new_arg = a
            # If `a` can be put into a graph
            elif a.maybe_fx_node() is not None:
                node = a.maybe_fx_node()
                assert node is not None
                example_value = node.meta.get("example_value", None)
                arg_name = node.name if sub_args_names is None else sub_args_names[idx]
                new_proxy = tracer.create_graph_input(
                    arg_name, a.python_type(), example_value
                )
                new_arg = wrap_fx_proxy_cls(
                    target_cls=type(a),
                    tx=tx,
                    proxy=new_proxy,
                    example_value=example_value,
                )
            # If `a` cannot be put into a graph
            else:
                # HOPs work much better if they use speculate_subgraph(set_subgraph_inputs="automatic").
                unimplemented(
                    gb_type="HOP body taking non-Tensor as input",
                    context=str(sub_args),
                    explanation=f"{description} with body that accepts non-Tensors as input. "
                    f"Got type {a.python_type()} at index {idx}.",
                    hints=[
                        *graph_break_hints.USER_ERROR,
                    ],
                )
            args.append(new_arg)
        return args


def _merge_graph_inputs(
    l_graph: torch.fx.Graph,
    l_lifted_freevars: dict[Proxy, Proxy],
    l_name: str,
    r_graph: torch.fx.Graph,
    r_lifted_freevars: dict[Proxy, Proxy],
    r_name: str,
) -> tuple[
    torch.fx.Graph, torch.fx.Graph, list[Proxy], list[Proxy], list[Proxy], list[Proxy]
]:
    def dedup_and_sort_lifted_freevars(
        l_lifted_freevars: dict[Proxy, Proxy], r_lifted_freevars: dict[Proxy, Proxy]
    ) -> tuple[list[Proxy], list[Proxy], list[Proxy], list[Proxy]]:
        # The nn module attributes are guaranteed to be registered into the top-level graph module during
        # higher order op speculation. Therefore, get_attr nodes in two branches with the same
        # target refer to the same attribute and we can safely deduplicate them with their target.
        #
        # Note: ideally, dynamo should just create a single proxy for the same attribute of a nn module. But
        # true_branch and false_branch belong to two separate tracing contexts, they may register the same
        # attribute to top level separately. This creates two get_attr proxies for the same attribute
        # that have different meta data such as stack_trace (one stack trace for the true_branch,
        # and the other for false_branch). It seems better to discard the proxy explicitly in cond
        # than make dynamo create a single proxy for the same get_attr target.
        def shared_getattrs(
            l_lifted_proxies: Sequence[Proxy], r_lifted_proxies: Sequence[Proxy]
        ) -> tuple[dict[Proxy, Proxy], dict[Proxy, Proxy]]:
            true_targets = {
                proxy.node.target: proxy
                for proxy in l_lifted_proxies
                if proxy.node.op == "get_attr"
            }
            l_shared_getattrs = {}
            r_shared_getattrs = {}

            for false_proxy in r_lifted_proxies:
                if (
                    false_proxy.node.op == "get_attr"
                    and false_proxy.node.target in true_targets
                ):
                    true_proxy = true_targets[false_proxy.node.target]
                    l_shared_getattrs[true_proxy] = true_proxy
                    r_shared_getattrs[false_proxy] = true_proxy
            return l_shared_getattrs, r_shared_getattrs

        l_shared_getattrs, r_shared_getattrs = shared_getattrs(
            l_lifted_freevars.keys(),  # type: ignore[arg-type]
            r_lifted_freevars.keys(),  # type: ignore[arg-type]
        )

        l_shared_freevars = (l_lifted_freevars.keys() & r_lifted_freevars.keys()).union(
            l_shared_getattrs.keys()
        )
        r_shared_freevars = (l_lifted_freevars.keys() & r_lifted_freevars.keys()).union(
            r_shared_getattrs.keys()
        )
        unique_l_freevars = l_lifted_freevars.keys() - l_shared_freevars
        unique_r_freevars = r_lifted_freevars.keys() - r_shared_freevars

        def _sort_by_name(vars: list[Proxy]) -> list[Proxy]:
            return sorted(vars, key=lambda var: var.node.name)

        return (
            list(_sort_by_name(list(l_shared_freevars))),
            list(_sort_by_name(list(r_shared_freevars))),
            list(_sort_by_name(list(unique_l_freevars))),
            list(_sort_by_name(list(unique_r_freevars))),
        )

    (l_shared, r_shared, unique_l, unique_r) = dedup_and_sort_lifted_freevars(
        l_lifted_freevars, r_lifted_freevars
    )

    # Let's say we capture cond(pred, true_fn, false_fn, (x,))
    # With set_graph_input set to automatic,
    # true_fn has lifted variables x, a, b, c
    # false_fn has lifted variables x, a, b, d
    # Then fixup_branch_inps make sure both branches have the same signature, i.e.:
    # - true_fn(x, a, b, c_true_branch, d_false_branch)
    # - false_fn(x, a, b, c_true_branch, d_false_branch)
    #
    # More formally, the signature has three parts in the following order:
    # 1. used in both branches: x, a, b
    # 2. only used in true branches: c, suffixed with _true_branch
    # 3. only used in false branches: d, suffixed with _false_branch
    # Within each part, we re-order the nodes by name to have a derterministic ordering for testing.
    def fixup_branch_inps(
        graph: torch.fx.Graph,
        lifted_freevars: dict[Proxy, Proxy],
        shared: Sequence[Proxy],
        unique_l: Sequence[Proxy],
        unique_r: Sequence[Proxy],
    ) -> None:
        def _insert_or_replace_phs(new_args: Sequence[Proxy], name_suffix: str) -> None:
            for arg in new_args:
                new_ph = graph.placeholder(arg.node.name + name_suffix)
                new_ph.meta = arg.node.meta
                # Override with new_ph if there exists a old placeholder.
                if arg in lifted_freevars:
                    old_ph = lifted_freevars[arg].node
                    old_ph.replace_all_uses_with(new_ph)
                    # replace_all_uses_with doesn't clean users. Clean it manually so that we could erase it.
                    old_ph.users = {}
                    graph.erase_node(old_ph)

        first_not_ph_node = next(
            node for node in graph.nodes if node.op != "placeholder"
        )
        with graph.inserting_before(first_not_ph_node):
            _insert_or_replace_phs(shared, "")
            _insert_or_replace_phs(unique_l, "_" + l_name)
            _insert_or_replace_phs(unique_r, "_" + r_name)

    fixup_branch_inps(l_graph, l_lifted_freevars, l_shared, unique_l, unique_r)
    fixup_branch_inps(r_graph, r_lifted_freevars, r_shared, unique_l, unique_r)
    return l_graph, r_graph, l_shared, r_shared, unique_l, unique_r


def move_lifted_freevars_phs_to_end(
    graph: torch.fx.Graph, lifted_freevars: dict[Proxy, Proxy]
) -> None:
    lifted_ph_set = {child_p.node for child_p in lifted_freevars.values()}

    prev_phs = [n for n in graph.nodes if n.op == "placeholder"]

    # No need to reorder when graph doesn't have args or doesn't
    # have lifted freevars or all inputs are lifted freevars.
    if (
        len(prev_phs) == 0
        or len(lifted_ph_set) == 0
        or len(prev_phs) == len(lifted_ph_set)
    ):
        return

    # Step 1: find first X1
    for x1 in prev_phs:
        if x1 in lifted_ph_set:
            break

    assert x1 is not None and x1.op == "placeholder"
    # Step 2: starting from the X1, skip Xs and prepend Os before X1.
    cand_x = x1.next
    while cand_x is not None and cand_x.op == "placeholder":
        if cand_x in lifted_ph_set:
            cand_x = cand_x.next
        else:
            nxt = cand_x.next
            cand_x._remove_from_list()
            x1.prepend(cand_x)
            cand_x = nxt

    # Step 3: assert that all placeholders are in the correct order as .
    # in lifted_freevars
    after_phs = [node for node in graph.nodes if node.op == "placeholder"][
        -len(lifted_freevars) :
    ]
    assert len(after_phs) == len(lifted_freevars)
    for child_proxy, ph in zip(lifted_freevars.values(), after_phs):
        assert child_proxy.node is ph, (
            "The order of placeholders is different from the order of lifted_freevars"
        )

    graph.lint()


def check_aliasing_and_input_mutation(
    subtracer: "SubgraphTracer",
    graph: torch.fx.Graph,
    supports_input_mutation: bool,
    supports_aliasing: bool,
    source_target: Optional["HigherOrderOperator"],
) -> None:
    name = source_target.name if source_target else "<UNKNOWN>"
    if not supports_input_mutation:
        mutation_info = subtracer.has_input_mutation()
        if mutation_info.has_mutation:
            context = f"{mutation_info.msg} in\n {graph}"
            unimplemented(
                gb_type="Encountered input mutation during higher order op tracing",
                context=context,
                explanation=f"Higher order ops do not support input mutation. Found in {name}",
                hints=[
                    "Consider using the debug context to change user code to avoid mutation.",
                    "Please open an issue.",
                ],
            )

    if not supports_aliasing:
        aliasing_info = subtracer.has_aliasing()
        if aliasing_info.has_aliasing:
            context = f"{aliasing_info.msg} in\n {graph}"
            unimplemented(
                gb_type="Encountered aliasing during higher order op tracing",
                context=context,
                explanation=f"Higher order ops do not support aliasing. Found in {name}",
                hints=[
                    "Replace `return input` with `return input.clone()` to avoid aliasing.",
                    "Consider using the debug context to change user code to avoid aliasing.",
                    "Please open an issue.",
                ],
            )


def trace_hop_function(
    f: VariableTracker,
    tx: "InstructionTranslator",
    subtracer: "SubgraphTracer",
    enable_grad: bool | None,
    restore_side_effects: bool,
    args: Sequence[VariableTracker],
    sub_kwargs: dict[str, VariableTracker],
) -> VariableTracker:
    # For autograd.Function and other legacy HOPs, we do NOT couple
    # restore_side_effects with allow_side_effects_in_hop.
    # This preserves the old behavior where:
    # - restore_side_effects=False means ctx mutations persist
    # - But non-ctx side effects still cause graph breaks (under_activation_checkpoint was False)
    enable_side_effects_with_extra_outputs = False

    autograd_ctx = (
        dynamo_enable_grad(tx, enable_grad)
        if enable_grad is not None
        else contextlib.nullcontext()
    )
    side_effects_ctx = (
        dynamo_allow_side_effects_in_hop(tx)
        if enable_side_effects_with_extra_outputs
        else contextlib.nullcontext()
    )

    # For handling side effects, we can make an argument that we don't
    # have to do anything here. The side effects infra does a good job
    # of graph breaking if we mutate any nonlocal or global variable
    # while subtracing. As a result if tracing succeeds, side effects
    # data structure will only contain read-only data structures that
    # are put there for tracking purposes.
    # But on the other hand, there is an argument that if we ever write
    # a new side effect in Dynamo which does not go through the side
    # effect infra, we can end up in bad state.
    # Therefore we restore the side effects after tracing. The catch is
    # that we have to special handle tensor variables. If we have seen a
    # nonlocal variable tensor during subtracing, we want to keep a
    # track of that tensor, so that later subtracing or the root tracer
    # itself does not create a new proxy for the already observed tensor
    # variable.
    prev_side_effects = None
    if restore_side_effects:
        prev_side_effects = tx.output.side_effects.clone()

    with autograd_ctx, side_effects_ctx:
        output = f.call_function(tx, args, sub_kwargs)

    if restore_side_effects:
        new_side_effects = tx.output.side_effects.clone()
        assert prev_side_effects is not None
        prev_side_effects.track_runahead_tensor_and_symvar_side_effects(
            new_side_effects
        )
        tx.output.side_effects = prev_side_effects
    return output


def trace_hop_function_with_auto_output_flattening(
    f: VariableTracker,
    tx: "InstructionTranslator",
    subtracer: "SubgraphTracer",
    enable_grad: bool | None,
    allow_side_effects: bool,
    args: Sequence[VariableTracker],
    sub_kwargs: dict[str, VariableTracker],
) -> VariableTracker:
    autograd_ctx = (
        dynamo_enable_grad(tx, enable_grad)
        if enable_grad is not None
        else contextlib.nullcontext()
    )
    side_effects_ctx = (
        dynamo_allow_side_effects_in_hop(tx)
        if allow_side_effects
        else contextlib.nullcontext()
    )

    with autograd_ctx, side_effects_ctx:
        output = f.call_function(tx, args, sub_kwargs)

    return output


def get_hop_args(
    tx: "InstructionTranslator",
    f: VariableTracker,
    subtracer: "SubgraphTracer",
    sub_args: list[VariableTracker],
    sub_kwargs: dict[str, VariableTracker],
    set_subgraph_inputs: str,
    description: str,
) -> list[VariableTracker]:
    sub_args_names = maybe_positional_arg_names(f)
    # User mismatch in the number of args. Will eventually lead to an error.
    if sub_args_names is not None and len(sub_args_names) < len(sub_args):
        sub_args_names = None
    args = validate_args_and_maybe_create_graph_inputs(
        sub_args,
        subtracer,
        tx,
        set_subgraph_inputs,
        description,
        sub_args_names,
    )

    validate_args_and_maybe_create_graph_inputs(
        sub_kwargs.values(),  # type: ignore[arg-type]
        subtracer,
        tx,
        set_subgraph_inputs="automatic",
        description=description,
    )
    return args


def speculate_subgraph_with_auto_output_flattening(
    tx: "InstructionTranslator",
    f: VariableTracker,
    sub_args: Sequence[VariableTracker],
    sub_kwargs: dict[str, VariableTracker] | None,
    description: str,
    *,
    # source_target is the .value of HigherOrderOpVariable and is the
    # target of the proxy that we created for the higherOrderOperator.
    source_target: HigherOrderOperator | None = None,
    enable_grad: bool | None = None,
    # automatic: relies on Dynamo to find the used tensors and lift them as
    # inputs.
    #
    # automatic_with_forced_inputs: relies on the function arg names to create
    # a new proxy. Also, it will always INSERT a tensor placeholder as input,
    # even though it might not be used in the graph and they will also be in the
    # same order as the original function (as opposed to automatic which will
    # not insert the unused placeholder and can insert other placeholders in the
    # order they are see while tracing). This is useful for autograd.Function
    # backward where we do need to account for all the inputs of the backwards
    # to be lifted as inputs for making the fwd-bwd graph consistent.
    set_subgraph_inputs: Literal[
        "automatic", "automatic_with_forced_inputs", "flatten_manual", "manual"
    ] = "automatic",
    # If True, exposes intermediates to subgraph outputs to allow later tensor ops to
    # access intermediates from the subgraph, this is useful for mutation
    allow_side_effects: bool = False,
    # Controls whether to filter aliased intermediates when collecting extra outputs.
    # This is only relevant when allow_side_effects=True.
    # - True: Filter out intermediates that alias with inputs or outputs (strict, for invoke_subgraph)
    # - False: Allow aliased intermediates (for checkpoint/autograd.Function which get desugared/inlined)
    #
    # Example where filtering is needed:
    #
    #   @invoke_subgraph
    #   def gn(x):
    #       view = x.view(2, 4)  # intermediate that aliases input x
    #       y = torch.sin(view)
    #       return torch.cos(view)
    #
    #   def fn(x):
    #       res = gn(x)
    #       return res + 4
    #
    # In this case, if we don't filter `view`, we would later error because some HOPs
    # have strict aliasing checks on inputs/outputs.
    #
    # This does however introduce a subtle issue when we do something like:
    #
    #   captured = []
    #
    #   @invoke_subgraph
    #   def gn(x):
    #       view = x.view(2, 4)  # intermediate that aliases input x
    #       y = torch.sin(view)
    #       captured.append(view)
    #       return torch.cos(view)
    #
    #   def fn(x):
    #       res = gn(x)
    #       return res + captured[0]
    #
    # In this case, we will not replay the side effect on `captured` in the graph,
    # which fails with a not-so-nice error. We will address this in a follow-up PR
    # because this case is rare. This is not a regression because side effects were
    # never supported for invoke_subgraph anyway.
    filter_aliased_intermediates: bool = False,
    # TODO - supports input_mutation and aliasing should be False by default for strictness
    supports_input_mutation: bool = True,
    supports_aliasing: bool = True,
    # Pass in an originating tracer - this is needed for preserving context
    # across fwd-bwd for autograd.Function
    tracer: Optional["SubgraphTracer"] = None,
) -> tuple[
    VariableTracker,  # output: The VT that Dynamo continues tracing with
    torch.fx.Graph,  # graph: The FX graph representing the subgraph computation
    dict[
        torch.fx.Proxy, torch.fx.Proxy
    ],  # lifted_freevars: Free variables lifted as inputs
    VariableTracker
    | tuple[
        VariableTracker, ...
    ],  # graph_output_vts: Tensor/symint VTs that are actual FX graph outputs
    SubgraphTracingInfo,  # tracing_info: Properties observed during subgraph tracing
]:
    """
    Speculate subgraph for Higher-Order Operators (HOPs) with automatic output flattening.

    ## Automatic output flattening

    For many HOPs, the representation exists only as a container for the
    subgraph. In later compiler stages or at runtime, the HOP is desugared and
    simply executes the subgraph directly, as if it were inlined. For such hops,
    we follow automatic output flattening.
    For example:
    - invoke_subgraph
    - activation checkpointing (torch.utils.checkpoint.checkpoint)
    - autograd.Function
    - nested_compile_region

    This is in contrast to control flow HOPs which do not follow this desugaring:
    - torch.cond (conditional execution based on predicate)
    - torch.while_loop (iterative execution)
    - torch.map (parallel execution over batch dimension)

    For control flow HOPs, the HOP behavior is fundamentally different from just
    running the body function once.

    ## Key Advantage: Disentangling VTs from Graph Outputs

    Desugaring simplify HOP processing by allowing us to disentangle the output
    variable trackers (VTs) from the HOP subgraph outputs. This mirrors typical
    Dynamo processing where:
    - VTs "run ahead" representing the program state for continued tracing
    - The graph is a side data structure tracking computation seen so far

    This separation is crucial for HOPs with non-proxyable outputs (e.g., custom
    user-defined objects containing tensors). The function may return complex Python
    objects for Dynamo to continue tracing, but only the tensor/symint VTs need to
    be registered as actual FX graph outputs.

    Example:
        class Foo:
            def __init__(self, a, b):
                self.a = a  # tensor
                self.b = b  # tensor

        def gn(x):
            return Foo(torch.sin(x), torch.cos(x))

        result = some_hop(gn, x)  # Returns Foo instance
        out = result.a + result.b  # Dynamo can continue tracing

    Here, `output` VT is a UserDefinedObjectVariable wrapping Foo, but
    `graph_output_vts` contains only the tensor VTs (a and b) that should be
    actual FX graph outputs. This allows Dynamo to continue tracing with the
    Foo object while the graph only needs to output the constituent tensors.

    ## Return Values

    Unlike `speculate_subgraph`, this function returns:
    - output: The VT that Dynamo continues tracing with (may be complex Python objects)
    - graph: The FX graph representing the subgraph computation
    - lifted_freevars: Free variables lifted as inputs to the subgraph
    - graph_output_vts: Only the tensor/symint VTs that are actual FX graph outputs

    The key difference is `graph_output_vts` instead of `treespec`, which gives more
    flexibility for handling non-proxyable outputs.
    """
    if sub_kwargs is None:
        sub_kwargs = {}

    assert set_subgraph_inputs in {
        "automatic",
        "automatic_with_forced_inputs",
        "flatten_manual",
        "manual",
    }, "Please use one of the supported set_subgraph_inputs options."

    # See NOTE [Temporary argument `set_subgraph_inputs`]
    if sub_kwargs and set_subgraph_inputs != "automatic":
        unimplemented(
            gb_type="invalid set_subgraph_inputs and sub_kwargs settings",
            context=f"set_subgraph_inputs: {set_subgraph_inputs}, sub_kwargs: {sub_kwargs}",
            explanation="`sub_kwargs` cannot be used when `set_subgraph_inputs` is not set to 'automatic'.",
            hints=[
                "Use `set_subgraph_inputs='automatic'` when passing `sub_kwargs`.",
                *graph_break_hints.USER_ERROR,
            ],
        )

    try:
        with tx.output.subtracer(source_target, tracer, description) as subtracer:
            args = get_hop_args(
                tx,
                f,
                subtracer,
                list(sub_args),
                sub_kwargs,
                set_subgraph_inputs,
                description,
            )

            # Special case - if users uses
            # `traced_with_externally_visible_side_effects`, we still need to
            # return the intermediates as outputs. However, this API gets
            # triggered during the hop tracing,  and we don't know at this point
            # of time, if the API will take into effect. To handle this, we have
            # a flag traced_with_externally_visible_side_effects (default=False)
            # that is set to True anytime
            # `traced_with_externally_visible_side_effects` is set. We reset it
            # with the old value after the hop is traced out.
            old_value = (
                tx.output.current_tracer.traced_with_externally_visible_side_effects
            )

            output = trace_hop_function_with_auto_output_flattening(
                f,
                tx,
                subtracer,
                enable_grad,
                allow_side_effects,
                args,
                sub_kwargs,
            )

            # NOTE: [Separation of graph outputs and output VTs]
            # In Dynamo (outside of speculate_subgraph), VTs and the graph are
            # separate concepts:
            # - VTs (VariableTrackers) can "run ahead" and continue Dynamo tracing
            # - The graph is just a side data structure tracking computation seen so far
            #
            # This separation is crucial for HOPs with non-proxyable outputs (e.g.,
            # custom user-defined objects containing tensors). The function may return
            # complex Python objects for Dynamo to continue tracing, but only the
            # tensor/symint VTs need to be registered as actual graph outputs.
            #
            # Example:
            #   class Foo:
            #       def __init__(self, a, b):
            #           self.a = a  # tensor
            #           self.b = b  # tensor
            #
            #   def gn(x):
            #       return Foo(torch.sin(x), torch.cos(x))
            #
            # Here, `output` VT is a UserDefinedObjectVariable wrapping Foo, but
            # `graph_output_vts` contains only the tensor VTs (a and b) that should
            # be actual FX graph outputs.
            # Collect only tensor and symint VTs that should be graph outputs.
            # We walk the output structure and extract proxyable VTs.
            graph_output_vt_list = []

            def visit(vt: VariableTracker) -> None:
                if vt.is_tensor() or isinstance(
                    vt, (SymNodeVariable, TorchScriptObjectVariable)
                ):
                    graph_output_vt_list.append(vt)

            VariableTracker.visit(visit, output)
            graph_output_vts = tuple(graph_output_vt_list)

            # NOTE - [Return subgraph intermediates as subgraph outputs]
            # This helps HOPs which allow side effects. Consider the
            # following example
            #
            # def gn(x, z):
            #     o = torch.matmul(x, x) @ x
            #     out = x.sin()
            #     z.append(out)
            #     return torch.cos(torch.sin(o))

            # def fn(x):
            #     z = []
            #     out1 = torch.utils.checkpoint.checkpoint(
            #         gn,
            #         x,
            #         z,
            #         use_reentrant=False,
            #     )
            #     return out1, z[0]
            #
            # In this example, list `z` is in outer scope and gets appended
            # in the subgraph with `out`. But `out` is not an output of the
            # subgraph. This can cause issue because later on when the outer
            # graph returns `z[0]` it needs to have access to the graph node
            # `out`. To solve this problem, we just return all intermediates
            # from the subgraph.

            # TODO - Today this is supported only for AC. AC HOP gets
            # desugared in AOTDispatcher so even though subgraph has extra
            # unused outputs in Dynamo, its ok even if we don't DCE them in
            # Dynamo. As AOTDispatcher desugars/inlines the subgraph, the
            # subgraph boundary disappears. And even for AC, today this only
            # works when the skip_fwd_side_effects_in_bwd_under_checkpoint
            # flag is True, i.e., only when we allow side-effects. But, we
            # want this to be supported for other Hops as well, specifically
            # nested_compile_region and autograd.Function. Today, its safe
            # because we error out on seeing a side-effect.

            allow_side_effects = (
                allow_side_effects
                or tx.output.current_tracer.traced_with_externally_visible_side_effects
            )
            if allow_side_effects:
                extra_outputs = collect_intermediate_outputs(
                    tx, subtracer, graph_output_vts, filter_aliased_intermediates
                )
                graph_output_vts = graph_output_vts + tuple(extra_outputs)

            tx.output.current_tracer.traced_with_externally_visible_side_effects = (
                old_value
            )

            validate_subgraph_output_types(graph_output_vts)

            # The output proxies might not belong to this SubgraphTracer
            # (if they are free variables that were never lifted)
            # so lift them here.
            # output_proxies = output.as_proxy()
            if isinstance(graph_output_vts, tuple):
                output_proxies = [a.as_proxy() for a in graph_output_vts]  # type: ignore[attr-defined]
                output_proxies = pytree.tree_map(
                    subtracer.maybe_lift_tracked_freevar_to_input, output_proxies
                )
                output_proxies = tuple(output_proxies)
            else:
                output_proxies = output.as_proxy()
                output_proxies = pytree.tree_map(
                    subtracer.maybe_lift_tracked_freevar_to_input, output_proxies
                )

            tx.output.create_node(
                "output",
                "output",
                (subtracer.create_arg((output_proxies,))),
                {},
            )
            graph = tx.output.graph
            graph.lint()
            lifted_freevars = subtracer.lifted_freevars

            if len(lifted_freevars) > 0:
                move_lifted_freevars_phs_to_end(graph, lifted_freevars)

            check_aliasing_and_input_mutation(
                subtracer,
                graph,
                supports_input_mutation,
                supports_aliasing,
                source_target,
            )
            # Return both the output VT and the graph output VTs separately:
            # - `output`: The VT that Dynamo continues tracing with (may be
            #   complex Python objects, tuples, dicts, etc.)
            # - `graph`: The FX graph representing the subgraph computation
            # - `lifted_freevars`: Free variables lifted as inputs to the subgraph
            # - `graph_output_vts`: Only the tensor/symint VTs that are actual
            #   FX graph outputs (basically the vts associated with graph outputs)
            # - `tracing_info`: Properties observed during subgraph tracing
            tracing_info = SubgraphTracingInfo(
                side_effect_stack=subtracer.side_effect_stack,
                traced_sources=subtracer.traced_sources,
            )

            return (
                output,
                graph,
                lifted_freevars,
                graph_output_vts,
                tracing_info,
            )
    except Unsupported as ex:
        f_name = f"{type(f).__name__}"
        if isinstance(f, UserFunctionVariable):
            f_name = f.get_name()
        msg = (
            f"speculate_subgraph: while introspecting {description}, we were unable "
            f"to trace function `{f_name}` into a single graph. This means "
            f"that Dynamo was unable to prove safety for this API and will "
            f"fall back to eager-mode PyTorch, which could lead to a slowdown."
        )
        log.info(msg)
        log.info(ex)  # noqa: G200
        raise ex


def speculate_subgraph(
    tx: "InstructionTranslator",
    f: VariableTracker,
    sub_args: Sequence[VariableTracker],
    sub_kwargs: dict[str, VariableTracker] | None,
    description: str,
    *,
    # source_target is the .value of HigherOrderOpVariable and is the
    # target of the proxy that we created for the higherOrderOperator.
    source_target: HigherOrderOperator | None = None,
    always_restore: bool = False,
    enable_grad: bool | None = None,
    # NOTE [argument `set_subgraph_inputs`]
    # set_subgraph_inputs controls what how to construct subgraphs' placeholders from sub_args.
    # 1. if your HOP supports arbitrary inputs, use set_subgraph_inputs="automatic" (most recommended).
    # 2. if your HOP supports only Tensor and symnode inputs, use set_subgraph_inputs="flatten_manual" (recommended).
    # If sub_args contain Pytree structure (e.g. dict/list/tuple/set), the sub_args will be flattened first.
    # Then the flattened args are manually set as subgraph's placeholders.
    # 3. if your HOP must preserve inputs that are not tensor or symnode as placeholders e.g. AutogradFunctionContextVariable
    # use set_subgraph_inputs="manual" (not recommended). We do not recommend it in general because it has the
    # restriction that user need to manually control how to create placeholders and VariableTrackers for the args.
    set_subgraph_inputs: Literal[
        "automatic", "semi_automatic", "flatten_manual", "manual"
    ] = "automatic",
    restore_side_effects: bool = True,
    should_flatten_outputs: bool = False,
    # if should_flatten_outputs is True, `remove_consts_from_outputs` remove the
    # const outputs from the subgraph output.
    remove_consts_from_outputs: bool = True,
    # TODO - supports input_mutation and aliasing should be False by default for strictness
    supports_input_mutation: bool = True,
    supports_aliasing: bool = True,
    # Pass in an originating tracer - this is needed for preserving context
    # across fwd-bwd for autograd.Function
    tracer: Optional["SubgraphTracer"] = None,
) -> tuple[tuple[VariableTracker, OutputSpec], torch.fx.Graph, dict[Proxy, Proxy]]:
    if sub_kwargs is None:
        sub_kwargs = {}

    from ..builder import SourcelessBuilder

    assert set_subgraph_inputs in {
        "automatic",
        "automatic_with_forced_inputs",
        "flatten_manual",
        "manual",
    }, "Please use one of the supported set_subgraph_inputs options."

    # See NOTE [Temporary argument `set_subgraph_inputs`]
    if sub_kwargs and set_subgraph_inputs != "automatic":
        unimplemented(
            gb_type="invalid set_subgraph_inputs and sub_kwargs settings",
            context=f"set_subgraph_inputs: {set_subgraph_inputs}, sub_kwargs: {sub_kwargs}",
            explanation="`sub_kwargs` cannot be used when `set_subgraph_inputs` is not set to 'automatic'.",
            hints=[
                "Use `set_subgraph_inputs='automatic'` when passing `sub_kwargs`.",
                *graph_break_hints.USER_ERROR,
            ],
        )

    try:
        # ensure guards on args get installed in parent subgraph
        f, sub_args, sub_kwargs = LazyVariableTracker.realize_all(
            (f, sub_args, sub_kwargs),
        )

        with tx.output.subtracer(source_target, tracer, description) as subtracer:
            args = get_hop_args(
                tx, f, subtracer, sub_args, sub_kwargs, set_subgraph_inputs, description
            )

            output = trace_hop_function(
                f,
                tx,
                subtracer,
                enable_grad,
                restore_side_effects,
                args,
                sub_kwargs,
            )

            treespec = None
            masks_to_filter_const_values = None
            const_values = None
            if should_flatten_outputs:
                from torch._dynamo.external_utils import filter_out_const_values

                # Flatten the speculated subgraph output.
                output, treespec = _make_inlined(tx, pytree.tree_flatten)(
                    output
                ).unpack_var_sequence(tx)

                # Actually, transform the list (returned by flatten) into a tuple
                # for dynamo consistency.
                output = SourcelessBuilder.create(tx, tuple).call_function(
                    tx, [output], {}
                )

                if remove_consts_from_outputs:
                    # Filter out the constants and save them into a spec. Filtering
                    # out constants makes the graph simpler for the backends. We
                    # need to ensure that after unflattening the constants are
                    # inserted back at the right positions for the Dynamo tracing to
                    # continue. This is done by filter_const_spec
                    output_proxies = output.as_proxy()
                    masks_to_filter_const_values = pytree.tree_map(
                        lambda x: not isinstance(x, torch.fx.Proxy), output_proxies
                    )
                    const_values = pytree.tree_map(
                        lambda x: None if isinstance(x, torch.fx.Proxy) else x,
                        output_proxies,
                    )
                    output = _make_inlined(tx, filter_out_const_values)(
                        output, masks_to_filter_const_values
                    )

            # TODO - clean up num_intermediate_nodes_as_outputs - we do not need
            # after AC moved to auto_output_flattening
            num_intermediate_nodes_as_outputs = 0
            # Register output to graph
            # Modeled off of compile_and_call_fx_graph
            # TODO: support pytree output
            # We check always_restore because we dont use the output or side effects of always_restore code,
            # like bwd.
            if always_restore:
                # Nothing left to do here
                return (
                    (
                        output,
                        OutputSpec(
                            treespec,  # type: ignore[arg-type]
                            masks_to_filter_const_values,
                            const_values,
                            num_intermediate_nodes_as_outputs,
                        ),
                    ),
                    tx.output.graph,
                    subtracer.lifted_freevars,
                )
            else:
                validate_subgraph_output_types(output)

                # The output proxies might not belong to this SubgraphTracer
                # (if they are free variables that were never lifted)
                # so lift them here.
                output_proxies = output.as_proxy()
                output_proxies = pytree.tree_map(
                    subtracer.maybe_lift_tracked_freevar_to_input, output_proxies
                )

                tx.output.create_node(
                    "output",
                    "output",
                    (subtracer.create_arg((output_proxies,))),
                    {},
                )
                graph = tx.output.graph
                graph.lint()
                lifted_freevars = subtracer.lifted_freevars

                if len(lifted_freevars) > 0:
                    move_lifted_freevars_phs_to_end(graph, lifted_freevars)
                check_aliasing_and_input_mutation(
                    subtracer,
                    graph,
                    supports_input_mutation,
                    supports_aliasing,
                    source_target,
                )

                return (
                    (
                        output,
                        OutputSpec(
                            treespec,  # type: ignore[arg-type]
                            masks_to_filter_const_values,
                            const_values,
                            num_intermediate_nodes_as_outputs,
                        ),
                    ),
                    graph,
                    lifted_freevars,
                )

    except Unsupported as ex:
        f_name = f"{type(f).__name__}"
        if isinstance(f, UserFunctionVariable):
            f_name = f.get_name()
        msg = (
            f"speculate_subgraph: while introspecting {description}, we were unable "
            f"to trace function `{f_name}` into a single graph. This means "
            f"that Dynamo was unable to prove safety for this API and will "
            f"fall back to eager-mode PyTorch, which could lead to a slowdown."
        )
        log.info(msg)
        log.info(ex)  # noqa: G200
        raise ex


def make_attr(tx: "InstructionTranslator", name: str) -> Proxy:
    node = tx.output.create_proxy(
        "get_attr",
        name,
        (),
        {},
    )
    return node


def add_hop_context(cls: type[HOP_VT_Alias]) -> type[HOP_VT_Alias]:
    """
    Class decorator that adds HOP context to exceptions raised in call_function.

    Requires the class to have _HOP_NAME and _ALLOW_FALLBACK_TO_EAGER set.
    """

    if hasattr(cls.call_method, "_hop_wrapped"):
        return cls

    if cls._HOP_NAME is None:
        raise TypeError(f"{cls.__name__} must define _HOP_NAME class attribute.")
    if cls._ALLOW_FALLBACK_TO_EAGER is None:
        raise TypeError(
            f"{cls.__name__} must define _ALLOW_FALLBACK_TO_EAGER class attribute."
        )

    original_call_function = cls.call_function

    @functools.wraps(original_call_function)
    def wrapped_call_function(self, *args: Any, **kwargs: Any) -> VariableTracker:
        try:
            return original_call_function(self, *args, **kwargs)
        except UncapturedHigherOrderOpError as e:
            if not hasattr(e, "_hop_name"):
                e._hop_name = self._HOP_NAME  # pyrefly: ignore[missing-attribute]
            raise
        except (Unsupported, ObservedException) as e:
            # Only tag if not already tagged (reports deepest HOP only)
            if hasattr(e, "_hop_name"):
                raise

            if self._ALLOW_FALLBACK_TO_EAGER:
                # Tag the exception with HOP name for later formatting in exc.py
                # NOTE: because nested graph breaks are NOT supported on HOPs, we will
                # NEVER log a HOP graph break before running this
                e._hop_name = self._HOP_NAME  # pyrefly: ignore[missing-attribute]
                raise
            else:
                real_stack = getattr(e, "real_stack", None)
                full_msg = (
                    "This higher order operator doesn't work unless it is "
                    "captured completely with torch.compile. Got graph break/error:"
                    f"\n\n{str(e)}"
                )
                exc = UncapturedHigherOrderOpError(full_msg, real_stack)
                exc._hop_name = self._HOP_NAME  # pyrefly: ignore[missing-attribute]
                raise exc.with_traceback(e.__traceback__) from None

    wrapped_call_function._hop_wrapped = True  # pyrefly: ignore[missing-attribute]
    cls.call_function = wrapped_call_function
    return cls


class TorchHigherOrderOperatorVariable(VariableTracker):
    # Subclasses should set _HOP_NAME to enable automatic HOP context in error messages
    _HOP_NAME: str | None = None
    # Set to False for HOPs that hard error on graph break (e.g., cond, map, scan); otherwise
    # HOPs will fall back to eager.
    _ALLOW_FALLBACK_TO_EAGER: bool = True

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        add_hop_context(cls)

    def __init__(
        self, value: HigherOrderOperator, source: Source | None = None, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.value = value
        self.source = source

    @staticmethod
    def make(
        value: HigherOrderOperator, source: Source | None = None, **kwargs: Any
    ) -> "TorchHigherOrderOperatorVariable":
        variable_class = _hop_name_to_variable_class.get(value.__name__)
        if variable_class is not None:
            return variable_class(value, source, **kwargs)

        from torch._higher_order_ops import BaseHOP

        from .autograd import BaseHOPVariable

        if isinstance(value, BaseHOP):
            return BaseHOPVariable(value, source, **kwargs)
        unimplemented(
            gb_type="unsupported HigherOrderOperator",
            context=str(value),
            explanation=f"Unable to create higher order operator variable for {value.__name__}.",
            hints=[
                *graph_break_hints.DYNAMO_BUG,
            ],
        )

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from ..torch_function import (
            can_dispatch_torch_function,
            dispatch_torch_function,
        )

        if can_dispatch_torch_function(tx, args, kwargs):
            return dispatch_torch_function(tx, self, args, kwargs)

        return self._call_function(tx, args, kwargs)

    def _call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        unimplemented(
            gb_type="unsupported HigherOrderOperator function call",
            context=str(self.value),
            explanation=f"Unable to trace calling higher order operator variable for {self.value.__name__}.",
            hints=[
                *graph_break_hints.DYNAMO_BUG,
            ],
        )

    def as_python_constant(self) -> HigherOrderOperator:
        return self.value

    def is_python_hashable(self) -> bool:
        return True

    def get_python_hash(self) -> int:
        return hash(self.as_python_constant())

    def is_python_equal(self, other: object) -> bool:
        return (
            isinstance(other, VariableTracker)
            and self.as_python_constant() == other.as_python_constant()
        )


def _get_fake_value(x: Union[VariableTracker, Proxy, "FakeTensor"]) -> "FakeTensor":
    if isinstance(x, variables.VariableTracker):
        return x.as_proxy().node.meta["example_value"]
    elif isinstance(x, torch.fx.Proxy):
        return x.node.meta["example_value"]
    else:
        return x


def maybe_positional_arg_names(func: VariableTracker) -> list[str] | None:
    result = []
    if not hasattr(func, "get_function"):
        return None
    try:
        fn = func.get_function()
    except (Unsupported, NotImplementedError):
        return None
    try:
        sig = inspect.signature(fn)
    except ValueError:
        return None
    for name, param in sig.parameters.items():
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            return None
        if (
            param.kind is inspect.Parameter.POSITIONAL_ONLY
            or param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        ):
            if name == "self":
                # FX graphs can't have a placeholder named self
                result.append("self_")
            else:
                result.append(name)
    return result


_hop_name_to_variable_class: dict[str, type[TorchHigherOrderOperatorVariable]] = {}
