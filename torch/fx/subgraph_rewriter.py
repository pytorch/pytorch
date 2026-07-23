import copy
import logging
import operator
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import Any, NamedTuple, TYPE_CHECKING

import torch
from torch._guards import detect_fake_mode

from ._compatibility import compatibility
from ._symbolic_trace import symbolic_trace
from .graph import Graph
from .graph_module import GraphModule
from .immutable_collections import immutable_dict, immutable_list
from .node import map_arg, Node
from .operator_schemas import get_signature_for_torch_op


if TYPE_CHECKING:
    from torch._subclasses.fake_tensor import FakeTensorMode

    from .passes.utils.matcher_with_name_node_map_utils import InternalMatch


_SAFE_META_PROPAGATION_BUILTINS = (
    operator.add,
    operator.floordiv,
    operator.getitem,
    operator.matmul,
    operator.mod,
    operator.mul,
    operator.neg,
    operator.pos,
    operator.pow,
    operator.sub,
    operator.truediv,
)
_TORCH_BUILTIN_TYPE = type(torch.sin)
_META_PROPAGATION_OPERATOR_EXCEPTIONS = (
    ArithmeticError,
    LookupError,
    RuntimeError,
    TypeError,
    ValueError,
)
_SAFE_META_SCALAR_TYPES = (
    bool,
    bytes,
    complex,
    float,
    int,
    str,
    type(None),
    torch.device,
    torch.dtype,
    torch.layout,
    torch.memory_format,
    torch.SymBool,
    torch.SymFloat,
    torch.SymInt,
)
_MISSING = object()
log = logging.getLogger(__name__)

__all__ = [
    "Match",
    "replace_pattern",
    "replace_pattern_with_filters",
    "ReplacedPatterns",
]


@compatibility(is_backward_compatible=True)
class Match(NamedTuple):
    # Node from which the match was found
    anchor: Node
    # Maps nodes in the pattern subgraph to nodes in the larger graph
    nodes_map: dict[Node, Node]


@compatibility(is_backward_compatible=False)
@dataclass
class ReplacedPatterns:
    # Node from which the match was found
    anchor: Node
    # Maps nodes in the pattern subgraph to nodes in the larger graph
    nodes_map: dict[Node, Node]
    # List of nodes that were added into the graph
    replacements: list[Node]


def _get_module_attr(module: torch.nn.Module | None, target: Any) -> Any:
    if module is None or type(target) is not str:
        return _MISSING
    module_path, _, attr_name = target.rpartition(".")
    try:
        owner = module.get_submodule(module_path)
    except AttributeError:
        return _MISSING
    attr = getattr(owner, attr_name, None)
    return _MISSING if attr is None else attr


def _get_module_attr_static(module: torch.nn.Module | None, target: Any) -> Any:
    if module is None or type(target) is not str:
        return _MISSING
    module_path, _, attr_name = target.rpartition(".")
    owner = module
    if module_path:
        for atom in module_path.split("."):
            modules = object.__getattribute__(owner, "_modules")
            if type(modules) is not dict or atom not in modules:
                return _MISSING
            owner = modules[atom]
            if owner is None:
                return _MISSING

    for mapping_name in ("_parameters", "_buffers", "_modules"):
        mapping = object.__getattribute__(owner, mapping_name)
        if type(mapping) is dict and attr_name in mapping:
            attr = mapping[attr_name]
            return _MISSING if attr is None else attr
    instance_attrs = object.__getattribute__(owner, "__dict__")
    attr = instance_attrs.get(attr_name, _MISSING)
    return _MISSING if attr is None else attr


def _replace_attributes(gm: GraphModule, replacement: torch.nn.Module) -> None:
    gm.delete_all_unused_submodules()

    if isinstance(replacement, GraphModule):
        replacement.graph.lint()

    for node in gm.graph.nodes:
        if node.op == "call_module" or node.op == "get_attr":
            gm_attr = _get_module_attr(gm, node.target)
            replacement_attr = _get_module_attr(replacement, node.target)

            # CASE 1: This target already exists as an attribute in our
            # result GraphModule. Whether or not it exists in
            # `replacement`, the existing submodule takes precedence.
            if gm_attr is not _MISSING:
                continue

            # CASE 2: The target exists as an attribute in `replacement`
            # only, so we need to copy it over.
            elif replacement_attr is not _MISSING:
                new_attr = copy.deepcopy(replacement_attr)
                if isinstance(replacement_attr, torch.nn.Module):
                    gm.add_submodule(node.target, new_attr)
                else:
                    setattr(gm, node.target, new_attr)

            # CASE 3: The target doesn't exist as an attribute in `gm`
            # or `replacement`
            else:
                raise RuntimeError(
                    'Attempted to create a "',
                    node.op,
                    '" node during subgraph rewriting '
                    f"with target {node.target}, but "
                    "the referenced attribute does not "
                    "exist in the replacement GraphModule",
                )

    gm.graph.lint()


def _is_exact_type(value_type: type[Any], allowed_types: tuple[type[Any], ...]) -> bool:
    return any(value_type is allowed_type for allowed_type in allowed_types)


@cache
def _torch_return_type_classes() -> tuple[type[Any], ...]:
    return tuple(
        candidate
        for name in dir(torch.return_types)
        if isinstance(candidate := getattr(torch.return_types, name), type)
    )


def _is_torch_return_type(value: Any) -> bool:
    value_type = type(value)
    return any(
        value_type is return_type for return_type in _torch_return_type_classes()
    )


def _is_safe_tensor_meta_value(value: Any) -> bool:
    value_type = type(value)
    if value_type is torch.Tensor or value_type is torch.nn.Parameter:
        return True
    FakeTensor, _, _ = _fake_tensor_meta_helpers()
    return value_type is FakeTensor


class _MetaValueAnalysis(NamedTuple):
    is_safe: bool
    contains_tensor: bool


def _analyze_meta_value(value: Any) -> _MetaValueAnalysis:
    active_container_ids: set[int] = set()
    completed_container_analyses: dict[int, _MetaValueAnalysis] = {}

    def visit(current: Any) -> _MetaValueAnalysis:
        current_type = type(current)
        if _is_safe_tensor_meta_value(current):
            return _MetaValueAnalysis(True, True)
        if (
            _is_exact_type(current_type, _SAFE_META_SCALAR_TYPES)
            or current_type is torch.Size
        ):
            return _MetaValueAnalysis(True, False)

        children: list[Any]
        if current_type is slice:
            children = [current.start, current.stop, current.step]
        elif current_type is not tuple and _is_torch_return_type(current):
            children = list(current)
        elif _is_exact_type(current_type, (list, tuple, immutable_list)):
            children = list(current)
        elif _is_exact_type(current_type, (dict, immutable_dict)):
            children = [child for item in current.items() for child in item]
        else:
            return _MetaValueAnalysis(False, False)

        current_id = id(current)
        completed_analysis = completed_container_analyses.get(current_id)
        if completed_analysis is not None:
            return completed_analysis
        if current_id in active_container_ids:
            return _MetaValueAnalysis(False, False)
        active_container_ids.add(current_id)
        try:
            contains_tensor = False
            for child in children:
                child_analysis = visit(child)
                if not child_analysis.is_safe:
                    analysis = _MetaValueAnalysis(False, False)
                    break
                contains_tensor = contains_tensor or child_analysis.contains_tensor
            else:
                analysis = _MetaValueAnalysis(True, contains_tensor)
        finally:
            active_container_ids.remove(current_id)
        completed_container_analyses[current_id] = analysis
        return analysis

    try:
        return visit(value)
    except RecursionError:
        return _MetaValueAnalysis(False, False)


def _contains_tensor(value: Any) -> bool:
    return _analyze_meta_value(value).contains_tensor


def _is_safe_meta_value(value: Any) -> bool:
    return _analyze_meta_value(value).is_safe


@cache
def _fake_tensor_meta_helpers() -> tuple[type[Any], type[Any], Callable[..., Any]]:
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
    from torch.fx.experimental.proxy_tensor import snapshot_fake

    return FakeTensor, FakeTensorMode, snapshot_fake


@cache
def _fake_tensor_propagation_exceptions() -> tuple[type[Exception], ...]:
    from torch._subclasses.fake_tensor import (
        DataDependentOutputException,
        DynamicOutputShapeException,
        UnsupportedFakeTensorException,
        UnsupportedOperatorException,
    )

    return (
        DataDependentOutputException,
        DynamicOutputShapeException,
        UnsupportedFakeTensorException,
        UnsupportedOperatorException,
    )


def _copy_meta_val(
    value: Any,
    fake_mode: "FakeTensorMode | None" = None,
    memo: dict[int, Any] | None = None,
) -> Any:
    if memo is None:
        memo = {}
    value_type = type(value)
    value_id = id(value)
    if value_id in memo:
        return memo[value_id]
    if value_type is torch.Tensor or value_type is torch.nn.Parameter:
        _, FakeTensorMode, snapshot_fake = _fake_tensor_meta_helpers()
        if fake_mode is None:
            fake_mode = FakeTensorMode(allow_fallback_kernels=False)
        with fake_mode:
            result = snapshot_fake(fake_mode.from_tensor(value, static_shapes=True))
        memo[value_id] = result
        return result
    if _is_safe_tensor_meta_value(value):
        _, _, snapshot_fake = _fake_tensor_meta_helpers()
        result = snapshot_fake(value)
        memo[value_id] = result
        return result
    if _is_exact_type(value_type, (list, immutable_list)):
        values: list[Any] = []
        if value_type is list:
            memo[value_id] = values
        values.extend(_copy_meta_val(v, fake_mode, memo) for v in value)
        result = immutable_list(values) if value_type is immutable_list else values
        memo[value_id] = result
        return result
    if value_type is not torch.Size and (
        value_type is tuple or _is_torch_return_type(value)
    ):
        if _is_torch_return_type(value):
            result = type(value)(*(_copy_meta_val(v, fake_mode, memo) for v in value))
        else:
            result = tuple(_copy_meta_val(v, fake_mode, memo) for v in value)
        memo[value_id] = result
        return result
    if _is_exact_type(value_type, (dict, immutable_dict)):
        values_dict: dict[Any, Any] = {}
        if value_type is dict:
            memo[value_id] = values_dict
        for key, item in value.items():
            values_dict[_copy_meta_val(key, fake_mode, memo)] = _copy_meta_val(
                item, fake_mode, memo
            )
        result = (
            immutable_dict(values_dict) if value_type is immutable_dict else values_dict
        )
        memo[value_id] = result
        return result
    return value


def _try_copy_meta_val(
    value: Any,
    fake_mode: "FakeTensorMode | None",
    memo: dict[int, Any] | None = None,
) -> Any:
    copy_memo = {} if memo is None else memo.copy()
    try:
        result = _copy_meta_val(value, fake_mode, copy_memo)
    except _fake_tensor_propagation_exceptions():
        return _MISSING
    if memo is not None:
        memo.update(copy_memo)
    return result


def _detect_fake_mode_from_values(values: list[Any]) -> "FakeTensorMode | None":
    try:
        fake_mode = detect_fake_mode(values)
    except AssertionError:
        # Inputs from multiple fake modes cannot be interpreted together.
        # Returning None makes propagation copy inputs but skip operator execution.
        return None
    if fake_mode is None and any(_contains_tensor(value) for value in values):
        _, FakeTensorMode, _ = _fake_tensor_meta_helpers()
        fake_mode = FakeTensorMode(allow_fallback_kernels=False)
    return fake_mode


def _resolve_rewrite_attr(
    destination_module: GraphModule,
    replacement_module: torch.nn.Module | None,
    target: Any,
) -> tuple[Any, bool]:
    destination_attr = _get_module_attr_static(destination_module, target)
    replacement_attr = _get_module_attr_static(replacement_module, target)
    if destination_attr is not _MISSING:
        return (
            destination_attr,
            replacement_attr is _MISSING or destination_attr is not replacement_attr,
        )
    return replacement_attr, False


def _meta_val_if_safe(node: Node) -> Any:
    meta_val = node.meta.get("val")
    if meta_val is None or not _is_safe_meta_value(meta_val):
        return _MISSING
    return meta_val


@cache
def _has_value_returning_torch_schema(target: Any) -> bool:
    try:
        _, schemas = get_signature_for_torch_op(target, return_schemas=True)
    except (NameError, RuntimeError, TypeError):
        return False
    return schemas is not None and any(schema.returns for schema in schemas)


def _is_trusted_torch_builtin(target: Any) -> bool:
    if type(target) is not _TORCH_BUILTIN_TYPE:
        return False
    target_name = getattr(target, "__name__", None)
    return (
        target.__module__ == "torch"
        and isinstance(target_name, str)
        and getattr(torch, target_name, None) is target
    )


def _is_safe_meta_propagation_target(
    target: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> bool:
    has_tensor_input = _contains_tensor((args, kwargs))
    if type(target) is torch._ops.OpOverload:
        return (
            target.namespace == "aten"
            and not target._defined_in_python
            and bool(target._schema.returns)
            and has_tensor_input
        )
    if type(target) is torch._ops.OpOverloadPacket:
        overloads = [getattr(target, name) for name in target.overloads()]
        return (
            bool(overloads)
            and all(
                overload.namespace == "aten" and not overload._defined_in_python
                for overload in overloads
            )
            and any(overload._schema.returns for overload in overloads)
            and has_tensor_input
        )
    return any(target is builtin for builtin in _SAFE_META_PROPAGATION_BUILTINS) or (
        has_tensor_input
        and _is_trusted_torch_builtin(target)
        and _has_value_returning_torch_schema(target)
    )


def _is_safe_meta_propagation_method(target: str) -> bool:
    if type(target) is not str:
        return False
    return _is_safe_meta_propagation_method_name(target)


@cache
def _is_safe_meta_propagation_method_name(target: str) -> bool:
    tensor_method = getattr(torch.Tensor, target, None)
    if tensor_method is None or not callable(tensor_method):
        return False
    try:
        packet = getattr(torch.ops.aten, target)
    except AttributeError:
        return False
    return isinstance(
        packet, torch._ops.OpOverloadPacket
    ) and _has_value_returning_torch_schema(packet)


def _propagate_replacement_meta(
    destination_module: GraphModule,
    replacement_graph: Graph,
    replacement_module: torch.nn.Module | None,
    val_map: dict[Node, Any],
    replacement_nodes: list[Node],
) -> None:
    # This intentionally does not use whole-graph FakeTensorProp. Only metadata
    # for newly copied replacement nodes is filled, and only through built-in ops
    # that are safe to interpret during rewriting.
    replacement_node_set = {
        node for node in replacement_nodes if isinstance(node, Node)
    }
    resolved_attrs: dict[Node, Any] = {}
    invalidated_nodes: set[Node] = set()
    for node in replacement_graph.nodes:
        if node.op in ("get_attr", "call_module"):
            attr_value, invalidates_preannotation = _resolve_rewrite_attr(
                destination_module, replacement_module, node.target
            )
            resolved_attrs[node] = attr_value
            if invalidates_preannotation:
                invalidated_nodes.add(node)
        if any(input_node in invalidated_nodes for input_node in node.all_input_nodes):
            invalidated_nodes.add(node)

    for node in invalidated_nodes:
        copied_node = val_map.get(node)
        if isinstance(copied_node, Node) and copied_node in replacement_node_set:
            copied_node.meta.pop("val", None)

    source_values: list[Any] = []
    for replacement_node, copied_node in val_map.items():
        if isinstance(copied_node, Node):
            if copied_node in replacement_node_set and replacement_node.op in (
                "get_attr",
                "call_function",
                "call_method",
            ):
                continue
            meta_val = _meta_val_if_safe(copied_node)
            if meta_val is not _MISSING:
                source_values.append(meta_val)
        elif _is_safe_meta_value(copied_node):
            source_values.append(copied_node)
    for node in replacement_graph.nodes:
        if node.op == "get_attr":
            attr_value = resolved_attrs[node]
            if attr_value is not _MISSING and _is_safe_meta_value(attr_value):
                source_values.append(attr_value)
    fake_mode = _detect_fake_mode_from_values(source_values)

    env: dict[Node, Any] = {}
    source_copy_memo: dict[int, Any] = {}

    def load_arg(arg_node: Node) -> Any:
        if arg_node not in env:
            raise KeyError(arg_node)
        return env[arg_node]

    def use_preannotated_meta(node: Node, meta_val: Any) -> None:
        if meta_val is not _MISSING:
            env[node] = meta_val

    for node in replacement_graph.nodes:
        if node.op == "output":
            continue

        copied_node = val_map.get(node)
        preannotated_meta_val = _MISSING
        if isinstance(copied_node, Node):
            infer_before_preannotation = (
                copied_node in replacement_node_set
                and node.op in ("get_attr", "call_function", "call_method")
            )
            if infer_before_preannotation:
                if node not in invalidated_nodes:
                    meta_val = _meta_val_if_safe(copied_node)
                    if meta_val is not _MISSING:
                        preannotated_meta_val = _try_copy_meta_val(
                            meta_val, fake_mode, source_copy_memo
                        )
            else:
                meta_val = _meta_val_if_safe(copied_node)
                if meta_val is not _MISSING:
                    copied_meta_val = _try_copy_meta_val(
                        meta_val, fake_mode, source_copy_memo
                    )
                    if copied_meta_val is _MISSING:
                        continue
                    env[node] = copied_meta_val
                    continue
                if copied_node.meta.get("val") is not None:
                    continue

        if node.op == "get_attr":
            attr_value = resolved_attrs[node]
            if attr_value is _MISSING or not _is_safe_meta_value(attr_value):
                use_preannotated_meta(node, preannotated_meta_val)
                continue
            copied_attr_value = _try_copy_meta_val(
                attr_value, fake_mode, source_copy_memo
            )
            if copied_attr_value is _MISSING:
                use_preannotated_meta(node, preannotated_meta_val)
                continue
            recorded_attr_value = _try_copy_meta_val(copied_attr_value, fake_mode)
            if recorded_attr_value is _MISSING:
                use_preannotated_meta(node, preannotated_meta_val)
                continue
            env[node] = copied_attr_value
            if isinstance(copied_node, Node) and copied_node in replacement_node_set:
                copied_node.meta["val"] = recorded_attr_value
            continue

        if node.op == "placeholder":
            if isinstance(copied_node, Node):
                meta_val = _meta_val_if_safe(copied_node)
                if meta_val is not _MISSING:
                    copied_meta_val = _try_copy_meta_val(
                        meta_val, fake_mode, source_copy_memo
                    )
                    if copied_meta_val is not _MISSING:
                        env[node] = copied_meta_val
            elif _is_safe_meta_value(copied_node):
                copied_meta_val = _try_copy_meta_val(
                    copied_node, fake_mode, source_copy_memo
                )
                if copied_meta_val is not _MISSING:
                    env[node] = copied_meta_val
            continue

        if node.op not in ("call_function", "call_method"):
            continue

        if not (isinstance(copied_node, Node) and copied_node in replacement_node_set):
            continue

        if fake_mode is None:
            # Never execute operators without a detected or freshly created mode.
            # None also covers metadata containing incompatible fake modes.
            use_preannotated_meta(node, preannotated_meta_val)
            continue

        try:
            args = map_arg(node.args, load_arg)
            kwargs = map_arg(node.kwargs, load_arg)
        except KeyError:
            # A predecessor could not be propagated, so its dependents cannot run.
            use_preannotated_meta(node, preannotated_meta_val)
            continue
        if not _is_safe_meta_value((args, kwargs)):
            use_preannotated_meta(node, preannotated_meta_val)
            continue

        try:
            if node.op == "call_function":
                if not _is_safe_meta_propagation_target(node.target, args, kwargs):
                    use_preannotated_meta(node, preannotated_meta_val)
                    continue
                with fake_mode:
                    result = node.target(*args, **kwargs)
            else:
                if not (
                    type(node.target) is str
                    and _is_safe_meta_propagation_method(node.target)
                    and len(args) > 0
                    and isinstance(args[0], torch.Tensor)
                ):
                    use_preannotated_meta(node, preannotated_meta_val)
                    continue
                with fake_mode:
                    result = getattr(args[0], node.target)(*args[1:], **kwargs)
        except _META_PROPAGATION_OPERATOR_EXCEPTIONS:
            # Invalid metadata inputs and unsupported fake implementations must
            # not turn this best-effort inference into a rewrite failure.
            log.debug(
                "Could not propagate replacement metadata for %s", node, exc_info=True
            )
            use_preannotated_meta(node, preannotated_meta_val)
            continue
        if not _is_safe_meta_value(result):
            use_preannotated_meta(node, preannotated_meta_val)
            continue
        copied_result = _try_copy_meta_val(result, fake_mode)
        if copied_result is _MISSING:
            use_preannotated_meta(node, preannotated_meta_val)
            continue
        recorded_result = _try_copy_meta_val(copied_result, fake_mode)
        if recorded_result is _MISSING:
            use_preannotated_meta(node, preannotated_meta_val)
            continue
        env[node] = copied_result
        copied_node.meta["val"] = recorded_result


@compatibility(is_backward_compatible=True)
def replace_pattern(
    gm: GraphModule,
    pattern: Callable[..., Any] | GraphModule,
    replacement: Callable[..., Any] | GraphModule,
) -> list[Match]:
    """
    Matches all possible non-overlapping sets of operators and their
    data dependencies (``pattern``) in the Graph of a GraphModule
    (``gm``), then replaces each of these matched subgraphs with another
    subgraph (``replacement``).

    Args:
        ``gm``: The GraphModule that wraps the Graph to operate on
        ``pattern``: The subgraph to match in ``gm`` for replacement
        ``replacement``: The subgraph to replace ``pattern`` with

    Returns:
        List[Match]: A list of ``Match`` objects representing the places
        in the original graph that ``pattern`` was matched to. The list
        is empty if there are no matches. ``Match`` is defined as:

        .. code-block:: python

            class Match(NamedTuple):
                # Node from which the match was found
                anchor: Node
                # Maps nodes in the pattern subgraph to nodes in the larger graph
                nodes_map: Dict[Node, Node]

    Examples:

    .. code-block:: python

        import torch
        from torch.fx import symbolic_trace, subgraph_rewriter


        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, w1, w2):
                m1 = torch.cat([w1, w2]).sum()
                m2 = torch.cat([w1, w2]).sum()
                return x + torch.max(m1) + torch.max(m2)


        def pattern(w1, w2):
            return torch.cat([w1, w2])


        def replacement(w1, w2):
            return torch.stack([w1, w2])


        traced_module = symbolic_trace(M())

        subgraph_rewriter.replace_pattern(traced_module, pattern, replacement)

    The above code will first match ``pattern`` in the ``forward``
    method of ``traced_module``. Pattern-matching is done based on
    use-def relationships, not node names. For example, if you had
    ``p = torch.cat([a, b])`` in ``pattern``, you could match
    ``m = torch.cat([a, b])`` in the original ``forward`` function,
    despite the variable names being different (``p`` vs ``m``).

    The ``return`` statement in ``pattern`` is matched based on its
    value only; it may or may not match to the ``return`` statement in
    the larger graph. In other words, the pattern doesn't have to extend
    to the end of the larger graph.

    When the pattern is matched, it will be removed from the larger
    function and replaced by ``replacement``. If there are multiple
    matches for ``pattern`` in the larger function, each non-overlapping
    match will be replaced. In the case of a match overlap, the first
    found match in the set of overlapping matches will be replaced.
    ("First" here being defined as the first in a topological ordering
    of the Nodes' use-def relationships. In most cases, the first Node
    is the parameter that appears directly after ``self``, while the
    last Node is whatever the function returns.)

    One important thing to note is that the parameters of the
    ``pattern`` Callable must be used in the Callable itself,
    and the parameters of the ``replacement`` Callable must match
    the pattern. The first rule is why, in the above code block, the
    ``forward`` function has parameters ``x, w1, w2``, but the
    ``pattern`` function only has parameters ``w1, w2``. ``pattern``
    doesn't use ``x``, so it shouldn't specify ``x`` as a parameter.
    As an example of the second rule, consider replacing

    .. code-block:: python

        def pattern(x, y):
            return torch.neg(x) + torch.relu(y)

    with

    .. code-block:: python

        def replacement(x, y):
            return torch.relu(x)

    In this case, ``replacement`` needs the same number of parameters
    as ``pattern`` (both ``x`` and ``y``), even though the parameter
    ``y`` isn't used in ``replacement``.

    After calling ``subgraph_rewriter.replace_pattern``, the generated
    Python code looks like this:

    .. code-block:: python

        def forward(self, x, w1, w2):
            stack_1 = torch.stack([w1, w2])
            sum_1 = stack_1.sum()
            stack_2 = torch.stack([w1, w2])
            sum_2 = stack_2.sum()
            max_1 = torch.max(sum_1)
            add_1 = x + max_1
            max_2 = torch.max(sum_2)
            add_2 = add_1 + max_2
            return add_2
    """
    match_and_replacements = _replace_pattern(gm, pattern, replacement)
    return [
        Match(anchor=m.anchor, nodes_map=m.nodes_map) for m in match_and_replacements
    ]


# Experimental API, not backward compatible
@compatibility(is_backward_compatible=False)
def replace_pattern_with_filters(
    gm: GraphModule,
    pattern: Callable[..., Any] | Graph | GraphModule,
    replacement: Callable[..., Any] | Graph | GraphModule | None = None,
    match_filters: list[Callable[["InternalMatch", Graph, Graph], bool]] | None = None,
    ignore_literals: bool = False,
    # Placed at the end to avoid breaking backward compatibility
    replacement_callback: Callable[["InternalMatch", Graph, Graph], Graph]
    | None = None,
    node_name_match: str = "",
) -> list[ReplacedPatterns]:
    """
    See replace_pattern for documentation. This function is an overload with an additional match_filter argument.

    Args:
        ``match_filters``: A list of functions that take in
            (match: InternalMatch, original_graph: Graph, pattern_graph: Graph) and return a boolean indicating
            whether the match satisfies the condition.
            See matcher_utils.py for definition of InternalMatch.
        ``replacement_callback``: A function that takes in a match and returns a
            Graph to be used as the replacement. This allows you to construct a
            replacement graph based on the match.
        ``node_name_match``: Node name to match. If not empty, it will try to match the node name.
    """

    return _replace_pattern(
        gm,
        pattern,
        replacement,
        match_filters,
        ignore_literals,
        replacement_callback,
        node_name_match,
    )


def _replace_pattern(
    gm: GraphModule,
    pattern: Callable[..., Any] | Graph | GraphModule,
    replacement: Callable[..., Any] | Graph | GraphModule | None = None,
    match_filters: list[Callable[["InternalMatch", Graph, Graph], bool]] | None = None,
    ignore_literals: bool = False,
    # Placed at the end to avoid breaking backward compatibility
    replacement_callback: Callable[["InternalMatch", Graph, Graph], Graph]
    | None = None,
    node_name_match: str = "",
) -> list[ReplacedPatterns]:
    from torch.fx.passes.utils.matcher_utils import InternalMatch, SubgraphMatcher

    if match_filters is None:
        match_filters = []

    # Get the graphs for `gm`, `pattern`, `replacement`
    original_graph: Graph = gm.graph

    if isinstance(pattern, GraphModule):
        pattern_graph = pattern.graph
    elif isinstance(pattern, Graph):
        pattern_graph = pattern
    else:
        pattern_graph = symbolic_trace(pattern).graph  # type: ignore[arg-type]

    matcher = SubgraphMatcher(
        pattern_graph,
        match_output=False,
        match_placeholder=False,
        remove_overlapping_matches=True,
        ignore_literals=ignore_literals,
    )
    _matches: list[InternalMatch] = matcher.match(
        original_graph, node_name_match=node_name_match
    )

    # Filter out matches that don't match the filter
    _matches = [
        m
        for m in _matches
        if all(
            match_filter(m, original_graph, pattern_graph)
            for match_filter in match_filters
        )
    ]

    if isinstance(replacement, GraphModule):
        common_replacement_graph = replacement.graph
        common_replacement_module: torch.nn.Module | None = replacement
    elif isinstance(replacement, Graph):
        common_replacement_graph = replacement
        common_replacement_module = None
    elif callable(replacement):
        common_replacement_graph = symbolic_trace(replacement).graph
        common_replacement_module = (
            replacement if isinstance(replacement, torch.nn.Module) else None
        )
    else:
        if replacement_callback is None:
            raise AssertionError(
                "Must provide either a replacement GraphModule or a replacement callback"
            )
        common_replacement_graph = None  # type: ignore[assignment]
        common_replacement_module = None

    # As we progressively replace nodes, we'll need to keep track of how the match results should change
    match_changed_node: dict[Node, Node] = {}

    match_and_replacements = []
    for match in _matches:
        if replacement_callback is not None:
            replacement_graph = replacement_callback(
                match, original_graph, pattern_graph
            )
            replacement_module = None
        else:
            if common_replacement_graph is None:
                raise AssertionError(
                    "Must provide either a replacement GraphModule or a replacement callback"
                )
            replacement_graph = common_replacement_graph
            replacement_module = common_replacement_module
        replacement_placeholders = [
            n for n in replacement_graph.nodes if n.op == "placeholder"
        ]

        # Build connecting between replacement graph's input and original graph input producer node

        # Initialize `val_map` with mappings from placeholder nodes in
        # `replacement` to their corresponding node in `original_graph`
        if len(match.placeholder_nodes) != len(replacement_placeholders):
            raise AssertionError(
                f"Placeholder count mismatch: {len(match.placeholder_nodes)} vs "
                f"{len(replacement_placeholders)}"
            )
        val_map: dict[Node, Node] = {}
        for rn, gn in zip(replacement_placeholders, match.placeholder_nodes):
            if isinstance(gn, Node):
                val_map[rn] = match_changed_node.get(gn, gn)
                if gn != val_map[rn]:
                    # Update match.placeholder_nodes and match.nodes_map with the node that replaced gn
                    gn_ind = match.placeholder_nodes.index(gn)
                    match.placeholder_nodes[gn_ind] = match_changed_node[gn]
                    map_key = list(match.nodes_map.keys())[
                        list(match.nodes_map.values()).index(gn)
                    ]
                    match.nodes_map[map_key] = match_changed_node[gn]
            else:
                val_map[rn] = gn

        # Copy the replacement graph over
        user_nodes: set[Node] = set()
        for n in match.returning_nodes:
            user_nodes.update(n.users)

        first_user_node = None
        if len(user_nodes) == 0:
            first_user_node = None
        elif len(user_nodes) == 1:
            first_user_node = next(iter(user_nodes))
        else:
            # If there are multiple user nodes, we need to find the first user node
            # in the current execution order of the `original_graph`
            for n in original_graph.nodes:
                if n in user_nodes:
                    first_user_node = n
                    break

        first_next_node = None
        if first_user_node is None:
            # no users, so we insert the replacement graph before the first next
            # node of returning nodes
            next_node = None
            for n in reversed(original_graph.nodes):
                if n in match.returning_nodes:
                    first_next_node = next_node
                    break
                else:
                    next_node = n
        insert_point = (
            first_user_node if first_user_node is not None else first_next_node
        )
        if insert_point is None:
            raise AssertionError("The insert point can't be None")
        with original_graph.inserting_before(insert_point):
            copied_returning_nodes = original_graph.graph_copy(
                replacement_graph, val_map
            )

        if isinstance(copied_returning_nodes, Node):
            copied_returning_nodes = (copied_returning_nodes,)

        # Get a list of nodes that have been replaced into the graph
        replacement_nodes: list[Node] = [
            v for v in val_map.values() if v not in match.placeholder_nodes
        ]
        _propagate_replacement_meta(
            gm,
            replacement_graph,
            replacement_module,
            val_map,
            replacement_nodes,
        )

        # Hook the output Node of the replacement subgraph in to the
        # original Graph at the correct location
        if len(match.returning_nodes) != len(copied_returning_nodes):  # type: ignore[arg-type]
            raise AssertionError(
                f"Returning nodes count mismatch: {len(match.returning_nodes)} vs "
                f"{len(copied_returning_nodes)}"  # pyrefly: ignore [bad-argument-type]
            )
        for gn, copied_node in zip(match.returning_nodes, copied_returning_nodes):  # type: ignore[arg-type]
            # pyrefly: ignore [bad-argument-type]
            gn.replace_all_uses_with(copied_node)
            # pyrefly: ignore [unsupported-operation]
            match_changed_node[gn] = copied_node
        # Remove the original nodes
        for node in reversed(pattern_graph.nodes):
            if node.op != "placeholder" and node.op != "output":
                gn = match.nodes_map[node]
                gm.graph.erase_node(gn)

        match_and_replacements.append(
            ReplacedPatterns(
                anchor=match.anchors[0],
                nodes_map=match.nodes_map,
                replacements=replacement_nodes,
            )
        )

    # Update the passed-in GraphModule to reflect the new state of
    # `original_graph`
    gm.recompile()

    # If `replacement` was an nn.Module, we'll need to make sure that
    # all the submodules have been copied over correctly
    if isinstance(replacement, torch.nn.Module):
        _replace_attributes(gm, replacement)

    return match_and_replacements
