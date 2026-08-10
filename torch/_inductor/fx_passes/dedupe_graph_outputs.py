# mypy: allow-untyped-defs
"""Share equivalent output computation without aliasing returned tensors."""

import struct
from collections import defaultdict

import torch
from torch import fx
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.utils import counters
from torch._subclasses.fake_tensor import FakeTensor
from torch.utils._ordered_set import OrderedSet

from .. import config


_MIN_OUTPUTS = 8
_MAX_OUTPUTS = 32
_SAFE_NAMESPACES = frozenset(("aten", "prims"))
_REJECTED_TAGS = frozenset(
    (
        torch.Tag.cudagraph_unsafe,
        torch.Tag.data_dependent_output,
        torch.Tag.dynamic_output_shape,
        torch.Tag.inplace,
        torch.Tag.maybe_aliasing_or_mutating,
        torch.Tag.nondeterministic_bitwise,
        torch.Tag.nondeterministic_seeded,
        torch.Tag.out,
        torch.Tag.out_variant,
    )
)
# These allocate unspecified values or carry tensor metadata that clone does not
# preserve. Descendants must not be CSE'd through them either.
_REJECTED_PACKETS = frozenset(
    (
        torch.ops.aten._conj,
        torch.ops.aten._empty_affine_quantized,
        torch.ops.aten._empty_per_channel_affine_quantized,
        torch.ops.aten._efficientzerotensor,
        torch.ops.aten._neg_view,
        torch.ops.aten.empty,
        torch.ops.aten.empty_like,
        torch.ops.aten.empty_permuted,
        torch.ops.aten.empty_quantized,
        torch.ops.aten.empty_strided,
        torch.ops.aten.new_empty,
        torch.ops.aten.new_empty_strided,
        torch.ops.prims.empty,
        torch.ops.prims.empty_permuted,
        torch.ops.prims.empty_strided,
    )
)
_StorageKey = tuple[int, torch.device]


def _output_node(graph: fx.Graph) -> fx.Node | None:
    return next((node for node in graph.nodes if node.op == "output"), None)


def is_output_computation_sharing_supported(gm: fx.GraphModule) -> bool:
    """Restrict default-on use to the audited NVIDIA SM100 Triton cohort."""
    if torch.version.hip is not None or config.cuda_backend != "triton":
        return False

    output = _output_node(gm.graph)
    if output is None:
        return False

    devices = OrderedSet(
        [
            leaf.device
            for arg in torch.utils._pytree.arg_tree_leaves(
                *output.args, **output.kwargs
            )
            if isinstance(arg, fx.Node)
            for leaf in torch.utils._pytree.tree_leaves(arg.meta.get("val"))
            if isinstance(leaf, torch.Tensor)
        ]
    )
    if len(devices) != 1:
        return False
    device = next(iter(devices))
    if device.type != "cuda":
        return False

    properties = get_interface_for_device("cuda").Worker.get_device_properties(device)
    return properties.major == 10


def _constant_key(value):
    """Return a type-sensitive key, or None for unsupported constants."""
    value_type = type(value)
    if value is None:
        return (value_type,)
    if value_type in (bool, int, str, bytes):
        return (value_type, value)
    if value_type is float:
        return (value_type, struct.pack("!d", value))
    if value_type is complex:
        return (
            value_type,
            struct.pack("!d", value.real),
            struct.pack("!d", value.imag),
        )
    if value_type in (torch.dtype, torch.device, torch.layout, torch.memory_format):
        return (value_type, value)
    return None


def _is_functional_op(node: fx.Node) -> bool:
    return (
        node.op == "call_function"
        and isinstance(node.target, torch._ops.OpOverload)
        and node.target.namespace in _SAFE_NAMESPACES
        and not node.is_impure()
    )


def _is_shareable_node(node: fx.Node) -> bool:
    if not _is_functional_op(node):
        return False
    target = node.target
    if not isinstance(target, torch._ops.OpOverload):
        return False
    if target.overloadpacket in _REJECTED_PACKETS or not _REJECTED_TAGS.isdisjoint(
        target.tags
    ):
        return False
    return not any(
        arg.alias_info is not None and arg.alias_info.is_write
        for arg in target._schema.arguments
    )


def _is_pure_functional_graph(graph: fx.Graph) -> bool:
    return all(
        node.op in ("placeholder", "get_attr", "output") or _is_functional_op(node)
        for node in graph.nodes
    )


def _structural_classes(graph: fx.Graph) -> dict[fx.Node, int]:
    """Intern collision-free structural keys in topological order."""
    classes: dict[fx.Node, int] = {}
    interned: dict[tuple[object, ...], int] = {}

    for node in graph.nodes:
        key = None
        if _is_shareable_node(node):
            leaves, spec = torch.utils._pytree.tree_flatten(
                (node.args, tuple(sorted(node.kwargs.items())))
            )
            leaf_keys: list[object] = []
            for leaf in leaves:
                if isinstance(leaf, fx.Node):
                    leaf_keys.append(("node", classes[leaf]))
                elif (constant := _constant_key(leaf)) is not None:
                    leaf_keys.append(("constant", constant))
                else:
                    break
            else:
                key = (node.target, spec, *leaf_keys)

        class_id = interned.get(key) if key is not None else None
        if class_id is None:
            class_id = len(classes)
            if key is not None:
                interned[key] = class_id
        classes[node] = class_id

    return classes


def _storage_key(value) -> _StorageKey | None:
    if (
        type(value) is not FakeTensor
        or value.layout is not torch.strided
        or not torch._C._has_storage(value)
    ):
        return None
    try:
        return (value.untyped_storage()._cdata, value.device)
    except RuntimeError:
        return None


def _storage_keys(value) -> OrderedSet[_StorageKey]:
    return OrderedSet(
        [
            key
            for leaf in torch.utils._pytree.tree_leaves(value)
            if (key := _storage_key(leaf)) is not None
        ]
    )


def _has_distinct_dense_storage(
    node: fx.Node,
    input_storage: OrderedSet[_StorageKey],
    output_storage_users: dict[_StorageKey, OrderedSet[fx.Node]],
) -> bool:
    value = node.meta.get("val")
    if type(value) is not FakeTensor:
        return False
    key = _storage_key(value)
    if (
        key is None
        or not all(isinstance(size, int) for size in value.size())
        or not all(isinstance(stride, int) for stride in value.stride())
        or not isinstance(value.storage_offset(), int)
        or value.storage_offset() != 0
        or not torch._prims_common.is_non_overlapping_and_dense_or_false(value)
    ):
        return False
    try:
        storage_nbytes = value.untyped_storage().nbytes()
    except RuntimeError:
        return False
    return (
        key not in input_storage
        and len(output_storage_users[key]) == 1
        and storage_nbytes == value.numel() * value.element_size()
    )


def dedupe_graph_outputs_pass(graph: fx.Graph) -> None:
    """Share 8-32 equivalent output branches and clone their returned values."""
    output = _output_node(graph)
    if output is None:
        return

    output_nodes = list(
        dict.fromkeys(
            node
            for node in torch.utils._pytree.arg_tree_leaves(
                *output.args, **output.kwargs
            )
            if isinstance(node, fx.Node)
        )
    )
    if len(output_nodes) < _MIN_OUTPUTS or not _is_pure_functional_graph(graph):
        return

    groups: dict[int, list[fx.Node]] = defaultdict(list)
    classes = _structural_classes(graph)
    for node in output_nodes:
        groups[classes[node]].append(node)
    candidates = [
        group for group in groups.values() if _MIN_OUTPUTS <= len(group) <= _MAX_OUTPUTS
    ]
    if not candidates:
        return

    input_storage: OrderedSet[_StorageKey] = OrderedSet()
    for node in graph.nodes:
        if node.op in ("placeholder", "get_attr"):
            input_storage.update(_storage_keys(node.meta.get("val")))

    output_storage_users: dict[_StorageKey, OrderedSet[fx.Node]] = defaultdict(
        OrderedSet
    )
    for node in output_nodes:
        if (key := _storage_key(node.meta.get("val"))) is not None:
            output_storage_users[key].add(node)

    replacements = 0
    for group in candidates:
        eligible = [
            node
            for node in group
            if _has_distinct_dense_storage(node, input_storage, output_storage_users)
        ]
        if len(eligible) < _MIN_OUTPUTS:
            continue

        def is_output_only(node: fx.Node) -> bool:
            return len(node.users) == 1 and output in node.users

        canonical = next(
            (node for node in eligible if not is_output_only(node)), eligible[0]
        )
        duplicates = [
            node for node in eligible if node is not canonical and is_output_only(node)
        ]
        if len(duplicates) + 1 < _MIN_OUTPUTS:
            continue

        for node in duplicates:
            with graph.inserting_before(output):
                clone = graph.call_function(torch.ops.aten.clone.default, (canonical,))
            node.replace_all_uses_with(clone)
            replacements += 1

    if replacements:
        graph.eliminate_dead_code()
        graph.lint()
        counters["inductor"]["dedupe_graph_outputs"] += replacements
