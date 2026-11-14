import torch
from torch.fx import GraphModule, Node
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
from torch.utils._ordered_set import OrderedSet

from ..compiler_bisector import CompilerBisector
from ..fx_utils import get_node_storage


aten = torch.ops.aten


def canonicalize_as_strided(
    gm: GraphModule,
    *,
    storage_bases: OrderedSet[Node] | None = None,
    is_subgraph: bool = False,
) -> int:
    """Redirect as_strided/copy to supported fresh allocations at storage offset zero."""
    from ..lowering import user_lowerings

    if is_subgraph or CompilerBisector.bisection_enabled:
        return 0
    changed = 0
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target not in (
            aten.as_strided.default,
            aten.as_strided_copy.default,
        ):
            continue
        if node.target in user_lowerings:
            continue
        args = dict(zip(("self", "size", "stride", "storage_offset"), node.args))
        args.update(node.kwargs)
        source = args["self"]
        if not isinstance(source, Node):
            continue
        base = source
        supported = True
        while True:
            if base.target in user_lowerings:
                supported = False
                break
            value = base.meta.get("val")
            if not isinstance(value, torch.Tensor) or value.is_conj() or value.is_neg():
                supported = False
                break
            if base.op != "call_function" or not isinstance(
                base.target, torch._ops.OpOverload
            ):
                break
            # FakeTensor storage identifies aliases; retain mutation dependencies.
            if base.target._schema.is_mutable:
                break
            parent = base.args[0] if base.args else base.kwargs.get("self")
            if not isinstance(parent, Node):
                break
            parent_value = parent.meta.get("val")
            if not isinstance(parent_value, torch.Tensor):
                supported = False
                break
            storage = get_node_storage(base)
            parent_storage = get_node_storage(parent)
            if storage is None or parent_storage is None:
                supported = False
                break
            if storage != parent_storage:
                break
            if value.device != parent_value.device or value.dtype != parent_value.dtype:
                # A dtype-changing alias is not a supported storage base.
                supported = False
                break
            base = parent
        if not supported:
            continue
        # Do not treat mutations or unresolved aliases as fresh storage bases.
        value = base.meta["val"]
        if (
            base.op != "call_function"
            or not isinstance(base.target, torch._ops.OpOverload)
            or base.target._schema.is_mutable
            or any(ret.alias_info is not None for ret in base.target._schema.returns)
            or not isinstance(value.storage_offset(), int)
            or value.storage_offset() != 0
            or free_unbacked_symbols(value.stride())
        ):
            continue
        if storage_bases is not None:
            storage_bases.add(base)
        if base is source:
            continue

        value = source.meta["val"]
        offset = args.get("storage_offset")
        if offset is None:
            # The default offset belongs to the original view, not its base.
            offset = value.storage_offset()
            if isinstance(offset, torch.SymInt):
                with gm.graph.inserting_before(node):
                    offset = gm.graph.create_storage_offset_node(source)
        node.args = (base, args["size"], args["stride"], offset)
        node.kwargs = {}
        # Redirecting the input preserves the result's metadata.
        changed += 1

    if changed:
        gm.graph.lint()
        gm.recompile()
    return changed
