import torch
from torch.fx import GraphModule, Node

from ..fx_utils import get_node_storage


aten = torch.ops.aten


def canonicalize_as_strided(gm: GraphModule) -> int:
    """Redirect as_strided and as_strided_copy through same-dtype aliases."""
    changed = 0
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target not in (
            aten.as_strided.default,
            aten.as_strided_copy.default,
        ):
            continue
        args = dict(zip(("self", "size", "stride", "storage_offset"), node.args))
        args.update(node.kwargs)
        source = args["self"]
        if not isinstance(source, Node):
            continue
        base = source
        while base.op == "call_function" and isinstance(
            base.target, torch._ops.OpOverload
        ):
            # FakeTensor storage identifies aliases; retain mutation dependencies.
            if base.target._schema.is_mutable:
                break
            parent = base.args[0] if base.args else base.kwargs.get("self")
            if not isinstance(parent, Node):
                break
            value = base.meta.get("val")
            parent_value = parent.meta.get("val")
            if not isinstance(value, torch.Tensor) or not isinstance(
                parent_value, torch.Tensor
            ):
                break
            storage = get_node_storage(base)
            if (
                storage is None
                or storage != get_node_storage(parent)
                or value.dtype != parent_value.dtype
                or value.device != parent_value.device
                or value.is_conj()
                or parent_value.is_conj()
                or value.is_neg()
                or parent_value.is_neg()
            ):
                break
            base = parent
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
