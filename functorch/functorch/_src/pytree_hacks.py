import torch.utils._pytree as _pytree
from torch.utils._pytree import tree_flatten, tree_unflatten

# TODO: The following function should only be used with vmap.
# torch.return_types should be registered as PyTree nodes.
# I can't figure out how to do that, so we are turning all of them
# into normal Tuples for now (this is what vmap used to do anyways).
# We probably want some special behavior for named tuples?
def tree_flatten_hack(pytree):
    if _pytree._is_leaf(pytree) and not isinstance(pytree, tuple):
        return [pytree], _pytree.LeafSpec()

    if isinstance(pytree, tuple):
        typ = tuple
    else:
        typ = type(pytree)

    flatten_fn = _pytree.SUPPORTED_NODES[typ].flatten_fn
    child_pytrees, context = flatten_fn(pytree)

    # Recursively flatten the children
    result : List[Any] = []
    children_specs : List['TreeSpec'] = []
    for child in child_pytrees:
        flat, child_spec = tree_flatten_hack(child)
        result += flat
        children_specs.append(child_spec)

    return result, _pytree.TreeSpec(typ, context, children_specs)

# TODO: replace this with tree_map from core
def tree_map(fn, pytree):
    flat_args, spec = tree_flatten(pytree)
    return tree_unflatten([fn(arg) for arg in flat_args], spec)

def tree_map_(fn_, pytree):
    flat_args, _ = tree_flatten(pytree)
    [fn_(arg) for arg in flat_args]
    return pytree

class PlaceHolder():
    def __repr__(self):
        return '*'

def treespec_pprint(spec):
    leafs = [PlaceHolder() for _ in range(spec.num_leaves)]
    result = tree_unflatten(leafs, spec)
    return repr(result)

