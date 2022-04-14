from executorch.pytree.pybindings import (
    flatten as flatten,
    PyTree as PyTree,
)


def pytree_map(leaf_fn, tree):
    (leaves, pytree) = flatten(tree)
    return pytree.unflatten([leaf_fn(x) for x in leaves])
