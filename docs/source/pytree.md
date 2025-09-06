(pytree)=

# PyTrees

```{warning}
The main PyTree functionality is available through `torch.utils._pytree`. Note
that this is currently a private API and may change in future versions.
```

## Overview

A *pytree* is a nested data structure composed of:

* **Leaves**: non-container objects that cannot be further decomposed (e.g. tensors, ints, floats, etc.)
* **Containers**: collections like list, tuple, dict, namedtuple, etc. that contain other pytrees or leaves

Specifically, the following types are defined as **leaf type**:

* {class}`torch.Tensor` and tensor metadata ({class}`torch.dtype`, {class}`torch.layout`, {class}`torch.device`, {class}`torch.memory_format`)
* Symbolic values: `torch.SymInt`, `torch.SymFloat`, `torch.SymBool`
* Scalars: `int`, `float`, `bool`, `str`
* Python classes
* Anything registered through {func}`torch.utils._pytree.register_constant`

The following types are defined as **container type**:

* Python `list`, `tuple`, `namedtuple`
* Python `dict` with scalar keys
* Python `dataclass` (must be registered through {func}`torch.utils._pytree.register_dataclass`)
* Python classes (must be registered through {func}`torch.utils._pytree.register_dataclass` or {func}`torch.utils._pytree.register_pytree_node`)

## Viewing the PyTree Structure

We can use {func}`torch.utils._pytree.tree_structure` to view the
structure of a pytree. The structure is represented as a
{func}`torch.utils._pytree.TreeSpec` object, which contains the type of the
container, a list of its children `TreeSpec`'s, and any context needed to
represent the pytree.

```{code-cell}
import torch
import torch.utils._pytree as pytree

simple_list = [torch.tensor([1, 2]), torch.tensor([3, 4])]
print("simple_list treespec:", pytree.tree_structure(simple_list))

list_with_dict = [
    {'a': torch.tensor([1]), 'b': torch.tensor([2])},
    torch.tensor([3])
]
print("list_with_dict treespec:", pytree.tree_structure(simple_dict))
```

## Manipulating PyTrees

We can use {func}`torch.utils._pytree.tree_flatten` to flatten a pytree into a
list of leaves. This function also returns a `TreeSpec` representing the
structure of the pytree. This can be used along with
{func}`torch.utils._pytree.tree_unflatten` to then reconstruct the original pytree.

```{code-cell}
tree = {
    'a': torch.tensor([1]),
    'b': [2, 3.0],
}

flattened_tree, treespec = pytree.tree_flatten(tree)
print("flattened_tree:", flattened_tree)
print("treespec:", treespec)
```

```{code-cell}
manipulated_tree = [x + 1 for x in flattened_tree]
unflattened_tree = pytree.tree_unflatten(manipulated_tree, treespec)
print("unflattened_tree:", unflattened_tree)
```

We can also simply use {func}`torch.utils._pytree.tree_map` to apply a function
to all leaves in a pytree, or {func}`torch.utils._pytree.tree_map_only` to
specific types of leaves.

```{code-cell}
print("map over all:", pytree.tree_map(lambda x: x + 1, tree))

print("map over tensors:", pytree.tree_map_only(torch.Tensor, lambda x: x + 1, tree))
```

leaves, treespec = tree_flatten(tree)
print("Leaves:", leaves)
print("TreeSpec:", treespec)
import torch
import torch.utils._pytree as pytree

## Custom PyTree Registration

You can register custom types to be treated as PyTree containers. To do so, you
must specify a flatten function that returns a flattened representation of the
pytree, and an unflattening function that takes a flattened representation
returns the original pytree.

```{code-cell}
class Data1:
    def __init__(self, a: torch.Tensor, b: tuple[str]):
        self.a = a
        self.b = b

data = Data1(torch.tensor(3), ("moo",))
print("TreeSpec without registration:", pytree.tree_structure(data))

pytree.register_pytree_node(
    Data1,
    flatten_fn=lambda x: (x.a, x.b),
    unflatten_fn=lambda a, b: Data1(a, b),
)
print("TreeSpec after registration:", pytree.tree_structure(data))
```


If the class is a dataclass, or has the semantics of a dataclass, a simpler
approach is to use {func}`torch.utils._pytree.register_dataclass`.

```{code-cell}
class Data2:
    def __init__(self, a: torch.Tensor, b: tuple[str]):
        self.a = a
        self.b = b

data = Data2(torch.tensor(3), ("moo",))
print("TreeSpec without registration:", pytree.tree_structure(data))

pytree.register_dataclass(Data2, field_names=["a", "b"])
print("TreeSpec after registration:", pytree.tree_structure(data))
```

## API Reference

```{eval-rst}
.. autofunction:: torch.utils._pytree.tree_flatten
.. autofunction:: torch.utils._pytree.tree_flatten_with_path
.. autofunction:: torch.utils._pytree.tree_unflatten
.. autofunction:: torch.utils._pytree.tree_map
.. autofunction:: torch.utils._pytree.tree_map_
.. autofunction:: torch.utils._pytree.tree_map_only
.. autofunction:: torch.utils._pytree.tree_map_with_path
.. autofunction:: torch.utils._pytree.register_pytree_node
.. autofunction:: torch.utils._pytree.register_dataclass
.. autofunction:: torch.utils._pytree.register_constant
.. autofunction:: torch.utils._pytree.tree_structure
.. autoclass:: torch.utils._pytree.TreeSpec
```
