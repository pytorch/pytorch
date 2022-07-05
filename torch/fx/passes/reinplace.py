import torch
import torch.fx as fx
from torch.fx._compatibility import compatibility

from collections import defaultdict
from typing import Iterable
import itertools

__all__ = ['reinplace']

# Another option would be to expose `Tensor::is_alias_of` to python.
# This wouldn't be valid for non-standard backends like opaque tensor impls,
# but works just fine for meta tensors.
# Using `_base`  seems to work fine though.
def is_view_of(a, b):
    # The two tensors are the same python object
    if a is b:
        return True
    # One tensor is a "base", and the other is a view of it
    if a._base is b or b._base is a:
        return True
    # Both tensors are views, of the same original base tensor
    if a._base is not None and b._base is not None and a._base is b._base:
        return True
    return False

class ComputeInputAliasNames:
    """
    This class takes a `GraphModule`.
    Then, its `__call__` method executes the `GraphModule`
    node-by-node with the given arguments. As each operation
    executes, we use the runtime aliasing information
    to determine which node inputs in the graph
    correspond to aliases of the input arguments.
    It returns a set of all graph inputs and intermediate
    variable names that alias with the inputs.
    """
    def __init__(self, mod):
        self.mod = mod
        self.graph = mod.graph
        self.modules = dict(self.mod.named_modules())

    def __call__(self, *args):
        args_iter = iter(args)
        env : Dict[str, Node] = {}
        tensor_inputs = [a for a in args if isinstance(a, torch.Tensor)]
        input_aliases = set()

        def load_arg(a):
            return torch.fx.graph.map_arg(a, lambda n: env[n.name])

        def fetch_attr(target : str):
            target_atoms = target.split('.')
            attr_itr = self.mod
            for i, atom in enumerate(target_atoms):
                if not hasattr(attr_itr, atom):
                    raise RuntimeError(f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}")
                attr_itr = getattr(attr_itr, atom)
            return attr_itr

        for node in self.graph.nodes:
            if node.op == 'placeholder':
                result = next(args_iter)
            elif node.op == 'get_attr':
                result = fetch_attr(node.target)
            elif node.op == 'call_function':
                result = node.target(*load_arg(node.args), **load_arg(node.kwargs))
            elif node.op == 'call_method':
                self_obj, *args = load_arg(node.args)
                kwargs = load_arg(node.kwargs)
                result = getattr(self_obj, node.target)(*args, **kwargs)
            elif node.op == 'call_module':
                result = self.modules[node.target](*load_arg(node.args), **load_arg(node.kwargs))

            # This is the only code specific to shape propagation.
            # you can delete this `if` branch and this becomes
            # a generic GraphModule interpreter.
            for a in itertools.chain(node.args, node.kwargs.values()):
                arg = load_arg(a)
                if isinstance(arg, torch.Tensor) and any(is_view_of(arg, tensor_input) for tensor_input in tensor_inputs):
                    input_aliases.add(a.name)
            env[node.name] = result

        return input_aliases


def schemas_match(functional_schema, inplace_schema):
    names_match = inplace_schema.name.endswith("_") and inplace_schema.name[:-1] == functional_schema.name
    arg_types_match = all(
        a1.type == a2.type for a1, a2 in zip(functional_schema.arguments, inplace_schema.arguments))
    # for the inplace op, its first argument should be mutable
    assert inplace_schema.arguments[0].alias_info is not None and inplace_schema.arguments[0].alias_info.is_write
    # and its remaining arguments shouldn't be.
    assert all(a.alias_info is None for a in inplace_schema.arguments[1:])
    return names_match and arg_types_match

@compatibility(is_backward_compatible=True)
def reinplace(gm: fx.GraphModule, *sample_args):

    # Maintain a mapping: for every variable in the graph,
    # what are all of nodes in the graph that use it as an input?
    variable_to_nodes = defaultdict(list)
    for idx, node in enumerate(gm.graph.nodes):
        for arg in itertools.chain(node.args, node.kwargs.values()):
            if not isinstance(arg, Iterable):
                arg = [arg]
            for a in arg:
                if isinstance(a, fx.node.Node):
                    variable_to_nodes[a.name].append([idx, node])

    # Keep track of the inputs to the graph and their aliases.
    # If the first argument to the an out-of-place op is a graph input,
    # we can't convert it to an inplace op.
    sample_args = [a.to(device='meta') if isinstance(a, torch.Tensor) else a for a in sample_args]
    graph_input_alias_names = ComputeInputAliasNames(gm)(*sample_args)

    for idx, node in enumerate(gm.graph.nodes):
        if node.op == 'call_function':
            # __module__ seems broken; it returns torch._ops.aten which doesn't exist
            op_namespace = node.target.__module__.split(".")[-1]
            op_base_name = node.target.overloadpacket.__name__
            # Step 1: Check to see if this operator has an inplace variant.
            maybe_namespace_module = getattr(torch.ops, op_namespace)
            maybe_inplace_op = None if maybe_namespace_module is None else getattr(maybe_namespace_module, f'{op_base_name}_', None)
            if maybe_inplace_op is not None:
                inplace_overloads = [
                    getattr(maybe_inplace_op, overload_name) for overload_name in maybe_inplace_op.overloads()
                ]
                inplace_overloads_with_matching_schemas = [
                    f
                    for f in inplace_overloads
                    if schemas_match(node.target._schema, f._schema)
                ]
                # This is for sanity: if foo() and foo_() are both operators,
                # we expect them to have compatible schemas.
                # (This is asserted by codegen for ATen, but might not be true
                # for other arbitrary operators).
                assert len(inplace_overloads_with_matching_schemas) == 1
                inplace_op = inplace_overloads_with_matching_schemas[0]

                # This is a proxy check for ensuring that the first argument is "tensor-like"
                # (should be the case for all ops with inplace variants).
                assert len(node.target._schema.arguments) > 0
                assert 'Tensor' in str(node.target._schema.arguments[0].type)

                # Step 2: Check to see if the input to the op is re-used later in the graph.
                self_arg = node.args[0]
                self_arg_name = self_arg.name
                # We also can't re-inplace an op if input was an (alias to an) input to the graph.
                if self_arg_name in graph_input_alias_names:
                    continue
                nodes_used = variable_to_nodes[self_arg_name]
                # for every node that uses the current variable, we've tracked what their index is
                # in the graph. If there aren't any nodes with a higher index that use the current
                # argument, that means it isn't re-used anywhere.
                if not any(node_idx > idx for node_idx, _ in nodes_used):
                    # Step 3: replace the current out-of-place op with its inplace variant.
                    node.target = inplace_op

                    # Step 4:
                    # Now that we've replaced b = a.foo() with a.foo_(),
                    # We need to replace any later usages of "b" with "a"
                    for _, node_to_update in variable_to_nodes[node.name]:
                        new_args = []
                        for a in node_to_update.args:
                            if isinstance(a, fx.node.Node) and a.name == node.name:
                                new_args.append(self_arg)
                            else:
                                new_args.append(a)
                        new_kwargs = {}
                        for k, v in node_to_update.kwargs:
                            if isinstance(v, fx.node.Node) and v.name == node.name:
                                new_kwargs[k] = self_arg
                            else:
                                new_kwargs[k] = v

                        node_to_update.args = new_args
                        node_to_update.kwargs = new_kwargs

    gm.recompile()
    return gm
