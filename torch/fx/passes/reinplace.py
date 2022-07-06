import torch
import torch.fx as fx
from torch.fx._compatibility import compatibility

from collections import defaultdict
from typing import Iterable
import itertools

__all__ = ['reinplace']

def concatMap(func, xs):
    for x in xs:
        for r in func(x):
            yield r


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

    It returns a list[set[str]] "input_aliases",
    where input_aliases[i] is a list of names of all arguments
    in the GraphModule that alias with the i'th input to the program.

    It also returns a dict[str, set[str]], "alias_map"
    containing alias information of all variables in the graph.
    For a given variable name "x", alias_map[x] gives you a set
    of all variables in the graph that alias with x (including x itself).
    """
    def __init__(self, mod):
        self.mod = mod
        self.graph = mod.graph
        self.modules = dict(self.mod.named_modules())

    def __call__(self, *args):
        args_iter = iter(args)
        env : Dict[str, Node] = {}
        # For every argument, keep track of a list of all aliases of that argument.
        input_aliases = [set() for _ in range(len(args))]
        tensor_to_node_name_map = {}
        base_tensor_to_aliases_map = defaultdict(set)

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

            # Maintain mapping from each tensor in the program to its name in the graph
            tensor_to_node_name_map[result] = node.name

            for a in itertools.chain(node.args, node.kwargs.values()):
                arg = load_arg(a)
                if isinstance(arg, torch.Tensor):
                    for i, inpt in enumerate(args):
                        if isinstance(inpt, torch.Tensor) and is_view_of(arg, inpt):
                            input_aliases[i].add(a.name)

                    # Keep track of, for every "base" tensor in the program, what are all of its aliases.
                    curr_base_tensor_name = a.name if arg._base is None else tensor_to_node_name_map[arg._base]
                    base_tensor_to_aliases_map[curr_base_tensor_name].add(a.name)

            env[node.name] = result

        # Post-process our base-tensor-to-aliases info into a more convenient lookup table
        alias_map = defaultdict(set)
        for base_tensor_name, alias_names in base_tensor_to_aliases_map.items():
            for name in [base_tensor_name] + list(alias_names):
                alias_map[name] = alias_names

        return input_aliases, alias_map


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
def reinplace(gm: fx.GraphModule, *sample_args, **sample_kwargs):
    """
    Given an fx.GraphModule, modifies it to perform "reinplacing".
    We look for out-of-place op call sites like `b = a.add(...)`,
    and convert them to be inplace (`b = a.add_(...)`),
    as long as the input to the current operator ("a") isn't re-used
    anywhere later in the graph.

    Sample inputs are needed to determine aliasing relationships of the inputs.
    In general, we can't reinplace node `b = a.add(...)` if "a" aliases any of the
    inputs to the program.

    There is one exception though: if "a" is copied to at any point later in the program,
    e.g. `a.copy_(...)`, then we are free to re-use a's buffer any time
    before that node executes.

    This is an important optimization to include, to ensure that after running
    functionalization and then reinplacing, we don't regress the net memory usage
    of the original model.
    """

    # Maintain a mapping: for every variable in the graph,
    # what are all of nodes in the graph that use it as an input, where the data is read from?
    variable_to_nodes = defaultdict(list)
    for idx, node in enumerate(gm.graph.nodes):
        node_args = node.args
        # copy_() doesn't read from its first argument; it writes to it, overwriting previous data.
        # We don't want to treat it as "being used as an input".
        if node.target is torch.ops.aten.copy_.default:
            node_args = node_args[1:]
        for arg in itertools.chain(node_args, node.kwargs.values()):
            if not isinstance(arg, Iterable):
                arg = [arg]
            for a in arg:
                if isinstance(a, fx.node.Node):
                    variable_to_nodes[a.name].append([idx, node])

    # Keep track of the inputs to the graph and their aliases.
    # If the first argument to the an out-of-place op is a graph input,
    # we can't convert it to an inplace op.
    sample_args = [a.to(device='meta') if isinstance(a, torch.Tensor) else a for a in sample_args]
    # E.g. for a program with two tensor inputs x and y, graph_input_alias_names will look like:
    # [
    #   ['x', 'x_alias_1', ...]
    #   ['y', 'y_alias_1', ...]
    # ]
    graph_input_alias_names, graph_alias_map = ComputeInputAliasNames(gm)(*sample_args, **sample_kwargs)

    # In general, we can't inplace-ify ops that use program inputs (or their aliases) as arguments.
    # However, if we know that later in the program the input will be overwritten anyway,
    # then inplace-ifying is safe to do as long as it happens before the input is overwritten.
    #
    # This optimization is especially important to do because of functionalization:
    # functionalization will remove input mutations from a program, but append a copy_()
    # to the end of the program.
    #
    # Here, we're keeping a mapping of, for every input arg, the last place
    # in the program (if any) that its data was overwritten using a different tensor's data..
    # Mapping is from (input arg idx) -> (idx of the node that overwrites the data for that input)
    last_node_where_input_alias_is_overwritten: Dict[int, int] = defaultdict(lambda: -1)
    for node_idx, node in enumerate(gm.graph.nodes):
        if node.op == 'call_function' and node.target is torch.ops.aten.copy_.default:
            copied_to_arg_name = node.args[0].name
            copied_from_arg_name = node.args[1].name
            for input_idx, input_alias_names in enumerate(graph_input_alias_names):
                # Check that the program input is written to, but not read from here.
                if copied_to_arg_name in input_alias_names and copied_from_arg_name not in input_alias_names:
                    last_node_where_input_alias_is_overwritten[input_idx] = node_idx

    # Now do the real work: inplace-ify functional ops, subject to the constraints written below.
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
                # (This should be the case for all ops with inplace variants in ATen,
                # although we technically don't have guarantees for custom ops).
                assert len(node.target._schema.arguments) > 0
                assert 'Tensor' in str(node.target._schema.arguments[0].type)

                # Step 2: Check to see if the input to the op is re-used later in the graph.
                self_arg = node.args[0]
                self_arg_name = self_arg.name

                input_aliases_of_current_tensor = [
                    input_idx for (input_idx, alias_names) in enumerate(graph_input_alias_names)
                    if self_arg_name in alias_names
                ]
                # In general, we can't re-inplace an op if input was
                # an (alias to an) input to the graph.
                if len(input_aliases_of_current_tensor) > 0:
                    # However, if that input is overwritten by a copy_() later in the graph, then
                    # it's fair game to re-inplace.
                    if not any(last_node_where_input_alias_is_overwritten[inpt_alias_idx] > idx for inpt_alias_idx in input_aliases_of_current_tensor):
                        continue


                # For the current self_tensor, we want to know:
                # What are all of the nodes in the graph that use self_tensor, or any of its aliases, as an input?
                self_arg_aliases = graph_alias_map[self_arg_name]
                nodes_used = concatMap(lambda alias_name: variable_to_nodes[alias_name], self_arg_aliases)
                # for every node that uses the current variable, we've tracked what their index is
                # in the graph. If there aren't any nodes with a higher index that use the current
                # argument, that means it isn't re-used anywhere.
                if not any(node_idx > idx for node_idx, _ in nodes_used):
                    # Step 3: replace the current out-of-place op with its inplace variant.
                    node.target = inplace_op

                    # Step 4:
                    # Now that we've replaced b = a.foo() with a.foo_(),
                    # We need to replace any later usages of "b" with "a"
                    #
                    # A lot of the extra below is because we have several data structures
                    # That we need to update when we change the nodes' arguments:
                    # - variable_to_nodes (some nodes now take in different variables as arguments)
                    # - graph_alias_map (updating a view nodes' arguments will change with variables alias)
                    for node_to_update_idx, node_to_update in variable_to_nodes[node.name]:
                        new_args = []
                        for arg_idx, a in enumerate(node_to_update.args):
                            if isinstance(a, fx.node.Node) and a.name == node.name:
                                new_args.append(self_arg)
                                if node_to_update.op == 'call_function':
                                    schema_arg = node_to_update.target._schema.arguments[arg_idx]
                                    # And if this argument is aliasing according to the schema,
                                    # then we'll need to update our graph alias map.
                                    if schema_arg.alias_info is not None and not schema_arg.alias_info.is_write:
                                        graph_alias_map[node.name].remove(node_to_update.name)
                                        graph_alias_map[self_arg_name].add(node_to_update.name)
                                        graph_alias_map[node_to_update.name] = graph_alias_map[self_arg_name]
                            else:
                                new_args.append(a)
                        new_kwargs = {}
                        for kwarg_idx, (k, v) in enumerate(node_to_update.kwargs):
                            if isinstance(v, fx.node.Node) and v.name == node.name:
                                new_kwargs[k] = self_arg
                                if node_to_update.op == 'call_function':
                                    schema_arg = node_to_update.target._schema.arguments[len(node_to_update.args) + kwarg_idx]
                                    # And if this argument is aliasing according to the schema,
                                    # then we'll need to update our graph alias map.
                                    if schema_arg.alias_info is not None and not schema_arg.alias_info.is_write:
                                        graph_alias_map[node.name].remove(node_to_update.name)
                                        graph_alias_map[self_arg_name].add(node_to_update.name)
                                        graph_alias_map[node_to_update.name] = graph_alias_map[self_arg_name]
                            else:
                                new_kwargs[k] = v
                        node_to_update.args = new_args
                        node_to_update.kwargs = new_kwargs

                    # We also need to update our variable-to-node mapping,
                    # now that we changed some of our nodes' inputs.

                    # {self_arg} now gets some extra node entries
                    variable_to_nodes[self_arg_name] += [x for x in variable_to_nodes[node.name] if x[0] > idx]
                    # And {node.name} loses some entries (since we replaced {node.name} with {self_arg}
                    variable_to_nodes[node.name] = [x for x in variable_to_nodes[node.name] if x[0] <= idx]

    gm.recompile()
    return gm
