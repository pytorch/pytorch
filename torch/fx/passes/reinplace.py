import torch
import torch.fx as fx
from torch.fx._compatibility import compatibility

from torch.multiprocessing.reductions import StorageWeakRef

from collections import defaultdict
from typing import Iterable
import itertools

__all__ = ['reinplace']

def concatMap(func, xs):
    for x in xs:
        for r in func(x):
            yield r

class GatherReinplaceDatastructures(torch.fx.Interpreter):
    def __init__(self, gm):
        super().__init__(gm)
        self.node_counter = -1
        # graph variable names of all program inputs
        self.input_names = []
        # Groups sets of the variable names of all aliased variables in the program
        self.alias_map = defaultdict(set)
        # For a given variable, tells you the names of all nodes in the program that use it.
        # (including the index of that node, so we know where in the program it is)
        self.names_to_used_nodes = defaultdict(list)
        # When you call `x.copy_(y)`, you overwrite x's data.
        # This stores the name of the node (if any) where a given variable in the program is overwritten.
        # We need to track this information for program inputs
        self.last_node_where_variable_is_overwritten = defaultdict(lambda: -1)

    def run_node(self, node):
        self.node_counter += 1
        args, kwargs = self.fetch_args_kwargs_from_env(node)

        # Update names_to_used_nodes map for this node
        # copy_() doesn't read from its first argument; it writes to it, overwriting previous data.
        # We don't want to treat it as "being used as an input".
        node_args = node.args
        if node.target is torch.ops.aten.copy_.default:
            node_args = node_args[1:]

        for arg in itertools.chain(node_args, node.kwargs.values()):
            if not isinstance(arg, Iterable):
                arg = [arg]
            for a in arg:
                if isinstance(a, fx.node.Node):
                    # We map to the actual fx.Node object, because later we might need to
                    # modify the arguments that this node is called with
                    self.names_to_used_nodes[a.name].append([self.node_counter, node])

        if node.op == 'placeholder':
            # Update alias map + input names when we see a placeholder
            r = super().placeholder(node.target, args, kwargs)
            self.input_names.append(node.name)
            self.alias_map[StorageWeakRef(r.storage()).cdata].add(node.name)
            return r
        elif node.op == 'call_function':
            # Update alias map when we see a call_function
            for runtime_arg, node_arg in itertools.chain(
                    zip(args, node.args), zip(kwargs.values(), node.kwargs.values())):
                if isinstance(runtime_arg, torch.Tensor):
                    self.alias_map[StorageWeakRef(runtime_arg.storage()).cdata].add(node_arg.name)

            # And update our mapping for variables that are copied to.
            if node.target is torch.ops.aten.copy_.default:
                self_arg = args[0]
                self.last_node_where_variable_is_overwritten[StorageWeakRef(self_arg.storage()).cdata] = self.node_counter

            return super().call_function(node.target, args, kwargs)
        else:
            return super().run_node(node)


    def __call__(self, *args, **kwargs):
        # Hmm, seems like fx.Interpreter.run() doesn't accept kwargs?
        super().run(*args)
        # Do some post-processing on our bookkeeping.
        # We want to avoid returning data structures that involve StorageImpl,
        # since that isn't the same from run to run, and replace it with variable names from the graph

        # For the alias map, map [variable name] -> [names of all aliased variables]
        alias_map = {}
        for aliased_var_names in self.alias_map.values():
            for v in aliased_var_names:
                alias_map[v] = aliased_var_names

        # For the copy_() map [program input var name] -> [node index of the last copy_() that wrote to the input's data]
        last_node_where_input_alias_is_overwritten = defaultdict(lambda: -1)
        for storage_ref, copy_node_idx in self.last_node_where_variable_is_overwritten.items():
            # We only actually care about variables used in copy_() ops that are program inputs (or their aliases)
            aliased_inputs = [inpt_name for inpt_name in self.input_names if inpt_name in self.alias_map[storage_ref]]
            for inpt_name in aliased_inputs:
                last_node_where_input_alias_is_overwritten[inpt_name] = copy_node_idx

        return self.input_names, alias_map, self.names_to_used_nodes, last_node_where_input_alias_is_overwritten

def schemas_match(functional_schema, inplace_schema):
    names_match = inplace_schema.name.endswith("_") and inplace_schema.name[:-1] == functional_schema.name
    arg_types_match = all(
        a1.type == a2.type for a1, a2 in zip(functional_schema.arguments, inplace_schema.arguments))
    # for the inplace op, its first argument should be mutable
    assert inplace_schema.arguments[0].alias_info is not None and inplace_schema.arguments[0].alias_info.is_write
    # and its remaining arguments shouldn't be.
    assert all(a.alias_info is None for a in inplace_schema.arguments[1:])
    return names_match and arg_types_match

def maybe_get_inplace_op(op):
    # __module__ seems broken; it returns torch._ops.aten which doesn't exist
    op_namespace = op.__module__.split(".")[-1]
    op_base_name = op.overloadpacket.__name__
    maybe_namespace_module = getattr(torch.ops, op_namespace)
    maybe_inplace_op = None if maybe_namespace_module is None else getattr(maybe_namespace_module, f'{op_base_name}_', None)
    if maybe_inplace_op is None:
        return None

    inplace_overloads = [
        getattr(maybe_inplace_op, overload_name) for overload_name in maybe_inplace_op.overloads()
    ]
    inplace_overloads_with_matching_schemas = [
        f
        for f in inplace_overloads
        if schemas_match(op._schema, f._schema)
    ]
    # This is for sanity: if foo() and foo_() are both operators,
    # we expect them to have compatible schemas.
    # (This is asserted by codegen for ATen, but might not be true
    # for other arbitrary operators).
    assert len(inplace_overloads_with_matching_schemas) == 1
    inplace_op = inplace_overloads_with_matching_schemas[0]
    return inplace_op

@compatibility(is_backward_compatible=True)
def reinplace(gm: fx.GraphModule, *sample_args, **sample_kwargs):
    """
    Given an fx.GraphModule, modifies it to perform "reinplacing",
    mutating the nodes of the graph.
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


    sample_args = [a.to(device='meta') if isinstance(a, torch.Tensor) else a for a in sample_args]
    bookkeeping_data = GatherReinplaceDatastructures(gm)(*sample_args, **sample_kwargs)

    input_names = bookkeeping_data[0]
    alias_map = bookkeeping_data[1]
    names_to_used_nodes = bookkeeping_data[2]
    last_node_where_input_alias_is_overwritten = bookkeeping_data[3]

    # inplace-ify functional ops, subject to the constraints written below.
    for idx, node in enumerate(gm.graph.nodes):
        if node.op == 'call_function':
            # Step 1: Check to see if this operator has an inplace variant.
            maybe_inplace_op = maybe_get_inplace_op(node.target)
            if maybe_inplace_op is not None:
                # This is a proxy check for ensuring that the first argument is "tensor-like"
                # (This should be the case for all ops with inplace variants in ATen,
                # although we technically don't have guarantees for custom ops).
                assert len(node.target._schema.arguments) > 0
                assert 'Tensor' in str(node.target._schema.arguments[0].type)

                # Step 2: Check to see if the input to the op is re-used later in the graph.
                self_arg = node.args[0]
                self_arg_name = self_arg.name

                input_alias_names = set()
                for inpt_name in input_names:
                    input_alias_names |= alias_map[inpt_name]

                input_aliases_of_current_tensor = [n for n in input_alias_names if n in alias_map[self_arg_name]]

                # In general, we can't re-inplace an op if input was
                # an (alias to an) input to the graph.
                if len(input_aliases_of_current_tensor) > 0:
                    # However, if that input is overwritten by a copy_() later in the graph, then
                    # it's fair game to re-inplace.
                    if not any(last_node_where_input_alias_is_overwritten[inpt_alias_idx] > idx for inpt_alias_idx in input_aliases_of_current_tensor):
                        continue

                # For the current self_tensor, we want to know:
                # What are all of the nodes in the graph that use self_tensor, or any of its aliases, as an input?
                self_arg_aliases = alias_map[self_arg_name]
                nodes_used = concatMap(lambda alias_name: names_to_used_nodes[alias_name], self_arg_aliases)
                # for every node that uses the current variable, we've tracked what their index is
                # in the graph. If there aren't any nodes with a higher index that use the current
                # argument, that means it isn't re-used anywhere.
                if not any(node_idx > idx for node_idx, _ in nodes_used):
                    # Step 3: replace the current out-of-place op with its inplace variant.
                    node.target = maybe_inplace_op

                    # Step 4:
                    # Now that we've replaced b = a.foo() with a.foo_(),
                    # We need to replace any later usages of "b" with "a"
                    #
                    # A lot of the extra below is because we have several data structures
                    # That we need to update when we change the nodes' arguments:
                    # - names_to_used_nodes (some nodes now take in different variables as arguments)
                    # - alias_map (updating a view nodes' arguments will change with variables alias)
                    for node_to_update_idx, node_to_update in names_to_used_nodes[node.name]:
                        new_args = []
                        for arg_idx, a in enumerate(node_to_update.args):
                            if isinstance(a, fx.node.Node) and a.name == node.name:
                                new_args.append(self_arg)
                                if node_to_update.op == 'call_function':
                                    schema_arg = node_to_update.target._schema.arguments[arg_idx]
                                    # And if this argument is aliasing according to the schema,
                                    # then we'll need to update our graph alias map.
                                    if schema_arg.alias_info is not None and not schema_arg.alias_info.is_write:
                                        alias_map[node.name].remove(node_to_update.name)
                                        alias_map[self_arg_name].add(node_to_update.name)
                                        alias_map[node_to_update.name] = alias_map[self_arg_name]
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
                                        alias_map[node.name].remove(node_to_update.name)
                                        alias_map[self_arg_name].add(node_to_update.name)
                                        alias_map[node_to_update.name] = alias_map[self_arg_name]
                            else:
                                new_kwargs[k] = v
                        node_to_update.args = new_args
                        node_to_update.kwargs = new_kwargs

                    # We also need to update our variable-to-node mapping,
                    # now that we changed some of our nodes' inputs.

                    # {self_arg} now gets some extra node entries
                    names_to_used_nodes[self_arg_name] += [x for x in names_to_used_nodes[node.name] if x[0] > idx]
                    # And {node.name} loses some entries (since we replaced {node.name} with {self_arg}
                    names_to_used_nodes[node.name] = [x for x in names_to_used_nodes[node.name] if x[0] <= idx]

    gm.recompile()
    return gm
