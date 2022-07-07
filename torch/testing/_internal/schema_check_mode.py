import torch
from torch.utils._pytree import tree_flatten, tree_map
from torch.fx.operator_schemas import normalize_function
from torch.testing._internal.jit_utils import clone_inputs
from torch.utils._python_dispatch import TorchDispatchMode
from itertools import combinations
from collections import namedtuple
from copy import deepcopy

# Named Tuples used within SchemaCheckMode
Mutation = namedtuple('Mutation', ['op_name', 'arg_name'])
Aliasing = namedtuple('Aliasing', ['op_name', 'arg_name', 'output_number'])

# This TorchDispatchMode Subclass is used to verify op schemas
# This TorchDispatchMode Scubclass currently:
#  - Records the called ops
#  - Checks for mutations on all inputs
#  - Checks for aliasing on all inputs

class SchemaCheckMode(TorchDispatchMode):
    def __init__(self):
        # Information recorded for testing purposes. For example:
        #  - incorrect schemas
        #  - overly conservative schemas
        self.ops = []
        self.mutated = []
        self.aliasing = []

    def reset_cache(self):
        self.ops.clear()
        self.mutated.clear()
        self.aliasing.clear()

    def display_ops(self):
        print(*self.ops, sep=",")

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        def has_mutated(before, after, md):
            if type(before) == torch.Tensor and type(after) == torch.Tensor:
                return not (
                    torch.equal(before, after) and
                    md[0] == after.stride() and
                    md[1] == after.storage()._cdata
                )
            return False

        def has_aliased(lhs, rhs):
            if type(lhs) == torch.Tensor and type(rhs) == torch.Tensor:
                return torch._C._is_alias_of(lhs, rhs)
            return False

        def is_mutable(arg_alias_pairs):
            for arg in arg_alias_pairs:
                if arg.alias_info is not None and arg.alias_info.is_write:
                    return True
            return False

        def is_aliasing(output, arg_alias_pairs):
            for arg in arg_alias_pairs:
                if arg.alias_info is not None:
                    if '*' in arg.alias_info.after_set:
                        same_types = output.type == arg.type
                        elems_same_types = (isinstance(output.type, torch._C.ListType) and output.type.getElementType() == arg.type)
                        if same_types or elems_same_types:
                            return True
                    elif output.alias_info is not None:
                        share_aliasing_sets = bool(len(output.alias_info.after_set & arg.alias_info.after_set))
                        if share_aliasing_sets:
                            return True
            return False

        def are_args_aliasing(lhs, rhs):
            for lhs_value in lhs:
                for rhs_value in rhs:
                    if (has_aliased(lhs_value, rhs_value)):
                        return True
            return False

        def standardize_name(name):
            return name if name != "self" else "input"

        def unwrap(e):
            if isinstance(e, torch.Tensor) and not type(e) == torch.Tensor:
                try:
                    return e.elem
                except AttributeError as t:
                    return e
            return e

        def parse_metadata(e):
            if isinstance(e, torch.Tensor):
                if not type(e) == torch.Tensor:
                    try:
                        current = e.elem
                        return (deepcopy(current.stride()), current.storage()._cdata)
                    except AttributeError as t:
                        return None
                else:
                    return (deepcopy(e.stride()), e.storage()._cdata)
            return None

        self.ops.append(func._schema.name)

        # Clone and process arguments and outputs
        pre_arguments = normalize_function(
            func,
            args,
            kwargs,
            normalize_to_only_use_kwargs=True
        ).kwargs

        c_p_args = dict(zip(pre_arguments.keys(), clone_inputs(pre_arguments.values())))
        cloned_arguments = {name : tree_map(unwrap, tree_flatten(c_p_args.get(name))[0]) for name in c_p_args}
        cloned_metadata = {name : tree_map(parse_metadata, tree_flatten(pre_arguments.get(name))[0]) for name in pre_arguments}

        out = func(*args, **kwargs)

        arguments = {name : tree_map(unwrap, tree_flatten(pre_arguments.get(name))[0]) for name in pre_arguments}
        tuple_out = out if isinstance(out, tuple) else (out, )
        u_out = [tree_map(unwrap, tree_flatten(i)[0]) for i in tuple_out]

        # Construct an aliasing map between op arguments for verifying aliasing pairs
        # between op arguments and op outputs. This is used to allow cases where two aliasing arguments
        # cause a non-mutable/non-aliasing argument to mutate or alias.
        arg_alias_pairs_map = {standardize_name(arg.name) : [arg] for arg in func._schema.arguments}

        # Construct an aliasing set for each output for verifying aliasing pairs
        # between op outputs
        out_alias_pairs_map = [set() for arg in func._schema.returns]

        # Aliasing between arguments
        for i_arg, j_arg in combinations(func._schema.arguments, 2):
            i_values = arguments.get(standardize_name(i_arg.name))
            j_values = arguments.get(standardize_name(j_arg.name))
            if are_args_aliasing(i_values, j_values):
                arg_alias_pairs_map[standardize_name(i_arg.name)].append(j_arg)
                arg_alias_pairs_map[standardize_name(j_arg.name)].append(i_arg)

        # Process arguments with outputs
        for arg in func._schema.arguments:
            name = standardize_name(arg.name)
            if arguments.get(name) is not None:
                arg_alias_pairs = arg_alias_pairs_map[name]
                before = cloned_arguments.get(name)
                md = cloned_metadata.get(name)
                after = arguments.get(name)
                for v in after:
                    for i in range(len(u_out)):
                        for j in range(len(u_out[i])):
                            if has_aliased(v, u_out[i][j]):
                                if not is_aliasing(func._schema.returns[i], arg_alias_pairs):
                                    raise RuntimeError(f'Argument {name} is not defined to alias output but was aliasing')
                                else:
                                    self.aliasing.append(Aliasing(func._schema.name, name, f"output_{i}"))
                                    out_alias_pairs_map[i].add(name)
                                    break
                if any(has_mutated(i, j, k) for i, j, k in zip(before, after, md)):
                    if not is_mutable(arg_alias_pairs):
                        raise RuntimeError(f"Argument {name} is not defined as mutable but was mutated")
                    else:
                        self.mutated.append(Mutation(func._schema.name, name))

        # Aliasing between outputs
        for i, j in combinations(range(len(func._schema.returns)), 2):
            if are_args_aliasing(u_out[i], u_out[j]):
                share_aliasing_inputs = bool(len(out_alias_pairs_map[i] & out_alias_pairs_map[j]))
                if (not share_aliasing_inputs):
                    raise RuntimeError(f'Outputs {i} and {j} alias unexpectedly')

        return out
