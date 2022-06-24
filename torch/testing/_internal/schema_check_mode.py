import torch
from torch.utils._pytree import tree_flatten, tree_map
from torch.fx.operator_schemas import normalize_function
from torch.testing._internal.jit_utils import clone_inputs
from torch.utils._python_dispatch import TorchDispatchMode
from itertools import combinations
from collections import namedtuple

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
        def has_mutated(before, after):
            return not torch.equal(before, after) if isinstance(before, torch.Tensor) and isinstance(after, torch.Tensor) else False

        def has_aliased(lhs, rhs):
            return torch._C._is_alias_of(lhs, rhs) if isinstance(lhs, torch.Tensor) and isinstance(rhs, torch.Tensor) else False

        def is_mutable(arg_alias_pairs):
            for arg in arg_alias_pairs:
                if arg.alias_info is not None and arg.alias_info.is_write:
                    return True
            return False

        def is_aliasing(output_alias_info, arg_alias_pairs):
            if output_alias_info is None:
                return False
            for arg in arg_alias_pairs:
                if arg.alias_info is not None and bool(len(output_alias_info.after_set & arg.alias_info.after_set)):
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
            else:
                return e
        self.ops.append(func._schema.name)
        arguments = normalize_function(
            func,
            args,
            kwargs,
            normalize_to_only_use_kwargs=True
        ).kwargs

        cloned_arguments = dict(zip(arguments.keys(), clone_inputs(arguments.values())))
        out = func(*args, **kwargs)

        # Construct an aliasing map between op arguments for verifying aliasing pairs
        # between op arguments and op outputs. This is used to allow cases where two aliasing arguments
        # cause a non-mutable/non-aliasing argument to mutate or alias.
        arg_alias_pairs_map = {arg.name : [arg] for arg in func._schema.arguments}
        out_alias_pairs_map = [set() for arg in func._schema.returns]

        # Aliasing between arguments
        for i_arg, j_arg in combinations(func._schema.arguments, 2):
            i_values = tree_map(
                unwrap,
                tree_flatten(arguments.get(standardize_name(i_arg.name)))[0])
            j_values = tree_map(
                unwrap,
                tree_flatten(arguments.get(standardize_name(j_arg.name)))[0])
            if are_args_aliasing(i_values, j_values):
                arg_alias_pairs_map[i_arg.name].append(j_arg)
                arg_alias_pairs_map[j_arg.name].append(i_arg)

        # Process arguments with outputs
        for arg in func._schema.arguments:
            name = standardize_name(arg.name)
            if arguments.get(name) is not None:
                arg_alias_pairs = arg_alias_pairs_map[arg.name]
                before = tree_flatten(cloned_arguments.get(name))[0]
                after = tree_flatten(arguments.get(name))[0]
                u_values = tree_map(unwrap, after)
                u_out = tree_map(unwrap, out)
                u_out = u_out if isinstance(u_out, tuple) else (u_out, )
                if any([has_mutated(i, j) for i, j in zip(before, after)]):
                    if not is_mutable(arg_alias_pairs):
                        raise RuntimeError(f"Argument {name} is not defined as mutable but was mutated")
                    else:
                        self.mutated.append(Mutation(func._schema.name, name))
                for v in u_values:
                    for j in range(len(u_out)):
                        if has_aliased(v, u_out[j]):
                            if not is_aliasing(func._schema.returns[j].alias_info, arg_alias_pairs):
                                raise RuntimeError(f'Argument {name} is not defined to alias output but was aliasing')
                            else:
                                self.aliasing.append(Aliasing(func._schema.name, name, f"output_{j}"))
                                out_alias_pairs_map[j].add(name)

        # Aliasing between outputs
        for i, j in combinations(range(len(func._schema.returns)), 2):
            i_values = tree_map(
                unwrap,
                tree_flatten(out[i])[0])
            j_values = tree_map(
                unwrap,
                tree_flatten(out[j])[0])
            if are_args_aliasing(i_values, j_values) and not bool(len(out_alias_pairs_map[i] & out_alias_pairs_map[j])):
                raise RuntimeError(f'Outputs {i} and {j} alias unexpectedly')

        return out
