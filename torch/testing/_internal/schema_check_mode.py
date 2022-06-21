import torch
from torch.utils._pytree import tree_flatten, tree_map
from torch.fx.operator_schemas import normalize_function
from torch.testing._internal.jit_utils import clone_inputs
from torch.utils._python_dispatch import TorchDispatchMode
from itertools import combinations

# This TorchDispatchMode Subclass is used to verify op schemas
# This TorchDispatchMode Scubclass currently:
#  - Records the called ops
#  - Checks for mutations on all inputs
#  - Checks for aliasing on all inputs

class SchemaCheckMode(TorchDispatchMode):
    def __init__(self):
        self.ops = []

    def reset_cache(self):
        self.ops.clear()

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
        # between op arguments and op outputs
        arg_alias_pairs_map = {arg.name : [arg] for arg in func._schema.arguments}
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

        for arg in func._schema.arguments:
            name = standardize_name(arg.name)
            if arguments.get(name) is not None:
                arg_alias_pairs = arg_alias_pairs_map[arg.name]
                before = tree_flatten(cloned_arguments.get(name))[0]
                after = tree_flatten(arguments.get(name))[0]
                u_values = tree_map(unwrap, after)
                u_out = tree_map(unwrap, out)
                u_out = u_out if isinstance(u_out, tuple) else (u_out, )
                if any([has_mutated(i, j) for i, j in zip(before, after)]) and not is_mutable(arg_alias_pairs):
                    raise RuntimeError(f"Argument {name} is not defined as mutable but was mutated")
                for v in u_values:
                    for j in range(len(u_out)):
                        if has_aliased(v, u_out[j]) and not is_aliasing(func._schema.returns[j].alias_info, arg_alias_pairs):
                            raise RuntimeError(f'Argument {name} is not defined to alias output but was aliasing')

        return out
