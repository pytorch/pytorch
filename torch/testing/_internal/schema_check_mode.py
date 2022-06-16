import torch
from torch.utils._pytree import tree_flatten
from torch.fx.operator_schemas import normalize_function
from torch.testing._internal.jit_utils import clone_inputs
from torch.utils._python_dispatch import TorchDispatchMode

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

        def is_mutable(alias_info):
            return alias_info is not None and alias_info.is_write

        def is_aliasing(lhs_alias_info, rhs_alias_info):
            if lhs_alias_info is None or rhs_alias_info is None:
                return False
            else:
                return bool(len(lhs_alias_info.before_set & rhs_alias_info.before_set))

        self.ops.append(func._schema.name)
        arguments = normalize_function(
            func,
            args,
            kwargs,
            normalize_to_only_use_kwargs=True
        ).kwargs

        cloned_arguments = dict(zip(arguments.keys(), clone_inputs(arguments.values())))
        out = func(*args, **kwargs)

        for argument in func._schema.arguments:
            name = argument.name if argument.name != "self" else "input"
            if arguments.get(name) is not None:
                before = tree_flatten(cloned_arguments.get(name))[0]
                after = tree_flatten(arguments.get(name))[0]
                if any([has_mutated(i, j) for i, j in zip(before, after)]) and not is_mutable(argument.alias_info):
                    raise RuntimeError(f"Argument {name} is not defined as mutable but was mutated")
                for b in before:
                    if not isinstance(out, tuple):
                        if has_aliased(b, out) and not is_aliasing(argument.alias_info, func._schema.returns[0].alias_info):
                            raise RuntimeError(f'Argument {name} is not defined to alias output but was aliasing')
                    else:
                        for j in range(len(out)):
                            if has_aliased(b, out[j]) and not is_aliasing(argument.alias_info, func._schema.returns[j].alias_info):
                                raise RuntimeError(f'Argument {name} is not defined to alias output but was aliasing')

        return out
