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

        def have_values_aliased(lhs, rhs):
            return torch._C._is_alias_of(lhs, rhs) if isinstance(lhs, torch.Tensor) and isinstance(rhs, torch.Tensor) else False

        def is_aliasing(lhs_argument, rhs_argument):
            return bool(len(lhs_argument.before_set & rhs_argument.before_set))

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
                if (any([has_mutated(i, j) for i, j in zip(before, after)]) and not argument.is_mutable):
                    raise RuntimeError(f"Argument {name} is not defined as mutable but was mutated")
                for i in before:
                    if not isinstance(out, tuple):
                        if (have_values_aliased(i, out) and not is_aliasing(argument, func._schema.returns[0])):
                            raise RuntimeError(f'Argument {name} is not defined to alias output but was aliasing')
                    else:
                        for j in range(len(out)):
                            if (have_values_aliased(i, out[j]) and not is_aliasing(argument, func._schema.returns[j])):
                                raise RuntimeError(f'Argument {name} is not defined to alias output but was aliasing')

        return out
