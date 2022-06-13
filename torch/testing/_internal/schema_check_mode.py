import torch
from torch.utils._pytree import tree_flatten
from torch.fx.operator_schemas import normalize_function
from torch.testing._internal.jit_utils import clone_inputs
from torch.utils._python_dispatch import TorchDispatchMode

# This TorchDispatchMode Subclass is used to verify op schemas
# This TorchDispatchMode Scubclass currently:
#  - Records the called ops
#  - Checks for mutations on all inputs

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
                before = tree_flatten(arguments.get(name))[0]
                after = tree_flatten(cloned_arguments.get(name))[0]
                if (any([has_mutated(i, j) for i, j in zip(before, after)]) and not argument.is_mutable):
                    raise RuntimeError(f"Argument {name} is not defined as mutable but was mutated")

        return out
