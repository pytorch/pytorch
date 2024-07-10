from collections import defaultdict
from typing import Callable, Dict, List, Tuple, Union

import torch

from torch import nn, Tensor

# Type helpers
InputsType = Union[Tensor, Tuple[Tensor, ...]]
# A Getter takes in a device and returns a callable and the inputs to that callable
GetterReturnType = Tuple[Callable[..., Tensor], InputsType]
GetterType = Callable[[torch.device], GetterReturnType]
# V here refers to the v in either vjp, jvp, vhp or hvp
VType = Union[None, Tensor, Tuple[Tensor, ...]]
# Type used to store timing results. The first key is the model name, the second key
# is the task name, the result is a Tuple of: speedup, mean_before, var_before, mean_after, var_after.
TimingResultType = Dict[str, Dict[str, Tuple[float, ...]]]


# Utilities to make nn.Module "functional"
# In particular the goal is to be able to provide a function that takes as input
# the parameters and evaluate the nn.Module using fixed inputs.
def _del_nested_attr(obj: nn.Module, names: List[str]) -> None:
    """
    Deletes the attribute specified by the given list of names.
    For example, to delete the attribute obj.conv.weight,
    use _del_nested_attr(obj, ['conv', 'weight'])
    """
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        _del_nested_attr(getattr(obj, names[0]), names[1:])


def _set_nested_attr(obj: nn.Module, names: List[str], value: Tensor) -> None:
    """
    Set the attribute specified by the given list of names to value.
    For example, to set the attribute obj.conv.weight,
    use _del_nested_attr(obj, ['conv', 'weight'], value)
    """
    if len(names) == 1:
        setattr(obj, names[0], value)
    else:
        _set_nested_attr(getattr(obj, names[0]), names[1:], value)


def extract_weights(mod: nn.Module) -> Tuple[Tuple[Tensor, ...], List[str]]:
    """
    This function removes all the Parameters from the model and
    return them as a tuple as well as their original attribute names.
    The weights must be re-loaded with `load_weights` before the model
    can be used again.
    Note that this function modifies the model in place and after this
    call, mod.parameters() will be empty.
    """
    orig_params = tuple(mod.parameters())
    # Remove all the parameters in the model
    names = []
    for name, p in list(mod.named_parameters()):
        _del_nested_attr(mod, name.split("."))
        names.append(name)

    # Make params regular Tensors instead of nn.Parameter
    params = tuple(p.detach().requires_grad_() for p in orig_params)
    return params, names


def load_weights(mod: nn.Module, names: List[str], params: Tuple[Tensor, ...]) -> None:
    """
    Reload a set of weights so that `mod` can be used again to perform a forward pass.
    Note that the `params` are regular Tensors (that can have history) and so are left
    as Tensors. This means that mod.parameters() will still be empty after this call.
    """
    for name, p in zip(names, params):
        _set_nested_attr(mod, name.split("."), p)


# Utilities to read/write markdown table-like content.
def to_markdown_table(res: TimingResultType, header: Tuple[str, ...] = None) -> str:
    if header is None:
        header = ("model", "task", "mean", "var")
    out = ""

    def write_line(*args):
        nonlocal out
        out += f"| {' | '.join(str(a) for a in args)} |\n"

    # Make it a markdown table
    write_line(*header)
    write_line(*["--"] * len(header))
    for model, tasks in res.items():
        for task, line in tasks.items():
            write_line(*(model, task) + line)

    return out


def from_markdown_table(data: str) -> TimingResultType:
    out = data.strip().split("\n")
    out = out[2:]  # Ignore the header lines

    res: TimingResultType
    res = defaultdict(defaultdict)

    for line in out:
        model, task, mean, var = (f.strip() for f in line.strip().split("|") if f)
        res[model][task] = (float(mean), float(var))

    return res


def check_for_functorch():
    try:
        import functorch  # noqa: F401

        return True
    except ImportError:
        return False
