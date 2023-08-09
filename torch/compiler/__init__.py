import torch
from typing import Any, List, Optional

from torch.utils._pytree import (
    FlattenFunc,
    MaybeFromStrFunc,
    ToStrFunc,
    UnflattenFunc,
)

__all__ = [
    "compile",
    "assume_constant_result",
    "reset",
    "allow_in_graph",
    "list_backends",
    "disable",
    "dynamic_dim",
    "register_dataclass_as_pytree_node",
]

def compile(*args, **kwargs):
    """
    See :func:`torch.compile` for details on the arguments for this function.
    """
    return torch.compile(*args, **kwargs)

def reset() -> None:
    """
    This function clears all compilation caches and restores the system to its initial state.
    It is recommended to call this function, especially after using operations like `torch.compile(...)`
    to ensure a clean state before another unrelated compilation
    """
    import torch._dynamo

    torch._dynamo.reset()

def allow_in_graph(fn):
    """
    Customize which functions compilation will include in the generated graph.
    It bypasses all introspection of the symbolic python code in favor of
    directly writing it to the graph.
    If fn is a list or tuple of callables it recursively applies :func:`allow_in_graph()`
    to each function and returns a new list or tuple containing the modified functions

    Args:
        fn: A callable representing the function to be included in the graph.

    .. warning::

        :func:`allow_in_graph` skips TorchDynamo completely on the decorated function
        skipping all TorchDynamo safety checks (graph breaks, handling closures, etc).
        Therefore, one has to be very careful with :func:`allow_in_graph` since subsystems
        like AOT Autograd rely on torchdynamo
        If not careful, this could lead to soundness and really hard-to-debug issues.

    """
    import torch._dynamo

    return torch._dynamo.allow_in_graph(fn)


def list_backends(exclude_tags=("debug", "experimental")) -> List[str]:
    """
    Return valid strings that can be passed to `torch.compile(..., backend="name")`.

    Args:
        exclude_tags(optional): A tuple of strings representing tags to exclude.
    """
    import torch._dynamo

    return torch._dynamo.list_backends(exclude_tags)

def assume_constant_result(fn):
    """
    This function is used to mark a function `fn` as having a constant result.
    This allows the compiler to optimize away your function
    Returns The same function `fn`

    Args:
        fn: The function to be marked as having a constant result.

    .. warning::
        `assume_constant_result` can if invalid cause safety and soundness issues, :func:`torch.compile`
        will not attempt to validate whether the constant assumption is true or not

    """
    import torch._dynamo

    return torch._dynamo.assume_constant_result(fn)

def disable(fn=None, recursive=True):
    """
    This function provides both a decorator and a context manager to disable compilation on a function
    It also provides the option of recursively disabling called functions

    Args:
        fn (optional): The function to disable
        recursive (optional): A boolean value indicating whether the disabling should be recursive.
    """
    import torch._dynamo

    return torch._dynamo.disable(fn, recursive)


def dynamic_dim(t: torch.Tensor, index: int):
    """
    Note - [On Export Dynamic Dimension UX]

    After a lot of discussion, we have settled on a dynamic marking API
    for export that meets the following constraints:
    1) Stateless
    2) Safe for numerous .export calls within a single process
    3) Simple to use
    4) Can be extended to constraints easily

    While the underlying API is still torch._dynamo.mark_dynamic, we offer a higher
    level API that meets the constraints above.

    This API produces an object that is meant to be passed into torch.export()
    constraints field. See docs on torch.export for more details.

    Note - The output type and structure here is NOT BC and NOT A CONTRACT, we reserve
    the right to change the output here at any time, and will do so as we extend the API.

    result = torch.export(
        my_model,
        args,
        kwargs,
        constraints=[
            # if you do only dynamic_dim, this is sugar for
            # -Inf <= dynamic_dim(blah, 0) <= Inf; we don’t otherwise
            # permit direct int->bool conversion
            dynamic_dim(blah, 0),
            # operator overloading because it makes it clear whether
            # or not you’re inclusive-exclusive range or not
            0 <= dynamic_dim(blah, 1) <= 100,
            # NB: But we actually truncate ranges to be >= 2, because of
            # 0/1 specialization
        ]
    )
    """
    return torch._export.dynamic_dim(t, index)


def register_dataclass_as_pytree_node(
    typ: Any,
    flatten_fn: Optional[FlattenFunc] = None,
    unflatten_fn: Optional[UnflattenFunc] = None,
    to_str_fn: Optional[ToStrFunc] = None,
    maybe_from_str_fn: Optional[MaybeFromStrFunc] = None,
    *,
    return_none_fields: bool = False,
) -> None:
    """
    Registers a customized flatten/unflatten/serialization/deserialization for a dataclass.

    Once registered, the custom dataclass type can be used as valid input/output types for
    torch.export()
    """
    from torch._export.utils import register_dataclass_as_pytree_node
    return register_dataclass_as_pytree_node(
        typ, flatten_fn, unflatten_fn, to_str_fn, maybe_from_str_fn,
        return_none_fields=return_none_fields)
