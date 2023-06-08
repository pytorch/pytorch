import torch
from typing import List

__all__ = [
    "compile",
    "assume_constant_result",
    "reset",
    "allow_in_graph",
    "list_backends",
    "disable",
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

    Example::

        .. code-block:: python

            import torch
            torch.compiler.reset()
            torch.compile(...)
            torch.compiler.reset()
    """

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

    return torch._dynamo.allow_in_graph(fn)


def list_backends(exclude_tags=("debug", "experimental")) -> List[str]:
    """
    Return valid strings that can be passed to `torch.compile(..., backend="name")`.

    Args:
        exclude_tags(optional): A tuple of strings representing tags to exclude.

    Example::

        .. code-block:: python

            valid_backends = list_backends(exclude_tags=("debug", "experimental"))

    """

    return torch._dynamo.list_backends(exclude_tags)

def assume_constant_result(fn):
    """
    This function is used to mark a function `fn` as having a constant result.
    This allows the compiler to optimize away your function
    Returns The same function `fn`

    Args:
        fn: The function to be marked as having a constant result.

    Example::

        .. code-block:: python

            marked_function = assume_constant_result(my_function)

    .. warning::
        `assume_constant_result` can if invalid cause safety and soundness issues, `torch.compile`
        will not attempt to validate whether the constant assumption is true or not

    """

    return torch._dynamo.assume_constant_result(fn)

def disable(fn=None, recursive=True):
    """
    This function provides both a decorator and a context manager to disable compilation on a function
    It also provides the option of recursively disabling called functions

    Args:
        fn (optional): The function to disable
        recursive (optional): A boolean value indicating whether the disabling should be recursive.

    Example::

        .. code-block:: python

            # The decorator without recursive disabling
            @disable(recursive=False)
            def my_function():

            # The context manager with recursive disabling:
            with disable(recursive=True):
                ...
    """

    return torch._dynamo.disable(fn, recursive)
