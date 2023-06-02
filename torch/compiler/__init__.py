import torch
import torch._dynamo
import torch._inductor
from typing import Callable, Union, List, Set, Tuple, Any, Dict

__all__ = [
    "compile",
    "is_enabled",
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
    to ensure a clean state before subsequent compilation.

    Usage:
    1. Call `reset()` to clear all compilation caches and restore the initial state.
    2. Perform any desired operations, such as `torch.compile(...)`.
    3. If you need to start fresh or perform another `torch.compile(...)`, call `reset()` to ensure a clean state.

    """

    torch._dynamo.reset()

def allow_in_graph(fn):
    """
    Customize which functions compilation will include in the generated graph.
    It bypasses all introspection of the symbolic python code in favor of
    directly writing it to the graph.

    Arguments:
    - fn: A callable representing the function to be included in the graph.

    Returns:
    - If `fn` is a single callable, it adds the function to the list of allowed functions
    in compilations internal storage and returns the function itself.
    - If `fn` is a list or tuple of callables, it recursively applies the `allow_in_graph()`
    function to each item in the list or tuple and returns a new list containing the
    modified functions.

    Note:
    - The function assumes that `fn` is a callable. If it is not, an assertion error is raised.

    Warning:
    - `allow_in_graph` skips TorchDynamo completely on the decorated function
    skipping all TorchDynamo safety checks (graph breaks, handling closures, etc).
    - Therefore, one has to be very careful with `allow_in_graph`
    Today, downstream components like AOT Autograd rely on TorchDynamo to take care of complex Python features
    but `allow_in_graph` bypasses TorchDynamo.
    - If not careful, this could lead to soundness and really hard-to-debug issues.
    """

    return torch._dynamo.allow_in_graph(fn)


def list_backends(exclude_tags=("debug", "experimental")) -> List[str]:
    """
    Return valid strings that can be passed to `torch.compile(..., backend="name")`.

    Arguments:
    - exclude_tags (optional): A tuple of strings representing tags to exclude.
    Backends with any of the specified tags will not be included in the returned list.
    By default, the tags "debug" and "experimental" are excluded.

    Returns:
    - A sorted list of backend names that can be passed to `torch.compile()`.

    Example:
    To retrieve a list of available backends excluding the tags "debug" and "experimental",
    we can call the `list_backends()` function as follows:

    ::
        valid_backends = list_backends(exclude_tags=("debug", "experimental"))

    """

    return torch._dynamo.list_backends(exclude_tags)

def assume_constant_result(fn):
    """
    This function is used to mark a function `fn` as having a constant result.
    This allows the compiler to optimize away your function

    Arguments:
    - fn: The function to be marked as having a constant result.

    Returns:
    - The same function `fn`

    Example:
    To mark a function `my_function()` as having a constant result, we can call the
    `assume_constant_result()` function as follows:

    ::
        marked_function = assume_constant_result(my_function)

    Warning:
    - `assume_constant_result` can if invalid cause safety and soundness issues, `torch.compile`
    will not attempt to validate whether the constant assumption is true or not

    """

    return torch._dynamo.assume_constant_result(fn)

def disable(fn=None, recursive=True):
    """
    This function provides both a decorator and a context manager to disable compilation.

    Arguments:
    - fn (optional): The function to be decorated or used as a context manager.
    If provided, compilation will be disabled for the decorated function frame and any
    recursively invoked functions within it. If not provided, a context manager will be returned.
    - recursive (optional): A boolean value indicating whether the disabling should be recursive.
    If set to True (default), compilation is completely skipped on the decorated function frame
    and any recursively invoked functions within it. If set to False, compilation skips frames
    associated with the function code but still processes recursively invoked frames.

    Returns:
    - If `recursive=True` and `fn` is provided, a decorated version of the function `fn` is returned,
    with compilation disabled for the decorated function frame and any recursively invoked functions.
    - If `recursive=True` and `fn` is not provided, a context manager is returned, allowing compilation
    to be disabled within a specific code block.
    - If `recursive=False`, the `skip()` function is returned, which allows compilation to skip frames
    associated with the function code but still process recursively invoked frames.

    Note:
    - When using the decorator or context manager compilation processing is selectively disabled for
    the decorated function frame and any recursive function calls, depending on the `recursive` flag.
    - The function internally uses the `innermost_fn()` function to ensure that the innermost function
    is decorated when `fn` is provided.
    - The `skip()` function is used when `recursive=False` to skip frames associated with the function code
    but still process recursively invoked frames.

    Example:
    1. Using the decorator with recursive disabling:

    ::
      @disable(recursive=True)
      def my_function():

    In this example, `my_function()` is decorated with compi disabled, meaning that compilations
    processing will be skipped for the function frame and any recursive function calls within it.

    2. Using the context manager with recursive disabling:

    ::
      with disable(recursive=True):

    In this example, the code block within the `with` statement will have compilation disabled, meaning
    that compilations processing will be skipped for the code within the block and any recursive function
    calls within that code.

    3. Using the skip function with non-recursive disabling:

    ::
      disable(recursive=False)(my_function)

    In this example, `my_function()` is wrapped with the `skip()` function, which disables compilations
    processing for the function frame but still processes recursively invoked functions.

    """

    return torch._dynamo.disable(fn, recursive)
