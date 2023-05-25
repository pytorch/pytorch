import torch
import torch._dynamo
import torch._inductor
from typing import Callable, Union, List, Set, Tuple, Any, Dict
from torch._dynamo.eval_frame import OptimizedModule

__all__ = [
    "OptimizedModule",
    "compile",
    "is_enabled",
    "reset",
    "allow_in_graph",
    "list_backends",
    "explain",
    "is_compiling",
    "disable",
    "list_options",
    "list_mode_options",
]

def compile(*args, **kwargs):
    """
    See :func:`torch.compile` for details on the arguments for this function.
    """

    return torch.compile(*args, **kwargs)

def is_enabled() -> bool:
    """
    Check if compilation is enabled in the current environment.

    This function checks if the `torch.compile` attribute exists in the `torch` module,
    indicating that compilation is enabled. It returns `True` if compilation is enabled,
    and `False` otherwise.

    Returns:
    - A boolean value indicating whether compilation is enabled (`True`) or not (`False`).

    Note:
    - The function checks the presence of the `torch.compile` attribute using the `hasattr()` function.
    - If `torch.compile` exists, it indicates that compilation is enabled, and the function returns `True`.
    - If `torch.compile` does not exist, it indicates that compilation is not enabled, and the function returns `False`.

    """

    if hasattr(torch, "compile"):
        return True
    else:
        return False

def reset() -> None:
    """
    Clear all compile caches and restore the initial state.

    This function clears all compile caches and resets various internal state variables to their initial values.

    It performs the following operations:
    1. Clears the input and output code caches.
    2. Clears the original code map.
    3. Clears the guard failures.
    4. Clears the cache for the `ContinueExecutionCache` class.
    5. If the most recent backend in the evaluation frame has a 'reset' method, it is called.
    6. Resets the most recent backend in the evaluation frame to None.
    7. Clears the compilation metrics.
    8. Resets the frame count to its initial value.

    Note:
    - Call this function if you're compiling more than one model in a program or if you're changing the compile options

    """

    torch._dynamo.reset()

def allow_in_graph(fn):
    """
    Customize which functions compilation will include in the generated graph.
    Similar to `torch.fx.wrap()` it can be used as an escape hatch from graph breaks

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
    """


def list_backends(exclude_tags=("debug", "experimental")) -> List[str]:
    """
    Return valid strings that can be passed to `torch.compile(..., backend="name")`.

    Arguments:
    - exclude_tags (optional): A tuple of strings representing tags to exclude.
    Backends with any of the specified tags will not be included in the returned list.
    By default, the tags "debug" and "experimental" are excluded.

    Returns:
    - A sorted list of backend names that can be passed to `torch.compile()`.

    Note:
    - The function internally performs a lazy import using the `_lazy_import()` function.
    - The `exclude_tags` argument allows users to filter out backends based on their tags.
    If the argument is not provided or is set to `None`, no tags will be excluded.

    Example:
    To retrieve a list of available backends excluding the tags "debug" and "experimental",
    we can call the `list_backends()` function as follows:

    ::
        valid_backends = list_backends(exclude_tags=("debug", "experimental"))

    """

    return torch._dynamo.list_backends(exclude_tags)

def explain(f, *args, **kwargs):
    """
    Run the function `f` with compilation and provide an explanation of the optimization process.

    This function runs the specified function `f` with compilations optimization process enabled.
    It captures information about the optimization process, including the number of graphs produced,
    the reasons for graph breaks, the operations per graph, and the output guards.

    Arguments:
    - `f`: The function to be optimized and analyzed.
    - `*args`, `**kwargs`: Arguments and keyword arguments to be passed to the function `f`.

    Returns:
    A tuple containing the following information:
    - explanation: A summary of the optimization process, including the number of graphs produced,
    the number of graph breaks, and the total number of operations.
    - out_guards: A list of output guards captured during the optimization process.
    - graphs: A list of TorchFX GraphModule objects representing the produced graphs.
    - ops_per_graph: A list of lists, where each inner list contains the operations (node targets)
    present in a graph.
    - break_reasons: A list of graph break reasons encountered during the optimization process.
    - explanation_verbose: A detailed explanation including the break reasons with formatted stack traces.

    Note:
    - The function imports and uses the `reset()` function from the same module to reset
    compilations internal state before running `f`.
    - The function internally defines several helper functions to accumulate graphs, count operations,
    export guards, and format explanations.
    - The function temporarily patches the `most_recent_backend` attribute to None using the `patch()` function
    from the `unittest.mock` module.

    Example:
    To run a function `my_function()` with compilation and obtain an explanation of the optimization process,
    we can call the `explain()` function as follows:

    ::
        explanation, out_guards, graphs, ops_per_graph, break_reasons, explanation_verbose =
        explain(my_function, arg1, arg2, keyword_arg=value)

    In this example, `my_function()` will be optimized using compilation, and the results and information
    about the optimization process will be stored in the respective variables.

    """

    return torch._dynamo.explain(f, *args, **kwargs)

def assume_constant_result(fn):
    """
    This function is used to mark a function `fn` as having a constant result.
    This allows the compiler to optimize away your function

    Arguments:
    - fn: The function to be marked as having a constant result.

    Returns:
    - The same function `fn` with the `_dynamo_marked_constant` attribute set to `True`.

    Example:
    To mark a function `my_function()` as having a constant result, we can call the
    `assume_constant_result()` function as follows:

    ::
        marked_function = assume_constant_result(my_function)

    """

    return torch._dynamo.assume_constant_result(fn)

def is_compiling() -> bool:
    """
    This function checks the status of compilation phase

    Returns:
    - A boolean value indicating if compilation is currently compiling

    Example:
    We can check if compilation is currently compiling by calling the function as follows:

    ::
        compiling = torch.compiler.is_compiling()

    """

    return torch._dynamo.is_compiling()

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

def list_options() -> Dict[str, Any]:
    """
    This function returns a dictionary that provides information about the available
    optimizations and debug configurations that can be used with the `torch.compile()`
    function. The options are documented in `torch._inductor.config`.

    Returns:
    - A dictionary containing the available optimizations and debug configurations.

    Example:
    To retrieve the available options for optimizations and debug configurations,
    we can call the `list_options()` function as follows:

    ::
        options = torch.compiler.list_options()

    """

    return torch._inductor.list_options()

def list_mode_options(mode: str = None) -> Dict[str, Any]:
    """
    This function returns a dictionary that provides information about the optimizations
    performed by each available mode passed to the `torch.compile()` function.

    Args:
    - mode (str, optional): The mode to return the optimizations for. If None, optimizations
    for all modes are returned.

    Returns:
    - A dictionary describing the optimizations performed by the specified mode(s).
    If `mode` is provided, the dictionary contains the optimizations specific to that mode.
    If `mode` is None, the dictionary contains optimizations for all available modes.

    Example:
    To retrieve the optimizations performed by each available mode passed to `torch.compile()`,
    we can call the `list_mode_options()` function as follows:

    ::
      options = torch.compiler.list_mode_options()

    To retrieve the optimizations for a specific mode, we can pass the mode name as an argument:

    ::
        options = torch.compiler.list_mode_options(mode="max-autotune")

    In this example, `options` will contain the optimizations specific to the "max-autotune" mode.

    """

    return torch._inductor.list_mode_options(mode)
