# mypy: allow-untyped-defs
import torch
from typing import List

__all__ = [
    "compile",
    "assume_constant_result",
    "reset",
    "allow_in_graph",
    "list_backends",
    "disable",
    "cudagraph_mark_step_begin",
    "wrap_numpy",
    "is_compiling",
    "is_dynamo_compiling",
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
    Tells the compiler frontend (Dynamo) to skip symbolic introspection of the function
    and instead directly write it to the graph when encountered.

    If you are using :func:`torch.compile` (with backend="inductor" (the default)), or
    :func:`torch.export.export`, and trying to black-box a Python function throughout
    all tracing, do not use this API.
    Instead, please create a custom operator (see :ref:`custom-ops-landing-page`)

    .. warning::

        If you're a typical torch.compile user (e.g. you're applying torch.compile to
        a model to make it run faster), you probably don't want to use this function.
        :func:`allow_in_graph` is a footgun because it skips the compiler frontend
        (Dynamo) that is responsible for doing safety checks (graph breaks, handling
        closures, etc). Incorrect usage will lead to difficult-to-debug silent
        incorrectness issues.

    Given a Python function with no allow_in_graph decorator, regular execution
    of torch.compile traces through the function. :func:`allow_in_graph` changes
    it so that the frontend does not trace inside the function, but the compiler
    backend still traces through it. Compare this to custom operators, which
    treats a function as a black box throughout the torch.compile stack. The following
    table compares these mechanisms.

    +------------------------+-----------------------+--------------------------------+
    | Mechanism              | Frontend (Dynamo)     | Backend (AOTAutograd+Inductor) |
    +========================+=======================+================================+
    | no decorator           | trace inside          | trace inside                   |
    +------------------------+-----------------------+--------------------------------+
    | allow_in_graph         | opaque callable       | trace inside                   |
    +------------------------+-----------------------+--------------------------------+
    | custom op              | opaque callable       | opaque callable                |
    +------------------------+-----------------------+--------------------------------+

    One common use case for :func:`allow_in_graph()` is as an escape hatch for the compiler
    frontend: if you know the function works w.r.t. to the downstream components of the
    compilation stack (AOTAutograd and Inductor) but there is a Dynamo bug that prevents it from
    symbolically introspecting the function properly (or if your code is in C/C++ and
    therefore cannot be introspected with Dynamo), then one can decorate said function
    with :func:`allow_in_graph` to bypass Dynamo.

    We require that ``fn`` adhere to the following restrictions. Failure to adhere
    results in undefined behavior:

    - The inputs to ``fn`` must be Proxy-able types in the FX graph. Valid types include:
      Tensor/int/bool/float/None/List[Tensor?]/List[int?]/List[float?]
      Tuple[Tensor?, ...]/Tuple[int?, ...]/Tuple[float?, ...]/torch.dtype/torch.device
    - The outputs to ``fn`` must be Proxy-able types in the FX graph (see previous bullet)
    - all Tensors used inside of ``fn`` must be passed directly as inputs to ``fn``
      (as opposed to being captured variables).

    Args:
        fn: A callable representing the function to be included in the graph.
            If ``fn`` is a list or tuple of callables it recursively applies
            :func:`allow_in_graph()` to each function and returns a new list or
            tuple containing the modified functions.

    Example::

        torch.compiler.allow_in_graph(my_custom_function)

        @torch.compile(...)
        def fn(a):
            x = torch.add(x, 1)
            x = my_custom_function(x)
            x = torch.add(x, 1)
            return x

        fn(...)

    Will capture a single graph containing ``my_custom_function()``.

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

def cudagraph_mark_step_begin():
    """
    Indicates that a new iteration of inference or training is about to begin.

    CUDA Graphs will free tensors of a prior iteration. A new iteration is started on each invocation of
    torch.compile, so long as there is not a pending backward that has not been called.

    If that heuristic is wrong, such as in the following example, manually mark it with this api.

    .. code-block:: python

        @torch.compile(mode="reduce-overhead")
        def rand_foo():
            return torch.rand([4], device="cuda")

        for _ in range(5):
            torch.compiler.cudagraph_mark_step_begin()
            rand_foo() + rand_foo()

    For more details, see `torch.compiler_cudagraph_trees <https://pytorch.org/docs/main/torch.compiler_cudagraph_trees.html>`__
    """
    from torch._inductor import cudagraph_trees

    cudagraph_trees.mark_step_begin()

def wrap_numpy(fn):
    r"""Decorator that turns a function from ``np.ndarray``s to ``np.ndarray``s into a function
    from ``torch.Tensor``s to ``torch.Tensor``s.

    It is designed to be used with :func:`torch.compile` with ``fullgraph=True``. It allows to
    compile a NumPy function as if it were a PyTorch function. This allows you to run NumPy code
    on CUDA or compute its gradients.

    .. note::

        This decorator does not work without :func:`torch.compile`.

    Example::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> # Compile a NumPy function as a Tensor -> Tensor function
        >>> @torch.compile(fullgraph=True)
        >>> @torch.compiler.wrap_numpy
        >>> def fn(a: np.ndarray):
        >>>     return np.sum(a * a)
        >>> # Execute the NumPy function using Tensors on CUDA and compute the gradients
        >>> x = torch.arange(6, dtype=torch.float32, device="cuda", requires_grad=True)
        >>> out = fn(x)
        >>> out.backward()
        >>> print(x.grad)
        tensor([ 0.,  2.,  4.,  6.,  8., 10.], device='cuda:0')
    """
    from torch._dynamo.external_utils import wrap_numpy as wrap
    return wrap(fn)

_is_compiling_flag: bool = False

def is_compiling() -> bool:
    """
    Indicates whether a graph is executed/traced as part of torch.compile() or torch.export().

    Note that there are 2 other related flags that should deprecated eventually:
      * torch._dynamo.external_utils.is_compiling()
      * torch._utils.is_compiling()

    Example::

        >>> def forward(self, x):
        >>>     if not torch.compiler.is_compiling():
        >>>        pass # ...logic that is not needed in a compiled/traced graph...
        >>>
        >>>     # ...rest of the function...
    """
    if torch.jit.is_scripting():
        return False
    else:
        return _is_compiling_flag

def is_dynamo_compiling() -> bool:
    """
    Indicates whether a graph is traced via TorchDynamo.

    It's stricter than is_compiling() flag, as it would only be set to True when
    TorchDynamo is used.

    Example::

        >>> def forward(self, x):
        >>>     if not torch.compiler.is_dynamo_compiling():
        >>>        pass # ...logic that is not needed in a TorchDynamo-traced graph...
        >>>
        >>>     # ...rest of the function...
    """
    return False
