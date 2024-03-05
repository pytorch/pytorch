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
        >>>        ...logic that is not needed in a compiled/traced graph...
        >>>
        >>>     ...rest of the function...
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
        >>>        ...logic that is not needed in a TorchDynamo-traced graph...
        >>>
        >>>     ...rest of the function...
    """
    return False
