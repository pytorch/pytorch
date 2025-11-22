# mypy: allow-untyped-defs
import io
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING, TypeVar, Union
from typing_extensions import ParamSpec

import torch
from torch._higher_order_ops.invoke_subgraph import NestedCompileRegionOptions

from . import config


try:
    from typing import LiteralString
except ImportError:
    from typing_extensions import LiteralString

if TYPE_CHECKING:
    from ._cache import CacheInfo


__all__ = [
    "compile",
    "config",
    "assume_constant_result",
    "reset",
    "allow_in_graph",
    "substitute_in_graph",
    "list_backends",
    "disable",
    "set_stance",
    "set_enable_guard_collectives",
    "cudagraph_mark_step_begin",
    "load_compiled_function",
    "wrap_numpy",
    "is_compiling",
    "is_dynamo_compiling",
    "is_exporting",
    "save_cache_artifacts",
    "load_cache_artifacts",
    "skip_guard_on_inbuilt_nn_modules_unsafe",
    "skip_guard_on_all_nn_modules_unsafe",
    "keep_tensor_guards_unsafe",
    "skip_guard_on_globals_unsafe",
    "skip_all_guards_unsafe",
    "nested_compile_region",
]


_P = ParamSpec("_P")
_R = TypeVar("_R")
FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)


def compile(*args, **kwargs):
    """
    See :func:`torch.compile` for details on the arguments for this function.
    """
    # pyrefly: ignore [not-iterable]
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
    Instead, please create a custom operator (see `PyTorch Custom Operators Landing Page
    <https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html>`_)

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
        def fn(x):
            x = torch.add(x, 1)
            x = my_custom_function(x)
            x = torch.add(x, 1)
            return x


        fn(...)

    Will capture a single graph containing ``my_custom_function()``.

    """
    import torch._dynamo

    return torch._dynamo.allow_in_graph(fn)


def substitute_in_graph(
    original_fn: Callable[_P, _R],
    *,
    can_constant_fold_through: bool = False,
    skip_signature_check: bool = False,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """
    Register a polyfill handler for a function, usually a C function from the C extension, to be
    used in place of the original function when inlining the original function in the graph.

    .. note::

        The polyfill handler is only used when inlining the original function. It is not used when
        the original function is called directly. In the eager mode, the decorated function calls
        the performant C function rather than the polyfill handler.

    The polyfill handler is a function that will be called in place of the original function when
    inlining the original function. The polyfill handler should have the same signature and the same
    behavior as the original function.

    Args:
        original_fn (callable): The original function, usually a C function, to register a polyfill
            handler for.
        can_constant_fold_through (bool, optional): Whether the polyfill handler can be constant
            folded through. That is, if the polyfill handler is a pure function and its arguments
            are constant, the result of the polyfill handler can be constant folded during the
            compilation. Defaults to ``False``.
        skip_signature_check (bool, optional): Whether to skip the signature check between the
            original function and the polyfill handler. Defaults to ``False``.

    Returns:
        A decorator that registers the polyfill handler for the original function.

    Example::

        >>> import operator
        >>> operator.indexOf([1, 2, 3, 4, 5], 3)
        2
        >>> torch.compile(operator.indexOf, fullgraph=True)([1, 2, 3, 4, 5], 3)
        ... # xdoctest: +SKIP("Long tracebacks")
        Traceback (most recent call last):
        ...
        torch._dynamo.exc.Unsupported: ...

        >>> @torch.compiler.substitute_in_graph(operator.indexOf)
        ... def indexOf(a, b, /):
        ...     for i, item in enumerate(a):
        ...         if item is b or item == b:
        ...             return i
        ...     raise ValueError("sequence.index(x): x not in sequence")
        >>>
        >>> torch.compile(operator.indexOf, fullgraph=True)([1, 2, 3, 4, 5], 3)
        2
    """
    import torch._dynamo

    return torch._dynamo.substitute_in_graph(
        original_fn,
        can_constant_fold_through=can_constant_fold_through,
        skip_signature_check=skip_signature_check,
    )


def list_backends(exclude_tags=("debug", "experimental")) -> list[str]:
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
    This allows the compiler to optimize away your function.
    Returns The same function `fn`

    Args:
        fn: The function to be marked as having a constant result.

    .. warning::
        `assume_constant_result` can if invalid cause safety and soundness issues, :func:`torch.compile`
        will not attempt to validate whether the constant assumption is true or not

    """
    import torch._dynamo

    return torch._dynamo.assume_constant_result(fn)


def disable(fn=None, recursive=True, *, reason=None):
    """
    This function provides a decorator to disable compilation on a function.
    It also provides the option of recursively disabling called functions.

    Args:
        fn (optional): The function to disable
        recursive (optional): A boolean value indicating whether the disabling should be recursive.
        reason (optional): A string value indicating the reason for disabling the function.
    """
    import torch._dynamo

    return torch._dynamo.disable(fn, recursive, reason=reason)


def set_stance(
    stance: str = "default",
    *,
    skip_guard_eval_unsafe: bool = False,
    force_backend: Union[str, Callable[..., Any], None] = None,
):
    """
    Set the current stance of the compiler.
    Can be used as a function, context manager, or decorator.
    Do not use this function inside a `torch.compile` region - an error will be raised otherwise.

    .. code-block:: python

        @torch.compile
        def foo(x): ...


        @torch.compiler.set_stance("force_eager")
        def bar():
            # will not be compiled
            foo(...)


        bar()

        with torch.compiler.set_stance("force_eager"):
            # will also not be compiled
            foo(...)

        torch.compiler.set_stance("force_eager")
        # will also not be compiled
        foo(...)
        torch.compiler.set_stance("default")

        # will be compiled
        foo(...)

    Args:
        stance: The stance to set the compiler to. Valid values are:

            - "default": The default stance, used for normal compilation.
            - "force_eager": Ignore all `torch.compile` directives.
            - "eager_on_recompile": Run code eagerly when a recompile is necessary.
              If there is cached compiled code valid for the input, it will still be used.
            - "fail_on_recompile": Raise an error when recompiling a function.
            - "eager_then_compile": Run the first invocation in eager mode, then compile on
              subsequent calls. This is beneficial for dynamic shapes as it allows inferring
              dynamism from the first two invocations instead of wasting a static compile on
              the first invocation.
            - "aot_eager_then_compile": Run the first invocation with AOT eager to get memory
              benefits from activation checkpointing, then compile on subsequent calls. Like
              eager_then_compile, this improves handling of dynamic shapes by avoiding an
              initial static compile.


        skip_guard_eval_unsafe: A flag to run only differentiating guards.
            CAUTION - This flag is unsafe and should only be used if your setup
            meets the following conditions.

            torch.compile uses a guard system to support recompilations and
            choose which compiled artifact to run at runtime.  These guards,
            though efficient, add some overhead, which may impact performance in
            scenarios where you need to optimize for minimal guard processing
            time.  This API enables you to disable guard evaluation, assuming
            that you have warmed up the compiled model with a sufficient variety
            of inputs. This assumption means that, after the warmup phase, no
            further recompilations will be necessary.  If this assumption fails,
            there is a risk of silently producing incorrect results (hence the
            term "unsafe" in the API name).

        force_backend: If `stance` is "default", this argument can be used to force `torch.compile`
            to use a specific backend. Otherwise, an error is raised.
    """
    import torch._dynamo

    return torch._dynamo.set_stance(
        stance,
        skip_guard_eval_unsafe=skip_guard_eval_unsafe,
        force_backend=force_backend,
    )


# forbid in graph
set_stance._dynamo_forbidden = True  # type: ignore[attr-defined]


def set_enable_guard_collectives(enabled: bool):
    """
    Enables use of collectives *during* guard evaluation to synchronize behavior
    across ranks.  This is expensive: we have to issue a collective every time
    we enter a compiled code region, even if no rank actually would need to
    compile.  This can help prevent NCCL hangs by ensuring that we never have a
    situation where one rank starts recompiling while other ranks don't compile;
    it is especially useful in conjunction with enable_compiler_collectives
    where such a situation would immediately cause a hang (as it is necessary
    for all ranks to compile at the same time to run compiler collectives).  Like
    compiler collectives, you can only run this on SPMD programs; you will hang
    otherwise.  Note that a guard collective is only issued if there is any
    compiled code to guard on; if this the first time we encounter a frame or
    the frame is skipped, we don't issue collectives.

    Returns the previous setting of enabled.
    """
    from torch._C._dynamo.eval_frame import set_guard_complete_hook  # noqa: F401
    from torch._dynamo.eval_frame import guard_collectives_hook

    if enabled:
        return set_guard_complete_hook(guard_collectives_hook) is not None  # type: ignore[arg-type]
    else:
        return set_guard_complete_hook(None) is not None


set_enable_guard_collectives._dynamo_forbidden = True  # type: ignore[attr-defined]


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
_is_exporting_flag: bool = False


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


def is_exporting() -> bool:
    """
    Indicated whether we're under exporting.

    It's stricter than is_compiling() flag, as it would only be set to True when
    torch.export is used.

    Example::

        >>> def forward(self, x):
        >>>     if not torch.compiler.is_exporting():
        >>>        pass # ...logic that is not needed in export...
        >>>
        >>>     # ...rest of the function...
    """
    return _is_exporting_flag


def save_cache_artifacts() -> Optional[tuple[bytes, "CacheInfo"]]:
    """
    Serializes all the cache artifacts that were created during the compilation

    Example:

    - Execute torch.compile
    - Call torch.compiler.save_cache_artifacts()
    """
    from ._cache import CacheArtifactManager

    if torch._dynamo.config.caching_precompile:
        from torch._dynamo.precompile_context import PrecompileContext

        PrecompileContext.save_to_dynamo_cache()

    return CacheArtifactManager.serialize()


def load_cache_artifacts(serialized_artifacts: bytes) -> Optional["CacheInfo"]:
    """
    Hot loads cache artifacts that were previously serialized via
    save_cache_artifacts

    Example:

    # From a previous invocation
    artifacts = torch.compiler.save_cache_artifacts()

    torch.compiler.load_cache_artifacts(artifacts[0])
    """
    from ._cache import CacheArtifactManager, CacheInfo

    artifacts = CacheArtifactManager.deserialize(serialized_artifacts)
    if artifacts is not None:
        return CacheArtifactManager.populate_caches(artifacts)
    return None


def skip_guard_on_inbuilt_nn_modules_unsafe(guard_entries):
    """
    A common function to skip guards on the inbuilt nn modules like
    torch.nn.Linear. This is unsafe to use by default. But for majority of
    torch.compile users, the model code does not modify the inbuilt nn module
    attributes. They can benefit from reduction in guard latency overhead using
    this API.

    To use this API, use guard_filter_fn argument while calling torch.compile

    >> opt_mod = torch.compile(
    >>     mod,
    >>     options={"guard_filter_fn": torch.compiler.skip_guard_on_all_nn_modules_unsafe},
    >> )
    """
    return [
        not entry.orig_guard.source.is_unspecialized_builtin_nn_module()
        for entry in guard_entries
    ]


def skip_guard_on_all_nn_modules_unsafe(guard_entries):
    """
    A common function to skip guards on all nn modules, both user defined as
    well inbuilt nn modules (like torch.nn.Linear). This is unsafe to use by
    default. But for majority of torch.compile users, the model code does not
    modify the nn module attributes. They can benefit from reduction in guard
    latency overhead using this API.

    To use this API, use guard_filter_fn argument while calling torch.compile

    >> opt_mod = torch.compile(
    >>     mod,
    >>     options={"guard_filter_fn": torch.compiler.skip_guard_on_all_nn_modules_unsafe},
    >> )
    """

    return [
        not entry.orig_guard.source.is_unspecialized_nn_module()
        for entry in guard_entries
    ]


def keep_tensor_guards_unsafe(guard_entries, keep_parameters=False):
    """
    A common function to keep tensor guards on all tensors. This is unsafe to
    use by default. But if you don't expect any changes in the model code, you
    can just keep the tensor guards.


    >> opt_mod = torch.compile(
    >>     mod,
    >>     options={"guard_filter_fn": torch.compiler.keep_tensor_guards},
    >> )
    """

    keep_flags = []
    for entry in guard_entries:
        if entry.guard_type == "TENSOR_MATCH":
            if not isinstance(entry.value, torch.nn.Parameter):
                keep_flags.append(True)
            elif keep_parameters:
                keep_flags.append(True)
            else:
                keep_flags.append(False)
        else:
            keep_flags.append(False)
    return keep_flags


def skip_guard_on_globals_unsafe(guard_entries):
    """
    A common function to skip guards on all globals. This is unsafe to use by
    default. But if you don't expect any changes in the globals, you can just
    keep the tensor guards.

    >> opt_mod = torch.compile(
    >>     mod,
    >>     options={"guard_filter_fn": torch.compiler.skip_guard_on_globals},
    >> )
    """

    return [not entry.is_global for entry in guard_entries]


def skip_all_guards_unsafe(guard_entries):
    """
    A function for skipping all guards on a compiled function.

    WARNING: This function will drop all the safety guarantees from Dynamo
             compiled function. Use this with caution.

    To use this API, use guard_filter_fn argument while calling torch.compile

    >> opt_mod = torch.compile(
    >>     mod,
    >>     options={"guard_filter_fn": torch.compiler.skip_all_guards_unsafe},
    >> )
    """
    return [False for entry in guard_entries]


def nested_compile_region(
    fn=None, backend_options: Optional[NestedCompileRegionOptions] = None
):
    """
    Tells **``torch.compile``** that the marked set of operations forms a nested
    compile region (which is often repeated in the full model) whose code can be
    compiled once and safely reused.  ``nested_compile_region`` can also be used
    as a decorator.

    During **``torch.compile``** tracing, the compiler applies *hierarchical
    compilation* with ``nested_compile_region``: it emits optimized code for the
    marked region the first time it is encountered and re-emits (or "stamps
    out") the previously compiled code on every subsequent invocation.  This can
    substantially reduce overall compile time for deeply-stacked,
    structurally-identical components such as the transformer layers of a
    large-language-model (LLM).

    Outside a ``torch.compile`` context—i.e., in standard eager execution—the
    call is a no-op, so existing workflows remain unaffected.

    Note that ``nested_compile_region`` **does not** promise that a region will
    be compiled exactly once.  If the compiler detects that new input conditions
    (shape, dtype, device, stride, globals etc.) make the cached version invalid
    to reuse, it will transparently re-compile the region.  Using it is
    therefore *safe*: correctness is always preserved, and you pay the extra
    compilation cost only when required.

    Args:
        fn: The function to wrap
        backend: Optional backend to use for compiling the subgraph.
    """

    from torch._higher_order_ops.invoke_subgraph import (
        mark_compile_region as _mark_compile_region,
    )

    return _mark_compile_region(fn, backend_options=backend_options)


def load_compiled_function(file: io.IOBase) -> Callable[..., Any]:
    """
    Load an aot-compiled function from a file.

    .. warning::

        This API is currently experimental and subject to change.

    Args:
        file: A file-like object containing the serialized compiled function.

    Returns:
        A torch-compiled function with compilation preloaded from disk.
    """
    from torch._dynamo.aot_compile import AOTCompiledFunction

    data = file.read()
    return AOTCompiledFunction.deserialize(data)
