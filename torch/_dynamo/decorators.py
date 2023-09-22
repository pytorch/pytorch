from typing import TYPE_CHECKING

import torch
from . import allowed_functions
from .eval_frame import DisableContext, innermost_fn, RunOnlyContext
from .exc import IncorrectUsage

if TYPE_CHECKING:
    from torch._C._dynamo.eval_frame import (  # noqa: F401
        reset_code,
        set_eval_frame,
        set_guard_error_hook,
        set_guard_fail_hook,
        skip_code,
        unsupported,
    )
else:
    for name in dir(torch._C._dynamo.eval_frame):
        if name.startswith("__"):
            continue
        globals()[name] = getattr(torch._C._dynamo.eval_frame, name)


def run(fn=None):
    """Don't do any dynamic compiles, just use prior optimizations"""
    if fn is not None:
        fn = innermost_fn(fn)
        assert callable(fn)
        return RunOnlyContext()(fn)
    return RunOnlyContext()


def disable(fn=None, recursive=True):
    """
    Decorator and context manager to disable TorchDynamo

    If recursive=True, Dynamo is completely skipped on the decorated function
    frame as well as the recursively invoked functions.

    If recursive=False, Dynamo skips frames associated with the function code,
    but still process recursively invoked frames.
    """
    if recursive:
        if fn is not None:
            fn = innermost_fn(fn)
            assert callable(fn)
            return DisableContext()(fn)
        return DisableContext()
    else:
        return skip(fn)


def skip(fn=None):
    """
    Skip frames associated with the function code, but still process recursively
    invoked frames
    """
    if fn is None:
        return skip
    fn = innermost_fn(fn)
    assert callable(fn)
    skip_code(fn.__code__)
    fn._torchdynamo_disable = True
    return fn


def assume_constant_result(fn):
    fn._dynamo_marked_constant = True
    return fn


def allow_in_graph(fn):
    """
    Customize which functions TorchDynamo will include in the generated
    graph. Similar to `torch.fx.wrap()`.
    ::

        torch._dynamo.allow_in_graph(my_custom_function)

        @torch._dynamo.optimize(...)
        def fn(a):
            x = torch.add(x, 1)
            x = my_custom_function(x)
            x = torch.add(x, 1)
            return x

        fn(...)

    Will capture a single graph containing `my_custom_function()`.
    """
    if isinstance(fn, (list, tuple)):
        return [allow_in_graph(x) for x in fn]
    assert callable(fn), "allow_in_graph expects a callable"
    allowed_functions._allowed_function_ids.add(id(fn))
    allowed_functions._disallowed_function_ids.remove(id(fn))
    allowed_functions._allowed_user_defined_function_ids.add(id(fn))
    return fn


def _disallow_in_graph_helper(throw_if_not_allowed):
    def inner(fn):
        if isinstance(fn, (list, tuple)):
            return [disallow_in_graph(x) for x in fn]
        assert callable(fn), "disallow_in_graph expects a callable"
        if throw_if_not_allowed and not allowed_functions.is_allowed(fn):
            raise IncorrectUsage(
                "disallow_in_graph is expected to be used on an already allowed callable (like torch.* ops). "
                "Allowed callables means callables that TorchDynamo puts as-is in the extracted graph."
            )
        allowed_functions._allowed_function_ids.remove(id(fn))
        allowed_functions._disallowed_function_ids.add(id(fn))
        allowed_functions._allowed_user_defined_function_ids.remove(id(fn))
        return fn

    return inner


def disallow_in_graph(fn):
    """
    Customize which functions TorchDynamo will exclude in the generated
    graph and force a graph break on.
    ::

        torch._dynamo.disallow_in_graph(torch.sub)

        @torch._dynamo.optimize(...)
        def fn(a):
            x = torch.add(x, 1)
            x = torch.sub(x, 1)
            x = torch.add(x, 1)
            return x

        fn(...)

    Will break the graph on `torch.sub`, and give two graphs each with a
    single `torch.add()` op.
    """
    return _disallow_in_graph_helper(throw_if_not_allowed=True)(fn)


@_disallow_in_graph_helper(throw_if_not_allowed=False)
def graph_break():
    """Force a graph break"""
    pass


def forbid_in_graph(fn):
    """
    Customize which functions TorchDynamo will assert are not present while tracing.

    If you want a graph break on this function instead, use disallow_in_graph.
    TODO(voz): We now have allow_in_graph, disallow_in_graph, forbid_in_graph - some more robust
    documentation would not be amiss.
    """
    if isinstance(fn, (list, tuple)):
        return [forbid_in_graph(x) for x in fn]
    assert callable(fn), "forbid_in_graph applies only to callables"
    fn._dynamo_forbidden = True
    return fn


@forbid_in_graph
def mark_dynamic(t, index):
    """
    Mark a tensor as having a dynamic dim.

    [Note - on the state of mark_dynamic]

    The behavior of having a dynamic dimension on a tensor is governed by a few factors:

    1) torch._dynamo.config dynamic_shapes True or False.
        a) dynamic_shapes=True - dynamic_shapes must be True for mark_dynamic to work.
        a) dynamic_shapes=False - This config will raise an exception when used in conjunction with
        mark_dyamic. We will eventually support this.

    2) If the dimension is fully constrained - as in, it does not allow more than a single value
    in both eager (torch.compile, torch._dynamo.optimize) mode and export mode (torch._dynamo.export),
    we will raise an error

    3) If the dimension is partially constrained - allowing at least 2 values but not the full unbounded
    range of shapes, in eager we will pass it through, but export will raise an error.

    4) Attempts to trace this function will explicitly raise. As such, all calls to mark_dynamic must be made
    before torch.compile.

    """
    if isinstance(index, int):
        if not hasattr(t, "_dynamo_dynamic_indices"):
            t._dynamo_dynamic_indices = set()
        # TODO(voz): Should we bounds check?
        t._dynamo_dynamic_indices.add(index)
        return

    assert isinstance(index, (list, tuple))
    for i in index:
        mark_dynamic(t, i)


@forbid_in_graph
def maybe_mark_dynamic(t, index):
    """
    Mark a tensor as having a dynamic dim, but don't enforce it (i.e., if this
    dimension ends up getting specialized, don't error).
    """
    if isinstance(index, int):
        if not hasattr(t, "_dynamo_weak_dynamic_indices"):
            t._dynamo_weak_dynamic_indices = set()
        # TODO(voz): Should we bounds check?
        t._dynamo_weak_dynamic_indices.add(index)
        return

    assert isinstance(index, (list, tuple))
    for i in index:
        maybe_mark_dynamic(t, i)


@forbid_in_graph
def mark_static(t, index=None):
    """
    Mark a tensor as having a static dim.

    This will prevent us from attempting to compile it dynamically
    when dynamic=True; this can improve trace-time performance.

    This has lower precedence than mark_dynamic.
    """
    if isinstance(index, int):
        if not hasattr(t, "_dynamo_static_indices"):
            t._dynamo_static_indices = set()
        # TODO(voz): Should we bounds check?
        t._dynamo_static_indices.add(index)
    elif index is None:
        for i in range(t.dim()):
            mark_static(t, i)
    else:
        assert isinstance(index, (list, tuple))
        for i in index:
            mark_static(t, i)


@forbid_in_graph
def mark_static_address(t, guard=True):
    """
    Marks an input tensor whose data_ptr will not change across multiple calls
    to a dynamo-compiled function. This indicates to cudagraphs that an extra allocation
    is not needed for this input. The data_ptr will be guarded if guard=True. Note:
    Tensors marked in this way will be kept alive until `torch._dynamo.reset()` is called.
    """
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"mark_static_address expects a tensor but recieved {type(t)}")

    if guard:
        t._dynamo_static_input_type = "guarded"
    else:
        t._dynamo_static_input_type = "unguarded"


# Note: it's preferable to not make `import torch` eagerly import other libs.
# However, we want to provide a grace period to make borderline versions of einops
# compatible with torch.compile.
# TODO: we should delete this whole _allow_in_graph_einops logic by approximately 2024 Q2
def _allow_in_graph_einops():
    try:
        import einops

        try:
            # requires einops > 0.6.1, torch >= 2.0
            from einops._torch_specific import (  # noqa: F401
                _ops_were_registered_in_torchdynamo,
            )

            # einops > 0.6.1 will call the op registration logic as it is imported.
            pass
        except ImportError:
            # einops <= 0.6.1
            allow_in_graph(einops.rearrange)
            allow_in_graph(einops.reduce)
            if hasattr(einops, "repeat"):
                allow_in_graph(einops.repeat)  # available since einops 0.2.0
            if hasattr(einops, "einsum"):
                allow_in_graph(einops.einsum)  # available since einops 0.5.0
            if hasattr(einops, "pack"):
                allow_in_graph(einops.pack)  # available since einops 0.6.0
            if hasattr(einops, "unpack"):
                allow_in_graph(einops.unpack)  # available since einops 0.6.0
    except ImportError:
        pass


_allow_in_graph_einops()
