# mypy: ignore-errors

import contextlib
import os

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._utils import wrapper_set_seed
import torch.utils._pytree as pytree


MAKE_FX_CPP_FAKE_TENSOR = (
    os.environ.get("MAKE_FX_CPP_FAKE_TENSOR", "0") == "1"
    and hasattr(torch._C, "_is_fake_tensor")
)


@contextlib.contextmanager
def cpp_fake_tensor_mode(shape_env=None):
    from torch._subclasses.fake_tensor import CppFakeTensorMode, FakeTensorConverter
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    if shape_env is None:
        shape_env = ShapeEnv()
    cpp_mode = CppFakeTensorMode.create_cpp_fake_tensor_mode(
        FakeTensorConverter(), shape_env
    )
    with cpp_mode.activated():
        yield shape_env


_cpp_fake_arg_counter = 0


def to_cpp_fake_symbolic(x):
    global _cpp_fake_arg_counter
    if not isinstance(x, torch.Tensor):
        return x
    from torch._dynamo.source import ConstantSource
    from torch.fx.experimental.symbolic_shapes import (
        DimDynamic,
        StatelessSymbolicContext,
    )

    from torch._subclasses.fake_tensor import CppFakeTensorMode

    _cpp_fake_arg_counter += 1
    source = ConstantSource(f"cpp_fake_arg_{_cpp_fake_arg_counter}")
    ctx = StatelessSymbolicContext(dynamic_sizes=[DimDynamic.DYNAMIC] * x.dim())
    return CppFakeTensorMode._get_active_cpp_fake_tensor_mode().from_tensor(
        x, source=source, symbolic_context=ctx
    )


def make_fx_cpp_fake(f, tracing_mode, decomposition_table=None, **kwargs):
    """Like make_fx for fake/symbolic, but traces under tracing_mode="real"
    inside an active C++ FakeTensorMode. Returns a callable, like make_fx."""
    from torch._dispatch.python import enable_python_dispatcher

    symbolic = tracing_mode == "symbolic"

    def wrapped(*args):
        with cpp_fake_tensor_mode():
            if symbolic:
                args = pytree.tree_map_only(torch.Tensor, to_cpp_fake_symbolic, args)
                with enable_python_dispatcher():
                    return make_fx(
                        f,
                        decomposition_table=decomposition_table,
                        tracing_mode="real",
                        **kwargs,
                    )(*args)
            return make_fx(
                f,
                decomposition_table=decomposition_table,
                tracing_mode="real",
                **kwargs,
            )(*args)

    return wrapped


def make_fx_check(
    func,
    args,
    kwargs,
    tracing_mode,
    assert_close=torch.testing.assert_close,
    randomize_data=False,
):
    f, *new_args = handle_sizes_for_dynamic_shapes(func, args, kwargs)

    def run(f, *args, **kwargs):
        return wrapper_set_seed(f, *args, **kwargs)

    if MAKE_FX_CPP_FAKE_TENSOR and tracing_mode in ("fake", "symbolic"):
        traced_f = make_fx_cpp_fake(f, tracing_mode)(*new_args)
    else:
        traced_f = make_fx(f, tracing_mode=tracing_mode)(*new_args)

    msg = (
        "op(*args, **kwargs) and make_fx(op)(*args, **kwargs) produced different "
        "values. This could mean that your abstract impls (meta/FakeTensor impls) "
        "are incorrect, that your operator is not completely traceable (e.g., "
        "it relies on some global state), or that there is a bug in make_fx. "
        "Note that if you passed a python function (and not an operator) to "
        "make_fx_check, it is still possible that the python function will still "
        "work with torch.compile because it handles capturing pieces of "
        "your python code to compile."
    )

    # Randomize the data and run the traced graph with it, to catch bugs
    # where we may have baked in Tensor data into the trace.
    # This is not guaranteed to succeed, because `f` might have preconditions
    # on the values of the inputs, so we just ignore if we used
    # random data and it fails.
    if randomize_data:
        new_args = randomize(new_args)
    try:
        expected = run(f, *new_args)
    except Exception:
        if randomize_data:
            return
        raise
    result = run(traced_f, *new_args)
    assert_close(result, expected, msg=msg)


# Arguably we should make make_fx promote torch.Size() objects to symbolic shapes.
# Absent that, here is our strategy:
#
# If any argument is a torch.Size(), maybe get dynamic shapes for it by:
# - Create a temporary Tensor whose size is the torch.Size() we want. Note that
#   we use an expanded Tensor as we cannot pass "meta" Tensors to make_fx.
# - Pass it to make_fx such that it is converted to a proxy Tensor
# - Unpack the size in the wrapper to get a torch.Size with dynamic shapes (in
#   symbolic mode, a no-op otherwise)
def handle_sizes_for_dynamic_shapes(func, args, kwargs):
    def f(args, kwargs, extra_args, extra_kwargs):
        if extra_args:
            for i, t in extra_args:
                args[i] = t.size()
        if extra_kwargs:
            for k, t in extra_kwargs.items():
                kwargs[k] = t.size()

        return func(*args, **kwargs)

    extra_args = []
    extra_kwargs = {}
    for i, arg in enumerate(args):
        if isinstance(arg, torch.Size):
            extra_args.append((i, torch.empty(arg, device="cpu")))
    for key, value in kwargs.items():
        if isinstance(value, torch.Size):
            extra_kwargs[key] = torch.empty(value, device="cpu")

    return f, args, kwargs, extra_args, extra_kwargs


def randomize(args):
    def transform(x):
        if not x.dtype.is_floating_point:
            return x
        return x.detach().clone().uniform_(0, 1).requires_grad_(x.requires_grad)
    return pytree.tree_map_only(torch.Tensor, transform, args)
