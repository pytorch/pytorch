"""Test if simple wrapper around autograd.Function is compilable."""
import torch
import torch._dynamo as dynamo
from contextvars import ContextVar
from torch.utils._pytree import tree_map

_in_autograd_function: ContextVar[bool] = ContextVar("in_autograd_function", default=False)


class MyFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * 2

    @staticmethod
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        return grad * 2


class MyFuncWithSetupContext(torch.autograd.Function):
    """autograd.Function with setup_context (required for functorch)."""
    @staticmethod
    def forward(x):
        return x * 2

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, = inputs
        ctx.save_for_backward(x)

    @staticmethod
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        return grad * 2


def simple_wrapper(autograd_function, *args, **kwargs):
    """Simple wrapper that sets context before/after."""
    token = _in_autograd_function.set(True)
    try:
        result = autograd_function.apply(*args, **kwargs)
    finally:
        _in_autograd_function.reset(token)
    return result


def test_direct_apply():
    """Test direct autograd.Function.apply()"""
    print("=" * 60)
    print("Test 1: Direct autograd.Function.apply()")
    print("=" * 60)

    def fn(x):
        return MyFunc.apply(x)

    x = torch.randn(10, requires_grad=True)

    # Eager
    out_eager = fn(x)
    print(f"Eager result: {out_eager.sum().item():.4f}")

    # Compiled
    fn_compiled = torch.compile(fn, backend="eager")
    out_compiled = fn_compiled(x)
    print(f"Compiled result: {out_compiled.sum().item():.4f}")

    # Check graph breaks
    explanation = dynamo.explain(fn)(x)
    print(f"Graph break count: {explanation.graph_break_count}")
    if explanation.graph_break_count > 0:
        print(f"Break reasons: {explanation.break_reasons}")
    print()


def test_simple_wrapper():
    """Test simple wrapper around autograd.Function"""
    print("=" * 60)
    print("Test 2: Simple wrapper around autograd.Function")
    print("=" * 60)

    def fn(x):
        return simple_wrapper(MyFunc, x)

    x = torch.randn(10, requires_grad=True)

    # Eager
    out_eager = fn(x)
    print(f"Eager result: {out_eager.sum().item():.4f}")

    # Compiled
    fn_compiled = torch.compile(fn, backend="eager")
    out_compiled = fn_compiled(x)
    print(f"Compiled result: {out_compiled.sum().item():.4f}")

    # Check graph breaks
    explanation = dynamo.explain(fn)(x)
    print(f"Graph break count: {explanation.graph_break_count}")
    if explanation.graph_break_count > 0:
        print(f"Break reasons: {explanation.break_reasons}")
    print()


def test_wrapper_with_setup_context():
    """Test wrapper with autograd.Function that has setup_context"""
    print("=" * 60)
    print("Test 3: Wrapper with setup_context autograd.Function")
    print("=" * 60)

    def fn(x):
        return simple_wrapper(MyFuncWithSetupContext, x)

    x = torch.randn(10, requires_grad=True)

    # Eager
    out_eager = fn(x)
    print(f"Eager result: {out_eager.sum().item():.4f}")

    # Compiled
    fn_compiled = torch.compile(fn, backend="eager")
    out_compiled = fn_compiled(x)
    print(f"Compiled result: {out_compiled.sum().item():.4f}")

    # Check graph breaks
    explanation = dynamo.explain(fn)(x)
    print(f"Graph break count: {explanation.graph_break_count}")
    if explanation.graph_break_count > 0:
        print(f"Break reasons: {explanation.break_reasons}")
    print()


def test_wrapper_checks_context():
    """Test that wrapper context is visible inside forward"""
    print("=" * 60)
    print("Test 4: Context visibility inside forward")
    print("=" * 60)

    class CheckContextFunc(torch.autograd.Function):
        @staticmethod
        def forward(x):
            in_autograd = _in_autograd_function.get()
            print(f"  Inside forward, _in_autograd_function = {in_autograd}")
            return x * (2 if in_autograd else 1)

        @staticmethod
        def setup_context(ctx, inputs, output):
            pass

        @staticmethod
        def backward(ctx, grad):
            return grad

    def fn(x):
        return simple_wrapper(CheckContextFunc, x)

    x = torch.randn(10)

    print("Eager:")
    out_eager = fn(x)

    print("Compiled:")
    fn_compiled = torch.compile(fn, backend="eager")
    out_compiled = fn_compiled(x)

    print(f"Results match: {torch.allclose(out_eager, out_compiled)}")
    print()


if __name__ == "__main__":
    dynamo.reset()
    test_direct_apply()

    dynamo.reset()
    test_simple_wrapper()

    dynamo.reset()
    test_wrapper_with_setup_context()

    dynamo.reset()
    test_wrapper_checks_context()
