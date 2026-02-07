from collections.abc import Callable
from typing import Optional

import torch
import torch.utils._pytree as pytree
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode


# Global registry for callback functions
_CALLBACK_REGISTRY: dict[int, Callable[[torch.Tensor], None]] = {}
_NEXT_CALLBACK_ID = 0


def _register_callback(fn: Callable[[torch.Tensor], None]) -> int:
    global _NEXT_CALLBACK_ID
    callback_id = _NEXT_CALLBACK_ID
    _NEXT_CALLBACK_ID += 1
    _CALLBACK_REGISTRY[callback_id] = fn
    return callback_id


def _get_callback(callback_id: int) -> Optional[Callable[[torch.Tensor], None]]:
    return _CALLBACK_REGISTRY.get(callback_id)


class CompilePrintFwd(HigherOrderOperator):
    """
    compile_print_fwd(fwd_callback_id, bwd_callback_id, *tensors) -> None

    Forward pass HOP that calls the forward callback on each tensor.
    Returns None (pure side-effect operation).
    """

    def __init__(self) -> None:
        super().__init__("compile_print_fwd")

    def __call__(
        self,
        fwd_callback_id: int,
        bwd_callback_id: int,
        *tensors: torch.Tensor,
    ) -> None:
        # pyrefly: ignore [missing-attribute]
        return super().__call__(fwd_callback_id, bwd_callback_id, *tensors)

    # pyrefly: ignore [bad-override]
    def gen_schema(
        self,
        fwd_callback_id: int,
        bwd_callback_id: int,
        *tensors: torch.Tensor,
    ) -> torch.FunctionSchema:
        from torch._higher_order_ops.schema import HopSchemaGenerator

        schema_gen = HopSchemaGenerator(self)
        schema_gen.add_arg("fwd_callback_id", fwd_callback_id)
        schema_gen.add_arg("bwd_callback_id", bwd_callback_id)
        for i, t in enumerate(tensors):
            schema_gen.add_arg(f"tensor{i}", t)
        # No outputs - pure side effect
        return schema_gen.gen_schema()


compile_print_fwd = CompilePrintFwd()


class CompilePrintBwd(HigherOrderOperator):
    """
    compile_print_bwd(bwd_callback_id, *grads) -> None

    Backward pass HOP that calls the backward callback on each gradient.
    Returns None (pure side-effect operation).
    """

    def __init__(self) -> None:
        super().__init__("compile_print_bwd")

    def __call__(self, bwd_callback_id: int, *grads: torch.Tensor) -> None:
        # pyrefly: ignore [missing-attribute]
        return super().__call__(bwd_callback_id, *grads)

    # pyrefly: ignore [bad-override]
    def gen_schema(
        self, bwd_callback_id: int, *grads: torch.Tensor
    ) -> torch.FunctionSchema:
        from torch._higher_order_ops.schema import HopSchemaGenerator

        schema_gen = HopSchemaGenerator(self)
        schema_gen.add_arg("bwd_callback_id", bwd_callback_id)
        for i, g in enumerate(grads):
            schema_gen.add_arg(f"grad{i}", g)
        # No outputs - pure side effect
        return schema_gen.gen_schema()


compile_print_bwd = CompilePrintBwd()


# Forward HOP implementations


@compile_print_fwd.py_impl(ProxyTorchDispatchMode)
def compile_print_fwd_proxy(
    mode: ProxyTorchDispatchMode,
    fwd_callback_id: int,
    bwd_callback_id: int,
    *tensors: torch.Tensor,
) -> None:
    proxy_args = (fwd_callback_id, bwd_callback_id) + pytree.tree_map(
        mode.tracer.unwrap_proxy,  # pyrefly: ignore [missing-attribute]
        tensors,
    )
    mode.tracer.create_proxy("call_function", compile_print_fwd, proxy_args, {})
    return None


@compile_print_fwd.py_impl(FakeTensorMode)
def compile_print_fwd_fake(
    mode: FakeTensorMode,
    fwd_callback_id: int,
    bwd_callback_id: int,
    *tensors: torch.Tensor,
) -> None:
    return None


@compile_print_fwd.py_impl(torch._C.DispatchKey.CompositeExplicitAutograd)
def compile_print_fwd_impl(
    fwd_callback_id: int,
    bwd_callback_id: int,
    *tensors: torch.Tensor,
) -> None:
    fwd_fn = _get_callback(fwd_callback_id)
    if fwd_fn is not None:
        for t in tensors:
            if isinstance(t, torch.Tensor):
                fwd_fn(t)
    return None


@compile_print_fwd.py_functionalize_impl
def compile_print_fwd_func(
    ctx,
    fwd_callback_id: int,
    bwd_callback_id: int,
    *tensors: torch.Tensor,
):
    unwrapped_tensors = ctx.unwrap_tensors(tensors)
    with ctx.redispatch_to_next():
        compile_print_fwd(fwd_callback_id, bwd_callback_id, *unwrapped_tensors)
    return None


# Backward HOP implementations


@compile_print_bwd.py_impl(ProxyTorchDispatchMode)
def compile_print_bwd_proxy(
    mode: ProxyTorchDispatchMode, bwd_callback_id: int, *grads: torch.Tensor
) -> None:
    # pyrefly: ignore [missing-attribute]
    proxy_args = (bwd_callback_id,) + pytree.tree_map(mode.tracer.unwrap_proxy, grads)
    mode.tracer.create_proxy("call_function", compile_print_bwd, proxy_args, {})
    return None


@compile_print_bwd.py_impl(FakeTensorMode)
def compile_print_bwd_fake(
    mode: FakeTensorMode, bwd_callback_id: int, *grads: torch.Tensor
) -> None:
    return None


@compile_print_bwd.py_impl(torch._C.DispatchKey.CompositeExplicitAutograd)
def compile_print_bwd_impl(bwd_callback_id: int, *grads: torch.Tensor) -> None:
    bwd_fn = _get_callback(bwd_callback_id)
    if bwd_fn is not None:
        for g in grads:
            if isinstance(g, torch.Tensor):
                bwd_fn(g)
    return None


@compile_print_bwd.py_functionalize_impl
def compile_print_bwd_func(ctx, bwd_callback_id: int, *grads: torch.Tensor):
    unwrapped_grads = ctx.unwrap_tensors(grads)
    with ctx.redispatch_to_next():
        compile_print_bwd(bwd_callback_id, *unwrapped_grads)
    return None


# Autograd fallthrough for backward HOP (it doesn't need gradients)
compile_print_bwd.fallthrough(torch._C.DispatchKey.AutogradCPU)
compile_print_bwd.fallthrough(torch._C.DispatchKey.AutogradCUDA)


# Autograd implementation for forward HOP
class _CompilePrintAutogradOp(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, fwd_callback_id, bwd_callback_id, *tensors):
        with torch._C._AutoDispatchBelowAutograd():
            compile_print_fwd(fwd_callback_id, bwd_callback_id, *tensors)

        # Register hooks on input tensors to call backward callback
        # when they receive gradients (even if our output isn't used).
        # We emit compile_print_bwd calls so they get traced properly.
        for t in tensors:
            if isinstance(t, torch.Tensor) and t.requires_grad:

                def make_hook(callback_id):
                    def hook(grad):
                        # Emit compile_print_bwd call (gets traced in compiled mode)
                        compile_print_bwd(callback_id, grad)
                        return None  # Don't modify the gradient

                    return hook

                t.register_hook(make_hook(bwd_callback_id))

        return None

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        # No outputs, so backward is never called
        return None


@compile_print_fwd.py_autograd_impl
def compile_print_fwd_autograd(
    fwd_callback_id: int,
    bwd_callback_id: int,
    *tensors: torch.Tensor,
):
    return _CompilePrintAutogradOp.apply(fwd_callback_id, bwd_callback_id, *tensors)


def make_compile_print(
    fwd_f: Callable[[torch.Tensor], None],
    bwd_f: Callable[[torch.Tensor], None],
) -> Callable[..., None]:
    """
    Create a compile_print function with the given forward and backward callbacks.

    This factory function creates a properly configured compile_print that can be used
    inside torch.compile regions and with make_fx. The returned function takes tensors,
    calls fwd_f on each tensor in forward and bwd_f on each gradient in backward,
    and returns None (pure side-effect operation).

    Example:
        >>> def print_fwd(t):
        ...     print(f"Forward: shape={t.shape}, mean={t.mean():.4f}")
        >>> def print_bwd(t):
        ...     print(f"Backward: shape={t.shape}, mean={t.mean():.4f}")
        >>> cp = make_compile_print(print_fwd, print_bwd)
        >>> @torch.compile
        ... def fn(x):
        ...     cp(x)
        ...     return x.sum()
        >>> x = torch.randn(3, 3, requires_grad=True)
        >>> fn(x).backward()

    Args:
        fwd_f: Function to call on each tensor in forward pass. Signature: (Tensor) -> None.
        bwd_f: Function to call on each gradient in backward pass. Signature: (Tensor) -> None.

    Returns:
        A function that takes tensors, calls the callbacks, and returns None.
    """
    fwd_callback_id = _register_callback(fwd_f)
    bwd_callback_id = _register_callback(bwd_f)

    def compile_print_impl(*tensors: torch.Tensor) -> None:
        return compile_print_fwd(fwd_callback_id, bwd_callback_id, *tensors)

    return compile_print_impl


def new_compile_print(
    fwd_f: Callable[[torch.Tensor], None],
    bwd_f: Callable[[torch.Tensor], None],
    *args: torch.Tensor,
) -> None:
    """
    Run fwd_f(tensor) on all tensor args in forward, and bwd_f(grad) on all gradients in backward.

    Both fwd_f and bwd_f have signature f(Tensor) -> None.
    This is a pure side-effect operation that returns None.

    Example:
        >>> def print_fwd(t):
        ...     print(f"Forward: shape={t.shape}, mean={t.mean():.4f}")
        >>> def print_bwd(t):
        ...     print(f"Backward: shape={t.shape}, mean={t.mean():.4f}")
        >>> x = torch.randn(3, 3, requires_grad=True)
        >>> new_compile_print(print_fwd, print_bwd, x)
        >>> x.sum().backward()

    Args:
        fwd_f: Function to call on each tensor in forward pass. Signature: (Tensor) -> None.
        bwd_f: Function to call on each gradient in backward pass. Signature: (Tensor) -> None.
        *args: Tensors to observe.

    Returns:
        None.
    """
    fn = make_compile_print(fwd_f, bwd_f)
    return fn(*args)
