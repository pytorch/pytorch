import functools

import torch.utils._pytree as pytree
from torch import Tensor
from torch._C import DispatchKey
from torch._ops import HigherOrderOperator
from torch._prims_common import clone_preserve_strides
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)


# Used for wrapping a Triton Kernel
class TritonKernelWrapperMutation(HigherOrderOperator):
    def __init__(self):
        super().__init__("triton_kernel_wrapper_mutation")


triton_kernel_wrapper_mutation = TritonKernelWrapperMutation()


# Used for wrapping a Triton Kernel in a functional manner
class TritonKernelWrapperFunctional(HigherOrderOperator):
    def __init__(self):
        super().__init__("triton_kernel_wrapper_functional")


triton_kernel_wrapper_functional = TritonKernelWrapperFunctional()


@triton_kernel_wrapper_mutation.py_impl(DispatchKey.CompositeExplicitAutograd)
def triton_kernel_wrapper_mutation_dense(*, kernel, grid, kwargs):
    kernel[grid](**kwargs)


@triton_kernel_wrapper_mutation.py_impl(FakeTensorMode)
def triton_kernel_wrapper_mutation_fake_tensor_mode(mode, *, kernel, grid, kwargs):
    with mode:
        return None


def prepare_triton_kernel_for_graph_node(f, kernel):
    # This is a hack to workaround how FX does not allow for functions
    # in the graph

    fn = functools.partial(f, kernel=kernel)
    # FX graph needs __name__ and __module__ attributes
    fn.__name__ = f.__name__  # type:ignore[attr-defined]
    if not hasattr(fn, "__module__"):
        # Super hacky but on AMD __module__ is not set
        fn.__module__ = "itertools"
    return fn


def trace_triton_kernel_wrapper(proxy_mode, func_overload, *, kernel, grid, kwargs):
    with disable_proxy_modes_tracing():
        out = func_overload(kernel=kernel, grid=grid, kwargs=kwargs)

    node_args = {"grid": grid, "kwargs": kwargs}
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, node_args)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function",
        prepare_triton_kernel_for_graph_node(func_overload, kernel),
        (),
        proxy_args,
        name=func_overload.__name__ + "_proxy",
    )
    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@triton_kernel_wrapper_mutation.py_impl(ProxyTorchDispatchMode)
def triton_kernel_wrapper_mutation_proxy_torch_dispatch_mode(
    mode, *, kernel, grid, kwargs
):
    if mode.enable_tracing:
        trace_triton_kernel_wrapper(
            mode,
            triton_kernel_wrapper_mutation,
            kernel=kernel,
            grid=grid,
            kwargs=kwargs,
        )
    else:
        triton_kernel_wrapper_mutation(kernel=kernel, grid=grid, kwargs=kwargs)

    return None


@triton_kernel_wrapper_mutation.py_functionalize_impl
def triton_kernel_wrapper_mutation_functionalize(ctx, kernel, grid, kwargs):
    unwrapped_kwargs = ctx.unwrap_tensors(kwargs)
    with ctx.redispatch_to_next():
        unwrapped_outputs = triton_kernel_wrapper_functional(
            kernel=kernel, grid=grid, kwargs=unwrapped_kwargs
        )

    assert unwrapped_outputs.keys() == kwargs.keys()
    for key, output_arg in unwrapped_outputs.items():
        if not isinstance(output_arg, Tensor):
            continue
        input_arg = kwargs[key]
        assert isinstance(input_arg, Tensor)

        ctx.replace(input_arg, output_arg)
        ctx.commit_update(input_arg)
        ctx.sync(input_arg)
    return None


@triton_kernel_wrapper_functional.py_impl(DispatchKey.CompositeExplicitAutograd)
def triton_kernel_wrapper_functional_dense(*, kernel, grid, kwargs):
    # TODO(oulgen): For performance reasons, we want to ensure that these
    # `clone_preserve_strides` calls are never executed at runtime
    # (inductor should always optimize them away).
    # Requires https://github.com/pytorch/pytorch/issues/109240
    kwargs = {
        key: (clone_preserve_strides(val) if isinstance(val, Tensor) else val)
        for key, val in kwargs.items()
    }
    triton_kernel_wrapper_mutation(kernel=kernel, grid=grid, kwargs=kwargs)
    return kwargs


@triton_kernel_wrapper_functional.py_impl(FakeTensorMode)
def triton_kernel_wrapper_functional_fake_tensor_mode(mode, *, kernel, grid, kwargs):
    # TODO(oulgen): For performance reasons, we want to ensure that these
    # `clone_preserve_strides` calls are never executed at runtime
    # (inductor should always optimize them away).
    # Requires https://github.com/pytorch/pytorch/issues/109240
    with mode:
        return {
            key: (clone_preserve_strides(val) if isinstance(val, Tensor) else val)
            for key, val in kwargs.items()
        }


@triton_kernel_wrapper_functional.py_impl(ProxyTorchDispatchMode)
def triton_kernel_wrapper_functional_proxy_torch_dispatch_mode(
    mode, *, kernel, grid, kwargs
):
    if mode.enable_tracing:
        return trace_triton_kernel_wrapper(
            mode,
            triton_kernel_wrapper_functional,
            kernel=kernel,
            grid=grid,
            kwargs=kwargs,
        )
    else:
        return triton_kernel_wrapper_functional(kernel=kernel, grid=grid, kwargs=kwargs)


@triton_kernel_wrapper_functional.py_functionalize_impl
def triton_kernel_wrapper_functional_functionalize(ctx, kernel, grid, kwargs):
    unwrapped_kwargs = ctx.unwrap_tensors(kwargs)
    with ctx.redispatch_to_next():
        outputs = triton_kernel_wrapper_functional(
            kernel=kernel, grid=grid, kwargs=unwrapped_kwargs
        )
        return ctx.wrap_tensors(outputs)


triton_kernel_wrapper_mutation.fallthrough(DispatchKey.PythonDispatcher)  # type: ignore[attr-defined]
triton_kernel_wrapper_mutation.fallthrough(DispatchKey.PythonTLSSnapshot)  # type: ignore[attr-defined]
triton_kernel_wrapper_mutation.fallthrough(DispatchKey.ADInplaceOrView)
triton_kernel_wrapper_mutation.fallthrough(DispatchKey.BackendSelect)
triton_kernel_wrapper_mutation.fallthrough(DispatchKey.AutocastCPU)  # type: ignore[attr-defined]
triton_kernel_wrapper_mutation.fallthrough(DispatchKey.AutocastCUDA)  # type: ignore[attr-defined]
triton_kernel_wrapper_mutation.fallthrough(DispatchKey.AutogradCUDA)
triton_kernel_wrapper_mutation.fallthrough(DispatchKey.AutogradCPU)

triton_kernel_wrapper_functional.fallthrough(DispatchKey.PythonDispatcher)  # type: ignore[attr-defined]
triton_kernel_wrapper_functional.fallthrough(DispatchKey.PythonTLSSnapshot)  # type: ignore[attr-defined]
triton_kernel_wrapper_functional.fallthrough(DispatchKey.ADInplaceOrView)
triton_kernel_wrapper_functional.fallthrough(DispatchKey.BackendSelect)
triton_kernel_wrapper_functional.fallthrough(DispatchKey.AutocastCPU)  # type: ignore[attr-defined]
triton_kernel_wrapper_functional.fallthrough(DispatchKey.AutocastCUDA)  # type: ignore[attr-defined]
triton_kernel_wrapper_functional.fallthrough(DispatchKey.AutogradCUDA)
triton_kernel_wrapper_functional.fallthrough(DispatchKey.AutogradCUDA)
triton_kernel_wrapper_functional.fallthrough(DispatchKey.AutogradCPU)
