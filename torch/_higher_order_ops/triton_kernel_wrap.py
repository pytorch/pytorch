import threading
from typing import Any, Dict

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


###############################################################################
# Kernel Side Table


# We cannot put Triton Kernels into the FX graph as the graph nodes
# do not support arbitrary functions.
# Use a side table.
# We use two dicts so that fetching both the kernel and id are O(1)
class KernelSideTable:
    id_to_kernel: Dict[int, Any] = dict()
    kernel_to_id: Dict[Any, int] = dict()
    lock = threading.Lock()

    # Returns index on the table
    def add_kernel(self, kernel) -> int:
        with self.lock:
            if kernel in self.kernel_to_id:
                return self.kernel_to_id[kernel]

            idx = len(self.id_to_kernel)
            self.id_to_kernel[idx] = kernel
            self.kernel_to_id[kernel] = idx
            return idx

    # Returns the triton kernel at the given index
    def get_kernel(self, idx: int):
        # No need to lock here as fetching from dict is atomic
        assert idx in self.id_to_kernel
        return self.id_to_kernel[idx]

    # Resets the table (only meant to be used in unit tests)
    # This is only safe assuming single threaded execution
    def reset_table(self) -> None:
        self.id_to_kernel = dict()
        self.kernel_to_id = dict()


kernel_side_table = KernelSideTable()


###############################################################################
# Triton Kernel Wrappers


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
def triton_kernel_wrapper_mutation_dense(*, kernel_idx, grid, kwargs):
    from torch._inductor.codegen.wrapper import user_defined_kernel_grid_fn_code

    kernel = kernel_side_table.get_kernel(kernel_idx)

    if len(grid) == 1:
        grid_fn = grid[0]
    else:
        fn_name, code = user_defined_kernel_grid_fn_code(
            kernel.fn.__name__, kernel.configs, grid
        )
        namespace: Dict[str, Any] = {}
        exec(code, namespace)
        grid_fn = namespace[fn_name]

    kernel[grid_fn](**kwargs)


@triton_kernel_wrapper_mutation.py_impl(FakeTensorMode)
def triton_kernel_wrapper_mutation_fake_tensor_mode(mode, *, kernel_idx, grid, kwargs):
    with mode:
        return None


def trace_triton_kernel_wrapper(proxy_mode, func_overload, node_args):
    with disable_proxy_modes_tracing():
        out = func_overload(**node_args)

    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, node_args)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function",
        func_overload,
        (),
        proxy_args,
        name=func_overload.__name__ + "_proxy",
    )
    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@triton_kernel_wrapper_mutation.py_impl(ProxyTorchDispatchMode)
def triton_kernel_wrapper_mutation_proxy_torch_dispatch_mode(
    mode, *, kernel_idx, grid, kwargs
):
    if mode.enable_tracing:
        trace_triton_kernel_wrapper(
            mode,
            triton_kernel_wrapper_mutation,
            {"kernel_idx": kernel_idx, "grid": grid, "kwargs": kwargs},
        )
    else:
        triton_kernel_wrapper_mutation(kernel_idx=kernel_idx, grid=grid, kwargs=kwargs)

    return None


@triton_kernel_wrapper_mutation.py_functionalize_impl
def triton_kernel_wrapper_mutation_functionalize(ctx, kernel_idx, grid, kwargs):
    unwrapped_kwargs = ctx.unwrap_tensors(kwargs)
    # TODO(oulgen): We can be smarter here and iterate through the triton kernel
    # to find which tensors are actually being mutated
    tensors_to_clone = [
        key for key, value in unwrapped_kwargs.items() if isinstance(value, Tensor)
    ]
    with ctx.redispatch_to_next():
        unwrapped_outputs = triton_kernel_wrapper_functional(
            kernel_idx=kernel_idx,
            grid=grid,
            kwargs=unwrapped_kwargs,
            tensors_to_clone=tensors_to_clone,
        )

    assert unwrapped_outputs.keys() == kwargs.keys()
    for key, output_arg in unwrapped_outputs.items():
        if not isinstance(output_arg, Tensor):
            continue
        input_arg = kwargs[key]
        assert isinstance(input_arg, Tensor)

        ctx.replace(input_arg, output_arg)
        # indicate that above replace is hidden from autograd
        ctx.mark_mutation_hidden_from_autograd(input_arg)
        ctx.commit_update(input_arg)
        ctx.sync(input_arg)
        # sync calls replace_ under the hood, so again indicate that
        # this indirect replace is hidden from autograd
        ctx.mark_mutation_hidden_from_autograd(input_arg)
    return None


@triton_kernel_wrapper_functional.py_impl(DispatchKey.CompositeExplicitAutograd)
def triton_kernel_wrapper_functional_dense(
    *, kernel_idx, grid, kwargs, tensors_to_clone
):
    # TODO(oulgen): For performance reasons, we want to ensure that these
    # `clone_preserve_strides` calls are never executed at runtime
    # (inductor should always optimize them away).
    # Requires https://github.com/pytorch/pytorch/issues/109240
    kwargs = {
        key: (clone_preserve_strides(val) if key in tensors_to_clone else val)
        for key, val in kwargs.items()
    }
    triton_kernel_wrapper_mutation(kernel_idx=kernel_idx, grid=grid, kwargs=kwargs)
    return kwargs


@triton_kernel_wrapper_functional.py_impl(FakeTensorMode)
def triton_kernel_wrapper_functional_fake_tensor_mode(
    mode, *, kernel_idx, grid, kwargs, tensors_to_clone
):
    # TODO(oulgen): For performance reasons, we want to ensure that these
    # `clone_preserve_strides` calls are never executed at runtime
    # (inductor should always optimize them away).
    # Requires https://github.com/pytorch/pytorch/issues/109240
    with mode:
        return {
            key: (clone_preserve_strides(val) if key in tensors_to_clone else val)
            for key, val in kwargs.items()
        }


@triton_kernel_wrapper_functional.py_impl(ProxyTorchDispatchMode)
def triton_kernel_wrapper_functional_proxy_torch_dispatch_mode(
    mode, *, kernel_idx, grid, kwargs, tensors_to_clone
):
    if mode.enable_tracing:
        return trace_triton_kernel_wrapper(
            mode,
            triton_kernel_wrapper_functional,
            {
                "kernel_idx": kernel_idx,
                "grid": grid,
                "kwargs": kwargs,
                "tensors_to_clone": tensors_to_clone,
            },
        )
    else:
        return triton_kernel_wrapper_functional(
            kernel_idx=kernel_idx,
            grid=grid,
            kwargs=kwargs,
            tensors_to_clone=tensors_to_clone,
        )


@triton_kernel_wrapper_functional.py_functionalize_impl
def triton_kernel_wrapper_functional_functionalize(
    ctx, kernel_idx, grid, kwargs, tensors_to_clone
):
    unwrapped_kwargs = ctx.unwrap_tensors(kwargs)
    with ctx.redispatch_to_next():
        outputs = triton_kernel_wrapper_functional(
            kernel_idx=kernel_idx,
            grid=grid,
            kwargs=unwrapped_kwargs,
            tensors_to_clone=tensors_to_clone,
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
