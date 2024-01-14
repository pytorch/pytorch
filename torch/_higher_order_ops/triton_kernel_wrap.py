import dataclasses
import inspect
import logging
import threading
from types import FunctionType
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
from torch.utils._triton import has_triton

log = logging.getLogger("torch._dynamo")


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
# Mutation Tracker


class IdentifyMutationException(Exception):
    pass


@dataclasses.dataclass
class Proxy:
    mutated: bool = False

    def __add__(self, o):
        if isinstance(o, Proxy):
            raise IdentifyMutationException("Proxy operation with Proxy")
        return self


class Scalar:
    @staticmethod
    def check(o):
        if isinstance(o, Proxy):
            raise IdentifyMutationException("Proxy operation with Scalar")

    def __radd__(self, o):
        if isinstance(o, Proxy):
            return o
        return self


# This operation is only needed when triton is available
if has_triton():
    import triton

    def replacement_fn(self, *args, **kwargs):
        if len(args) > 0:
            Scalar.check(args[0])
        return self

    for name, _ in inspect.getmembers(triton.language.core.tensor, inspect.isfunction):
        if name in ["__init__", "__radd__"]:
            continue

        setattr(Scalar, name, replacement_fn)


# Given a triton kernel and the arguments for this kernel, this function traces
# through the triton kernel and identifies which input pointers are mutated.
# Tracing is done by replacing the input pointers with Proxy objects that
# track mutation. Each triton language function is monkey patched to
# either detect the mutation or return a fresh scalar object.
def identify_mutated_tensors(kernel, kwargs):
    from triton import language as tl
    from triton.runtime.autotuner import Autotuner

    # Replace tensor pointers with Proxy object. Do not replace the other
    # inputs since tl.constexprs can be inspected by the kernel.
    proxy_kwargs = {
        key: (Proxy() if isinstance(value, Tensor) else value)
        for key, value in kwargs.items()
    }

    if isinstance(kernel, Autotuner):
        if len(kernel.configs) > 0:
            # If we are autotuning, then it doesn't matter which version gets
            # picked for tracing purposes, so lets pick the first one
            proxy_kwargs = {**proxy_kwargs, **kernel.configs[0].kwargs}
        kernel = kernel.fn

    default_impls: Dict[str, FunctionType] = {}
    mutation_ops = {
        "store",
        "atomic_add",
        "atomic_cas",
        "atomic_max",
        "atomic_min",
        "atomic_xchg",
    }
    try:
        # Monkey patch all triton language functions
        for name, impl in inspect.getmembers(tl, inspect.isfunction):
            default_impls[name] = impl

            if name in mutation_ops:
                # Ops that mutate the first argument tensor pointer

                def fn(pointer, *args, **kwargs):
                    assert isinstance(pointer, Proxy)
                    pointer.mutated = True
                    return Scalar()

            elif name == "make_block_ptr":
                # Creates indirection to base tensor pointer

                def fn(base, *args, **kwargs):  # type: ignore[misc] # different conditional signatures
                    assert isinstance(base, Proxy)
                    return base

            elif name == "inline_asm_elementwise":
                # If there's inline asm in the kernel, anything can happen.
                # Do not optimize

                def fn(*args, **kwargs):  # type: ignore[misc] # different conditional signatures
                    raise IdentifyMutationException("inline_asm_elementwise in kernel")

            else:
                # Any operation not listed above, do not mutate a kernel
                # pointer. They either return some sort of scalar or
                # for the purposes of tracing cause an uninteresting
                # side effect.

                def fn(*args, **kwargs):  # type: ignore[misc] # different conditional signatures
                    return Scalar()

            setattr(tl, name, fn)

        kernel.fn(**proxy_kwargs)
        return [
            key
            for key, value in proxy_kwargs.items()
            if isinstance(value, Proxy) and value.mutated
        ]
    except (AttributeError, IdentifyMutationException, RuntimeError, ValueError) as e:
        # - AttributeError: Triton converts constexpr to special objects
        #   TODO(oulgen): we also need to wrap them appropriately
        #
        # - IdentifyMutationException: An indication for us to not optimize
        #
        # - RuntimeError: Throw when the kernel calls another @triton.jit kernel
        #   "Cannot call @triton.jit'd outside of the scope of a kernel"
        #   TODO(oulgen): Find a way to monkey patch @triton.jit away
        #
        # - ValueError: If triton language function are imported ahead of time,
        #   we are no longer able to monkey patch them, so the execution throws
        #   "Did you forget to add @triton.jit ?"
        #   TODO(oulgen): Find a way to monkey patch this
        import traceback

        log.debug(
            "Encountered an exception in identify_mutated_tensors, assuming every input is mutated"
        )
        log.debug(
            "".join(
                traceback.TracebackException.from_exception(e).format()  # noqa: G001
            )
        )
        return [key for key, value in proxy_kwargs.items() if isinstance(value, Proxy)]
    finally:
        for name, impl in default_impls.items():
            setattr(tl, name, impl)


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
    kernel = kernel_side_table.get_kernel(kernel_idx)
    # TODO(oulgen): Preexisting bug, if two kernel inputs are views of each
    # other, and one gets mutated in kernel, and later another gets mutated,
    # they are no longer equal. Fix this by graph breaking on this condition
    # earlier in dynamo.
    tensors_to_clone = identify_mutated_tensors(kernel, unwrapped_kwargs)
    with ctx.redispatch_to_next():
        unwrapped_outputs = triton_kernel_wrapper_functional(
            kernel_idx=kernel_idx,
            grid=grid,
            kwargs=unwrapped_kwargs,
            tensors_to_clone=tensors_to_clone,
        )

    assert set(unwrapped_outputs.keys()).issubset(set(kwargs.keys()))
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
    return {key: val for key, val in kwargs.items() if key in tensors_to_clone}


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
            key: clone_preserve_strides(val)
            for key, val in kwargs.items()
            if key in tensors_to_clone
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
