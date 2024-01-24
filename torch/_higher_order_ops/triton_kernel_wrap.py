import dataclasses
import logging
import threading
from typing import Any, Dict, List, Union

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


# Given a triton kernel and the arguments for this kernel, this function traces
# through the triton kernel and identifies which input pointers are mutated.
# Tracing is done by replacing the input pointers with Proxy objects that
# track mutation. Each triton language function is monkey patched to
# either detect the mutation or return a fresh scalar object.
def identify_mutated_tensors(kernel, kwargs):
    import functools

    import triton
    from triton.compiler.code_generator import ast_to_ttir
    from triton.runtime.jit import JITFunction

    assert isinstance(kernel, JITFunction)

    args = [val for key, val in kwargs.items()]

    specialization = kernel._get_config(*args)
    constants = {i: arg for i, arg in enumerate(args) if not isinstance(arg, Tensor)}
    debug = None
    target = None

    # Build kernel signature -- doesn't include constexpr arguments.
    signature = {
        i: kernel._type_of(kernel._key_of(arg))
        for i, arg in enumerate(args)
        if i not in kernel.constexprs
    }

    @dataclasses.dataclass(frozen=True)
    class TensorParam:
        idx: int

    @dataclasses.dataclass(frozen=True)
    class NonTensorParam:
        source: Any
        pass

    @dataclasses.dataclass(frozen=True)
    class Intermediate:
        idx: int

    mappings: Dict[Any, Union[TensorParam, NonTensorParam, Intermediate]] = dict()
    next_intermediate = 0

    def convert(arg):
        if isinstance(arg, triton._C.libtriton.triton.ir.block_argument):
            if arg not in mappings:
                mappings[arg] = NonTensorParam(arg)
            return mappings[arg]
        if isinstance(arg, triton._C.libtriton.triton.ir.value):
            if arg not in mappings:
                nonlocal next_intermediate
                mappings[arg] = Intermediate(next_intermediate)
                next_intermediate += 1
            return mappings[arg]
        return arg

    # Name of mutation op to mutated parameter indices
    MUTATION_OPS = {"masked_store": [0]}

    ops: Dict[Union[TensorParam, NonTensorParam, Intermediate], "Op"] = dict()
    sinks: List["Op"] = []

    @dataclasses.dataclass
    class Op:
        name: str
        args: List[Any]
        kwargs: Dict[str, Any]

        @staticmethod
        def create(name, args, kwargs, result):
            op = Op(name, [convert(a) for a in args], kwargs)
            ops[convert(result)] = op
            if name in MUTATION_OPS:
                sinks.append(op)

    class MutationAnalysisWrapper:
        def __init__(self, builder):
            self.builder = builder

        def __getattr__(self, name):
            def generic_indirection(name, *args, **kwargs):
                result = getattr(self.builder, name)(*args, **kwargs)

                if name.startswith("create_"):
                    Op.create(name[len("create_") :], args, kwargs, result)
                elif name in {"get_int32"}:
                    Op.create(name[len("get_") :], args, kwargs, result)
                elif name in {"get_loc", "set_loc"}:
                    pass
                else:
                    pass
                return result

            return functools.partial(generic_indirection, name)

    original_add_entry_block = triton._C.libtriton.triton.ir.function.add_entry_block

    def custom_add_entry_block(self):
        result = self.original_add_entry_block()
        for i, arg in enumerate(args):
            if isinstance(arg, Tensor):
                mappings[self.args(i)] = TensorParam(i)
        return result

    triton._C.libtriton.triton.ir.function.add_entry_block = custom_add_entry_block
    triton._C.libtriton.triton.ir.function.original_add_entry_block = (
        original_add_entry_block
    )

    OriginalCodeGenerator = triton.compiler.code_generator.CodeGenerator

    class CustomCodeGenerator(OriginalCodeGenerator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.builder = MutationAnalysisWrapper(self.builder)

        def visit_compound_statement(self, *args, **kwargs):
            super().visit_compound_statement(*args, **kwargs)
            breakpoint()

    triton.compiler.code_generator.CodeGenerator = CustomCodeGenerator

    try:
        ttir_module = ast_to_ttir(
            kernel, signature, specialization, constants, debug, target
        )
    except Exception as e:
        import traceback

        log.debug(
            "Encountered an exception in identify_mutated_tensors, assuming every input is mutated"
        )
        log.debug(
            "".join(
                traceback.TracebackException.from_exception(e).format()  # noqa: G001
            )
        )
        return [key for key, value in kwargs.items() if isinstance(value, Tensor)]
    finally:
        triton._C.libtriton.triton.ir.function.add_entry_block = (
            original_add_entry_block
        )
        triton.compiler.code_generator.CodeGenerator = OriginalCodeGenerator

    stack = []
    for sink in sinks:
        for idx in MUTATION_OPS[sink.name]:
            stack.append(sink.args[idx])

    mutated = [False] * len(kwargs)
    while len(stack):
        arg = stack[-1]
        stack.pop()

        if isinstance(arg, TensorParam):
            mutated[arg.idx] = True
        elif isinstance(arg, NonTensorParam):
            pass
        elif isinstance(arg, Intermediate):
            for a in ops[arg].args:
                stack.append(a)
        else:
            # There are some scalar args
            pass

    return [
        key
        for i, (key, value) in enumerate(kwargs.items())
        if isinstance(value, Tensor) and mutated[i]
    ]


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
