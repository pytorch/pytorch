"""User-defined CuTeDSL kernels in PT2: the counterpart of triton_kernel_wrap.py.

A CuTeDSL kernel is launched by a ``@cute.jit`` host function. PT2 captures a call to
one as ``cutedsl_kernel_wrapper_mutation(kernel_idx, constant_args_idx, kwargs)``: the
host function and the user's write set are a side-table record (CuteDSLKernel), the
tensors and dynamic scalars are ``kwargs``, and every other argument is a constant
arg in the side table. Functionalization rewrites the node into
``cutedsl_kernel_wrapper_functional``, exactly as for Triton, and the DSL-agnostic
passes treat both DSLs the same way through the user-kernel HOP registry
(user_kernel_hop.py).

The stream parameter of the host function (the one annotated ``cuda.CUstream``) is
never part of the node: the kernel is launched on the current stream when the node
runs.
"""

import copy
import dataclasses
import inspect
import weakref
from collections.abc import Callable, Mapping
from typing import Any, Literal, TYPE_CHECKING
from typing_extensions import Never

import sympy

import torch.utils._pytree as pytree
from torch import SymBool, SymFloat, SymInt, Tensor
from torch._C import DispatchKey
from torch._higher_order_ops.user_kernel_hop import (
    _object_state,
    functional_hop_dense,
    functional_hop_fake,
    functionalize_functional_hop,
    functionalize_mutation_hop,
    KernelSideTable,
    register_user_kernel_hop,
    register_user_kernel_hop_dispatch,
    store_non_graphable_args,
    trace_user_kernel_wrapper,
    UserKernelHOPifier,
    UserKernelHOPInfo,
    UserKernelWrapperHOP,
)
from torch._higher_order_ops.utils import register_fake
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode
from torch.fx.experimental.symbolic_shapes import guard_scalar


if TYPE_CHECKING:
    from torch._subclasses.functional_tensor import BaseFunctionalizeAPI


###############################################################################
# Kernel Side Table


# The parameter names the kernel writes, or "unknown": every tensor argument.
MutatesArgs = tuple[str, ...] | Literal["unknown"]

# The write set Dynamo gives a raw call (a host function called as in eager code, not
# through wrap_cutedsl): every tensor argument.
RAW_CALL_MUTATES_ARGS: MutatesArgs = "unknown"


@dataclasses.dataclass(frozen=True)
class CuteDSLKernel:
    """Side-table entry for a CuTeDSL host function: plays the part a JITFunction
    plays for Triton. The write set is part of the record, so one host function
    wrapped with two different lists gets two entries. ``mutates_args`` may be any
    iterable of parameter names (stored as a tuple without repeats), or "unknown"."""

    fn: Callable[..., Any]
    mutates_args: MutatesArgs

    def __post_init__(self) -> None:
        if self.mutates_args == "unknown":
            return
        if isinstance(self.mutates_args, str):
            raise ValueError(
                f'CuTeDSL kernel {self.name}: mutates_args must be "unknown" or a '
                f"sequence of parameter names, got {self.mutates_args!r}"
            )
        names = tuple(dict.fromkeys(self.mutates_args))
        object.__setattr__(self, "mutates_args", names)
        if not names:
            return
        info = get_host_fn_info(self.fn)
        writable = tuple(p for p in info.params if p != info.stream_param)
        for name in names:
            if name not in writable:
                raise ValueError(
                    f"CuTeDSL kernel {self.name}: mutates_args names {name!r}, but "
                    f"its parameters other than the stream {info.stream_param!r} "
                    f"are {writable}"
                )

    @property
    def name(self) -> str:
        return getattr(inspect.unwrap(self.fn), "__name__", "cutedsl_kernel")


# Unlike Triton's table, torch._dynamo.reset() leaves this one alone: graphs and
# exported programs captured before a reset keep resolving their indices.
cutedsl_kernel_side_table: KernelSideTable[CuteDSLKernel] = KernelSideTable(
    dedupe_constant_args=True
)


def get_kernel(kernel_idx: int) -> CuteDSLKernel:
    return cutedsl_kernel_side_table.get_kernel(kernel_idx)


###############################################################################
# Host function introspection


@dataclasses.dataclass(frozen=True)
class CuteDSLHostFnInfo:
    """What PT2 needs from a host function's signature.

    Arguments fall into four classes: the stream (supplied at launch); tensors and
    runtime scalars, which are kernel arguments; and constants, which the DSL bakes
    into the compiled kernel: Constexpr parameters and values that are neither
    tensors nor scalars (None, objects). A runtime scalar is a value for a parameter
    annotated with a cutlass Numeric type such as cutlass.Int32, or a Python
    int/float/bool for an unannotated parameter, which the DSL converts to
    Int32/Float32/Boolean.

    ``defaults`` are the parameters' defaults as written: like a passed value, a
    default that is a runtime scalar goes through python_scalar at launch, and any
    other default reaches the DSL unchanged.
    """

    params: tuple[str, ...]
    stream_param: str
    constexpr_params: frozenset[str]
    numeric_params: Mapping[str, type]
    defaults: Mapping[str, Any]

    def is_runtime_scalar(self, name: str, value: object) -> bool:
        if name in self.numeric_params:
            return True
        if name in self.constexpr_params:
            return False
        # sympy.Basic: Inductor's symbolic sizes (Expr) and symbolic bools (Boolean).
        return isinstance(
            value, (bool, int, float, SymInt, SymFloat, SymBool, sympy.Basic)
        )


def python_scalar(value: Any) -> Any:
    """A runtime scalar as the Python value the DSL expects: a cutlass Numeric value
    as its .value, a subclass of bool/int/float (an IntEnum, a numpy float) as the
    builtin, which the DSL's argument adapters require exactly."""
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if (getattr(type(value), "__module__", "") or "").startswith("cutlass"):
        from cutlass.base_dsl.typing import Boolean, Numeric

        if isinstance(value, Boolean):
            return bool(value.value)  # .value is the int 0 or 1
        if isinstance(value, Numeric):
            return value.value
    return value


def is_dsl_runtime_object(cls: type) -> bool:
    """Objects PT2 has to make itself: cute tensors and pointers (it converts torch
    tensors) and CUDA streams (it passes the current stream). Other DSL values, such
    as cutlass dtype classes, enums and Numeric values, are ordinary constants."""
    module = getattr(cls, "__module__", "") or ""
    name = getattr(cls, "__name__", "")
    return module.startswith("cutlass.cute.runtime") or (
        module.startswith("cuda.") and name in ("CUstream", "CUdeviceptr")
    )


_host_fn_infos: "weakref.WeakKeyDictionary[Callable[..., Any], CuteDSLHostFnInfo]" = (
    weakref.WeakKeyDictionary()
)


def get_host_fn_info(fn: Callable[..., Any]) -> CuteDSLHostFnInfo:
    """Parse a ``@cute.jit`` host function's signature; raises ValueError for the
    ones PT2 cannot launch."""
    info = _host_fn_infos.get(fn)
    if info is not None:
        return info

    import cuda.bindings.driver as cuda  # pyrefly: ignore[missing-import]
    from cutlass.base_dsl.dsl import is_arg_spec_constexpr
    from cutlass.base_dsl.typing import NumericMeta

    inner = inspect.unwrap(fn)
    name = getattr(inner, "__qualname__", repr(inner))
    params, constexpr, defaults = [], set(), {}
    numeric: dict[str, type] = {}
    streams = []
    for i, p in enumerate(inspect.signature(inner).parameters.values()):
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            raise ValueError(f"{name}: *args and **kwargs parameters are not supported")
        ann = p.annotation
        if isinstance(ann, str):
            raise ValueError(
                f"{name}: annotation of {p.name!r} is a string; CuTeDSL needs real "
                "annotations (remove `from __future__ import annotations`)"
            )
        params.append(p.name)
        if p.default is not p.empty:
            defaults[p.name] = p.default
        if inspect.isclass(ann) and issubclass(ann, cuda.CUstream):
            streams.append(p.name)
        elif is_arg_spec_constexpr(ann, p.name, i, inner):
            constexpr.add(p.name)
        elif isinstance(ann, NumericMeta):
            numeric[p.name] = ann
    if len(streams) != 1:
        raise ValueError(
            f"{name}: expected exactly one parameter annotated cuda.CUstream (PT2 "
            f"launches the kernel on the current CUDA stream), found {streams}"
        )
    info = CuteDSLHostFnInfo(
        params=tuple(params),
        stream_param=streams[0],
        constexpr_params=frozenset(constexpr),
        numeric_params=numeric,
        defaults=defaults,
    )
    _host_fn_infos[fn] = info
    return info


###############################################################################
# CuTeDSL Kernel Wrappers


# Used for wrapping a CuTeDSL host function
class CuteDSLKernelWrapperMutation(UserKernelWrapperHOP):
    def __init__(self) -> None:
        # Not cacheable: the generated code binds the host function in-process.
        super().__init__("cutedsl_kernel_wrapper_mutation", cacheable=False)

    def __call__(
        self,
        kernel_idx: int,
        constant_args_idx: int,
        kwargs: dict[str, Any],
    ) -> Any:
        # pyrefly: ignore [missing-attribute]
        return super().__call__(
            kernel_idx=kernel_idx,
            constant_args_idx=constant_args_idx,
            kwargs=kwargs,
        )


cutedsl_kernel_wrapper_mutation = CuteDSLKernelWrapperMutation()


# Used for wrapping a CuTeDSL host function in a functional manner
class CuteDSLKernelWrapperFunctional(UserKernelWrapperHOP):
    def __init__(self) -> None:
        super().__init__("cutedsl_kernel_wrapper_functional", cacheable=False)

    def __call__(
        self,
        kernel_idx: int,
        constant_args_idx: int,
        kwargs: dict[str, Any],
        tensors_to_clone: list[str],
        tensor_bases: dict[str, Tensor] | None = None,
    ) -> dict[str, Any]:
        hop_kwargs: dict[str, Any] = {
            "kernel_idx": kernel_idx,
            "constant_args_idx": constant_args_idx,
            "kwargs": kwargs,
            "tensors_to_clone": tensors_to_clone,
        }
        if tensor_bases:
            hop_kwargs["tensor_bases"] = tensor_bases

        # pyrefly: ignore [missing-attribute]
        return super().__call__(**hop_kwargs)


cutedsl_kernel_wrapper_functional = CuteDSLKernelWrapperFunctional()


@cutedsl_kernel_wrapper_mutation.py_impl(DispatchKey.CompositeExplicitAutograd)
def cutedsl_kernel_wrapper_mutation_dense(
    *,
    kernel_idx: int,
    constant_args_idx: int,
    kwargs: dict[str, Any],
) -> None:
    from torch._inductor.runtime.cutedsl_user_kernel import launch_cutedsl_kernel

    kernel = cutedsl_kernel_side_table.get_kernel(kernel_idx)
    constant_args = cutedsl_kernel_side_table.get_constant_args(constant_args_idx)
    launch_cutedsl_kernel(kernel.fn, {**kwargs, **constant_args})


@register_fake(cutedsl_kernel_wrapper_mutation, skip_cache=True)
def cutedsl_kernel_wrapper_mutation_fake_tensor_mode(
    *,
    kernel_idx: int,
    constant_args_idx: int,
    kwargs: dict[str, Any],
) -> None:
    return None


@cutedsl_kernel_wrapper_mutation.py_impl(DispatchKey.Meta)
def _(
    *,
    kernel_idx: int,
    constant_args_idx: int,
    kwargs: dict[str, Any],
) -> None:
    return None


@cutedsl_kernel_wrapper_mutation.py_impl(ProxyTorchDispatchMode)
def cutedsl_kernel_wrapper_mutation_proxy_torch_dispatch_mode(
    mode: ProxyTorchDispatchMode,
    *,
    kernel_idx: int,
    constant_args_idx: int,
    kwargs: dict[str, Any],
) -> None:
    trace_user_kernel_wrapper(
        mode,
        cutedsl_kernel_wrapper_mutation,
        {
            "kernel_idx": kernel_idx,
            "constant_args_idx": constant_args_idx,
            "kwargs": kwargs,
        },
    )

    return None


def get_cutedsl_mutated_tensors(
    kernel_idx: int,
    constant_args_idx: int,
    kwargs: dict[str, Any],
) -> list[str]:
    """Names of the tensor kwargs the kernel writes: the mirror of Triton's
    get_mutated_tensors, read from the user's mutates_args instead of TTIR."""
    kernel = cutedsl_kernel_side_table.get_kernel(kernel_idx)
    if kernel.mutates_args == "unknown":
        names: tuple[str, ...] = tuple(kwargs)
    else:
        names = kernel.mutates_args
    return [name for name in names if isinstance(kwargs.get(name), Tensor)]


def _same_storage(a: Tensor, b: Tensor) -> bool:
    from torch.multiprocessing.reductions import StorageWeakRef

    try:
        a_storage, b_storage = a.untyped_storage(), b.untyped_storage()
    except RuntimeError:  # no storage (NotImplementedError is a RuntimeError)
        return False
    return StorageWeakRef(a_storage) == StorageWeakRef(b_storage)


def _may_overlap(a: Tensor, b: Tensor) -> bool:
    """Whether two tensors may share memory: the same tensor, or two on one storage
    that are not provably disjoint (statically_known_true, so no guards). Provably
    disjoint are an empty tensor and any other, tensors whose byte ranges do not
    meet, and 2-D tensors with contiguous rows the same stride apart whose rows do
    not meet modulo that stride, such as the q, k and v columns of a fused qkv
    buffer. Bytes, not elements: views of one storage can differ in dtype. (Not
    compute_overlapping_tensors, which reinplacing uses: it reports some
    overlapping 2-D views as disjoint.)"""
    if a is b:
        return True
    if not _same_storage(a, b):
        return False
    from torch.fx.experimental.symbolic_shapes import statically_known_true

    if any(statically_known_true(t.numel() == 0) for t in (a, b)):
        return False

    def start(t: Tensor) -> Any:
        return t.storage_offset() * t.element_size()

    def end(t: Tensor) -> Any:
        # Exact unless t is empty, and an empty tensor overlaps nothing.
        last = sum((n - 1) * s for n, s in zip(t.shape, t.stride()))
        return start(t) + (last + 1) * t.element_size()

    if statically_known_true(end(a) <= start(b)) or statically_known_true(
        end(b) <= start(a)
    ):
        return False
    if a.dim() != 2 or b.dim() != 2:
        return True
    # Modulo a symbolic row stride, only a difference known to be >= 0 simplifies.
    if statically_known_true(start(b) < start(a)):
        a, b = b, a
    row, b_row = (t.stride(0) * t.element_size() for t in (a, b))
    same_rows = (a.stride(1) == 1, b.stride(1) == 1, row > 0, b_row == row)
    if not all(map(statically_known_true, same_rows)):
        return True
    # Every row of b starts `shift` bytes into a period of a's rows.
    shift = (start(b) - start(a)) % row
    return not (
        statically_known_true(a.shape[1] * a.element_size() <= shift)
        and statically_known_true(shift + b.shape[1] * b.element_size() <= row)
    )


def check_written_args_disjoint(
    kernel_idx: int, written: list[str], kwargs: dict[str, Any], wrapped: dict[str, Any]
) -> list[str]:
    """``written``, after checking that no two of those arguments may overlap in
    memory. The functional HOP clones each written argument and writes each clone
    back on its own, in kwargs order, so where two written arguments overlap the
    later write-back overwrites the earlier one: an in-place call under
    mutates_args="unknown" (the same tensor twice), or overlapping views of one base,
    would lose a write.

    ``kwargs`` holds the values unwrapped from the functional tensors in
    ``wrapped``. Functional tensors that alias share a storage; where their
    unwrapped values do not, functionalization copied the views
    (torch.func.functionalize with remove="mutations_and_views"), and the copies do
    not show where the views sit, so they count as overlapping."""
    for i, later in enumerate(written):
        for earlier in written[:i]:
            a, b = kwargs[earlier], kwargs[later]
            aliases = _same_storage(wrapped[earlier], wrapped[later])
            if _may_overlap(a, b) or (aliases and not _same_storage(a, b)):
                kernel = cutedsl_kernel_side_table.get_kernel(kernel_idx)
                raise RuntimeError(
                    f"CuTeDSL kernel {kernel.name}: {earlier!r} and {later!r} are the "
                    "same tensor or views that may overlap, and both count as written "
                    f"(mutates_args={kernel.mutates_args!r}), which PT2 cannot "
                    "functionalize: one write would be lost. Pass tensors that do not "
                    "overlap, or give mutates_args naming only the parameters the "
                    "kernel writes."
                )
    return written


@cutedsl_kernel_wrapper_mutation.py_functionalize_impl
def cutedsl_kernel_wrapper_mutation_functionalize(
    ctx: "BaseFunctionalizeAPI",
    kernel_idx: int,
    constant_args_idx: int,
    kwargs: dict[str, Any],
) -> None:
    functionalize_mutation_hop(
        ctx,
        cutedsl_kernel_wrapper_functional,
        lambda unwrapped_kwargs: check_written_args_disjoint(
            kernel_idx,
            get_cutedsl_mutated_tensors(
                kernel_idx, constant_args_idx, unwrapped_kwargs
            ),
            unwrapped_kwargs,
            kwargs,
        ),
        hop_kwargs={"kernel_idx": kernel_idx, "constant_args_idx": constant_args_idx},
        kwargs=kwargs,
    )
    return None


@cutedsl_kernel_wrapper_functional.py_impl(DispatchKey.CompositeExplicitAutograd)
def cutedsl_kernel_wrapper_functional_dense(
    *,
    kernel_idx: int,
    constant_args_idx: int,
    kwargs: dict[str, Any],
    tensors_to_clone: list[str],
    tensor_bases: dict[str, Tensor] | None = None,
) -> dict[str, Any]:
    return functional_hop_dense(
        cutedsl_kernel_wrapper_mutation,
        hop_kwargs={"kernel_idx": kernel_idx, "constant_args_idx": constant_args_idx},
        kwargs=kwargs,
        tensors_to_clone=tensors_to_clone,
        tensor_bases=tensor_bases,
    )


@register_fake(cutedsl_kernel_wrapper_functional, skip_cache=True)
def cutedsl_kernel_wrapper_functional_fake_tensor_mode(
    *,
    kernel_idx: int,
    constant_args_idx: int,
    kwargs: dict[str, Any],
    tensors_to_clone: list[str],
    tensor_bases: dict[str, Tensor] | None = None,
) -> dict[str, Any]:
    return functional_hop_fake(
        kwargs=kwargs, tensors_to_clone=tensors_to_clone, tensor_bases=tensor_bases
    )


@cutedsl_kernel_wrapper_functional.py_impl(ProxyTorchDispatchMode)
def cutedsl_kernel_wrapper_functional_proxy_torch_dispatch_mode(
    mode: ProxyTorchDispatchMode,
    *,
    kernel_idx: int,
    constant_args_idx: int,
    kwargs: dict[str, Any],
    tensors_to_clone: list[str],
    tensor_bases: dict[str, Tensor] | None = None,
) -> dict[str, Any]:
    node_args: dict[str, Any] = {
        "kernel_idx": kernel_idx,
        "constant_args_idx": constant_args_idx,
        "kwargs": kwargs,
        "tensors_to_clone": tensors_to_clone,
    }
    if tensor_bases:
        node_args["tensor_bases"] = tensor_bases

    ret = trace_user_kernel_wrapper(mode, cutedsl_kernel_wrapper_functional, node_args)
    if ret is None:
        raise AssertionError("trace_user_kernel_wrapper returned None")
    return ret


@cutedsl_kernel_wrapper_functional.py_functionalize_impl
def cutedsl_kernel_wrapper_functional_functionalize(
    ctx: "BaseFunctionalizeAPI",
    kernel_idx: int,
    constant_args_idx: int,
    kwargs: dict[str, Any],
    tensors_to_clone: list[str],
    tensor_bases: dict[str, Tensor] | None = None,
) -> dict[str, Any]:
    return functionalize_functional_hop(
        ctx,
        cutedsl_kernel_wrapper_functional,
        hop_kwargs={"kernel_idx": kernel_idx, "constant_args_idx": constant_args_idx},
        kwargs=kwargs,
        tensors_to_clone=tensors_to_clone,
        tensor_bases=tensor_bases,
    )


register_user_kernel_hop_dispatch(
    cutedsl_kernel_wrapper_mutation, cutedsl_kernel_wrapper_functional
)


###############################################################################
# The "CuteDSLHOPifier": a class that transforms a call to a CuTeDSL host function
# into a call to the cutedsl_kernel_wrapper_mutation HOP.


def normalize_mutates_args(mutates_args: object) -> MutatesArgs:
    if mutates_args == "unknown":
        return "unknown"
    error = ValueError(
        'mutates_args must be a sequence of parameter names or "unknown", '
        f"got {mutates_args!r}"
    )
    if isinstance(mutates_args, str):
        raise error
    try:
        # Materialize once: a generator would be empty on a second pass.
        names = tuple(mutates_args)  # type: ignore[call-overload]
    except TypeError:
        raise error from None
    if not all(isinstance(name, str) for name in names):
        raise error
    return tuple(dict.fromkeys(names))


class CuteDSLHOPifier(UserKernelHOPifier):
    """Orchestrator for converting a call to a CuTeDSL host function into a call to
    the cutedsl_kernel_wrapper_mutation HOP: the counterpart of TritonHOPifier.

    It has two main use cases.

    1. When Dynamo sees a host function, raw or wrapped, it becomes a
    CuteDSLKernelVariable, whose calls DynamoCuteDSLHOPifier turns into a call to the
    HOP. A raw call is written as for eager, with ``from_dlpack(t)`` for each tensor
    and ``cutlass.torch.current_stream()`` for the stream; it is captured as the call
    of ``wrap_cutedsl(fn, mutates_args=RAW_CALL_MUTATES_ARGS)`` with the torch
    tensors and no stream.

    2. For make_fx and non-strict export, which cannot see a raw call, wrap_cutedsl
    returns a TraceableCuteDSLKernelWrapper, whose calls TracingCuteDSLHOPifier turns
    into a call to the HOP (which can then be traced).

    A CuTeDSL launch has no grid, autotuner or heuristics, so what is left is binding
    the call to the host function's signature: the stream is supplied at launch, so a
    wrapper's caller does not pass it; Constexpr arguments are baked into the compiled
    kernel, so symbolic values for them are specialized; tensors and scalars are
    kernel arguments (see CuteDSLHostFnInfo).
    """

    def is_tensor(self, val: Any) -> bool:
        raise NotImplementedError("abstract method")

    def python_type(self, val: Any) -> type:
        raise NotImplementedError("abstract method")

    def leaves(self, val: Any) -> list[Any]:
        """The values inside a container argument (the value itself otherwise);
        dataclass instances and plain objects are containers (see _object_state)."""
        raise NotImplementedError("abstract method")

    def specialize_nested(self, variable: Any, val: Any) -> Any:
        """val with every symbolic value in it (or itself) specialized."""
        raise NotImplementedError("abstract method")

    def cute_tensor_arg(self, val: Any) -> Any:
        """The torch tensor ``t`` if ``val`` is ``from_dlpack(t)`` (in a raw call),
        else None."""
        raise NotImplementedError("abstract method")

    def is_current_stream(self, val: Any, tx: Any) -> bool:
        """Whether ``val`` is ``cutlass.torch.current_stream()`` read on the stream
        that is current at the call (in a raw call)."""
        raise NotImplementedError("abstract method")

    def wrap_raw(self, variable: Any) -> Any:
        """The wrapper a raw call of ``variable`` is captured through:
        ``wrap_cutedsl(variable.kernel, mutates_args=RAW_CALL_MUTATES_ARGS)``."""
        raise NotImplementedError("abstract method")

    def may_overlap(self, a: Any, b: Any) -> bool:
        """Whether tensors ``a`` and ``b`` may share memory (see _may_overlap)."""
        raise NotImplementedError("abstract method")

    def overlapping_args(self, tensors: Mapping[str, Any]) -> tuple[str, str] | None:
        """The names of the first two of ``tensors`` that may share memory."""
        items = list(tensors.items())
        for i, (later, val) in enumerate(items):
            for earlier, other in items[:i]:
                if self.may_overlap(other, val):
                    return earlier, later
        return None

    def call_HOP(  # type: ignore[no-untyped-def]
        self,
        variable,
        combined_args: dict[str, Any],
        tx,
    ) -> Any:
        raise NotImplementedError("abstract method")

    def host_fn_info(self, kernel: Callable[..., Any]) -> CuteDSLHostFnInfo:
        try:
            return get_host_fn_info(kernel)
        except ValueError as e:
            self.raise_unsupported(str(e))

    def init_variable(
        self,
        variable: Any,
        kernel: Callable[..., Any],
        mutates_args: object,
        kernel_idx: int | None = None,
        *,
        raw: bool = False,
    ) -> None:
        """Record the host function on ``variable`` and add it, with its write set,
        to the side table. A ``raw`` host function (Dynamo met the @cute.jit function
        itself, not a wrapper) has no write set and gets no record of its own: a call
        of it is captured through ``wrap_raw``."""
        variable.kernel = kernel
        variable.mutates_args = None
        variable.kernel_idx = None
        if raw:
            if mutates_args is not None:
                raise AssertionError(
                    f"a raw host function has no write set, got {mutates_args!r}"
                )
            return
        try:
            normalized = normalize_mutates_args(mutates_args)
        except ValueError as e:
            self.raise_unsupported(str(e))
        info = self.host_fn_info(kernel)
        if normalized != "unknown":
            scalar_params = (
                {info.stream_param} | info.constexpr_params | set(info.numeric_params)
            )
            for name in normalized:
                if name not in info.params or name in scalar_params:
                    self.raise_unsupported(
                        f"mutates_args: {name!r} is not a tensor parameter of "
                        f"{CuteDSLKernel(kernel, ()).name} (parameters: {info.params})"
                    )
        variable.mutates_args = normalized
        variable.kernel_idx = cutedsl_kernel_side_table.add_kernel(
            CuteDSLKernel(kernel, normalized)
        )
        if kernel_idx is not None and variable.kernel_idx != kernel_idx:
            raise AssertionError(
                f"kernel_idx mismatch: expected {kernel_idx}, got {variable.kernel_idx}"
            )

    def bind_args(
        self,
        variable: Any,
        args: tuple[Any, ...] | list[Any],
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Bind a call of a wrapper (torch tensors, no stream) to the host function's
        parameters, minus the stream."""
        kernel = variable.kernel
        name = CuteDSLKernel(kernel, ()).name
        info = self.host_fn_info(kernel)
        params = [p for p in info.params if p != info.stream_param]
        if len(args) > len(params):
            self.raise_unsupported(
                f"{name}() takes {len(params)} arguments without the stream but "
                f"{len(args)} were given"
            )
        combined_args = dict(zip(params, args))
        for key, val in kwargs.items():
            if key == info.stream_param:
                self.raise_unsupported(
                    f"{name}(): do not pass {key!r}; the kernel is launched on the "
                    "current CUDA stream"
                )
            if key not in params:
                self.raise_unsupported(
                    f"{name}() got an unexpected keyword argument {key!r}"
                )
            if key in combined_args:
                self.raise_unsupported(
                    f"{name}() got multiple values for argument {key!r}"
                )
            combined_args[key] = val
        missing = [
            p for p in params if p not in combined_args and p not in info.defaults
        ]
        if missing:
            self.raise_unsupported(f"{name}() missing arguments {missing}")

        for key, val in combined_args.items():
            if self.is_tensor(val):
                if key in info.constexpr_params or key in info.numeric_params:
                    self.raise_unsupported(
                        f"{name}(): a tensor was passed to the scalar parameter {key!r}"
                    )
                continue
            leaves = self.leaves(val)
            for leaf in leaves:
                if self.is_tensor(leaf):
                    # Only top-level tensors become graph arguments; one inside a
                    # container would be baked in as a trace-time constant.
                    self.raise_unsupported(
                        f"{name}(): {key!r} holds a tensor inside a container; pass "
                        "each tensor as its own argument"
                    )
                if is_dsl_runtime_object(self.python_type(leaf)):
                    self.raise_unsupported(
                        f"{name}(): pass torch.Tensor and Python values, not "
                        f"{self.python_type(leaf).__name__} ({key!r}); PT2 converts "
                        "tensors and supplies the stream"
                    )
            is_container = not (len(leaves) == 1 and leaves[0] is val)
            if key in info.constexpr_params or is_container:
                # The DSL bakes a Constexpr argument into the compiled kernel, and a
                # container can only be a side-table constant, so symbolic values in
                # either are specialized (see [Note: Specialize tl.constexpr args in
                # user-defined triton kernels]).
                combined_args[key] = self.specialize_nested(variable, val)
        return combined_args

    def unwrap_raw_args(
        self,
        variable: Any,
        args: tuple[Any, ...] | list[Any],
        kwargs: dict[str, Any],
        tx: Any,
    ) -> dict[str, Any]:
        """A raw call's arguments, by name, as its wrapper takes them: each
        ``from_dlpack(t)`` as ``t``, and no stream. The stream has to be
        ``cutlass.torch.current_stream()`` read on the stream current at the call,
        because the wrapper launches on that stream."""
        kernel = variable.kernel
        name = CuteDSLKernel(kernel, ()).name
        info = self.host_fn_info(kernel)
        if len(args) > len(info.params):
            self.raise_unsupported(
                f"{name}() takes {len(info.params)} arguments but got {len(args)}"
            )
        bound = dict(zip(info.params, args))
        for key, val in kwargs.items():
            if key not in info.params:
                self.raise_unsupported(
                    f"{name}() got an unexpected keyword argument {key!r}"
                )
            if key in bound:
                self.raise_unsupported(
                    f"{name}() got multiple values for argument {key!r}"
                )
            bound[key] = val
        stream = bound.pop(info.stream_param, None)
        if stream is None or not self.is_current_stream(stream, tx):
            self.raise_unsupported(
                f"{name}(): pass cutlass.torch.current_stream(), read on the stream "
                f"current at this call, for {info.stream_param!r}; a captured kernel "
                "is launched on the current CUDA stream"
            )
        for key, val in bound.items():
            tensor = self.cute_tensor_arg(val)
            if tensor is not None:
                bound[key] = tensor
            elif self.is_tensor(val):
                self.raise_unsupported(
                    f"{name}(): pass from_dlpack(tensor) for {key!r}, not a "
                    "torch.Tensor (eager CuTeDSL refuses one too)"
                )
            elif is_dsl_runtime_object(self.python_type(val)):
                self.raise_unsupported(
                    f"{name}(): {key!r} is a {self.python_type(val).__name__} made "
                    "outside the compiled region; only from_dlpack(tensor) and "
                    "cutlass.torch.current_stream() called inside it are captured"
                )
        # Overlapping tensors for two parameters would fail check_written_args_disjoint
        # in the backend, which cannot fall back to eager; here it is a graph break.
        overlap = self.overlapping_args(
            {key: val for key, val in bound.items() if self.is_tensor(val)}
        )
        if overlap is not None:
            earlier, later = overlap
            self.raise_unsupported(
                f"{name}(): {earlier!r} and {later!r} are the same tensor or views "
                "that may overlap, and a raw call counts every tensor as written, "
                "which PT2 cannot functionalize (one write would be lost); call "
                f"wrap_cutedsl({name}, mutates_args=[...]) naming only the "
                "parameters the kernel writes"
            )
        return bound

    def call_cutedsl_kernel(
        self,
        variable: Any,
        args: tuple[Any, ...] | list[Any],
        kwargs: dict[str, Any],
        tx: Any,
        *,
        raw: bool = False,
    ) -> Any:
        if raw:
            # A raw call: captured as the call of its wrapper.
            kwargs = self.unwrap_raw_args(variable, args, kwargs, tx)
            args = ()
            variable = self.wrap_raw(variable)
        return self.call_HOP(variable, self.bind_args(variable, args, kwargs), tx)


###############################################################################
# Helpers for the wrap_cutedsl API, which makes a CuTeDSL host function traceable
# into a graph via make_fx or non-strict export


class TracingCuteDSLHOPifier(CuteDSLHOPifier):
    def raise_unsupported(self, msg: str) -> Never:
        raise RuntimeError(msg)

    def get_value(self, val: Any) -> Any:
        return val

    def is_tensor(self, val: Any) -> bool:
        return isinstance(val, Tensor)

    def python_type(self, val: Any) -> type:
        return type(val)

    def leaves(self, val: Any) -> list[Any]:
        leaves: list[Any] = []
        for leaf in pytree.tree_leaves(val):
            state = None if self.is_tensor(leaf) else _object_state(leaf)
            leaves += [leaf] if state is None else self.leaves(list(state.values()))
        return leaves

    def specialize_nested(self, variable: Any, val: Any) -> Any:
        def specialize(leaf: Any) -> Any:
            state = _object_state(leaf)
            if state is None:
                return variable.specialize_symbolic(leaf)
            values = {k: self.specialize_nested(variable, v) for k, v in state.items()}
            if all(values[k] is v for k, v in state.items()):
                return leaf
            new = copy.copy(leaf)
            for name, value in values.items():
                object.__setattr__(new, name, value)
            return new

        return pytree.tree_map(specialize, val)

    def call_HOP(
        self,
        variable: "TraceableCuteDSLKernelWrapper",
        combined_args: dict[str, Any],
        tx: None,
    ) -> None:
        if tx is not None:
            raise AssertionError("tx must be None for TracingCuteDSLHOPifier")
        graphable_args, constant_args_idx = store_non_graphable_args(
            cutedsl_kernel_side_table, combined_args
        )
        return cutedsl_kernel_wrapper_mutation(
            kernel_idx=variable.kernel_idx,
            constant_args_idx=constant_args_idx,
            kwargs=graphable_args,
        )


tracing_cutedsl_hopifier_singleton = TracingCuteDSLHOPifier()


class TraceableCuteDSLKernelWrapper:
    """What wrap_cutedsl returns: calling it launches the host function eagerly, or
    records a cutedsl_kernel_wrapper_mutation node under make_fx / non-strict export."""

    kernel: Callable[..., Any]
    mutates_args: MutatesArgs
    kernel_idx: int

    def __init__(self, kernel: Callable[..., Any], mutates_args: object) -> None:
        tracing_cutedsl_hopifier_singleton.init_variable(self, kernel, mutates_args)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        from torch._library.triton import is_wrap_triton_enabled

        if is_wrap_triton_enabled():
            return tracing_cutedsl_hopifier_singleton.call_cutedsl_kernel(
                self, args, kwargs, None
            )
        # Eager fast path (e.g. inside a triton_op's eager backend): launch directly,
        # skipping the HOP dispatch.
        from torch._inductor.runtime.cutedsl_user_kernel import launch_cutedsl_kernel

        launch_cutedsl_kernel(
            self.kernel,
            tracing_cutedsl_hopifier_singleton.bind_args(self, args, kwargs),
        )

    def specialize_symbolic(self, arg: Any) -> Any:
        import torch

        # See [Note: Specialize tl.constexpr args in user-defined triton kernels]
        if isinstance(arg, (torch.SymInt, torch.SymBool, torch.SymFloat)):
            return guard_scalar(arg)
        return arg


# Imported after every definition: torch._dynamo imports CuteDSLHOPifier and
# other names from this module.
from torch._dynamo.utils import ReInplaceTrigger


register_user_kernel_hop(
    UserKernelHOPInfo(
        dsl="cutedsl",
        mutation=cutedsl_kernel_wrapper_mutation,
        functional=cutedsl_kernel_wrapper_functional,
        get_mutated_tensors=lambda hop_kwargs, kwargs: get_cutedsl_mutated_tensors(
            hop_kwargs["kernel_idx"], hop_kwargs["constant_args_idx"], kwargs
        ),
        get_kernel_name=lambda hop_kwargs: get_kernel(hop_kwargs["kernel_idx"]).name,
        functional_dense=cutedsl_kernel_wrapper_functional_dense,
        reinplace_trigger=ReInplaceTrigger.CUTEDSL_OPS,
    )
)
