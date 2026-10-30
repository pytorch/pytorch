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

import dataclasses
import inspect
import weakref
from collections.abc import Callable, Mapping
from typing import Any, Literal, TYPE_CHECKING

import sympy

from torch import SymBool, SymFloat, SymInt, Tensor
from torch._C import DispatchKey
from torch._higher_order_ops.user_kernel_hop import (
    functional_hop_dense,
    functional_hop_fake,
    functionalize_functional_hop,
    functionalize_mutation_hop,
    KernelSideTable,
    register_user_kernel_hop,
    register_user_kernel_hop_dispatch,
    trace_user_kernel_wrapper,
    UserKernelHOPInfo,
    UserKernelWrapperHOP,
)
from torch._higher_order_ops.utils import register_fake
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode


if TYPE_CHECKING:
    from torch._subclasses.functional_tensor import BaseFunctionalizeAPI


###############################################################################
# Kernel Side Table


# The parameter names the kernel writes, or "unknown": every tensor argument.
MutatesArgs = tuple[str, ...] | Literal["unknown"]


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
    that compute_overlapping_tensors, which reinplacing uses too, does not find
    disjoint. It guards on the symbolic sizes it compares, and a size it cannot guard
    on (an unbacked one) counts as an overlap. It counts offsets and strides in
    elements, so views of one storage with different element sizes are compared by
    their byte ranges instead, without guards (statically_known_true)."""
    if a is b:
        return True
    if not _same_storage(a, b):
        return False
    from torch._C._dynamo.guards import compute_overlapping_tensors
    from torch.fx.experimental.symbolic_shapes import (
        GuardOnDataDependentSymNode,
        statically_known_true,
    )

    if a.element_size() == b.element_size():
        try:
            return bool(compute_overlapping_tensors([a, b]))
        except GuardOnDataDependentSymNode:
            return True

    def byte_range(t: Tensor) -> tuple[Any, Any]:
        # Exact unless t is empty, and an empty tensor overlaps nothing.
        start = t.storage_offset() * t.element_size()
        last = sum((n - 1) * s for n, s in zip(t.shape, t.stride()))
        return start, start + (last + 1) * t.element_size()

    (a_start, a_end), (b_start, b_end) = byte_range(a), byte_range(b)
    return not (
        statically_known_true(a_end <= b_start)
        or statically_known_true(b_end <= a_start)
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


# Imported after every definition, as in triton_kernel_wrap.py.
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
        serializable=False,
    )
)
