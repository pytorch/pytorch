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
import sys
import weakref
from collections.abc import Callable, Collection, Mapping
from typing import Any, Literal, TYPE_CHECKING

import sympy

from torch import SymBool, SymFloat, SymInt, Tensor
from torch._C import DispatchKey
from torch._higher_order_ops.reinplace_trigger import ReInplaceTrigger
from torch._higher_order_ops.user_kernel_hop import (
    ConstantRules,
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


def cutedsl_kernel_name(fn: Callable[..., Any]) -> str:
    return getattr(inspect.unwrap(fn), "__name__", "cutedsl_kernel")


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
        error = TypeError(
            f'CuTeDSL kernel {self.name}: mutates_args must be "unknown" or an '
            f"iterable of parameter names, got {self.mutates_args!r}"
        )
        if isinstance(self.mutates_args, str):
            raise error
        try:
            names = tuple(dict.fromkeys(self.mutates_args))
        except TypeError:
            raise error from None
        if not all(isinstance(name, str) for name in names):
            raise error
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
        return cutedsl_kernel_name(self.fn)


def _is_cutlass_type(cls: type) -> bool:
    """Whether ``cls`` comes from the cutlass package (not cutlass_library, say)."""
    module = getattr(cls, "__module__", "") or ""
    return module == "cutlass" or module.startswith("cutlass.")


class _CuteDSLConstantRules(ConstantRules):
    def is_dsl_value(self, value: object) -> bool:
        return _is_cutlass_type(type(value))

    def scalar_value(self, value: object) -> bool | int | float | None:
        # A cutlass Numeric, such as cutlass.Float32(2.0), is keyed by its value.
        inner = getattr(value, "value", None) if self.is_dsl_value(value) else None
        return inner if isinstance(inner, (bool, int, float)) else None


CUTEDSL_CONSTANT_RULES = _CuteDSLConstantRules()

# Unlike Triton's table, torch._dynamo.reset() leaves this one alone: graphs and
# exported programs captured before a reset keep resolving their indices.
cutedsl_kernel_side_table: KernelSideTable[CuteDSLKernel] = KernelSideTable(
    constant_rules=CUTEDSL_CONSTANT_RULES
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
    tensors nor runtime scalars (None, objects). See is_runtime_scalar and
    split_args.

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
        """Whether ``value``, for parameter ``name``, is a runtime scalar: any value
        but None for a parameter annotated with a cutlass Numeric type such as
        cutlass.Int32; for an unannotated parameter, a Python bool, int or float, a
        numpy scalar that python_scalar converts to one, or a symbolic one (a SymBool,
        SymInt or SymFloat, or Inductor's sympy expression for one), which the DSL
        converts to Boolean, Int32 or Float32; never for a Constexpr parameter."""
        if value is None or name in self.constexpr_params:
            return False
        if name in self.numeric_params:
            return True
        # sympy.Basic: Inductor's symbolic sizes (Expr) and symbolic bools (Boolean).
        scalar_types = (bool, int, float, SymInt, SymFloat, SymBool, sympy.Basic)
        return isinstance(value, scalar_types) or _numpy_scalar_item(value) is not None

    def check_call(self, name: str, names: Collection[str]) -> None:
        """Check that ``names``, the arguments of a call by name, bind to the
        parameters of kernel ``name`` but the stream, which the launch supplies:
        raises a TypeError for a name that is not a parameter, for the stream, and
        for missing parameters without a default."""
        for arg in names:
            if arg == self.stream_param:
                raise TypeError(
                    f"CuTeDSL kernel {name}: got a value for its stream parameter "
                    f"{arg!r}, which the launch supplies"
                )
            if arg not in self.params:
                raise TypeError(
                    f"CuTeDSL kernel {name}: got an unexpected argument {arg!r}; its "
                    f"parameters are {self.params}"
                )
        missing = [
            p
            for p in self.params
            if p != self.stream_param and p not in names and p not in self.defaults
        ]
        if missing:
            raise TypeError(f"CuTeDSL kernel {name}: missing arguments {missing}")

    def split_args(
        self, name: str, args: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """The runtime arguments (tensors and runtime scalars) and the constants of a
        call of kernel ``name`` that binds (see check_call): every parameter but the
        stream, in signature order, those missing from ``args`` taking their
        defaults. Raises a TypeError for a tensor for a Constexpr or cutlass Numeric
        parameter, and a ValueError for a parameter left to a tensor default."""
        runtime_args: dict[str, Any] = {}
        constant_args: dict[str, Any] = {}
        for param in self.params:
            if param == self.stream_param:
                continue
            if param in args:
                value = args[param]
            else:
                value = self.defaults[param]
                if isinstance(value, Tensor):
                    raise ValueError(
                        f"CuTeDSL kernel {name}: parameter {param!r} defaults to a "
                        "tensor; pass the tensor"
                    )
            if isinstance(value, Tensor):
                if param in self.constexpr_params or param in self.numeric_params:
                    raise TypeError(
                        f"CuTeDSL kernel {name}: parameter {param!r} takes a scalar "
                        "(it is Constexpr or annotated with a cutlass Numeric type), "
                        "not a tensor"
                    )
                runtime_args[param] = value
            elif self.is_runtime_scalar(param, value):
                runtime_args[param] = value
            else:
                constant_args[param] = value
        return runtime_args, constant_args


def _numpy_scalar_item(value: object) -> bool | int | float | None:
    """``value.item()`` for a numpy bool, integer or floating scalar whose item is a
    Python bool, int or float (not a long double wider than a float), else None.
    numpy is not imported: without it, no value is one of its scalars."""
    np = sys.modules.get("numpy")
    if np is None or not isinstance(value, (np.bool_, np.integer, np.floating)):
        return None
    item = value.item()
    return item if type(item) in (bool, int, float) else None


def python_scalar(value: Any, annotation: type | None = None) -> Any:
    """A runtime scalar as the Python value the DSL expects, whose argument adapters
    take only an exact bool, int or float: a numpy bool, integer or floating scalar
    as its .item(); a subclass of bool, int or float (an IntEnum, np.float64) as that
    builtin; a cutlass Numeric value as its .value; and an int for a parameter
    annotated with a cutlass float type (``annotation``) as a float."""
    item = _numpy_scalar_item(value)
    if item is not None:
        value = item
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        if annotation is not None:
            from cutlass.base_dsl.typing import Boolean, Integer

            if not issubclass(annotation, (Boolean, Integer)):
                # nvidia-cutlass-dsl 4.5 and later would pass the int through Int32
                # first, overflowing or wrapping one beyond its range.
                return float(value)
        return int(value)
    if isinstance(value, float):
        return float(value)
    if _is_cutlass_type(type(value)):
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
    """Parse a ``@cute.jit`` host function's signature; raises a ValueError for the
    ones PT2 cannot launch."""
    info = _host_fn_infos.get(fn)
    if info is not None:
        return info

    import cuda.bindings.driver as cuda  # pyrefly: ignore[missing-import]
    from cutlass.base_dsl.runtime import jit_arg_adapters
    from cutlass.base_dsl.typing import NumericMeta

    # Named is_arg_spec_constexpr before nvidia-cutlass-dsl 4.5.
    is_constexpr = getattr(jit_arg_adapters, "is_arg_annotation_constexpr", None)
    if is_constexpr is None:
        is_constexpr = jit_arg_adapters.is_arg_spec_constexpr

    inner = inspect.unwrap(fn)
    name = cutedsl_kernel_name(fn)
    params, constexpr, defaults = [], set(), {}
    numeric: dict[str, type] = {}
    streams = []
    for i, p in enumerate(inspect.signature(inner).parameters.values()):
        if p.kind is not p.POSITIONAL_OR_KEYWORD:
            # The launch passes the arguments to cute.compile by name and to the
            # compiled function by position, which CuTeDSL binds to keyword-only
            # parameters only on its fast path, for one value per runtime parameter.
            raise ValueError(
                f"CuTeDSL kernel {name}: parameter {p.name!r} is "
                f"{p.kind.description}, but PT2 needs every parameter to accept both "
                "positional and keyword arguments"
            )
        ann = p.annotation
        if isinstance(ann, str):
            raise ValueError(
                f"CuTeDSL kernel {name}: parameter {p.name!r} has a string "
                "annotation; CuTeDSL needs real annotations (remove `from __future__ "
                "import annotations`)"
            )
        params.append(p.name)
        if p.default is not p.empty:
            defaults[p.name] = p.default
        if inspect.isclass(ann) and issubclass(ann, cuda.CUstream):
            streams.append(p.name)
        elif is_constexpr(ann, p.name, i, inner):
            constexpr.add(p.name)
        elif isinstance(ann, NumericMeta):
            numeric[p.name] = ann
    if len(streams) != 1:
        raise ValueError(
            f"CuTeDSL kernel {name}: expected exactly one parameter annotated "
            "cuda.CUstream (PT2 launches the kernel on the current CUDA stream), "
            f"found {streams}"
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
        # Not cacheable: its side-table indices exist only in the process that
        # captured them.
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
        # Not cacheable: its side-table indices exist only in the process that
        # captured them.
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
    from torch._native.cutedsl.user_kernel import launch_cutedsl_kernel

    table = cutedsl_kernel_side_table
    kernel = table.get_kernel(kernel_idx)
    constant_args = table.get_constant_args(constant_args_idx)
    launch_cutedsl_kernel(
        kernel.fn,
        {**kwargs, **constant_args},
        constant_keys=table.get_constant_args_keys(constant_args_idx),
    )


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


def _rows_disjoint(a: Tensor, b: Tensor) -> bool:
    """Whether two views of one storage, both 2-D with stride 1 in dim 1 and one
    stride s in dim 0, are disjoint. With x the view with the lower storage offset,
    y the other and d the distance between their offsets: one is empty, their
    element ranges do not meet, or each row of y lies in the gap after a row of x
    (s > 0, d % s >= x.size(1) and d % s + y.size(1) <= s)."""
    if a.numel() == 0 or b.numel() == 0:
        return True
    x, y = (b, a) if a.storage_offset() > b.storage_offset() else (a, b)
    s, d = x.stride(0), y.storage_offset() - x.storage_offset()
    if (x.size(0) - 1) * s + x.size(1) <= d:
        return True
    if s == 0:
        return False
    return bool(d % s >= x.size(1) and d % s + y.size(1) <= s)


def _may_overlap(a: Tensor, b: Tensor) -> bool:
    """Whether two tensors may share memory: the same tensor, or two on one storage
    that compute_overlapping_tensors, which reinplacing uses too, does not find
    disjoint (with _rows_disjoint agreeing for 2-D views with contiguous rows of one
    stride). It guards on the symbolic sizes it compares, and a size it cannot guard
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
            if compute_overlapping_tensors([a, b]):
                return True
            # TODO(zoranzhao): remove this check and _rows_disjoint once
            # https://github.com/pytorch/pytorch/pull/202848 lands. Until then,
            # compute_overlapping_tensors reports 2-D views whose rows start inside
            # the other's rows, such as base[:4, :3] and base[1:5, 2:5], as disjoint.
            rows = a.dim() == b.dim() == 2 and a.stride(1) == b.stride(1) == 1
            if rows and a.stride(0) == b.stride(0):
                return not _rows_disjoint(a, b)
            return False
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
            if _may_overlap(a, b):
                problem = "are the same tensor or views that may overlap"
                fix = "Pass tensors that do not overlap."
            elif aliases and not _same_storage(a, b):
                problem = (
                    "are views of one tensor, which functionalization with "
                    'remove="mutations_and_views" copied, so PT2 cannot tell whether '
                    "they overlap"
                )
                fix = (
                    'Functionalize with remove="mutations" (the default), which keeps '
                    "views, or pass tensors that are not views of one tensor."
                )
            else:
                continue
            kernel = cutedsl_kernel_side_table.get_kernel(kernel_idx)
            if kernel.mutates_args == "unknown":
                fix += " Or give mutates_args naming only the parameters it writes."
            raise RuntimeError(
                f"CuTeDSL kernel {kernel.name}: {earlier!r} and {later!r} {problem}, "
                f"and both count as written (mutates_args={kernel.mutates_args!r}). "
                "PT2 clones each written argument and writes it back on its own, so "
                f"one write could be lost. {fix}"
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


register_user_kernel_hop(
    UserKernelHOPInfo(
        dsl="cutedsl",
        mutation=cutedsl_kernel_wrapper_mutation,
        functional=cutedsl_kernel_wrapper_functional,
        get_mutated_tensors=lambda hop_kwargs, kwargs: get_cutedsl_mutated_tensors(
            hop_kwargs["kernel_idx"], hop_kwargs["constant_args_idx"], kwargs
        ),
        get_kernel_name=lambda hop_kwargs: get_kernel(hop_kwargs["kernel_idx"]).name,
        get_layout_constraint=lambda: "needs_fixed_stride_order",
        reinplace_trigger=ReInplaceTrigger.CUTEDSL_OPS,
        serializable=False,
    )
)
