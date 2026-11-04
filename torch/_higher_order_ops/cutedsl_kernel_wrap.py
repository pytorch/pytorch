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
import functools
import inspect
import sys
import weakref
from collections.abc import Callable, Collection, Mapping
from typing import Any, Literal, TYPE_CHECKING
from typing_extensions import Never

import sympy

from torch import SymBool, SymFloat, SymInt, Tensor
from torch._C import DispatchKey
from torch._higher_order_ops.reinplace_trigger import ReInplaceTrigger
from torch._higher_order_ops.user_kernel_hop import (
    constant_key,
    ConstantRules,
    functional_hop_dense,
    functional_hop_fake,
    functionalize_functional_hop,
    functionalize_mutation_hop,
    KernelSideTable,
    map_constant,
    register_user_kernel_hop,
    register_user_kernel_hop_dispatch,
    specialize_symbolic,
    store_non_graphable_args,
    trace_user_kernel_wrapper,
    UserKernelHOPifier,
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

# The write set Dynamo gives a raw call (a host function called as in eager code, not
# through wrap_cutedsl): every tensor argument.
RAW_CALL_MUTATES_ARGS: MutatesArgs = "unknown"


def cutedsl_kernel_name(fn: Callable[..., Any]) -> str:
    return getattr(inspect.unwrap(fn), "__name__", "cutedsl_kernel")


def normalize_mutates_args(fn: Callable[..., Any], mutates_args: object) -> MutatesArgs:
    """The write set of host function ``fn``: "unknown", or the parameter names, read
    once (a generator works) into a tuple without repeats, of parameters the kernel
    can write: not the stream, nor a Constexpr or cutlass Numeric one. Raises a
    TypeError for anything but "unknown" or an iterable of str (exactly str: not the
    numpy strings of an array of names), and a ValueError for a name the kernel
    cannot write, naming ``fn``; an empty write set does not parse the signature.
    CuteDSLHOPifier.bind_call checks that a call passes each written parameter a
    tensor or None, or leaves it to a default of None."""
    if isinstance(mutates_args, str) and mutates_args == "unknown":
        return "unknown"
    name = cutedsl_kernel_name(fn)
    error = TypeError(
        f'CuTeDSL kernel {name}: mutates_args must be "unknown" or an iterable of '
        "parameter names, each exactly a str (not a numpy string or another str "
        f"subclass), got {mutates_args!r}"
    )
    if isinstance(mutates_args, str):
        raise error
    try:
        # Materialize once: a generator would be empty on a second pass.
        names = tuple(mutates_args)  # type: ignore[call-overload]
    except TypeError:
        raise error from None
    if not all(type(n) is str for n in names):
        raise error
    names = tuple(dict.fromkeys(names))
    if not names:
        return names
    info = get_host_fn_info(fn)
    scalars = {info.stream_param, *info.constexpr_params, *info.numeric_params}
    writable = tuple(p for p in info.params if p not in scalars)
    for n in names:
        if n not in writable:
            raise ValueError(
                f"CuTeDSL kernel {name}: mutates_args names {n!r}, but the parameters "
                "it can write (all but the stream, Constexpr and cutlass Numeric "
                f"ones) are {writable}"
            )
    return names


@dataclasses.dataclass(frozen=True)
class CuteDSLKernel:
    """Side-table entry for a CuTeDSL host function: plays the part a JITFunction
    plays for Triton. The write set is part of the record, so one host function
    wrapped with two different lists gets two entries. ``mutates_args`` may be any
    iterable of parameter names (stored as a tuple without repeats), or "unknown",
    as normalize_mutates_args validates it."""

    fn: Callable[..., Any]
    mutates_args: MutatesArgs

    def __post_init__(self) -> None:
        mutates_args = normalize_mutates_args(self.fn, self.mutates_args)
        object.__setattr__(self, "mutates_args", mutates_args)

    @property
    def name(self) -> str:
        return cutedsl_kernel_name(self.fn)


def _is_cutlass_type(cls: type) -> bool:
    """Whether ``cls`` comes from the cutlass package (not cutlass_library, say)."""
    module = getattr(cls, "__module__", "") or ""
    return module == "cutlass" or module.startswith("cutlass.")


# What to pass instead of a DSL runtime object (see is_dsl_runtime_object).
_DSL_RUNTIME_OBJECT_FIX = (
    "pass torch.Tensor and Python values; PT2 converts tensors and supplies the stream"
)

# class -> whether its instances are kernel objects (see keyed_by_content).
_kernel_object_classes: "weakref.WeakKeyDictionary[type, bool]" = (
    weakref.WeakKeyDictionary()
)


class _CuteDSLConstantRules(ConstantRules):
    def is_dsl_value(self, value: object) -> bool:
        return _is_cutlass_type(type(value))

    def scalar_value(self, value: object) -> bool | int | float | None:
        # A cutlass Numeric, such as cutlass.Float32(2.0), is keyed by its value.
        inner = getattr(value, "value", None) if self.is_dsl_value(value) else None
        return inner if isinstance(inner, (bool, int, float)) else None

    def keyed_by_content(self, value: object) -> bool:
        # A kernel object: its class's __call__ is a @cute.jit host function.
        cls = type(value)
        answer = _kernel_object_classes.get(cls)
        if answer is None:
            from torch._library.cutedsl import is_cutedsl_jit_function

            answer = is_cutedsl_jit_function(cls.__call__)
            _kernel_object_classes[cls] = answer
        return answer

    def refusal(self, cls: type) -> str | None:
        return _DSL_RUNTIME_OBJECT_FIX if is_dsl_runtime_object(cls) else None


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


def _tensor_default_error(name: str, param: str) -> ValueError:
    return ValueError(
        f"CuTeDSL kernel {name}: parameter {param!r} defaults to a tensor; pass the "
        "tensor"
    )


def _scalar_param_error(name: str, param: str) -> TypeError:
    return TypeError(
        f"CuTeDSL kernel {name}: parameter {param!r} takes a scalar (it is Constexpr "
        "or annotated with a cutlass Numeric type), not a tensor"
    )


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
    other default reaches the DSL unchanged. ``name`` is the host function's
    cutedsl_kernel_name, which error messages give.
    """

    params: tuple[str, ...]
    stream_param: str
    constexpr_params: frozenset[str]
    numeric_params: Mapping[str, type]
    defaults: Mapping[str, Any]
    name: str = "cutedsl_kernel"

    @functools.cached_property
    def stream_index(self) -> int:
        """The position of the stream parameter, which no positional argument of a
        call reaches (see CuteDSLHOPifier.bind_call)."""
        return self.params.index(self.stream_param)

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
                    raise _tensor_default_error(name, param)
            if isinstance(value, Tensor):
                if param in self.constexpr_params or param in self.numeric_params:
                    raise _scalar_param_error(name, param)
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
        name=name,
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


###############################################################################
# The "CuteDSLHOPifier": a class that transforms a call to a CuTeDSL host function
# into a call to the cutedsl_kernel_wrapper_mutation HOP.


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

    Every mistake of the user's (a write set, host function or call that PT2 cannot
    take) goes through raise_user_error as a TypeError or ValueError naming the
    kernel and any parameter involved: the one CuteDSLKernel or the eager launch
    raises for the same mistake, where they check it.
    """

    def raise_user_error(self, exc: TypeError | ValueError) -> Never:
        """Refuse a write set, host function or call with ``exc``."""
        raise NotImplementedError("abstract method")

    def is_tensor(self, val: Any) -> bool:
        raise NotImplementedError("abstract method")

    def python_type(self, val: Any) -> type:
        raise NotImplementedError("abstract method")

    def leaves(self, val: Any) -> list[Any]:
        """The values inside a container argument (the value itself otherwise):
        containers are what the side table keys by content (see constant_contents).
        Refuses a tensor or a DSL runtime object in it (only top-level tensors become
        graph arguments; one inside a container would be baked in as a trace-time
        constant) and a cycle, with the TypeError or ValueError of the side table's
        walk, which says where it sits; bind_args re-raises it naming the kernel and
        the parameter."""
        raise NotImplementedError("abstract method")

    def specialize_nested(self, variable: Any, val: Any) -> Any:
        """val with every symbolic value in it (or itself) specialized."""
        raise NotImplementedError("abstract method")

    def cute_tensor_arg(self, name: str, key: str, val: Any) -> Any:
        """The torch tensor ``t`` if ``val`` is ``from_dlpack(t)`` (in a raw call),
        else None; refused if ``t`` lost the layout it had at from_dlpack. ``name``
        and ``key`` name the host function and the parameter."""
        raise NotImplementedError("abstract method")

    def is_current_stream(self, val: Any, tx: Any) -> bool:
        """Whether ``val`` is ``cutlass.torch.current_stream()`` read on the stream
        that is current at the call (in a raw call)."""
        raise NotImplementedError("abstract method")

    def wrap_raw(self, variable: Any) -> Any:
        """The wrapper a raw call of ``variable`` is captured through:
        ``wrap_cutedsl(variable.kernel, mutates_args=RAW_CALL_MUTATES_ARGS)``."""
        raise NotImplementedError("abstract method")

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
            self.raise_user_error(e)

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
            normalized = normalize_mutates_args(kernel, mutates_args)
        except (TypeError, ValueError) as e:
            self.raise_user_error(e)
        self.host_fn_info(kernel)  # refuses a host function PT2 cannot launch
        variable.mutates_args = normalized
        variable.kernel_idx = cutedsl_kernel_side_table.add_kernel(
            CuteDSLKernel(kernel, normalized)
        )
        if kernel_idx is not None and variable.kernel_idx != kernel_idx:
            raise AssertionError(
                f"kernel_idx mismatch: expected {kernel_idx}, got {variable.kernel_idx}"
            )

    def bind_call(
        self,
        variable: Any,
        args: tuple[Any, ...] | list[Any],
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Bind a call of a wrapper (torch tensors, no stream) to the host function's
        parameters, minus the stream, and
        refuse what PT2 cannot launch without looking inside an argument: a call
        that does not bind, a tensor for a scalar parameter, a DSL runtime object,
        a parameter left to a tensor default, and a written parameter given or left
        to anything but a tensor or None. The eager fast path launches the result,
        and the launch walks each constant (see bind_args)."""
        info = self.host_fn_info(variable.kernel)
        name = info.name
        if len(args) > info.stream_index:
            msg = (
                f"CuTeDSL kernel {name}: takes {info.stream_index} positional "
                f"arguments (the parameters before its stream parameter "
                f"{info.stream_param!r}, which the launch supplies) but got {len(args)}"
            )
            by_keyword = info.params[info.stream_index + 1 : len(args) + 1]
            if by_keyword:
                msg += f"; pass {', '.join(map(repr, by_keyword))} by keyword"
            self.raise_user_error(TypeError(msg))
        combined_args = dict(zip(info.params, args))
        for key, val in kwargs.items():
            if key in combined_args:
                self.raise_user_error(
                    TypeError(
                        f"CuTeDSL kernel {name}: got multiple values for argument "
                        f"{key!r}"
                    )
                )
            combined_args[key] = val

        try:
            info.check_call(name, combined_args)
        except TypeError as e:
            self.raise_user_error(e)

        # PT2 can only functionalize a tensor that the call passes: a written
        # parameter takes a tensor, or None for an optional tensor that the call
        # leaves out, and no parameter is left to a tensor default.
        written = () if variable.mutates_args == "unknown" else variable.mutates_args
        for key, default in info.defaults.items():
            if key in combined_args:
                continue
            if isinstance(default, Tensor):
                self.raise_user_error(_tensor_default_error(name, key))
            if key in written and default is not None:
                self.raise_user_error(
                    ValueError(
                        f"CuTeDSL kernel {name}: mutates_args names parameter {key!r}, "
                        "which takes a tensor or None, but the call leaves it to its "
                        f"default, of type {type(default).__name__}"
                    )
                )
        for key, val in combined_args.items():
            if self.is_tensor(val):
                if key in info.constexpr_params or key in info.numeric_params:
                    self.raise_user_error(_scalar_param_error(name, key))
                continue
            val_type = self.python_type(val)
            if is_dsl_runtime_object(val_type):
                self.raise_user_error(
                    TypeError(
                        f"CuTeDSL kernel {name}: parameter {key!r} got a "
                        f"{val_type.__name__}: {_DSL_RUNTIME_OBJECT_FIX}"
                    )
                )
            if key in written and val_type is not type(None):
                self.raise_user_error(
                    TypeError(
                        f"CuTeDSL kernel {name}: mutates_args names parameter {key!r}, "
                        f"which takes a tensor or None, not {val_type.__name__}"
                    )
                )
        return combined_args

    def bind_args(
        self,
        variable: Any,
        args: tuple[Any, ...] | list[Any],
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """bind_call, then each argument that is not a tensor walked as the side
        table walks a constant (leaves), which refuses what the side table refuses in
        it, and the symbolic values in it specialized where the DSL bakes them into
        the compiled kernel."""
        combined_args = self.bind_call(variable, args, kwargs)
        info = self.host_fn_info(variable.kernel)
        for key, val in combined_args.items():
            if self.is_tensor(val):
                continue
            try:
                leaves = self.leaves(val)
            except (TypeError, ValueError) as e:
                msg = f"CuTeDSL kernel {info.name}: parameter {key!r}: {e}"
                self.raise_user_error(type(e)(msg))
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
        because the wrapper launches on that stream. A call that eager CuTeDSL
        refuses too (one that does not bind, or with a torch.Tensor for a parameter)
        is a user error (see raise_user_error)."""
        kernel = variable.kernel
        name = cutedsl_kernel_name(kernel)
        info = self.host_fn_info(kernel)
        if len(args) > len(info.params):
            self.raise_user_error(
                TypeError(
                    f"CuTeDSL kernel {name}: takes {len(info.params)} positional "
                    f"arguments but got {len(args)}"
                )
            )
        bound = dict(zip(info.params, args))
        for key, val in kwargs.items():
            if key not in info.params:
                self.raise_user_error(
                    TypeError(
                        f"CuTeDSL kernel {name}: got an unexpected keyword argument "
                        f"{key!r}"
                    )
                )
            if key in bound:
                self.raise_user_error(
                    TypeError(
                        f"CuTeDSL kernel {name}: got multiple values for argument "
                        f"{key!r}"
                    )
                )
            bound[key] = val
        missing = [p for p in info.params if p not in bound and p not in info.defaults]
        if missing:
            self.raise_user_error(
                TypeError(f"CuTeDSL kernel {name}: missing arguments {missing}")
            )
        stream = bound.pop(info.stream_param, None)
        if stream is None or not self.is_current_stream(stream, tx):
            self.raise_unsupported(
                f"CuTeDSL kernel {name}: pass cutlass.torch.current_stream(), read on "
                f"the stream current at this call, for {info.stream_param!r}; a "
                "captured kernel is launched on the current CUDA stream"
            )
        for key, val in bound.items():
            tensor = self.cute_tensor_arg(name, key, val)
            if tensor is not None:
                bound[key] = tensor
            elif self.is_tensor(val):
                self.raise_user_error(
                    TypeError(
                        f"CuTeDSL kernel {name}: parameter {key!r}: pass "
                        "from_dlpack(tensor), not a torch.Tensor (eager CuTeDSL "
                        "refuses one too)"
                    )
                )
            elif is_dsl_runtime_object(self.python_type(val)):
                self.raise_unsupported(
                    f"CuTeDSL kernel {name}: parameter {key!r} is a "
                    f"{self.python_type(val).__name__} made outside the compiled "
                    "region; only from_dlpack(tensor) and "
                    "cutlass.torch.current_stream() called inside it are captured"
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
        # The UserKernelHOPifier interface's: what this path refuses is a mistake of
        # the user's (raise_user_error).
        raise RuntimeError(msg)

    def raise_user_error(self, exc: TypeError | ValueError) -> Never:
        # from None: where exc is an error caught, or one that translates it,
        # chaining would print its message twice.
        raise exc from None

    def get_value(self, val: Any) -> Any:
        return val

    def is_tensor(self, val: Any) -> bool:
        return isinstance(val, Tensor)

    def python_type(self, val: Any) -> type:
        return type(val)

    def leaves(self, val: Any) -> list[Any]:
        leaves: list[Any] = []

        def collect(leaf: Any) -> Any:
            if not isinstance(leaf, (SymInt, SymFloat, SymBool)):
                # Also an unhashable value, as the side table would refuse it.
                constant_key(leaf, CUTEDSL_CONSTANT_RULES)
            leaves.append(leaf)
            return leaf

        map_constant(val, CUTEDSL_CONSTANT_RULES, collect)
        return leaves

    def specialize_nested(self, variable: Any, val: Any) -> Any:
        # bind_args has walked val with leaves, which refuses a cycle.
        return map_constant(val, CUTEDSL_CONSTANT_RULES, specialize_symbolic)

    def call_HOP(
        self,
        variable: "TraceableCuteDSLKernelWrapper",
        combined_args: dict[str, Any],
        tx: None,
    ) -> None:
        if tx is not None:
            raise AssertionError("tx must be None for TracingCuteDSLHOPifier")
        info = get_host_fn_info(variable.kernel)
        # Runtime scalars as the Python values the launch passes, which are
        # graphable: a numpy scalar is not, and would be a side-table constant.
        hop_args = {
            key: python_scalar(val, info.numeric_params.get(key))
            if info.is_runtime_scalar(key, val)
            else val
            for key, val in combined_args.items()
        }
        graphable_args, constant_args_idx = store_non_graphable_args(
            cutedsl_kernel_side_table, hop_args
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
        # skipping the HOP dispatch. Nothing is symbolic, and the launch walks each
        # constant once, to key it, refusing what bind_args' walk refuses.
        from torch._native.cutedsl.user_kernel import launch_cutedsl_kernel

        launch_cutedsl_kernel(
            self.kernel,
            tracing_cutedsl_hopifier_singleton.bind_call(self, args, kwargs),
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
