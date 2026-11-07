"""User-defined FlyDSL kernels in PT2: the counterpart of triton_kernel_wrap.py for
``@flyc.jit`` host functions (flydsl >= 0.4).

A FlyDSL kernel (``@flyc.kernel``) is launched by a ``@flyc.jit`` host function. PT2
captures a call to one as ``flydsl_kernel_wrapper_mutation(*, kernel_idx,
constant_args_idx, kwargs)``: the host function and the user's write set are a
side-table record (FlyDSLKernel), the tensors and dynamic scalars are ``kwargs``, and
every other argument is a constant arg in the side table. Functionalization rewrites
the node into ``flydsl_kernel_wrapper_functional``, exactly as for Triton, and the
DSL-agnostic passes treat every DSL the same way through the user-kernel HOP
registry (user_kernel_hop.py).

The stream parameter of the host function (the one annotated ``fx.Stream``) is never
part of the node: the kernel is launched on the current HIP stream when the node
runs.

``@flyc.jit`` returns a JitFunction whose ``func`` is the user's function, with its
annotations (FlyDSL's own pre-rewrite copy, ``_original_func``, has none). This
module never imports flydsl at module level: Dynamo and Inductor import it whether
or not a FlyDSL kernel is used.
"""

import dataclasses
import inspect
import types
import typing
import weakref
from collections.abc import Callable, Mapping
from typing import Any, Literal, Never, TYPE_CHECKING

import sympy

import torch
import torch.utils._pytree as pytree
from torch import SymBool, SymFloat, SymInt, Tensor
from torch._higher_order_ops.reinplace_trigger import ReInplaceTrigger
from torch._higher_order_ops.user_kernel_hop import (
    constant_key,
    ConstantRules,
    KernelSideTable,
    register_user_kernel_hop_impls,
    specialize_symbolic,
    UserKernelHOPifier,
    UserKernelWrapperHOP,
)


if TYPE_CHECKING:
    from torch._guards import Source


###############################################################################
# FlyDSL signatures and values


def _host_function(fn: Any) -> Callable[..., Any]:
    # The user's function (JitFunction.func), which keeps its annotations.
    return fn.func


def _is_stream(annotation: object) -> bool:
    # How FlyDSL itself recognizes its stream parameter (fx.Stream).
    return isinstance(annotation, type) and bool(
        getattr(annotation, "_is_stream_param", False)
    )


def _is_constexpr(
    annotation: object, name: str, index: int, fn: Callable[..., Any]
) -> bool:
    from flydsl.expr.typing import Constexpr

    # FlyDSL also bakes the values of type / Type / type[T] parameters into the
    # compiled kernel (the origin of typing.Type and of type[T] is type).
    return (
        Constexpr.is_constexpr_annotation(annotation)
        or annotation is type
        or typing.get_origin(annotation) is type
    )


def _is_numeric(annotation: object) -> bool:
    from flydsl.expr.numeric import NumericMeta

    return isinstance(annotation, NumericMeta)


def _is_flydsl_type(cls: type) -> bool:
    """Whether ``cls`` comes from the flydsl package (not a module merely named like
    it)."""
    module = getattr(cls, "__module__", "") or ""
    return module == "flydsl" or module.startswith("flydsl.")


def _numeric_value(value: Any) -> Any:
    if _is_flydsl_type(type(value)):
        from flydsl.expr.numeric import Boolean, Numeric

        if isinstance(value, Boolean):
            return bool(value.value)  # .value is the int 0 or 1
        if isinstance(value, Numeric) and isinstance(value.value, (bool, int, float)):
            return value.value
    return value


def _leaf_key(value: object) -> object:
    # FlyDSL keys a Constexpr lambda by its code, constants and defaults
    # (Constexpr.value_signature), also inside a Constexpr tuple, so equal lambdas
    # made per call share one compile and one side-table entry. Anything else (and a
    # lambda FlyDSL refuses) is keyed as constant_key keys it (None).
    if isinstance(value, types.FunctionType) and value.__name__ == "<lambda>":
        from flydsl.expr.typing import Constexpr

        try:
            return ("flydsl-lambda", Constexpr.value_signature(value))
        except TypeError:
            pass
    return None


class _FlyDSLConstantRules(ConstantRules):
    def is_dsl_value(self, value: object) -> bool:
        return _is_flydsl_type(type(value))

    def scalar_value(self, value: object) -> bool | int | float | None:
        # A FlyDSL Numeric, such as fx.Int32(5), is keyed by its value; another FlyDSL
        # value with a .value is not (a Basis, fx.E(0), has modes besides it).
        if not self.is_dsl_value(value):
            return None
        from flydsl.expr.numeric import Numeric

        inner = value.value if isinstance(value, Numeric) else None
        return inner if isinstance(inner, (bool, int, float)) else None

    def leaf_key(self, value: object) -> object:
        return _leaf_key(value)


FLYDSL_CONSTANT_RULES = _FlyDSLConstantRules()


def _constant_key(value: object) -> object:
    """The key of a FlyDSL constant argument, in the side table and the compile
    caches: constant_key with FlyDSL's rules (Numerics by value, Constexpr lambdas
    by their code, at any depth)."""
    return constant_key(value, FLYDSL_CONSTANT_RULES)


def _is_runtime_object(cls: type) -> bool:
    """Objects PT2 makes itself: FlyDSL tensor and pointer arguments (it passes torch
    tensors) and streams (it passes the current stream). Other DSL values, such as
    Numeric values, are ordinary constants."""
    module = getattr(cls, "__module__", "") or ""
    return (
        module.startswith("flydsl.compiler.jit_argument")
        or _is_stream(cls)
        or (isinstance(cls, type) and issubclass(cls, torch.cuda.Stream))
    )


###############################################################################
# Kernel Side Table


# The parameter names the kernel writes; "unknown": every tensor argument.
MutatesArgs = tuple[str, ...] | Literal["unknown"]


@dataclasses.dataclass(frozen=True)
class FlyDSLKernel:
    """Side-table entry for a FlyDSL host function: plays the part a JITFunction
    plays for Triton. The write set is part of the record, so one host function
    wrapped with two different lists gets two entries."""

    fn: Callable[..., Any]
    mutates_args: MutatesArgs

    @property
    def name(self) -> str:
        return getattr(_host_function(self.fn), "__name__", "flydsl_kernel")


# Unlike Triton's table, torch._dynamo.reset() leaves this one alone: graphs and
# exported programs captured before a reset keep resolving their indices. Constant
# args are keyed with FlyDSL's rules (_constant_key).
flydsl_kernel_side_table: KernelSideTable[FlyDSLKernel] = KernelSideTable(
    constant_rules=FLYDSL_CONSTANT_RULES
)


def get_kernel(kernel_idx: int) -> FlyDSLKernel:
    return flydsl_kernel_side_table.get_kernel(kernel_idx)


###############################################################################
# Host function introspection


@dataclasses.dataclass(frozen=True)
class FlyDSLHostFnInfo:
    """What PT2 needs from a host function's signature.

    Arguments fall into four classes: the stream (supplied at launch); tensors and
    runtime scalars, which are kernel arguments; and constants, which FlyDSL bakes
    into the compiled kernel: Constexpr parameters and values that are neither
    tensors nor scalars (None, objects). A runtime scalar is a value for a parameter
    annotated with a FlyDSL Numeric type such as fx.Int32, or a Python
    int/float/bool for an unannotated parameter, which FlyDSL converts to
    Int32/Float32/Boolean.
    """

    params: tuple[str, ...]
    stream_param: str
    constexpr_params: frozenset[str]
    numeric_params: Mapping[str, type]
    defaults: Mapping[str, Any]
    # The keyword-only parameters (after * in the signature): a call passes them by
    # name, and so does PT2's compile.
    kwonly_params: frozenset[str]

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
    """A runtime scalar as the Python value FlyDSL expects: a FlyDSL Numeric value as
    its .value, a subclass of bool/int/float (an IntEnum, a numpy float) as the
    builtin, which FlyDSL's argument adapters require exactly."""
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    return _numeric_value(value)


_host_fn_infos: "weakref.WeakKeyDictionary[Callable[..., Any], FlyDSLHostFnInfo]" = (
    weakref.WeakKeyDictionary()
)


def get_host_fn_info(fn: Callable[..., Any]) -> FlyDSLHostFnInfo:
    """Parse a ``@flyc.jit`` host function's signature; raises ValueError for the
    ones PT2 cannot launch."""
    info = _host_fn_infos.get(fn)
    if info is not None:
        return info

    inner = _host_function(fn)
    name = getattr(inner, "__qualname__", repr(inner))
    params, constexpr, defaults, kwonly = [], set(), {}, set()
    numeric: dict[str, type] = {}
    streams = []
    for i, p in enumerate(inspect.signature(inner).parameters.values()):
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            raise ValueError(f"{name}: *args and **kwargs parameters are not supported")
        if p.kind == p.POSITIONAL_ONLY:
            # PT2 binds and compiles host functions by parameter name.
            raise ValueError(
                f"{name}: positional-only parameters are not supported ({p.name!r})"
            )
        if i == 0 and p.name == "self":
            raise ValueError(
                f"{name}: @flyc.jit methods are not supported (FlyDSL binds the first "
                f"parameter, {p.name!r}, to the object); wrap a @flyc.jit function "
                f"without {p.name!r}"
            )
        ann = p.annotation
        if isinstance(ann, str):
            raise ValueError(
                f"{name}: annotation of {p.name!r} is a string; FlyDSL needs real "
                "annotations (remove `from __future__ import annotations`)"
            )
        params.append(p.name)
        if p.kind == p.KEYWORD_ONLY:
            kwonly.add(p.name)
        if p.default is not p.empty:
            if isinstance(p.default, Tensor):
                # PT2 passes tensors as graph arguments; a default one would be a
                # constant of the compiled kernel, which constant keys refuse.
                raise ValueError(
                    f"{name}: the default of {p.name!r} is a tensor; pass each tensor "
                    "as its own argument"
                )
            if ann is p.empty:
                # As written: FlyDSL types a Numeric default (fx.Int64(5)) of an
                # unannotated parameter from its class, and a Python scalar from its
                # value (Int32 for an int), which is converted at launch.
                defaults[p.name] = p.default
            else:
                # A Numeric default (fx.Float32(0.5)) as its Python value, which
                # generated code can print.
                defaults[p.name] = python_scalar(p.default)
        if _is_stream(ann):
            streams.append(p.name)
        elif _is_constexpr(ann, p.name, i, inner):
            constexpr.add(p.name)
        elif _is_numeric(ann):
            numeric[p.name] = ann
    if len(streams) != 1:
        raise ValueError(
            f"{name}: expected exactly one parameter annotated fx.Stream (PT2 "
            f"launches the kernel on the current HIP stream), found {streams}"
        )
    info = FlyDSLHostFnInfo(
        params=tuple(params),
        stream_param=streams[0],
        constexpr_params=frozenset(constexpr),
        numeric_params=numeric,
        defaults=defaults,
        kwonly_params=frozenset(kwonly),
    )
    _host_fn_infos[fn] = info
    return info


###############################################################################
# FlyDSL Kernel Wrappers


# Used for wrapping a FlyDSL host function
class FlyDSLKernelWrapperMutation(UserKernelWrapperHOP):
    def __init__(self) -> None:
        # Not cacheable: the generated code binds the host function in-process.
        super().__init__("flydsl_kernel_wrapper_mutation", cacheable=False)


flydsl_kernel_wrapper_mutation = FlyDSLKernelWrapperMutation()


# Used for wrapping a FlyDSL host function in a functional manner
class FlyDSLKernelWrapperFunctional(UserKernelWrapperHOP):
    def __init__(self) -> None:
        super().__init__("flydsl_kernel_wrapper_functional", cacheable=False)


flydsl_kernel_wrapper_functional = FlyDSLKernelWrapperFunctional()


def flydsl_kernel_wrapper_mutation_dense(
    *,
    kernel_idx: int,
    constant_args_idx: int,
    kwargs: dict[str, Any],
) -> None:
    from torch._inductor.runtime.flydsl_user_kernel import launch_flydsl_kernel

    kernel = flydsl_kernel_side_table.get_kernel(kernel_idx)
    launch_flydsl_kernel(
        kernel.fn, flydsl_kernel_side_table.kernel_kwargs(constant_args_idx, kwargs)
    )


def get_flydsl_mutated_tensors(
    kernel_idx: int,
    constant_args_idx: int,
    kwargs: dict[str, Any],
) -> list[str]:
    """Names of the tensor kwargs the kernel writes (the user's mutates_args): the
    mirror of Triton's get_mutated_tensors."""
    kernel = flydsl_kernel_side_table.get_kernel(kernel_idx)
    if kernel.mutates_args == "unknown":
        names: tuple[str, ...] = tuple(kwargs)
    else:
        names = kernel.mutates_args
    return [name for name in names if isinstance(kwargs.get(name), Tensor)]


flydsl_kernel_hop_info = register_user_kernel_hop_impls(
    flydsl_kernel_wrapper_mutation,
    flydsl_kernel_wrapper_functional,
    dsl="flydsl",
    display_name="FlyDSL",
    side_table=flydsl_kernel_side_table,
    launch=flydsl_kernel_wrapper_mutation_dense,
    get_mutated_tensors=lambda hop_kwargs, kwargs: get_flydsl_mutated_tensors(
        hop_kwargs["kernel_idx"], hop_kwargs["constant_args_idx"], kwargs
    ),
    # A FlyDSL kernel is compiled for the strides its tensors have when it is
    # called, which may be what its indexing assumes: Inductor keeps them.
    get_layout_constraint=lambda: "needs_fixed_stride_order",
    reinplace_trigger=ReInplaceTrigger.FLYDSL_OPS,
    # The generated code binds the host function in-process.
    serializable=False,
    refuse_aliased_writes=True,
)


###############################################################################
# The "FlyDSLHOPifier": a class that transforms a call to a FlyDSL host function
# into a call to the flydsl_kernel_wrapper_mutation HOP.


def normalize_mutates_args(mutates_args: object) -> MutatesArgs:
    """``mutates_args`` as wrap_flydsl received it, checked: a sequence of names
    becomes a tuple (in order, without duplicates). None is refused too."""
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


class FlyDSLHOPifier(UserKernelHOPifier):
    """Orchestrator for converting a call to a FlyDSL host function into a call to
    the flydsl_kernel_wrapper_mutation HOP: the counterpart of TritonHOPifier.

    It has two main use cases.

    1. When Dynamo sees a wrapped host function, it becomes a FlyDSLKernelVariable,
    whose calls DynamoFlyDSLHOPifier turns into a call to the HOP.

    2. For make_fx and non-strict export, wrap_flydsl returns a
    TraceableFlyDSLKernelWrapper, whose calls TracingFlyDSLHOPifier turns into a
    call to the HOP (which can then be traced).

    A FlyDSL launch has no grid, autotuner or heuristics, so what is left is binding
    the call to the host function's signature: the stream parameter is supplied at
    launch and must not be passed; keyword-only parameters are passed by name;
    Constexpr arguments are baked into the compiled kernel, so symbolic values for
    them are specialized; tensors and scalars are kernel arguments (see
    FlyDSLHostFnInfo).
    """

    hop_info = flydsl_kernel_hop_info

    def is_tensor(self, val: Any) -> bool:
        raise NotImplementedError("abstract method")

    def python_type(self, val: Any) -> type:
        raise NotImplementedError("abstract method")

    def leaves(self, val: Any) -> list[Any]:
        """The values inside a list/tuple argument (the value itself otherwise)."""
        raise NotImplementedError("abstract method")

    def specialize_nested(self, variable: Any, val: Any) -> Any:
        """val with every symbolic value in it (or itself) specialized."""
        raise NotImplementedError("abstract method")

    def call_HOP(  # type: ignore[no-untyped-def]
        self,
        variable,
        combined_args: dict[str, Any],
        tx,
    ) -> Any:
        raise NotImplementedError("abstract method")

    def host_fn_info(self, kernel: Callable[..., Any]) -> FlyDSLHostFnInfo:
        try:
            return get_host_fn_info(kernel)
        except ValueError as e:
            self.raise_unsupported(str(e))

    def normalized_mutates_args(self, mutates_args: object) -> MutatesArgs:
        """``mutates_args`` as wrap_flydsl received it, checked (see
        normalize_mutates_args). None is invalid too: init_variable takes it for a
        host function that was not wrapped."""
        try:
            return normalize_mutates_args(mutates_args)
        except ValueError as e:
            self.raise_unsupported(str(e))

    def init_variable(
        self,
        variable: Any,
        kernel: Callable[..., Any],
        mutates_args: object,
        kernel_idx: int | None,
    ) -> None:
        """Record the host function on ``variable`` and, once the write set is known,
        add it to the side table. ``mutates_args`` is None for a host function that
        has not been wrapped (Dynamo met it raw): calling that is unsupported."""
        variable.kernel = kernel
        variable.mutates_args = None
        variable.kernel_idx = None
        if mutates_args is None:
            return
        normalized = self.normalized_mutates_args(mutates_args)
        info = self.host_fn_info(kernel)
        if not isinstance(normalized, str):
            scalar_params = (
                {info.stream_param} | info.constexpr_params | set(info.numeric_params)
            )
            for name in normalized:
                if name not in info.params or name in scalar_params:
                    self.raise_unsupported(
                        f"mutates_args: {name!r} is not a tensor parameter of "
                        f"{FlyDSLKernel(kernel, ()).name} (parameters: {info.params})"
                    )
        variable.mutates_args = normalized
        self.record_kernel(variable, FlyDSLKernel(kernel, normalized), kernel_idx)

    def bind_args(
        self,
        variable: Any,
        args: tuple[Any, ...] | list[Any],
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Bind a call to the host function's parameters, minus the stream."""
        kernel = variable.kernel
        name = FlyDSLKernel(kernel, ()).name
        if variable.kernel_idx is None:
            self.raise_unsupported(
                f"Calling the FlyDSL host function {name} directly is not supported: "
                "wrap it with torch._library.flydsl.wrap_flydsl(fn) and call the "
                "wrapper with torch.Tensor arguments and no stream"
            )
        info = self.host_fn_info(kernel)
        params = [p for p in info.params if p != info.stream_param]
        positional = [p for p in params if p not in info.kwonly_params]
        if len(args) > len(positional):
            if len(positional) < len(params):
                self.raise_unsupported(
                    f"{name}() takes {len(positional)} positional arguments without "
                    f"the stream but {len(args)} were given"
                )
            self.raise_unsupported(
                f"{name}() takes {len(params)} arguments without the stream but "
                f"{len(args)} were given"
            )
        combined_args = dict(zip(positional, args))
        for key, val in kwargs.items():
            if key == info.stream_param:
                self.raise_unsupported(
                    f"{name}(): do not pass {key!r}; the kernel is launched on the "
                    "current HIP stream"
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
                if _is_runtime_object(self.python_type(leaf)):
                    self.raise_unsupported(
                        f"{name}(): pass torch.Tensor and Python values, not "
                        f"{self.python_type(leaf).__name__} ({key!r}); PT2 converts "
                        "tensors and supplies the stream"
                    )
            if key in info.constexpr_params or not (
                len(leaves) == 1 and leaves[0] is val
            ):
                # FlyDSL bakes a Constexpr argument into the compiled kernel, and a
                # container can only be a side-table constant, so symbolic values in
                # either are specialized (see [Note: Specialize tl.constexpr args in
                # user-defined triton kernels]).
                combined_args[key] = self.specialize_nested(variable, val)
        return combined_args

    def call_flydsl_kernel(
        self,
        variable: Any,
        args: tuple[Any, ...] | list[Any],
        kwargs: dict[str, Any],
        tx: Any,
    ) -> Any:
        return self.call_HOP(variable, self.bind_args(variable, args, kwargs), tx)


###############################################################################
# Helpers for the wrap_flydsl API, which makes a FlyDSL host function traceable
# into a graph via make_fx or non-strict export


class TracingFlyDSLHOPifier(FlyDSLHOPifier):
    def raise_unsupported(self, msg: str) -> Never:
        raise RuntimeError(msg)

    def get_value(self, val: Any) -> Any:
        return val

    def is_tensor(self, val: Any) -> bool:
        return isinstance(val, Tensor)

    def python_type(self, val: Any) -> type:
        return type(val)

    def leaves(self, val: Any) -> list[Any]:
        return pytree.tree_leaves(val)

    def specialize_nested(self, variable: Any, val: Any) -> Any:
        return pytree.tree_map(variable.specialize_symbolic, val)

    def call_HOP(
        self,
        variable: "TraceableFlyDSLKernelWrapper",
        combined_args: dict[str, Any],
        tx: None,
    ) -> None:
        if tx is not None:
            raise AssertionError("tx must be None for TracingFlyDSLHOPifier")
        return self.call_mutation_hop(variable.kernel_idx, combined_args)


tracing_flydsl_hopifier_singleton = TracingFlyDSLHOPifier()


class TraceableFlyDSLKernelWrapper:
    """What wrap_flydsl returns: calling it launches the host function eagerly, or
    records a flydsl_kernel_wrapper_mutation node under make_fx / non-strict export.
    Dynamo traces it as a FlyDSLKernelVariable."""

    kernel: Callable[..., Any]
    mutates_args: MutatesArgs | None
    kernel_idx: int | None
    kernel_source: "Source | None"

    def __init__(
        self,
        kernel: Callable[..., Any],
        mutates_args: object,
        kernel_idx: int | None = None,
        kernel_source: "Source | None" = None,
    ) -> None:
        self.kernel_source = kernel_source
        tracing_flydsl_hopifier_singleton.init_variable(
            self, kernel, mutates_args, kernel_idx
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        from torch._library.triton import is_wrap_triton_enabled

        if is_wrap_triton_enabled():
            return tracing_flydsl_hopifier_singleton.call_flydsl_kernel(
                self, args, kwargs, None
            )
        # Eager fast path (e.g. inside a triton_op's eager backend): launch directly,
        # skipping the HOP dispatch.
        from torch._inductor.runtime.flydsl_user_kernel import launch_flydsl_kernel

        launch_flydsl_kernel(
            self.kernel,
            tracing_flydsl_hopifier_singleton.bind_args(self, args, kwargs),
        )

    def specialize_symbolic(self, arg: Any) -> Any:
        return specialize_symbolic(arg)
