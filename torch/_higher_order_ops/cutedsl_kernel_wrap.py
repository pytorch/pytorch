"""User-defined CuTeDSL kernels in PT2: the counterpart of triton_kernel_wrap.py."""

import dataclasses
import inspect
import sys
import weakref
from collections.abc import Callable, Collection, Mapping
from typing import Any

import sympy

from torch import SymBool, SymFloat, SymInt, Tensor
from torch._higher_order_ops.user_kernel_hop import ConstantRules


def cutedsl_kernel_name(fn: Callable[..., Any]) -> str:
    return getattr(inspect.unwrap(fn), "__name__", "cutedsl_kernel")


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
                # nvidia-cutlass-dsl 4.8 refuses to compile an int beyond int64 and
                # uint64 for a float parameter (OverflowError); 4.4 casts it itself.
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
