"""
Normalization of collective overrides on Python subclasses of ProcessGroup.

C++ callers reach a Python override through the ``PyProcessGroup`` pybind
trampoline, which always passes the canonical arguments of the C++ virtual, e.g.
``allreduce(tensors, opts)``. Python callers can additionally use convenience
overloads bound in ``init.cpp``, e.g. ``pg.allreduce(tensor, op=ReduceOp.MAX)``.
Defining ``allreduce`` on a subclass shadows the whole pybind overload set, so
without help every override would have to parse all of those forms itself.

``ProcessGroup.__init_subclass__`` (installed by :func:`_install`) therefore:

* wraps overrides of methods that have convenience overloads so they always
  receive the canonical ``(..., opts)`` arguments. Non-canonical calls are
  converted by the C++ convenience overloads themselves: they are invoked on a
  private recorder ProcessGroup whose overrides capture the arguments the C++
  virtual is called with, so the defaults live only in ``init.cpp``, and the
  subclass's own trampoline is never re-entered;
* rejects subclasses that override both names of a deprecated alias pair, since
  both names dispatch to the same C++ virtual and ``super()`` from one override
  would re-enter the other.
"""

import functools
import warnings
from collections.abc import Callable
from types import FunctionType
from typing import Any

from torch._C._distributed_c10d import (
    AllgatherOptions,
    AllreduceOptions,
    AllToAllOptions,
    BarrierOptions,
    BroadcastOptions,
    GatherOptions,
    ProcessGroup,
    ReduceOptions,
    ReduceScatterOptions,
    ScatterOptions,
)


__all__: list[str] = []


# Methods with convenience overloads in init.cpp, mapped to the number of
# arguments of the canonical form (including opts) and the type of opts.
_NORMALIZED_METHODS: dict[str, tuple[int, type]] = {
    "allgather": (3, AllgatherOptions),
    "allreduce": (2, AllreduceOptions),
    "all_to_all_single": (5, AllToAllOptions),
    "alltoall_base": (5, AllToAllOptions),
    "barrier": (1, BarrierOptions),
    "broadcast": (2, BroadcastOptions),
    "gather": (3, GatherOptions),
    "reduce": (2, ReduceOptions),
    "reduce_scatter": (3, ReduceScatterOptions),
    "scatter": (3, ScatterOptions),
}

# Pairs of Python names bound to the same C++ virtual, as
# (canonical, deprecated, whether the trampoline falls back to the deprecated
# name when only it is overridden).
_ALIAS_PAIRS: tuple[tuple[str, str, bool], ...] = (
    ("all_gather_single", "_allgather_base", False),
    ("all_gather_single_coalesced", "allgather_into_tensor_coalesced", True),
    ("all_to_all_single", "alltoall_base", True),
    ("gather_single", "gather_into_tensor", False),
    ("reduce_scatter_single", "_reduce_scatter_base", False),
    ("reduce_scatter_single_coalesced", "reduce_scatter_tensor_coalesced", True),
)


class _CapturedArgs(Exception):
    pass


class _ArgsRecorder(ProcessGroup):
    """Captures the canonical arguments a C++ convenience overload produces."""


def _capture(self: ProcessGroup, *args: Any) -> Any:
    raise _CapturedArgs(*args)


# The C++ virtuals that the convenience overloads call.
for _name in (
    "allgather",
    "allreduce",
    "all_to_all_single",
    "barrier",
    "broadcast",
    "gather",
    "reduce",
    "reduce_scatter",
    "scatter",
):
    setattr(_ArgsRecorder, _name, _capture)


_recorder = _ArgsRecorder(0, 1)


def _normalize(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    nargs, opts_type = _NORMALIZED_METHODS[name]
    overloads = getattr(ProcessGroup, name)

    @functools.wraps(fn)
    def wrapper(self: ProcessGroup, *args: Any, **kwargs: Any) -> Any:
        if not kwargs:
            canonical = len(args) == nargs and isinstance(args[-1], opts_type)
        else:
            canonical = (
                len(args) == nargs - 1
                and kwargs.keys() == {"opts"}
                and isinstance(kwargs["opts"], opts_type)
            )
        if not canonical:
            try:
                overloads(_recorder, *args, **kwargs)
            except _CapturedArgs as captured:
                args, kwargs = captured.args, {}
            except TypeError:
                # Matches no pybind overload (e.g. keyword names specific to
                # this override): leave the call to the override as written.
                pass
        return fn(self, *args, **kwargs)

    wrapper._pg_normalized = True  # type: ignore[attr-defined]
    return wrapper


def _init_subclass(cls: type, **kwargs: Any) -> None:
    super(ProcessGroup, cls).__init_subclass__(**kwargs)  # type: ignore[misc]
    namespace = vars(cls)
    for canonical, deprecated, falls_back in _ALIAS_PAIRS:
        if canonical in namespace and deprecated in namespace:
            raise TypeError(
                f"{cls.__qualname__} overrides both ProcessGroup.{canonical} and "
                f"its deprecated alias {deprecated}. Both names dispatch to the "
                f"same C++ method, so super() in one override would re-enter the "
                f"other. Override only {canonical}."
            )
        if deprecated in namespace and not falls_back:
            warnings.warn(
                f"{cls.__qualname__} overrides the deprecated "
                f"ProcessGroup.{deprecated} but not {canonical}. C++ callers and "
                f"torch.distributed dispatch to {canonical}, so this override is "
                f"only reached by direct calls to {deprecated}. Override "
                f"{canonical} instead.",
                FutureWarning,
                stacklevel=2,
            )
    for name in _NORMALIZED_METHODS:
        fn = namespace.get(name)
        if isinstance(fn, FunctionType) and not getattr(fn, "_pg_normalized", False):
            setattr(cls, name, _normalize(name, fn))


def _install() -> None:
    ProcessGroup.__init_subclass__ = classmethod(_init_subclass)  # type: ignore[assignment, method-assign]
