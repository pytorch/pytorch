"""DSL-agnostic pieces of the user-kernel HOPs.

A user-defined kernel (a ``@triton.jit`` kernel, a ``@cute.jit`` host function) is
captured as a pair of HOPs: a mutation HOP that launches the kernel and writes its
arguments in place, and a functional HOP that functionalization rewrites it into
(clones of the written arguments, then the launch on the clones). Everything here is
shared by every DSL's pair; each DSL's module (triton_kernel_wrap.py,
cutedsl_kernel_wrap.py) adds the parts that depend on the kernel object: how to launch
it, how to find its write set, and how to capture a call to it (the HOPifier).
"""

import copy
import dataclasses
import threading
import types
from collections.abc import Callable, Mapping
from typing import Any, Generic, TYPE_CHECKING, TypeVar
from typing_extensions import Never

import torch.fx as fx
import torch.utils._pytree as pytree
from torch import Tensor
from torch._C import _dispatch_keys, DispatchKey
from torch._higher_order_ops.utils import redirect_to_mode
from torch._ops import HigherOrderOperator
from torch._prims_common import (
    clone_preserve_strides,
    is_non_overlapping_and_dense_or_false,
)
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)
from torch.fx.experimental.symbolic_shapes import guard_or_false
from torch.utils.checkpoint import _CachedTorchDispatchMode, _CachingTorchDispatchMode


if TYPE_CHECKING:
    from torch._dynamo.utils import ReInplaceTrigger
    from torch._subclasses.functional_tensor import BaseFunctionalizeAPI


KernelT = TypeVar("KernelT")


###############################################################################
# Kernel Side Table


def _has_slot_state(cls: type) -> bool:
    """Whether cls or a base declares __slots__ other than __dict__ and __weakref__.
    Like copyreg._slotnames, but without caching the answer on the class: Dynamo
    replays that store after the graph, outside copyreg's try/except, and it fails
    for built-in types such as types.SimpleNamespace."""
    for c in cls.__mro__:
        slots = c.__dict__.get("__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        if any(s not in ("__dict__", "__weakref__") for s in slots):
            return True
    return False


def _object_state(value: object) -> dict[str, object] | None:
    """The attributes constant_key keys an object by, and _snapshot copies: the
    fields of a dataclass instance, or the __dict__ of a plain object (one with no
    __slots__ attributes that is not callable or a module, of a class that does not
    hash by value), such as a mutable config object. None for any other value,
    cutlass values included.

    Limitation: any other object with __slots__ attributes, or a callable one (a
    functor), is keyed as itself, by identity unless its class defines equality and
    hashing: mutating it between launches reuses the kernel compiled for its old
    contents, and a new object with the same contents is compiled again."""
    cls = type(value)
    if (getattr(cls, "__module__", "") or "").startswith("cutlass"):
        return None
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: getattr(value, f.name) for f in dataclasses.fields(value)}
    if (
        (cls.__hash__ is None or cls.__hash__ is object.__hash__)
        and hasattr(value, "__dict__")
        and not callable(value)
        and not isinstance(value, types.ModuleType)
        and not _has_slot_state(cls)
    ):
        return vars(value)
    return None


def constant_key(value: object) -> object:
    """A hashable key for a constant kernel argument, equal for two values only when
    they are interchangeable: lists, tuples, dicts and objects (see _object_state)
    by type and content, floats by hex() (-0.0 is not 0.0, and every NaN is the
    same), a cutlass Numeric by its type and Python value, other values by their own
    equality. Any other unhashable value is refused."""
    if isinstance(value, (tuple, list)):
        return (type(value), tuple(constant_key(v) for v in value))
    if isinstance(value, dict):
        return (
            type(value),
            tuple((constant_key(k), constant_key(v)) for k, v in value.items()),
        )
    if (getattr(type(value), "__module__", "") or "").startswith("cutlass"):
        inner = getattr(value, "value", None)
        if isinstance(inner, (bool, int, float)):
            return (type(value), constant_key(inner))
    if isinstance(value, float):
        return (type(value), value.hex())
    state = _object_state(value)
    if state is not None:
        return (type(value), constant_key(state))
    try:
        hash(value)
    except TypeError:
        raise TypeError(
            f"a constant kernel argument of type {type(value).__name__} is "
            "unhashable: pass a hashable value, or a list, tuple, dict, dataclass or "
            "plain object of them"
        ) from None
    return (type(value), value)


def _snapshot(value: object) -> object:
    """``value`` with everything constant_key keys by content copied, so that the
    caller mutating it later changes neither the stored value nor its key."""
    if isinstance(value, tuple):
        items = [_snapshot(v) for v in value]
        if all(a is b for a, b in zip(items, value)):
            return value
        # A namedtuple is rebuilt with _make, any other tuple from the items.
        return getattr(type(value), "_make", type(value))(items)
    if isinstance(value, list):
        new_list = copy.copy(value)
        new_list[:] = [_snapshot(v) for v in value]
        return new_list
    if isinstance(value, dict):
        new_dict = copy.copy(value)
        new_dict.clear()
        new_dict.update((_snapshot(k), _snapshot(v)) for k, v in value.items())
        return new_dict
    state = _object_state(value)
    if state is None:
        return value
    new = copy.copy(value)
    for name, v in state.items():
        object.__setattr__(new, name, _snapshot(v))
    return new


# We cannot put user kernels into the FX graph as the graph nodes
# do not support arbitrary functions.
# Use a side table.
# We use two dicts so that fetching both the kernel and id are O(1)
class KernelSideTable(Generic[KernelT]):
    def __init__(self, dedupe_constant_args: bool = False) -> None:
        # Per instance, so that each DSL's table has its own index space.
        self.id_to_kernel: dict[int, KernelT] = {}
        self.kernel_to_id: dict[KernelT, int] = {}
        self.constant_args: dict[int, dict[str, object]] = {}
        self.lock = threading.Lock()
        # With dedupe_constant_args, equal constant args (see constant_key) share one
        # index, so a table that is never reset does not grow with every eager launch,
        # and each entry is a snapshot of the args it was added with.
        self._constant_args_index: dict[object, int] | None = (
            {} if dedupe_constant_args else None
        )

    # Returns index on the table
    def add_kernel(self, kernel: KernelT) -> int:
        with self.lock:
            if kernel in self.kernel_to_id:
                return self.kernel_to_id[kernel]

            idx = len(self.id_to_kernel)
            self.id_to_kernel[idx] = kernel
            self.kernel_to_id[kernel] = idx
            return idx

    # Returns the kernel at the given index
    def get_kernel(self, idx: int) -> KernelT:
        # No need to lock here as fetching from dict is atomic
        if idx not in self.id_to_kernel:
            raise AssertionError(f"Kernel index {idx} not found in id_to_kernel")
        return self.id_to_kernel[idx]

    # Not every constant arg can be added to the graph. Use this side table
    # for constant args.
    def add_constant_args(self, args: dict[str, object]) -> int:
        with self.lock:
            key = None
            if self._constant_args_index is not None:
                key = tuple((k, constant_key(v)) for k, v in sorted(args.items()))
                if key in self._constant_args_index:
                    return self._constant_args_index[key]
                args = {k: _snapshot(v) for k, v in args.items()}
            idx = len(self.constant_args)
            self.constant_args[idx] = args
            if key is not None:
                self._constant_args_index[key] = idx  # type: ignore[index]
            return idx

    # Returns the constant args
    def get_constant_args(self, idx: int) -> dict[str, object]:
        # No need to lock here as fetching from dict is atomic
        if idx not in self.constant_args:
            raise AssertionError(
                f"Constant args index {idx} not found in constant_args"
            )
        return self.constant_args[idx]

    # Resets the table (only meant to be used in unit tests)
    # This is only safe assuming single threaded execution
    def reset_table(self) -> None:
        self.id_to_kernel = {}
        self.kernel_to_id = {}
        self.constant_args = {}
        if self._constant_args_index is not None:
            self._constant_args_index = {}


def store_non_graphable_args(
    side_table: KernelSideTable[Any], combined_args: dict[str, Any]
) -> tuple[dict[str, Any], int]:
    """
    Some args cannot be stored in the FX graph.
    Put them in the side table.
    """

    def is_graphable(val: object) -> bool:
        return isinstance(val, (fx.node.base_types, fx.Node))

    non_graphable_args = {k: v for k, v in combined_args.items() if not is_graphable(v)}
    graphable_args = {k: v for k, v in combined_args.items() if is_graphable(v)}

    constant_args_idx = side_table.add_constant_args(non_graphable_args)

    return graphable_args, constant_args_idx


###############################################################################
# HOP base class and the dispatch rules shared by every user-kernel HOP pair


class UserKernelWrapperHOP(HigherOrderOperator):
    """Base class of the user-kernel mutation and functional HOPs.

    Their tensors live inside the ``kwargs`` (and ``tensor_bases``) dicts, which the
    default overloaded-args scan does not look into.
    """

    def _get_overloaded_args(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[Tensor, ...]:
        overloaded_args = list(super()._get_overloaded_args(args, kwargs))
        for tensor_dict_name in ("kwargs", "tensor_bases"):
            tensor_dict = kwargs.get(tensor_dict_name)
            if isinstance(tensor_dict, dict):
                overloaded_args.extend(
                    arg
                    for arg in tensor_dict.values()
                    if isinstance(arg, Tensor) and _dispatch_keys(arg).has("Python")
                )
        return tuple(overloaded_args)


def trace_user_kernel_wrapper(
    proxy_mode: ProxyTorchDispatchMode,
    func_overload: Callable[..., Any],
    node_args: dict[str, Any],
) -> dict[str, Any] | None:
    with disable_proxy_modes_tracing():
        out = func_overload(**node_args)

    proxy_args = pytree.tree_map(
        proxy_mode.tracer.unwrap_proxy,  # type: ignore[union-attr]
        node_args,
    )
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function",
        func_overload,
        (),
        proxy_args,
        name=func_overload.__name__ + "_proxy",
    )

    ret = track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)
    return ret


def clone_mutated_arg(
    key: str, val: Tensor, tensor_bases: dict[str, Tensor] | None
) -> Tensor:
    """Clone a mutated pointer arg.

    clone_preserve_strides allocates storage_offset + span elements and reads them
    out of val's storage. Whenever that is more than val's own numel -- a non-zero
    storage_offset, or gaps between val's elements -- part of the read lies outside
    val, which a backend that realizes val as a buffer sized to val cannot supply.
    """
    if is_non_overlapping_and_dense_or_false(val):
        if guard_or_false(val.storage_offset() == 0):
            # storage_offset + span is exactly val's numel, so nothing outside val
            # is read and this stays the cheapest thing that works.
            return clone_preserve_strides(val)
        # clone() reproduces the strides of a dense tensor and lands it at offset 0,
        # so the clone spans exactly val's own elements. That is all a user kernel
        # needs: it only ever sees a data pointer plus the strides passed to it.
        return val.clone()
    # Gaps have to be materialized as well, so the storage val sits in gets copied.
    return clone_preserve_strides(
        val, base=tensor_bases.get(key) if tensor_bases else None
    )


def functionalize_mutation_hop(
    ctx: "BaseFunctionalizeAPI",
    functional_hop: HigherOrderOperator,
    get_mutated_tensors: Callable[[dict[str, Any]], list[str]],
    *,
    hop_kwargs: dict[str, Any],
    kwargs: dict[str, Any],
) -> None:
    """Functionalize rule of a mutation HOP: call the functional HOP on clones of the
    written arguments (``get_mutated_tensors`` names them, given the unwrapped kwargs),
    then write its outputs back into the functional wrappers of the inputs.

    ``hop_kwargs`` holds the HOP's arguments other than ``kwargs``; they are passed
    to the functional HOP unchanged.
    """
    unwrapped_kwargs = ctx.unwrap_tensors(kwargs)  # type: ignore[arg-type]
    tensors_to_clone = get_mutated_tensors(unwrapped_kwargs)
    # A mutated arg that is a view has to be cloned through its base, and the base is
    # only a tracked value out here -- reading _base inside the functional impl yields
    # an untracked tensor that the tracer would lift to a constant. See
    # clone_preserve_strides for why cloning the view itself is not graph-safe.
    tensor_bases = {}
    for key in tensors_to_clone:
        tensor = kwargs[key]
        base = tensor._base if isinstance(tensor, Tensor) else None
        if base is not None:
            tensor_bases[key] = ctx.unwrap_tensors(base)
    with ctx.redispatch_to_next():
        functional_kwargs: dict[str, Any] = {
            **hop_kwargs,
            "kwargs": unwrapped_kwargs,
            "tensors_to_clone": tensors_to_clone,
        }
        if tensor_bases:
            functional_kwargs["tensor_bases"] = tensor_bases
        unwrapped_outputs = functional_hop(**functional_kwargs)

    if not set(unwrapped_outputs.keys()).issubset(set(kwargs.keys())):
        raise AssertionError(
            f"Output keys {set(unwrapped_outputs.keys())} not subset of input keys {set(kwargs.keys())}"
        )
    for key, output_arg in unwrapped_outputs.items():
        if not isinstance(output_arg, Tensor):
            continue
        input_arg = kwargs[key]
        if not isinstance(input_arg, Tensor):
            raise AssertionError(
                f"Expected input_arg for key {key} to be a Tensor, got {type(input_arg)}"
            )

        ctx.replace(input_arg, output_arg)
        # indicate that above replace is hidden from autograd
        ctx.mark_mutation_hidden_from_autograd(input_arg)
        ctx.commit_update(input_arg)
        ctx.sync(input_arg)


def functional_hop_dense(
    mutation_hop: HigherOrderOperator,
    *,
    hop_kwargs: dict[str, Any],
    kwargs: dict[str, Any],
    tensors_to_clone: list[str],
    tensor_bases: dict[str, Tensor] | None,
) -> dict[str, Any]:
    """Dense impl of a functional HOP: launch the mutation HOP on clones of the
    arguments in ``tensors_to_clone`` and return those clones."""
    # TODO(oulgen): For performance reasons, we want to ensure that these
    # `clone_preserve_strides` calls are never executed at runtime
    # (inductor should always optimize them away).
    # Requires https://github.com/pytorch/pytorch/issues/109240
    kwargs = {
        key: (
            clone_mutated_arg(key, val, tensor_bases)
            if key in tensors_to_clone
            else val
        )
        for key, val in kwargs.items()
    }
    mutation_hop(**hop_kwargs, kwargs=kwargs)
    return {key: val for key, val in kwargs.items() if key in tensors_to_clone}


def functional_hop_fake(
    *,
    kwargs: dict[str, Any],
    tensors_to_clone: list[str],
    tensor_bases: dict[str, Tensor] | None,
) -> dict[str, Any]:
    # TODO(oulgen): For performance reasons, we want to ensure that these
    # `clone_preserve_strides` calls are never executed at runtime
    # (inductor should always optimize them away).
    # Requires https://github.com/pytorch/pytorch/issues/109240
    return {
        key: clone_mutated_arg(key, val, tensor_bases)
        for key, val in kwargs.items()
        if key in tensors_to_clone
    }


def functionalize_functional_hop(
    ctx: "BaseFunctionalizeAPI",
    functional_hop: HigherOrderOperator,
    *,
    hop_kwargs: dict[str, Any],
    kwargs: dict[str, Any],
    tensors_to_clone: list[str],
    tensor_bases: dict[str, Tensor] | None,
) -> dict[str, Any]:
    unwrapped_kwargs = ctx.unwrap_tensors(kwargs)  # type: ignore[arg-type]
    unwrapped_tensor_bases = ctx.unwrap_tensors(tensor_bases)  # type: ignore[arg-type]
    with ctx.redispatch_to_next():
        functional_kwargs: dict[str, Any] = {
            **hop_kwargs,
            "kwargs": unwrapped_kwargs,
            "tensors_to_clone": tensors_to_clone,
        }
        if unwrapped_tensor_bases:
            functional_kwargs["tensor_bases"] = unwrapped_tensor_bases
        outputs = functional_hop(**functional_kwargs)
        return ctx.wrap_tensors(outputs)  # type: ignore[return-value,arg-type]


def register_user_kernel_hop_dispatch(
    mutation_hop: HigherOrderOperator, functional_hop: HigherOrderOperator
) -> None:
    """Dispatch keys a user-kernel HOP pair falls through, plus SAC support."""
    for hop in (mutation_hop, functional_hop):
        hop.fallthrough(DispatchKey.PythonDispatcher)  # type: ignore[attr-defined]
        hop.fallthrough(DispatchKey.PythonTLSSnapshot)  # type: ignore[attr-defined]
        hop.fallthrough(DispatchKey.ADInplaceOrView)
        hop.fallthrough(DispatchKey.BackendSelect)
        hop.fallthrough(DispatchKey.AutocastCPU)  # type: ignore[attr-defined]
        hop.fallthrough(DispatchKey.AutocastCUDA)  # type: ignore[attr-defined]
        hop.fallthrough(DispatchKey.AutogradCUDA)
        hop.fallthrough(DispatchKey.AutogradCPU)

    # Adds SAC support for user kernels
    redirect_to_mode(mutation_hop, _CachingTorchDispatchMode)
    redirect_to_mode(mutation_hop, _CachedTorchDispatchMode)


###############################################################################
# Registry: what the DSL-agnostic passes need to know about each DSL's HOP pair


@dataclasses.dataclass(frozen=True)
class UserKernelHOPInfo:
    """One DSL's user-kernel HOP pair, as seen by the passes that handle every DSL
    the same way (reinplacing, the functional decomposition, Inductor's layout
    constraint and mutation propagation, CSE, partitioning, ...).

    Each DSL module registers one at import time; a pass looks a node's target up
    with get_user_kernel_hop_info instead of comparing it with a DSL's HOP.

    Those passes read the HOP nodes' kwargs by name, so a registered pair must be
    traced with keyword arguments only, in this form:

    - ``kwargs``: the kernel's arguments by parameter name. The mutation HOP writes
      the tensors among them that ``get_mutated_tensors`` names.
    - The functional HOP takes the mutation HOP's arguments plus
      ``tensors_to_clone``, the names in ``kwargs`` whose tensors it clones and
      writes instead, and, if any of those is a view, ``tensor_bases``: their
      bases by name. It returns ``{name: written clone}`` for each name in
      ``tensors_to_clone``. Reinplacing removes the names whose tensors the kernel
      may write in place.
    - Any other argument is the DSL's own; the passes only hand it on.
    """

    dsl: str
    mutation: UserKernelWrapperHOP
    functional: UserKernelWrapperHOP
    # (hop_kwargs, kernel_kwargs) -> names of the kernel kwargs the kernel writes.
    # Inductor's propagate_mutation passes the mutation HOP's kwargs as lowered
    # (each FX node replaced by its lowered value, e.g. an IR node for a tensor, in
    # "kwargs" too) and that node's "kwargs" with each FX node replaced by its meta
    # value. Read only the DSL's own arguments (e.g. kernel_idx) from hop_kwargs,
    # and the kernel's arguments from kernel_kwargs.
    get_mutated_tensors: Callable[[Mapping[str, Any], dict[str, Any]], list[str]]
    # HOP kwargs -> kernel name, for logs and the reinplacing counters.
    get_kernel_name: Callable[[Mapping[str, Any]], str]
    # The functional HOP's dense impl: the post-grad decomposition traces it into
    # clones plus the mutation HOP.
    functional_dense: Callable[..., dict[str, Any]]
    # Counts this DSL's reinplacing decisions.
    reinplace_trigger: "ReInplaceTrigger"


_user_kernel_hops: dict[UserKernelWrapperHOP, UserKernelHOPInfo] = {}


def register_user_kernel_hop(info: UserKernelHOPInfo) -> None:
    from torch._dynamo.utils import ReInplaceTrigger

    if not isinstance(info.reinplace_trigger, ReInplaceTrigger):
        raise TypeError(f"{info.reinplace_trigger!r} is not a ReInplaceTrigger")
    hops = (info.mutation, info.functional)
    for hop in hops:
        if not isinstance(hop, UserKernelWrapperHOP):
            raise TypeError(f"{hop} is not a UserKernelWrapperHOP")
        if _user_kernel_hops.get(hop, info) is not info:
            raise AssertionError(f"{hop} is already registered as a user-kernel HOP")
    for hop in hops:
        _user_kernel_hops[hop] = info


def get_user_kernel_hop_info(target: object) -> UserKernelHOPInfo | None:
    """The registry entry whose mutation or functional HOP is ``target``."""
    if not isinstance(target, UserKernelWrapperHOP):
        return None
    return _user_kernel_hops.get(target)


def is_user_kernel_mutation_hop(target: object) -> bool:
    info = get_user_kernel_hop_info(target)
    return info is not None and target is info.mutation


def is_user_kernel_functional_hop(target: object) -> bool:
    info = get_user_kernel_hop_info(target)
    return info is not None and target is info.functional


def user_kernel_hop_infos() -> list[UserKernelHOPInfo]:
    """Every registered DSL, in registration order."""
    return list(dict.fromkeys(_user_kernel_hops.values()))


###############################################################################
# The HOPifier contract


class UserKernelHOPifier:
    """Base of each DSL's HOPifier (e.g. TritonHOPifier), which turns a call to a user
    kernel into a call to the DSL's mutation HOP.

    Each DSL has one subclass per capture environment: Dynamo, which works on
    VariableTrackers and reports unsupported features as graph breaks, and tracing
    (make_fx, non-strict export), which works on real or fake values.
    """

    def raise_unsupported(self, msg: str) -> Never:
        raise NotImplementedError("abstract method")

    def get_value(self, val: Any) -> Any:
        raise NotImplementedError("abstract method")
