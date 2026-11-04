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
from typing import Any, Generic, Literal, NamedTuple, TYPE_CHECKING, TypeVar
from typing_extensions import Never

import torch.fx as fx
import torch.utils._pytree as pytree
from torch import SymBool, SymFloat, SymInt, Tensor
from torch._C import _dispatch_keys, DispatchKey
from torch._higher_order_ops.reinplace_trigger import ReInplaceTrigger
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
from torch.fx.experimental.symbolic_shapes import guard_or_false, guard_scalar
from torch.utils.checkpoint import _CachedTorchDispatchMode, _CachingTorchDispatchMode


if TYPE_CHECKING:
    from torch._subclasses.functional_tensor import BaseFunctionalizeAPI


KernelT = TypeVar("KernelT")


###############################################################################
# Kernel Side Table


def _has_slot_state(value: object) -> bool:
    """Whether the class of ``value`` or a base declares __slots__ other than
    __dict__, __weakref__ and, for a dataclass instance, its fields. Like
    copyreg._slotnames, but without caching the answer on the class: Dynamo replays
    that store after the graph, outside copyreg's try/except, and it fails for
    built-in types such as types.SimpleNamespace."""
    fields: tuple[str, ...] = ()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = tuple(f.name for f in dataclasses.fields(value))
    for c in type(value).__mro__:
        slots = c.__dict__.get("__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        if any(s not in ("__dict__", "__weakref__", *fields) for s in slots):
            return True
    return False


class ConstantRules:
    """What a DSL tells constant_key and _snapshot about its own values. The
    defaults treat every value as a plain Python one; a DSL whose side table dedupes
    its constant args supplies its own (cutedsl_kernel_wrap.CUTEDSL_CONSTANT_RULES)."""

    def is_dsl_value(self, value: object) -> bool:
        """Whether ``value`` is of one of the DSL's own types, which is never keyed or
        copied by its attributes."""
        return False

    def scalar_value(self, value: object) -> bool | int | float | None:
        """The Python value that ``value``, a DSL scalar, is keyed by (with its type),
        or None for any other value."""
        return None

    def keyed_by_content(self, value: object) -> bool:
        """Whether ``value``, a callable object, is keyed and copied by its attributes
        as a plain object is (other callables are keyed as themselves)."""
        return False

    def refusal(self, cls: type) -> str | None:
        """What to pass instead of a value of type ``cls``, one of the DSL's objects
        that PT2 makes for each launch itself (a DSL tensor or stream, say), which a
        constant cannot be or hold; None for any other type."""
        return None


def _object_state(value: object, rules: ConstantRules) -> dict[str, object] | None:
    """The attributes constant_key keys an object by, and _snapshot copies: the
    fields of a dataclass instance followed by the other entries of its __dict__
    (attributes set ad hoc or in __post_init__), or the __dict__ of a plain object
    (one with no __slots__ attributes that is not callable or a module, of a class
    that does not hash by value), such as a mutable config object, or of a callable
    object the DSL keys by content (ConstantRules.keyed_by_content), such as a
    CuTeDSL kernel object, one that would be a plain object but for its class's
    @cute.jit __call__ (a host function takes one as a Constexpr and calls it). Equal
    objects then share one side-table entry and one compile, kernel objects built
    for each call included, and an object mutated between launches is compiled
    again. None for any other value, the DSL's own values and tensors (of a subclass
    that does not hash too) included.

    Limitation: an object with state in __slots__ attributes that are not dataclass
    fields, or a callable object that is not a dataclass instance and that the DSL
    does not key by content (a functor), is keyed as itself, by identity unless its
    class defines equality and hashing (a dataclass's compares its fields alone), and
    refused if it is unhashable (as a dataclass with eq=True that is not frozen is):
    mutating it between launches reuses the kernel compiled for its old contents,
    and a new object with the same contents is compiled again. Such an object also
    stays in memory, with anything it holds, a tensor included (constant_key does
    not look inside it): a side-table entry holds it for the life of the process
    (see KernelSideTable), and the eager compile cache as long as the host function
    lives (see torch/_native/cutedsl/user_kernel.py)."""
    cls = type(value)
    if rules.is_dsl_value(value) or isinstance(value, Tensor) or _has_slot_state(value):
        return None
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        state = {f.name: getattr(value, f.name) for f in dataclasses.fields(value)}
        state.update(
            (k, v) for k, v in getattr(value, "__dict__", {}).items() if k not in state
        )
        return state
    if (
        (cls.__hash__ is None or cls.__hash__ is object.__hash__)
        and hasattr(value, "__dict__")
        and not isinstance(value, types.ModuleType)
        and (not callable(value) or rules.keyed_by_content(value))
    ):
        return vars(value)
    return None


class ConstantContents(NamedTuple):
    """One level of a constant kernel argument that constant_key keys by content."""

    # The attribute names an object is keyed by; none for a tuple, list or dict.
    names: tuple[str, ...]
    # (format, key, item) for each value directly inside, in walk order, where
    # format.format(key) names the step to the item, such as "[0]".
    steps: list[tuple[str, object, object]]
    # A copy of the container with new items in their places (a tuple with the same
    # items is returned as is).
    rebuild: Callable[[list[object]], object]


def constant_contents(value: object, rules: ConstantRules) -> ConstantContents | None:
    """The values directly inside a constant kernel argument that constant_key keys,
    and _snapshot copies, by content, for any walk of a constant to agree with the
    side table: the items of a tuple or list, the keys and then the values of a dict
    (of a subclass too), the attributes _object_state reads of any other object.
    None for a value that is not such a container."""
    names: tuple[str, ...] = ()
    steps: list[tuple[str, object, object]]
    if isinstance(value, (tuple, list)):
        steps = [("[{!r}]", i, item) for i, item in enumerate(value)]
    elif isinstance(value, dict):
        steps = [(".keys()[{!r}]", i, key) for i, key in enumerate(value)]
        steps += [("[{!r}]", key, item) for key, item in value.items()]
    else:
        state = _object_state(value, rules)
        if state is None:
            return None
        names = tuple(state)
        steps = [(".{}", name, item) for name, item in state.items()]

    def rebuild(items: list[object]) -> object:
        if isinstance(value, list):
            new_list = copy.copy(value)
            new_list[:] = items
            return new_list
        if isinstance(value, tuple):
            if all(a is b for a, b in zip(items, value)):
                return value
            # A namedtuple is rebuilt with _make, any other tuple from the items.
            return getattr(type(value), "_make", type(value))(items)
        if isinstance(value, dict):
            half = len(items) // 2
            new_dict = copy.copy(value)
            new_dict.clear()
            new_dict.update(zip(items[:half], items[half:]))
            return new_dict
        new = copy.copy(value)
        for name, item in zip(names, items):
            object.__setattr__(new, name, item)
        return new

    return ConstantContents(names, steps, rebuild)


def constant_key(value: object, rules: ConstantRules) -> object:
    """A hashable key for a constant kernel argument, equal for two values only when
    they are interchangeable: containers (see constant_contents) by type and
    content, floats by hex() (-0.0 is not 0.0, and every NaN is the same), a DSL
    scalar by its type and Python value (see ConstantRules), other values by their
    own equality. Any other unhashable value, and a tensor or a value the DSL
    refuses (ConstantRules.refusal), as the value or in a container, are refused
    with a TypeError, and a value that contains itself with a ValueError."""
    return _ContentWalk(rules).key(value)


def _snapshot(value: object, rules: ConstantRules) -> object:
    """``value`` with everything constant_key keys by content copied, so that the
    caller mutating it later changes neither the stored value nor its key."""
    return _ContentWalk(rules).snapshot(value)


def map_constant(
    value: object, rules: ConstantRules, fn: Callable[[object], object]
) -> object:
    """``value`` with ``fn`` applied to each value in it that is not a container (see
    constant_contents), or to ``value`` itself if it is not one. A container is
    copied only if ``fn`` changed something in it. A tensor or a value the DSL
    refuses, as the value or in a container, and a value that contains itself are
    refused, as constant_key refuses them."""
    return _ContentWalk(rules).map(value, fn)


class _ContentWalk:
    """One walk of a constant kernel argument's contents (see constant_contents), by
    constant_key, _snapshot or map_constant. A cycle would be followed forever, so
    entering a container that the walk is already inside raises a ValueError naming
    the cycle."""

    def __init__(self, rules: ConstantRules) -> None:
        self.rules = rules
        # The containers the walk is inside, outermost first, by id, each with the
        # step it took out of it: a format and the index, key or attribute name.
        self.path: dict[int, tuple[str, object]] = {}

    def visit(
        self,
        container: object,
        steps: list[tuple[str, object, object]],
        fn: Callable[[object], object],
    ) -> list[object]:
        """``fn`` of the item of each (format, key, item) step of ``container`` (see
        ConstantContents), called with the container on the path."""
        if id(container) in self.path:
            start = list(self.path).index(id(container))
            cycle = list(self.path.values())[start:]
            raise ValueError(
                f"a constant kernel argument of type {type(container).__name__} "
                f"refers to itself through {''.join(f.format(k) for f, k in cycle)}"
            )
        results = []
        for step, key, item in steps:
            self.path[id(container)] = (step, key)
            results.append(fn(item))
        self.path.pop(id(container), None)
        return results

    def refuse(self, value: object) -> None:
        """Refuse ``value``, met at the walk's path, if it is a tensor or a value the
        DSL refuses (ConstantRules.refusal)."""
        if isinstance(value, Tensor):
            # Keyed by identity, a tensor would stay alive with its key, and it would
            # reach the DSL without from_dlpack.
            held, fix = "a tensor", "pass each tensor as its own argument"
        else:
            fix = self.rules.refusal(type(value))
            if fix is None:
                return
            held = f"a {type(value).__name__}"
        where = "".join(f.format(k) for f, k in self.path.values())
        what = f"holds {held} at {where}" if where else f"is {held}"
        raise TypeError(f"a constant kernel argument {what}: {fix}")

    def key(self, value: object) -> object:
        self.refuse(value)
        scalar = self.rules.scalar_value(value)
        if scalar is not None:
            return (type(value), self.key(scalar))
        if isinstance(value, float):
            return (type(value), value.hex())
        contents = constant_contents(value, self.rules)
        if contents is not None:
            items = self.visit(value, contents.steps, self.key)
            return (type(value), contents.names, tuple(items))
        try:
            hash(value)
        except TypeError:
            fix = (
                "pass a hashable value, or a list, tuple, dict, dataclass or plain "
                "object of them"
            )
            if not self.rules.is_dsl_value(value) and _has_slot_state(value):
                fix = (
                    "it has state in __slots__ attributes that are not dataclass "
                    "fields, so it is keyed as itself, not by content; make it "
                    "hashable, or keep that state in dataclass fields or its __dict__"
                )
            raise TypeError(
                f"a constant kernel argument of type {type(value).__name__} is "
                f"unhashable: {fix}"
            ) from None
        return (type(value), value)

    def snapshot(self, value: object) -> object:
        if self.rules.scalar_value(value) is not None:
            # A DSL scalar is keyed by its value, which its owner can still change.
            return copy.copy(value)
        contents = constant_contents(value, self.rules)
        if contents is None:
            return value
        return contents.rebuild(self.visit(value, contents.steps, self.snapshot))

    def map(self, value: object, fn: Callable[[object], object]) -> object:
        self.refuse(value)
        contents = constant_contents(value, self.rules)
        if contents is None:
            return fn(value)
        items = self.visit(value, contents.steps, lambda item: self.map(item, fn))
        if all(new is old for new, (_, _, old) in zip(items, contents.steps)):
            return value
        return contents.rebuild(items)


# We cannot put user kernels into the FX graph as the graph nodes
# do not support arbitrary functions.
# Use a side table.
# We use two dicts so that fetching both the kernel and id are O(1)
class KernelSideTable(Generic[KernelT]):
    def __init__(self, constant_rules: ConstantRules | None = None) -> None:
        # Per instance, so that each DSL's table has its own index space.
        self.id_to_kernel: dict[int, KernelT] = {}
        self.kernel_to_id: dict[KernelT, int] = {}
        self.constant_args: dict[int, dict[str, object]] = {}
        self.lock = threading.Lock()
        # With constant_rules, equal constant args (see constant_key) share one index,
        # so a table that is never reset does not grow with every eager launch, and
        # each entry is a snapshot of the args it was added with. A constant keyed by
        # identity (see _object_state) still adds an entry per object, which such a
        # table keeps, with the object, for the life of the process: each capture adds
        # its constants, and captured graphs hold their indexes, so entries cannot be
        # evicted.
        self.constant_rules = constant_rules
        self._constant_args_index: dict[object, int] | None = (
            {} if constant_rules is not None else None
        )
        # index -> {name: constant_key of the arg}, for a launch to reuse.
        self._constant_args_keys: dict[int, dict[str, object]] = {}

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
            rules = self.constant_rules
            if self._constant_args_index is not None and rules is not None:
                key = tuple(
                    (k, constant_key(v, rules)) for k, v in sorted(args.items())
                )
                if key in self._constant_args_index:
                    return self._constant_args_index[key]
                args = {k: _snapshot(v, rules) for k, v in args.items()}
            idx = len(self.constant_args)
            self.constant_args[idx] = args
            if key is not None:
                self._constant_args_index[key] = idx  # type: ignore[index]
                self._constant_args_keys[idx] = dict(key)
            return idx

    # Returns the constant args
    def get_constant_args(self, idx: int) -> dict[str, object]:
        # No need to lock here as fetching from dict is atomic
        if idx not in self.constant_args:
            raise AssertionError(
                f"Constant args index {idx} not found in constant_args"
            )
        return self.constant_args[idx]

    def get_constant_args_keys(self, idx: int) -> dict[str, object]:
        """The constant_key of each constant arg at ``idx``, as add_constant_args
        computed it (none in a table without constant_rules)."""
        self.get_constant_args(idx)
        return self._constant_args_keys.get(idx, {})

    # Resets the table (only meant to be used in unit tests)
    # This is only safe assuming single threaded execution
    def reset_table(self) -> None:
        self.id_to_kernel = {}
        self.kernel_to_id = {}
        self.constant_args = {}
        if self._constant_args_index is not None:
            self._constant_args_index = {}
        self._constant_args_keys = {}


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


UserKernelLayoutConstraint = Literal["needs_fixed_stride_order", "flexible_layout"]


# eq=False: infos compare and hash by identity, as register_user_kernel_hop compares
# them, so a DSL's callables need not be hashable.
@dataclasses.dataclass(frozen=True, eq=False)
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

    The functional HOP's CompositeExplicitAutograd py_impl, which eager runs, must
    call the mutation HOP once, on clones of the tensors named in
    ``tensors_to_clone`` (functional_hop_dense does this). The post-grad
    decomposition traces that impl, so a pair without one cannot be registered.

    The FX graph cache keys user kernels by source for Triton only
    (FxGraphHashDetails, and the has_user_defined_triton_kernels flag computed from
    it), so a DSL whose HOPs are cacheable must extend the cache key first. HOPs are
    not cacheable by default, and the cache bypasses every graph that contains an
    uncacheable HOP, such as CuTeDSL's (cacheable=False).
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
    # HOP kwargs -> kernel name, for the reinplacing pass's log.
    get_kernel_name: Callable[[Mapping[str, Any]], str]
    # () -> the layout constraint Inductor gives the mutation HOP's tensor arguments:
    # "needs_fixed_stride_order" (the strides they had when the graph was traced) or
    # "flexible_layout" (Inductor picks them). Inductor calls it each time it lowers
    # one of this DSL's mutation HOP nodes.
    get_layout_constraint: Callable[[], UserKernelLayoutConstraint]
    # Only keys this DSL's reinplacing counters: ReInplaceTrigger.USER_KERNEL_OPS, or
    # a member of the DSL's own. auto_functionalize's AUTO_FUNC_V1 and AUTO_FUNC_V2
    # are refused; the reinplacing pass also reads AUTO_FUNC_V2 as a switch that
    # stops it from rewriting getitem users.
    reinplace_trigger: ReInplaceTrigger
    # torch.export's serializer can write this DSL's nodes.
    serializable: bool


_user_kernel_hops: dict[UserKernelWrapperHOP, UserKernelHOPInfo] = {}


def register_user_kernel_hop(info: UserKernelHOPInfo) -> None:
    """Register a DSL's HOP pair with the passes that handle every DSL. Each DSL
    module calls it once, at import time.

    Everything is checked before the registry is written, so a refused registration
    leaves it as it was. Raises:

    - TypeError if the trigger is not a ReInplaceTrigger member (its name, for
      example), if either HOP is not a UserKernelWrapperHOP, or if the functional HOP
      has no CompositeExplicitAutograd py_impl for the decomposition to trace.
    - ValueError if the trigger is auto_functionalize's AUTO_FUNC_V1 or AUTO_FUNC_V2,
      or if the mutation and functional HOPs are the same HOP.
    - RuntimeError if either HOP is registered already, with another info. Infos
      compare by identity: registering the same object again is accepted and changes
      nothing, but an equal copy of it is refused.
    """
    trigger = info.reinplace_trigger
    if not isinstance(trigger, ReInplaceTrigger):
        raise TypeError(f"{trigger!r} is not a ReInplaceTrigger")
    if trigger in (ReInplaceTrigger.AUTO_FUNC_V1, ReInplaceTrigger.AUTO_FUNC_V2):
        raise ValueError(
            f"{trigger} belongs to auto_functionalize; register the {info.dsl} DSL "
            "with ReInplaceTrigger.USER_KERNEL_OPS or a member of its own"
        )
    if info.mutation is info.functional:
        raise ValueError(
            f"{info.mutation} cannot be both the mutation and the functional HOP of "
            f"the {info.dsl} DSL"
        )
    hops = (info.mutation, info.functional)
    for hop in hops:
        if not isinstance(hop, UserKernelWrapperHOP):
            raise TypeError(f"{hop} is not a UserKernelWrapperHOP")
        owner = _user_kernel_hops.get(hop, info)
        if owner is not info:
            raise RuntimeError(
                f"{hop} is already registered by the {owner.dsl} DSL with another "
                "UserKernelHOPInfo object; only that object can be registered again"
            )
    if DispatchKey.CompositeExplicitAutograd not in info.functional.py_kernels:
        raise TypeError(f"{info.functional} has no CompositeExplicitAutograd py_impl")
    for hop in hops:
        _user_kernel_hops[hop] = info


def get_user_kernel_hop_info(
    target: object, role: Literal["mutation", "functional"] | None = None
) -> UserKernelHOPInfo | None:
    """The registry entry whose mutation or functional HOP is ``target``; given a
    ``role``, only if ``target`` is the HOP of its pair with that role."""
    if not isinstance(target, UserKernelWrapperHOP):
        return None
    info = _user_kernel_hops.get(target)
    if info is None or role is None:
        return info
    hop = info.mutation if role == "mutation" else info.functional
    return info if target is hop else None


def is_user_kernel_mutation_hop(target: object) -> bool:
    return get_user_kernel_hop_info(target, "mutation") is not None


def is_user_kernel_functional_hop(target: object) -> bool:
    return get_user_kernel_hop_info(target, "functional") is not None


def user_kernel_hop_infos() -> list[UserKernelHOPInfo]:
    """Every registered DSL, in registration order."""
    return list(dict.fromkeys(_user_kernel_hops.values()))


###############################################################################
# The HOPifier contract


class UserKernelHOPifier:
    """Base of each DSL's HOPifier (TritonHOPifier, CuteDSLHOPifier), which turns a
    call to a user kernel into a call to the DSL's mutation HOP.

    Each DSL has one subclass per capture environment it supports: Dynamo, which
    works on VariableTrackers and reports unsupported features as graph breaks, and
    tracing (make_fx, non-strict export), which works on real or fake values.
    """

    def raise_unsupported(self, msg: str) -> Never:
        raise NotImplementedError("abstract method")

    def get_value(self, val: Any) -> Any:
        raise NotImplementedError("abstract method")


def specialize_symbolic(arg: Any) -> Any:
    # See [Note: Specialize tl.constexpr args in user-defined triton kernels]
    if isinstance(arg, (SymInt, SymBool, SymFloat)):
        return guard_scalar(arg)
    return arg
