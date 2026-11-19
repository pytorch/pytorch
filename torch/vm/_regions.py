"""Runs of a basic block compiled by a graph backend.

With ``torch.vm.passes.Graphs()`` in its list of passes the VM hands the tensor work
of each basic block to AOTAutograd and Inductor. The unit is a *region*: a run
of consecutive instructions of one block that have no effect a trace cannot
capture (calls, arithmetic, attribute reads, building containers). A region is
compiled lazily:

- the first calls run its instructions as they are;
- then the instructions are run once more on fake tensors under ``make_fx``,
  with what they call left to the interpreter, which gives one ATen graph of
  everything the region does, the functions it calls included. Parameters and
  buffers of the modules the region takes are inputs of that graph;
- the graph goes to the backend. For Inductor that is
  ``torch._inductor.compile``: AOTAutograd traces the backward of the graph
  ahead of time and both directions are compiled, so a region that runs with
  gradients enabled returns tensors whose ``grad_fn`` is one compiled autograd
  function.

Regions are tried from the outside in. The region around ``model(x)`` in the
function the user compiled is tried first; if all of it traces, nothing below
it is looked at again. If an instruction does not trace (it branches on the
value of a tensor, calls ``backward``, sets an attribute of a module, touches a
tensor the region was not given, ...), the region is split around that
instruction, which then runs in the VM as before, so that the blocks of what it
calls get their turn. Control flow that depends on data therefore stays control
flow of the VM, at whatever depth it occurs, with compiled regions in between.

A trace runs the instructions a second time and a compiled region does not run
them at all, so a region may only hold instructions without an effect on
Python objects. For the instructions of the region itself that is checked on
the values they are about to run on, before each one runs (``_problem``): a
call must be of a module, a function, a method of a tensor or of a module, or
a builtin that only reads; and unless it is a known pure builtin it must
involve a tensor or a module, so that ``time.time()`` or ``opt.step()`` stay
with the VM. For what those instructions call, it is the assumption every
tracer of PyTorch programs makes, with the usual offenders turned into errors
while the trace runs (``_traps``).

The inputs of a region are tensors, modules, plain values, functions and
classes, and tuples, lists, dicts, pytree nodes and plain objects of those
(an object that is its class and its ``__dict__``: a cache, say). A trace
runs on copies of the lists, dicts and objects, so that what the code does to
them shows: storing an item or an attribute is then something the region does
to the real ones after its graph has run (``_Variant.effects``), as is
storing a tensor in an attribute of a module that held one. An instruction
that takes anything else from before the region stays with the VM. What a
region gives back may also be an object it made itself, which every call then
makes again from the results of the graph (``_Recipes``).

The last region of a block does not end with the block. Its trace goes on into
the blocks that follow, along the way the values of the trace take through
them: a loop over a list of layers or a ``range``, a branch on a flag or on
whether an argument is None, a statement that has no tensor in it. The graph
is then the tensor work of that whole path, and the region tells the VM where
in the function to go on (``Region.leap``): at the instruction that ended the
path, which is one a region cannot hold, a branch on the data of a tensor, or
the end of the function. What a branch on the way looked at is either an input
of the region, which every call checks, or an attribute of a module or a
global that the path read, which is checked along with them (``_Walk.guards``).

A compiled region is valid for the inputs it was traced with. Every call
compares the tensors (rank, dtype, device, ``requires_grad``, sizes and
strides, which of them are the same object), the values of numbers and
strings, the identity of functions and classes, the modules (their submodules
and ``training`` flags) and the grad and autocast modes; a new combination is
traced and compiled as another variant. What the trace read from anywhere
else, such as an attribute of a module that is not a tensor or a global that
is rebound later, is not checked.
"""

import collections
import contextlib
import enum
import gc
import random
import sys
import threading
import time
import types
import weakref
from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensor, is_fake_tensor
from torch.utils import _pytree as pytree


Backend = Callable[[torch.fx.GraphModule, list[torch.Tensor]], Callable[..., Any]]


class Untraceable(Exception):
    """What a region does at this point cannot be captured by a trace."""


class _Failed(Exception):
    # A region could not be compiled. `index` is the instruction to blame, or
    # None if no single one is.
    def __init__(self, index: int | None, reason: str) -> None:
        super().__init__(reason)
        self.index = index
        self.reason = reason


class _Opaque(Exception):
    # An input of a region that cannot be an input of a graph.
    def __init__(self, position: int, reason: str) -> None:
        super().__init__(reason)
        self.position = position
        self.reason = reason


class _State(threading.local):
    # Regions that are running their instructions while it is still open
    # whether they will be compiled.
    pending = 0
    # Ids of the modules whose tensors the trace in progress replaced, and
    # what they held when it began.
    traced: set[int] | None = None
    watch: "_Watch | None" = None


_state = _State()
_runtime: Any = None

# Instructions that can do tensor work. A region must contain one.
_WORK = frozenset({"call", "call_ex", "binary", "unary", "compare", "get_item"})
# Instructions that read something and do nothing else.
_READS = frozenset(
    {"load_attr", "load_global", "load_deref", "load_closure", "load_slot", "cell_get"}
)
# Instructions that change their first operand.
_MUTATORS = frozenset(
    {"list_append", "list_extend", "dict_update", "dict_merge", "map_add"}
)

_VALUES = frozenset(
    {
        int,
        float,
        bool,
        str,
        bytes,
        complex,
        type(None),
        type(...),
        range,
        slice,
        torch.Size,
        torch.dtype,
        torch.device,
        torch.layout,
        torch.memory_format,
    }
)
# Objects that are what they are: checked by identity, not looked into.
_PLAIN = frozenset(
    {
        types.FunctionType,
        types.ModuleType,
        types.MethodDescriptorType,
        types.WrapperDescriptorType,
        type,
        object,
    }
)
_TENSORS = frozenset({torch.Tensor, nn.Parameter})
_TENSOR_TYPES = tuple(_TENSORS)
# How many times regions wait for one that was surveyed before them to be
# compiled, before they go ahead without it.
_MOST_WAITS = 256
_HEAPTYPE = 1 << 9
# How many variants of a region may leave what they were given with other
# items, keys or attributes than it had. A cache that is filled on the first
# call does that once.
_MOST_GROWTH = 2
# How many lists, dicts and objects with state one call of a region may hand
# it: each is copied for a trace and compared after it.
_MOST_HELD = 4096


_CONTAINERS = frozenset(
    {
        list,
        dict,
        set,
        bytearray,
        collections.OrderedDict,
        collections.defaultdict,
        collections.deque,
    }
)
# Methods of containers that leave them as they are.
_READERS = frozenset(
    {
        "get",
        "keys",
        "values",
        "items",
        "copy",
        "index",
        "count",
        "__getitem__",
        "__contains__",
        "__len__",
        "__iter__",
        "union",
        "intersection",
        "difference",
        "symmetric_difference",
        "issubset",
        "issuperset",
        "isdisjoint",
    }
)
# Methods of a dict that put something into it or take something out.
_DICT_STORES = frozenset({"__setitem__", "__delitem__", "update", "setdefault", "pop"})
# Methods of tensors whose effect is not a tensor operation.
_TENSOR_EFFECTS = frozenset(
    {
        "backward",
        "register_hook",
        "register_post_accumulate_grad_hook",
        "retain_grad",
        "requires_grad_",
        "share_memory_",
    }
)
# Methods and conversions that give the data of a tensor to Python: a graph
# has no place for what Python then does with it.
_TENSOR_READS = frozenset(
    {"item", "tolist", "numpy", "__bool__", "__int__", "__float__", "__index__"}
)
_SCALARS = (bool, int, float, complex)
_IMPURE_BUILTINS = frozenset(
    {
        "print",
        "input",
        "setattr",
        "delattr",
        "exec",
        "eval",
        "open",
        "next",
        "breakpoint",
        "compile",
        "globals",
        "locals",
        "vars",
        "__import__",
    }
)
# Where functions live that compute a value from their arguments and do
# nothing else, whatever the arguments are.
_PURE_MODULES = frozenset({"builtins", "math", "operator"})
# The parts of torch itself that hold the tensor work of a model. The rest of
# the library (the optimizers, the autograd front end, the plumbing around
# operators) is Python that moves tensors about: regions there would be tried
# one after the other for nothing.
_TORCH = torch.__file__.rsplit("/", 1)[0] + "/"
_TORCH_MODEL_CODE = (
    "nn/",
    "functional.py",
    "_tensor.py",
    "distributions/",
    "_functorch/",
    "func/",
)


def _wants_regions(filename: str) -> bool:
    if not filename.startswith(_TORCH):
        return True
    return filename[len(_TORCH) :].startswith(_TORCH_MODEL_CODE)


# ----------------------------------------------------------------------
# Which instructions a region may hold
# ----------------------------------------------------------------------


def _involves_tensor(obj: Any, depth: int = 3) -> bool:
    if isinstance(obj, (torch.Tensor, nn.Module)):
        return True
    if depth == 0:
        return False
    if isinstance(obj, (tuple, list)):
        return any(_involves_tensor(item, depth - 1) for item in obj)
    if isinstance(obj, dict):
        return any(_involves_tensor(item, depth - 1) for item in obj.values())
    receiver = getattr(obj, "__self__", None)
    return receiver is not None and isinstance(receiver, (torch.Tensor, nn.Module))


def _is_outside_tensor(value: Any, depth: int = 2) -> bool:
    """Whether a value a trace came by is, or holds, a tensor that is not of
    the trace: one that existed before it and was not handed to it."""
    if type(value) in _TENSORS:
        return True
    if depth and type(value) in (tuple, list):
        return any(_is_outside_tensor(item, depth - 1) for item in value)
    if depth and type(value) is dict:
        return any(_is_outside_tensor(item, depth - 1) for item in value.values())
    return False


def _receiver_problem(
    receiver: Any, name: str, local: dict[int, Any], given: Any = ()
) -> str | None:
    cls = type(receiver)
    if isinstance(receiver, torch.Tensor):
        if name in _TENSOR_READS:
            return "reads the data of a tensor"
        return (
            f"Tensor.{name} has an effect a graph does not hold"
            if name in _TENSOR_EFFECTS
            else None
        )
    if (
        isinstance(receiver, (nn.Module, type, types.ModuleType, tuple, frozenset))
        or cls in _VALUES
    ):
        return None
    if cls in _CONTAINERS:
        if id(receiver) in local or name in _READERS:
            return None
        if (
            id(receiver) in given
            and isinstance(receiver, dict)
            and name in _DICT_STORES
        ):
            # Done to a copy in a trace, and to the dict itself after the
            # graph (``_Variant.effects``). A list that is appended to is
            # another list on every call: that stays with the VM.
            return None
        return f"changes a {cls.__name__} from before the region"
    if id(receiver) in given:
        # An object whose state the region keeps track of.
        return None
    return f"a method of a {cls.__name__}, which may have effects"


def _callee_problem(callee: Any, local: dict[int, Any], given: Any = ()) -> str | None:
    """Why a region cannot hold a call of ``callee``, or None if it can."""
    cls = type(callee)
    if cls is types.FunctionType or isinstance(callee, (nn.Module, type)):
        return None
    if cls is types.MethodType:
        name = callee.__func__.__name__
        return _receiver_problem(callee.__self__, name, local, given)
    if cls is types.BuiltinFunctionType:
        receiver = callee.__self__
        if receiver is None or type(receiver) is types.ModuleType:
            impure = (
                callee.__module__ == "builtins" and callee.__name__ in _IMPURE_BUILTINS
            )
            return f"{callee.__name__} has an effect" if impure else None
        return _receiver_problem(receiver, callee.__name__, local, given)
    if cls is types.MethodWrapperType:
        return _receiver_problem(callee.__self__, callee.__name__, local, given)
    if cls is types.MethodDescriptorType or cls is types.WrapperDescriptorType:
        owner = callee.__objclass__
        if owner in _CONTAINERS and callee.__name__ not in _READERS:
            return f"changes a {owner.__name__}"
        return None
    if isinstance(callee, (torch._ops.OpOverload, torch._ops.OpOverloadPacket)):
        return None
    return f"calls a {cls.__name__}, which may have effects"


def _is_pure(callee: Any) -> bool:
    if type(callee) is types.BuiltinFunctionType or isinstance(callee, type):
        return getattr(callee, "__module__", None) in _PURE_MODULES
    return False


def _problem(
    inst: dict[str, Any], env: dict[int, Any], local: dict[int, Any], given: Any = ()
) -> str | None:
    """Why a region cannot hold this instruction, on the values it is about
    to run on, or None if it can. ``given`` has the ids of the lists, dicts
    and objects the region was handed and keeps track of."""
    op = inst["op"]
    if op == "call" or op == "call_ex":
        uses = inst["uses"]
        callee = env.get(uses[0])
        reason = _callee_problem(callee, local, given)
        if reason is not None:
            return reason
        if any(callee is scalar for scalar in _SCALARS):
            if any(isinstance(env.get(v), torch.Tensor) for v in uses[1:]):
                return "reads the data of a tensor"
        if not _is_pure(callee) and not any(_involves_tensor(env.get(v)) for v in uses):
            return "no tensor or module involved"
    elif op == "to_bool" or (op == "unary" and inst["text"].split()[1] == "not"):
        if isinstance(env.get(inst["uses"][0]), torch.Tensor):
            return "branches on the data of a tensor"
    elif op == "binary" and inst["text"].split()[1].endswith("="):
        target = env.get(inst["uses"][0])
        if type(target) in _CONTAINERS and id(target) not in local:
            return f"changes a {type(target).__name__} from before the region"
    return None


def _note_local(
    inst: dict[str, Any], env: dict[int, Any], local: dict[int, Any]
) -> None:
    # Containers the region itself made: changing those is no effect.
    op = inst["op"]
    made = op in ("build_list", "build_map", "build_set")
    if op == "call":
        callee = env.get(inst["uses"][0])
        made = any(callee is maker for maker in (list, dict, set, sorted))
    if made and inst["dst"] is not None:
        value = env.get(inst["dst"])
        local[id(value)] = value


# ----------------------------------------------------------------------
# Modules: what they hold, and noticing when that changes
# ----------------------------------------------------------------------

# Hooks are code that runs at a call of a module and is not in its forward: a
# graph of the call would not hold what they do.
_HOOK_DICTS = (
    "_forward_hooks",
    "_forward_pre_hooks",
    "_backward_hooks",
    "_backward_pre_hooks",
)
_HOOK_REGISTRATIONS = (
    "register_forward_hook",
    "register_forward_pre_hook",
    "register_full_backward_hook",
    "register_full_backward_pre_hook",
    "register_backward_hook",
)
_GLOBAL_HOOK_DICTS = tuple(
    getattr(torch.nn.modules.module, "_global" + name) for name in _HOOK_DICTS
)
# What an RNN computes from its parameters and keeps: its weights in a list,
# with weak references to notice when one is replaced. It rebuilds both when
# that happens, as it does in a trace.
_DERIVED = (sys.intern("_flat_weights"), sys.intern("_flat_weight_refs"))


# The dicts of a module that are changed in place.
_MODULE_DICTS = ("_parameters", "_buffers", "_modules", *_HOOK_DICTS)
# The methods that change a module. A trace of a region that was given the
# module must not run one: the instructions will, when they run for real.
_MODULE_CHANGES = {
    getattr(nn.Module, name).__code__: name
    for name in (
        "__setattr__",
        "__delattr__",
        "register_buffer",
        "register_parameter",
        "add_module",
        *_HOOK_REGISTRATIONS,
    )
}
# The tool of sys.monitoring that sees them start: -1 if there is none to
# have, None before the first trace.
_monitor: int | None = None


def _module_change_started(code: types.CodeType, offset: int) -> None:
    traced = _state.traced
    if traced is None:
        return
    arguments = sys._getframe(1).f_locals
    module: Any = arguments.get("self")
    if id(module) not in traced:
        return
    name = _MODULE_CHANGES[code]
    if name == "__setattr__":
        attribute: Any = arguments.get("name")
        if isinstance(module, nn.RNNBase) and attribute in _DERIVED:
            # Derived from the parameters, which the trace replaced: put
            # back when the trace is over (see ``_Watch``).
            return
        current = vars(module).get(attribute, module)
        value: Any = arguments.get("value", code)
        if current is value:
            return
        watch = _state.watch
        if watch is not None and type(value) is FakeTensor:
            if type(current) is FakeTensor or type(current) in _TENSORS:
                # A tensor the module keeps as a plain attribute (a mask it
                # builds, a cache) gets another tensor of the trace. The
                # trace goes on with it, and the region stores what the
                # graph computes for it when it has run (``_Watch.stored``).
                watch.stores[(id(vars(module)), attribute)] = (module, attribute)
                return
    raise Untraceable(f"{name} on a module the region was given")


def _watch_module_changes(on: bool) -> None:
    """Has the methods that change a module report to the trace, or stops it.

    The methods themselves stay what they are: replacing them on nn.Module,
    even for the time of a trace, takes the caches of attribute lookups from
    every module class, and what else runs in the process (torch.compile
    traces Module.__setattr__) would find something it does not know.
    """
    global _monitor
    monitoring: Any = getattr(sys, "monitoring", None)
    if monitoring is None:
        return
    if _monitor is None:
        _monitor = -1
        for tool in (4, 3):
            try:
                monitoring.use_tool_id(tool, "torch.vm")
            except ValueError:
                continue
            start = monitoring.events.PY_START
            monitoring.register_callback(tool, start, _module_change_started)
            _monitor = tool
            break
    if _monitor >= 0:
        events = monitoring.events.PY_START if on else 0
        for code in _MODULE_CHANGES:
            monitoring.set_local_events(_monitor, code, events)


class _Watch:
    """What the modules of a trace hold, to see whether a step of the trace
    gave one of them something else, and to put everything back.

    A trace runs code on the modules themselves. An attribute it stores, a
    buffer it registers or a hook it adds would be there twice once the
    instructions run for real, and is not in the graph. The methods that do
    such things report to the trace (``_watch_module_changes``); this is
    for what is done without them, and for what they are allowed to do.
    """

    def __init__(self, modules: dict[int, nn.Module]) -> None:
        self.modules = list(modules.values())
        # Attributes that hold a tensor and were given another one of the
        # trace, by the dict they are in and their name.
        self.stores: dict[tuple[int, str], tuple[nn.Module, str]] = {}
        self.saved: list[tuple[dict[Any, Any], dict[Any, Any]]] = []
        for module in self.modules:
            for m in module.modules():
                attributes = vars(m)
                self.saved.append((attributes, dict(attributes)))
                for name in _MODULE_DICTS:
                    held = attributes.get(name)
                    if isinstance(held, dict):
                        self.saved.append((held, dict(held)))
        self.saved.extend((held, dict(held)) for held in _GLOBAL_HOOK_DICTS)
        self.before = self.stamp()

    def stamp(self) -> tuple[int, ...]:
        hooks = [len(held) for held in _GLOBAL_HOOK_DICTS]
        return (*(_runtime.module_identity(m, _DERIVED) for m in self.modules), *hooks)

    def changed(self) -> bool:
        now = self.stamp()
        if now == self.before:
            return False
        if self.stores and not self.changed_otherwise():
            self.before = now
            return False
        return True

    def changed_otherwise(self) -> bool:
        """Whether more is different than the tensors that were stored."""
        for held, saved in self.saved:
            if len(held) != len(saved):
                return True
            for key, value in saved.items():
                if held.get(key, self) is value or key in _DERIVED:
                    continue
                if (id(held), key) not in self.stores:
                    return True
        return False

    def stored(self) -> list[tuple[nn.Module, str, Any]]:
        """The tensors a trace left in attributes of the modules."""
        before = {id(held): saved for held, saved in self.saved}
        found = []
        for (held, name), (module, _) in self.stores.items():
            value = vars(module).get(name)
            if value is not before[held].get(name):
                found.append((module, name, value))
        return found

    def restore(self) -> None:
        for held, saved in self.saved:
            same = len(held) == len(saved) and all(
                held.get(key, self) is value for key, value in saved.items()
            )
            if not same:
                held.clear()
                held.update(saved)


class _ModuleState:
    """The tensors of a module and of its submodules, for one stamp."""

    __slots__ = ("accessors", "hooked", "ids", "module", "stamp", "token")

    def __init__(self, module: nn.Module, stamp: int) -> None:
        self.module = weakref.ref(module)
        self.refresh(module, stamp)

    def refresh(self, module: nn.Module, stamp: int) -> None:
        self.stamp = stamp
        # Where each tensor is kept: the dicts are read again on every call,
        # so a parameter that was replaced by another tensor is picked up.
        self.accessors: list[tuple[Any, Any]] = []
        ids, training = [], []
        # Whether the module or one inside it has a hook.
        self.hooked = False
        for m in module.modules():
            ids.append(id(m))
            training.append(m.training)
            self.hooked = self.hooked or any(getattr(m, name) for name in _HOOK_DICTS)
            for store in (m._parameters, m._buffers):
                for name, tensor in store.items():
                    if tensor is not None:
                        self.accessors.append((store, name))
            # Tensors kept as plain attributes, or in a list that is one (as
            # the weights of an RNN are, next to being its parameters).
            attributes = vars(m)
            for name, value in attributes.items():
                if type(value) in _TENSORS:
                    self.accessors.append((attributes, name))
                elif type(value) is list:
                    for i, item in enumerate(value):
                        if type(item) in _TENSORS:
                            self.accessors.append((value, i))
        self.ids = frozenset(ids)
        shape = (tuple(ids), tuple(training), tuple(n for _, n in self.accessors))
        # A small number that stands for the shape in the key of a variant.
        self.token = _tokens.setdefault(shape, len(_tokens))


_tokens: dict[Any, int] = {}
_module_states: dict[int, _ModuleState] = {}


def _forget_module(key: int) -> None:
    _module_states.pop(key, None)


def _module_state(module: nn.Module) -> _ModuleState:
    # What is known about a module holds while its stamp is what it was: a
    # parameter, buffer, submodule or hook more or less, a tensor attribute
    # that was none, another training flag all change it.
    stamp = _runtime.module_stamp(module, _TENSOR_TYPES)
    state = _module_states.get(id(module))
    if state is None or state.module() is not module:
        state = _module_states[id(module)] = _ModuleState(module, stamp)
        weakref.finalize(module, _forget_module, id(module))
    elif state.stamp != stamp:
        state.refresh(module, stamp)
    return state


# ----------------------------------------------------------------------
# Inputs of a region
# ----------------------------------------------------------------------


class _Inputs:
    """What one call hands a region, as ``_describe`` finds it."""

    __slots__ = ("flat", "held", "seen")

    def __init__(self) -> None:
        # The tensors, each object once.
        self.flat: list[torch.Tensor] = []
        # The objects whose state a region may change: lists, dicts, sets,
        # cells, plain instances.
        self.held: list[Any] = []
        # Where a tensor is in ``flat``, or, counted down from -1, an object
        # in ``held``; by id.
        self.seen: dict[int, int] = {}


def _hold(obj: Any, inputs: _Inputs) -> int | None:
    """Notes an object a region may change. Returns where it was noted before,
    or None if it is new: found a second time, it is the same object, and a
    trace must see one copy of it, not two."""
    known = inputs.seen.get(id(obj))
    if known is None:
        if len(inputs.held) >= _MOST_HELD:
            raise _Opaque(0, "more objects than a region keeps track of")
        inputs.seen[id(obj)] = -1 - len(inputs.held)
        inputs.held.append(obj)
    return known


def _describe(obj: Any, inputs: _Inputs) -> Any:
    """What a compiled region depends on in ``obj``, as something hashable,
    but for its tensors: those go to ``inputs`` in the order they are found,
    and are compared by what they are like (``_key``)."""
    cls = type(obj)
    if cls in _TENSORS:
        position = inputs.seen.get(id(obj))
        if position is None:
            position = inputs.seen[id(obj)] = len(inputs.flat)
            inputs.flat.append(obj)
        return position
    if cls in _VALUES:
        return (cls, obj)
    if cls in _PLAIN:
        return ("object", id(obj))
    if cls is tuple:
        return (cls, tuple([_describe(item, inputs) for item in obj]))
    if cls is list:
        known = _hold(obj, inputs)
        if known is not None:
            return ("again", known)
        return (cls, tuple([_describe(item, inputs) for item in obj]))
    if cls is dict or cls is collections.OrderedDict:
        known = _hold(obj, inputs)
        if known is not None:
            return ("again", known)
        for key in obj:
            if type(key) not in _VALUES:
                raise _Opaque(0, "a dict with keys that are not plain values")
        values = tuple([_describe(item, inputs) for item in obj.values()])
        return (cls, tuple(obj), values)
    if cls is set or cls is frozenset:
        if all(type(item) in _VALUES for item in obj):
            if cls is set and _hold(obj, inputs) is not None:
                return ("again", inputs.seen[id(obj)])
            return (cls, frozenset(obj))
        raise _Opaque(0, "a set of things that are not plain values")
    if cls is types.CellType:
        # A variable that a nested function shares: what it holds.
        known = _hold(obj, inputs)
        if known is not None:
            return ("again", known)
        try:
            return ("cell", _describe(obj.cell_contents, inputs))
        except ValueError:
            raise _Opaque(0, "a variable that has no value yet") from None
    if isinstance(obj, nn.Module):
        state = _module_state(obj)
        if state.hooked or any(_GLOBAL_HOOK_DICTS):
            raise _Opaque(
                0, "a module with hooks, which run code a graph does not hold"
            )
        flat, seen = inputs.flat, inputs.seen
        try:
            for store, name in state.accessors:
                tensor = store[name]
                if id(tensor) not in seen:
                    if type(tensor) not in _TENSORS:
                        raise _Opaque(0, "a module that holds a tensor subclass")
                    seen[id(tensor)] = len(flat)
                    flat.append(tensor)
        except LookupError:
            # Changed since the stamp was taken.
            state.refresh(obj, _runtime.module_stamp(obj, _TENSOR_TYPES))
            return _describe(obj, inputs)
        return ("module", id(obj), state.token)
    if cls is types.MethodType:
        return (
            "method",
            id(obj.__func__),
            _describe(obj.__self__, inputs),
        )
    if cls is types.BuiltinFunctionType:
        receiver = obj.__self__
        if receiver is None or type(receiver) is types.ModuleType:
            return ("object", id(obj))
        if type(receiver) in _CONTAINERS:
            raise _Opaque(
                0, f"a method of a {type(receiver).__name__} from before the region"
            )
        return ("method", obj.__name__, _describe(receiver, inputs))
    if isinstance(obj, tuple):
        # Named tuples and the like.
        return (cls, tuple([_describe(item, inputs) for item in obj]))
    node = pytree.SUPPORTED_NODES.get(cls)
    if node is not None:
        # Looked into once, like the copy a trace makes of it: what it holds
        # is then found once, in the order the trace finds it.
        known = inputs.seen.get(id(obj))
        if known is not None:
            return ("again", known)
        inputs.seen[id(obj)] = -_MOST_HELD - len(inputs.seen)
        children, context = node.flatten_fn(obj)
        return (
            cls,
            repr(context),
            tuple([_describe(item, inputs) for item in children]),
        )
    if isinstance(
        obj, (type, enum.Enum, torch._ops.OpOverload, torch._ops.OpOverloadPacket)
    ):
        return ("object", id(obj))
    if _has_plain_state(obj):
        # An object that is its class and its attributes: a cache, say. What
        # it holds is looked at like the items of a dict.
        known = _hold(obj, inputs)
        if known is not None:
            return ("again", known)
        state = vars(obj)
        values = tuple([_describe(item, inputs) for item in state.values()])
        return ("instance", cls, tuple(state), values)
    # Anything else may have state that a trace would change unseen.
    raise _Opaque(0, f"a {cls.__name__} from before the region")


def _acceptable(obj: Any) -> str | None:
    """Why ``obj`` cannot be an input of a region, or None if it can."""
    try:
        _describe(obj, _Inputs())
    except _Opaque as e:
        return e.reason
    return None


def _key(parts: list[Any], inputs: _Inputs) -> Any:
    """The key of the variant for a call: what its inputs are apart from
    tensors, what kinds of tensors they hold, the numbers of the call (the
    sizes and strides of the tensors), and the modes it runs in."""
    try:
        kinds, numbers, _ = _runtime.tensor_key(inputs.flat, [])
    except TypeError:
        raise _Opaque(0, "a tensor without sizes of its own") from None
    return (tuple(parts), kinds, numbers, _modes())


class _Copies:
    """The copies a trace runs on in place of what a region was given.

    A trace runs the code of the region, and that code may change a list, a
    dict or an object it was handed. It changes a copy; what it did to the
    copy is what the region does to the real one after its graph has run
    (``_Variant.effects``).
    """

    def __init__(self, fakes: dict[int, torch.Tensor]) -> None:
        self.fakes = fakes
        # By the id of what they are copies of.
        self.of: dict[int, Any] = {}
        # Pytree nodes, with their children: these must stay as they are.
        self.nodes: list[tuple[Any, list[Any]]] = []
        self.tracked: list[tuple[Any, Any]] = []

    def track(self, held: list[Any]) -> None:
        """Remembers what the copies of ``held`` hold before the trace."""
        self.tracked = [
            (self.of[id(real)], _state_of(self.of[id(real)])) for real in held
        ]

    def given(self) -> dict[int, int]:
        return {id(copy): k for k, (copy, _) in enumerate(self.tracked)}


def _state_of(copy: Any) -> Any:
    cls = type(copy)
    if cls is list:
        return list(copy)
    if cls is set:
        return set(copy)
    if cls is types.CellType:
        try:
            return [copy.cell_contents]
        except ValueError:
            return []
    return dict(copy) if isinstance(copy, dict) else dict(vars(copy))


def _rebuild(obj: Any, copies: _Copies) -> Any:
    """``obj`` with its tensors replaced by the ones the trace runs on, and
    with everything a trace could change in it built anew."""
    cls = type(obj)
    if cls in _TENSORS:
        return copies.fakes[id(obj)]
    if cls is tuple:
        return tuple([_rebuild(item, copies) for item in obj])
    copy = copies.of.get(id(obj))
    if copy is not None:
        return copy
    if cls is list:
        copy = copies.of[id(obj)] = []
        copy.extend([_rebuild(item, copies) for item in obj])
        return copy
    if cls is dict or cls is collections.OrderedDict:
        copy = copies.of[id(obj)] = cls()
        for key, value in obj.items():
            copy[key] = _rebuild(value, copies)
        return copy
    if cls is set:
        copy = copies.of[id(obj)] = set(obj)
        return copy
    if cls is types.CellType:
        cell: Any = types.CellType()
        copies.of[id(obj)] = cell
        try:
            cell.cell_contents = _rebuild(obj.cell_contents, copies)
        except ValueError:
            pass
        return cell
    if cls is types.MethodType:
        receiver = _rebuild(obj.__self__, copies)
        return (
            obj
            if receiver is obj.__self__
            else types.MethodType(obj.__func__, receiver)
        )
    if cls is types.BuiltinFunctionType:
        receiver = _rebuild(obj.__self__, copies)
        return obj if receiver is obj.__self__ else getattr(receiver, obj.__name__)
    if isinstance(obj, tuple) and not isinstance(obj, torch.Size):
        items = [_rebuild(item, copies) for item in obj]
        return cls(*items) if hasattr(cls, "_fields") else cls(items)
    node = pytree.SUPPORTED_NODES.get(cls)
    if node is not None and not isinstance(obj, nn.Module):
        children, context = node.flatten_fn(obj)
        children = [_rebuild(item, copies) for item in children]
        copy = copies.of[id(obj)] = node.unflatten_fn(children, context)
        copies.nodes.append((copy, children))
        return copy
    if not isinstance(obj, nn.Module) and _has_plain_state(obj):
        copy = copies.of[id(obj)] = object.__new__(cls)
        for name, value in vars(obj).items():
            copy.__dict__[name] = _rebuild(value, copies)
        return copy
    return obj


def _same_items(now: list[Any], before: list[Any]) -> bool:
    return len(now) == len(before) and all(a is b for a, b in zip(now, before))


def _unchanged(copy: Any, children: list[Any]) -> bool:
    """Whether the copy of a pytree node still holds what it started with."""
    return _same_items(pytree.SUPPORTED_NODES[type(copy)].flatten_fn(copy)[0], children)


def _effects(copies: _Copies, recipes: "_Recipes") -> tuple[list[Any], bool]:
    """What a trace did to the copies of what the region was given, as what
    to do to the real things after the graph, and whether one of them is not
    of the shape it was (more items, other keys, other attributes): the next
    call then finds the region's inputs changed."""
    effects: list[Any] = []
    grows = False
    for k, (copy, before) in enumerate(copies.tracked):
        cls = type(copy)
        if cls is list:
            if not _same_items(copy, before):
                grows |= len(copy) != len(before)
                effects.append(("list", k, [recipes.of(item) for item in copy]))
        elif cls is set:
            if copy != before:
                if not all(type(item) in _VALUES for item in copy):
                    raise Untraceable("puts something that is no plain value in a set")
                grows = True
                effects.append(("set", k, frozenset(copy)))
        elif cls is types.CellType:
            now = _state_of(copy)
            if not _same_items(now, before):
                if not now:
                    raise Untraceable("deletes a variable of an enclosing function")
                effects.append(("cell", k, recipes.of(now[0])))
        else:
            now = copy if isinstance(copy, dict) else vars(copy)
            if list(now) == list(before) and _same_items(
                list(now.values()), list(before.values())
            ):
                continue
            if isinstance(copy, dict):
                for key in now:
                    if type(key) not in _VALUES:
                        raise Untraceable("gives a dict a key that is no plain value")
                grows |= list(now) != list(before)
                parts = [recipes.of(item) for item in now.values()]
                effects.append(("dict", k, list(now), parts))
            else:
                grows |= set(now) != set(before)
                changed = [
                    name
                    for name, value in now.items()
                    if name not in before or before[name] is not value
                ]
                parts = [recipes.of(now[name]) for name in changed]
                gone = [name for name in before if name not in now]
                effects.append(("attrs", k, changed, parts, gone))
    return effects, grows


def _checked(
    copies: _Copies,
    fake: Any,
    grads: list[bool],
    reachable: set[int],
    given: dict[int, int],
) -> "_Recipes":
    """After the instructions of a trace ran: what a region must not have
    done, and the recipes for what it gives back."""
    for copy, children in copies.nodes:
        if not _unchanged(copy, children):
            raise _Failed(None, f"changes a {type(copy).__name__} it was given")
    if grads != [f.requires_grad for f in fake]:
        raise _Failed(None, "changes requires_grad of a tensor it was given")
    return _Recipes(reachable, given, {id(f): i for i, f in enumerate(fake)})


def _effects_of(
    copies: _Copies, recipes: "_Recipes", stored: list[tuple[nn.Module, str, Any]]
) -> tuple[list[Any], bool]:
    """What a region does after its graph has run: to what it was given,
    and to the modules whose tensors its trace replaced."""
    try:
        effects, grows = _effects(copies, recipes)
        for module, name, value in stored:
            effects.append(("module", module, name, recipes.of(value)))
        return effects, grows
    except Untraceable as e:
        raise _Failed(None, str(e)) from None


def _apply(
    effect: Any, results: Any, built: dict[int, Any], flat: Any, held: Any
) -> None:
    kind = effect[0]
    if kind == "module":
        setattr(effect[1], effect[2], _build(effect[3], results, built, flat, held))
        return
    target = held[effect[1]]
    if kind == "list":
        target[:] = [_build(part, results, built, flat, held) for part in effect[2]]
    elif kind == "set":
        target.clear()
        target.update(effect[2])
    elif kind == "cell":
        target.cell_contents = _build(effect[2], results, built, flat, held)
    elif kind == "dict":
        values = [_build(part, results, built, flat, held) for part in effect[3]]
        target.clear()
        target.update(zip(effect[2], values))
    else:
        state = vars(target)
        for name, part in zip(effect[2], effect[3]):
            state[name] = _build(part, results, built, flat, held)
        for name in effect[4]:
            state.pop(name, None)


def _identities(
    obj: Any, modules: dict[int, nn.Module], others: list[Any], seen: set[int]
) -> None:
    """The modules in an input of a region, and the other objects that
    ``_describe`` names by their id."""
    if isinstance(obj, torch.Tensor) or type(obj) in _VALUES or id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, nn.Module):
        modules[id(obj)] = obj
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            _identities(item, modules, others, seen)
    elif isinstance(obj, dict):
        for item in obj.values():
            _identities(item, modules, others, seen)
    elif type(obj) in (types.MethodType, types.BuiltinFunctionType):
        others.append(obj)
        _identities(obj.__self__, modules, others, seen)
    elif type(obj) is types.CellType:
        try:
            _identities(obj.cell_contents, modules, others, seen)
        except ValueError:
            pass
    elif type(obj) in pytree.SUPPORTED_NODES:
        for item in pytree.SUPPORTED_NODES[type(obj)].flatten_fn(obj)[0]:
            _identities(item, modules, others, seen)
    elif _has_plain_state(obj):
        for item in vars(obj).values():
            _identities(item, modules, others, seen)
    else:
        others.append(obj)


def _same_but_values(a: Any, b: Any) -> bool:
    """Whether two keys of variants differ in numbers and strings only."""
    if type(a) is not tuple or type(b) is not tuple:
        return a == b
    if len(a) != len(b):
        return False
    if len(a) == 2 and a[0] is b[0] and a[0] in (int, float, str, bool):
        return True
    return all(_same_but_values(x, y) for x, y in zip(a, b))


def _modes() -> tuple[Any, ...]:
    return (
        torch.is_grad_enabled(),
        torch.is_inference_mode_enabled(),
        torch.is_autocast_enabled("cuda"),
        torch.is_autocast_enabled("cpu"),
        torch.get_default_dtype(),
    )


def _trace(fn: Callable[..., Any], example: list[Any]) -> Any:
    """The graph of what ``fn`` does to tensors like ``example``."""
    from torch.fx.experimental.proxy_tensor import make_fx

    error: BaseException | None = None
    graph = None
    # What the trace makes is told from what was there before by its age
    # (``_Recipes.made``): everything older is out of the youngest
    # generation now, and nothing leaves it while the trace runs.
    collecting = gc.isenabled()
    gc.collect(0)
    gc.disable()
    try:
        graph = make_fx(fn, tracing_mode="fake")(*example)
    except Exception as e:
        error = e
    finally:
        if collecting:
            gc.enable()
    if error is None:
        return graph
    if isinstance(error, (_Failed, _Retry)):
        raise error
    reason = f"{type(error).__name__}: {error}".split("\n")[0][:200]
    raise _Failed(getattr(error, "cfg_index", None), reason) from None


# ----------------------------------------------------------------------
# Tracing
# ----------------------------------------------------------------------


_GRAD_READERS = ("torch/_subclasses/meta_utils.py", "torch/_tensor.py")


def _refuse(what: str) -> Callable[..., Any]:
    def refused(*args: Any, **kwargs: Any) -> Any:
        raise Untraceable(what)

    return refused


class _NoEngine:
    # Stands in for the autograd engine while a trace runs.
    def __getattr__(self, name: str) -> Any:
        raise Untraceable("runs the autograd engine")


def _set_grad_enabled_checked(mode: bool) -> None:
    # The graph does not say where gradients were switched off, so the
    # backward that is derived from it would run through such a stretch.
    if bool(mode) != torch.is_grad_enabled():
        raise Untraceable("changes whether gradients are recorded")


@contextlib.contextmanager
def _traps(modules: dict[int, nn.Module]) -> Any:
    """Turns what a trace would silently get wrong into an error."""
    python_random = random.getstate()
    numpy = sys.modules.get("numpy")
    numpy_random = numpy.random.get_state() if numpy is not None else None
    torch_random = torch.get_rng_state()
    reachable: set[int] = set()
    for module in modules.values():
        reachable |= _module_state(module).ids
    # Nothing of torch.Tensor or nn.Module is replaced: see
    # ``_watch_module_changes``.
    patches = [
        (torch.autograd, "backward", _refuse("calls torch.autograd.backward")),
        (torch.autograd, "grad", _refuse("calls torch.autograd.grad")),
        (torch.autograd.Variable, "_execution_engine", _NoEngine()),
        (torch._C, "_set_grad_enabled", _set_grad_enabled_checked),
    ]
    saved = [(owner, name, owner.__dict__.get(name)) for owner, name, _ in patches]
    watch: _Watch | None = None
    refuse_grad = _refuse("reads or sets the .grad of a tensor")
    real_grad = torch._C.TensorBase.grad.__get__  # type: ignore[attr-defined]

    def read_grad(tensor: torch.Tensor) -> Any:
        # Making a fake tensor of a real one looks at its grad; that is the
        # machinery of the trace, not the program.
        if sys._getframe(1).f_code.co_filename.endswith(_GRAD_READERS):
            return real_grad(tensor)
        return refuse_grad()

    try:
        for owner, name, replacement in patches:
            setattr(owner, name, replacement)
        # Shadows the attribute of torch.Tensor for the tensors of the trace.
        FakeTensor.grad = property(read_grad, refuse_grad, refuse_grad)  # type: ignore[assignment,method-assign]
        _state.traced = reachable
        _state.watch = watch = _Watch(modules)
        _watch_module_changes(True)
        # Library code that asks torch.compiler.is_compiling() takes the path
        # it takes under torch.compile.
        with torch.compiler._compile_session_context():
            yield
    finally:
        _state.traced = None
        _state.watch = None
        _watch_module_changes(False)
        if watch is not None:
            # Also what a module derives from its parameters and keeps, as an
            # RNN does with its weights.
            watch.restore()
        if "grad" in vars(FakeTensor):
            del FakeTensor.grad
        for owner, name, original in saved:
            if original is None:
                delattr(owner, name)
            else:
                setattr(owner, name, original)
    # A trace is an extra run: the generators are put back where they were,
    # and a region that drew from one that a graph does not know is no region.
    torch.set_rng_state(torch_random)
    if random.getstate() != python_random:
        random.setstate(python_random)
        raise Untraceable("draws from Python's random module")
    if numpy_random is not None:
        now = sys.modules["numpy"].random.get_state()
        if numpy_random[2] != now[2] or (numpy_random[1] != now[1]).any():
            sys.modules["numpy"].random.set_state(numpy_random)
            raise Untraceable("draws from numpy's global random state")


@contextlib.contextmanager
def _swapped(modules: dict[int, nn.Module], fakes: dict[int, torch.Tensor]) -> Any:
    """Puts the tensors of the trace in place of the modules' own."""
    originals = []
    try:
        for module in modules.values():
            for store, name in _module_state(module).accessors:
                tensor = store[name]
                if id(tensor) in fakes:
                    originals.append((store, name, tensor))
                    store[name] = fakes[id(tensor)]
        yield
    finally:
        for store, name, tensor in originals:
            store[name] = tensor


def _static_output(leaf: Any, reachable: set[int]) -> bool:
    """Whether a compiled region may return ``leaf`` itself on every call."""
    cls = type(leaf)
    if cls is types.BuiltinFunctionType or cls is types.MethodType:
        # A method is as good as what it is bound to: not one of a tensor of
        # the trace, or of the copy of a list the trace ran on.
        receiver = leaf.__self__
        if receiver is None or type(receiver) is types.ModuleType:
            return True
        return type(receiver) not in _CONTAINERS and _static_output(receiver, reachable)
    if cls in _VALUES or cls in _PLAIN or cls in _TENSORS:
        return True
    if isinstance(leaf, nn.Module):
        return id(leaf) in reachable
    return isinstance(leaf, (type, enum.Enum))


def _is_plain_instance(obj: Any) -> bool:
    """Whether ``obj`` is its class and its ``__dict__`` and nothing more, as
    ``copy.copy`` assumes of an object that does not say otherwise."""
    cls = type(obj)
    if isinstance(obj, (nn.Module, BaseException, type)) or not hasattr(
        obj, "__dict__"
    ):
        return False
    for base in cls.__mro__[:-1]:
        if not base.__flags__ & _HEAPTYPE or vars(base).get("__slots__"):
            return False
    return (
        cls.__reduce_ex__ is object.__reduce_ex__
        and cls.__reduce__ is object.__reduce__
        and cls.__getstate__ is object.__getstate__
        and not hasattr(cls, "__setstate__")
    )


_PLAIN_STATE: dict[type, bool] = {}


def _has_plain_state(obj: Any) -> bool:
    """Whether ``obj`` is its class and its ``__dict__``, read and written
    the way ``object`` does it: then what a trace does to a copy of it can be
    seen, and done to the object itself afterwards."""
    cls = type(obj)
    known = _PLAIN_STATE.get(cls)
    if known is None:
        known = _PLAIN_STATE[cls] = (
            _is_plain_instance(obj)
            and cls.__setattr__ is object.__setattr__
            and cls.__delattr__ is object.__delattr__
            and cls.__getattribute__ is object.__getattribute__
            and not hasattr(cls, "__getattr__")
            and not hasattr(cls, "__del__")
        )
    return known


class _Recipes:
    """How to make the outputs of a trace again from the results of its graph.

    A recipe is a list ``[kind, dynamic, ...]``: ``result`` (a tensor the
    graph returns), ``input`` (a tensor the region was given), ``given`` (a
    list, dict or object it was given), ``same`` (the object itself, on every
    call), ``sequence``, ``dict``, ``node`` (a pytree node) or ``instance``
    (an object the region made: a new one of its class with the same
    attributes). ``dynamic`` says whether something of the call is inside.
    """

    def __init__(
        self, reachable: set[int], given: dict[int, int], inputs: dict[int, int]
    ) -> None:
        self.reachable = reachable
        # Copies of what the region was given, and the tensors of the trace
        # that stand for its inputs: the call has the real ones.
        self.given = given
        self.inputs = inputs
        self.results: list[torch.Tensor] = []
        self.memo: dict[int, list[Any]] = {}
        self.alive: list[Any] = []
        self.young: set[int] | None = None

    def made(self, obj: Any) -> bool:
        """Whether the trace made ``obj``, rather than found it."""
        if self.young is None:
            self.young = {id(o) for o in gc.get_objects(generation=0)}
        return id(obj) in self.young

    def of(self, obj: Any, depth: int = 0) -> list[Any]:
        known = self.memo.get(id(obj))
        if known is not None:
            return known
        if depth > 40:
            raise Untraceable("gives an object that is nested too deeply")
        cls = type(obj)
        children: list[Any] | None = None
        recipe: list[Any]
        if id(obj) in self.given:
            recipe = ["given", True, self.given[id(obj)]]
        elif id(obj) in self.inputs:
            recipe = ["input", True, self.inputs[id(obj)]]
        elif is_fake_tensor(obj):
            recipe = ["result", True, len(self.results)]
            self.results.append(obj)
        elif _static_output(obj, self.reachable):
            recipe = ["same", False, obj]
        elif cls in (tuple, list, set, frozenset):
            recipe = ["sequence", False, cls]
            children = list(obj)
        elif cls is dict or cls is collections.OrderedDict:
            recipe = ["dict", False, cls, list(obj)]
            children = list(obj.values())
        elif cls in pytree.SUPPORTED_NODES:
            children, context = pytree.SUPPORTED_NODES[cls].flatten_fn(obj)
            recipe = ["node", False, pytree.SUPPORTED_NODES[cls].unflatten_fn, context]
        elif isinstance(obj, tuple) and hasattr(cls, "_fields"):
            recipe = ["sequence", False, cls._make]
            children = list(obj)
        elif _is_plain_instance(obj):
            recipe = ["instance", False, cls, list(vars(obj))]
            children = list(vars(obj).values())
        else:
            raise Untraceable(f"gives a {cls.__name__}, which a graph cannot return")
        # Before its parts, so that an object that refers to itself ends.
        self.memo[id(obj)] = recipe
        # Kept alive, so that no other object gets its id while the trace runs.
        self.alive.append(obj)
        if children is not None:
            parts = [self.of(child, depth + 1) for child in children]
            recipe.append(parts)
            recipe[1] = any(part[1] for part in parts)
            if recipe[0] == "instance" and not recipe[1] and not self.made(obj):
                # Nothing of the graph in it, and there before the trace: a
                # configuration, say. One the region makes (an empty cache)
                # is made on every call.
                recipe[:] = ["same", False, obj]
        return recipe


def _build(
    recipe: list[Any], results: Any, built: dict[int, Any], flat: Any, held: Any
) -> Any:
    kind = recipe[0]
    if kind == "result":
        return results[recipe[2]]
    if kind == "same":
        return recipe[2]
    if kind == "input":
        return flat[recipe[2]]
    if kind == "given":
        return held[recipe[2]]
    value = built.get(id(recipe))
    if value is not None:
        return value
    if kind == "instance":
        value = built[id(recipe)] = object.__new__(recipe[2])
        for name, part in zip(recipe[3], recipe[4]):
            value.__dict__[name] = _build(part, results, built, flat, held)
        return value
    parts = [_build(part, results, built, flat, held) for part in recipe[-1]]
    if kind == "sequence":
        value = recipe[2](parts)
    elif kind == "dict":
        value = recipe[2](zip(recipe[3], parts))
    else:
        value = recipe[2](parts, recipe[3])
    built[id(recipe)] = value
    return value


class _Variant:
    """A region compiled for one combination of inputs."""

    __slots__ = (
        "compiled",
        "direct",
        "effects",
        "grows",
        "guards",
        "keep",
        "leap",
        "ops",
        "proven",
        "recipes",
    )

    def __init__(
        self, compiled: Callable[..., Any], recipes: list[Any], ops: int, keep: Any
    ) -> None:
        self.compiled = compiled
        # One per output of the region.
        self.recipes = recipes
        # The outputs are exactly the results of the graph, in order.
        self.direct = all(r[0] == "result" and r[2] == i for i, r in enumerate(recipes))
        self.ops = ops
        self.keep = keep
        # Whether it has run once without an error.
        self.proven = False
        # For a region that goes on into other blocks: where the VM goes on
        # (block, instruction), the values the recipes are of, and the ones
        # that are unset there; and what the path read that is no input.
        self.leap: tuple[int, int, list[int], list[int]] | None = None
        self.guards: list[tuple[str, Any, str, Any]] = []
        # What the region does to the lists, dicts and objects it was given,
        # once its graph has run (see ``_effects``), and whether that leaves
        # one of them with other items, keys or attributes than it had.
        self.effects: list[Any] = []
        self.grows = False

    def run(self, inputs: _Inputs) -> Any:
        flat, held = inputs.flat, inputs.held
        results = self.compiled(*flat)
        if self.direct:
            return tuple(results)
        built: dict[int, Any] = {}
        made = [_build(recipe, results, built, flat, held) for recipe in self.recipes]
        for effect in self.effects:
            _apply(effect, results, built, flat, held)
        if self.leap is None:
            return tuple(made)
        block, index, values, unset = self.leap
        return (block, index, dict(zip(values, made)), unset)

    def affects(self, effects: list[Any], grows: bool) -> None:
        self.effects, self.grows = effects, grows
        self.direct = self.direct and not effects

    def holds(self) -> bool:
        """Whether what the path read is still what it was."""
        for kind, owner, name, value in self.guards:
            now = (
                getattr(owner, name, self) if kind == "attr" else owner.get(name, self)
            )
            if now is not value and (type(now) is not type(value) or now != value):
                return False
        return True


class _Region:
    """A run of instructions of a block; what the VM calls in their place."""

    def __init__(self, plan: "_Plan", lo: int, hi: int) -> None:
        self.plan = plan
        self.lo = lo
        self.hi = hi
        self.inputs, self.outputs = plan.replay.io(lo, hi)
        # None: these inputs run the instructions as they are.
        self.variants: dict[Any, _Variant | None] = {}
        self.hot = False
        # Whether the instructions were checked on real values.
        self.surveyed = False
        self.calls = 0
        # Graphs given to the backend, and the time traces and backend took.
        self.compiles = 0
        self.seconds = 0.0

    def __call__(self, *args: Any) -> Any:
        if not self.hot:
            return self.cold(args)
        inputs = _Inputs()
        try:
            key = self.key(args, inputs)
        except _Opaque:
            return self.decline(args)
        variant = self.variants.get(key, self)
        if variant is self or (variant is not None and not variant.holds()):  # type: ignore[union-attr]
            variant = self.specialize(key, args, inputs)
        if variant is None:
            return self.decline(args)
        if variant.proven:  # type: ignore[union-attr]
            return variant.run(inputs)  # type: ignore[union-attr]
        return self.first_run(key, variant, args, inputs)  # type: ignore[arg-type]

    def install(self, region: Any) -> bool:
        """Puts the region into the block a pass is working on."""
        return region.fuse(self.lo, self.hi, self)

    def bind(self, args: tuple[Any, ...]) -> dict[int, Any]:
        """The values the instructions start from."""
        return dict(zip(self.inputs, args))

    def give(self, env: dict[int, Any]) -> Any:
        """What the VM gets after the instructions ran on ``env``."""
        return tuple([env[value] for value in self.outputs])

    def decline(self, args: tuple[Any, ...]) -> Any:
        """For inputs there is no graph for: the instructions as they are."""
        return self.replay(args)

    def first_run(
        self, key: Any, variant: "_Variant", args: tuple[Any, ...], inputs: _Inputs
    ) -> tuple[Any, ...]:
        """Runs a variant that has not run yet. What a backend made can fail
        where the instructions would not (or would with another error): then
        they run instead, this time and from now on."""
        generator = torch.get_rng_state()
        try:
            outputs = variant.run(inputs)
        except Exception as e:
            torch.set_rng_state(generator)
            self.variants[key] = None
            if not any(self.variants.values()):
                self.hot = False
                reason = f"{type(e).__name__}: {e}".split("\n")[0][:200]
                self.plan.split(self, None, f"the compiled graph failed: {reason}")
            return self.decline(args)
        variant.proven = True
        return outputs

    def key(self, args: tuple[Any, ...], inputs: _Inputs) -> Any:
        parts = []
        for position, arg in enumerate(args):
            try:
                parts.append(_describe(arg, inputs))
            except _Opaque as e:
                raise _Opaque(position, e.reason) from None
        return _key(parts, inputs)

    def replay(self, args: tuple[Any, ...]) -> Any:
        env = self.bind(args)
        self.plan.replay.run(self.lo, self.hi, env)
        return self.give(env)

    def cold(self, args: tuple[Any, ...]) -> Any:
        # An enclosing region or block that has not made up its mind may
        # still take all of this one in.
        ready = (
            self.calls >= self.plan.options.warmup
            and _state.pending == 0
            and _state.traced is None
            and _runtime.profiling_depth() == 0
        )
        if ready and self.surveyed and not self.plan.options.held_back(self):
            self.decide(args)
            if self.hot:
                return self(*args)
        self.calls += 1
        _state.pending += 1
        try:
            if not ready or self.surveyed:
                return self.replay(args)
            # Compiled at the next call, not now: the regions of the blocks
            # that follow are surveyed in this call too, and if this one
            # goes on into those blocks, their graphs would be for nothing.
            return self.survey(args)
        finally:
            _state.pending -= 1

    def survey(self, args: tuple[Any, ...]) -> Any:
        """Runs the instructions one by one and takes those out of the plan
        that a region cannot hold, on the values of this call."""
        plan = self.plan
        env = self.bind(args)
        local: dict[int, Any] = {}
        bad: dict[int, str] = {}
        tensors = 0
        refused: dict[int, str | None] = {}
        handed = _Inputs()
        for value, arg in zip(self.inputs, args):
            try:
                _describe(arg, handed)
                refused[value] = None
            except _Opaque as e:
                refused[value] = e.reason
        given = {id(obj) for obj in handed.held}
        for i in range(self.lo, self.hi):
            inst = plan.insts[i]
            reason = _problem(inst, env, local, given)
            if reason is None and inst["op"] != "hold":
                reason = next(
                    (refused[v] for v in inst["uses"] if refused.get(v)), None
                )
                reason = reason and "takes " + reason
            if reason is not None:
                bad[i] = reason
            elif inst["op"] in _WORK:
                tensors += any(_involves_tensor(env.get(v)) for v in inst["uses"])
            plan.replay.run(i, i + 1, env)
            _note_local(inst, env, local)
        outputs = self.give(env)
        if bad:
            plan.split(self, bad)
        elif tensors == 0:
            plan.split(self, None, "no tensor or module involved")
        else:
            self.surveyed = True
            plan.options.wait(self)
        return outputs

    def decide(self, args: tuple[Any, ...]) -> None:
        """Compiles the region for the first time, or takes it out of the plan."""
        inputs = _Inputs()
        try:
            key = self.key(args, inputs)
        except _Opaque as e:
            value = self.inputs[e.position]
            users = [
                i
                for i in range(self.lo, self.hi)
                if value in self.plan.insts[i]["uses"]
                and self.plan.insts[i]["op"] != "hold"
            ]
            reason = "takes " + e.reason
            self.plan.split(
                self, dict.fromkeys(users, reason) if users else None, reason
            )
            return
        try:
            self.variants[key] = self.compile(key, args, inputs)
            self.hot = True
        except _Failed as e:
            self.plan.split(
                self, None if e.index is None else {e.index: e.reason}, e.reason
            )

    def specialize(
        self, key: Any, args: tuple[Any, ...], inputs: _Inputs
    ) -> _Variant | None:
        """Another variant of a region that has one, for inputs not seen yet."""
        if len(self.variants) >= 3 and all(
            _same_but_values(key, k) for k in self.variants
        ):
            # A loop counter, say: there is no end to these, and every one
            # would be a graph of its own.
            self.hot = False
            self.plan.split(self, None, "its inputs take ever new values")
            return None
        if (
            key not in self.variants
            and len(self.variants) >= self.plan.options.max_variants
        ):
            return None
        growing = sum(1 for v in self.variants.values() if v is not None and v.grows)
        try:
            variant: _Variant | None = self.compile(key, args, inputs)
        except _Failed:
            # These inputs do what the first ones did not: they run the
            # instructions as they are.
            variant = None
        if variant is not None and variant.grows and growing >= _MOST_GROWTH:
            # A list it appends to on every call, say: what it is given is
            # never what a graph was made for.
            self.hot = False
            self.plan.split(self, None, "changes what it is given on every call")
            return None
        self.variants[key] = variant
        return variant

    def compile(self, key: Any, args: tuple[Any, ...], inputs: _Inputs) -> _Variant:
        start = time.perf_counter()
        flat, held = inputs.flat, inputs.held
        modules: dict[int, nn.Module] = {}
        # What the key names by id is kept alive, so the ids stay theirs. Not
        # the tensors: a variant must not hold on to the batch it was made on.
        keep: list[Any] = []
        visited: set[int] = set()
        for arg in args:
            _identities(arg, modules, keep, visited)
        keep.extend(modules.values())
        reachable: set[int] = set()
        for module in modules.values():
            reachable |= _module_state(module).ids
        found: dict[str, Any] = {}

        def region(*fake: torch.Tensor) -> tuple[Any, ...]:
            fakes = {id(real): f for real, f in zip(flat, fake)}
            copies = _Copies(fakes)
            env = dict(zip(self.inputs, [_rebuild(arg, copies) for arg in args]))
            copies.track(held)
            given = copies.given()
            grads = [f.requires_grad for f in fake]
            local: dict[int, Any] = {}
            insts, replay = self.plan.insts, self.plan.replay
            with _swapped(modules, fakes), _traps(modules):
                for i in range(self.lo, self.hi):
                    reason = _problem(insts[i], env, local, given)
                    if reason is not None:
                        raise _Failed(i, reason)
                    replay.run(i, i + 1, env, True)
                    if _state.watch is not None and _state.watch.changed():
                        raise _Failed(i, "changes a module the region was given")
                    _note_local(insts[i], env, local)
                    if _is_outside_tensor(env.get(insts[i]["dst"])):
                        # Read from a global, a closure or a plain object. Left
                        # to the VM, it is an input of the next region.
                        raise _Failed(i, "gives a tensor from outside the region")
                stored = _state.watch.stored() if _state.watch is not None else []
            recipes = _checked(copies, fake, grads, reachable, given)
            found["recipes"] = []
            for value in self.outputs:
                try:
                    found["recipes"].append(recipes.of(env[value]))
                except Untraceable as e:
                    raise _Failed(self.definition(value), str(e)) from None
            found["effects"] = _effects_of(copies, recipes, stored)
            return tuple(recipes.results)

        graph = _trace(region, list(flat))
        ops = self.ops_of(graph)
        variant = _Variant(self.finish(graph, inputs, key), found["recipes"], ops, keep)
        variant.affects(*found["effects"])
        self.seconds += time.perf_counter() - start
        _runtime.plan_changed()
        return variant

    @staticmethod
    def ops_of(graph: torch.fx.GraphModule) -> int:
        return sum(1 for node in graph.graph.nodes if node.op == "call_function")

    def finish(self, graph: torch.fx.GraphModule, inputs: _Inputs, key: Any) -> Any:
        """Hands a traced graph to the backend, unless it has seen it."""
        options = self.plan.options
        ops = self.ops_of(graph)
        if ops < options.min_ops:
            raise _Failed(None, f"{ops} tensor operations, not worth a graph")
        _views_to_reshapes(graph)
        # The parameters are inputs, so the layers of a model that are built
        # alike give the same graph: it is compiled once. A graph that holds a
        # tensor of its own is not looked up, the code does not say which.
        same = None
        if not any(node.op == "get_attr" for node in graph.graph.nodes):
            same = (graph.code, key[1], key[2], key[3])
        compiled = options.compiled.get(same)
        if compiled is None:
            try:
                compiled = options.backend(graph, list(inputs.flat))
            except Exception as e:
                reason = f"{type(e).__name__}: {e}".split("\n")[0][:200]
                raise _Failed(None, f"the backend failed: {reason}") from None
            if same is not None:
                options.compiled[same] = compiled
            self.compiles += 1
        return compiled

    def definition(self, value: int) -> int | None:
        """The instruction of the region that defines a value."""
        for i in range(self.lo, self.hi):
            if self.plan.insts[i]["dst"] == value:
                return i
        return None

    def __repr__(self) -> str:
        return f"{self.plan.name}[{self.lo}:{self.hi}]"


class _Retry(Exception):
    # The trace of a path has to be made again: over fewer blocks, or with
    # other inputs.
    def __init__(
        self, limit: int | None = None, positions: list[int] | None = None
    ) -> None:
        super().__init__()
        self.limit = limit
        self.positions = positions


# How many instructions a path may run: a loop over plain values has no end
# that a trace would notice.
_PATH_STEPS = 20000


class _Walk:
    """Takes a region on from its block: runs the instructions of the path
    the values of a trace take through the function, one by one, until one
    comes that a region cannot hold or that the trace cannot decide."""

    def __init__(
        self,
        region: "_Path",
        env: dict[int, Any],
        refused: set[int],
        limit: int | None,
        handed: dict[int, int],
    ) -> None:
        self.region = region
        self.plan = region.plan
        self.env = env
        # The copies of the lists, dicts and objects the region was given.
        self.handed = handed
        # Values from before the region that cannot be inputs of a graph.
        self.refused = refused
        # How many blocks to go on into: None for as far as it gets, -1 to
        # stop where the region ends in its own block.
        self.limit = limit
        self.block = self.plan.block
        self.index = region.lo
        self.entered = 0
        self.steps = 0
        self.given = set(env) | refused
        # Of the values from before: the ones the path looked at.
        self.used: set[int] = set()
        # What the path defined, and how many blocks in it was (None: among
        # the instructions of the region itself).
        self.born: dict[int, int | None] = {}
        self.local: dict[int, Any] = {}
        self.guards: list[tuple[str, Any, str, Any]] = []

    def read(self, value: int) -> None:
        if value in self.given and value not in self.born:
            self.used.add(value)

    def run(self) -> tuple[int, int]:
        """Returns where the path ends: a block and an instruction of it."""
        region, env = self.region, self.env
        while True:
            insts = self.plan.insts_of(self.block)
            while self.index < len(insts):
                inst = insts[self.index]
                inside = self.entered == 0 and self.index < region.hi
                if inside:
                    reason = _problem(inst, env, self.local, self.handed)
                    if reason is not None:
                        raise _Failed(self.index, reason)
                elif (self.limit == -1 and self.entered == 0) or self.stops(inst):
                    return self.block, self.index
                before = dict(env)
                receiver = (
                    env.get(inst["uses"][0]) if inst["op"] == "load_attr" else None
                )
                try:
                    self.plan.replay.step(
                        self.block, self.index, self.index + 1, env, True
                    )
                    outside = _is_outside_tensor(env.get(inst["dst"]))
                except Exception:
                    if inside:
                        raise
                    outside = True
                    began = inst["op"] not in _READS
                else:
                    began = outside and inst["op"] not in _READS
                watch = _state.watch
                if watch is not None and watch.changed():
                    if inside:
                        raise _Failed(
                            self.index, "changes a module the region was given"
                        )
                    outside = began = True
                if outside:
                    if inside:
                        raise _Failed(
                            self.index, "gives a tensor from outside the region"
                        )
                    if began:
                        # It ran, in part at least, and what it did up to
                        # there is in the trace (an append to a list the
                        # region was given, a tensor changed in place), while
                        # the VM is to run all of it. The path is traced
                        # again, and ends in front of it.
                        self.plan.fails.add((self.block, self.index))
                        raise _Retry
                    # The VM runs this one: the path ends before it.
                    env.clear()
                    env.update(before)
                    return self.block, self.index
                if inst["op"] != "hold":
                    for value in inst["uses"]:
                        self.read(value)
                if inst["dst"] is not None:
                    self.born[inst["dst"]] = None if inside else self.entered
                self.note(inst, receiver)
                self.index += 1
                self.steps += 1
            if not self.leave():
                return self.block, self.index

    def stops(self, inst: dict[str, Any]) -> bool:
        """Whether the path ends in front of an instruction."""
        op, uses, env = inst["op"], inst["uses"], self.env
        if not inst["steppable"] or op in ("frame_enter", "frame_leave"):
            return True
        if self.steps >= _PATH_STEPS or (self.block, self.index) in self.plan.fails:
            return True
        if op == "hold":
            return False
        if any(value in self.refused or value not in env for value in uses):
            return True
        if op == "set_item" and type(env[uses[0]]) in (list, dict):
            # On what the path made itself, or on a copy of what it was
            # given: the store is then done to the real one after the graph.
            target = id(env[uses[0]])
            return target not in self.local and target not in self.handed
        if op == "iter_next" or op == "set_item" or op in _MUTATORS:
            # Only on what the path made itself.
            return id(env[uses[0]]) not in self.local
        return _problem(inst, env, self.local, self.handed) is not None

    def note(self, inst: dict[str, Any], receiver: Any) -> None:
        op, value = inst["op"], self.env.get(inst["dst"])
        _note_local(inst, self.env, self.local)
        if op == "make_cell" or (
            op == "get_iter" and value is not self.env.get(inst["uses"][0])
        ):
            self.local[id(value)] = value
        elif type(value) in _VALUES:
            # A flag of a module or a global: no input of the region, so it
            # is checked on its own.
            if op == "load_attr" and isinstance(
                receiver, (nn.Module, types.ModuleType)
            ):
                self.guards.append(
                    ("attr", receiver, inst["text"].split("'")[1], value)
                )
            elif op == "load_global":
                name = inst["text"].split("'")[1]
                if name in self.plan.globals:
                    self.guards.append(("global", self.plan.globals, name, value))

    def leave(self) -> bool:
        """Goes on into the block the values choose; False if they cannot."""
        kind, value, targets = self.plan.terminator(self.block)
        env = self.env
        if kind == "jump":
            successor = targets[0]
        elif kind == "br":
            if type(env.get(value, self)) not in _VALUES:
                return False
            successor = targets[0] if env[value] else targets[1]
        elif kind == "br_null":
            successor = targets[0] if value not in env else targets[1]
        else:
            return False
        if self.limit is not None and self.entered >= self.limit:
            return False
        if self.steps >= _PATH_STEPS:
            return False
        if value is not None:
            self.read(value)
        moves, kills = self.plan.replay.edge(self.block, successor)
        self.entered += 1
        # The phis of the block: all at once.
        arriving = [
            (to, env.get(source, self) if source is not None else self)
            for to, source in moves
        ]
        for _, source in moves:
            if source is not None:
                self.read(source)
        for to, arrived in arriving:
            self.born[to] = self.entered
            if arrived is self:
                env.pop(to, None)
            else:
                env[to] = arrived
        for dead in kills:
            env.pop(dead, None)
        self.block, self.index = successor, 0
        return True


class _Path(_Region):
    """The last region of a block, which goes on into the blocks after it.

    The VM calls it in front of its instructions, with every value that is
    needed from there on, and gets None (it runs the instructions itself) or
    where to go on and with what (see ``Region.leap``).
    """

    def __init__(self, plan: "_Plan", lo: int, hi: int) -> None:
        super().__init__(plan, lo, hi)
        self.inputs = plan.replay.live(plan.block, lo)
        # Which of the inputs the variants are keyed on: the ones a path
        # looked at. None before the first trace.
        self.positions: tuple[int, ...] | None = None

    def install(self, region: Any) -> bool:
        return region.leap(self.lo, self) == self.inputs

    def bind(self, args: tuple[Any, ...]) -> dict[int, Any]:
        undefined = _runtime.UNDEFINED
        return {v: arg for v, arg in zip(self.inputs, args) if arg is not undefined}

    def give(self, env: dict[int, Any]) -> Any:
        values = {value: env[value] for value in self.outputs if value in env}
        return (self.plan.block, self.hi, values, [])

    def decline(self, args: tuple[Any, ...]) -> Any:
        return None

    def key(self, args: tuple[Any, ...], inputs: _Inputs, positions: Any = None) -> Any:
        parts = []
        for position in positions if positions is not None else self.positions or ():
            try:
                parts.append(_describe(args[position], inputs))
            except _Opaque as e:
                raise _Opaque(position, e.reason) from None
        return _key(parts, inputs)

    def decide(self, args: tuple[Any, ...]) -> None:
        try:
            key, variant = self.follow(args)
        except _Failed as e:
            self.plan.split(
                self, None if e.index is None else {e.index: e.reason}, e.reason
            )
            return
        self.variants[key] = variant
        self.hot = True

    def compile(self, key: Any, args: tuple[Any, ...], inputs: _Inputs) -> _Variant:
        # The key is made again: the path may look at other inputs.
        self.variants.pop(key, None)
        key, variant = self.follow(args)
        self.variants[key] = variant
        return variant

    def specialize(
        self, key: Any, args: tuple[Any, ...], inputs: _Inputs
    ) -> _Variant | None:
        variant = super().specialize(key, args, inputs)
        if variant is None:
            self.variants.pop(key, None)
        return variant

    def follow(self, args: tuple[Any, ...]) -> tuple[Any, _Variant]:
        """Traces the path these inputs take and compiles its graph."""
        start = time.perf_counter()
        undefined = _runtime.UNDEFINED
        refused = {
            self.inputs[p]
            for p, arg in enumerate(args)
            if arg is not undefined and _acceptable(arg) is not None
        }
        positions = [
            p
            for p, arg in enumerate(args)
            if arg is not undefined and self.inputs[p] not in refused
        ]
        limit: int | None = None
        for _ in range(8):
            inputs = _Inputs()
            key = self.key(args, inputs, positions)
            try:
                graph, found, keep = self.trace(args, inputs, positions, refused, limit)
            except _Retry as again:
                limit = again.limit if again.limit is not None else limit
                positions = (
                    again.positions if again.positions is not None else positions
                )
                continue
            # Keyed on what some path looked at, and on nothing else.
            wanted = sorted(set(found["used"]) | set(self.positions or ()))
            if wanted != positions:
                fewer = _Inputs()
                self.key(args, fewer, wanted)
                # The recipes and effects count the tensors and the objects
                # the region was given: they must be the same ones.
                same = _same_items(fewer.flat, inputs.flat) and _same_items(
                    fewer.held, inputs.held
                )
                positions = wanted
                if not same:
                    continue
            break
        else:
            raise _Failed(None, "the path of the region does not settle")
        if tuple(positions) != self.positions:
            self.positions = tuple(positions)
            self.variants.clear()
        key = self.key(args, _Inputs())
        variant = _Variant(
            self.finish(graph, inputs, key), found["recipes"], self.ops_of(graph), keep
        )
        variant.direct = False
        variant.leap = found["leap"]
        variant.guards = found["guards"]
        variant.affects(*found["effects"])
        self.seconds += time.perf_counter() - start
        _runtime.plan_changed()
        return key, variant

    def trace(
        self,
        args: tuple[Any, ...],
        inputs: _Inputs,
        positions: list[int],
        refused: set[int],
        limit: int | None,
    ) -> tuple[Any, dict[str, Any], list[Any]]:
        plan = self.plan
        flat, held = inputs.flat, inputs.held
        undefined = _runtime.UNDEFINED
        modules: dict[int, nn.Module] = {}
        keep: list[Any] = []
        visited: set[int] = set()
        for position in positions:
            _identities(args[position], modules, keep, visited)
        keep.extend(modules.values())
        reachable: set[int] = set()
        for module in modules.values():
            reachable |= _module_state(module).ids
        found: dict[str, Any] = {}
        keyed = set(positions)

        def path(*fake: torch.Tensor) -> tuple[Any, ...]:
            fakes = {id(real): f for real, f in zip(flat, fake)}
            copies = _Copies(fakes)
            env: dict[int, Any] = {}
            for position, (value, arg) in enumerate(zip(self.inputs, args)):
                if position in keyed:
                    env[value] = _rebuild(arg, copies)
                elif arg is not undefined and value in refused:
                    # There for the instructions that only hold it.
                    env[value] = arg
            copies.track(held)
            given = copies.given()
            grads = [f.requires_grad for f in fake]
            walk = _Walk(
                self,
                env,
                refused | {v for p, v in enumerate(self.inputs) if p not in keyed},
                limit,
                given,
            )
            try:
                with _swapped(modules, fakes), _traps(modules):
                    block, index = walk.run()
                    stored = _state.watch.stored() if _state.watch is not None else []
            except Untraceable as e:
                if walk.entered and getattr(e, "cfg_index", None) is None:
                    # One of the generators moved somewhere on the path.
                    raise _Retry(limit=-1) from None
                raise
            if walk.entered == 0 and index < self.hi:
                raise _Failed(index, "the trace ends inside the region")
            rest = plan.insts_of(block)[index:]
            if walk.entered and index and any(inst["op"] in _WORK for inst in rest):
                # More to compile in that block: it is left whole, to its own
                # regions.
                raise _Retry(limit=walk.entered - 1)
            recipes = _checked(copies, fake, grads, reachable, given)
            live = [v for v in plan.replay.live(block, index) if v in walk.born]
            # What an instruction reads from there on is made. What a block
            # only keeps alive to the end of its scope is not: it would be a
            # result of the graph for nothing.
            read = set(plan.replay.live(block, index, True))
            values = [v for v in live if v in env and v in read]
            found["recipes"] = []
            for value in values:
                try:
                    found["recipes"].append(recipes.of(env[value]))
                except Untraceable as e:
                    # The path ends before what cannot be handed back: an
                    # iterator half way through, say.
                    born = walk.born[value]
                    if born is None:
                        raise _Failed(self.definition(value), str(e)) from None
                    raise _Retry(limit=born - 1) from None
            found["leap"] = (block, index, values, [v for v in live if v not in values])
            found["effects"] = _effects_of(copies, recipes, stored)
            found["guards"] = walk.guards
            found["used"] = [
                p for p, v in enumerate(self.inputs) if v in walk.used and p in keyed
            ]
            return tuple(recipes.results)

        graph = _trace(path, list(flat))
        return graph, found, keep


class _Plan:
    """The regions of one basic block. Kept on the block as its note."""

    def __init__(
        self, options: "Regions", replay: Any, insts: list[dict[str, Any]]
    ) -> None:
        self.options = options
        self.replay = replay
        self.name = replay.name
        self.block = replay.block
        self.globals = replay.globals
        # What the blocks of one function have in common.
        self.function = (id(replay.globals), replay.name.rsplit(" (block", 1)[0])
        self.insts = insts
        # The other blocks of the function, as a path comes by them.
        self.blocks: dict[int, list[dict[str, Any]]] = {self.block: insts}
        self.ends: dict[int, Any] = {}
        # Instructions that stay with the VM, and why.
        self.bad: dict[int, str] = {}
        # Instructions of the function, by block, that a path ends in front
        # of: one tried to go through them and could not.
        self.fails: set[tuple[int, int]] = set()
        # Runs that are not worth a region, and why.
        self.skipped: dict[tuple[int, int], str] = {}
        self.regions: list[_Region] = []
        self.lay_out(None)

    def lay_out(self, parent: _Region | None) -> None:
        """Maximal runs of instructions that can be replayed, as regions."""
        existing = {(r.lo, r.hi): r for r in self.regions}
        self.regions = []
        start = None
        defined: set[int] = set()
        for i, inst in enumerate(self.insts):
            usable = inst["replayable"] and i not in self.bad
            if usable and inst["op"] in _MUTATORS and inst["uses"][0] not in defined:
                # Changes a container from before the run.
                usable = False
            if usable:
                if start is None:
                    start, defined = i, set()
                if inst["dst"] is not None:
                    defined.add(inst["dst"])
                continue
            if start is not None:
                self.add(start, i, existing, parent)
            start = None
        if start is not None:
            self.add(start, len(self.insts), existing, parent)
        self.extend()

    def extend(self) -> None:
        """Lets the last region go on into the blocks after this one, if all
        that comes after it in the block is something a path can run."""
        if not self.regions or not self.replay.leaps:
            return
        last = self.regions[-1]
        rest = range(last.hi, len(self.insts))
        open_end = all(self.insts[i]["steppable"] and i not in self.bad for i in rest)
        if open_end == isinstance(last, _Path) or last.hot:
            return
        other = (_Path if open_end else _Region)(self, last.lo, last.hi)
        other.calls, other.surveyed = last.calls, last.surveyed
        self.regions[-1] = other

    def insts_of(self, block: int) -> list[dict[str, Any]]:
        if block not in self.blocks:
            self.blocks[block] = self.replay.instructions_of(block)
        return self.blocks[block]

    def terminator(self, block: int) -> Any:
        if block not in self.ends:
            self.ends[block] = self.replay.terminator(block)
        return self.ends[block]

    def add(
        self,
        lo: int,
        hi: int,
        existing: dict[tuple[int, int], _Region],
        parent: _Region | None,
    ) -> None:
        # Loads and constants at either end are as well off in the VM, and
        # what they give is then an input, which every call checks.
        while lo < hi and self.insts[lo]["op"] not in _WORK:
            lo += 1
        while hi > lo and self.insts[hi - 1]["op"] not in _WORK:
            hi -= 1
        if lo == hi or (lo, hi) in self.skipped:
            return
        region = existing.get((lo, hi))
        if region is None:
            region = _Region(self, lo, hi)
            if parent is not None and parent.lo <= lo and hi <= parent.hi:
                # What was checked on the whole was checked on the part.
                region.calls, region.surveyed = parent.calls, True
        self.regions.append(region)

    def split(
        self, region: _Region, bad: dict[int, str] | None, reason: str = ""
    ) -> None:
        """Takes instructions of a region, or all of it, out of the plan."""
        if bad is None:
            self.skipped[(region.lo, region.hi)] = reason
        else:
            self.bad.update(bad)
        self.lay_out(region)
        # The block is specialized again, around the regions that are left.
        self.replay.settle(self)
        _runtime.plan_changed()


class Regions:
    """The pass that puts regions into a block. One per compiled function."""

    __name__ = "regions"

    def __init__(
        self, backend: Backend, min_ops: int, max_variants: int, warmup: int
    ) -> None:
        self.backend = backend
        self.min_ops = min_ops
        self.max_variants = max_variants
        self.warmup = warmup
        self.plans: list[_Plan] = []
        # What the backend made of each graph, by the code of the graph and
        # what its inputs are like.
        self.compiled: dict[Any, Callable[..., Any]] = {}
        # Per function: the regions that were surveyed and are compiled at
        # their next call, in the order they were surveyed, and how often
        # another region waited for each.
        self.waiting: dict[Any, dict[_Region, int]] = {}

    def wait(self, region: _Region) -> None:
        self.waiting.setdefault(region.plan.function, {})[region] = 0

    def held_back(self, region: _Region) -> bool:
        """Whether a region is to put off its first graph: one that was
        surveyed before it, in the same function, has not made up its mind.
        It may go on into the block of this one (a loop body, say, which is
        called many times before the other is called again), and the graphs
        of this one would then be for nothing."""
        waiting = self.waiting.get(region.plan.function)
        if not waiting:
            return False
        for other in list(waiting):
            if other is region:
                del waiting[other]
                return False
            if other.hot or other not in other.plan.regions:
                del waiting[other]
                continue
            waiting[other] += 1
            if waiting[other] > _MOST_WAITS:
                # It is not called again, it seems.
                del waiting[other]
                continue
            return True
        return False

    def __call__(self, region: Any) -> bool:
        insts = region.instructions()
        if any(inst["op"] in ("call_object", "leap") for inst in insts):
            return False
        plan = region.note
        if plan is None:
            if not _wants_regions(region.filename):
                return False
            if not any(inst["replayable"] and inst["op"] in _WORK for inst in insts):
                return False
            plan = _Plan(self, region.replayer(0, len(insts)), insts)
            region.note = plan
            self.plans.append(plan)
        changed = False
        for part in reversed(plan.regions):
            changed |= part.install(region)
        return changed

    def report(self) -> list[dict[str, Any]]:
        """One entry per region and per instruction that was left to the VM."""
        entries: list[dict[str, Any]] = []
        for plan in self.plans:
            for part in plan.regions:
                variants = [v for v in part.variants.values() if v is not None]
                entries.append(
                    {
                        "where": repr(part),
                        "state": "compiled" if part.hot else "waiting",
                        "variants": len(variants),
                        "compiles": part.compiles,
                        "ends": sorted({v.leap[:2] for v in variants if v.leap}),
                        "ops": max((v.ops for v in variants), default=0),
                        "seconds": part.seconds,
                    }
                )
            for index, reason in plan.bad.items():
                where = f"{plan.name}[{index}]"
                entries.append(
                    {
                        "where": where,
                        "state": "vm",
                        "reason": reason,
                        "text": plan.insts[index]["text"],
                    }
                )
            for (lo, hi), reason in plan.skipped.items():
                entries.append(
                    {
                        "where": f"{plan.name}[{lo}:{hi}]",
                        "state": "vm",
                        "reason": reason,
                    }
                )
        return entries


def _views_to_reshapes(graph: torch.fx.GraphModule) -> None:
    """A trace records a reshape as the view it was for the tensor at hand.
    What the backend makes of the operations before it may lay that tensor
    out otherwise, and then the view is no longer one: a reshape is the view
    wherever there is one."""
    views = (torch.ops.aten.view.default, torch.ops.aten._unsafe_view.default)
    changed = False
    for node in graph.graph.nodes:
        if node.op == "call_function" and node.target in views:
            node.target = torch.ops.aten.reshape.default
            changed = True
    if changed:
        graph.recompile()


def _inductor(graph: torch.fx.GraphModule, inputs: list[Any]) -> Callable[..., Any]:
    return torch._inductor.compile(graph, inputs)


def _aot_eager(
    graph: torch.fx.GraphModule, inputs: list[torch.Tensor]
) -> Callable[..., Any]:
    # AOTAutograd without a compiler behind it: the same graphs and the same
    # compiled autograd function, run by the eager kernels.
    from torch._dynamo.backends.debugging import aot_eager

    return aot_eager(graph, inputs)


_BACKENDS: dict[str, Backend] = {"inductor": _inductor, "aot_eager": _aot_eager}


def make_pass(
    runtime: Any, backend: str | Backend, min_ops: int, max_variants: int, warmup: int
) -> Regions:
    global _runtime
    if isinstance(backend, str):
        if backend not in _BACKENDS:
            raise ValueError(f"unknown backend {backend!r}; known: {sorted(_BACKENDS)}")
        backend = _BACKENDS[backend]
    _runtime = runtime
    return Regions(backend, min_ops, max_variants, warmup)
