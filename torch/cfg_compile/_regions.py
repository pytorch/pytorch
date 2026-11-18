"""Runs of a basic block compiled by a graph backend.

With ``torch.cfg_compile(fn, backend="inductor")`` the VM hands the tensor work
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
classes, and tuples, lists and dicts of those. An instruction that takes
anything else from before the region, say a cache object, stays with the VM:
a trace could change such an object without anyone noticing. What a region
gives back is made of the same, with the tensors from the graph.

A compiled region is valid for the inputs it was traced with. Every call
compares the tensors (shape, dtype, device, ``requires_grad``, which of them
are the same object), the values of numbers and strings, the identity of
functions and classes, the modules (their submodules and ``training`` flags)
and the grad and autocast modes; a new combination is traced and compiled as
another variant. What the trace read from anywhere else, such as an attribute
of a module that is not a tensor or a global that is rebound later, is not
checked.
"""

import collections
import contextlib
import enum
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
from torch._subclasses.fake_tensor import is_fake_tensor


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
    # Ids of the modules whose tensors the trace in progress replaced.
    traced: set[int] | None = None


_state = _State()
_runtime: Any = None

# Instructions that can do tensor work. A region must contain one.
_WORK = frozenset({"call", "call_ex", "binary", "unary", "compare", "get_item"})
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


def _receiver_problem(receiver: Any, name: str, local: dict[int, Any]) -> str | None:
    cls = type(receiver)
    if isinstance(receiver, torch.Tensor):
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
        return f"changes a {cls.__name__} from before the region"
    return f"a method of a {cls.__name__}, which may have effects"


def _callee_problem(callee: Any, local: dict[int, Any]) -> str | None:
    """Why a region cannot hold a call of ``callee``, or None if it can."""
    cls = type(callee)
    if cls is types.FunctionType or isinstance(callee, (nn.Module, type)):
        return None
    if cls is types.MethodType:
        return _receiver_problem(callee.__self__, callee.__func__.__name__, local)
    if cls is types.BuiltinFunctionType:
        receiver = callee.__self__
        if receiver is None or type(receiver) is types.ModuleType:
            impure = (
                callee.__module__ == "builtins" and callee.__name__ in _IMPURE_BUILTINS
            )
            return f"{callee.__name__} has an effect" if impure else None
        return _receiver_problem(receiver, callee.__name__, local)
    if cls is types.MethodWrapperType:
        return _receiver_problem(callee.__self__, callee.__name__, local)
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
    inst: dict[str, Any], env: dict[int, Any], local: dict[int, Any]
) -> str | None:
    """Why a region cannot hold this instruction, on the values it is about
    to run on, or None if it can."""
    op = inst["op"]
    if op == "call" or op == "call_ex":
        uses = inst["uses"]
        callee = env.get(uses[0])
        reason = _callee_problem(callee, local)
        if reason is not None:
            return reason
        if not _is_pure(callee) and not any(_involves_tensor(env.get(v)) for v in uses):
            return "no tensor or module involved"
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

# Bumped whenever a module gets or loses a parameter, buffer or submodule, or
# changes its training flag. What is known about a module holds for one epoch.
_epoch = 0
_hooked = False
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


def _install_module_hooks() -> None:
    """Makes ``nn.Module`` report changes of its structure, once per process."""
    global _hooked
    if _hooked:
        return
    _hooked = True
    module = nn.Module

    def reporting(name: str, structural: Callable[..., bool]) -> None:
        original = getattr(module, name)

        def method(self: nn.Module, *args: Any, **kwargs: Any) -> Any:
            global _epoch
            traced = _state.traced
            if traced is not None and id(self) in traced:
                raise Untraceable(f"{name} on a module the region was given")
            if structural(*args, **kwargs):
                _epoch += 1
            return original(self, *args, **kwargs)

        method.__name__ = original.__name__
        method.__qualname__ = original.__qualname__
        method.__doc__ = original.__doc__
        setattr(module, name, method)

    def setattr_is_structural(name: str, value: Any) -> bool:
        return (
            name == "training"
            or value is None
            or isinstance(value, (torch.Tensor, nn.Module))
        )

    reporting("__setattr__", setattr_is_structural)
    for name in (
        "__delattr__",
        "register_buffer",
        "register_parameter",
        "add_module",
        *_HOOK_REGISTRATIONS,
    ):
        reporting(name, lambda *args, **kwargs: True)
    module.register_module = module.add_module  # type: ignore[method-assign]


class _ModuleState:
    """The tensors of a module and of its submodules, for one epoch."""

    __slots__ = ("accessors", "epoch", "hooked", "ids", "module", "token")

    def __init__(self, module: nn.Module) -> None:
        self.module = weakref.ref(module)
        self.refresh(module)

    def refresh(self, module: nn.Module) -> None:
        self.epoch = _epoch
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
        self.ids = frozenset(ids)
        shape = (tuple(ids), tuple(training), tuple(n for _, n in self.accessors))
        # A small number that stands for the shape in the key of a variant.
        self.token = _tokens.setdefault(shape, len(_tokens))


_tokens: dict[Any, int] = {}
_module_states: dict[int, _ModuleState] = {}


def _forget_module(key: int) -> None:
    _module_states.pop(key, None)


def _module_state(module: nn.Module) -> _ModuleState:
    state = _module_states.get(id(module))
    if state is None or state.module() is not module:
        state = _module_states[id(module)] = _ModuleState(module)
        weakref.finalize(module, _forget_module, id(module))
    elif state.epoch != _epoch:
        state.refresh(module)
    return state


# ----------------------------------------------------------------------
# Inputs of a region
# ----------------------------------------------------------------------


def _describe(obj: Any, flat: list[torch.Tensor], seen: dict[int, int]) -> Any:
    """What a compiled region depends on in ``obj``, as something hashable.

    The tensors found on the way are appended to ``flat``, each object once.
    """
    cls = type(obj)
    if cls in _TENSORS:
        position = seen.get(id(obj))
        if position is None:
            position = seen[id(obj)] = len(flat)
            flat.append(obj)
        return position
    if cls in _VALUES:
        return (cls, obj)
    if cls in _PLAIN:
        return ("object", id(obj))
    if cls is tuple or cls is list:
        return (cls, tuple([_describe(item, flat, seen) for item in obj]))
    if cls is dict or cls is collections.OrderedDict:
        for key in obj:
            if type(key) not in _VALUES:
                raise _Opaque(0, "a dict with keys that are not plain values")
        values = tuple([_describe(item, flat, seen) for item in obj.values()])
        return (cls, tuple(obj), values)
    if isinstance(obj, nn.Module):
        state = _module_state(obj)
        if state.hooked or any(_GLOBAL_HOOK_DICTS):
            raise _Opaque(
                0, "a module with hooks, which run code a graph does not hold"
            )
        try:
            for store, name in state.accessors:
                tensor = store[name]
                if id(tensor) not in seen:
                    if type(tensor) not in _TENSORS:
                        raise _Opaque(0, "a module that holds a tensor subclass")
                    seen[id(tensor)] = len(flat)
                    flat.append(tensor)
        except LookupError:
            # Changed behind the back of the hooks.
            state.refresh(obj)
            return _describe(obj, flat, seen)
        return ("module", id(obj), state.token)
    if cls is types.MethodType:
        return ("method", id(obj.__func__), _describe(obj.__self__, flat, seen))
    if cls is types.BuiltinFunctionType:
        receiver = obj.__self__
        if receiver is None or type(receiver) is types.ModuleType:
            return ("object", id(obj))
        if type(receiver) in _CONTAINERS:
            raise _Opaque(
                0, f"a method of a {type(receiver).__name__} from before the region"
            )
        return ("method", obj.__name__, _describe(receiver, flat, seen))
    if isinstance(obj, tuple):
        # Named tuples and the like.
        return (cls, tuple([_describe(item, flat, seen) for item in obj]))
    if isinstance(
        obj, (type, enum.Enum, torch._ops.OpOverload, torch._ops.OpOverloadPacket)
    ):
        return ("object", id(obj))
    # Anything else may have state that a trace would change.
    raise _Opaque(0, f"a {cls.__name__} from before the region")


def _acceptable(obj: Any) -> str | None:
    """Why ``obj`` cannot be an input of a region, or None if it can."""
    try:
        _describe(obj, [], {})
    except _Opaque as e:
        return e.reason
    return None


def _rebuild(obj: Any, fakes: dict[int, torch.Tensor], copies: list[Any]) -> Any:
    """``obj`` with its tensors replaced by the ones the trace runs on.

    Containers are built anew, so that a trace that changes one changes a
    copy; ``copies`` collects them with what they started as.
    """
    cls = type(obj)
    if cls in _TENSORS:
        return fakes[id(obj)]
    if cls is tuple:
        return tuple([_rebuild(item, fakes, copies) for item in obj])
    if cls is list:
        copy = [_rebuild(item, fakes, copies) for item in obj]
        copies.append((copy, list(copy)))
        return copy
    if cls is dict or cls is collections.OrderedDict:
        copy = cls((k, _rebuild(v, fakes, copies)) for k, v in obj.items())
        copies.append((copy, dict(copy)))
        return copy
    if cls is types.MethodType:
        receiver = _rebuild(obj.__self__, fakes, copies)
        return (
            obj
            if receiver is obj.__self__
            else types.MethodType(obj.__func__, receiver)
        )
    if cls is types.BuiltinFunctionType:
        receiver = _rebuild(obj.__self__, fakes, copies)
        return obj if receiver is obj.__self__ else getattr(receiver, obj.__name__)
    if isinstance(obj, tuple) and not isinstance(obj, torch.Size):
        items = [_rebuild(item, fakes, copies) for item in obj]
        return cls(*items) if hasattr(cls, "_fields") else cls(items)
    return obj


def _unchanged(copy: Any, before: Any) -> bool:
    """Whether a copy a trace ran on still holds what it started with."""
    if isinstance(copy, dict):
        if list(copy) != list(before):
            return False
        copy, before = list(copy.values()), list(before.values())
    return len(copy) == len(before) and all(a is b for a, b in zip(copy, before))


def _identities(obj: Any, modules: dict[int, nn.Module], others: list[Any]) -> None:
    """The modules in an input of a region, and the other objects that
    ``_describe`` names by their id."""
    if isinstance(obj, nn.Module):
        modules[id(obj)] = obj
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            _identities(item, modules, others)
    elif isinstance(obj, dict):
        for item in obj.values():
            _identities(item, modules, others)
    elif type(obj) in (types.MethodType, types.BuiltinFunctionType):
        others.append(obj)
        _identities(obj.__self__, modules, others)
    elif not isinstance(obj, torch.Tensor) and type(obj) not in _VALUES:
        others.append(obj)


def _modes() -> tuple[Any, ...]:
    return (
        torch.is_grad_enabled(),
        torch.is_inference_mode_enabled(),
        torch.is_autocast_enabled("cuda"),
        torch.is_autocast_enabled("cpu"),
        torch.get_default_dtype(),
    )


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
    patches = [
        (torch.Tensor, "backward", _refuse("calls backward")),
        (torch.autograd, "backward", _refuse("calls torch.autograd.backward")),
        (torch.autograd, "grad", _refuse("calls torch.autograd.grad")),
        (torch.autograd.Variable, "_execution_engine", _NoEngine()),
        (torch._C, "_set_grad_enabled", _set_grad_enabled_checked),
    ]
    saved = [(owner, name, owner.__dict__.get(name)) for owner, name, _ in patches]
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
        # Shadows the attribute of the base class for the time of the trace.
        torch.Tensor.grad = property(read_grad, refuse_grad, refuse_grad)  # type: ignore[assignment,method-assign]
        _state.traced = reachable
        # Library code that asks torch.compiler.is_compiling() takes the path
        # it takes under torch.compile.
        with torch.compiler._compile_session_context():
            yield
    finally:
        _state.traced = None
        del torch.Tensor.grad
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


class _Recipes:
    """How to make the outputs of a trace again from the results of its graph.

    A recipe is a list ``[kind, dynamic, ...]``: ``result`` (a tensor the
    graph returns), ``same`` (the object itself, on every call), ``sequence``
    or ``dict``. ``dynamic`` says whether a result is somewhere inside.
    """

    def __init__(self, reachable: set[int], given: set[int]) -> None:
        self.reachable = reachable
        # Copies of the containers the region was given: not its to return.
        self.given = given
        self.results: list[torch.Tensor] = []
        self.memo: dict[int, list[Any]] = {}

    def of(self, obj: Any, depth: int = 0) -> list[Any]:
        known = self.memo.get(id(obj))
        if known is not None:
            return known
        if id(obj) in self.given:
            raise Untraceable("gives back a list or dict it was given")
        if depth > 40:
            raise Untraceable("gives an object that is nested too deeply")
        cls = type(obj)
        children: list[Any] | None = None
        recipe: list[Any]
        if is_fake_tensor(obj):
            recipe = ["result", True, len(self.results)]
            self.results.append(obj)
        elif _static_output(obj, self.reachable):
            recipe = ["same", False, obj]
        elif cls is tuple or cls is list:
            recipe = ["sequence", False, cls]
            children = list(obj)
        elif cls is dict or cls is collections.OrderedDict:
            recipe = ["dict", False, cls, list(obj)]
            children = list(obj.values())
        else:
            raise Untraceable(f"gives a {cls.__name__}, which a graph cannot return")
        self.memo[id(obj)] = recipe
        if children is not None:
            parts = [self.of(child, depth + 1) for child in children]
            recipe.append(parts)
            recipe[1] = any(part[1] for part in parts)
        return recipe


def _build(recipe: list[Any], results: Any, built: dict[int, Any]) -> Any:
    kind = recipe[0]
    if kind == "result":
        return results[recipe[2]]
    if kind == "same":
        return recipe[2]
    value = built.get(id(recipe))
    if value is not None:
        return value
    parts = [_build(part, results, built) for part in recipe[-1]]
    value = recipe[2](parts) if kind == "sequence" else recipe[2](zip(recipe[3], parts))
    built[id(recipe)] = value
    return value


class _Variant:
    """A region compiled for one combination of inputs."""

    __slots__ = ("compiled", "direct", "keep", "ops", "proven", "recipes")

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

    def run(self, flat: list[torch.Tensor]) -> tuple[Any, ...]:
        results = self.compiled(*flat)
        if self.direct:
            return tuple(results)
        built: dict[int, Any] = {}
        return tuple([_build(recipe, results, built) for recipe in self.recipes])


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
        # The time traces and the backend took.
        self.seconds = 0.0

    def __call__(self, *args: Any) -> tuple[Any, ...]:
        if not self.hot:
            return self.cold(args)
        flat: list[torch.Tensor] = []
        try:
            key = self.key(args, flat)
        except _Opaque:
            return self.replay(args)
        variant = self.variants.get(key, self)
        if variant is self:
            variant = self.specialize(key, args, flat)
        if variant is None:
            return self.replay(args)
        if variant.proven:  # type: ignore[union-attr]
            return variant.run(flat)  # type: ignore[union-attr]
        return self.first_run(key, variant, args, flat)  # type: ignore[arg-type]

    def first_run(
        self,
        key: Any,
        variant: "_Variant",
        args: tuple[Any, ...],
        flat: list[torch.Tensor],
    ) -> tuple[Any, ...]:
        """Runs a variant that has not run yet. What a backend made can fail
        where the instructions would not (or would with another error): then
        they run instead, this time and from now on."""
        generator = torch.get_rng_state()
        try:
            outputs = variant.run(flat)
        except Exception as e:
            torch.set_rng_state(generator)
            self.variants[key] = None
            if not any(self.variants.values()):
                self.hot = False
                reason = f"{type(e).__name__}: {e}".split("\n")[0][:200]
                self.plan.split(self, None, f"the compiled graph failed: {reason}")
            return self.replay(args)
        variant.proven = True
        return outputs

    def key(self, args: tuple[Any, ...], flat: list[torch.Tensor]) -> Any:
        seen: dict[int, int] = {}
        parts = []
        for position, arg in enumerate(args):
            try:
                parts.append(_describe(arg, flat, seen))
            except _Opaque as e:
                raise _Opaque(position, e.reason) from None
        tensors = [(t.shape, t.dtype, t.device, t.requires_grad) for t in flat]
        return (tuple(parts), tuple(tensors), _modes())

    def replay(self, args: tuple[Any, ...]) -> tuple[Any, ...]:
        env = dict(zip(self.inputs, args))
        self.plan.replay.run(self.lo, self.hi, env)
        return tuple([env[value] for value in self.outputs])

    def cold(self, args: tuple[Any, ...]) -> tuple[Any, ...]:
        # An enclosing region or block that has not made up its mind may
        # still take all of this one in.
        ready = (
            self.calls >= self.plan.options.warmup
            and _state.pending == 0
            and _state.traced is None
            and _runtime.profiling_depth() == 0
        )
        if ready and self.surveyed:
            self.decide(args)
            if self.hot:
                return self(*args)
        self.calls += 1
        _state.pending += 1
        try:
            if not ready or self.surveyed:
                return self.replay(args)
            outputs = self.survey(args)
            if self.surveyed and self in self.plan.regions:
                # Nothing to take out: compiled now, for the next call.
                self.decide(args)
            return outputs
        finally:
            _state.pending -= 1

    def survey(self, args: tuple[Any, ...]) -> tuple[Any, ...]:
        """Runs the instructions one by one and takes those out of the plan
        that a region cannot hold, on the values of this call."""
        plan = self.plan
        env = dict(zip(self.inputs, args))
        local: dict[int, Any] = {}
        bad: dict[int, str] = {}
        tensors = 0
        refused = {v: _acceptable(arg) for v, arg in zip(self.inputs, args)}
        for i in range(self.lo, self.hi):
            inst = plan.insts[i]
            reason = _problem(inst, env, local)
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
        outputs = tuple([env[value] for value in self.outputs])
        if bad:
            plan.split(self, bad)
        elif tensors == 0:
            plan.split(self, None, "no tensor or module involved")
        else:
            self.surveyed = True
        return outputs

    def decide(self, args: tuple[Any, ...]) -> None:
        """Compiles the region for the first time, or takes it out of the plan."""
        flat: list[torch.Tensor] = []
        try:
            key = self.key(args, flat)
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
            self.variants[key] = self.compile(key, args, flat)
            self.hot = True
        except _Failed as e:
            self.plan.split(
                self, None if e.index is None else {e.index: e.reason}, e.reason
            )

    def specialize(
        self, key: Any, args: tuple[Any, ...], flat: list[torch.Tensor]
    ) -> _Variant | None:
        """Another variant of a region that has one, for inputs not seen yet."""
        if len(self.variants) >= self.plan.options.max_variants:
            return None
        try:
            variant: _Variant | None = self.compile(key, args, flat)
        except _Failed:
            # These inputs do what the first ones did not: they run the
            # instructions as they are.
            variant = None
        self.variants[key] = variant
        return variant

    def compile(
        self, key: Any, args: tuple[Any, ...], flat: list[torch.Tensor]
    ) -> _Variant:
        from torch.fx.experimental.proxy_tensor import make_fx

        options = self.plan.options
        start = time.perf_counter()
        modules: dict[int, nn.Module] = {}
        # What the key names by id is kept alive, so the ids stay theirs. Not
        # the tensors: a variant must not hold on to the batch it was made on.
        keep: list[Any] = []
        for arg in args:
            _identities(arg, modules, keep)
        keep.extend(modules.values())
        reachable: set[int] = set()
        for module in modules.values():
            reachable |= _module_state(module).ids
        found: dict[str, Any] = {}

        def region(*fake: torch.Tensor) -> tuple[torch.Tensor, ...]:
            fakes = {id(real): f for real, f in zip(flat, fake)}
            copies: list[Any] = []
            env = dict(zip(self.inputs, [_rebuild(arg, fakes, copies) for arg in args]))
            grads = [f.requires_grad for f in fake]
            local: dict[int, Any] = {}
            insts, replay = self.plan.insts, self.plan.replay
            with _swapped(modules, fakes), _traps(modules):
                for i in range(self.lo, self.hi):
                    reason = _problem(insts[i], env, local)
                    if reason is not None:
                        raise _Failed(i, reason)
                    replay.run(i, i + 1, env, True)
                    _note_local(insts[i], env, local)
            if not all(_unchanged(copy, before) for copy, before in copies):
                raise _Failed(None, "changes a list or dict it was given")
            if grads != [f.requires_grad for f in fake]:
                raise _Failed(None, "changes requires_grad of a tensor it was given")
            recipes = _Recipes(reachable, {id(copy) for copy, _ in copies})
            found["recipes"] = []
            for value in self.outputs:
                try:
                    found["recipes"].append(recipes.of(env[value]))
                except Untraceable as e:
                    raise _Failed(self.definition(value), str(e)) from None
            return tuple(recipes.results)

        try:
            graph = make_fx(region, tracing_mode="fake")(*flat)
        except _Failed:
            raise
        except Exception as e:
            reason = f"{type(e).__name__}: {e}".split("\n")[0][:200]
            raise _Failed(getattr(e, "cfg_index", None), reason) from None
        ops = sum(1 for node in graph.graph.nodes if node.op == "call_function")
        if ops < options.min_ops:
            raise _Failed(None, f"{ops} tensor operations, not worth a graph")
        try:
            compiled = options.backend(graph, list(flat))
        except Exception as e:
            reason = f"{type(e).__name__}: {e}".split("\n")[0][:200]
            raise _Failed(None, f"the backend failed: {reason}") from None
        self.seconds += time.perf_counter() - start
        _runtime.plan_changed()
        return _Variant(compiled, found["recipes"], ops, keep)

    def definition(self, value: int) -> int | None:
        """The instruction of the region that defines a value."""
        for i in range(self.lo, self.hi):
            if self.plan.insts[i]["dst"] == value:
                return i
        return None

    def __repr__(self) -> str:
        return f"{self.plan.name}[{self.lo}:{self.hi}]"


class _Plan:
    """The regions of one basic block. Kept on the block as its note."""

    def __init__(
        self, options: "Regions", replay: Any, insts: list[dict[str, Any]]
    ) -> None:
        self.options = options
        self.replay = replay
        self.name = replay.name
        self.insts = insts
        # Instructions that stay with the VM, and why.
        self.bad: dict[int, str] = {}
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

    def __call__(self, region: Any) -> bool:
        insts = region.instructions()
        if any(inst["op"] == "call_object" for inst in insts):
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
            changed |= region.fuse(part.lo, part.hi, part)
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
    _install_module_hooks()
    return Regions(backend, min_ops, max_variants, warmup)
