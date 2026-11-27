# mypy: allow-untyped-defs
"""The lowering of a call list: the aten-level FX graph a block pass is given.

A call list (torch/_rust/src/exec/calls) is the body's own C calls in program
order. ``lower`` makes its graph without the program: the same Rust
interpreter runs the same entries once more, under ``make_fx`` with an empty
decomposition table, on fake tensors made from the keys of the inputs and of
the pins. The pins are the trailing placeholders (``const_0``, ...), as a
region's constants are today.

Given: the list, the spec it was made from, the keys of its inputs (a float
input: its traced value) and pins, and what the traced run left (``Traced``).
Called with no mode on, in the grad mode and autocast state the list runs in.

Guarantees: a ``Lowered`` only if the run reached the end, every entry is of
a kind and under flags this module knows, and the fake results have the
recorded keys (the outputs, every input and pin after the run, what the
entries' asserts say, the class every tensor operand is pinned to). Every
other list is ``NotLowerable`` with the reason and the entry. No tensor of
the program is an operand of the run, and no mode is left on a stack.

Releases. A list states where each result is released (a moved operand, a
drop entry); a graph states none, so the graph of a list is the same graph
wherever the list's releases are, and a pass may keep, fuse or free what is
between the inputs and the outputs as it likes. That is sound only where
nothing can see a release: a list lowers only if everything it releases is,
in the data-free run, a tensor of that run, an exact int, float, bool,
complex, str, bytes or None, or a tuple, list or struct sequence of those. For
any other object neither its class's deallocator nor a weak reference to
it can be known here, so there is no list fact for it: the list has no
graph. When the memory of a tensor goes back is not kept by a graph.

Values read out of tensors (lower_scalars.py). A number an entry reads out
of a tensor (``Tensor.item``), and what entries make of it, is a symbolic
scalar of the data-free run: the graph has the read and each step of the
arithmetic as nodes and computes the number of every call, the list has the
fact ``reads_value``, and what was recorded of the number is its class, not
its value. A list that needs the value (a size, an index, a branch inside a
callee, the number as an output) is ``NotLowerable`` at that entry.

May not assume: values (the graph of a callee whose kernel asks whether a
mode is on may differ from the callee in the last bits). A float input is a
constant of the graph (``Lowered.specialized``: for that value only). A tensor
an entry makes of constants alone (``torch.tensor(1.0)``) is one more trailing
placeholder (``Lowered.constants``). How the tensors of a later call share
memory, and what a pass may do with ``list_facts``, are the caller's.

Imports lower_scalars of this package and nothing else of it. The form of a
spec is known to ``_Read`` (and, for the entries that are scalars, to
lower_scalars).
"""

import dataclasses
from typing import Any, NamedTuple

import torch
from torch._subclasses import fake_tensor as _fake
from torch.fx.experimental.proxy_tensor import get_proxy_mode, make_fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from .lower_scalars import in_run, NEEDS_VALUE, Reads, settled, SYMBOL, symbols


class Key(NamedTuple):
    """What a region compares of a tensor (exec/tensor_key.rs), by field."""

    parameter: bool
    dtype: torch.dtype
    device: torch.device
    sizes: tuple[int, ...]
    strides: tuple[int, ...]
    requires_grad: bool
    conj: bool = False
    neg: bool = False
    inference: bool = False


def key_of(t: Any, parameter: bool | None = None) -> Key | None:
    """The key of a dense, strided tensor: of an object whose class is exactly
    Tensor or Parameter, or of any tensor when the class is given."""
    if parameter is None:
        if type(t) is not torch.Tensor and type(t) is not torch.nn.Parameter:
            return None
        parameter = type(t) is torch.nn.Parameter
    if not isinstance(t, torch.Tensor) or t.layout is not torch.strided:
        return None
    shape, grad = (tuple(t.shape), tuple(t.stride())), t.requires_grad
    bits = (t.is_conj(), t.is_neg(), t.is_inference())
    return Key(parameter, t.dtype, t.device, *shape, grad, *bits)


class Traced(NamedTuple):
    """What the traced run left. ``outputs``: per output of the list a Key, a
    list or tuple of those, or None. ``inputs`` and ``pins``: the key of each
    after the run (a float input: anything)."""

    outputs: tuple[Any, ...]
    inputs: tuple[Any, ...]
    pins: tuple[Any, ...]


# What a pass must declare to be given the graph of a list that has it.
FACTS = ("writes_input", "draws", "sets_state", "effects", "reads_value", "data_attr")


@dataclasses.dataclass(frozen=True)
class Lowered:
    # Called as gm(*inputs, *pins, *constants): the tuple of the list's outputs.
    gm: torch.fx.GraphModule
    # The float inputs the graph holds as constants.
    specialized: tuple[int, ...]
    list_facts: frozenset[str]
    constants: tuple[torch.Tensor, ...] = ()


@dataclasses.dataclass(frozen=True)
class NotLowerable:
    reason: str
    entry: int | None = None
    list_facts: frozenset[str] = frozenset()


class _Differs(Exception):
    def __init__(self, why, entry=None):
        super().__init__(why)
        self.why, self.entry = why, entry


# The callees lowered: no state setter, store to an attribute or deleted item.
_CALLEES = ("vector", "binary", "compare", "unary", "getitem", "setitem", "getattr")
# The flags an entry may have; one of the first three says what it does.
_DOES = ("pure", "writes", "draws")
_BESIDE = ("fresh", "covered")
_REFUSED = {
    "effect": ("effects", "an effect is not lowered"),
    "state": ("sets_state", "a state setter is not lowered"),
    "opaque": (None, "an opaque call has no graph"),
    "rekeys": (None, "an entry that changes a tensor's key in place is not lowered"),
}
# What a tensor operand or result is in the data-free run, whatever its class
# at the trace; the classes a key can name; the callees with no result.
_FAKE = _fake.FakeTensor
_TENSORS = (torch.Tensor, torch.nn.Parameter)
_NO_RESULT = ("setitem", "delitem", "setattr")
# The classes of which nothing can see an instance released.
_QUIET = (int, float, bool, complex, str, bytes, type(None))
_HEAP = 1 << 9
# How the data-free run stopped, where no entry raised.
_STOPPED = {
    "assert": "what the entry asserts of its result does not hold of the data-free run",
    "part": "a part of a result is not there",
    "class": "an operand of the data-free run is not of the class it is pinned to",
    "refused": "the list of the data-free run was not entered",
}
_DATA = (_fake.DataDependentOutputException, _fake.DynamicOutputShapeException)
_NO_KERNEL = (_fake.UnsupportedOperatorException, _fake.UnsupportedFakeTensorException)
_RUNS = "the data-free run and the traced run"
_STRICT = {"allow_fallback_kernels": False, "allow_non_fake_inputs": False}


def _held(t):
    """The template ``t`` with every move a use: the data-free run releases
    nothing."""
    kind = t[0]
    if kind == "move":
        return ("res", t[1])
    if kind in ("seq", "map"):
        return (kind, t[1], [_held(part) for part in t[2]])
    return (kind, *map(_held, t[1:])) if kind == "slice" else t


class _Read:
    """A spec of ``_calls_make``, read. Everything this module knows of the
    form of a spec is in this class: what the list uses and has, the first
    thing in it that is not lowered (``refused``), the spec of the data-free
    run (``fake_spec``, the one place a spec is built), what the asserts say
    of a tensor (``checks``) and which operands are pinned to a tensor class
    (``tensor_pins``)."""

    def __init__(self, spec, input_keys, pins):
        if len(spec) != 5 or any(
            len(e) != (3 if e[0] == "drop" else 8) for e in spec[3]
        ):
            raise ValueError(
                "a spec of 5 fields with entries of 8 and drops of 3 is read"
            )
        self.ninputs, self.npins, self.consts, self.entries, self.outputs = spec
        if (self.ninputs, self.npins) != (len(input_keys), len(pins)):
            raise _Differs("the keys are not as many as the list's inputs and pins")
        self.input_keys = input_keys
        self.facts: set[str] = set()
        self.floats: set[int] = set()
        self.refused: tuple[str, int | None] | None = None
        # By entry: whether its callee has a result (a drop has none).
        self.results = [
            e[0] != "drop" and e[0][0] not in _NO_RESULT for e in self.entries
        ]
        # (entry, operand template, class) of every operand, but a constant,
        # that is pinned to a class of tensors.
        self.tensor_pins: list[tuple[int, Any, type]] = []
        # The pins of the data-free run, by entry.
        self.fake_pins: dict[int, tuple[Any, ...]] = {}
        # By entry whose result is a scalar read from a tensor: its class.
        self.symbols, no_scalar = symbols(self.entries)
        if self.symbols or no_scalar is not None:
            self.facts.add("reads_value")
        if no_scalar is not None:
            self.refuse(*no_scalar)
        # The slots an output names whole are the caller's: not released.
        self.handed: set[int] = set()
        fresh: list[bool] = []
        for k, e in enumerate(self.entries):
            if e[0] == "drop":
                # Nothing in a graph. Its slots are read by the interpreter.
                fresh.append(False)
                continue
            callee, args, flags = e[0], e[1], e[5]
            if callee[0] not in _CALLEES:
                self.refuse(f"a {callee[0]!r} entry is not lowered", k)
            if tuple(callee) == ("getattr", "data"):
                # A graph has aten.detach for it, which counts the versions
                # of the tensor it is read of; what .data gives does not.
                self.facts.add("data_attr")
                self.refuse("a read of .data is not lowered", k)
            names = {f if type(f) is str else f[0] for f in flags}
            writes = [f[1] for f in flags if type(f) is not str and f[0] == "writes"]
            written = set().union(*writes)
            for name in sorted(names):
                if name in _REFUSED:
                    fact, why = _REFUSED[name]
                    self.facts.update([fact] if fact else [])
                    self.refuse(why, k)
                elif name not in _DOES and name not in _BESIDE:
                    self.refuse(f"the flag {name!r} is not understood", k)
            if not names.intersection(_DOES):
                self.refuse("no flag says what the entry does", k)
            if "draws" in names:
                self.facts.add("draws")
            for i, t in enumerate(args):
                self.operand(t, k, i in written, fresh)
            fresh.append("fresh" in names)
            pinned = zip(args, e[7], strict=True)
            self.fake_pins[k] = tuple(self.pin(t, c, k) for t, c in pinned)
        for t in self.outputs:
            self.operand(t, None, False, fresh)

    def refuse(self, why, k):
        self.refused = self.refused or (why, k)

    def operand(self, t, k, written, fresh):
        kind = t[0]
        if kind == "in":
            if type(self.input_keys[t[1]]) is float:
                self.floats.add(t[1])
        elif kind in ("seq", "map"):
            for part in t[2]:
                self.operand(part, k, written, fresh)
        elif kind == "slice":
            for part in t[1:]:
                self.operand(part, k, False, fresh)
        elif kind not in ("pin", "res", "move", "part", "const", "null"):
            self.refuse(f"the operand template {kind!r} is not understood", k)
        if kind == "res" and k is None:
            self.handed.add(t[1])
        # Written: an input, a pin, or a result that may be over the memory
        # of one (a part of a result is not known to be fresh).
        own = kind in ("res", "move") and fresh[t[1]]
        if written and kind in ("in", "pin", "res", "move", "part") and not own:
            self.facts.add("writes_input")

    def pin(self, t, c, k):
        """The class the operand ``t`` of entry ``k``, pinned to ``c``, has in
        the data-free run: a tensor is a fake tensor there, whatever the
        pin; a constant is the program's own object."""
        kind = t[0]
        if kind in ("seq", "map", "slice"):
            parts = t[2] if kind != "slice" else t[1:]
            return tuple(self.pin(p, pc, k) for p, pc in zip(parts, c, strict=True))
        if kind in ("res", "move") and t[1] in self.symbols and c in SYMBOL:
            return SYMBOL[c]
        if kind == "const" or c not in _TENSORS:
            return c
        self.tensor_pins.append((k, t, c))
        return _FAKE

    def fake_spec(self):
        """The spec of the same entries for the data-free run. Nothing is
        released (a move is a use, a drop drops nothing) and every result is
        handed out behind the list's outputs. A tensor is pinned and asserted
        to be a fake tensor: a fake tensor has no key for the interpreter to
        read, so what the list says of tensors is compared by ``lower``
        (``checks``, ``tensor_pins``). The other asserts and pins (a count,
        "the result is operand i", the class of what is no tensor) are the
        list's own, held by the interpreter as in any run; a scalar read
        from a tensor is of its symbolic class there (lower_scalars)."""
        entries = []
        for k, e in enumerate(self.entries):
            if e[0] == "drop":
                entries.append(("drop", (), e[2]))
                continue
            callee, said = in_run(e, k, self.symbols)
            tensor = [
                a[0] == "key" or (a[0] == "class" and a[1] in _TENSORS) for a in said
            ]
            asserts = tuple(("class", _FAKE) if t else a for t, a in zip(tensor, said))
            out = k if self.results[k] else None
            args = [_held(t) for t in e[1]]
            entries.append(
                (callee, args, e[2], out, e[4], e[5], asserts, self.fake_pins[k])
            )
        every = [("res", k) for k, has in enumerate(self.results) if has]
        return (self.ninputs, self.npins, self.consts, entries, [*self.outputs, *every])

    def checks(self):
        """What the asserts say of a tensor, as (entry, what): the Key of the
        asserted tensor, or the class (Tensor or Parameter)."""
        out = []
        for k, e in enumerate(self.entries):
            for a in () if e[0] == "drop" else e[6]:
                if a[0] == "key":
                    out.append((k, key_of(a[1])))
                elif a[0] == "class" and a[1] in _TENSORS:
                    out.append((k, a[1]))
                elif a[0] not in ("class", "count", "is"):
                    raise _Differs(f"the assert {a[0]!r} is not understood", k)
        return out

    def value(self, t, fakes, results):
        """What a template of one input, pin, result or part names in the
        data-free run."""
        kind = t[0]
        if kind in ("in", "pin"):
            return fakes[t[1] + (self.ninputs if kind == "pin" else 0)]
        x = results.get(t[1]) if kind in ("res", "move", "part") else None
        for i in t[2] if kind == "part" else ():
            x = tuple.__getitem__(x, i)
        return x


def _fake_of(key, what):
    """A fake tensor of ``key``, in the fake mode that is on."""
    if type(key) is not Key:
        raise _Differs(f"{what} has no key (no plain, dense, strided tensor)")
    if key.conj or key.neg or key.inference:
        why = "the conjugate or the negative bit, or is an inference tensor"
        raise _Differs(f"{what} has {why}: not lowered")
    sizes, strides = key.sizes, key.strides
    t = torch.empty_strided(sizes, strides, dtype=key.dtype, device=key.device)
    return t.requires_grad_(True) if key.requires_grad else t


def _same(got, want, where, classes, mode):
    """L2: ``got``, of the data-free run, has what the traced run recorded."""
    why = None
    if type(want) is Key:
        fake = type(got) is _fake.FakeTensor and got.fake_mode is mode
        have = key_of(got, classes.get(id(got), False)) if fake else None
        fields = [f for f, a, b in zip(Key._fields, have or (), want) if a != b]
        if have != want:
            why = f"differ in {', '.join(fields) or 'what the tensor is'}"
    elif type(want) in (tuple, list):
        if type(got) is not type(want) or len(got) != len(want):
            why = "differ in what holds the tensors"
        else:
            for i, (g, w) in enumerate(zip(got, want)):
                _same(g, w, f"{where}[{i}]", classes, mode)
    elif want is not None or got is not None:
        why = f"are not compared ({type(got).__name__}, {type(want).__name__})"
    if why:
        raise _Differs(f"{where}: {_RUNS} {why}")


def _is(x, parameter, classes, mode):
    """Whether ``x`` is a tensor of the data-free run, of the class a key
    with ``parameter`` names: its input's or pin's if it is that object."""
    fake = type(x) is _FAKE and x.fake_mode is mode
    return fake and classes.get(id(x), False) == parameter


def _asserted(read, checks, fakes, results, classes, mode):
    """What the list says of tensors holds of the data-free run, entry by
    entry: the class every operand is pinned to and the key or class the
    entry asserts of its result, against the classes the recorded keys give."""
    said = [(k, 0, t, pinned) for k, t, pinned in read.tensor_pins]
    said += [(k, 1, None, want) for k, want in checks]
    for k, _, t, want in sorted(said, key=lambda one: one[:2]):
        x = results.get(k) if t is None else read.value(t, fakes, results)
        if type(want) is Key:
            try:
                _same(x, want, "the result the entry asserts a key of", classes, mode)
            except _Differs as e:
                raise _Differs(e.why, k) from None
        elif not _is(x, want is torch.nn.Parameter, classes, mode):
            what = "the result is asserted" if t is None else "an operand is pinned"
            why = "and the keys of the trace give it another class"
            raise _Differs(f"{what} to be a {want.__name__}, {why}", k)


def _unseen(x, mode):
    """Whether nothing of a program can see ``x`` released: a tensor of the
    data-free run, a plain value (or the symbolic scalar that stands for
    one), or an exact tuple or list or a struct sequence (a subclass of
    tuple that C code defines) of those. None of these classes has a
    finalizer or takes a weak reference."""
    cls = type(x)
    if cls is _FAKE:
        return x.fake_mode is mode
    c_tuple = type(cls) is type and cls.__base__ is tuple and not cls.__flags__ & _HEAP
    if cls is tuple or cls is list or c_tuple:
        return all(_unseen(i, mode) for i in x)
    return cls in _QUIET or cls in SYMBOL.values()


def _released(read, results, mode):
    """Everything the list releases is of a class whose release is unseen."""
    for k, x in results.items():
        if k not in read.handed and not _unseen(x, mode):
            why = f"the list releases its result, a {type(x).__name__}"
            raise _Differs(f"{why}: a graph cannot keep to when, and it may be seen", k)


def _as_today(gm, n):
    """Makes ``gm`` what a region's graph is today: dead code gone, the pins
    and then the tensors the run made (returned) its last placeholders."""
    gm.graph.eliminate_dead_code()
    given, made = list(gm.graph.find_nodes(op="placeholder")), {}
    for node in list(gm.graph.find_nodes(op="get_attr")):
        value = getattr(gm, node.target, None)
        if not isinstance(value, torch.Tensor) or value.grad_fn is not None:
            raise _Differs(f"the graph names {node.target}, no tensor to hand it")
        if node.target not in made:
            # The graph has a node (this one) behind every placeholder.
            behind = (given[-1] if given else gm.graph._root).next
            with gm.graph.inserting_before(behind):
                given.append(gm.graph.placeholder(node.target))
            given[-1].meta = dict(node.meta)
            made[node.target] = (given[-1], value)
        node.replace_all_uses_with(made[node.target][0])
        gm.graph.erase_node(node)
    for target in made:
        delattr(gm, target)
    for i, node in enumerate(given[n:]):
        node._rename(f"const_{i}")
        node.target = node.name
    gm.recompile()
    return tuple(value for _, value in made.values())


def _modes():
    return torch._C._len_torch_dispatch_stack(), torch._C._len_torch_function_stack()


def lower(call_list, spec, input_keys, pins, facts):
    """The graph of ``call_list`` (made from ``spec``), or why it has none.
    ``input_keys``: a Key or the traced float; ``pins``: Keys; ``facts``: Traced."""
    from torch._rust import _calls_make, _calls_run, _calls_text

    def no(why, entry=None, have=()):
        if entry is not None:
            lines = _calls_text(call_list).split("\n")
            line = lines[entry].strip() if entry < len(lines) else "?"
            why = f"entry {entry}: {why}: {line}"
        return NotLowerable(why, entry, frozenset(have))

    try:
        read = _Read(spec, input_keys, pins)
        if tuple(map(len, facts)) != (len(read.outputs), read.ninputs, read.npins):
            raise _Differs("the facts are not as many as the outputs, inputs and pins")
        if read.refused is not None:
            return no(*read.refused, have=read.facts)
        checks = read.checks()
        fake_list = _calls_make(read.fake_spec())
    except _Differs as e:
        return no(e.why, e.entry)
    except (TypeError, ValueError, IndexError, KeyError, AttributeError) as e:
        return no(f"a spec that is not understood: {type(e).__name__}: {e}")
    before = _modes()
    if before != (0, 0):
        return no("a dispatch mode or a function mode is on", have=read.facts)
    n, nout, keys = read.ninputs, len(read.outputs), [*input_keys, *pins]
    names = [f"input {i}" if i < n else f"pin {i - n}" for i in range(len(keys))]
    floats = {i for i, key in enumerate(input_keys) if type(key) is float}
    # With a value read from a tensor the run is symbolic in its scalars.
    env = ShapeEnv() if read.symbols else None
    fake_mode = _fake.FakeTensorMode if env is None else Reads
    mode = fake_mode(**_STRICT, shape_env=env)
    tracers, results, outputs, fakes, classes, left = [], {}, [], [], {}, []

    def stopped_at():
        at = fake_list.stopped
        return at[0] if at is not None and at[0] < len(read.entries) else None

    def replay(*given):
        tracers.append(get_proxy_mode().tracer)
        if any(a is not b for a, b in zip(given, fakes, strict=True)):
            raise _Differs("the data-free run was not handed the tensors of the keys")
        try:
            got = _calls_run(fake_list, given[:n], given[n:])
        finally:
            # A stop where no entry raised: a miss, or the interpreter's error.
            at = fake_list.stopped
            if at is not None and at[1] != "raised":
                raise _Differs(_STOPPED[at[1]], stopped_at())
        results.update(
            zip([k for k, has in enumerate(read.results) if has], got[nout:])
        )
        outputs.extend(got[:nout])
        return got[:nout]

    try:
        with mode:
            for i, key in enumerate(keys):
                fakes.append(key if i in floats else _fake_of(key, names[i]))
                classes[id(fakes[i])] = i not in floats and key.parameter
            # "real": make_fx takes the fakes as they are (its symbolic mode
            # would make new ones, of symbolic sizes).
            gm = make_fx(replay, decomposition_table={}, tracing_mode="real")(*fakes)
            after = [*facts.inputs, *facts.pins]
            for i, got in enumerate(outputs):
                _same(got, facts.outputs[i], f"output {i}", classes, mode)
            for i, got in enumerate(fakes):
                if i not in floats:
                    _same(got, after[i], f"{names[i]} after the run", classes, mode)
            _asserted(read, checks, fakes, results, classes, mode)
            _released(read, results, mode)
            needs = None if env is None else settled(gm, results, env)
            if needs is not None:
                raise _Differs(*needs)
            constants = _as_today(gm, n)
    except _Differs as e:
        return no(e.why, e.entry, read.facts)
    except Exception as e:
        text = f"{type(e).__name__}: {e}"
        if isinstance(e, NEEDS_VALUE):
            first = str(e).split("\n")[0]
            text = f"it needs the value of a number read from a tensor ({first})"
        elif isinstance(e, _DATA):
            read.facts.add("reads_value")
            text = f"its result depends on a tensor's data ({text})"
        elif isinstance(e, _NO_KERNEL):
            text = f"it has no fake kernel ({text})"
        else:
            text = f"the data-free run raised {text}"
        return no(text, stopped_at(), read.facts)
    finally:
        # A stopped run keeps its slots. fx tracks each tensor weakly with a
        # callback, which would run under whichever instruction of the
        # program the collector frees it at.
        fake_list.release()
        for tracer in tracers:
            tracer.tensor_tracker.clear()
            tracer.script_object_tracker.clear()
        left.append(_modes() != before)
        while torch._C._len_torch_dispatch_stack() > before[0]:
            torch._C._pop_torch_dispatch_stack(None)
        while torch._C._len_torch_function_stack() > before[1]:
            torch._C._pop_torch_function_stack()
    if left[0]:
        return no("the data-free run left a mode on a stack", have=read.facts)
    return Lowered(gm, tuple(sorted(read.floats)), frozenset(read.facts), constants)
