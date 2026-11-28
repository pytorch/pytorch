# mypy: allow-untyped-defs
"""The writes of a region's traced run, on the tracer's side.

While a region's body is traced, the executor records each write the run makes
to an object that outlives it (torch/_rust/src/exec/writes.rs): ``how(target,
*values)``, C code such as ``object.__setattr__`` or ``list.append``. This
module turns that record into what the region does again on every run: one
call per write, made in the run's order after the region's graph ran.

Given (by ``torch._rust_inductor.trace_region``)
  - the record of the recorder (``torch._rust._region_writes()``), and the
    make_fx tracer of the trace (``own``);
  - ``plain(value)``: ``value`` with the tracer's stand-ins for float
    arguments replaced by the floats they stand for;
  - ``describe(value)``: the template by which a region makes ``value`` on a
    later run (a graph output, an argument, a constant, a container of
    those); it raises if there is none;
  - ``shared(values, old)``: whether a container the run made can be reached
    twice from ``values``;
  - ``args`` of the traced call, and ``old``: the containers that were there
    before the run, by id (those inside an argument with their path).

Guarantees
  - ``take`` leaves the program's state as eager leaves it after the traced
    call: no stand-in of the tracer's stays where the run wrote one. Every
    write is tried whatever became of one before it, and ``finish`` goes on
    where ``take`` was interrupted: the tracer calls one of them on every
    way out of a body whose run stands.
  - ``plan`` gives one call template per write that outlives the run, in the
    run's order, or raises ``Unrepeatable``. The target is an argument, a
    container inside one (found by its path on every run), or the object
    itself, which the region then holds (``pinned``). A write into a
    container or an object the region builds anew, for an output or for a
    written value, is none: it is built as the run left it.
  - ``unreachable`` names a held target that nothing the run read leads to:
    the checks of such a region would not notice that the program uses
    another object in its place.

May not assume
  - that a target is one of the region's arguments;
  - that a written value is a tensor, or that it can be made again;
  - that only the body writes while it is traced: make_fx's own records
    change too, which is why they are named to the recorder (``own``).
"""

from typing import Any

from torch.utils.weak import WeakIdKeyDictionary


class Unrepeatable(Exception):
    """A write of the traced run that a region cannot make again."""


_NOTHING = object()


def own(tracer) -> list[Any]:
    """The containers the make_fx trace keeps its own records in (which
    tensor is which proxy): what is read from one of them or written to it
    while the body runs (the entry of a tensor that died is taken out) is
    the trace's, not the body's (``torch._rust._region_own``)."""
    out: list[Any] = []
    for v in vars(tracer).values():
        if isinstance(v, (dict, list, set)):
            out.append(v)
        elif isinstance(v, WeakIdKeyDictionary):
            out.extend(x for x in vars(v).values() if isinstance(x, (dict, list)))
    return out


def callbacks(tracer) -> list[Any]:
    """The function objects the make_fx trace's own tables run when a tensor
    they hold weakly dies (the entry is taken out): a frame of one of them
    under the traced body is the trace's, not code of the program's that no
    instruction called (``torch._rust._region_callbacks``)."""
    return [
        v._remove for v in vars(tracer).values() if isinstance(v, WeakIdKeyDictionary)
    ]


def _put_plain(how, target, values, real) -> None:
    """Gives every place of ``target`` that holds what the write ``how``
    put there (``values``, with a stand-in of the tracer's in them) the
    plain value (``real``). The place is found by the object it holds,
    in a list never by the index of the write: a later write of the run
    may have moved it (a deletion in front of it) or replaced it, and a
    place that holds something else is left. Runs no code of the target's
    class. A way of writing this does not know is an error: it
    would leave a stand-in in the program's state."""
    name = how.__name__
    if name in ("__delattr__", "__delitem__", "discard"):
        # Nothing was put anywhere.
        return
    if name == "add":
        # The member itself: one that a later write took out is not there.
        if set.__contains__(target, values[0]):
            set.discard(target, values[0])
            set.add(target, real[0])
    elif name in ("append", "__setitem__") and isinstance(target, list):
        value, plain = (
            (values[0], real[0]) if name == "append" else (values[1], real[1])
        )
        for i in range(list.__len__(target)):
            if list.__getitem__(target, i) is value:
                list.__setitem__(target, i, plain)
    elif name == "__setitem__" and isinstance(target, dict):
        # (The recorder takes a write under a key whose lookup is C code
        # alone: exec/keys.rs.)
        if dict.get(target, values[0], _NOTHING) is values[1]:
            how(target, *real)
    elif name == "__setattr__":
        # An instance's dict, or a slot (which reads as an AttributeError
        # while it holds nothing).
        try:
            there = object.__getattribute__(target, values[0])
        except AttributeError:
            there = _NOTHING
        if there is values[1]:
            how(target, *real)
    else:
        raise AssertionError(f"a write by {name} of a {type(target).__name__}")


def _built(values, old) -> set[int]:
    """The ids of the lists, dicts and sets a region builds anew for
    ``values`` (those not of ``old``, which were there before the run)."""
    seen: set[int] = set()
    todo = list(values)
    while todo:
        x = todo.pop()
        if isinstance(x, (list, dict, set)):
            if id(x) in seen or id(x) in old:
                continue
            seen.add(id(x))
        if isinstance(x, (list, tuple, set)):
            todo.extend(x)
        elif isinstance(x, dict):
            todo.extend(x.values())
    return seen


class Written:
    """The writes of one traced run, in the order it made them."""

    def __init__(self) -> None:
        # (how, target, values) as the recorder gave them.
        self.recorded: list[tuple[Any, Any, tuple[Any, ...]]] = []
        # What the region does again: a call template per write.
        self.templates: list[Any] = []
        # The same as text, for torch.vm.regions ("setattr 'grid' of a Layer").
        self.text: list[str] = []
        # The targets the region holds itself.
        self.pinned: list[Any] = []
        # Whether the recorder's list was taken, and the writes of it whose
        # places still wait for their plain values.
        self.taken = False
        self.pending: list[tuple[Any, Any, tuple[Any, ...]]] = []

    @property
    def targets(self) -> set[int]:
        return {id(target) for _, target, _ in self.recorded}

    def take(self, recorded, plain) -> None:
        """Takes the recorder's list when the body is over. The run made its
        writes with the tracer's stand-ins in them: each place that still
        holds one is given the plain value; a place written again since is
        left."""
        self.recorded = recorded
        self.pending = list(recorded)
        self.taken = True
        self.finish(plain)

    def finish(self, plain) -> None:
        """Gives the places of the writes that still wait their plain
        values. A write that fails does not keep the ones behind it from
        theirs: the first failure is raised when all were tried."""
        failed = None
        while self.pending:
            how, target, values = self.pending.pop(0)
            try:
                real = [plain(v) for v in values]
                if any(r is not v for r, v in zip(real, values)):
                    _put_plain(how, target, values, real)
            except BaseException as e:
                # (An interrupt too: the writes behind it get their floats
                # first, and it is raised then.)
                failed = failed or e
        if failed is not None:
            raise failed

    def plan(self, outs, args, old, objects, describe, shared, made) -> None:
        """Fills ``templates``, ``text`` and ``pinned`` (appending what the
        templates refer to to ``objects``, as ``describe`` does). ``outs``:
        the outputs of the traced run, which the region builds too.
        ``made(x)``: whether the run made ``x``, a list, dict or set, or an
        object that a region builds anew from its fields."""
        # What the run stored into an object it made is in that object as
        # the region builds it, and nowhere else.
        values = [
            v
            for _, target, vs in self.recorded
            if isinstance(target, (list, dict, set)) or not made(target)
            for v in vs
        ]
        if shared([*outs, *values], old):
            raise Unrepeatable("a container the run made is held in two places")
        built = _built([*outs, *values], old)
        for how, target, vs in self.recorded:
            if id(target) in built:
                continue
            # A container the run made and hands to nobody (the list a loop
            # collects tensors in for one operator call): nothing of the
            # write outlives the run. An object the run made is built as the
            # run left it, if it is handed on at all.
            if made(target):
                continue
            inside = (
                isinstance(target, (list, dict, set))
                and type(old.get(id(target))) is tuple
            )
            if inside or any(target is a for a in args):
                where = describe(target)
            else:
                objects.append(target)
                self.pinned.append(target)
                where = ("obj", len(objects) - 1)
            try:
                items = [where, *(describe(v) for v in vs)]
            except Exception as e:
                # Whatever describe raises for a value it has no template
                # for, said of a write.
                e.args = (f"it writes what a region cannot make again: {e}",)
                raise
            objects.append(how)
            self.templates.append(("call", len(objects) - 1, items))
            name = getattr(how, "__name__", "?").strip("_")
            key = f" {vs[0]!r}" if vs and type(vs[0]) in (str, int) else ""
            self.text.append(f"{name}{key} of a {type(target).__name__}")

    def unreachable(self, reads, given) -> str | None:
        """Why the region cannot hold one of its targets (see the module's
        guarantees), or None. ``reads``: what the trace read
        (torch._rust.Reads), or None for a trace that read nothing;
        ``given``: the region's arguments, which were there before the run."""
        for target in self.pinned:
            if reads is None or not reads.knows(target, given):
                return f"it writes to a {type(target).__name__} that nothing it read leads to"
        return None
