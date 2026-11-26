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
    call: no stand-in of the tracer's stays where the run wrote one.
  - ``plan`` gives one call template per write that outlives the run, in the
    run's order, or raises ``Unrepeatable``. The target is an argument, a
    container inside one (found by its path on every run), or the object
    itself, which the region then holds (``pinned``). A write into a
    container the region builds anew, for an output or for a written value,
    is none: the container is built as the run left it.
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


def _stored(how: str, target, key):
    """What the write ``how`` left in ``target`` under ``key``, read without
    running code of the target's class."""
    try:
        if how == "__setattr__":
            return object.__getattribute__(target, key)
        if isinstance(target, dict):
            return dict.get(target, key, _NOTHING)
        return list.__getitem__(target, key)
    except (AttributeError, IndexError, TypeError):
        return _NOTHING


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

    @property
    def targets(self) -> set[int]:
        return {id(target) for _, target, _ in self.recorded}

    def take(self, recorded, plain) -> None:
        """Takes the recorder's list when the body is over. The run made its
        writes with the tracer's stand-ins in them: each place that still
        holds one is given the plain value; a place written again since is
        left."""
        self.recorded = recorded
        for how, target, values in recorded:
            real = [plain(v) for v in values]
            if all(r is v for r, v in zip(real, values)):
                continue
            name = getattr(how, "__name__", "")
            if name == "append":
                for i, v in enumerate(target):
                    if v is values[0]:
                        target[i] = real[0]
            elif name == "add":
                if values[0] in target:
                    target.discard(values[0])
                    target.add(real[0])
            elif name in ("__setattr__", "__setitem__"):
                if _stored(name, target, values[0]) is values[1]:
                    how(target, *real)

    def plan(self, outs, args, old, objects, describe, shared) -> None:
        """Fills ``templates``, ``text`` and ``pinned`` (appending what the
        templates refer to to ``objects``, as ``describe`` does). ``outs``:
        the outputs of the traced run, which the region builds too."""
        values = [v for _, _, vs in self.recorded for v in vs]
        if shared([*outs, *values], old):
            raise Unrepeatable("a container the run made is held in two places")
        built = _built([*outs, *values], old)
        for how, target, vs in self.recorded:
            if id(target) in built:
                continue
            inside = isinstance(target, (list, dict, set)) and type(old.get(id(target))) is tuple
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

    def unreachable(self, reads) -> str | None:
        """Why the region cannot hold one of its targets (see the module's
        guarantees), or None. ``reads``: what the trace read
        (torch._rust.Reads), or None for a trace that read nothing."""
        for target in self.pinned if reads is not None else ():
            if not reads.knows(target):
                return f"it writes to a {type(target).__name__} that nothing it read leads to"
        return None
