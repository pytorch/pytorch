# mypy: allow-untyped-defs
"""What a cfg pass gets: ``(cfg: vm.Cfg) -> None``.

A cfg pass changes what a region is. The VM calls it on each function's
control-flow graph (built from its bytecode) when it optimizes that function,
after watching it run: so the CFG carries what the runs recorded at each
instruction (the torch ops it ran, the Python frames it entered, the types it
produced, the callee it called) and which instructions may sit in a region.

Analyses: ``blocks``, ``calls()``, ``opaque_calls()`` (calls that end a
region), ``spans()`` (stretches of blocks the VM will trace as one region),
``loops()``, ``branches()``, ``collectives()``, ``effects(v)``,
``type_of(v)``. Edits: ``merge_across(call)`` lets a call sit in a region,
``merge_blocks(entry, exit)`` makes a stretch of blocks one region, and
``split(block)`` keeps a block out of any. The rest of the generic edits
(inlining, specializing, software-pipelining a loop, autograd regions) are
declared but not implemented yet, and raise.

The VM's own passes (``vm.DEFAULT_PASSES``, ``Skip``) configure it once,
when ``torch.vm`` is called: they get the program's ``Cfg``, with no
function's CFG in it.
"""

import dataclasses
from collections.abc import Callable, Iterable
from typing import Any


class CannotGuard(Exception):
    """A value can't be specialized: the slice behind it reads tensor data or
    something no check kind expresses."""


def _todo(what: str):
    qualified = what if "." in what else f"Cfg.{what}"

    def method(self, *args, **kwargs):
        raise NotImplementedError(f"vm.{qualified} isn't implemented yet")

    method.__name__ = what.rpartition(".")[2]
    return method


class CfgValue:
    """An SSA value of the function and the instruction defining it
    (``ir::ValueId`` and its ``Instr``): its op as the VM prints it, its args,
    and what the runs recorded there."""

    def __init__(self, cfg: "Cfg", id: int, data: dict[str, Any]) -> None:
        self._cfg = cfg
        self.id = id
        self.op: str = data["op"]
        self.kind: str = data["kind"]
        self.line: int = data["line"]
        self.origin: int = data["origin"]
        self.movable: bool = data["movable"]
        # Why the VM does not take it for something that only computes (the
        # reason TORCH_RUST_PURE_DEBUG prints), if it judged it and said no.
        self.why: str | None = data.get("why")
        self._effects: list[str] = data["effects"]
        self._type: str | None = data["type"]
        self._args: list[int] = data["args"]
        self._block: int = data["block"]

    @property
    def target(self) -> str:
        return self.op

    @property
    def args(self) -> list["CfgValue | None"]:
        """Its args that are instructions of the function (None for a block
        param or a value the VM made)."""
        return [self._cfg._values.get(a) for a in self._args]

    @property
    def users(self) -> list["CfgValue"]:
        return [v for v in self._cfg._values.values() if self.id in v._args]

    @property
    def block(self) -> "Block":
        return self._cfg.block(self._block)

    replace_input = _todo("CfgValue.replace_input")

    def __repr__(self) -> str:
        return f"%{self.id} = {self.op}  (line {self.line})"


class Call(CfgValue):
    """A call, with the callee the runs recorded (None if it varied)."""

    def __init__(self, cfg: "Cfg", id: int, data: dict[str, Any]) -> None:
        super().__init__(cfg, id, data)
        self.callee: Any = data["callee"]

    def resolves_to(self) -> Any:
        return self.callee

    def callee_type(self) -> type:
        return type(self.callee)

    def __repr__(self) -> str:
        name = getattr(self.callee, "__qualname__", None) or type(self.callee).__name__
        return f"%{self.id} = call {name}  (line {self.line})"


class Block:
    """A block of the function's CFG (``ir::Block``): its instructions as the
    runs saw them, and where it goes."""

    def __init__(self, cfg: "Cfg", data: dict[str, Any]) -> None:
        self._cfg = cfg
        self.id: int = data["id"]
        self._params: list[int] = data["params"]
        self._instrs: list[int] = data["instrs"]
        self._succs: list[int] = data["succs"]
        self.term: str = data["term"]
        self.seen: bool = data["seen"]

    @property
    def params(self) -> list[int]:
        return list(self._params)

    @property
    def instrs(self) -> list[CfgValue]:
        return [self._cfg._values[v] for v in self._instrs]

    @property
    def succs(self) -> list["Block"]:
        return [self._cfg.block(s) for s in self._succs]

    @property
    def lines(self) -> tuple[int, int]:
        ls = [v.line for v in self.instrs if v.line]
        return (min(ls), max(ls)) if ls else (0, 0)

    def find(self, pred: Callable[[CfgValue], bool]) -> list[CfgValue]:
        return [v for v in self.instrs if pred(v)]

    defines = _todo("Block.defines")
    clone = _todo("Block.clone")
    clone_slice = _todo("Block.clone_slice")
    replace_all = _todo("Block.replace_all")
    remove_slice = _todo("Block.remove_slice")

    def __repr__(self) -> str:
        lo, hi = self.lines
        return f"bb{self.id} (lines {lo}-{hi}, {len(self._instrs)} instrs, -> {self._succs})"


@dataclasses.dataclass(frozen=True)
class Span:
    """Blocks from ``entry`` to ``exit`` (None: to a return) traced as one
    region."""

    entry: Block
    exit: Block | None

    def __repr__(self) -> str:
        return f"span bb{self.entry.id} -> {'return' if self.exit is None else f'bb{self.exit.id}'}"


@dataclasses.dataclass(frozen=True)
class Loop:
    """A loop: ``header`` is the block its back-edge (from ``latch``) goes to;
    ``blocks`` are those on a path from the header back to it."""

    header: Block
    latch: Block
    blocks: tuple[Block, ...]

    item_at = _todo("Loop.item_at")
    unroll = _todo("Loop.unroll")
    peel_last = _todo("Loop.peel_last")
    add_carried = _todo("Loop.add_carried")
    is_invariant = _todo("Loop.is_invariant")
    trip_counts = _todo("Loop.trip_counts")
    body_guards = _todo("Loop.body_guards")


@dataclasses.dataclass(frozen=True)
class Branch:
    block: Block
    targets: tuple[Block, Block]


@dataclasses.dataclass(frozen=True)
class Merge:
    """What ``cfg.merge_across`` did: the call may now sit in a region."""

    call: Call
    reads: tuple[Any, ...]
    writes: tuple[Any, ...]
    capturable: bool
    traced: bool

    guard = _todo("Merge.guard")
    tensor_inputs = _todo("Merge.tensor_inputs")


class Cfg:
    """The program ``torch.vm`` was given (for the VM's own passes), or one
    function's control-flow graph as its runs observed it (for a cfg pass:
    ``qualname`` is set)."""

    def __init__(self, fn: Callable, view: dict[str, Any] | None = None) -> None:
        self.fn = fn
        # The VM's own passes, by name (each sets itself here).
        self.builtin: dict[str, Any] = {}
        self.skipped: list[Callable] = []
        self.autograd: Any = None
        self.partition: Any = None
        self.qualname: str | None = None
        self.filename: str | None = None
        self.code: int | None = None
        self.notes: list[str] = []
        self._edits: dict[str, list[Any]] = {
            "movable": [],
            "spans": [],
            "split": [],
            "opaque": [],
        }
        self._values: dict[int, CfgValue] = {}
        self._blocks: dict[int, Block] = {}
        self._spans: list[tuple[int, int | None]] = []
        if view is not None:
            self.qualname = view["qualname"]
            self.filename = view["filename"]
            self.code = view["code"]
            for k, data in view["instrs"].items():
                call = data["kind"] == "Call" or "CALL_FUNCTION_EX" in data["op"]
                self._values[k] = (Call if call else CfgValue)(self, k, data)
            self._blocks = {b["id"]: Block(self, b) for b in view["blocks"]}
            self._spans = list(view["spans"])

    def skip(self, *fns: Callable) -> None:
        """Runs these functions, and everything they call, on CPython."""
        self.skipped.extend(fns)

    def _function(self) -> None:
        if self.qualname is None:
            raise TypeError(
                "this is the program's Cfg (the VM's own passes get it); a cfg pass gets each function's"
            )

    # Analyses.

    @property
    def blocks(self) -> list[Block]:
        self._function()
        return list(self._blocks.values())

    def block(self, id: int) -> Block:
        self._function()
        return self._blocks[id]

    def values(self) -> list[CfgValue]:
        self._function()
        return list(self._values.values())

    def calls(self, to: str | Callable | None = None) -> list[Call]:
        """The calls the runs made, or those to ``to`` (a callable, or a
        substring of its qualified name)."""
        self._function()
        out = [v for v in self._values.values() if isinstance(v, Call)]
        if to is None:
            return out
        if callable(to) and not isinstance(to, str):
            return [c for c in out if c.callee is to]
        return [
            c
            for c in out
            if to
            in f"{getattr(c.callee, '__module__', '')}.{getattr(c.callee, '__qualname__', '')}"
        ]

    def opaque_calls(self) -> list[Call]:
        """The calls that can't sit in a region (they end one): those that
        ran something other than torch ops, or no torch op the VM could see."""
        return [c for c in self.calls() if not c.movable and c.block.seen]

    def why_opaque(self, call: Call) -> str | None:
        """Why ``call`` can't sit in a region, in the VM's words: the chain
        from the call to the instruction that decides (``"%66 Call -> f: %16
        Call called <built-in function print>"``, ``"... ran aten::item"``,
        ``"... writes"``). None for a call that is movable, and for one the
        VM didn't judge (pure calls are off)."""
        return None if call.movable else call.why

    def spans(self) -> list[Span]:
        """The stretches of blocks the VM will trace as one region each."""
        self._function()
        return [
            Span(self.block(e), None if x is None else self.block(x))
            for e, x in self._spans
        ]

    def loops(self) -> list[Loop]:
        """Loops, from their back-edges (blocks are numbered in reverse
        postorder, so an edge to a block numbered no higher is one)."""
        self._function()
        out = []
        for b in self._blocks.values():
            for s in b._succs:
                if s <= b.id:
                    blocks = [x for x in self._blocks.values() if s <= x.id <= b.id]
                    out.append(Loop(self.block(s), b, tuple(blocks)))
        return out

    def branches(self) -> list[Branch]:
        self._function()
        return [
            Branch(b, (b.succs[0], b.succs[1]))
            for b in self._blocks.values()
            if b.term == "branch" and len(b._succs) == 2
        ]

    def effects(self, v: CfgValue) -> list[str]:
        """What the runs did at v: the torch ops it ran and Python frames it
        entered."""
        return list(v._effects)

    def type_of(self, v: CfgValue) -> str | None:
        """The join of the values v produced (``types::Ty``), as text."""
        return v._type

    def collectives(self) -> list[CfgValue]:
        return [v for v in self.values() if any("c10d" in e for e in v._effects)]

    # Edits.

    def merge_across(
        self,
        call: Call,
        reads: Iterable[Any] = (),
        writes: Iterable[Any] = (),
        capturable: bool = False,
        trace: bool = False,
    ) -> Merge:
        """Lets ``call`` sit in a region, so the regions on either side of it
        become one. By default a region's trace records this call (not other
        calls of its callee) as one node calling the callee, so every run
        calls it as eager does (an opaque
        effect: block passes see the node, not what it does). With ``trace``,
        the trace records the torch ops it runs instead: only sound if it
        does nothing else (no Python-level effects)."""
        self._function()
        if not isinstance(call, Call):
            raise TypeError(f"merge_across takes a Call, not {call!r}")
        if call.callee is None:
            raise ValueError(f"{call!r} called different things on different runs")
        if not trace:
            self._edits["opaque"].append((self.code, call.origin))
        self._edits["movable"].append(call.id)
        self.notes.append(f"merge_across {call!r}{' (traced)' if trace else ''}")
        return Merge(call, tuple(reads), tuple(writes), capturable, trace)

    def merge_blocks(self, entry: Block, exit: Block | None = None) -> Span:
        """Makes the blocks from ``entry`` to ``exit`` (None: to a return) one
        region, however few ops they run. They must only compute (their calls
        movable, see merge_across), and leave only to ``exit``; if they
        don't, the VM keeps its own spans (``TORCH_RUST_SPAN_DEBUG=1`` says
        why)."""
        self._function()
        self._edits["spans"].append((entry.id, None if exit is None else exit.id))
        self.notes.append(
            f"merge_blocks bb{entry.id} -> {'return' if exit is None else f'bb{exit.id}'}"
        )
        return Span(entry, exit)

    def split(self, block: Block) -> None:
        """Keeps ``block`` out of any span."""
        self._function()
        self._edits["split"].append(block.id)
        self.notes.append(f"split bb{block.id}")

    edit = _todo("edit")
    guard = _todo("guard")
    remove = _todo("remove")
    autograd_region = _todo("autograd_region")
    inline = _todo("inline")
    specialize = _todo("specialize")
    simplify = _todo("simplify")
    slice = _todo("slice")
    live_bytes = _todo("live_bytes")
    peak_live_bytes = _todo("peak_live_bytes")
    straight_chains = _todo("straight_chains")
    guard_checks = _todo("guard_checks")

    @property
    def raw(self):
        raise NotImplementedError("vm.Cfg.raw isn't implemented yet")

    def __repr__(self) -> str:
        if self.qualname is None:
            return f"Cfg(program {getattr(self.fn, '__qualname__', self.fn)})"
        return f"Cfg({self.qualname}: {len(self._blocks)} blocks, {len(self._spans)} spans)"


class Slice:
    """An op and the ops feeding it (``cfg.slice``)."""

    root: CfgValue
    ops: list[CfgValue]
    inputs: list[CfgValue]


class Carried:
    """A value passed across a loop's back-edge (a header block param)."""

    param: CfgValue
    set_next = _todo("Carried.set_next")


class Edit:
    rollback = _todo("Edit.rollback")


class Region:
    """An autograd unit (``cfg.autograd_region``)."""

    blocks: list[Block]
    entry: Block
    exit: Block
