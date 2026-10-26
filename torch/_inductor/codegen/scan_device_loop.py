"""
NOTE [scan device loop]

``decompose_scan_to_while_loop`` turns ``scan`` into a ``while_loop`` whose body
runs one step per host iteration: every step launches the body kernel and
round-trips the carry through global memory. With ``config.scan_device_loop``,
a scan whose body schedules to exactly one Triton kernel is instead compiled
into a kernel that runs every step itself::

    carry = tl.load(carry_in + idx)            # hoisted above the loop
    for scan_idx in range(0, T):
        ... per-step loads, compute, ys stores at scan_idx ...
        carry = new_carry                      # stays in registers
    tl.store(carry_out + idx, carry)           # sunk below the loop

The wrapper then calls the body subgraph once instead of looping on the host.

This is only valid when no program of the kernel reads a carry element that a
different program wrote in the previous step. The kernel therefore has to load
every carry at exactly the index it stores it at, so each program owns its
carry elements across steps; pointwise and persistent-reduction kernels keep a
whole row in one program, so reductions inside a row are allowed. Any condition
that cannot be proven while generating the kernel makes the kernel fall back to
the per-step form and the wrapper keeps the host loop.
"""

from __future__ import annotations

import dataclasses
import re
from typing import TYPE_CHECKING

import sympy

from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.symbol import free_symbol_is_type, symbol_is_type, SymT

from .. import ir
from ..utils import sympy_subs
from ..virtualized import V
from .common import IndentedBuffer


if TYPE_CHECKING:
    from ..scheduler import BaseSchedulerNode
    from .common import CSEVariable


@dataclasses.dataclass
class ScanDeviceLoopSpec:
    """How the body subgraph of a scan-derived while_loop maps onto the loop."""

    # Symbol bound to ``loop_idx.item()`` in the body; becomes the loop variable.
    loop_index: sympy.Symbol
    trip_count: sympy.Expr
    # Body input buffer name -> body output buffer name of the same carry.
    carries: dict[str, str]
    # ys buffers: body inputs that the body writes in place at the loop index.
    ys: OrderedSet[str]
    # The loop counter's body input and the ``loop_idx + 1`` body output.
    counter_input: str
    counter_output: str
    applied: bool = False

    @classmethod
    def from_while_loop(cls, while_loop: ir.WhileLoop) -> ScanDeviceLoopSpec | None:
        """Return the spec if ``while_loop`` came from a scan the loop can cover."""
        if while_loop.scan_layout is None or while_loop.body_subgraph is None:
            return None
        graph = while_loop.body_subgraph.graph
        if graph is None:
            raise AssertionError("expected a lowered body subgraph")
        num_ys, num_carries = while_loop.scan_layout
        names = graph.graph_input_names
        outputs = graph.graph_outputs
        xs_start = 1 + num_ys + num_carries
        num_carried = len(while_loop.carried_inputs or ())
        if xs_start >= num_carried:
            # A scan without xs has no trip count to read off the inputs.
            return None
        for name in names[1:num_carried]:
            buf = graph.graph_inputs[name]
            if not isinstance(buf, ir.TensorBox):
                return None
            device = buf.get_device()
            if device is None or device.type == "cpu":
                return None

        loop_index = None
        for op in graph.operations:
            if isinstance(op, ir.DynamicScalar) and op.inputs[0].get_name() == names[0]:
                loop_index = op.sym
        if loop_index is None:
            return None

        ys = names[1 : 1 + num_ys]
        if any(outputs[1 + i].get_name() != ys[i] for i in range(num_ys)):
            return None
        carry_positions = range(1 + num_ys, xs_start)
        for i in carry_positions:
            # Equal load and store indices only mean "same element" when the
            # carry output is a whole buffer laid out like the carry input.
            buf = graph.try_get_buffer(outputs[i].get_name())
            if (
                buf is None
                or buf.get_layout() != outputs[i].get_layout()
                or buf.get_layout().offset != 0
                or graph.graph_inputs[names[i]].get_layout().offset != 0
            ):
                return None
        return cls(
            loop_index=loop_index,
            trip_count=graph.graph_inputs[names[xs_start]].get_size()[0],
            carries={names[i]: outputs[i].get_name() for i in carry_positions},
            ys=OrderedSet(ys),
            counter_input=names[0],
            counter_output=outputs[0].get_name(),
        )

    def is_loop_bookkeeping(self, node: BaseSchedulerNode) -> bool:
        """Whether ``node`` is the scan's ``loop_idx.item()``, an assert on its
        range, or ``loop_idx + 1``.

        The device loop runs every other body op once instead of once per step,
        so only these may sit outside the looped kernel.
        """
        ir_node = node.node
        if isinstance(ir_node, ir.DynamicScalar):
            return ir_node.sym == self.loop_index
        if isinstance(ir_node, ir.AssertScalar):
            allowed = OrderedSet([self.loop_index, *self.trip_count.free_symbols])
            return OrderedSet(ir_node.scalar.free_symbols) <= allowed
        return (
            isinstance(ir_node, ir.ComputedBuffer)
            and node.is_cpu()
            and ir_node.get_name() == self.counter_output
            and all(dep.name == self.counter_input for dep in node.read_writes.reads)
        )


@dataclasses.dataclass
class CarryLoad:
    index: sympy.Expr
    var: CSEVariable


class ScanDeviceLoopKernelState:
    """Per-kernel record of carry accesses while a scan body kernel is generated."""

    # Not a SymT prefix, so rename_indexing leaves it alone.
    loop_symbol = sympy.Symbol("loop_idx", integer=True, nonnegative=True)

    def __init__(self, spec: ScanDeviceLoopSpec) -> None:
        self.spec = spec
        self.carry_inputs = {out: inp for inp, out in spec.carries.items()}
        self.loads: dict[str, CarryLoad] = {}
        self.stored: OrderedSet[str] = OrderedSet()
        # Carries that keep a per-step store because their input was not loaded
        # before the store.
        self.unhoisted: OrderedSet[str] = OrderedSet()
        self.carry_updates: list[str] = []
        self.epilogue = IndentedBuffer()
        self.trip_count = ""
        self.failure: str | None = None

    def fail(self, reason: str) -> None:
        if self.failure is None:
            self.failure = reason

    def substitute_loop_index(self, index: sympy.Expr) -> sympy.Expr:
        loop_index = self.spec.loop_index
        for sym in index.free_symbols:
            if sym == loop_index:
                continue
            # Any other unbacked symbol is passed by value, which would freeze a
            # per-step value at the first step.
            if symbol_is_type(sym, (SymT.UNBACKED_INT, SymT.UNBACKED_FLOAT)) or (
                symbol_is_type(sym, SymT.PRECOMPUTED_SIZE)
                and loop_index
                in V.graph.sizevars.inv_precomputed_replacements[sym].free_symbols
            ):
                self.fail(f"index {index} uses {sym}, which the loop cannot update")
        return sympy_subs(index, {loop_index: self.loop_symbol})

    def check_carry_index(self, name: str, index: sympy.Expr) -> bool:
        if free_symbol_is_type(index, SymT.TMP) or self.spec.loop_index in (
            index.free_symbols
        ):
            self.fail(f"carry {name} is indexed indirectly or by the loop index")
            return False
        return True

    def uses_loop_var(self, buffer: IndentedBuffer) -> bool:
        pattern = re.compile(rf"\b{self.loop_symbol.name}\b")
        return any(
            pattern.search(str(getattr(line, "line", line))) for line in buffer._lines
        )
