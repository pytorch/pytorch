import dataclasses
import dis
import sys
from numbers import Real

TERMINAL_OPCODES = {
    dis.opmap["RETURN_VALUE"],
    dis.opmap["JUMP_ABSOLUTE"],
    dis.opmap["JUMP_FORWARD"],
    dis.opmap["RAISE_VARARGS"],
    # TODO(jansel): double check exception handling
}
if sys.version_info >= (3, 9):
    TERMINAL_OPCODES.add(dis.opmap["RERAISE"])
JUMP_OPCODES = set(dis.hasjrel + dis.hasjabs)
HASLOCAL = set(dis.haslocal)
HASFREE = set(dis.hasfree)

if sys.version_info < (3, 8):

    def stack_effect(opcode, arg, jump=None):
        # jump= was added in python 3.8, we just ingore it here
        if dis.opname[opcode] in ("NOP", "EXTENDED_ARG"):
            # for some reason NOP isn't supported in python 3.7
            return 0
        return dis.stack_effect(opcode, arg)

else:
    stack_effect = dis.stack_effect


def remove_dead_code(instructions):
    """Dead code elimination"""
    indexof = {id(inst): i for i, inst in enumerate(instructions)}
    live_code = set()

    def find_live_code(start):
        for i in range(start, len(instructions)):
            if i in live_code:
                return
            live_code.add(i)
            inst = instructions[i]
            if inst.opcode in JUMP_OPCODES:
                find_live_code(indexof[id(inst.target)])
            if inst.opcode in TERMINAL_OPCODES:
                return

    find_live_code(0)
    return [inst for i, inst in enumerate(instructions) if i in live_code]


def remove_pointless_jumps(instructions):
    """Eliminate jumps to the next instruction"""
    pointless_jumps = {
        id(a)
        for a, b in zip(instructions, instructions[1:])
        if a.opname == "JUMP_ABSOLUTE" and a.target is b
    }
    return [inst for inst in instructions if id(inst) not in pointless_jumps]


def propagate_line_nums(instructions):
    """Ensure every instruction has line number set in case some are removed"""
    cur_line_no = None

    def populate_line_num(inst):
        nonlocal cur_line_no
        if inst.starts_line:
            cur_line_no = inst.starts_line

        inst.starts_line = cur_line_no

    for inst in instructions:
        populate_line_num(inst)


def remove_extra_line_nums(instructions):
    """Remove extra starts line properties before packing bytecode"""

    cur_line_no = None

    def remove_line_num(inst):
        nonlocal cur_line_no
        if inst.starts_line is None:
            return
        elif inst.starts_line == cur_line_no:
            inst.starts_line = None
        else:
            cur_line_no = inst.starts_line

    for inst in instructions:
        remove_line_num(inst)


@dataclasses.dataclass
class ReadsWrites:
    reads: set
    writes: set
    visited: set


def livevars_analysis(instructions, instruction):
    indexof = {id(inst): i for i, inst in enumerate(instructions)}
    must = ReadsWrites(set(), set(), set())
    may = ReadsWrites(set(), set(), set())

    def walk(state, start):
        if start in state.visited:
            return
        state.visited.add(start)

        for i in range(start, len(instructions)):
            inst = instructions[i]
            if inst.opcode in HASLOCAL or inst.opcode in HASFREE:
                if "LOAD" in inst.opname or "DELETE" in inst.opname:
                    if inst.argval not in must.writes:
                        state.reads.add(inst.argval)
                elif "STORE" in inst.opname:
                    state.writes.add(inst.argval)
                else:
                    raise NotImplementedError(f"unhandled {inst.opname}")
            if inst.opcode in JUMP_OPCODES:
                walk(may, indexof[id(inst.target)])
                state = may
            if inst.opcode in TERMINAL_OPCODES:
                return

    walk(must, indexof[id(instruction)])
    return must.reads | may.reads


@dataclasses.dataclass
class FixedPointBox:
    value: bool = True


@dataclasses.dataclass
class StackSize:
    low: Real
    high: Real
    fixed_point: FixedPointBox

    def zero(self):
        self.low = 0
        self.high = 0
        self.fixed_point.value = False

    def offset_of(self, other, n):
        prior = (self.low, self.high)
        self.low = min(self.low, other.low + n)
        self.high = max(self.high, other.high + n)
        if (self.low, self.high) != prior:
            self.fixed_point.value = False


def stacksize_analysis(instructions):
    assert instructions
    fixed_point = FixedPointBox()
    stack_sizes = {
        inst: StackSize(float("inf"), float("-inf"), fixed_point)
        for inst in instructions
    }
    stack_sizes[instructions[0]].zero()

    for _ in range(100):
        if fixed_point.value:
            break
        fixed_point.value = True

        for inst, next_inst in zip(instructions, instructions[1:] + [None]):
            stack_size = stack_sizes[inst]
            if inst.opcode not in TERMINAL_OPCODES:
                assert next_inst is not None, f"missing next inst: {inst}"
                stack_sizes[next_inst].offset_of(
                    stack_size, stack_effect(inst.opcode, inst.arg, jump=False)
                )
            if inst.opcode in JUMP_OPCODES:
                stack_sizes[inst.target].offset_of(
                    stack_size, stack_effect(inst.opcode, inst.arg, jump=True)
                )

    if False:
        for inst in instructions:
            stack_size = stack_sizes[inst]
            print(stack_size.low, stack_size.high, inst)

    low = min([x.low for x in stack_sizes.values()])
    high = max([x.high for x in stack_sizes.values()])

    if sys.version_info < (3, 8) and not fixed_point.value:
        # This is a rare issue in python 3.7 that still needs debugging
        # see test/test_nops.py::NopTests::test3
        return low + 32

    assert fixed_point.value, "failed to reach fixed point"
    assert low >= 0
    return high
