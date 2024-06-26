# mypy: allow-untyped-defs
import copy
import dataclasses
import dis
import itertools
import sys
import types
from typing import Any, Callable, cast, Dict, Iterator, List, Optional, Tuple, Union

from .bytecode_analysis import (
    get_indexof,
    propagate_line_nums,
    remove_extra_line_nums,
    stacksize_analysis,
)


@dataclasses.dataclass
class InstructionExnTabEntry:
    start: "Instruction"
    end: "Instruction"
    target: "Instruction"
    depth: int
    lasti: bool

    def __repr__(self) -> str:
        return (
            f"InstructionExnTabEntry(start={self.start.short_inst_repr()}, "
            f"end={self.end.short_inst_repr()}, "
            f"target={self.target.short_inst_repr()}, "
            f"depth={self.depth}, lasti={self.lasti})"
        )

    def __eq__(self, o) -> bool:
        return (
            self.start is o.start
            and self.end is o.end
            and self.target is o.target
            and self.depth == o.depth
            and self.lasti == o.lasti
        )


@dataclasses.dataclass
class Instruction:
    """A mutable version of dis.Instruction"""

    opcode: int
    opname: str
    arg: Optional[int]
    argval: Any
    offset: Optional[int] = None
    starts_line: Optional[int] = None
    is_jump_target: bool = False
    positions: Optional["dis.Positions"] = None
    # extra fields to make modification easier:
    target: Optional["Instruction"] = None
    exn_tab_entry: Optional[InstructionExnTabEntry] = None

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other) -> bool:
        return id(self) == id(other)

    def short_inst_repr(self) -> str:
        return f"Instruction(opname={self.opname}, offset={self.offset})"


def convert_instruction(i: dis.Instruction) -> Instruction:
    return Instruction(
        i.opcode,
        i.opname,
        i.arg,
        i.argval,
        i.offset,
        i.starts_line,
        i.is_jump_target,
        getattr(i, "positions", None),
    )


class _NotProvided:
    def __repr__(self) -> str:
        return "_NotProvided"


def inst_has_op_bits(name):
    return (sys.version_info >= (3, 11) and name == "LOAD_GLOBAL") or (
        sys.version_info >= (3, 12) and name in ("LOAD_ATTR", "LOAD_SUPER_ATTR")
    )


def create_instruction(
    name, *, arg=None, argval=_NotProvided, target=None
) -> Instruction:
    """
    At most one of `arg`, `argval`, and `target` can be not None/_NotProvided.
    This is to prevent ambiguity, e.g. does
        create_instruction("LOAD_CONST", 5)
    mean load the constant at co_consts[5], or load the constant 5?

    If `arg` is not provided, it will be computed during assembly from
    `argval` or `target`.

    Bits in the args of instructions LOAD_GLOBAL, LOAD_ATTR (3.12+), and LOAD_SUPER_ATTR
    modify the behavior of the instruction. In this case, we allow both `arg`
    and `argval` to be set. The value of `arg` here is expected to be the value of
    the op bits and the true value of `arg` will be computed during assembly.
    If `arg` is not set, the bits are assumed to be 0.
    """

    # allow for instructions with op bits to have both arg and argval specified
    if inst_has_op_bits(name):
        if target is not None:
            raise RuntimeError("target cannot be specified for instruction")
        if arg is None:
            arg = 0
    else:
        cnt = (arg is not None) + (argval is not _NotProvided) + (target is not None)
        if cnt > 1:
            raise RuntimeError(
                "only one of arg, argval, and target can be not None/_NotProvided"
            )
    if arg is not None and not isinstance(arg, int):
        raise RuntimeError("instruction arg must be int or None")
    return Instruction(
        opcode=dis.opmap[name], opname=name, arg=arg, argval=argval, target=target
    )


# Python 3.11 remaps
def create_jump_absolute(target) -> Instruction:
    inst = "JUMP_FORWARD" if sys.version_info >= (3, 11) else "JUMP_ABSOLUTE"
    return create_instruction(inst, target=target)


def create_dup_top() -> Instruction:
    if sys.version_info >= (3, 11):
        return create_instruction("COPY", arg=1)
    return create_instruction("DUP_TOP")


def create_rot_n(n) -> List[Instruction]:
    """
    Returns a "simple" sequence of instructions that rotates TOS to the n-th
    position in the stack. For Python < 3.11, returns a single ROT_*
    instruction. If no such instruction exists, an error is raised and the
    caller is expected to generate an equivalent sequence of instructions.
    For Python >= 3.11, any rotation can be expressed as a simple sequence of
    swaps.
    """
    if n <= 1:
        # don't rotate
        return []

    if sys.version_info >= (3, 11):
        # rotate can be expressed as a sequence of swap operations
        # e.g. rotate 3 is equivalent to swap 3, swap 2
        return [create_instruction("SWAP", arg=i) for i in range(n, 1, -1)]

    # ensure desired rotate function exists
    if sys.version_info < (3, 8) and n >= 4:
        raise AttributeError(f"rotate {n} not supported for Python < 3.8")
    if sys.version_info < (3, 10) and n >= 5:
        raise AttributeError(f"rotate {n} not supported for Python < 3.10")

    if n <= 4:
        return [create_instruction("ROT_" + ["TWO", "THREE", "FOUR"][n - 2])]
    return [create_instruction("ROT_N", arg=n)]


def add_push_null(
    inst_or_insts: Union[Instruction, List[Instruction]],
) -> List[Instruction]:
    """
    Appends or prepends a PUSH_NULL instruction to `inst_or_insts`,
    depending on Python version. Used when you know that
    `inst_or_insts` generates a callable that will be called.

    NOTE: Assumes `inst_or_insts` is a single instruction or sequence of
    instructions that pushes exactly 1 object to the stack that is to
    be called. It is important that you include ALL instructions that
    construct the callable - not just the first instruction/a prefix.

    Will attempt to use the NULL push bit for instructions
    with such bits (LOAD_GLOBAL 3.11+, LOAD_ATTR 3.12+, LOAD_SUPER_ATTR).
    In this case, instructions WILL be modified.
    """
    if isinstance(inst_or_insts, Instruction):
        insts = [inst_or_insts]
    else:
        insts = inst_or_insts

    def inst_has_bit_set(idx):
        assert insts[idx].arg is not None
        return insts[idx].arg & 1 == 1

    def set_inst_bit(idx):
        assert insts[idx].arg is not None
        insts[idx].arg |= 1

    if sys.version_info >= (3, 13):
        # In 3.13, NULL follows the callable
        if inst_has_op_bits(insts[-1].opname) and not inst_has_bit_set(-1):
            # All insts with op bits have the push_null bit as the last one.
            # Only set the bit if it hasn't been set - otherwise, we need
            # to add another PUSH_NULL.
            set_inst_bit(-1)
        else:
            insts = insts + [create_instruction("PUSH_NULL")]
    elif sys.version_info >= (3, 12):
        # LOAD_ATTR/LOAD_SUPER_ATTR at the end
        # We assume that `insts` will only load 1 object, so
        # LOAD_GLOBAL at the end doesn't need to be checked
        if inst_has_op_bits(insts[-1].opname) and not inst_has_bit_set(-1):
            set_inst_bit(-1)
        elif insts[0].opname == "LOAD_GLOBAL" and not inst_has_bit_set(0):
            set_inst_bit(0)
        else:
            insts = [create_instruction("PUSH_NULL")] + insts
    elif sys.version_info >= (3, 11):
        # 3.11 introduced NULL preceding callable
        if inst_has_op_bits(insts[0].opname) and not inst_has_bit_set(0):
            set_inst_bit(0)
        else:
            insts = [create_instruction("PUSH_NULL")] + insts
    return insts


def add_push_null_call_function_ex(
    inst_or_insts: Union[Instruction, List[Instruction]],
) -> List[Instruction]:
    """Like add_push_null, but the low bit of LOAD_ATTR/LOAD_SUPER_ATTR
    is not set, due to an expected CALL_FUNCTION_EX instruction.
    """
    if isinstance(inst_or_insts, Instruction):
        insts = [inst_or_insts]
    else:
        insts = inst_or_insts

    if sys.version_info < (3, 11):
        return insts

    idx = -1 if sys.version_info >= (3, 13) else 0
    if insts[idx].opname == "LOAD_GLOBAL":
        assert insts[idx].arg is not None
        if insts[idx].arg & 1 == 0:  # type: ignore[operator]
            insts[idx].arg |= 1  # type: ignore[operator]
            return insts

    if sys.version_info >= (3, 13):
        insts = insts + [create_instruction("PUSH_NULL")]
    else:
        insts = [create_instruction("PUSH_NULL")] + insts

    return insts


def create_call_function(nargs, push_null) -> List[Instruction]:
    """
    Creates a sequence of instructions that makes a function call.

    `push_null` is used in Python 3.11+ only. It is used in codegen when
    a function call is intended to be made with the NULL + fn convention,
    and we know that the NULL has not been pushed yet. We will push a
    NULL and rotate it to the correct position immediately before making
    the function call.

    `push_null` should be True if no NULL is pushed for the callable.
    Conversely, `push_null` should be False if a NULL was pushed for the callable.
    Prefer using `push_null=False` when possible since we will not need to rotate
    NULL to the right place, which is less efficient.

    Generally, you should codegen a function by using `add_push_null` then
    `create_call_function` with `push_null=False`.

    Example of when to set push_null False:

    insts = [
        create_instruction("LOAD_GLOBAL", argval="torch"),
        create_instruction("LOAD_ATTR", argval="nn"),
        create_instruction("LOAD_ATTR", argval="functional"),
        create_instruction("LOAD_ATTR", argval="relu"),
    ]
    insts = add_push_null(insts)
    insts.append(create_instruction("LOAD_FAST", argval="x"))
    insts.extend(create_call_function(1, False))

    Example of when to set push_null True:

    insts = [create_instruction("LOAD_FAST", x)]
    for should_wrap, wrapper_name in wrappers:
        if should_wrap:
            insts.extend([
                create_instruction("LOAD_GLOBAL", argval="wrapper1"),
                create_instruction("SWAP", arg=2),
                *create_call_function(1, True),
            )
    """
    if sys.version_info >= (3, 11):
        output = []
        if push_null:
            output.append(create_instruction("PUSH_NULL"))
            # 3.13 swapped NULL and callable
            rots = nargs + 1 if sys.version_info >= (3, 13) else nargs + 2
            output.extend(create_rot_n(rots))
        if sys.version_info < (3, 12):
            output.append(create_instruction("PRECALL", arg=nargs))
        output.append(create_instruction("CALL", arg=nargs))
        return output
    return [create_instruction("CALL_FUNCTION", arg=nargs)]


def create_call_method(nargs) -> List[Instruction]:
    if sys.version_info >= (3, 12):
        return [create_instruction("CALL", arg=nargs)]
    if sys.version_info >= (3, 11):
        return [
            create_instruction("PRECALL", arg=nargs),
            create_instruction("CALL", arg=nargs),
        ]
    return [create_instruction("CALL_METHOD", arg=nargs)]


def create_load_method(name) -> Instruction:
    if sys.version_info >= (3, 12):
        # in 3.12, create a LOAD_ATTR instruction with the low bit set
        return create_instruction("LOAD_ATTR", arg=1, argval=name)
    return create_instruction("LOAD_METHOD", argval=name)


def create_setup_with(target) -> Instruction:
    opname = "BEFORE_WITH" if sys.version_info >= (3, 11) else "SETUP_WITH"
    return create_instruction(opname, target=target)


def create_swap(n) -> List[Instruction]:
    if sys.version_info >= (3, 11):
        return [create_instruction("SWAP", arg=n)]
    # in Python < 3.11, SWAP is a macro that expands to multiple instructions
    if n == 1:
        return []
    """
    e.g. swap "a" and "b" in this stack:
    0 a 1 2 3 b
    0 a [1 2 3 b]
    0 a [1 2 3 b] [1 2 3 b]
    0 a [1 2 3 b] [1 2 3 b] -1
    0 a [1 2 3 b] b
    0 b a [1 2 3 b]
    0 b a [1 2 3 b] [1 2 3 b]
    0 b [1 2 3 b] a [1 2 3 b]
    0 b [1 2 3 b] a [1 2 3 b] -1
    0 b [1 2 3 a]
    0 b [1 2 3 a] [1 2 3 a]
    0 b [1 2 3 a] [1 2 3 a] reverse
    0 b [a 3 2 1] None
    0 b [a 3 2 1]
    0 b 1 2 3 a
    """
    return [
        create_instruction("BUILD_LIST", arg=n - 1),
        create_instruction("DUP_TOP"),
        create_instruction("LOAD_CONST", argval=-1),
        create_instruction("BINARY_SUBSCR"),
        create_instruction("ROT_THREE"),
        create_instruction("DUP_TOP"),
        create_instruction("ROT_THREE"),
        create_instruction("LOAD_CONST", argval=-1),
        create_instruction("STORE_SUBSCR"),
        create_instruction("DUP_TOP"),
        create_load_method("reverse"),
        *create_call_method(0),
        create_instruction("POP_TOP"),
        create_instruction("UNPACK_SEQUENCE", arg=n - 1),
    ]


def lnotab_writer(
    lineno: int, byteno: int = 0
) -> Tuple[List[int], Callable[[int, int], None]]:
    """
    Used to create typing.CodeType.co_lnotab
    See https://github.com/python/cpython/blob/main/Objects/lnotab_notes.txt
    This is the internal format of the line number table if Python < 3.10
    """
    assert sys.version_info < (3, 10)
    lnotab: List[int] = []

    def update(lineno_new, byteno_new):
        nonlocal byteno, lineno
        while byteno_new != byteno or lineno_new != lineno:
            byte_offset = max(0, min(byteno_new - byteno, 255))
            line_offset = max(-128, min(lineno_new - lineno, 127))
            assert byte_offset != 0 or line_offset != 0
            byteno += byte_offset
            lineno += line_offset
            lnotab.extend((byte_offset, line_offset & 0xFF))

    return lnotab, update


def linetable_310_writer(first_lineno):
    """
    Used to create typing.CodeType.co_linetable
    See https://github.com/python/cpython/blob/main/Objects/lnotab_notes.txt
    This is the internal format of the line number table for Python 3.10
    """
    assert sys.version_info >= (3, 10) and sys.version_info < (3, 11)
    linetable: List[int] = []
    lineno = first_lineno
    lineno_delta = 0
    byteno = 0

    def _update(byteno_delta, lineno_delta):
        while byteno_delta != 0 or lineno_delta != 0:
            byte_offset = max(0, min(byteno_delta, 254))
            line_offset = max(-127, min(lineno_delta, 127))
            assert byte_offset != 0 or line_offset != 0
            byteno_delta -= byte_offset
            lineno_delta -= line_offset
            linetable.extend((byte_offset, line_offset & 0xFF))

    def update(lineno_new, byteno_new):
        nonlocal lineno, lineno_delta, byteno
        byteno_delta = byteno_new - byteno
        byteno = byteno_new
        _update(byteno_delta, lineno_delta)
        lineno_delta = lineno_new - lineno
        lineno = lineno_new

    def end(total_bytes):
        _update(total_bytes - byteno, lineno_delta)

    return linetable, update, end


def encode_varint(n: int) -> List[int]:
    """
    6-bit chunk encoding of an unsigned integer
    See https://github.com/python/cpython/blob/3.11/Objects/locations.md
    """
    assert n >= 0
    b = [n & 63]
    n >>= 6
    while n > 0:
        b[-1] |= 64
        b.append(n & 63)
        n >>= 6
    return b


def linetable_311_writer(first_lineno: int):
    """
    Used to create typing.CodeType.co_linetable
    See https://github.com/python/cpython/blob/3.11/Objects/locations.md
    This is the internal format of the line number table for Python 3.11
    """
    assert sys.version_info >= (3, 11)
    linetable = []
    lineno = first_lineno

    def update(positions: "dis.Positions", inst_size):
        nonlocal lineno
        lineno_new = positions.lineno if positions else None

        def _update(delta, size):
            assert 0 < size <= 8
            # first byte - use 13 (no column info) is positions is
            # malformed, otherwise use 14 (long form)
            other_varints: Tuple[int, ...] = ()
            if (
                positions
                and positions.lineno is not None
                and positions.end_lineno is not None
                and positions.col_offset is not None
                and positions.end_col_offset is not None
            ):
                linetable.append(0b1_1110_000 + size - 1)
                # for whatever reason, column offset needs `+ 1`
                # https://github.com/python/cpython/blob/1931c2a438c50e6250725c84dff94fc760b9b951/Python/compile.c#L7603
                other_varints = (
                    positions.end_lineno - positions.lineno,
                    positions.col_offset + 1,
                    positions.end_col_offset + 1,
                )
            else:
                linetable.append(0b1_1101_000 + size - 1)
            # encode signed int
            if delta < 0:
                delta = ((-delta) << 1) | 1
            else:
                delta <<= 1
            # encode unsigned int
            linetable.extend(encode_varint(delta))
            for n in other_varints:
                linetable.extend(encode_varint(n))

        if lineno_new is None:
            lineno_delta = 0
        else:
            lineno_delta = lineno_new - lineno
            lineno = lineno_new
        while inst_size > 8:
            _update(lineno_delta, 8)
            inst_size -= 8
        _update(lineno_delta, inst_size)

    return linetable, update


@dataclasses.dataclass
class ExceptionTableEntry:
    start: int
    end: int
    target: int
    depth: int
    lasti: bool


def encode_exception_table_varint(n: int) -> List[int]:
    """
    Similar to `encode_varint`, but the 6-bit chunks are ordered in reverse.
    """
    assert n >= 0
    b = [n & 63]
    n >>= 6
    while n > 0:
        b.append(n & 63)
        n >>= 6
    b.reverse()
    for i in range(len(b) - 1):
        b[i] |= 64
    return b


def decode_exception_table_varint(bytes_iter: Iterator[int]) -> int:
    """
    Inverse of `encode_exception_table_varint`.
    """
    b = next(bytes_iter)
    val = b & 63
    while b & 64:
        val <<= 6
        b = next(bytes_iter)
        val |= b & 63
    return val


def check_exception_table(tab: List[ExceptionTableEntry]) -> None:
    """
    Verifies that a list of ExceptionTableEntries will make a well-formed
    jump table: entries are non-empty, sorted, and do not overlap.
    """
    for i in range(len(tab) - 1):
        assert (
            tab[i].start <= tab[i].end
            and tab[i].end < tab[i + 1].start
            and tab[i + 1].start <= tab[i + 1].end
        )


def parse_exception_table(exntab: bytes) -> List[ExceptionTableEntry]:
    """
    Parse the exception table according to
    https://github.com/python/cpython/blob/3.11/Objects/exception_handling_notes.txt
    """
    exntab_iter = iter(exntab)
    tab = []
    try:
        while True:
            start = decode_exception_table_varint(exntab_iter) * 2
            length = decode_exception_table_varint(exntab_iter) * 2
            end = start + length - 2
            target = decode_exception_table_varint(exntab_iter) * 2
            dl = decode_exception_table_varint(exntab_iter)
            depth = dl >> 1
            lasti = bool(dl & 1)
            tab.append(ExceptionTableEntry(start, end, target, depth, lasti))
    except StopIteration:
        check_exception_table(tab)
        return tab


def assemble_exception_table(tab: List[ExceptionTableEntry]) -> bytes:
    """
    Inverse of parse_exception_table - encodes list of exception
    table entries into bytes.
    """
    b = []
    for entry in tab:
        first_entry = encode_exception_table_varint(entry.start // 2)
        first_entry[0] |= 1 << 7
        b.extend(first_entry)
        length = entry.end - entry.start + 2
        b.extend(encode_exception_table_varint(length // 2))
        b.extend(encode_exception_table_varint(entry.target // 2))
        dl = (entry.depth << 1) + entry.lasti
        b.extend(encode_exception_table_varint(dl))
    return bytes(b)


def assemble(instructions: List[Instruction], firstlineno: int) -> Tuple[bytes, bytes]:
    """Do the opposite of dis.get_instructions()"""
    code: List[int] = []
    if sys.version_info >= (3, 11):
        lnotab, update_lineno = linetable_311_writer(firstlineno)
        num_ext = 0
        for i, inst in enumerate(instructions):
            if inst.opname == "EXTENDED_ARG":
                inst_size = 1
                num_ext += 1
                # copy positions from the actual instruction
                for j in (1, 2, 3):
                    if instructions[i + j].opname != "EXTENDED_ARG":
                        inst.positions = instructions[i + j].positions
                        break
            else:
                inst_size = instruction_size(inst) // 2 + num_ext
                num_ext = 0
            update_lineno(inst.positions, inst_size)
            num_ext = 0
            arg = inst.arg or 0
            code.extend((inst.opcode, arg & 0xFF))
            for _ in range(instruction_size(inst) // 2 - 1):
                code.extend((0, 0))
    else:
        if sys.version_info < (3, 10):
            lnotab, update_lineno = lnotab_writer(firstlineno)
        else:
            lnotab, update_lineno, end = linetable_310_writer(firstlineno)

        for inst in instructions:
            if inst.starts_line is not None:
                update_lineno(inst.starts_line, len(code))
            arg = inst.arg or 0
            code.extend((inst.opcode, arg & 0xFF))

        if sys.version_info >= (3, 10):
            end(len(code))

    return bytes(code), bytes(lnotab)


def _get_instruction_by_offset(offset_to_inst: Dict[int, Instruction], offset: int):
    """
    Get the instruction located at a given offset, accounting for EXTENDED_ARGs
    """
    for n in (0, 2, 4, 6):
        if offset_to_inst[offset + n].opcode != dis.EXTENDED_ARG:
            return offset_to_inst[offset + n]
    return None


def virtualize_jumps(instructions) -> None:
    """Replace jump targets with pointers to make editing easier"""
    jump_targets = {inst.offset: inst for inst in instructions}

    for inst in instructions:
        if inst.opcode in dis.hasjabs or inst.opcode in dis.hasjrel:
            inst.target = _get_instruction_by_offset(jump_targets, inst.argval)


_REL_JUMPS = set(dis.hasjrel)


def flip_jump_direction(instruction: Instruction) -> None:
    if sys.version_info < (3, 11):
        raise RuntimeError("Cannot flip jump direction in Python < 3.11")
    if "FORWARD" in instruction.opname:
        instruction.opname = instruction.opname.replace("FORWARD", "BACKWARD")
    elif "BACKWARD" in instruction.opname:
        instruction.opname = instruction.opname.replace("BACKWARD", "FORWARD")
    else:
        raise AttributeError("Instruction is not a forward or backward jump")
    instruction.opcode = dis.opmap[instruction.opname]
    assert instruction.opcode in _REL_JUMPS


def _get_instruction_front(instructions: List[Instruction], idx: int):
    """
    i.e. get the first EXTENDED_ARG instruction (if any) when targeting
    instructions[idx] with a jump.
    """
    target = instructions[idx]
    for offset in (1, 2, 3):
        if idx >= offset and instructions[idx - offset].opcode == dis.EXTENDED_ARG:
            target = instructions[idx - offset]
        else:
            break
    return target


def devirtualize_jumps(instructions):
    """Fill in args for virtualized jump target after instructions may have moved"""
    indexof = get_indexof(instructions)
    jumps = set(dis.hasjabs).union(set(dis.hasjrel))

    for inst in instructions:
        if inst.opcode in jumps:
            target = _get_instruction_front(instructions, indexof[inst.target])
            if inst.opcode in dis.hasjabs:
                if sys.version_info < (3, 10):
                    inst.arg = target.offset
                elif sys.version_info < (3, 11):
                    # `arg` is expected to be bytecode offset, whereas `offset` is byte offset.
                    # Divide since bytecode is 2 bytes large.
                    inst.arg = int(target.offset / 2)
                else:
                    raise RuntimeError("Python 3.11+ should not have absolute jumps")
            else:  # relative jump
                # byte offset between target and next instruction
                inst.arg = int(target.offset - inst.offset - instruction_size(inst))
                if inst.arg < 0:
                    if sys.version_info < (3, 11):
                        raise RuntimeError("Got negative jump offset for Python < 3.11")
                    inst.arg = -inst.arg
                    # forward jumps become backward
                    if "FORWARD" in inst.opname:
                        flip_jump_direction(inst)
                elif inst.arg > 0:
                    # backward jumps become forward
                    if sys.version_info >= (3, 11) and "BACKWARD" in inst.opname:
                        flip_jump_direction(inst)
                if sys.version_info >= (3, 10):
                    # see bytecode size comment in the absolute jump case above
                    inst.arg //= 2
            inst.argval = target.offset
            inst.argrepr = f"to {target.offset}"


def virtualize_exception_table(exn_tab_bytes: bytes, instructions: List[Instruction]):
    """Replace exception table entries with pointers to make editing easier"""
    exn_tab = parse_exception_table(exn_tab_bytes)
    offset_to_inst = {cast(int, inst.offset): inst for inst in instructions}
    offsets = sorted(offset_to_inst.keys())
    end_offset_idx = 0
    exn_tab_iter = iter(exn_tab)
    try:

        def step():
            nonlocal end_offset_idx
            entry = next(exn_tab_iter)
            # find rightmost offset <= entry.end, since entry.end may not be
            # an actual instruction, e.g. if the end instruction is LOAD_GLOBAL,
            # which takes more than 2 bytes, then entry.end points to the end
            # of the LOAD_GLOBAL instruction, not the beginning.
            while (
                end_offset_idx < len(offsets) and offsets[end_offset_idx] <= entry.end
            ):
                end_offset_idx += 1
            assert end_offset_idx > 0
            end_offset = offsets[end_offset_idx - 1]
            inst_entry = InstructionExnTabEntry(
                _get_instruction_by_offset(offset_to_inst, entry.start),
                _get_instruction_by_offset(offset_to_inst, end_offset),
                _get_instruction_by_offset(offset_to_inst, entry.target),
                entry.depth,
                entry.lasti,
            )
            return entry, inst_entry

        entry, inst_entry = step()
        for inst in instructions:
            while inst.offset > entry.end:
                entry, inst_entry = step()
            if inst.offset >= entry.start:
                inst.exn_tab_entry = copy.copy(inst_entry)
    except StopIteration:
        pass


def compute_exception_table(
    instructions: List[Instruction],
) -> List[ExceptionTableEntry]:
    """Compute exception table in list format from instructions with exn_tab_entries"""
    exn_dict: Dict[Tuple[int, int], Tuple[int, int, bool]] = {}
    indexof = get_indexof(instructions)

    for inst in instructions:
        if inst.exn_tab_entry:
            # account for prefixed EXTENDED_ARGS
            start = _get_instruction_front(
                instructions, indexof[inst.exn_tab_entry.start]
            ).offset
            # point to the last 2 bytes of the end instruction
            end = (
                cast(int, inst.exn_tab_entry.end.offset)
                + instruction_size(inst.exn_tab_entry.end)
                - 2
            )
            target = _get_instruction_front(
                instructions, indexof[inst.exn_tab_entry.target]
            ).offset
            key = (start, end)
            val = (target, inst.exn_tab_entry.depth, inst.exn_tab_entry.lasti)
            if key in exn_dict:
                assert exn_dict[key] == val
            exn_dict[key] = val

    # Dynamo may construct nested exception table entries for convenience,
    # but Python expects exception table entries to not overlap.
    # NOTE: below, "keys" refer to old instruction entries' starts and ends,
    # and "entries" refer to the generated exception table entries.

    # Sort keys by increasing start, then decreasing end
    keys_sorted = sorted(exn_dict.keys(), key=lambda t: (t[0], -t[1]))
    # smallest byte that the next exception table entry can start at
    nexti = 0
    # stack of current nested keys
    key_stack: List[Tuple[int, int]] = []
    exn_tab: List[ExceptionTableEntry] = []

    def pop():
        """
        Pop the key_stack and append an exception table entry if possible.
        """
        nonlocal nexti
        if key_stack:
            key = key_stack.pop()
            if nexti <= key[1]:
                exn_tab.append(
                    ExceptionTableEntry(max(key[0], nexti), key[1], *exn_dict[key])
                )
                nexti = key[1] + 2

    for key in keys_sorted:
        # pop keys that are no longer nested over the current key
        while key_stack and key_stack[-1][1] < key[0]:
            pop()
        if key_stack:
            # create an entry covering to the current key, if possible
            assert key_stack[-1][0] <= key[0] <= key[1] <= key_stack[-1][1]
            left = max(nexti, key_stack[-1][0])
            if left < key[0]:
                exn_tab.append(
                    ExceptionTableEntry(left, key[0] - 2, *exn_dict[key_stack[-1]])
                )
            nexti = key[0]
        key_stack.append(key)
    while key_stack:
        pop()
    check_exception_table(exn_tab)
    return exn_tab


def check_inst_exn_tab_entries_nested(
    tab: List[InstructionExnTabEntry], indexof
) -> None:
    """
    Checks `tab` is a properly sorted list of nested InstructionExnTabEntry's,
    i.e. no entries partially overlap.
    "Properly sorted" means entries are sorted by increasing starts, then
    decreasing ends.
    """
    entry_stack: List[Tuple[int, int]] = []
    for entry in tab:
        key = (indexof[entry.start], indexof[entry.end])
        while entry_stack and entry_stack[-1][1] < key[0]:
            entry_stack.pop()
        if entry_stack:
            assert entry_stack[-1][0] <= key[0] <= key[1] <= entry_stack[-1][1]
        entry_stack.append(key)


def propagate_inst_exn_table_entries(instructions: List[Instruction]) -> None:
    """
    Copies exception table entries to all instructions in an entry's range.
    Supports nested exception table entries.
    """
    indexof = get_indexof(instructions)
    entries: Dict[Tuple[int, int], InstructionExnTabEntry] = {}
    for inst in instructions:
        if inst.exn_tab_entry:
            key = (
                indexof[inst.exn_tab_entry.start],
                indexof[inst.exn_tab_entry.end],
            )
            if key in entries:
                assert inst.exn_tab_entry == entries[key]
            entries[key] = inst.exn_tab_entry
    sorted_entries = [
        entries[key] for key in sorted(entries.keys(), key=lambda t: (t[0], -t[1]))
    ]
    check_inst_exn_tab_entries_nested(sorted_entries, indexof)
    # Propagation of nested entries works since nested entries come later
    # in sorted order.
    for entry in sorted_entries:
        for i in range(indexof[entry.start], indexof[entry.end] + 1):
            instructions[i].exn_tab_entry = copy.copy(entry)


def check_inst_exn_tab_entries_valid(instructions: List[Instruction]):
    """
    Checks that exn_tab_entries of instructions are valid.
    An entry's start, end, and target must be in instructions.
    Instructions with an exn_tab_entry are located within
    the entry's start and end instructions.
    Instructions do not share exn_tab_entries.

    Implicitly checks for no duplicate instructions.
    """
    indexof = get_indexof(instructions)
    exn_tab_entry_set = set()
    for i, inst in enumerate(instructions):
        if inst.exn_tab_entry:
            assert sys.version_info >= (3, 11)
            assert id(inst.exn_tab_entry) not in exn_tab_entry_set
            exn_tab_entry_set.add(id(inst.exn_tab_entry))
            entry = inst.exn_tab_entry
            assert entry.start in indexof
            assert entry.end in indexof
            assert entry.target in indexof
            assert indexof[entry.start] <= i <= indexof[entry.end]


def strip_extended_args(instructions: List[Instruction]) -> None:
    instructions[:] = [i for i in instructions if i.opcode != dis.EXTENDED_ARG]


def remove_load_call_method(instructions: List[Instruction]) -> List[Instruction]:
    """LOAD_METHOD puts a NULL on the stack which causes issues, so remove it"""
    assert sys.version_info < (3, 11)
    rewrites = {"LOAD_METHOD": "LOAD_ATTR", "CALL_METHOD": "CALL_FUNCTION"}
    for inst in instructions:
        if inst.opname in rewrites:
            inst.opname = rewrites[inst.opname]
            inst.opcode = dis.opmap[inst.opname]
    return instructions


def remove_jump_if_none(instructions: List[Instruction]) -> None:
    new_insts = []
    for inst in instructions:
        new_insts.append(inst)
        if "_NONE" in inst.opname:
            is_op = create_instruction("IS_OP", arg=int("NOT" in inst.opname))
            is_op.argval = is_op.arg
            is_op.positions = inst.positions
            if sys.version_info < (3, 12):
                jump_op = create_instruction(
                    "POP_JUMP_FORWARD_IF_TRUE"
                    if "FORWARD" in inst.opname
                    else "POP_JUMP_BACKWARD_IF_TRUE",
                    target=inst.target,
                )
            else:
                jump_op = create_instruction("POP_JUMP_IF_TRUE", target=inst.target)
            jump_op.positions = inst.positions
            # update inst.exn_tab_entry.end if necessary
            if inst.exn_tab_entry and inst.exn_tab_entry.end is inst:
                inst.exn_tab_entry.end = jump_op
            # preserve exception table entries
            is_op.exn_tab_entry = copy.copy(inst.exn_tab_entry)
            jump_op.exn_tab_entry = copy.copy(inst.exn_tab_entry)
            # modify inst in-place to preserve jump target
            inst.opcode = dis.opmap["LOAD_CONST"]
            inst.opname = "LOAD_CONST"
            inst.arg = None
            inst.argval = None
            new_insts.extend([is_op, jump_op])
    instructions[:] = new_insts


def remove_binary_store_slice(instructions: List[Instruction]) -> None:
    new_insts = []
    for inst in instructions:
        new_insts.append(inst)
        if inst.opname in ("BINARY_SLICE", "STORE_SLICE"):
            # new instruction
            subscr_inst = create_instruction(inst.opname.replace("SLICE", "SUBSCR"))
            if inst.exn_tab_entry and inst.exn_tab_entry.end is inst:
                inst.exn_tab_entry.end = subscr_inst
            subscr_inst.exn_tab_entry = copy.copy(inst.exn_tab_entry)
            subscr_inst.positions = inst.positions
            # modify inst in-place to preserve jump target
            inst.opcode = dis.opmap["BUILD_SLICE"]
            inst.opname = "BUILD_SLICE"
            inst.arg = 2
            inst.argval = 2
            new_insts.append(subscr_inst)
    instructions[:] = new_insts


def explicit_super(code: types.CodeType, instructions: List[Instruction]) -> None:
    """convert super() with no args into explicit arg form"""
    cell_and_free = (code.co_cellvars or tuple()) + (code.co_freevars or tuple())
    if not len(code.co_varnames):
        # A function with no argument cannot contain a valid "super()" call
        return
    output = []
    for idx, inst in enumerate(instructions):
        output.append(inst)
        if inst.opname == "LOAD_GLOBAL" and inst.argval == "super":
            nexti = instructions[idx + 1]
            if nexti.arg == 0 and (
                (sys.version_info >= (3, 12) and nexti.opname == "CALL")
                or (
                    sys.version_info >= (3, 11)
                    and sys.version_info < (3, 12)
                    and nexti.opname == "PRECALL"
                )
                or (sys.version_info < (3, 11) and nexti.opname == "CALL_FUNCTION")
            ):
                assert "__class__" in cell_and_free
                output.append(create_instruction("LOAD_DEREF", argval="__class__"))
                first_var = code.co_varnames[0]
                if first_var in cell_and_free:
                    output.append(create_instruction("LOAD_DEREF", argval=first_var))
                else:
                    output.append(create_instruction("LOAD_FAST", argval=first_var))
                nexti.arg = 2
                nexti.argval = 2
                if nexti.opname == "PRECALL":
                    # also update the following CALL instruction
                    call_inst = instructions[idx + 2]
                    call_inst.arg = 2
                    call_inst.argval = 2

    instructions[:] = output


def fix_extended_args(instructions: List[Instruction]) -> int:
    """Fill in correct argvals for EXTENDED_ARG ops"""
    output: List[Instruction] = []

    def maybe_pop_n(n):
        for _ in range(n):
            if output and output[-1].opcode == dis.EXTENDED_ARG:
                output.pop()

    for inst in instructions:
        if inst.opcode == dis.EXTENDED_ARG:
            # Leave this instruction alone for now so we never shrink code
            inst.arg = 0
        elif inst.arg and inst.arg > 0xFFFFFF:
            maybe_pop_n(3)
            output.append(create_instruction("EXTENDED_ARG", arg=inst.arg >> 24))
            output.append(create_instruction("EXTENDED_ARG", arg=inst.arg >> 16))
            output.append(create_instruction("EXTENDED_ARG", arg=inst.arg >> 8))
        elif inst.arg and inst.arg > 0xFFFF:
            maybe_pop_n(2)
            output.append(create_instruction("EXTENDED_ARG", arg=inst.arg >> 16))
            output.append(create_instruction("EXTENDED_ARG", arg=inst.arg >> 8))
        elif inst.arg and inst.arg > 0xFF:
            maybe_pop_n(1)
            output.append(create_instruction("EXTENDED_ARG", arg=inst.arg >> 8))
        output.append(inst)

    added = len(output) - len(instructions)
    assert added >= 0
    instructions[:] = output
    return added


def instruction_size(inst) -> int:
    import torch

    if sys.version_info >= (3, 11):
        return 2 * (torch._C._dynamo.eval_frame.py_opcode_caches[inst.opcode] + 1)
    return 2


def check_offsets(instructions) -> None:
    offset = 0
    for inst in instructions:
        assert inst.offset == offset
        offset += instruction_size(inst)


def update_offsets(instructions) -> None:
    offset = 0
    for inst in instructions:
        inst.offset = offset
        offset += instruction_size(inst)


def debug_bytes(*args) -> str:
    index = range(max(map(len, args)))
    result = []
    for arg in (
        [index] + list(args) + [[int(a != b) for a, b in zip(args[-1], args[-2])]]
    ):
        result.append(" ".join(f"{x:03}" for x in arg))

    return "bytes mismatch\n" + "\n".join(result)


def debug_checks(code):
    """Make sure our assembler produces same bytes as we start with"""
    dode = transform_code_object(code, lambda x, y: None, safe=True)
    assert code.co_code == dode.co_code, debug_bytes(code.co_code, dode.co_code)
    assert code.co_lnotab == dode.co_lnotab, debug_bytes(code.co_lnotab, dode.co_lnotab)


HAS_LOCAL = set(dis.haslocal)
HAS_NAME = set(dis.hasname)
HAS_FREE = set(dis.hasfree)
HAS_CONST = set(dis.hasconst)


def get_const_index(code_options, val) -> int:
    for i, v in enumerate(code_options["co_consts"]):
        # NOTE: stronger comparison is required, since we have
        # examples where two values compare equal but have
        # different semantic meaning in some cases, e.g.
        # 0.0 == -0.0 but have different effects in torch.copysign.
        if val is v:
            return i
    code_options["co_consts"] += (val,)
    return len(code_options["co_consts"]) - 1


def fix_vars(instructions: List[Instruction], code_options, varname_from_oparg=None):
    # compute instruction arg from argval if arg is not provided
    names = {name: idx for idx, name in enumerate(code_options["co_names"])}

    def get_name_index(name) -> int:
        try:
            idx = names[name]
        except KeyError:
            # Add a missing item to co_names
            idx = names[name] = len(names)
            code_options["co_names"] = (*code_options["co_names"], name)
            assert len(code_options["co_names"]) == len(names)
        return idx

    if sys.version_info < (3, 11):
        assert varname_from_oparg is None
        varnames = {name: idx for idx, name in enumerate(code_options["co_varnames"])}
        freenames = {
            name: idx
            for idx, name in enumerate(
                code_options["co_cellvars"] + code_options["co_freevars"]
            )
        }
    else:
        assert callable(varname_from_oparg)
        allnames = {}
        for idx in itertools.count():
            try:
                name = varname_from_oparg(idx)
                allnames[name] = idx
            except IndexError:
                break
        varnames = {name: allnames[name] for name in code_options["co_varnames"]}
        freenames = {
            name: allnames[name]
            for name in code_options["co_cellvars"] + code_options["co_freevars"]
        }
    for i in range(len(instructions)):

        def should_compute_arg():
            # argval is prioritized over arg
            return instructions[i].argval is not _NotProvided

        if instructions[i].opname == "LOAD_GLOBAL":
            # 3.11 LOAD_GLOBAL requires both arg and argval - see create_instruction
            assert instructions[i].argval is not _NotProvided
            if sys.version_info >= (3, 11):
                assert instructions[i].arg is not None
                instructions[i].arg = (get_name_index(instructions[i].argval) << 1) + (
                    cast(int, instructions[i].arg) % 2
                )
            else:
                instructions[i].arg = get_name_index(instructions[i].argval)
        elif instructions[i].opname == "LOAD_ATTR":
            # 3.12 LOAD_ATTR requires both arg and argval, like LOAD_GLOBAL
            assert instructions[i].argval is not _NotProvided
            if sys.version_info >= (3, 12):
                assert instructions[i].arg is not None
                instructions[i].arg = (get_name_index(instructions[i].argval) << 1) + (
                    cast(int, instructions[i].arg) % 2
                )
            else:
                instructions[i].arg = get_name_index(instructions[i].argval)
        elif instructions[i].opname == "LOAD_SUPER_ATTR":
            assert instructions[i].arg is not None
            assert instructions[i].argval is not _NotProvided
            # Copy low bit, force second bit on for explicit super (the "+ 2")
            instructions[i].arg = (
                (get_name_index(instructions[i].argval) << 2)
                + (cast(int, instructions[i].arg) % 2)
                + 2
            )
        elif instructions[i].opcode in HAS_LOCAL:
            if should_compute_arg():
                instructions[i].arg = varnames[instructions[i].argval]
        elif instructions[i].opcode in HAS_NAME:
            if should_compute_arg():
                instructions[i].arg = get_name_index(instructions[i].argval)
        elif instructions[i].opcode in HAS_FREE:
            if should_compute_arg():
                instructions[i].arg = freenames[instructions[i].argval]
        elif instructions[i].opcode in HAS_CONST:
            # NOTE: only update argval if arg is not provided. This assumes
            # that any additions to co_consts are appended.
            if instructions[i].arg is None:
                # cannot use a dictionary since consts may not be hashable
                idx = get_const_index(code_options, instructions[i].argval)
                assert idx >= 0
                instructions[i].arg = idx


def clear_instruction_args(instructions):
    # Clear the instruction arg for instructions that have argvals.
    # Useful for using dis'd bytecode within generated bytecode.
    for inst in instructions:
        if (
            inst.argval is not _NotProvided
            and (
                inst.opcode in HAS_LOCAL
                or inst.opcode in HAS_NAME
                or inst.opcode in HAS_FREE
                or inst.opcode in HAS_CONST
            )
            and inst.opname not in ("LOAD_GLOBAL", "LOAD_ATTR", "LOAD_SUPER_ATTR")
        ):
            inst.arg = None


def get_code_keys() -> List[str]:
    # Python 3.11 changes to code keys are not fully documented.
    # See https://github.com/python/cpython/blob/3.11/Objects/clinic/codeobject.c.h#L24
    # for new format.
    keys = ["co_argcount"]
    keys.append("co_posonlyargcount")
    keys.extend(
        [
            "co_kwonlyargcount",
            "co_nlocals",
            "co_stacksize",
            "co_flags",
            "co_code",
            "co_consts",
            "co_names",
            "co_varnames",
            "co_filename",
            "co_name",
        ]
    )
    if sys.version_info >= (3, 11):
        keys.append("co_qualname")
    keys.append("co_firstlineno")
    if sys.version_info >= (3, 10):
        keys.append("co_linetable")
    else:
        keys.append("co_lnotab")
    if sys.version_info >= (3, 11):
        # not documented, but introduced in https://github.com/python/cpython/issues/84403
        keys.append("co_exceptiontable")
    keys.extend(
        [
            "co_freevars",
            "co_cellvars",
        ]
    )
    return keys


def transform_code_object(code, transformations, safe=False) -> types.CodeType:
    keys = get_code_keys()
    code_options = {k: getattr(code, k) for k in keys}
    assert len(code_options["co_varnames"]) == code_options["co_nlocals"]

    instructions = cleaned_instructions(code, safe)
    propagate_line_nums(instructions)

    transformations(instructions, code_options)
    return clean_and_assemble_instructions(instructions, keys, code_options)[1]


def clean_and_assemble_instructions(
    instructions: List[Instruction], keys: List[str], code_options: Dict[str, Any]
) -> Tuple[List[Instruction], types.CodeType]:
    # also implicitly checks for no duplicate instructions
    check_inst_exn_tab_entries_valid(instructions)

    code_options["co_nlocals"] = len(code_options["co_varnames"])
    varname_from_oparg = None
    if sys.version_info >= (3, 11):
        # temporary code object with updated names
        tmp_code = types.CodeType(*[code_options[k] for k in keys])
        varname_from_oparg = tmp_code._varname_from_oparg  # type: ignore[attr-defined]
    fix_vars(instructions, code_options, varname_from_oparg=varname_from_oparg)

    dirty = True
    while dirty:
        update_offsets(instructions)
        devirtualize_jumps(instructions)
        # this pass might change offsets, if so we need to try again
        dirty = bool(fix_extended_args(instructions))

    remove_extra_line_nums(instructions)
    bytecode, lnotab = assemble(instructions, code_options["co_firstlineno"])
    if sys.version_info < (3, 10):
        code_options["co_lnotab"] = lnotab
    else:
        code_options["co_linetable"] = lnotab

    code_options["co_code"] = bytecode
    code_options["co_stacksize"] = stacksize_analysis(instructions)
    assert set(keys) - {"co_posonlyargcount"} == set(code_options.keys()) - {
        "co_posonlyargcount"
    }
    if sys.version_info >= (3, 11):
        code_options["co_exceptiontable"] = assemble_exception_table(
            compute_exception_table(instructions)
        )

    return instructions, types.CodeType(*[code_options[k] for k in keys])


def populate_kw_names_argval(instructions, consts):
    for inst in instructions:
        if inst.opname == "KW_NAMES":
            inst.argval = consts[inst.arg]


def cleaned_instructions(code, safe=False) -> List[Instruction]:
    instructions = list(map(convert_instruction, dis.get_instructions(code)))
    check_offsets(instructions)
    if sys.version_info >= (3, 11):
        populate_kw_names_argval(instructions, code.co_consts)
        virtualize_exception_table(code.co_exceptiontable, instructions)
    virtualize_jumps(instructions)
    strip_extended_args(instructions)
    if not safe:
        if sys.version_info < (3, 11):
            remove_load_call_method(instructions)
        if sys.version_info < (3, 12):
            explicit_super(code, instructions)
    if sys.version_info >= (3, 11):
        remove_jump_if_none(instructions)
        if sys.version_info >= (3, 12):
            remove_binary_store_slice(instructions)
        update_offsets(instructions)
        devirtualize_jumps(instructions)
    return instructions


_unique_id_counter = itertools.count()


def unique_id(name) -> str:
    return f"{name}_{next(_unique_id_counter)}"


def is_generator(code: types.CodeType) -> bool:
    co_generator = 0x20
    return (code.co_flags & co_generator) > 0


def bytecode_from_template(fn, varname_map=None, noreturn=True, noprefix=True):
    """Generates bytecode from a template function `fn` for use in
    dynamo bytecode generation.

    For example, we can generate Python-version-independent bytecode
    for looping through a dictionary and copying the values to a new dictionary.

    def template(d1, d2):
        for k, v in d1.items():
            d2[k] = v


    or a try block:

    def template():
        try:
            dummy1
        except:
            dummy2
            raise
        dummy3

    Args:
        fn: a function template to generate bytecode from
        varname_map: a mapping of `fn`'s varnames to new names. This
            map will be applied to the generated bytecode's varnames.
            For example, local variables in `fn` can be replaced with
            new names that are generated by `OutputGraph.new_var`.
        noreturn: remove all RETURN_* bytecodes and replace them with a jump
            to the end of the bytecode.
        noprefix: remove prefix bytecodes (all bytecode before the first RESUME, inclusive).
    """
    insts = cleaned_instructions(fn.__code__)
    clear_instruction_args(insts)

    if noprefix:
        for i, inst in enumerate(insts):
            if inst.opname == "RESUME":
                insts = insts[i + 1 :]
                break

    for inst in insts:
        # If we don't reset starts_line, then the generated
        # bytecode's line number will be based on fn's.
        inst.starts_line = None
        if varname_map and inst.argval in varname_map:
            inst.argval = varname_map[inst.argval]

    if noreturn:
        if sys.version_info >= (3, 12):
            # replace RETURN_CONST with LOAD_CONST RETURN_VALUE
            new_insts = []
            for inst in insts:
                if inst.opname == "RETURN_CONST":
                    inst.opcode = dis.opmap["LOAD_CONST"]
                    inst.opname = "LOAD_CONST"
                    new_insts.append(inst)
                    # no need to propagate target/exn table
                    new_insts.append(create_instruction("RETURN_VALUE"))
                else:
                    new_insts.append(inst)
            insts = new_insts

        returns = []
        for inst in insts:
            if inst.opname == "RETURN_VALUE":
                returns.append(inst)

        if len(returns) == 1 and returns[0] is insts[-1]:
            # only 1 return at the end - just pop it
            insts.pop(-1)
        elif len(returns) > 0:
            # create jump target - if the last inst is a return,
            # we can replace it with a NOP and make that the jump target.
            if insts[-1] is returns[-1]:
                insts[-1].opname = "NOP"
                insts[-1].opcode = dis.opmap["NOP"]
                insts[-1].arg = None
                insts[-1].argval = _NotProvided
                returns.pop(-1)
            else:
                insts.append(create_instruction("NOP"))

            # replace returns with jumps
            for inst in returns:
                # don't replace inst with new instruction
                # due to targetting/exn table/etc.
                jump_inst = create_jump_absolute(insts[-1])
                inst.opname = jump_inst.opname
                inst.opcode = jump_inst.opcode
                inst.arg = jump_inst.arg
                inst.argval = jump_inst.argval
                inst.target = jump_inst.target

    return insts
