import dataclasses
import dis
import itertools
import sys
import types
from typing import Any, List, Optional

from .bytecode_analysis import (
    propagate_line_nums,
    remove_extra_line_nums,
    stacksize_analysis,
)


@dataclasses.dataclass
class Instruction:
    """A mutable version of dis.Instruction"""

    opcode: int
    opname: str
    arg: int
    argval: Any
    offset: Optional[int] = None
    starts_line: Optional[int] = None
    is_jump_target: bool = False
    # extra fields to make modification easier:
    target: Optional["Instruction"] = None

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return id(self) == id(other)


def convert_instruction(i: dis.Instruction):
    return Instruction(
        i.opcode,
        i.opname,
        i.arg,
        i.argval,
        i.offset,
        i.starts_line,
        i.is_jump_target,
    )


class _NotProvided:
    pass


def create_instruction(name, arg=None, argval=_NotProvided, target=None):
    if argval is _NotProvided:
        argval = arg
    return Instruction(
        opcode=dis.opmap[name], opname=name, arg=arg, argval=argval, target=target
    )


def lnotab_writer(lineno, byteno=0):
    """
    Used to create typing.CodeType.co_lnotab
    See https://github.com/python/cpython/blob/main/Objects/lnotab_notes.txt
    This is the internal format of the line number table if Python < 3.10
    """
    assert sys.version_info < (3, 10)
    lnotab = []

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


def linetable_writer(first_lineno):
    """
    Used to create typing.CodeType.co_linetable
    See https://github.com/python/cpython/blob/main/Objects/lnotab_notes.txt
    This is the internal format of the line number table if Python >= 3.10
    """
    assert sys.version_info >= (3, 10)
    linetable = []
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


def assemble(instructions: List[dis.Instruction], firstlineno):
    """Do the opposite of dis.get_instructions()"""
    code = []
    if sys.version_info < (3, 10):
        lnotab, update_lineno = lnotab_writer(firstlineno)
    else:
        lnotab, update_lineno, end = linetable_writer(firstlineno)

    for inst in instructions:
        if inst.starts_line is not None:
            update_lineno(inst.starts_line, len(code))
        arg = inst.arg or 0
        code.extend((inst.opcode, arg & 0xFF))

    if sys.version_info >= (3, 10):
        end(len(code))

    return bytes(code), bytes(lnotab)


def virtualize_jumps(instructions):
    """Replace jump targets with pointers to make editing easier"""
    jump_targets = {inst.offset: inst for inst in instructions}

    for inst in instructions:
        if inst.opcode in dis.hasjabs or inst.opcode in dis.hasjrel:
            for offset in (0, 2, 4, 6):
                if jump_targets[inst.argval + offset].opcode != dis.EXTENDED_ARG:
                    inst.target = jump_targets[inst.argval + offset]
                    break


def devirtualize_jumps(instructions):
    """Fill in args for virtualized jump target after instructions may have moved"""
    indexof = {id(inst): i for i, inst, in enumerate(instructions)}
    jumps = set(dis.hasjabs).union(set(dis.hasjrel))

    for inst in instructions:
        if inst.opcode in jumps:
            target = inst.target
            target_index = indexof[id(target)]
            for offset in (1, 2, 3):
                if (
                    target_index >= offset
                    and instructions[target_index - offset].opcode == dis.EXTENDED_ARG
                ):
                    target = instructions[target_index - offset]
                else:
                    break

            if inst.opcode in dis.hasjabs:
                if sys.version_info < (3, 10):
                    inst.arg = target.offset
                else:
                    # arg is offset of the instruction line rather than the bytecode
                    # for all jabs/jrel since python 3.10
                    inst.arg = int(target.offset / 2)
            else:  # relative jump
                if sys.version_info < (3, 10):
                    inst.arg = target.offset - inst.offset - instruction_size(inst)
                else:
                    inst.arg = int(
                        (target.offset - inst.offset - instruction_size(inst)) / 2
                    )
            inst.argval = target.offset
            inst.argrepr = f"to {target.offset}"


def strip_extended_args(instructions: List[Instruction]):
    instructions[:] = [i for i in instructions if i.opcode != dis.EXTENDED_ARG]


def remove_load_call_method(instructions: List[Instruction]):
    """LOAD_METHOD puts a NULL on the stack which causes issues, so remove it"""
    rewrites = {"LOAD_METHOD": "LOAD_ATTR", "CALL_METHOD": "CALL_FUNCTION"}
    for inst in instructions:
        if inst.opname in rewrites:
            inst.opname = rewrites[inst.opname]
            inst.opcode = dis.opmap[inst.opname]
    return instructions


def explicit_super(code: types.CodeType, instructions: List[Instruction]):
    """convert super() with no args into explict arg form"""
    cell_and_free = (code.co_cellvars or tuple()) + (code.co_freevars or tuple())
    output = []
    for idx, inst in enumerate(instructions):
        output.append(inst)
        if inst.opname == "LOAD_GLOBAL" and inst.argval == "super":
            nexti = instructions[idx + 1]
            if nexti.opname == "CALL_FUNCTION" and nexti.arg == 0:
                assert "__class__" in cell_and_free
                output.append(
                    create_instruction(
                        "LOAD_DEREF", cell_and_free.index("__class__"), "__class__"
                    )
                )
                first_var = code.co_varnames[0]
                if first_var in cell_and_free:
                    output.append(
                        create_instruction(
                            "LOAD_DEREF", cell_and_free.index(first_var), first_var
                        )
                    )
                else:
                    output.append(create_instruction("LOAD_FAST", 0, first_var))
                nexti.arg = 2
                nexti.argval = 2

    instructions[:] = output


def fix_extended_args(instructions: List[Instruction]):
    """Fill in correct argvals for EXTENDED_ARG ops"""
    output = []

    def maybe_pop_n(n):
        for _ in range(n):
            if output and output[-1].opcode == dis.EXTENDED_ARG:
                output.pop()

    for i, inst in enumerate(instructions):
        if inst.opcode == dis.EXTENDED_ARG:
            # Leave this instruction alone for now so we never shrink code
            inst.arg = 0
        elif inst.arg and inst.arg > 0xFFFFFF:
            maybe_pop_n(3)
            output.append(create_instruction("EXTENDED_ARG", inst.arg >> 24))
            output.append(create_instruction("EXTENDED_ARG", inst.arg >> 16))
            output.append(create_instruction("EXTENDED_ARG", inst.arg >> 8))
        elif inst.arg and inst.arg > 0xFFFF:
            maybe_pop_n(2)
            output.append(create_instruction("EXTENDED_ARG", inst.arg >> 16))
            output.append(create_instruction("EXTENDED_ARG", inst.arg >> 8))
        elif inst.arg and inst.arg > 0xFF:
            maybe_pop_n(1)
            output.append(create_instruction("EXTENDED_ARG", inst.arg >> 8))
        output.append(inst)

    added = len(output) - len(instructions)
    assert added >= 0
    instructions[:] = output
    return added


def instruction_size(inst):
    return 2


def check_offsets(instructions):
    offset = 0
    for inst in instructions:
        assert inst.offset == offset
        offset += instruction_size(inst)


def update_offsets(instructions):
    offset = 0
    for inst in instructions:
        inst.offset = offset
        offset += instruction_size(inst)


def debug_bytes(*args):
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


def fix_vars(instructions: List[Instruction], code_options):
    varnames = {name: idx for idx, name in enumerate(code_options["co_varnames"])}
    names = {name: idx for idx, name in enumerate(code_options["co_names"])}
    for i in range(len(instructions)):
        if instructions[i].opcode in HAS_LOCAL:
            instructions[i].arg = varnames[instructions[i].argval]
        elif instructions[i].opcode in HAS_NAME:
            instructions[i].arg = names[instructions[i].argval]


def transform_code_object(code, transformations, safe=False):
    keys = [
        "co_argcount",
        "co_posonlyargcount",  # python 3.8+
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
        "co_firstlineno",
        "co_lnotab",  # changed to "co_linetable" if python 3.10+
        "co_freevars",
        "co_cellvars",
    ]
    if sys.version_info < (3, 8):
        keys.pop(1)
    if sys.version_info >= (3, 10):
        keys = list(map(lambda x: x.replace("co_lnotab", "co_linetable"), keys))
    code_options = {k: getattr(code, k) for k in keys}
    assert len(code_options["co_varnames"]) == code_options["co_nlocals"]

    instructions = cleaned_instructions(code, safe)
    propagate_line_nums(instructions)

    transformations(instructions, code_options)

    fix_vars(instructions, code_options)

    dirty = True
    while dirty:
        update_offsets(instructions)
        devirtualize_jumps(instructions)
        # this pass might change offsets, if so we need to try again
        dirty = fix_extended_args(instructions)

    remove_extra_line_nums(instructions)
    bytecode, lnotab = assemble(instructions, code_options["co_firstlineno"])
    if sys.version_info < (3, 10):
        code_options["co_lnotab"] = lnotab
    else:
        code_options["co_linetable"] = lnotab

    code_options["co_code"] = bytecode
    code_options["co_nlocals"] = len(code_options["co_varnames"])
    code_options["co_stacksize"] = stacksize_analysis(instructions)
    assert set(keys) - {"co_posonlyargcount"} == set(code_options.keys()) - {
        "co_posonlyargcount"
    }
    return types.CodeType(*[code_options[k] for k in keys])


def cleaned_instructions(code, safe=False):
    instructions = list(map(convert_instruction, dis.get_instructions(code)))
    check_offsets(instructions)
    virtualize_jumps(instructions)
    strip_extended_args(instructions)
    if not safe:
        remove_load_call_method(instructions)
        explicit_super(code, instructions)
    return instructions


_unique_id_counter = itertools.count()


def unique_id(name):
    return f"{name}_{next(_unique_id_counter)}"


def is_generator(code: types.CodeType):
    co_generator = 0x20
    return (code.co_flags & co_generator) > 0
