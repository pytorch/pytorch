import copy
import dataclasses
import sys
import types
from typing import Any, Dict, List, Optional, Tuple

from .bytecode_transformation import (
    create_call_function,
    create_call_method,
    create_dup_top,
    create_instruction,
    create_jump_absolute,
    Instruction,
    InstructionExnTabEntry,
    transform_code_object,
    unique_id,
)
from .utils import ExactWeakKeyDictionary

# taken from code.h in cpython
CO_OPTIMIZED = 0x0001
CO_NEWLOCALS = 0x0002
CO_VARARGS = 0x0004
CO_VARKEYWORDS = 0x0008
CO_NESTED = 0x0010
CO_GENERATOR = 0x0020
CO_NOFREE = 0x0040
CO_COROUTINE = 0x0080
CO_ITERABLE_COROUTINE = 0x0100
CO_ASYNC_GENERATOR = 0x0200


@dataclasses.dataclass(frozen=True)
class ReenterWith:
    stack_index: int = None
    target_values: Optional[Tuple] = None

    # If we do not want to destroy the stack, we can do the same thing as a
    # `SETUP_WITH` block, only that we store the context manager in a local_symbol
    def try_except(self, code_options, cleanup: List[Instruction]):
        """
        Codegen based off of:
        load args
        enter context
        try:
            (rest)
        finally:
            exit context
        """
        load_args = []
        if self.target_values:
            load_args = [
                create_instruction("LOAD_CONST", argval=val)
                for val in self.target_values
            ]
        ctx_name = unique_id(f"___context_manager_{self.stack_index}")
        if ctx_name not in code_options["co_varnames"]:
            code_options["co_varnames"] += (ctx_name,)
        for name in ["__enter__", "__exit__"]:
            if name not in code_options["co_names"]:
                code_options["co_names"] += (name,)

        except_jump_target = create_instruction(
            "NOP" if sys.version_info < (3, 11) else "PUSH_EXC_INFO"
        )
        cleanup_complete_jump_target = create_instruction("NOP")

        setup_finally = [
            *load_args,
            *create_call_function(len(load_args), True),
            create_instruction("STORE_FAST", argval=ctx_name),
            create_instruction("LOAD_FAST", argval=ctx_name),
            create_instruction("LOAD_METHOD", argval="__enter__"),
            *create_call_method(0),
            create_instruction("POP_TOP"),
        ]

        if sys.version_info < (3, 11):
            setup_finally.append(
                create_instruction("SETUP_FINALLY", target=except_jump_target)
            )
        else:
            exn_tab_begin = create_instruction("NOP")
            exn_tab_end = create_instruction("NOP")
            exn_tab_begin.exn_tab_entry = InstructionExnTabEntry(
                exn_tab_begin,
                exn_tab_end,
                except_jump_target,
                self.stack_index + 1,
                False,
            )
            setup_finally.append(exn_tab_begin)

        def create_reset():
            return [
                create_instruction("LOAD_FAST", argval=ctx_name),
                create_instruction("LOAD_METHOD", argval="__exit__"),
                create_instruction("LOAD_CONST", argval=None),
                create_dup_top(),
                create_dup_top(),
                *create_call_method(3),
                create_instruction("POP_TOP"),
            ]

        if sys.version_info < (3, 9):
            epilogue = [
                create_instruction("POP_BLOCK"),
                create_instruction("BEGIN_FINALLY"),
                except_jump_target,
                *create_reset(),
                create_instruction("END_FINALLY"),
            ]
        elif sys.version_info < (3, 11):
            epilogue = [
                create_instruction("POP_BLOCK"),
                *create_reset(),
                create_instruction("JUMP_FORWARD", target=cleanup_complete_jump_target),
                except_jump_target,
                *create_reset(),
                create_instruction("RERAISE"),
                cleanup_complete_jump_target,
            ]
        else:
            finally_exn_tab_end = create_instruction("RERAISE", arg=0)
            finally_exn_tab_target = create_instruction("COPY", arg=3)
            except_jump_target.exn_tab_entry = InstructionExnTabEntry(
                except_jump_target,
                finally_exn_tab_end,
                finally_exn_tab_target,
                self.stack_index + 2,
                True,
            )
            epilogue = [
                exn_tab_end,
                *create_reset(),
                create_instruction("JUMP_FORWARD", target=cleanup_complete_jump_target),
                except_jump_target,  # PUSH_EXC_INFO
                *create_reset(),
                finally_exn_tab_end,  # RERAISE 0
                finally_exn_tab_target,  # COPY 3
                create_instruction("POP_EXCEPT"),
                create_instruction("RERAISE", arg=1),
                cleanup_complete_jump_target,
            ]

        cleanup[:] = epilogue + cleanup
        return setup_finally

    def __call__(self, code_options, cleanup):
        """
        Codegen based off of:
        with ctx(args):
            (rest)
        """
        load_args = []
        if self.target_values:
            load_args = [
                create_instruction("LOAD_CONST", argval=val)
                for val in self.target_values
            ]
        if sys.version_info < (3, 9):
            with_cleanup_start = create_instruction("WITH_CLEANUP_START")
            begin_finally = create_instruction("BEGIN_FINALLY")
            cleanup[:] = [
                create_instruction("POP_BLOCK"),
                begin_finally,
                with_cleanup_start,
                create_instruction("WITH_CLEANUP_FINISH"),
                create_instruction("END_FINALLY"),
            ] + cleanup

            return [
                *load_args,
                create_instruction("CALL_FUNCTION", arg=len(load_args)),
                create_instruction("SETUP_WITH", target=with_cleanup_start),
                create_instruction("POP_TOP"),
            ], None
        elif sys.version_info < (3, 11):
            with_except_start = create_instruction("WITH_EXCEPT_START")
            pop_top_after_with_except_start = create_instruction("POP_TOP")

            cleanup_complete_jump_target = create_instruction("NOP")

            cleanup[:] = [
                create_instruction("POP_BLOCK"),
                create_instruction("LOAD_CONST", argval=None),
                create_instruction("DUP_TOP"),
                create_instruction("DUP_TOP"),
                create_instruction("CALL_FUNCTION", arg=3),
                create_instruction("POP_TOP"),
                create_instruction("JUMP_FORWARD", target=cleanup_complete_jump_target),
                with_except_start,
                create_instruction(
                    "POP_JUMP_IF_TRUE", target=pop_top_after_with_except_start
                ),
                create_instruction("RERAISE"),
                pop_top_after_with_except_start,
                create_instruction("POP_TOP"),
                create_instruction("POP_TOP"),
                create_instruction("POP_EXCEPT"),
                create_instruction("POP_TOP"),
                cleanup_complete_jump_target,
            ] + cleanup

            return [
                *load_args,
                create_instruction("CALL_FUNCTION", arg=len(load_args)),
                create_instruction("SETUP_WITH", target=with_except_start),
                create_instruction("POP_TOP"),
            ], None
        else:
            pop_top_after_with_except_start = create_instruction("POP_TOP")
            cleanup_complete_jump_target = create_instruction("NOP")

            def create_load_none():
                return create_instruction("LOAD_CONST", argval=None)

            exn_tab_1_begin = create_instruction("POP_TOP")
            exn_tab_1_end = create_instruction("NOP")
            exn_tab_1_target = create_instruction("PUSH_EXC_INFO")
            exn_tab_2_end = create_instruction("RERAISE", arg=2)
            exn_tab_2_target = create_instruction("COPY", arg=3)

            exn_tab_1_begin.exn_tab_entry = InstructionExnTabEntry(
                exn_tab_1_begin,
                exn_tab_1_end,
                exn_tab_1_target,
                self.stack_index + 1,
                True,
            )
            exn_tab_1_target.exn_tab_entry = InstructionExnTabEntry(
                exn_tab_1_target,
                exn_tab_2_end,
                exn_tab_2_target,
                self.stack_index + 3,
                True,
            )
            pop_top_after_with_except_start.exn_tab_entry = InstructionExnTabEntry(
                pop_top_after_with_except_start,
                pop_top_after_with_except_start,
                exn_tab_2_target,
                self.stack_index + 3,
                True,
            )

            cleanup[:] = [
                exn_tab_1_end,
                create_load_none(),
                create_load_none(),
                create_load_none(),
                *create_call_function(2, False),
                create_instruction("POP_TOP"),
                create_instruction("JUMP_FORWARD", target=cleanup_complete_jump_target),
                exn_tab_1_target,  # PUSH_EXC_INFO
                create_instruction("WITH_EXCEPT_START"),
                create_instruction(
                    "POP_JUMP_FORWARD_IF_TRUE",
                    target=pop_top_after_with_except_start,
                ),
                exn_tab_2_end,  # RERAISE 2
                exn_tab_2_target,  # COPY 3
                create_instruction("POP_EXCEPT"),
                create_instruction("RERAISE", arg=1),
                pop_top_after_with_except_start,
                create_instruction("POP_EXCEPT"),
                create_instruction("POP_TOP"),
                create_instruction("POP_TOP"),
                cleanup_complete_jump_target,
            ] + cleanup

            return [
                *load_args,
                *create_call_function(len(load_args), True),
                create_instruction("BEFORE_WITH"),
                exn_tab_1_begin,  # POP_TOP
            ], exn_tab_1_target


@dataclasses.dataclass
class ResumeFunctionMetadata:
    code: types.CodeType
    instructions: List[Instruction] = None
    # Python 3.11+ fields
    # NOTE: Python 3.11 removed blocks, but for our purposes, a "block" consists
    # of instructions of all exception table entries that have the same target.

    # map from PUSH_EXC_INFO's in the prefix to original block target offset
    prefix_block_target_offset_remap: List[int] = None
    # map from new block target offsets to original block target offsets
    block_target_offset_remap: Dict[int, int] = None


def _filter_iter(l1, l2, cond):
    """
    Two-pointer conditional filter.
    e.g. _filter_iter(insts, sorted_offsets, lambda i, o: i.offset == o)
    returns the instructions with offsets in sorted_offsets
    """
    it = iter(l2)
    res = []
    try:
        cur = next(it)
        for val in l1:
            if cond(val, cur):
                res.append(val)
                cur = next(it)
    except StopIteration:
        pass
    return res


class ContinueExecutionCache:
    cache = ExactWeakKeyDictionary()
    generated_code_metadata = ExactWeakKeyDictionary()

    @classmethod
    def lookup(cls, code, lineno, *key):
        if code not in cls.cache:
            cls.cache[code] = dict()
        key = tuple(key)
        if key not in cls.cache[code]:
            cls.cache[code][key] = cls.generate(code, lineno, *key)
        return cls.cache[code][key]

    @classmethod
    def generate(
        cls,
        code,
        lineno,
        offset: int,
        setup_fn_target_offsets: Tuple[int],  # only used in Python 3.11+
        nstack: int,
        argnames: Tuple[str],
        setup_fns: Tuple[ReenterWith],
        null_idxes: Tuple[int],
    ):
        assert offset is not None
        assert not (
            code.co_flags
            & (CO_GENERATOR | CO_COROUTINE | CO_ITERABLE_COROUTINE | CO_ASYNC_GENERATOR)
        )
        assert code.co_flags & CO_OPTIMIZED
        if code in ContinueExecutionCache.generated_code_metadata:
            return cls.generate_based_on_original_code_object(
                code,
                lineno,
                offset,
                setup_fn_target_offsets,
                nstack,
                argnames,
                setup_fns,
                null_idxes,
            )

        is_py311_plus = sys.version_info >= (3, 11)
        meta = ResumeFunctionMetadata(code)
        if is_py311_plus:
            meta.prefix_block_target_offset_remap = []

        def update(instructions: List[Instruction], code_options: Dict[str, Any]):
            meta.instructions = copy.deepcopy(instructions)

            args = [f"___stack{i}" for i in range(nstack)]
            args.extend(v for v in argnames if v not in args)
            freevars = tuple(code_options["co_cellvars"] or []) + tuple(
                code_options["co_freevars"] or []
            )
            code_options["co_name"] = f"<resume in {code_options['co_name']}>"
            if is_py311_plus:
                code_options[
                    "co_qualname"
                ] = f"<resume in {code_options['co_qualname']}>"
            code_options["co_firstlineno"] = lineno
            code_options["co_cellvars"] = tuple()
            code_options["co_freevars"] = freevars
            code_options["co_argcount"] = len(args)
            code_options["co_posonlyargcount"] = 0
            code_options["co_kwonlyargcount"] = 0
            code_options["co_varnames"] = tuple(
                args + [v for v in code_options["co_varnames"] if v not in args]
            )
            code_options["co_flags"] = code_options["co_flags"] & ~(
                CO_VARARGS | CO_VARKEYWORDS
            )
            target = next(i for i in instructions if i.offset == offset)

            prefix = []
            if is_py311_plus:
                if freevars:
                    prefix.append(
                        create_instruction("COPY_FREE_VARS", arg=len(freevars))
                    )
                prefix.append(create_instruction("RESUME", arg=0))

            cleanup = []
            hooks = {fn.stack_index: fn for fn in setup_fns}
            hook_target_offsets = {
                fn.stack_index: setup_fn_target_offsets[i]
                for i, fn in enumerate(setup_fns)
            }
            offset_to_inst = {inst.offset: inst for inst in instructions}
            # map old hook targets to new targets generated by the hook
            old_hook_target_remap = {}
            null_idxes_i = 0
            for i in range(nstack):
                while (
                    null_idxes_i < len(null_idxes)
                    and null_idxes[null_idxes_i] == i + null_idxes_i
                ):
                    prefix.append(create_instruction("PUSH_NULL"))
                    null_idxes_i += 1
                prefix.append(create_instruction("LOAD_FAST", argval=f"___stack{i}"))
                if i in hooks:
                    hook = hooks.pop(i)
                    hook_insts, exn_target = hook(code_options, cleanup)
                    prefix.extend(hook_insts)
                    if is_py311_plus:
                        hook_target_offset = hook_target_offsets.pop(i)
                        old_hook_target = offset_to_inst[hook_target_offset]
                        meta.prefix_block_target_offset_remap.append(hook_target_offset)
                        old_hook_target_remap[old_hook_target] = exn_target
            if is_py311_plus:
                # reverse the mapping since targets of later/nested contexts are inserted
                # into the mapping later, but show up earlier in the prefix.
                meta.prefix_block_target_offset_remap = list(
                    reversed(meta.prefix_block_target_offset_remap)
                )

            assert not hooks

            prefix.append(create_jump_absolute(target))

            # because the line number table monotonically increases from co_firstlineno
            # remove starts_line for any instructions before the graph break instruction
            # this will ensure the instructions after the break have the correct line numbers
            for inst in instructions:
                if inst.offset == target.offset:
                    break
                inst.starts_line = None

            if cleanup:
                prefix.extend(cleanup)
                prefix.extend(cls.unreachable_codes(code_options))

            # remap original instructions' exception table entries
            if old_hook_target_remap:
                assert is_py311_plus
                for inst in instructions:
                    if (
                        inst.exn_tab_entry
                        and inst.exn_tab_entry.target in old_hook_target_remap
                    ):
                        inst.exn_tab_entry.target = old_hook_target_remap[
                            inst.exn_tab_entry.target
                        ]

            # TODO(jansel): add dead code elimination here
            instructions[:] = prefix + instructions

        new_code = transform_code_object(code, update)
        ContinueExecutionCache.generated_code_metadata[new_code] = meta
        return new_code

    @staticmethod
    def unreachable_codes(code_options):
        """Codegen a `raise None` to make analysis work for unreachable code"""
        return [
            create_instruction("LOAD_CONST", argval=None),
            create_instruction("RAISE_VARARGS", arg=1),
        ]

    @classmethod
    def generate_based_on_original_code_object(
        cls, code, lineno, offset: int, setup_fn_target_offsets: Tuple[int], *args
    ):
        """
        This handles the case of generating a resume into code generated
        to resume something else.  We want to always generate starting
        from the original code object so that if control flow paths
        converge we only generated 1 resume function (rather than 2^n
        resume functions).
        """

        meta: ResumeFunctionMetadata = ContinueExecutionCache.generated_code_metadata[
            code
        ]
        new_offset = None

        def find_new_offset(
            instructions: List[Instruction], code_options: Dict[str, Any]
        ):
            nonlocal new_offset
            (target,) = [i for i in instructions if i.offset == offset]
            # match the functions starting at the last instruction as we have added a prefix
            (new_target,) = [
                i2
                for i1, i2 in zip(reversed(instructions), reversed(meta.instructions))
                if i1 is target
            ]
            assert target.opcode == new_target.opcode
            new_offset = new_target.offset

        transform_code_object(code, find_new_offset)

        if sys.version_info >= (3, 11):
            # setup_fn_target_offsets currently contains the target offset of
            # each setup_fn, based on `code`. When we codegen the resume function
            # based on the original code object, `meta.code`, the offsets in
            # setup_fn_target_offsets must be based on `meta.code` instead.
            if not meta.block_target_offset_remap:
                meta.block_target_offset_remap = {}

                def remap_block_offsets(
                    instructions: List[Instruction], code_options: Dict[str, Any]
                ):
                    # NOTE: each prefix block generates exactly one PUSH_EXC_INFO,
                    # so we can tell which block a prefix PUSH_EXC_INFO belongs to,
                    # by counting. Then we can use meta.prefix_block-target_offset_remap
                    # to determine where in the original code the PUSH_EXC_INFO offset
                    # replaced.
                    prefix_blocks = []
                    for idx, inst in enumerate(instructions):
                        if len(prefix_blocks) == len(
                            meta.prefix_block_target_offset_remap
                        ):
                            break
                        if inst.opname == "PUSH_EXC_INFO":
                            prefix_blocks.append(inst)

                    # offsets into prefix
                    for inst, o in zip(
                        prefix_blocks, meta.prefix_block_target_offset_remap
                    ):
                        meta.block_target_offset_remap[inst.offset] = o

                    # old bytecode targets are after the prefix PUSH_EXC_INFO's
                    old_start_offset = prefix_blocks[-1].offset if prefix_blocks else -1
                    # offsets into old bytecode
                    old_inst_offsets = sorted(
                        n for n in setup_fn_target_offsets if n > old_start_offset
                    )
                    targets = _filter_iter(
                        instructions, old_inst_offsets, lambda inst, o: inst.offset == o
                    )
                    new_targets = _filter_iter(
                        zip(reversed(instructions), reversed(meta.instructions)),
                        targets,
                        lambda v1, v2: v1[0] is v2,
                    )
                    for new, old in zip(new_targets, targets):
                        meta.block_target_offset_remap[old.offset] = new[1].offset

                transform_code_object(code, remap_block_offsets)

            # if offset is not in setup_fn_target_offsets, it is an error
            setup_fn_target_offsets = tuple(
                meta.block_target_offset_remap[n] for n in setup_fn_target_offsets
            )
        return ContinueExecutionCache.lookup(
            meta.code, lineno, new_offset, setup_fn_target_offsets, *args
        )


"""
# partially finished support for with statements

def convert_locals_to_cells(
        instructions: List[Instruction],
        code_options: Dict[str, Any]):

    code_options["co_cellvars"] = tuple(
        var
        for var in code_options["co_varnames"]
        if var not in code_options["co_freevars"]
        and not var.startswith("___stack")
    )
    cell_and_free = code_options["co_cellvars"] + code_options["co_freevars"]
    for inst in instructions:
        if str(inst.argval).startswith("___stack"):
            continue
        elif inst.opname == "LOAD_FAST":
            inst.opname = "LOAD_DEREF"
        elif inst.opname == "STORE_FAST":
            inst.opname = "STORE_DEREF"
        elif inst.opname == "DELETE_FAST":
            inst.opname = "DELETE_DEREF"
        else:
            continue
        inst.opcode = dis.opmap[inst.opname]
        assert inst.argval in cell_and_free, inst.argval
        inst.arg = cell_and_free.index(inst.argval)

def patch_setup_with(
    instructions: List[Instruction],
    code_options: Dict[str, Any]
):
    nonlocal need_skip
    need_skip = True
    target_index = next(
        idx for idx, i in enumerate(instructions) if i.offset == offset
    )
    assert instructions[target_index].opname == "SETUP_WITH"
    convert_locals_to_cells(instructions, code_options)

    stack_depth_before = nstack + stack_effect(instructions[target_index].opcode,
                                               instructions[target_index].arg)

    inside_with = []
    inside_with_resume_at = None
    stack_depth = stack_depth_before
    idx = target_index + 1
    for idx in range(idx, len(instructions)):
        inst = instructions[idx]
        if inst.opname == "BEGIN_FINALLY":
            inside_with_resume_at = inst
            break
        elif inst.target is not None:
            unimplemented("jump from with not supported")
        elif inst.opname in ("BEGIN_FINALLY", "WITH_CLEANUP_START", "WITH_CLEANUP_FINISH", "END_FINALLY",
                             "POP_FINALLY", "POP_EXCEPT",
                             "POP_BLOCK", "END_ASYNC_FOR"):
            unimplemented("block ops not supported")
        inside_with.append(inst)
        stack_depth += stack_effect(inst.opcode, inst.arg)
    assert inside_with_resume_at

    instructions = [
        create_instruction("LOAD_FAST", f"___stack{i}") for i in range(nstack)
    ] + [
        create_instruction("SETUP_WITH", target=instructions[target_index].target)
        ... call the function ...
        unpack_tuple
    ] + [
        create_instruction("JUMP_ABSOLUTE", target=inside_with_resume_at)
    ]
"""
