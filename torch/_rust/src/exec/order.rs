// In which order a step lets go of what it holds.
//
// CPython's instruction takes its operands from the stack and lets go of
// those it consumes one after the other; an object whose last reference
// goes there dies there, and a program sees the order of two deaths (a
// __del__, the callback of a weak reference). The executor holds each value
// once and drops, behind a step, the operands nothing reads again (live.rs).
// This module says in which order, per kind of instruction, as CPython
// 3.12's definition of the instruction has it (Python/bytecodes.c), never
// by the numbers the values happen to have.
//
// What it is given: the op of a step and its operands (ir.rs), or what a
// block holds.
// What it guarantees
//   - `releases`: the operands of the step, each once, in the order in
//     which the GENERIC form of the instruction lets go of them; an operand
//     that stands at two places dies at the later one. Every op of the IR
//     has a row, and every opcode the generic instruction runs (exec_ops.c
//     tr_generic); an opcode with no row has no order and its code is not
//     planned (it runs on CPython).
//   - `top_down`: what a block holds of the stack, in the order CPython
//     pops it: the last one the block got first (the body's values, then
//     what the version read on entry, then the params, each from its end).
//     An unwind uses it for what the raising instruction did not let go
//     of, and a version for the params it does not use.
//   - `as_got`: the same values in the order the block got them. The end of
//     a version uses it for what no step dropped: in a plan as written that
//     is one value at most on any path the table of deaths knows; in an
//     optimized plan they are the values of stores to locals the plan does
//     not make, which CPython lets go of in the order of the stores (or of
//     the slots, at the frame's exit).
// What it does not cover
//   - an instruction CPython specialized after its first runs may let go in
//     another order (CALL of list.append: the receiver before the callable);
//   - an operand CPython keeps on the stack and pops later (the list under
//     LIST_APPEND, the iterator of a loop) is dropped behind its last use,
//     with the operands of that step;
//   - the stack's own order where the compiler swapped it (SWAP): `top_down`
//     goes by when the block got a value.

use pyo3::prelude::*;

use crate::cfg::ir::*;
use crate::cfg::opcodes as oc;

enum Order {
    // As the instruction took them from the stack, bottom first: the order
    // of the IR's args. The row of an instruction whose definition lets go
    // of its inputs with DECREF_INPUTS(), or one by one in that order, or
    // that has at most one operand, or that moves its operands into what it
    // makes (no operand dies with it).
    Stack,
    // These places among the args, in this order.
    Places(&'static [usize]),
}

fn order(op: &Op) -> Option<Order> {
    Some(match op {
        // No operand.
        Op::Null
        | Op::Unknown
        | Op::Param(_)
        | Op::Const(_)
        | Op::Lit(_)
        | Op::Pinned(_)
        | Op::LoadLocal { .. }
        | Op::LoadLocalChecked { .. }
        | Op::DeleteLocal { .. }
        | Op::LoadGlobal(_)
        | Op::LoadGlobalIn { .. }
        | Op::Resume
        | Op::EvalBreaker
        | Op::MakeCell { .. }
        | Op::CopyFreeVars(_)
        | Op::LoadDeref { .. }
        | Op::DeleteDeref { .. }
        | Op::Removed => Order::Stack,
        // One operand: the instr a Project reads; POP_JUMP_IF_*'s condition;
        // STORE_FAST's and STORE_DEREF's value (it moves into the slot, and
        // what the slot held dies there); LOAD_ATTR's owner; GET_ITER's
        // iterable; FOR_ITER's iterator, which goes when it is exhausted;
        // LOAD_FROM_DICT_OR_DEREF's dict.
        Op::Project(_)
        | Op::ToBool
        | Op::IsNone
        | Op::StoreLocal { .. }
        | Op::StoreDeref { .. }
        | Op::LoadAttr(_)
        | Op::LoadMethod(_)
        | Op::GetIter
        | Op::IterNext
        | Op::LoadFromDictOrDeref { .. } => Order::Stack,
        // DECREF_INPUTS(): STORE_ATTR (v, owner), BINARY_OP (lhs, rhs),
        // COMPARE_OP (left, right).
        Op::StoreAttr(_) | Op::Binary(_) | Op::Compare(_) => Order::Stack,
        // CALL: the callable, then the arguments in their order (with a
        // method: the function, the receiver, the arguments).
        Op::Call { .. } => Order::Stack,
        // BUILD_TUPLE and MAKE_FUNCTION move their operands into the
        // object; SEND lets go of the value sent and keeps the receiver;
        // YIELD_VALUE hands the value out and the frame keeps its stack.
        Op::BuildTuple(_) | Op::MakeFunction(_) | Op::Send | Op::Yield => Order::Stack,
        // The optimizer's own instructions stand for calls: as CALL.
        Op::Aten { .. } | Op::Expect(_) | Op::Autograd(_) | Op::Region(_) => Order::Stack,
        Op::Bytecode { opcode, .. } => match *opcode {
            // (container, start, stop): the slice of start and stop is let
            // go of first, and takes both with it; then the container.
            oc::BINARY_SLICE => Order::Places(&[1, 2, 0]),
            // (v, container, start, stop): the slice, the value, the
            // container.
            oc::STORE_SLICE => Order::Places(&[2, 3, 0, 1]),
            // No operand, or one.
            oc::NOP
            | oc::RETURN_GENERATOR
            | oc::LOAD_BUILD_CLASS
            | oc::LOAD_ASSERTION_ERROR
            | oc::SETUP_ANNOTATIONS
            | oc::LOAD_LOCALS
            | oc::LOAD_NAME
            | oc::DELETE_NAME
            | oc::DELETE_GLOBAL
            | oc::UNARY_NEGATIVE
            | oc::UNARY_NOT
            | oc::UNARY_INVERT
            | oc::GET_LEN
            | oc::MATCH_MAPPING
            | oc::MATCH_SEQUENCE
            | oc::PUSH_EXC_INFO
            | oc::POP_EXCEPT
            | oc::BEFORE_WITH
            | oc::BEFORE_ASYNC_WITH
            | oc::STORE_NAME
            | oc::STORE_GLOBAL
            | oc::LOAD_FROM_DICT_OR_GLOBALS
            | oc::DELETE_ATTR
            | oc::UNPACK_SEQUENCE
            | oc::UNPACK_EX
            | oc::IMPORT_FROM
            | oc::CALL_INTRINSIC_1
            | oc::GET_YIELD_FROM_ITER
            | oc::GET_AWAITABLE
            | oc::GET_AITER
            | oc::GET_ANEXT => Order::Stack,
            // DECREF_INPUTS(), or the inputs one by one from the bottom:
            // (container, sub), (v, container, sub), (subject, type, names),
            // (left, right), (exc, type), (level, fromlist), (value2,
            // value1), (awaitable, exc), (sub_iter, last_sent, exc),
            // (global_super, class, self), (func, callargs, kwargs),
            // (value, spec), the items of a set, the keys and values of a
            // dict in pairs, the values and then the keys tuple, (start,
            // stop, step), the pieces of a string.
            oc::BINARY_SUBSCR
            | oc::STORE_SUBSCR
            | oc::DELETE_SUBSCR
            | oc::MATCH_CLASS
            | oc::IS_OP
            | oc::CONTAINS_OP
            | oc::CHECK_EXC_MATCH
            | oc::CHECK_EG_MATCH
            | oc::IMPORT_NAME
            | oc::META_LAZY_IMPORT_NAME
            | oc::CALL_INTRINSIC_2
            | oc::END_ASYNC_FOR
            | oc::CLEANUP_THROW
            | oc::LOAD_SUPER_ATTR
            | oc::CALL_FUNCTION_EX
            | oc::FORMAT_VALUE
            | oc::BUILD_SET
            | oc::BUILD_MAP
            | oc::BUILD_CONST_KEY_MAP
            | oc::BUILD_SLICE
            | oc::BUILD_STRING => Order::Stack,
            // The container stays on the stack and what is put into it
            // moves in or is let go of in its order: (list, v), (set, v),
            // (dict, key, value), (container, iterable), (func, dict,
            // update); MATCH_KEYS and WITH_EXCEPT_START keep their
            // operands; BUILD_LIST moves its items into the list.
            oc::LIST_APPEND
            | oc::SET_ADD
            | oc::MAP_ADD
            | oc::LIST_EXTEND
            | oc::SET_UPDATE
            | oc::DICT_UPDATE
            | oc::DICT_MERGE
            | oc::MATCH_KEYS
            | oc::WITH_EXCEPT_START
            | oc::BUILD_LIST => Order::Stack,
            _ => return None,
        },
    })
}

// The operands of an instr with `op` and `args` in the order the instr lets
// go of them, each once at its last place; None: the op has no row.
pub fn releases(op: &Op, args: &[ValueId]) -> Option<Vec<ValueId>> {
    let mut out: Vec<ValueId> = Vec::with_capacity(args.len());
    let mut put = |a: ValueId| {
        if !a.is_reserved() {
            out.retain(|x| *x != a);
            out.push(a);
        }
    };
    match order(op)? {
        Order::Stack => args.iter().for_each(|a| put(*a)),
        // An instr the optimizer gave other operands has no places.
        Order::Places(places) if places.len() == args.len() => places.iter().for_each(|p| put(args[*p])),
        Order::Places(_) => return None,
    }
    Some(out)
}

// What a version of a block holds, in the order the block got it.
pub fn as_got<'a>(blk: &'a Block, ver: &'a Version) -> impl DoubleEndedIterator<Item = ValueId> + 'a {
    blk.params.iter().chain(&ver.inputs).chain(&ver.body).copied()
}

// What a version of a block holds, the last one it got first.
pub fn top_down<'a>(blk: &'a Block, ver: &'a Version) -> impl Iterator<Item = ValueId> + 'a {
    as_got(blk, ver).rev()
}

/// The places of a generic instruction's operands (among its args, bottom
/// first) in the order in which the instruction lets go of them, for one
/// with `nargs` operands; None for an opcode with no row (for tests).
#[pyfunction]
pub fn _cfg_releases(opcode: u8, nargs: u32) -> Option<Vec<u32>> {
    let args: Vec<ValueId> = (0..nargs).map(|i| ValueId(NUM_RESERVED + i)).collect();
    releases(&Op::Bytecode { opcode, arg: 0 }, &args).map(|vs| vs.iter().map(|v| v.0 - NUM_RESERVED).collect())
}
