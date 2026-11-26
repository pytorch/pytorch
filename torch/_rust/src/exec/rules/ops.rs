// The first family of the rule table: the instruction kinds.
//
// Every Op and every opcode of CPython 3.12 that cfg/opcodes.rs names is in
// one of four lists here, and nowhere by default:
//   - a ROW, where what the instruction does is a fact of its kind, whatever
//     the classes of its operands: slots of the frame, constants, cells,
//     globals, moves on the stack, new exact containers that only hold their
//     operands, identity tests, the handler's bookkeeping;
//   - AS the kinds an opcode is lowered to (cfg/interp.rs), which have their
//     own entries;
//   - LATER, with the family that will take it: what the instruction does
//     depends on the class of an operand (an attribute, a subscript, a truth
//     test, an iteration, an operator, a call, a store), or it is something
//     the optimizer put in place of an instruction. Until that family is
//     written the answer is "not mine" and the code that decided before
//     decides;
//   - a STOP, with the reason: no trace runs it and no region holds it.
// A new Op does not compile (what() and kind() have no wildcard arm); an
// opcode in none of the lists is Unlisted, which is a stop, and
// test/test_vm_gate.py fails for one that cfg/opcodes.rs names.
//
// What it may not assume
//   - that a ROW's instruction ran on operands the rest of the engine
//     checked: the row says what the kind reads and writes, not that the
//     trace recorded it.

use pyo3::ffi::PyObject;

use super::{Class, Enters, Family, Group, Kind, Read, Res, Row, Scalars, Taken, What, Write, KINDS};
use crate::cfg::ir::Op;
use crate::cfg::opcodes::{self, *};
use Enters::{Foreign, Never};

pub const FAMILY: Family = Family { name: "ops", rule, table };

// What an instruction is: a call of a known callee is that callee.
pub fn what(op: &Op, callee: Option<*mut PyObject>) -> What {
    if let Some(f) = callee.filter(|f| !f.is_null()) {
        return What::Callee(f);
    }
    What::Op(match op {
        Op::Bytecode { opcode, arg } => return What::Opcode(*opcode, *arg),
        Op::Null => Kind::Null,
        Op::Unknown => Kind::Unknown,
        Op::Param(_) => Kind::Param,
        Op::Const(_) => Kind::Const,
        Op::Project(_) => Kind::Project,
        Op::ToBool => Kind::ToBool,
        Op::IsNone => Kind::IsNone,
        Op::LoadLocal { .. } => Kind::LoadLocal,
        Op::LoadLocalChecked { .. } => Kind::LoadLocalChecked,
        Op::StoreLocal { .. } => Kind::StoreLocal,
        Op::DeleteLocal { .. } => Kind::DeleteLocal,
        Op::LoadGlobal(_) => Kind::LoadGlobal,
        Op::LoadGlobalIn { .. } => Kind::LoadGlobalIn,
        Op::LoadAttr(_) => Kind::LoadAttr,
        Op::LoadMethod(_) => Kind::LoadMethod,
        Op::StoreAttr(_) => Kind::StoreAttr,
        Op::Call { .. } => Kind::Call,
        Op::Binary(_) => Kind::Binary,
        Op::Compare(_) => Kind::Compare,
        Op::BuildTuple(_) => Kind::BuildTuple,
        Op::GetIter => Kind::GetIter,
        Op::MakeFunction(_) => Kind::MakeFunction,
        Op::Yield => Kind::Yield,
        Op::Resume => Kind::Resume,
        Op::IterNext => Kind::IterNext,
        Op::Send => Kind::Send,
        Op::EvalBreaker => Kind::EvalBreaker,
        Op::MakeCell { .. } => Kind::MakeCell,
        Op::CopyFreeVars(_) => Kind::CopyFreeVars,
        Op::LoadDeref { .. } => Kind::LoadDeref,
        Op::StoreDeref { .. } => Kind::StoreDeref,
        Op::DeleteDeref { .. } => Kind::DeleteDeref,
        Op::LoadFromDictOrDeref { .. } => Kind::LoadFromDictOrDeref,
        Op::Aten { .. } => Kind::Aten,
        Op::Lit(_) => Kind::Lit,
        Op::Pinned(_) => Kind::Pinned,
        Op::Expect(_) => Kind::Expect,
        Op::Autograd(_) => Kind::Autograd,
        Op::Region(_) => Kind::Region,
        Op::Removed => Kind::Removed,
    })
}

const fn row(operands: &'static [(i8, Class)], reads: &'static [Read], writes: Write, result: Res, enters: Enters, scalars: Scalars, why: &'static str) -> Row {
    Row { operands, reads, writes, result, enters, scalars, why }
}

const CONST_ROW: Row = row(&[], &[Read::Constant], Write::None, Res::Lit, Never, Scalars::None, "A constant is part of the code object, and the code is what a region is found by.");
const PROJECT_ROW: Row = row(&[], &[], Write::None, Res::Part, Never, Scalars::None, "It names one output of the instruction it is an operand of, which has written it already.");
const IS_NONE_ROW: Row = row(&[], &[Read::Identity], Write::None, Res::Lit, Never, Scalars::None, "Whether an operand is None is which object it is, and is read from no object.");
const LOAD_LOCAL_ROW: Row = row(&[], &[Read::Slot], Write::None, Res::Slot, Never, Scalars::None, "A slot of the frame holds what the run put there: the value comes with whatever is known of it.");
const STORE_LOCAL_ROW: Row = row(&[], &[Read::Slot], Write::Slot, Res::None, Never, Scalars::None, "A slot of the frame is the run's own: nothing outside the frame sees it change.");
const LOAD_GLOBAL_ROW: Row = row(&[], &[Read::Global], Write::None, Res::Pinned, Never, Scalars::None, "The value is the item of a globals dict, else of its builtins dict, under an interned name: a lookup in C, and that item is what a region is checked on.");
const BUILD_ROW: Row = row(&[], &[], Write::None, Res::Made, Never, Scalars::Parts, "A new exact tuple, list or slice holds its operands as they are and reads nothing of them.");
const MAKE_FUNCTION_ROW: Row = row(&[], &[], Write::None, Res::Made, Never, Scalars::None, "A new function holds the code object, the defaults and the cells it is handed and reads none of them.");
const BREAKER_ROW: Row = row(&[], &[Read::Pending], Write::None, Res::None, Foreign, Scalars::None, "The interpreter's pending work (a signal handler, the collector, a switch of threads) runs here: frames that nobody named, each a stop of its own.");
const MAKE_CELL_ROW: Row = row(&[], &[Read::Slot], Write::Slot, Res::None, Never, Scalars::None, "It wraps the value of a slot of the frame in a new cell, in that slot.");
const COPY_FREE_VARS_ROW: Row = row(&[], &[Read::Closure], Write::Slot, Res::None, Never, Scalars::None, "It copies the cells of the frame's own function into slots of the frame, as the objects they are.");
const LOAD_DEREF_ROW: Row = row(&[], &[Read::Slot, Read::Cell], Write::None, Res::Pinned, Never, Scalars::None, "The value is the content of a cell: the frame's own if the frame made the cell, else what a region is checked on.");
const STACK_ROW: Row = row(&[], &[], Write::None, Res::None, Never, Scalars::None, "It moves values on the frame's own stack, or is no instruction at all: no object is read.");
const TERMINATOR_ROW: Row = row(&[], &[Read::Slot], Write::None, Res::None, Never, Scalars::None, "It ends a block: control moves within the frame, or a value of the frame's stack is handed on as it is.");
const BUILD_STRING_ROW: Row = row(&[], &[Read::Text], Write::None, Res::Made, Never, Scalars::None, "It joins the characters of its operands, which are strs: read in C, and a str never changes.");
const CONST_KEY_MAP_ROW: Row = row(&[(-1, Class::Group(Group::ExactTuple))], &[Read::Constant], Write::None, Res::Made, Never, Scalars::Parts, "A new exact dict whose keys are the items of a tuple of constants, hashed in C, and whose values it holds as they are.");
const LIST_APPEND_ROW: Row = row(&[(0, Class::Group(Group::ExactList))], &[], Write::Fills(0), Res::None, Never, Scalars::Parts, "It appends its operand, as it is, to the exact list of a comprehension, which the frame made and holds on its stack.");
const IS_ROW: Row = row(&[], &[Read::Identity], Write::None, Res::Lit, Never, Scalars::None, "Whether two operands are one object is read from neither.");
const LIST_TO_TUPLE_ROW: Row = row(&[(0, Class::Group(Group::ExactList))], &[], Write::None, Res::Made, Never, Scalars::Parts, "A new exact tuple of the items of the exact list the frame built for a starred call or display.");
const HANDLED_ROW: Row = row(&[], &[Read::Handled], Write::Handled, Res::Slot, Never, Scalars::None, "It saves, or puts back, the exception the thread is handling: a handler's own bookkeeping, undone before the handler is left.");
const EXC_MATCH_ROW: Row = row(&[], &[Read::Class], Write::None, Res::Lit, Never, Scalars::None, "Whether the exception is of the handler's classes is read from the MRO of its class in C: no __instancecheck__ runs.");

fn kind(k: Kind) -> Taken {
    match k {
        Kind::Const => Taken::Rows(&[CONST_ROW]),
        Kind::Project => Taken::Rows(&[PROJECT_ROW]),
        Kind::IsNone => Taken::Rows(&[IS_NONE_ROW]),
        Kind::LoadLocal | Kind::LoadLocalChecked => Taken::Rows(&[LOAD_LOCAL_ROW]),
        Kind::StoreLocal | Kind::DeleteLocal => Taken::Rows(&[STORE_LOCAL_ROW]),
        Kind::LoadGlobal | Kind::LoadGlobalIn => Taken::Rows(&[LOAD_GLOBAL_ROW]),
        Kind::BuildTuple => Taken::Rows(&[BUILD_ROW]),
        Kind::MakeFunction => Taken::Rows(&[MAKE_FUNCTION_ROW]),
        Kind::Resume | Kind::EvalBreaker => Taken::Rows(&[BREAKER_ROW]),
        Kind::MakeCell => Taken::Rows(&[MAKE_CELL_ROW]),
        Kind::CopyFreeVars => Taken::Rows(&[COPY_FREE_VARS_ROW]),
        Kind::LoadDeref => Taken::Rows(&[LOAD_DEREF_ROW]),
        Kind::LoadAttr | Kind::LoadMethod => Taken::Later("attributes"),
        Kind::ToBool | Kind::Compare => Taken::Later("subscripts"),
        Kind::GetIter | Kind::IterNext | Kind::Send | Kind::Yield => Taken::Later("iteration"),
        Kind::Binary => Taken::Later("torch"),
        Kind::Call => Taken::Later("functions"),
        Kind::StoreAttr | Kind::StoreDeref | Kind::DeleteDeref => Taken::Later("stores"),
        // What the optimizer puts in place of an instruction. A trace still
        // runs optimized plans; when it runs the code as written these stop.
        Kind::Aten | Kind::Lit | Kind::Pinned | Kind::Expect | Kind::Autograd | Kind::Region => Taken::Later("recorder"),
        Kind::LoadFromDictOrDeref => Taken::Stop(NAMESPACE),
        Kind::Null | Kind::Unknown | Kind::Param | Kind::Removed => Taken::Stop("it is a value or a hole of a plan, not an instruction: no block's body holds one"),
    }
}

const NAMESPACE: &str = "it reads or writes the namespace of a class body or a module by name: a mapping of any class, and a write that outlives every region";
const MATCH_FLAG: &str = "it reads a flag of the subject's class that registering the class with an ABC changes, and nothing records that read";
const WITH: &str = "it looks up and calls __enter__ or __exit__ of an object of any class";
const AWAIT: &str = "it drives a coroutine or an asynchronous iterator through methods of an object of any class";
const INTRINSIC: &str = "it is an intrinsic of the interpreter (print, import *, a type parameter, an exception group) that no rule describes";
const NEVER: &str = "no code object of a program holds it";

// The opcodes, each in one entry.
const OPCODES: &[(&[u8], Taken)] = &[
    (&[CACHE, NOP, POP_TOP, PUSH_NULL, END_FOR, END_SEND, COPY, SWAP, KW_NAMES, EXTENDED_ARG], Taken::Rows(&[STACK_ROW])),
    (&[JUMP_FORWARD, JUMP_BACKWARD_NO_INTERRUPT, RETURN_VALUE, RERAISE], Taken::Rows(&[TERMINATOR_ROW])),
    (&[BUILD_LIST, BUILD_SLICE], Taken::Rows(&[BUILD_ROW])),
    (&[BUILD_STRING], Taken::Rows(&[BUILD_STRING_ROW])),
    (&[BUILD_CONST_KEY_MAP], Taken::Rows(&[CONST_KEY_MAP_ROW])),
    (&[LIST_APPEND], Taken::Rows(&[LIST_APPEND_ROW])),
    (&[IS_OP], Taken::Rows(&[IS_ROW])),
    (&[PUSH_EXC_INFO, POP_EXCEPT], Taken::Rows(&[HANDLED_ROW])),
    (&[CHECK_EXC_MATCH], Taken::Rows(&[EXC_MATCH_ROW])),
    (&[RESUME], Taken::As(&[Kind::Resume])),
    (&[LOAD_CONST, RETURN_CONST], Taken::As(&[Kind::Const])),
    (&[LOAD_FAST, LOAD_CLOSURE], Taken::As(&[Kind::LoadLocal])),
    (&[LOAD_FAST_CHECK], Taken::As(&[Kind::LoadLocalChecked])),
    (&[LOAD_FAST_AND_CLEAR], Taken::As(&[Kind::LoadLocal, Kind::StoreLocal])),
    (&[STORE_FAST], Taken::As(&[Kind::StoreLocal])),
    (&[DELETE_FAST], Taken::As(&[Kind::DeleteLocal])),
    (&[MAKE_CELL], Taken::As(&[Kind::MakeCell])),
    (&[COPY_FREE_VARS], Taken::As(&[Kind::CopyFreeVars])),
    (&[LOAD_DEREF], Taken::As(&[Kind::LoadDeref])),
    (&[STORE_DEREF], Taken::As(&[Kind::StoreDeref])),
    (&[DELETE_DEREF], Taken::As(&[Kind::DeleteDeref])),
    (&[LOAD_FROM_DICT_OR_DEREF], Taken::As(&[Kind::LoadFromDictOrDeref])),
    (&[LOAD_GLOBAL], Taken::As(&[Kind::LoadGlobal])),
    (&[LOAD_ATTR], Taken::As(&[Kind::LoadAttr, Kind::LoadMethod, Kind::Project])),
    (&[STORE_ATTR], Taken::As(&[Kind::StoreAttr])),
    (&[COMPARE_OP], Taken::As(&[Kind::Compare])),
    (&[BINARY_OP], Taken::As(&[Kind::Binary])),
    (&[BUILD_TUPLE], Taken::As(&[Kind::BuildTuple])),
    (&[GET_ITER], Taken::As(&[Kind::GetIter])),
    (&[MAKE_FUNCTION], Taken::As(&[Kind::MakeFunction])),
    (&[CALL], Taken::As(&[Kind::Call])),
    (&[YIELD_VALUE], Taken::As(&[Kind::Yield])),
    (&[POP_JUMP_IF_TRUE, POP_JUMP_IF_FALSE], Taken::As(&[Kind::ToBool])),
    (&[POP_JUMP_IF_NONE, POP_JUMP_IF_NOT_NONE], Taken::As(&[Kind::IsNone])),
    (&[JUMP_BACKWARD], Taken::As(&[Kind::EvalBreaker])),
    (&[FOR_ITER], Taken::As(&[Kind::IterNext, Kind::Project])),
    (&[SEND], Taken::As(&[Kind::Send, Kind::Project])),
    (&[LOAD_SUPER_ATTR], Taken::Later("attributes")),
    // The last four hash their operands: code of the keys' classes.
    (&[UNARY_NOT, CONTAINS_OP, BINARY_SUBSCR, BINARY_SLICE, BUILD_MAP, BUILD_SET, MAP_ADD, SET_ADD], Taken::Later("subscripts")),
    (&[RETURN_GENERATOR, UNPACK_SEQUENCE, UNPACK_EX, LIST_EXTEND, SET_UPDATE, DICT_UPDATE, DICT_MERGE], Taken::Later("iteration")),
    (&[UNARY_NEGATIVE, UNARY_INVERT], Taken::Later("torch")),
    (&[FORMAT_VALUE], Taken::Later("descriptors")),
    (&[CALL_FUNCTION_EX], Taken::Later("functions")),
    // Given a class, the raise makes an instance of it.
    (&[RAISE_VARARGS], Taken::Later("classes")),
    (&[STORE_SUBSCR, STORE_SLICE, DELETE_SUBSCR, DELETE_ATTR], Taken::Later("stores")),
    (&[MATCH_CLASS], Taken::Stop("it reads __match_args__ of the class and attributes of the subject by name, and nothing records those reads")),
    (&[MATCH_KEYS], Taken::Stop("it looks the keys of the pattern up through the subject's get method: code of any mapping class")),
    (&[MATCH_MAPPING, MATCH_SEQUENCE], Taken::Stop(MATCH_FLAG)),
    (&[GET_LEN], Taken::Stop("it asks the subject of a match for its length: __len__ of any class")),
    (&[STORE_GLOBAL, DELETE_GLOBAL, STORE_NAME, DELETE_NAME, LOAD_NAME, LOAD_LOCALS, LOAD_FROM_DICT_OR_GLOBALS, SETUP_ANNOTATIONS], Taken::Stop(NAMESPACE)),
    (&[LOAD_BUILD_CLASS], Taken::Stop("it begins a class statement, which runs a metaclass and the class body")),
    (&[IMPORT_NAME, IMPORT_FROM], Taken::Stop("it runs the import machinery (sys.modules, __import__, a module's code, a module's __getattr__), and a region checks none of it")),
    (&[BEFORE_WITH, BEFORE_ASYNC_WITH, WITH_EXCEPT_START], Taken::Stop(WITH)),
    (&[GET_AWAITABLE, GET_AITER, GET_ANEXT, END_ASYNC_FOR, CLEANUP_THROW, GET_YIELD_FROM_ITER], Taken::Stop(AWAIT)),
    (&[CHECK_EG_MATCH], Taken::Stop("it splits an exception group through its split method, which a subclass may define")),
    (&[CALL_INTRINSIC_1, CALL_INTRINSIC_2], Taken::Stop(INTRINSIC)),
    (&[LOAD_ASSERTION_ERROR], Taken::Stop("it begins the raise of a failed assert, which no traced run met")),
    (&[RESERVED, INTERPRETER_EXIT], Taken::Stop(NEVER)),
];

// CALL_INTRINSIC_1's opargs that are not stops: 5 is unary +, a slot of the
// operand's class; 6 makes a tuple of the list of a starred call.
const POSITIVE: u32 = 5;
const TO_TUPLE: u32 = 6;

fn opcode(op: u8, arg: u32) -> Taken {
    match (op, arg) {
        (CALL_INTRINSIC_1, POSITIVE) => Taken::Later("descriptors"),
        (CALL_INTRINSIC_1, TO_TUPLE) => Taken::Rows(&[LIST_TO_TUPLE_ROW]),
        _ => OPCODES.iter().find(|(ops, _)| ops.contains(&op)).map_or(Taken::Unlisted, |(_, taken)| *taken),
    }
}

pub fn opcode_name(op: u8, arg: u32) -> String {
    match (opcodes::caches(op), op, arg) {
        (None, ..) => format!("<{op}>"),
        (_, CALL_INTRINSIC_1, POSITIVE | TO_TUPLE) => format!("{} {arg}", opcodes::name(op)),
        _ => opcodes::name(op).to_string(),
    }
}

fn rule(what: &What) -> Taken {
    match *what {
        What::Op(k) => kind(k),
        What::Opcode(op, arg) => opcode(op, arg),
        What::Callee(_) => Taken::No,
    }
}

fn table() -> Vec<(String, Taken)> {
    let kinds = KINDS.iter().map(|k| What::Op(*k));
    let opcodes = (0..=u8::MAX).flat_map(|op| [0, POSITIVE, TO_TUPLE].into_iter().filter(move |arg| *arg == 0 || op == CALL_INTRINSIC_1).map(move |arg| What::Opcode(op, arg)));
    kinds.chain(opcodes).map(|what| (what.name(), rule(&what))).collect()
}
