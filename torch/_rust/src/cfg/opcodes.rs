// CPython 3.12 opcode table: numbers, inline cache counts, and stack signatures
// transcribed from 3.12's Python/bytecodes.c. test/test_rust.py checks this
// against `dis`.

// IMPORT_NAME under Meta's lazy-imports CPython 3.12; see bytecode::decode.
pub const META_LAZY_IMPORT_NAME: u8 = 183;

macro_rules! opcodes {
    ($($name:ident = $num:expr, $caches:expr;)*) => {
        $(pub const $name: u8 = $num;)*
        pub const TABLE: &[(&str, u8, u8)] = &[$((stringify!($name), $num, $caches),)*];
    };
}

opcodes! {
    CACHE = 0, 0;
    POP_TOP = 1, 0;
    PUSH_NULL = 2, 0;
    INTERPRETER_EXIT = 3, 0;
    END_FOR = 4, 0;
    END_SEND = 5, 0;
    NOP = 9, 0;
    UNARY_NEGATIVE = 11, 0;
    UNARY_NOT = 12, 0;
    UNARY_INVERT = 15, 0;
    RESERVED = 17, 0;
    BINARY_SUBSCR = 25, 1;
    BINARY_SLICE = 26, 0;
    STORE_SLICE = 27, 0;
    GET_LEN = 30, 0;
    MATCH_MAPPING = 31, 0;
    MATCH_SEQUENCE = 32, 0;
    MATCH_KEYS = 33, 0;
    PUSH_EXC_INFO = 35, 0;
    CHECK_EXC_MATCH = 36, 0;
    CHECK_EG_MATCH = 37, 0;
    WITH_EXCEPT_START = 49, 0;
    GET_AITER = 50, 0;
    GET_ANEXT = 51, 0;
    BEFORE_ASYNC_WITH = 52, 0;
    BEFORE_WITH = 53, 0;
    END_ASYNC_FOR = 54, 0;
    CLEANUP_THROW = 55, 0;
    STORE_SUBSCR = 60, 1;
    DELETE_SUBSCR = 61, 0;
    GET_ITER = 68, 0;
    GET_YIELD_FROM_ITER = 69, 0;
    LOAD_BUILD_CLASS = 71, 0;
    LOAD_ASSERTION_ERROR = 74, 0;
    RETURN_GENERATOR = 75, 0;
    RETURN_VALUE = 83, 0;
    SETUP_ANNOTATIONS = 85, 0;
    LOAD_LOCALS = 87, 0;
    POP_EXCEPT = 89, 0;
    STORE_NAME = 90, 0;
    DELETE_NAME = 91, 0;
    UNPACK_SEQUENCE = 92, 1;
    FOR_ITER = 93, 1;
    UNPACK_EX = 94, 0;
    STORE_ATTR = 95, 4;
    DELETE_ATTR = 96, 0;
    STORE_GLOBAL = 97, 0;
    DELETE_GLOBAL = 98, 0;
    SWAP = 99, 0;
    LOAD_CONST = 100, 0;
    LOAD_NAME = 101, 0;
    BUILD_TUPLE = 102, 0;
    BUILD_LIST = 103, 0;
    BUILD_SET = 104, 0;
    BUILD_MAP = 105, 0;
    LOAD_ATTR = 106, 9;
    COMPARE_OP = 107, 1;
    IMPORT_NAME = 108, 0;
    IMPORT_FROM = 109, 0;
    JUMP_FORWARD = 110, 0;
    POP_JUMP_IF_FALSE = 114, 0;
    POP_JUMP_IF_TRUE = 115, 0;
    LOAD_GLOBAL = 116, 4;
    IS_OP = 117, 0;
    CONTAINS_OP = 118, 0;
    RERAISE = 119, 0;
    COPY = 120, 0;
    RETURN_CONST = 121, 0;
    BINARY_OP = 122, 1;
    SEND = 123, 1;
    LOAD_FAST = 124, 0;
    STORE_FAST = 125, 0;
    DELETE_FAST = 126, 0;
    LOAD_FAST_CHECK = 127, 0;
    POP_JUMP_IF_NOT_NONE = 128, 0;
    POP_JUMP_IF_NONE = 129, 0;
    RAISE_VARARGS = 130, 0;
    GET_AWAITABLE = 131, 0;
    MAKE_FUNCTION = 132, 0;
    BUILD_SLICE = 133, 0;
    JUMP_BACKWARD_NO_INTERRUPT = 134, 0;
    MAKE_CELL = 135, 0;
    LOAD_CLOSURE = 136, 0;
    LOAD_DEREF = 137, 0;
    STORE_DEREF = 138, 0;
    DELETE_DEREF = 139, 0;
    JUMP_BACKWARD = 140, 0;
    LOAD_SUPER_ATTR = 141, 1;
    CALL_FUNCTION_EX = 142, 0;
    LOAD_FAST_AND_CLEAR = 143, 0;
    EXTENDED_ARG = 144, 0;
    LIST_APPEND = 145, 0;
    SET_ADD = 146, 0;
    MAP_ADD = 147, 0;
    COPY_FREE_VARS = 149, 0;
    YIELD_VALUE = 150, 0;
    RESUME = 151, 0;
    MATCH_CLASS = 152, 0;
    FORMAT_VALUE = 155, 0;
    BUILD_CONST_KEY_MAP = 156, 0;
    BUILD_STRING = 157, 0;
    LIST_EXTEND = 162, 0;
    SET_UPDATE = 163, 0;
    DICT_MERGE = 164, 0;
    DICT_UPDATE = 165, 0;
    CALL = 171, 3;
    KW_NAMES = 172, 0;
    CALL_INTRINSIC_1 = 173, 0;
    CALL_INTRINSIC_2 = 174, 0;
    LOAD_FROM_DICT_OR_GLOBALS = 175, 0;
    LOAD_FROM_DICT_OR_DEREF = 176, 0;
}

fn entry(op: u8) -> Option<&'static (&'static str, u8, u8)> {
    TABLE.iter().find(|e| e.1 == op)
}

pub fn name(op: u8) -> &'static str {
    entry(op).map_or("<unknown>", |e| e.0)
}

pub fn caches(op: u8) -> Option<u8> {
    entry(op).map(|e| e.2)
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum In {
    Used,
    // Read through but not an operand (e.g. the list below LIST_APPEND's item).
    Unused,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Out {
    // The input at this index (bottom first), pushed back unchanged.
    Keep(usize),
    New,
}

#[derive(Clone, Debug)]
pub struct Sig {
    pub inputs: Vec<In>,
    pub outputs: Vec<Out>,
}

fn used(n: usize) -> Vec<In> {
    vec![In::Used; n]
}

fn new(n: usize) -> Vec<Out> {
    vec![Out::New; n]
}

fn sig(inputs: Vec<In>, outputs: Vec<Out>) -> Option<Sig> {
    Some(Sig { inputs, outputs })
}

// `(container, unused[n-1], items... -- container, unused[n-1])`.
fn container_update(n: usize, items: usize) -> Option<Sig> {
    if n == 0 {
        return None;
    }
    let mut inputs = vec![In::Used];
    inputs.extend(vec![In::Unused; n - 1]);
    inputs.extend(used(items));
    sig(inputs, (0..n).map(Out::Keep).collect())
}

// Stack signature (inputs bottom first). `jump` selects the taken-branch
// signature for opcodes whose effect differs (FOR_ITER).
pub fn signature(op: u8, arg: u32, jump: bool) -> Option<Sig> {
    let n = arg as usize;
    match op {
        NOP | RESUME | EXTENDED_ARG | SETUP_ANNOTATIONS | DELETE_NAME | DELETE_GLOBAL
        | JUMP_FORWARD | JUMP_BACKWARD | JUMP_BACKWARD_NO_INTERRUPT | RETURN_CONST
        | DELETE_FAST | DELETE_DEREF | MAKE_CELL | COPY_FREE_VARS | KW_NAMES => sig(vec![], vec![]),
        POP_TOP | RETURN_VALUE | POP_EXCEPT | STORE_NAME | STORE_GLOBAL | DELETE_ATTR
        | STORE_FAST | STORE_DEREF | POP_JUMP_IF_FALSE | POP_JUMP_IF_TRUE
        | POP_JUMP_IF_NONE | POP_JUMP_IF_NOT_NONE => sig(used(1), vec![]),
        PUSH_NULL | LOAD_BUILD_CLASS | LOAD_ASSERTION_ERROR | RETURN_GENERATOR | LOAD_LOCALS
        | LOAD_CONST | LOAD_NAME | LOAD_FAST | LOAD_FAST_CHECK | LOAD_CLOSURE
        | LOAD_FAST_AND_CLEAR | LOAD_DEREF => sig(vec![], new(1)),
        END_FOR | DELETE_SUBSCR | STORE_ATTR | END_ASYNC_FOR => sig(used(2), vec![]),
        END_SEND => sig(vec![In::Unused, In::Used], vec![Out::Keep(1)]),
        UNARY_NEGATIVE | UNARY_NOT | UNARY_INVERT | GET_AITER | GET_ITER | GET_YIELD_FROM_ITER
        | GET_AWAITABLE | YIELD_VALUE | CALL_INTRINSIC_1 | LOAD_FROM_DICT_OR_GLOBALS
        | LOAD_FROM_DICT_OR_DEREF => sig(used(1), new(1)),
        BINARY_SUBSCR | COMPARE_OP | IMPORT_NAME | IS_OP | CONTAINS_OP | BINARY_OP
        | CALL_INTRINSIC_2 => sig(used(2), new(1)),
        BINARY_SLICE | MATCH_CLASS => sig(used(3), new(1)),
        STORE_SUBSCR => sig(used(3), vec![]),
        STORE_SLICE => sig(used(4), vec![]),
        GET_LEN | MATCH_MAPPING | MATCH_SEQUENCE | GET_ANEXT | IMPORT_FROM => {
            sig(used(1), vec![Out::Keep(0), Out::New])
        }
        MATCH_KEYS => sig(used(2), vec![Out::Keep(0), Out::Keep(1), Out::New]),
        PUSH_EXC_INFO => sig(used(1), vec![Out::New, Out::Keep(0)]),
        CHECK_EXC_MATCH | SEND => sig(used(2), vec![Out::Keep(0), Out::New]),
        CHECK_EG_MATCH => sig(used(2), new(2)),
        WITH_EXCEPT_START => sig(
            vec![In::Used, In::Unused, In::Unused, In::Used],
            vec![Out::Keep(0), Out::Keep(1), Out::Keep(2), Out::Keep(3), Out::New],
        ),
        BEFORE_ASYNC_WITH | BEFORE_WITH => sig(used(1), new(2)),
        CLEANUP_THROW => sig(used(3), new(2)),
        UNPACK_SEQUENCE => sig(used(1), new(n)),
        UNPACK_EX => sig(used(1), new((n & 0xff) + 1 + (n >> 8))),
        FOR_ITER if jump => sig(used(1), vec![]),
        FOR_ITER => sig(used(1), vec![Out::Keep(0), Out::New]),
        SWAP if n >= 2 => {
            let mut outs: Vec<Out> = (0..n).map(Out::Keep).collect();
            outs.swap(0, n - 1);
            let mut ins = vec![In::Used];
            ins.extend(vec![In::Unused; n - 2]);
            ins.push(In::Used);
            sig(ins, outs)
        }
        COPY if n >= 1 => {
            let mut ins = vec![In::Used];
            ins.extend(vec![In::Unused; n - 1]);
            let mut outs: Vec<Out> = (0..n).map(Out::Keep).collect();
            outs.push(Out::Keep(0));
            sig(ins, outs)
        }
        BUILD_TUPLE | BUILD_LIST | BUILD_SET | BUILD_STRING => sig(used(n), new(1)),
        BUILD_MAP => sig(used(2 * n), new(1)),
        BUILD_CONST_KEY_MAP => sig(used(n + 1), new(1)),
        BUILD_SLICE => sig(used(if n == 3 { 3 } else { 2 }), new(1)),
        LOAD_ATTR => sig(used(1), new(1 + (n & 1))),
        LOAD_GLOBAL => sig(vec![], new(1 + (n & 1))),
        LOAD_SUPER_ATTR => sig(used(3), new(1 + (n & 1))),
        RERAISE => {
            let mut ins = vec![In::Unused; n];
            ins.push(In::Used);
            sig(ins, (0..n).map(Out::Keep).collect())
        }
        RAISE_VARARGS => sig(used(n), vec![]),
        MAKE_FUNCTION => sig(used((arg & 0xf).count_ones() as usize + 1), new(1)),
        CALL_FUNCTION_EX => {
            let mut ins = vec![In::Unused, In::Used, In::Used];
            if n & 1 != 0 {
                ins.push(In::Used);
            }
            sig(ins, new(1))
        }
        LIST_APPEND | SET_ADD | LIST_EXTEND | SET_UPDATE | DICT_UPDATE => container_update(n, 1),
        // (func, callargs, dict, unused[n-1], update -- ...): the func, two
        // below the dict, is read for error messages.
        DICT_MERGE if n >= 1 => {
            let mut inputs = vec![In::Used, In::Unused, In::Used];
            inputs.extend(vec![In::Unused; n - 1]);
            inputs.push(In::Used);
            sig(inputs, (0..n + 2).map(Out::Keep).collect())
        }
        MAP_ADD => container_update(n, 2),
        FORMAT_VALUE => sig(used(if n & 0x04 != 0 { 2 } else { 1 }), new(1)),
        CALL => sig(used(n + 2), new(1)),
        _ => None,
    }
}

// Can-raise for opcodes lowered to the generic Op::Bytecode. MemoryError and
// asynchronous exceptions are not modeled.
pub fn generic_can_raise(op: u8) -> bool {
    !matches!(
        op,
        NOP | POP_TOP
            | PUSH_NULL
            | END_FOR
            | END_SEND
            | MATCH_MAPPING
            | MATCH_SEQUENCE
            | PUSH_EXC_INFO
            | POP_EXCEPT
            | LOAD_ASSERTION_ERROR
            | RETURN_GENERATOR
            | IS_OP
            | BUILD_LIST
            | BUILD_SLICE
            | BUILD_STRING
            | LIST_APPEND
    )
}
