//! Decoding of CPython bytecode: 3.12 and 3.14.
//!
//! Opcode numbers and inline cache sizes are read from the running
//! interpreter's `opcode` module, so the decoder itself only knows opcode
//! names. A `Bc` means one thing (in frontend.rs) whichever version the
//! bytecode is of. Where 3.14 gave an instruction of 3.12 another meaning
//! under the same name, this is where it shows: the name stands for another
//! `Bc` (`bc_from_name`), the instruction is rewritten into what the frontend
//! knows (`simplify`), or the table says which of the two it is
//! (`callable_first`).

use crate::pyffi::PyObj;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bc {
    Cache,
    Resume,
    Nop,
    ExtendedArg,
    PopTop,
    PushNull,
    EndFor,
    Copy,
    Swap,
    LoadConst,
    LoadFast,
    LoadFastCheck,
    LoadFastAndClear,
    StoreFast,
    DeleteFast,
    GetLen,
    MatchSequence,
    MatchMapping,
    MatchKeys,
    MatchClass,
    DeleteAttr,
    DeleteGlobal,
    DeleteDeref,
    LoadGlobal,
    StoreGlobal,
    LoadAttr,
    LoadSuperAttr,
    ImportName,
    ImportFrom,
    StoreAttr,
    LoadDeref,
    StoreDeref,
    LoadClosure,
    MakeCell,
    MakeFunction,
    CopyFreeVars,
    BinaryOp,
    UnaryNegative,
    UnaryNot,
    UnaryInvert,
    CompareOp,
    IsOp,
    ContainsOp,
    BinarySubscr,
    StoreSubscr,
    DeleteSubscr,
    BinarySlice,
    StoreSlice,
    BuildTuple,
    BuildList,
    BuildSet,
    BuildMap,
    BuildConstKeyMap,
    BuildString,
    BuildSlice,
    ListExtend,
    ListAppend,
    SetAdd,
    MapAdd,
    SetUpdate,
    DictUpdate,
    DictMerge,
    UnpackSequence,
    UnpackEx,
    Call,
    KwNames,
    CallFunctionEx,
    CallIntrinsic1,
    GetIter,
    ForIter,
    JumpForward,
    JumpBackward,
    JumpBackwardNoInterrupt,
    PopJumpIfTrue,
    PopJumpIfFalse,
    PopJumpIfNone,
    PopJumpIfNotNone,
    ReturnValue,
    ReturnConst,
    RaiseVarargs,
    Reraise,
    PushExcInfo,
    PopExcept,
    CheckExcMatch,
    WithExceptStart,
    BeforeWith,
    LoadAssertionError,
    LoadBuildClass,
    FormatValue,
    // The rest is 3.14 only.
    ToBool,
    LoadSmallInt,
    LoadCommonConstant,
    CallKw,
    /// CALL_FUNCTION_EX of 3.14: the keyword arguments always have their
    /// place on the stack, which holds NULL if there are none.
    CallFunctionExKwargs,
    SetFunctionAttribute,
    ConvertValue,
    FormatSimple,
    FormatWithSpec,
    LoadSpecial,
    /// What the code of a generator has and no other: where it hands the
    /// generator to its caller, and where it gives a value.
    ReturnGenerator,
    YieldValue,
    /// WITH_EXCEPT_START of 3.14: `__exit__` is on the stack the way CALL
    /// finds a callable, with the place for `self` on top of it.
    WithExceptStartCall,
    Other,
}

fn bc_from_name(name: &str, v314: bool) -> Bc {
    if v314 {
        match name {
            // Pops what FOR_ITER pushed for an exhausted iterator, and
            // POP_ITER then pops the iterator; in 3.12 END_FOR pops both.
            "END_FOR" => return Bc::PopTop,
            "CALL_FUNCTION_EX" => return Bc::CallFunctionExKwargs,
            "WITH_EXCEPT_START" => return Bc::WithExceptStartCall,
            _ => {}
        }
    }
    match name {
        "CACHE" => Bc::Cache,
        "RESUME" => Bc::Resume,
        "NOP" => Bc::Nop,
        "EXTENDED_ARG" => Bc::ExtendedArg,
        "POP_TOP" => Bc::PopTop,
        "PUSH_NULL" => Bc::PushNull,
        "END_FOR" => Bc::EndFor,
        "COPY" => Bc::Copy,
        "SWAP" => Bc::Swap,
        "LOAD_CONST" => Bc::LoadConst,
        "LOAD_FAST" => Bc::LoadFast,
        "LOAD_FAST_CHECK" => Bc::LoadFastCheck,
        "LOAD_FAST_AND_CLEAR" => Bc::LoadFastAndClear,
        "STORE_FAST" => Bc::StoreFast,
        "DELETE_FAST" => Bc::DeleteFast,
        "GET_LEN" => Bc::GetLen,
        "MATCH_SEQUENCE" => Bc::MatchSequence,
        "MATCH_MAPPING" => Bc::MatchMapping,
        "MATCH_KEYS" => Bc::MatchKeys,
        "MATCH_CLASS" => Bc::MatchClass,
        "DELETE_ATTR" => Bc::DeleteAttr,
        "DELETE_GLOBAL" => Bc::DeleteGlobal,
        "DELETE_DEREF" => Bc::DeleteDeref,
        "LOAD_GLOBAL" => Bc::LoadGlobal,
        "STORE_GLOBAL" => Bc::StoreGlobal,
        "LOAD_ATTR" => Bc::LoadAttr,
        "STORE_ATTR" => Bc::StoreAttr,
        "LOAD_DEREF" => Bc::LoadDeref,
        "LOAD_SUPER_ATTR" => Bc::LoadSuperAttr,
        "IMPORT_NAME" => Bc::ImportName,
        "IMPORT_FROM" => Bc::ImportFrom,
        "STORE_DEREF" => Bc::StoreDeref,
        "LOAD_CLOSURE" => Bc::LoadClosure,
        "MAKE_CELL" => Bc::MakeCell,
        "MAKE_FUNCTION" => Bc::MakeFunction,
        "COPY_FREE_VARS" => Bc::CopyFreeVars,
        "BINARY_OP" => Bc::BinaryOp,
        "UNARY_NEGATIVE" => Bc::UnaryNegative,
        "UNARY_NOT" => Bc::UnaryNot,
        "UNARY_INVERT" => Bc::UnaryInvert,
        "COMPARE_OP" => Bc::CompareOp,
        "IS_OP" => Bc::IsOp,
        "CONTAINS_OP" => Bc::ContainsOp,
        "BINARY_SUBSCR" => Bc::BinarySubscr,
        "STORE_SUBSCR" => Bc::StoreSubscr,
        "DELETE_SUBSCR" => Bc::DeleteSubscr,
        "BINARY_SLICE" => Bc::BinarySlice,
        "STORE_SLICE" => Bc::StoreSlice,
        "BUILD_TUPLE" => Bc::BuildTuple,
        "BUILD_LIST" => Bc::BuildList,
        "BUILD_SET" => Bc::BuildSet,
        "BUILD_MAP" => Bc::BuildMap,
        "BUILD_CONST_KEY_MAP" => Bc::BuildConstKeyMap,
        "BUILD_STRING" => Bc::BuildString,
        "BUILD_SLICE" => Bc::BuildSlice,
        "LIST_EXTEND" => Bc::ListExtend,
        "LIST_APPEND" => Bc::ListAppend,
        "SET_ADD" => Bc::SetAdd,
        "MAP_ADD" => Bc::MapAdd,
        "SET_UPDATE" => Bc::SetUpdate,
        "DICT_UPDATE" => Bc::DictUpdate,
        "DICT_MERGE" => Bc::DictMerge,
        "UNPACK_SEQUENCE" => Bc::UnpackSequence,
        "UNPACK_EX" => Bc::UnpackEx,
        "CALL" => Bc::Call,
        "KW_NAMES" => Bc::KwNames,
        "CALL_FUNCTION_EX" => Bc::CallFunctionEx,
        "CALL_INTRINSIC_1" => Bc::CallIntrinsic1,
        "RETURN_GENERATOR" => Bc::ReturnGenerator,
        "YIELD_VALUE" => Bc::YieldValue,
        "GET_ITER" => Bc::GetIter,
        "FOR_ITER" => Bc::ForIter,
        "JUMP_FORWARD" => Bc::JumpForward,
        "JUMP_BACKWARD" => Bc::JumpBackward,
        "JUMP_BACKWARD_NO_INTERRUPT" => Bc::JumpBackwardNoInterrupt,
        "POP_JUMP_IF_TRUE" => Bc::PopJumpIfTrue,
        "POP_JUMP_IF_FALSE" => Bc::PopJumpIfFalse,
        "POP_JUMP_IF_NONE" => Bc::PopJumpIfNone,
        "POP_JUMP_IF_NOT_NONE" => Bc::PopJumpIfNotNone,
        "RETURN_VALUE" => Bc::ReturnValue,
        "RETURN_CONST" => Bc::ReturnConst,
        "RAISE_VARARGS" => Bc::RaiseVarargs,
        "RERAISE" => Bc::Reraise,
        "PUSH_EXC_INFO" => Bc::PushExcInfo,
        "POP_EXCEPT" => Bc::PopExcept,
        "CHECK_EXC_MATCH" => Bc::CheckExcMatch,
        "WITH_EXCEPT_START" => Bc::WithExceptStart,
        "BEFORE_WITH" => Bc::BeforeWith,
        "LOAD_ASSERTION_ERROR" => Bc::LoadAssertionError,
        "LOAD_BUILD_CLASS" => Bc::LoadBuildClass,
        "FORMAT_VALUE" => Bc::FormatValue,
        // A borrowed reference is the interpreter's business.
        "LOAD_FAST_BORROW" => Bc::LoadFast,
        "NOT_TAKEN" => Bc::Nop,
        "POP_ITER" => Bc::PopTop,
        "TO_BOOL" => Bc::ToBool,
        "LOAD_SMALL_INT" => Bc::LoadSmallInt,
        "LOAD_COMMON_CONSTANT" => Bc::LoadCommonConstant,
        "CALL_KW" => Bc::CallKw,
        "SET_FUNCTION_ATTRIBUTE" => Bc::SetFunctionAttribute,
        "CONVERT_VALUE" => Bc::ConvertValue,
        "FORMAT_SIMPLE" => Bc::FormatSimple,
        "FORMAT_WITH_SPEC" => Bc::FormatWithSpec,
        "LOAD_SPECIAL" => Bc::LoadSpecial,
        _ => Bc::Other,
    }
}

impl Bc {
    pub fn is_forward_jump(self) -> bool {
        matches!(
            self,
            Bc::JumpForward
                | Bc::PopJumpIfTrue
                | Bc::PopJumpIfFalse
                | Bc::PopJumpIfNone
                | Bc::PopJumpIfNotNone
                | Bc::ForIter
        )
    }

    pub fn is_backward_jump(self) -> bool {
        matches!(self, Bc::JumpBackward | Bc::JumpBackwardNoInterrupt)
    }

    pub fn is_conditional_jump(self) -> bool {
        matches!(
            self,
            Bc::PopJumpIfTrue
                | Bc::PopJumpIfFalse
                | Bc::PopJumpIfNone
                | Bc::PopJumpIfNotNone
                | Bc::ForIter
        )
    }

    /// Control never falls through to the next instruction.
    pub fn ends_flow(self) -> bool {
        matches!(
            self,
            Bc::JumpForward
                | Bc::JumpBackward
                | Bc::JumpBackwardNoInterrupt
                | Bc::ReturnValue
                | Bc::ReturnConst
                | Bc::RaiseVarargs
                | Bc::Reraise
        )
    }
}

/// The two instructions a superinstruction of 3.14 stands for. The high four
/// bits of its argument are the argument of the first, the low four that of
/// the second.
fn halves(name: &str) -> Option<(Bc, Bc)> {
    match name {
        "LOAD_FAST_LOAD_FAST" | "LOAD_FAST_BORROW_LOAD_FAST_BORROW" => Some((Bc::LoadFast, Bc::LoadFast)),
        "STORE_FAST_LOAD_FAST" => Some((Bc::StoreFast, Bc::LoadFast)),
        "STORE_FAST_STORE_FAST" => Some((Bc::StoreFast, Bc::StoreFast)),
        _ => None,
    }
}

pub struct OpcodeTable {
    pub kinds: Vec<Bc>,
    pub names: Vec<String>,
    pub caches: Vec<u32>,
    /// What a call finds under its arguments. In 3.12 it is the method or
    /// NULL, and then the callable or `self`; in 3.14 the callable, and then
    /// `self` or NULL. Whatever leaves a callable for a call (PUSH_NULL,
    /// LOAD_GLOBAL, LOAD_ATTR, LOAD_SUPER_ATTR) follows that order.
    pub callable_first: bool,
    /// The names LOAD_SPECIAL takes by index.
    pub special_names: Vec<String>,
    /// The objects LOAD_COMMON_CONSTANT takes by index.
    pub common_constants: Vec<PyObj>,
    halves: Vec<Option<(Bc, Bc)>>,
    /// The argument of BINARY_OP that is a subscript, in 3.14.
    subscr: Option<u32>,
    /// In 3.14 the comparison of COMPARE_OP is its argument shifted by 5, not
    /// 4, and the bit under it asks for the result as a bool.
    compare_as_bool: bool,
}

impl OpcodeTable {
    pub fn load(py: Python<'_>) -> PyResult<OpcodeTable> {
        let version = py.version_info();
        let v314 = match (version.major, version.minor) {
            (3, 12) => false,
            (3, 14) => true,
            _ => {
                return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
                    "torch.vm only understands CPython 3.12 and 3.14 bytecode, not {}.{}",
                    version.major, version.minor
                )))
            }
        };
        let opcode = py.import("opcode")?;
        let names: Vec<String> = opcode.getattr("opname")?.extract()?;
        // By opcode in 3.12, by name in 3.14.
        let caches = opcode.getattr("_inline_cache_entries")?;
        let caches: Vec<u32> = if v314 {
            let by_name: HashMap<String, u32> = caches.extract()?;
            names.iter().map(|n| by_name.get(n).copied().unwrap_or(0)).collect()
        } else {
            caches.extract()?
        };
        let binary: Vec<(String, String)> = opcode.getattr("_nb_ops")?.extract()?;
        let (special_names, common_constants) = if v314 {
            let constants = opcode.getattr("_common_constants")?;
            let constants = constants.try_iter()?.map(|c| Ok(unsafe { PyObj::from_borrowed(c?.as_ptr()) }));
            (opcode.getattr("_special_method_names")?.extract()?, constants.collect::<PyResult<_>>()?)
        } else {
            (Vec::new(), Vec::new())
        };
        Ok(OpcodeTable {
            kinds: names.iter().map(|n| bc_from_name(n, v314)).collect(),
            halves: names.iter().map(|n| halves(n.as_str())).collect(),
            names,
            caches,
            callable_first: v314,
            special_names,
            common_constants,
            subscr: binary.iter().position(|(name, _)| name == "NB_SUBSCR").map(|i| i as u32),
            compare_as_bool: v314,
        })
    }
}

#[derive(Clone, Debug)]
pub struct Ins {
    pub off: u32,
    /// Where the instruction starts, EXTENDED_ARG prefixes included. Jumps
    /// and the exception table can name that offset.
    pub start: u32,
    pub op: Bc,
    pub code: u8,
    pub arg: u32,
    /// Byte offset of the jump target, for jumps.
    pub target: Option<u32>,
}

/// The instructions as they are in the code object.
pub fn decode(table: &OpcodeTable, code: &Bound<'_, PyBytes>) -> Vec<Ins> {
    let bytes = code.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    let mut ext: u32 = 0;
    let mut prefix: Option<usize> = None;
    while i + 1 < bytes.len() {
        let opcode = bytes[i];
        let arg = (ext << 8) | bytes[i + 1] as u32;
        let kind = table.kinds[opcode as usize];
        if kind == Bc::ExtendedArg {
            ext = arg;
            prefix.get_or_insert(i);
            i += 2;
            continue;
        }
        ext = 0;
        let start = prefix.take().unwrap_or(i) as u32;
        let caches = table.caches[opcode as usize] as usize;
        let next = i + 2 + 2 * caches;
        let target = if kind.is_forward_jump() {
            Some((next + 2 * arg as usize) as u32)
        } else if kind.is_backward_jump() {
            Some((next - 2 * arg as usize) as u32)
        } else {
            None
        };
        if kind != Bc::Cache {
            out.push(Ins { off: i as u32, start, op: kind, code: opcode, arg, target });
        }
        i = next;
    }
    // A jump to a prefixed instruction names its first prefix.
    let prefixed: std::collections::HashMap<u32, u32> =
        out.iter().filter(|ins| ins.start != ins.off).map(|ins| (ins.start, ins.off)).collect();
    for ins in out.iter_mut() {
        if let Some(target) = ins.target.and_then(|t| prefixed.get(&t)) {
            ins.target = Some(*target);
        }
    }
    out
}

/// Rewrites the instructions of 3.14 that are another way to write what 3.12
/// has an instruction for, so that the frontend only knows the latter. Where
/// one instruction becomes two, the second gets the odd offset that follows
/// the instruction's own: no instruction is there, and it is in the same
/// line and under the same exception handler.
pub fn simplify(table: &OpcodeTable, decoded: Vec<Ins>) -> Vec<Ins> {
    let mut out = Vec::with_capacity(decoded.len());
    for mut ins in decoded {
        let (off, code) = (ins.off + 1, ins.code);
        let second = |op: Bc, arg: u32| Ins { off, start: off, op, code, arg, target: None };
        if let Some((first, then)) = table.halves[ins.code as usize] {
            let then = second(then, ins.arg & 15);
            ins.op = first;
            ins.arg >>= 4;
            out.extend([ins, then]);
        } else if ins.op == Bc::CompareOp && table.compare_as_bool {
            let as_bool = (ins.arg & 16 != 0).then(|| second(Bc::ToBool, 0));
            ins.arg = (ins.arg >> 5) << 4;
            out.push(ins);
            out.extend(as_bool);
        } else {
            if ins.op == Bc::BinaryOp && Some(ins.arg) == table.subscr {
                ins.op = Bc::BinarySubscr;
            }
            out.push(ins);
        }
    }
    out
}

/// One entry of a code object's exception table: an exception raised by an
/// instruction in [start, end) unwinds the stack to `depth`, pushes the offset
/// of the instruction if `lasti`, pushes the exception, and continues at
/// `target`. All offsets are in bytes.
pub struct ExceptionEntry {
    pub start: u32,
    pub end: u32,
    pub target: u32,
    pub depth: u32,
    pub lasti: bool,
}

pub fn parse_exception_table(table: &[u8]) -> Vec<ExceptionEntry> {
    let mut entries = Vec::new();
    let mut pos = 0usize;
    let varint = |pos: &mut usize| -> Option<u32> {
        let mut b = *table.get(*pos)?;
        *pos += 1;
        let mut val = (b & 63) as u32;
        while b & 64 != 0 {
            b = *table.get(*pos)?;
            *pos += 1;
            val = (val << 6) | (b & 63) as u32;
        }
        Some(val)
    };
    while pos < table.len() {
        // start, length, target, depth and lasti; offsets are in code units.
        let (Some(start), Some(length), Some(target), Some(depth_lasti)) =
            (varint(&mut pos), varint(&mut pos), varint(&mut pos), varint(&mut pos))
        else {
            break;
        };
        entries.push(ExceptionEntry {
            start: start * 2,
            end: (start + length) * 2,
            target: target * 2,
            depth: depth_lasti >> 1,
            lasti: depth_lasti & 1 != 0,
        });
    }
    entries
}
