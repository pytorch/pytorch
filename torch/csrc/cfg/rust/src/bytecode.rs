//! Decoding of CPython 3.12 bytecode.
//!
//! Opcode numbers and inline cache sizes are read from the running
//! interpreter's `opcode` module, so the decoder itself only knows opcode
//! names. The meaning of the opcodes (in frontend.rs) is that of CPython 3.12.

use pyo3::prelude::*;
use pyo3::types::PyBytes;

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
    LoadGlobal,
    StoreGlobal,
    LoadAttr,
    StoreAttr,
    LoadDeref,
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
    LoadAssertionError,
    FormatValue,
    Other,
}

fn bc_from_name(name: &str) -> Bc {
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
        "LOAD_GLOBAL" => Bc::LoadGlobal,
        "STORE_GLOBAL" => Bc::StoreGlobal,
        "LOAD_ATTR" => Bc::LoadAttr,
        "STORE_ATTR" => Bc::StoreAttr,
        "LOAD_DEREF" => Bc::LoadDeref,
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
        "CALL" => Bc::Call,
        "KW_NAMES" => Bc::KwNames,
        "CALL_FUNCTION_EX" => Bc::CallFunctionEx,
        "CALL_INTRINSIC_1" => Bc::CallIntrinsic1,
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
        "LOAD_ASSERTION_ERROR" => Bc::LoadAssertionError,
        "FORMAT_VALUE" => Bc::FormatValue,
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

pub struct OpcodeTable {
    pub kinds: Vec<Bc>,
    pub names: Vec<String>,
    pub caches: Vec<u32>,
}

impl OpcodeTable {
    pub fn load(py: Python<'_>) -> PyResult<OpcodeTable> {
        let version = py.version_info();
        if (version.major, version.minor) != (3, 12) {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
                "torch.cfg_compile only understands CPython 3.12 bytecode, not {}.{}",
                version.major, version.minor
            )));
        }
        let opcode = py.import_bound("opcode")?;
        let names: Vec<String> = opcode.getattr("opname")?.extract()?;
        let caches: Vec<u32> = opcode.getattr("_inline_cache_entries")?.extract()?;
        let kinds = names.iter().map(|n| bc_from_name(n)).collect();
        Ok(OpcodeTable { kinds, names, caches })
    }
}

#[derive(Clone, Debug)]
pub struct Ins {
    pub off: u32,
    pub op: Bc,
    pub code: u8,
    pub arg: u32,
    /// Byte offset of the jump target, for jumps.
    pub target: Option<u32>,
}

pub fn decode(table: &OpcodeTable, code: &Bound<'_, PyBytes>) -> Vec<Ins> {
    let bytes = code.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    let mut ext: u32 = 0;
    while i + 1 < bytes.len() {
        let opcode = bytes[i];
        let arg = (ext << 8) | bytes[i + 1] as u32;
        let kind = table.kinds[opcode as usize];
        if kind == Bc::ExtendedArg {
            ext = arg;
            i += 2;
            continue;
        }
        ext = 0;
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
            out.push(Ins { off: i as u32, op: kind, code: opcode, arg, target });
        }
        i = next;
    }
    out
}

pub struct ExceptionEntry {
    pub target: u32,
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
        let (Some(_start), Some(_length), Some(target), Some(_depth_lasti)) =
            (varint(&mut pos), varint(&mut pos), varint(&mut pos), varint(&mut pos))
        else {
            break;
        };
        entries.push(ExceptionEntry { target: target * 2 });
    }
    entries
}
