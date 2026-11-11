//! Python bytecode -> control-flow graph -> SSA.
//!
//! The bytecode is split into basic blocks, and each block is translated by
//! abstract interpretation of the operand stack. Locals and the stack slots
//! that are live across block boundaries are turned into SSA values with the
//! on-the-fly construction of Braun et al. ("Simple and Efficient
//! Construction of Static Single Assignment Form"), so the whole function,
//! with all of its branches and loops, becomes one graph.

use crate::bytecode::{self, Bc, Ins, OpcodeTable};
use crate::ir::*;
use crate::pyffi::PyObj;
use pyo3::exceptions::PyNotImplementedError;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyTuple};
use std::collections::{HashMap, HashSet};

/// Marks the NULL that PUSH_NULL and LOAD_GLOBAL leave under a callable. It
/// only lives on the abstract stack.
const NULLMARK: V = u32::MAX - 1;

const CO_VARARGS: u32 = 0x04;
const CO_VARKEYWORDS: u32 = 0x08;
const CO_GENERATOR_LIKE: u32 = 0x20 | 0x80 | 0x100 | 0x200;

fn unsupported<T>(msg: impl Into<String>) -> PyResult<T> {
    Err(PyNotImplementedError::new_err(msg.into()))
}

#[derive(Clone)]
struct RawBlock {
    start: usize,
    end: usize,
    succs: Vec<usize>,
    preds: Vec<usize>,
}

struct Builder<'a> {
    instrs: &'a [Ins],
    table: &'a OpcodeTable,
    cfg: Vec<RawBlock>,
    block_of_off: HashMap<u32, usize>,
    nlocals: usize,
    names_base: u32,
    nfree_base: u32,
    varnames: Vec<String>,
    lines: &'a [(u32, u32, i32)],

    // SSA construction state
    defs: Vec<HashMap<u32, V>>,
    sealed: Vec<bool>,
    filled: Vec<bool>,
    incomplete: Vec<Vec<(u32, V)>>,
    entry_depth: Vec<Option<usize>>,
    /// The values each block found on the stack when it was translated.
    entry_stacks: Vec<Vec<V>>,
    /// Per block and stack slot: the slot holds the NULL marker. A slot of a
    /// loop header is read before the loop is complete, so whether it is NULL
    /// is only known once construction is done; a second run uses the answer.
    null_slots: Vec<Vec<bool>>,
    forward: Vec<V>,
    phi_block: HashMap<V, usize>,
    phi_ops: HashMap<V, Vec<V>>,
    phi_users: HashMap<V, Vec<V>>,
    phi_order: Vec<V>,
    removed_phis: HashSet<V>,
    value_names: Vec<Option<String>>,
    bool_values: HashSet<V>,

    out_insts: Vec<Vec<Inst>>,
    out_terms: Vec<Option<Term>>,
    extra_consts: Vec<PyObj>,
    nconsts: u32,
}

impl<'a> Builder<'a> {
    fn new_value(&mut self) -> V {
        let v = self.forward.len() as V;
        self.forward.push(v);
        self.value_names.push(None);
        v
    }

    fn resolve(&self, mut v: V) -> V {
        if v >= NULLMARK {
            return v;
        }
        while self.forward[v as usize] != v {
            v = self.forward[v as usize];
            if v >= NULLMARK {
                return v;
            }
        }
        v
    }

    fn new_phi(&mut self, block: usize, var: u32) -> V {
        let v = self.new_value();
        self.value_names[v as usize] = self.varnames.get(var as usize).cloned();
        self.phi_block.insert(v, block);
        self.phi_ops.insert(v, Vec::new());
        self.phi_order.push(v);
        v
    }

    fn write_var(&mut self, var: u32, block: usize, v: V) {
        self.defs[block].insert(var, v);
    }

    fn read_var(&mut self, var: u32, block: usize) -> V {
        if let Some(v) = self.defs[block].get(&var) {
            return self.resolve(*v);
        }
        let v = if !self.sealed[block] {
            let phi = self.new_phi(block, var);
            self.incomplete[block].push((var, phi));
            phi
        } else if self.cfg[block].preds.len() == 1 {
            let pred = self.cfg[block].preds[0];
            self.read_var(var, pred)
        } else if self.cfg[block].preds.is_empty() {
            UNDEF
        } else {
            let phi = self.new_phi(block, var);
            self.write_var(var, block, phi);
            self.add_phi_operands(var, phi)
        };
        self.write_var(var, block, v);
        v
    }

    fn add_phi_operands(&mut self, var: u32, phi: V) -> V {
        let block = self.phi_block[&phi];
        let preds = self.cfg[block].preds.clone();
        for pred in preds {
            let v = self.read_var(var, pred);
            self.phi_ops.get_mut(&phi).unwrap().push(v);
            if self.phi_block.contains_key(&v) {
                self.phi_users.entry(v).or_default().push(phi);
            }
        }
        self.try_remove_trivial_phi(phi)
    }

    fn try_remove_trivial_phi(&mut self, phi: V) -> V {
        let mut same = NONE;
        let ops = self.phi_ops[&phi].clone();
        for op in ops {
            let op = self.resolve(op);
            if op == same || op == phi {
                continue;
            }
            if same != NONE {
                return phi;
            }
            same = op;
        }
        if same == NONE {
            same = UNDEF;
        }
        self.forward[phi as usize] = same;
        self.removed_phis.insert(phi);
        let users = self.phi_users.remove(&phi).unwrap_or_default();
        if same < NULLMARK && self.phi_block.contains_key(&same) {
            // The users of the removed phi now use `same`.
            let entry = self.phi_users.entry(same).or_default();
            entry.extend(users.iter().copied().filter(|u| *u != phi));
        }
        for user in users {
            if user != phi && !self.removed_phis.contains(&user) {
                self.try_remove_trivial_phi(user);
            }
        }
        self.resolve(same)
    }

    fn seal(&mut self, block: usize) {
        if self.sealed[block] {
            return;
        }
        let pending = std::mem::take(&mut self.incomplete[block]);
        self.sealed[block] = true;
        for (var, phi) in pending {
            self.add_phi_operands(var, phi);
        }
    }

    fn line_of(&self, off: u32) -> i32 {
        for (start, end, line) in self.lines {
            if off >= *start && off < *end {
                return *line;
            }
        }
        0
    }

    fn extra_const(&mut self, obj: PyObj) -> u32 {
        self.extra_consts.push(obj);
        self.nconsts + self.extra_consts.len() as u32 - 1
    }

    fn translate_block(&mut self, b: usize) -> PyResult<()> {
        let depth = self.entry_depth[b].unwrap_or(0);
        let mut stack: Vec<V> = Vec::with_capacity(depth + 8);
        for i in 0..depth {
            let v = if self.null_slots[b].get(i).copied().unwrap_or(false) {
                NULLMARK
            } else {
                self.read_var((self.nlocals + i) as u32, b)
            };
            stack.push(v);
        }
        self.entry_stacks[b] = stack.clone();
        let mut insts: Vec<Inst> = Vec::new();
        let mut kwnames: u32 = NONE;
        let mut term: Option<Term> = None;
        let (start, end) = (self.cfg[b].start, self.cfg[b].end);

        macro_rules! pop {
            () => {
                match stack.pop() {
                    Some(v) => v,
                    None => return unsupported("bytecode underflows the stack"),
                }
            };
        }
        macro_rules! pop_value {
            () => {{
                let v = pop!();
                if v == NULLMARK {
                    return unsupported("unexpected NULL on the stack");
                }
                v
            }};
        }
        macro_rules! peek {
            ($n:expr) => {{
                let n: usize = $n;
                if n == 0 || n > stack.len() {
                    return unsupported("bytecode underflows the stack");
                }
                stack[stack.len() - n]
            }};
        }
        macro_rules! emit {
            ($off:expr, $op:expr) => {{
                let dst = self.new_value();
                insts.push(Inst { dst, op: $op, bc: $off, kills: Vec::new() });
                dst
            }};
        }
        macro_rules! effect {
            ($off:expr, $op:expr) => {
                insts.push(Inst { dst: NONE, op: $op, bc: $off, kills: Vec::new() })
            };
        }
        macro_rules! pop_n {
            ($n:expr) => {{
                let n: usize = $n;
                if n > stack.len() {
                    return unsupported("bytecode underflows the stack");
                }
                let vs: Vec<V> = stack.split_off(stack.len() - n);
                if vs.iter().any(|v| *v == NULLMARK) {
                    return unsupported("unexpected NULL on the stack");
                }
                vs
            }};
        }

        for idx in start..end {
            let ins = self.instrs[idx].clone();
            let off = ins.off;
            let arg = ins.arg;
            match ins.op {
                Bc::Cache | Bc::Resume | Bc::Nop | Bc::CopyFreeVars | Bc::ExtendedArg => {}
                Bc::PopTop => {
                    pop!();
                }
                Bc::PushNull => stack.push(NULLMARK),
                Bc::EndFor => {
                    pop!();
                    pop!();
                }
                Bc::Copy => {
                    let v = peek!(arg as usize);
                    stack.push(v);
                }
                Bc::Swap => {
                    let n = arg as usize;
                    if n == 0 || n > stack.len() {
                        return unsupported("bytecode underflows the stack");
                    }
                    let (top, other) = (stack.len() - 1, stack.len() - n);
                    stack.swap(top, other);
                }
                Bc::LoadConst => {
                    let v = emit!(off, Op::Const(arg));
                    stack.push(v);
                }
                Bc::LoadFast | Bc::LoadFastCheck => {
                    let v = self.read_var(arg, b);
                    if ins.op == Bc::LoadFastCheck || v == UNDEF {
                        let name = self.varname_const(arg)?;
                        effect!(off, Op::CheckBound(v, name));
                    }
                    stack.push(v);
                }
                Bc::LoadFastAndClear => {
                    let v = self.read_var(arg, b);
                    stack.push(v);
                    self.write_var(arg, b, UNDEF);
                }
                Bc::StoreFast => {
                    let v = pop!();
                    if v == NULLMARK {
                        return unsupported("unexpected NULL on the stack");
                    }
                    if v != UNDEF && self.value_names[v as usize].is_none() {
                        self.value_names[v as usize] = self.varnames.get(arg as usize).cloned();
                    }
                    self.write_var(arg, b, v);
                }
                Bc::DeleteFast => {
                    let v = self.read_var(arg, b);
                    let name = self.varname_const(arg)?;
                    effect!(off, Op::CheckBound(v, name));
                    self.write_var(arg, b, UNDEF);
                }
                Bc::LoadGlobal => {
                    if arg & 1 != 0 {
                        stack.push(NULLMARK);
                    }
                    let v = emit!(off, Op::LoadGlobal(self.names_base + (arg >> 1)));
                    stack.push(v);
                }
                Bc::StoreGlobal => {
                    let v = pop_value!();
                    effect!(off, Op::StoreGlobal(self.names_base + arg, v));
                }
                Bc::LoadAttr => {
                    let obj = pop_value!();
                    if arg & 1 != 0 {
                        // The method form leaves (method, self) or (NULL, attr)
                        // on the stack; we always produce the latter, binding
                        // the method.
                        stack.push(NULLMARK);
                    }
                    let v = emit!(off, Op::LoadAttr(obj, self.names_base + (arg >> 1)));
                    stack.push(v);
                }
                Bc::StoreAttr => {
                    let obj = pop_value!();
                    let val = pop_value!();
                    effect!(off, Op::StoreAttr { obj, name: self.names_base + arg, val });
                }
                Bc::LoadDeref => {
                    if arg < self.nfree_base {
                        return unsupported("functions that create closures are not supported");
                    }
                    let v = emit!(off, Op::LoadDeref(arg - self.nfree_base));
                    stack.push(v);
                }
                Bc::BinaryOp => {
                    let rhs = pop_value!();
                    let lhs = pop_value!();
                    if arg as usize >= BINARY_SYMBOLS.len() {
                        return unsupported("unknown binary operator");
                    }
                    let v = emit!(off, Op::Binary(arg as u8, lhs, rhs));
                    stack.push(v);
                }
                Bc::UnaryNegative | Bc::UnaryInvert | Bc::UnaryNot => {
                    let a = pop_value!();
                    let kind = match ins.op {
                        Bc::UnaryNegative => UnaryKind::Neg,
                        Bc::UnaryInvert => UnaryKind::Invert,
                        _ => UnaryKind::Not,
                    };
                    let v = emit!(off, Op::Unary(kind, a));
                    if kind == UnaryKind::Not {
                        self.bool_values.insert(v);
                    }
                    stack.push(v);
                }
                Bc::CompareOp => {
                    let rhs = pop_value!();
                    let lhs = pop_value!();
                    let v = emit!(off, Op::Compare((arg >> 4) as u8, lhs, rhs));
                    stack.push(v);
                }
                Bc::IsOp => {
                    let rhs = pop_value!();
                    let lhs = pop_value!();
                    let v = emit!(off, Op::Is(lhs, rhs, arg != 0));
                    self.bool_values.insert(v);
                    stack.push(v);
                }
                Bc::ContainsOp => {
                    let container = pop_value!();
                    let item = pop_value!();
                    let v = emit!(off, Op::Contains(item, container, arg != 0));
                    self.bool_values.insert(v);
                    stack.push(v);
                }
                Bc::BinarySubscr => {
                    let key = pop_value!();
                    let obj = pop_value!();
                    let v = emit!(off, Op::GetItem(obj, key));
                    stack.push(v);
                }
                Bc::StoreSubscr => {
                    let key = pop_value!();
                    let obj = pop_value!();
                    let val = pop_value!();
                    effect!(off, Op::SetItem { obj, key, val });
                }
                Bc::DeleteSubscr => {
                    let key = pop_value!();
                    let obj = pop_value!();
                    effect!(off, Op::DelItem(obj, key));
                }
                Bc::BinarySlice => {
                    let stop = pop_value!();
                    let start_ = pop_value!();
                    let obj = pop_value!();
                    let slice = emit!(off, Op::BuildSlice(start_, stop, NONE));
                    let v = emit!(off, Op::GetItem(obj, slice));
                    stack.push(v);
                }
                Bc::StoreSlice => {
                    let stop = pop_value!();
                    let start_ = pop_value!();
                    let obj = pop_value!();
                    let val = pop_value!();
                    let slice = emit!(off, Op::BuildSlice(start_, stop, NONE));
                    effect!(off, Op::SetItem { obj, key: slice, val });
                }
                Bc::BuildTuple => {
                    let vs = pop_n!(arg as usize);
                    let v = emit!(off, Op::BuildTuple(vs));
                    stack.push(v);
                }
                Bc::BuildList => {
                    let vs = pop_n!(arg as usize);
                    let v = emit!(off, Op::BuildList(vs));
                    stack.push(v);
                }
                Bc::BuildSet => {
                    let vs = pop_n!(arg as usize);
                    let v = emit!(off, Op::BuildSet(vs));
                    stack.push(v);
                }
                Bc::BuildMap => {
                    let vs = pop_n!(2 * arg as usize);
                    let v = emit!(off, Op::BuildMap(vs));
                    stack.push(v);
                }
                Bc::BuildConstKeyMap => {
                    let keys = pop_value!();
                    let vals = pop_n!(arg as usize);
                    let mut kvs = Vec::with_capacity(2 * vals.len());
                    for (i, val) in vals.into_iter().enumerate() {
                        let key = emit!(off, Op::TupleGet(keys, i as u32));
                        kvs.push(key);
                        kvs.push(val);
                    }
                    let v = emit!(off, Op::BuildMap(kvs));
                    stack.push(v);
                }
                Bc::BuildString => {
                    let vs = pop_n!(arg as usize);
                    let v = emit!(off, Op::BuildString(vs));
                    stack.push(v);
                }
                Bc::BuildSlice => {
                    let step = if arg == 3 { pop_value!() } else { NONE };
                    let stop = pop_value!();
                    let start_ = pop_value!();
                    let v = emit!(off, Op::BuildSlice(start_, stop, step));
                    stack.push(v);
                }
                Bc::ListExtend | Bc::ListAppend | Bc::SetAdd | Bc::SetUpdate | Bc::DictUpdate
                | Bc::DictMerge => {
                    let item = pop_value!();
                    let container = peek!(arg as usize);
                    let op = match ins.op {
                        Bc::ListExtend => Op::ListExtend(container, item),
                        Bc::ListAppend => Op::ListAppend(container, item),
                        Bc::SetAdd => Op::SetAdd(container, item),
                        Bc::SetUpdate => Op::SetUpdate(container, item),
                        Bc::DictUpdate => Op::DictUpdate(container, item),
                        _ => Op::DictMerge(container, item),
                    };
                    effect!(off, op);
                }
                Bc::MapAdd => {
                    let val = pop_value!();
                    let key = pop_value!();
                    let container = peek!(arg as usize);
                    effect!(off, Op::MapAdd(container, key, val));
                }
                Bc::UnpackSequence => {
                    let seq = pop_value!();
                    let tuple = emit!(off, Op::Unpack(seq, arg));
                    for i in (0..arg).rev() {
                        let v = emit!(off, Op::TupleGet(tuple, i));
                        stack.push(v);
                    }
                }
                Bc::KwNames => kwnames = arg,
                Bc::Call => {
                    let mut args = pop_n!(arg as usize);
                    let second = pop_value!();
                    let first = pop!();
                    let callee = if first == NULLMARK {
                        second
                    } else {
                        args.insert(0, second);
                        first
                    };
                    let v = emit!(off, Op::Call { callee, args, kwnames });
                    kwnames = NONE;
                    stack.push(v);
                }
                Bc::CallFunctionEx => {
                    let kwargs = if arg & 1 != 0 { pop_value!() } else { NONE };
                    let args = pop_value!();
                    let callee = pop_value!();
                    if pop!() != NULLMARK {
                        return unsupported("unexpected CALL_FUNCTION_EX stack layout");
                    }
                    let v = emit!(off, Op::CallEx { callee, args, kwargs });
                    stack.push(v);
                }
                Bc::CallIntrinsic1 => {
                    let a = pop_value!();
                    let v = match arg {
                        5 => emit!(off, Op::Unary(UnaryKind::Pos, a)),
                        6 => emit!(off, Op::ListToTuple(a)),
                        _ => return unsupported(format!("intrinsic {} is not supported", arg)),
                    };
                    stack.push(v);
                }
                Bc::GetIter => {
                    let a = pop_value!();
                    let v = emit!(off, Op::GetIter(a));
                    stack.push(v);
                }
                Bc::ForIter => {
                    let iter = peek!(1);
                    let v = emit!(off, Op::IterNext(iter));
                    stack.push(v);
                    term = Some(Term::BrNull {
                        v,
                        null: self.block_at(ins.target.unwrap())? as B,
                        nonnull: self.block_at_index(idx + 1)? as B,
                    });
                }
                Bc::JumpForward | Bc::JumpBackward | Bc::JumpBackwardNoInterrupt => {
                    term = Some(Term::Jump(self.block_at(ins.target.unwrap())? as B));
                }
                Bc::PopJumpIfTrue | Bc::PopJumpIfFalse => {
                    let c = pop_value!();
                    let cond = if self.bool_values.contains(&c) {
                        c
                    } else {
                        emit!(off, Op::ToBool(c))
                    };
                    let target = self.block_at(ins.target.unwrap())? as B;
                    let fall = self.block_at_index(idx + 1)? as B;
                    term = Some(if ins.op == Bc::PopJumpIfTrue {
                        Term::Br { cond, then_: target, else_: fall }
                    } else {
                        Term::Br { cond, then_: fall, else_: target }
                    });
                }
                Bc::PopJumpIfNone | Bc::PopJumpIfNotNone => {
                    let c = pop_value!();
                    let cond = emit!(off, Op::IsNone(c, false));
                    let target = self.block_at(ins.target.unwrap())? as B;
                    let fall = self.block_at_index(idx + 1)? as B;
                    term = Some(if ins.op == Bc::PopJumpIfNone {
                        Term::Br { cond, then_: target, else_: fall }
                    } else {
                        Term::Br { cond, then_: fall, else_: target }
                    });
                }
                Bc::ReturnValue => {
                    let v = pop_value!();
                    term = Some(Term::Return(v));
                }
                Bc::ReturnConst => {
                    let v = emit!(off, Op::Const(arg));
                    term = Some(Term::Return(v));
                }
                Bc::RaiseVarargs => {
                    let (exc, cause) = match arg {
                        1 => (pop_value!(), NONE),
                        2 => {
                            let cause = pop_value!();
                            (pop_value!(), cause)
                        }
                        _ => return unsupported("bare raise is not supported"),
                    };
                    term = Some(Term::Raise { exc, cause });
                }
                Bc::LoadAssertionError => {
                    let obj = unsafe { PyObj::from_borrowed(ffi::PyExc_AssertionError) };
                    let c = self.extra_const(obj);
                    let v = emit!(off, Op::Const(c));
                    stack.push(v);
                }
                Bc::FormatValue => {
                    let spec = if arg & 4 != 0 { pop_value!() } else { NONE };
                    let val = pop_value!();
                    let v = emit!(off, Op::Format { val, conv: (arg & 3) as u8, spec });
                    stack.push(v);
                }
                Bc::Reraise | Bc::Other => {
                    return unsupported(format!(
                        "bytecode {} (line {}) is not supported",
                        self.table.names[ins.code as usize],
                        self.line_of(off)
                    ));
                }
            }
        }

        let term = match term {
            Some(t) => t,
            None => Term::Jump(self.block_at_index(end)? as B),
        };
        for (i, v) in stack.iter().enumerate() {
            self.write_var((self.nlocals + i) as u32, b, *v);
        }
        for s in term.successors() {
            let s = s as usize;
            match self.entry_depth[s] {
                None => self.entry_depth[s] = Some(stack.len()),
                Some(d) if d != stack.len() => {
                    return unsupported("inconsistent stack depth at a join point");
                }
                _ => {}
            }
        }
        self.out_insts[b] = insts;
        self.out_terms[b] = Some(term);
        Ok(())
    }

    fn varname_const(&mut self, local: u32) -> PyResult<u32> {
        let name = self.varnames.get(local as usize).cloned().unwrap_or_default();
        let obj = unsafe {
            let s = ffi::PyUnicode_FromStringAndSize(name.as_ptr() as *const _, name.len() as isize);
            if s.is_null() {
                return Err(Python::with_gil(PyErr::fetch));
            }
            PyObj::from_owned(s)
        };
        Ok(self.extra_const(obj))
    }

    fn block_at(&self, off: u32) -> PyResult<usize> {
        match self.block_of_off.get(&off) {
            Some(b) => Ok(*b),
            None => unsupported("jump into unreachable code"),
        }
    }

    fn block_at_index(&self, idx: usize) -> PyResult<usize> {
        match self.instrs.get(idx) {
            Some(ins) => self.block_at(ins.off),
            None => unsupported("control falls off the end of the bytecode"),
        }
    }
}

/// A handler that only restores a local and re-raises (the cleanup of an
/// inlined comprehension) has no effect once the exception leaves the
/// function, which is what happens to every exception in compiled code.
fn handler_is_transparent(instrs: &[Ins], index_of_off: &HashMap<u32, usize>, target: u32) -> bool {
    let Some(&start) = index_of_off.get(&target) else {
        return false;
    };
    for ins in &instrs[start..] {
        match ins.op {
            Bc::Swap | Bc::PopTop | Bc::StoreFast | Bc::Copy => {}
            Bc::Reraise => return true,
            _ => return false,
        }
    }
    false
}

pub fn build(py: Python<'_>, table: &OpcodeTable, func: &Bound<'_, PyAny>) -> PyResult<Function> {
    let code = func.getattr("__code__")?;
    let flags: u32 = code.getattr("co_flags")?.extract()?;
    if flags & CO_GENERATOR_LIKE != 0 {
        return unsupported("generators and coroutines are not supported");
    }
    let cellvars: Vec<String> = code.getattr("co_cellvars")?.extract()?;
    if !cellvars.is_empty() {
        return unsupported("functions that create closures are not supported");
    }
    let co_code = code.getattr("co_code")?;
    let co_code = co_code.downcast::<PyBytes>()?;
    let instrs = bytecode::decode(table, co_code);
    if instrs.is_empty() {
        return unsupported("empty bytecode");
    }
    let index_of_off: HashMap<u32, usize> =
        instrs.iter().enumerate().map(|(i, ins)| (ins.off, i)).collect();

    let exc_table = code.getattr("co_exceptiontable")?;
    let exc_table = exc_table.downcast::<PyBytes>()?;
    for entry in bytecode::parse_exception_table(exc_table.as_bytes()) {
        if !handler_is_transparent(&instrs, &index_of_off, entry.target) {
            return unsupported("try/except/finally and with blocks are not supported");
        }
    }

    let mut lines: Vec<(u32, u32, i32)> = Vec::new();
    for item in code.call_method0("co_lines")?.iter()? {
        let (start, end, line): (u32, u32, Option<i32>) = item?.extract()?;
        if let Some(line) = line {
            lines.push((start, end, line));
        }
    }

    // ---- basic blocks ----
    let mut leaders: HashSet<usize> = HashSet::new();
    leaders.insert(0);
    for (i, ins) in instrs.iter().enumerate() {
        if let Some(target) = ins.target {
            match index_of_off.get(&target) {
                Some(t) => {
                    leaders.insert(*t);
                }
                None => return unsupported("jump target is not an instruction"),
            }
        }
        if (ins.op.ends_flow() || ins.op.is_conditional_jump()) && i + 1 < instrs.len() {
            leaders.insert(i + 1);
        }
    }
    let mut starts: Vec<usize> = leaders.into_iter().collect();
    starts.sort_unstable();
    let raw_of_start: HashMap<usize, usize> =
        starts.iter().enumerate().map(|(b, s)| (*s, b)).collect();
    let mut raw: Vec<RawBlock> = Vec::new();
    for (b, start) in starts.iter().enumerate() {
        let end = if b + 1 < starts.len() { starts[b + 1] } else { instrs.len() };
        let last = &instrs[end - 1];
        let mut succs = Vec::new();
        let target = last.target.map(|t| raw_of_start[&index_of_off[&t]]);
        if last.op == Bc::ForIter {
            succs.push(raw_of_start.get(&end).copied());
            succs.push(target);
        } else if last.op.is_conditional_jump() {
            succs.push(target);
            succs.push(raw_of_start.get(&end).copied());
        } else if last.op.ends_flow() {
            if target.is_some() {
                succs.push(target);
            }
        } else {
            succs.push(raw_of_start.get(&end).copied());
        }
        let mut out = Vec::new();
        for s in succs {
            match s {
                Some(s) => {
                    if !out.contains(&s) {
                        out.push(s)
                    }
                }
                None => return unsupported("control falls off the end of the bytecode"),
            }
        }
        raw.push(RawBlock { start: *start, end, succs: out, preds: Vec::new() });
    }

    // Reverse postorder over the reachable blocks; this is also the block
    // numbering of the function.
    let mut order: Vec<usize> = Vec::new();
    let mut state = vec![0u8; raw.len()];
    let mut work: Vec<(usize, usize)> = vec![(0, 0)];
    state[0] = 1;
    while let Some((b, i)) = work.pop() {
        if i < raw[b].succs.len() {
            work.push((b, i + 1));
            let s = raw[b].succs[i];
            if state[s] == 0 {
                state[s] = 1;
                work.push((s, 0));
            }
        } else {
            order.push(b);
        }
    }
    order.reverse();
    let mut new_id = vec![usize::MAX; raw.len()];
    for (i, b) in order.iter().enumerate() {
        new_id[*b] = i;
    }
    let mut cfg: Vec<RawBlock> = order
        .iter()
        .map(|b| RawBlock {
            start: raw[*b].start,
            end: raw[*b].end,
            succs: raw[*b].succs.iter().map(|s| new_id[*s]).collect(),
            preds: Vec::new(),
        })
        .collect();
    for b in 0..cfg.len() {
        for s in cfg[b].succs.clone() {
            if !cfg[s].preds.contains(&b) {
                cfg[s].preds.push(b);
            }
        }
    }
    if !cfg[0].preds.is_empty() {
        return unsupported("the entry block is a jump target");
    }
    let block_of_off: HashMap<u32, usize> =
        cfg.iter().enumerate().map(|(b, blk)| (instrs[blk.start].off, b)).collect();

    // ---- function metadata ----
    let consts_tuple = code.getattr("co_consts")?;
    let consts_tuple = consts_tuple.downcast::<PyTuple>()?;
    let names_tuple = code.getattr("co_names")?;
    let names_tuple = names_tuple.downcast::<PyTuple>()?;
    let mut consts: Vec<PyObj> = Vec::new();
    for c in consts_tuple.iter() {
        consts.push(unsafe { PyObj::from_borrowed(c.as_ptr()) });
    }
    let names_base = consts.len() as u32;
    for n in names_tuple.iter() {
        consts.push(unsafe { PyObj::from_borrowed(n.as_ptr()) });
    }
    let varnames_tuple = code.getattr("co_varnames")?;
    let varnames_tuple = varnames_tuple.downcast::<PyTuple>()?;
    let varnames: Vec<String> = varnames_tuple.extract()?;
    let argcount: usize = code.getattr("co_argcount")?.extract()?;
    let posonlyargcount: usize = code.getattr("co_posonlyargcount")?.extract()?;
    let kwonlyargcount: usize = code.getattr("co_kwonlyargcount")?.extract()?;
    let has_varargs = flags & CO_VARARGS != 0;
    let has_varkw = flags & CO_VARKEYWORDS != 0;
    let nparams = argcount + kwonlyargcount + has_varargs as usize + has_varkw as usize;
    let nlocals = varnames.len();

    let nblocks = cfg.len();
    let nconsts = consts.len() as u32;
    let mut null_slots: Vec<Vec<bool>> = vec![Vec::new(); nblocks];
    let mut attempts = 0;
    let (mut builder, params) = loop {
        let mut builder = Builder {
            instrs: &instrs,
            table,
            cfg: cfg.clone(),
            block_of_off: block_of_off.clone(),
            nlocals,
            names_base,
            nfree_base: nlocals as u32,
            varnames: varnames.clone(),
            lines: &lines,
            defs: (0..nblocks).map(|_| HashMap::new()).collect(),
            sealed: vec![false; nblocks],
            filled: vec![false; nblocks],
            incomplete: (0..nblocks).map(|_| Vec::new()).collect(),
            entry_depth: vec![None; nblocks],
            entry_stacks: vec![Vec::new(); nblocks],
            null_slots: null_slots.clone(),
            forward: Vec::new(),
            phi_block: HashMap::new(),
            phi_ops: HashMap::new(),
            phi_users: HashMap::new(),
            phi_order: Vec::new(),
            removed_phis: HashSet::new(),
            value_names: Vec::new(),
            bool_values: HashSet::new(),
            out_insts: (0..nblocks).map(|_| Vec::new()).collect(),
            out_terms: (0..nblocks).map(|_| None).collect(),
            extra_consts: Vec::new(),
            nconsts,
        };
        let undef = builder.new_value();
        debug_assert_eq!(undef, UNDEF);
        let mut params = Vec::with_capacity(nparams);
        for i in 0..nlocals {
            if i < nparams {
                let v = builder.new_value();
                builder.value_names[v as usize] = Some(varnames[i].clone());
                builder.write_var(i as u32, 0, v);
                params.push(v);
            } else {
                builder.write_var(i as u32, 0, UNDEF);
            }
        }
        builder.entry_depth[0] = Some(0);
        builder.sealed[0] = true;
        for b in 0..nblocks {
            if builder.entry_depth[b].is_none() {
                return unsupported("could not determine the stack depth of a block");
            }
            builder.translate_block(b)?;
            builder.filled[b] = true;
            for s in builder.cfg[b].succs.clone() {
                if builder.cfg[s].preds.iter().all(|p| builder.filled[*p]) {
                    builder.seal(s);
                }
            }
        }
        for b in 0..nblocks {
            builder.seal(b);
        }
        let mut learned = false;
        for b in 0..nblocks {
            let slots: Vec<bool> = builder.entry_stacks[b]
                .iter()
                .map(|v| builder.resolve(*v) == NULLMARK)
                .collect();
            learned |= slots
                .iter()
                .zip(&builder.entry_stacks[b])
                .any(|(is_null, v)| *is_null && *v != NULLMARK);
            null_slots[b] = slots;
        }
        if !learned {
            break (builder, params);
        }
        attempts += 1;
        if attempts > 2 {
            return unsupported("could not resolve the NULL stack markers");
        }
    };

    // ---- assemble ----
    let resolve_checked = |builder: &Builder, v: V| -> PyResult<V> {
        if v == NONE {
            return Ok(NONE);
        }
        let r = builder.resolve(v);
        if r == NULLMARK {
            return unsupported("a NULL stack marker reached an instruction");
        }
        Ok(r)
    };
    let mut blocks: Vec<Block> = Vec::with_capacity(nblocks);
    for b in 0..nblocks {
        let mut insts = std::mem::take(&mut builder.out_insts[b]);
        for inst in insts.iter_mut() {
            rewrite_uses(&mut inst.op, &mut |v| resolve_checked(&builder, v))?;
        }
        let mut term = builder.out_terms[b].take().unwrap();
        rewrite_term_uses(&mut term, &mut |v| resolve_checked(&builder, v))?;
        blocks.push(Block {
            phis: Vec::new(),
            insts,
            term,
            preds: builder.cfg[b].preds.iter().map(|p| *p as B).collect(),
            bc_start: instrs[builder.cfg[b].start].off,
        });
    }
    for phi in &builder.phi_order {
        if builder.removed_phis.contains(phi) {
            continue;
        }
        let block = builder.phi_block[phi];
        let mut srcs = Vec::new();
        for op in &builder.phi_ops[phi] {
            srcs.push(resolve_checked(&builder, *op)?);
        }
        if srcs.len() != blocks[block].preds.len() {
            return unsupported("internal error: phi does not match its predecessors");
        }
        blocks[block].phis.push(Phi { dst: *phi, srcs });
    }
    remove_dead_phis(&mut blocks, builder.forward.len());

    let mut all_consts = consts;
    all_consts.extend(builder.extra_consts.drain(..));
    let nvalues = builder.forward.len() as u32;
    let value_names = std::mem::take(&mut builder.value_names);
    drop(builder);

    let opt_attr = |name: &str| -> PyResult<Option<PyObj>> {
        let v = func.getattr(name)?;
        if v.is_none() {
            Ok(None)
        } else {
            Ok(Some(unsafe { PyObj::from_borrowed(v.as_ptr()) }))
        }
    };
    let globals = func.getattr("__globals__")?;
    globals.downcast::<PyDict>()?;
    let builtins = func.getattr("__builtins__")?;
    let builtins = if builtins.downcast::<PyDict>().is_ok() {
        builtins
    } else {
        builtins.getattr("__dict__")?
    };
    let mut param_names = Vec::new();
    for i in 0..nparams {
        let name = varnames_tuple.get_item(i)?;
        unsafe {
            let mut ptr = ffi::Py_NewRef(name.as_ptr());
            ffi::PyUnicode_InternInPlace(&mut ptr);
            param_names.push(PyObj::from_owned(ptr));
        }
    }
    let _ = py;
    Ok(Function {
        name: code.getattr("co_name")?.extract()?,
        qualname: code.getattr("co_qualname")?.extract()?,
        filename: code.getattr("co_filename")?.extract()?,
        firstlineno: code.getattr("co_firstlineno")?.extract()?,
        consts: all_consts,
        blocks,
        nvalues,
        params,
        value_names,
        lines,
        globals: unsafe { PyObj::from_borrowed(globals.as_ptr()) },
        builtins: unsafe { PyObj::from_borrowed(builtins.as_ptr()) },
        closure: opt_attr("__closure__")?,
        defaults: opt_attr("__defaults__")?,
        kwdefaults: opt_attr("__kwdefaults__")?,
        argcount,
        posonlyargcount,
        kwonlyargcount,
        has_varargs,
        has_varkw,
        param_names,
    })
}

fn remove_dead_phis(blocks: &mut [Block], nvalues: usize) {
    loop {
        let mut used = vec![false; nvalues];
        for block in blocks.iter() {
            for phi in &block.phis {
                for s in &phi.srcs {
                    // A phi feeding only itself is not a use.
                    if *s != phi.dst {
                        used[*s as usize] = true;
                    }
                }
            }
            for inst in &block.insts {
                inst.op.for_each_use(|v| used[v as usize] = true);
            }
            block.term.for_each_use(|v| used[v as usize] = true);
        }
        let mut removed = false;
        for block in blocks.iter_mut() {
            let before = block.phis.len();
            block.phis.retain(|phi| used[phi.dst as usize]);
            removed |= block.phis.len() != before;
        }
        if !removed {
            break;
        }
    }
}

pub fn rewrite_uses(op: &mut Op, f: &mut dyn FnMut(V) -> PyResult<V>) -> PyResult<()> {
    macro_rules! r {
        ($($v:expr),*) => {{ $( *$v = f(*$v)?; )* }};
    }
    match op {
        Op::Const(_) | Op::LoadGlobal(_) | Op::LoadDeref(_) => {}
        Op::LoadAttr(a, _)
        | Op::StoreGlobal(_, a)
        | Op::Unary(_, a)
        | Op::ToBool(a)
        | Op::IsNone(a, _)
        | Op::ListToTuple(a)
        | Op::Unpack(a, _)
        | Op::TupleGet(a, _)
        | Op::GetIter(a)
        | Op::IterNext(a)
        | Op::CheckBound(a, _) => r!(a),
        Op::Binary(_, a, b)
        | Op::Compare(_, a, b)
        | Op::Is(a, b, _)
        | Op::Contains(a, b, _)
        | Op::GetItem(a, b)
        | Op::DelItem(a, b)
        | Op::ListAppend(a, b)
        | Op::SetAdd(a, b)
        | Op::ListExtend(a, b)
        | Op::SetUpdate(a, b)
        | Op::DictUpdate(a, b)
        | Op::DictMerge(a, b) => r!(a, b),
        Op::StoreAttr { obj, val, .. } => r!(obj, val),
        Op::SetItem { obj, key, val } => r!(obj, key, val),
        Op::MapAdd(a, b, c) | Op::BuildSlice(a, b, c) => r!(a, b, c),
        Op::Format { val, spec, .. } => r!(val, spec),
        Op::BuildTuple(vs)
        | Op::BuildList(vs)
        | Op::BuildSet(vs)
        | Op::BuildMap(vs)
        | Op::BuildString(vs) => {
            for v in vs.iter_mut() {
                r!(v);
            }
        }
        Op::Call { callee, args, .. } => {
            r!(callee);
            for v in args.iter_mut() {
                r!(v);
            }
        }
        Op::CallEx { callee, args, kwargs } => r!(callee, args, kwargs),
    }
    Ok(())
}

fn rewrite_term_uses(term: &mut Term, f: &mut dyn FnMut(V) -> PyResult<V>) -> PyResult<()> {
    match term {
        Term::Jump(_) => {}
        Term::Br { cond, .. } => *cond = f(*cond)?,
        Term::BrNull { v, .. } => *v = f(*v)?,
        Term::Return(v) => *v = f(*v)?,
        Term::Raise { exc, cause } => {
            *exc = f(*exc)?;
            *cause = f(*cause)?;
        }
    }
    Ok(())
}
