//! Inlining of calls to Python functions.
//!
//! A call `g(a, b)` whose callee is a global that names a plain Python
//! function when the caller is compiled is replaced by the callee's graph. The
//! assumption is a guard like any other: a branch on `is_function`, whose
//! other side keeps the original call, so rebinding the global or swapping
//! the function's code only changes which side runs.
//!
//! ```text
//!     %g = load_global 'g'              %g = load_global 'g'
//!     ...                               ...
//!     %r = call %g(%a, %b)      =>      %t = is_function %g <g>
//!     ...                               br %t <entry of g> <call>
//!                                   <blocks of g, returning into join>
//!                                   call:  %r' = call %g(%a, %b); jump join
//!                                   join:  %r = phi [...: returned values, call: %r']
//!                                          ...
//! ```

use crate::bytecode::OpcodeTable;
use crate::frontend::{self, rewrite_term_uses, rewrite_uses};
use crate::ir::*;
use crate::pyffi::PyObj;
use pyo3::ffi;
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};

const CO_VARARGS_OR_KEYWORDS: u32 = 0x04 | 0x08;
/// Inlined functions nested deeper than this stay calls.
const MAX_DEPTH: usize = 4;
const MAX_CALLEE_INSTS: usize = 400;
const MAX_ADDED_INSTS: usize = 4000;

struct Inliner<'a, 'py> {
    py: Python<'py>,
    table: &'a OpcodeTable,
    /// Per value: the name it loads, if it is defined by a load_global.
    global_name: HashMap<V, u32>,
    /// Blocks holding the call an inlined body falls back to.
    fallback_blocks: HashSet<usize>,
    /// Code object of each origin.
    origin_code: Vec<*mut ffi::PyObject>,
    added_insts: usize,
}

fn origin_of(bc: u32) -> usize {
    (bc >> ORIGIN_SHIFT) as usize
}

/// Adds `offset` to every constant an instruction refers to.
fn shift_consts(op: &mut Op, offset: u32) {
    let shift = |c: &mut u32| {
        if *c != NONE {
            *c += offset
        }
    };
    match op {
        Op::Const(c)
        | Op::LoadGlobal(c)
        | Op::LoadAttr(_, c)
        | Op::StoreGlobal(c, _)
        | Op::CheckBound(_, c)
        | Op::StoreAttr { name: c, .. }
        | Op::Call { kwnames: c, .. } => shift(c),
        Op::CallMethod { name, kwnames, .. } => {
            shift(name);
            shift(kwnames)
        }
        Op::IsFunction { func, code, .. } => {
            shift(func);
            shift(code)
        }
        _ => {}
    }
}

impl Inliner<'_, '_> {
    /// The function a call inlines, built as a graph of its own, if the call
    /// qualifies.
    fn callee(&self, func: &Function, b: usize, i: usize) -> Option<(Function, PyObj, PyObj)> {
        let inst = &func.blocks[b].insts[i];
        let Op::Call { callee, args, kwnames } = &inst.op else {
            return None;
        };
        if *kwnames != NONE || self.fallback_blocks.contains(&b) {
            return None;
        }
        let name = *self.global_name.get(callee)?;
        let target = unsafe {
            let key = func.consts[name as usize].as_ptr();
            let found = ffi::PyDict_GetItemWithError(func.globals.as_ptr(), key);
            if found.is_null() {
                // A builtin, or nothing yet: either way not a Python function.
                ffi::PyErr_Clear();
                return None;
            }
            // A function of another module would load its globals from ours.
            if ffi::PyFunction_Check(found) == 0
                || ffi::PyFunction_GetGlobals(found) != func.globals.as_ptr()
            {
                return None;
            }
            Bound::from_borrowed_ptr(self.py, found)
        };
        let code = target.getattr("__code__").ok()?;
        let mut origin = origin_of(inst.bc);
        let mut depth = 1;
        loop {
            if self.origin_code[origin] == code.as_ptr() {
                return None; // recursion
            }
            if origin == 0 {
                break;
            }
            origin = origin_of(func.origins[origin].call_bc);
            depth += 1;
        }
        let flags: u32 = code.getattr("co_flags").ok()?.extract().ok()?;
        let argcount: usize = code.getattr("co_argcount").ok()?.extract().ok()?;
        let kwonly: usize = code.getattr("co_kwonlyargcount").ok()?.extract().ok()?;
        // Exactly the positional parameters, so defaults do not come into it.
        if depth > MAX_DEPTH
            || flags & CO_VARARGS_OR_KEYWORDS != 0
            || kwonly != 0
            || argcount != args.len()
            || !target.getattr("__closure__").ok()?.is_none()
        {
            return None;
        }
        // What the frontend cannot compile stays a call.
        let graph = frontend::build(self.py, self.table, &target).ok()?;
        let size: usize = graph.blocks.iter().map(|block| block.insts.len()).sum();
        if size > MAX_CALLEE_INSTS
            || self.added_insts + size > MAX_ADDED_INSTS
            || !graph.blocks[0].preds.is_empty()
        {
            return None;
        }
        unsafe {
            Some((graph, PyObj::from_borrowed(target.as_ptr()), PyObj::from_borrowed(code.as_ptr())))
        }
    }

    /// Replaces the call at instruction `i` of block `b` by `callee`.
    fn splice(&mut self, func: &mut Function, b: usize, i: usize, callee: Function, target: PyObj, code: PyObj) {
        let tail = func.blocks[b].insts.split_off(i + 1);
        let call = func.blocks[b].insts.pop().unwrap();
        let Op::Call { callee: callee_value, args, .. } = &call.op else {
            unreachable!()
        };
        let (callee_value, args) = (*callee_value, args.clone());

        let value_offset = func.nvalues;
        let const_offset = func.consts.len() as u32;
        let block_offset = func.blocks.len() as B;
        let origin = func.origins.len() as u32;
        let fallback = block_offset + callee.blocks.len() as B;
        let join = fallback + 1;

        func.origins.push(Origin {
            name: callee.name.clone(),
            filename: callee.filename.clone(),
            call_bc: call.bc,
        });
        self.origin_code.push(code.as_ptr());
        let tag = |bc: u32| (origin << ORIGIN_SHIFT) | bc;
        func.lines.extend(callee.lines.iter().map(|(start, end, line)| (tag(*start), tag(*end), *line)));
        func.consts.extend(callee.consts.iter().cloned());
        let func_const = func.consts.len() as u32;
        func.consts.push(target);
        func.consts.push(code);
        func.value_names.extend(callee.value_names.iter().cloned());
        func.nvalues += callee.nvalues;
        let mut new_value = |func: &mut Function| {
            func.value_names.push(None);
            func.nvalues += 1;
            func.nvalues - 1
        };
        let guard = new_value(func);
        let fallback_result = new_value(func);

        // Parameters become the arguments; every other value gets a new number.
        let params: HashMap<V, V> = callee.params.iter().copied().zip(args.iter().copied()).collect();
        let map = |v: V| -> V {
            if v == NONE || v == UNDEF {
                v
            } else {
                params.get(&v).copied().unwrap_or(v + value_offset)
            }
        };
        let mut returns: Vec<(B, V)> = Vec::new();
        for (j, block) in callee.blocks.into_iter().enumerate() {
            let this = block_offset + j as B;
            let mut insts = block.insts;
            for inst in insts.iter_mut() {
                inst.dst = map(inst.dst);
                inst.bc = tag(inst.bc);
                let _ = rewrite_uses(&mut inst.op, &mut |v| Ok(map(v)));
                shift_consts(&mut inst.op, const_offset);
                if let Op::LoadGlobal(name) = &inst.op {
                    self.global_name.insert(inst.dst, *name);
                }
            }
            self.added_insts += insts.len();
            let mut term = block.term;
            let _ = rewrite_term_uses(&mut term, &mut |v| Ok(map(v)));
            let term = match term {
                Term::Jump(t) => Term::Jump(t + block_offset),
                Term::Br { cond, then_, else_ } => {
                    Term::Br { cond, then_: then_ + block_offset, else_: else_ + block_offset }
                }
                Term::BrNull { v, null, nonnull } => {
                    Term::BrNull { v, null: null + block_offset, nonnull: nonnull + block_offset }
                }
                Term::Return(v) => {
                    returns.push((this, v));
                    Term::Jump(join)
                }
                raise @ Term::Raise { .. } => raise,
            };
            let phis = block
                .phis
                .into_iter()
                .map(|phi| Phi { dst: map(phi.dst), srcs: phi.srcs.into_iter().map(map).collect() })
                .collect();
            let preds =
                if j == 0 { vec![b as B] } else { block.preds.iter().map(|p| p + block_offset).collect() };
            func.blocks.push(Block { phis, insts, term, preds, bc_start: tag(block.bc_start) });
        }

        // The rest of the caller's block runs after either side.
        let term = std::mem::replace(
            &mut func.blocks[b].term,
            Term::Br { cond: guard, then_: block_offset, else_: fallback },
        );
        for succ in term.successors() {
            for pred in func.blocks[succ as usize].preds.iter_mut() {
                if *pred == b as B {
                    *pred = join;
                }
            }
        }
        func.blocks[b].insts.push(Inst {
            dst: guard,
            op: Op::IsFunction { v: callee_value, func: func_const, code: func_const + 1 },
            bc: call.bc,
            kills: Vec::new(),
        });
        self.fallback_blocks.insert(fallback as usize);
        func.blocks.push(Block {
            phis: Vec::new(),
            insts: vec![Inst { dst: fallback_result, op: call.op, bc: call.bc, kills: Vec::new() }],
            term: Term::Jump(join),
            preds: vec![b as B],
            bc_start: call.bc,
        });
        let mut preds: Vec<B> = returns.iter().map(|(block, _)| *block).collect();
        let mut srcs: Vec<V> = returns.iter().map(|(_, v)| *v).collect();
        preds.push(fallback);
        srcs.push(fallback_result);
        func.blocks.push(Block {
            phis: vec![Phi { dst: call.dst, srcs }],
            insts: tail,
            term,
            preds,
            bc_start: call.bc,
        });
    }
}

/// Inlines what qualifies, including calls inside inlined functions, and
/// returns how many calls were replaced.
pub fn inline_calls(
    py: Python<'_>,
    table: &OpcodeTable,
    func: &mut Function,
    code: &Bound<'_, PyAny>,
) -> usize {
    let mut inliner = Inliner {
        py,
        table,
        global_name: HashMap::new(),
        fallback_blocks: HashSet::new(),
        origin_code: vec![code.as_ptr()],
        added_insts: 0,
    };
    for block in &func.blocks {
        for inst in &block.insts {
            if let Op::LoadGlobal(name) = &inst.op {
                inliner.global_name.insert(inst.dst, *name);
            }
        }
    }
    let mut inlined = 0;
    // Splicing appends the callee's blocks and the rest of the caller's
    // block, so this loop reaches the calls inside them too.
    let mut b = 0;
    while b < func.blocks.len() {
        let mut i = 0;
        while i < func.blocks[b].insts.len() {
            if let Some((callee, target, callee_code)) = inliner.callee(func, b, i) {
                inliner.splice(func, b, i, callee, target, callee_code);
                inlined += 1;
                break;
            }
            i += 1;
        }
        b += 1;
    }
    inlined
}
