// How the recorder (record.rs) makes an entry of an instruction: the callee,
// the operand templates from their origins (origins.rs), the classes, the
// flags. Nothing here decides WHETHER an instruction is an entry by a C
// callee's row: record.rs asks the rule table first.
//
// What it is given: the instruction, its operands as the objects they are,
// and where the recorded run is (the activation, the body's step).
// What it guarantees
//   - vector(): the entry of a call of a C callee, in the order of the
//     call's operands, flagged pure only for a callee whose every row says
//     that it writes no operand (rules::torch::only_computes); Err for a form of call whose
//     operands are not all on the instruction (a star call, a partial, a
//     builtin bound to its receiver) and for an operand with no origin;
//   - Call::stated(): what the ROW of the call states beyond that, taken
//     from the row and from nothing else: the operands the call writes
//     (flags.writes: each place the row names, a position or a keyword,
//     among this call's operands; Err if the call does not hand it one
//     there), which of them no operator's schema declares (the witness
//     cannot see those), that the call draws from the default random
//     generators if the row says so (the flag `draws`, and never `pure`
//     beside it: a list with such an entry is not run a second time, as one
//     with an entry that writes is not), and the operand the result is, if
//     the row says the call hands one back. A row that writes anything else
//     than operands and the generators is no entry: Err;
//   - slot(): the entry of an operator, a subscript or an attribute
//     instruction that has a plain tensor where its line of the license
//     wants one: Some(Ok) only if slots::license takes it on the classes of
//     ALL operands; an in-place operator says it writes operand 0, a store
//     its container; Some(Err), with the reason, if no line takes it or an
//     operand has no origin; None for any other instruction and for an
//     attribute of FACTS on a tensor (counted: it is a fact of the key);
//   - built(): for an instruction that builds a list, a tuple, a dict or a
//     slice, the template of what it builds.

use pyo3::ffi::{self, PyObject};
use pyo3::prelude::*;

use super::origins::{plain_tensor, Built, Made, Origins};
use super::{check, slots, Callee, Entry, Flags, Tmpl, DRAWS, PURE};
use crate::bindings::ffi::op_hook::TensorIdent;
use crate::cfg::ir::{Instr, Op};
use crate::cfg::opcodes::*;
use crate::exec::ffi::tr_partial_func;
use crate::exec::hist;
use crate::exec::rules::{self, Site};

// The key-fact attributes of a tensor: read off its key, which a region
// compares (an input) or asserts (a result).
const FACTS: [&str; 10] = ["shape", "dtype", "device", "layout", "ndim", "requires_grad", "is_cpu", "is_cuda", "is_sparse", "is_nested"];

pub struct Call {
    pub entry: Entry,
    pub label: String,
    // The tensors the call can reach, each with its operand's position; the
    // operands themselves.
    pub ids: Vec<(TensorIdent, usize)>,
    pub operands: Vec<*mut PyObject>,
    // What the callee's row states (stated): the operand the result is, and
    // the written operands that no schema declares.
    pub returns: Option<u32>,
    pub undeclared: Vec<u32>,
}

impl Call {
    pub unsafe fn stated(mut self, row: &rules::Row, kwnames: *mut PyObject) -> Result<Call, String> {
        unsafe {
            let n = self.entry.args.len();
            let named = if kwnames.is_null() { 0 } else { ffi::PyTuple_GET_SIZE(kwnames) as usize };
            let place = |at: &str| -> Option<u32> {
                match at.parse::<usize>() {
                    Ok(k) => (k + named < n).then_some(k as u32),
                    Err(_) => (0..named).find(|i| text(ffi::PyTuple_GET_ITEM(kwnames, *i as ffi::Py_ssize_t)) == at).map(|i| (n - named + i) as u32),
                }
            };
            match row.writes {
                rules::Write::None => {}
                rules::Write::Operands(written) | rules::Write::Draws(written) => {
                    if matches!(row.writes, rules::Write::Draws(_)) {
                        self.entry.flags.bits = (self.entry.flags.bits & !PURE) | DRAWS;
                    }
                    let mut writes = Vec::new();
                    for w in written {
                        let Some(p) = place(w.at) else { return Err(format!("its row says it writes operand {}, which this call does not hand it in that place", w.at)) };
                        writes.push(p);
                        if !w.declared {
                            self.undeclared.push(p);
                        }
                    }
                    self.entry.flags.writes = writes.into();
                }
                other => return Err(format!("its row writes what no entry of a list states: {other:?}")),
            }
            if let rules::Res::Operand(k) = row.result {
                self.returns = Some(k as u32);
            }
            Ok(self)
        }
    }
}

pub fn blank(callee: Callee, at: u32) -> Entry {
    Entry { callee, args: Box::default(), kwnames: None, out: None, at, flags: Flags::default(), assert: None, classes: Box::default(), decided: Box::default(), release: Box::default() }
}

pub unsafe fn text(s: *mut PyObject) -> String {
    unsafe { hist::text(s) }.unwrap_or_else(|| "?".into())
}

pub struct Maker<'a> {
    pub origins: &'a mut Origins,
    pub facts: &'a mut u64,
    // The body's step in flight, and the activation the instruction is of.
    pub at: u32,
    pub depth: u32,
}

impl Maker<'_> {
    pub unsafe fn built(&mut self, ins: &Instr, all: &[*mut PyObject]) -> Option<Made> {
        unsafe {
            let values = |from: usize| ins.args[from..].iter().map(|x| x.0).zip(all[from..].iter().copied()).map(|(v, o)| (o, v)).collect::<Vec<_>>();
            let (kind, parts) = match ins.op {
                Op::BuildTuple(_) => (Built::Tuple, values(0)),
                Op::Bytecode { opcode: BUILD_LIST, .. } => (Built::List, values(0)),
                Op::Bytecode { opcode: BUILD_SLICE, .. } => (Built::Slice, values(0)),
                Op::Bytecode { opcode: BUILD_MAP, .. } => (Built::Map(all.iter().step_by(2).copied().collect()), values(0).into_iter().skip(1).step_by(2).collect()),
                Op::Bytecode { opcode: BUILD_CONST_KEY_MAP, .. } if all.last().is_some_and(|k| !k.is_null() && ffi::PyTuple_CheckExact(*k) != 0) => {
                    let keys = *all.last().unwrap();
                    let mut parts = values(0);
                    parts.pop();
                    (Built::Map((0..ffi::PyTuple_GET_SIZE(keys)).map(|i| ffi::PyTuple_GET_ITEM(keys, i)).collect()), parts)
                }
                _ => return None,
            };
            Some(self.origins.build(kind, &parts, self.depth))
        }
    }

    unsafe fn operands(&mut self, objects: &[*mut PyObject], values: &[u32]) -> Result<(Vec<Tmpl>, Vec<Option<Py<PyAny>>>), String> {
        let (mut tmpls, mut classes) = (Vec::new(), Vec::new());
        for (o, v) in objects.iter().zip(values) {
            let (t, c) = unsafe { self.origins.template(*o, (self.depth, *v)) }?;
            tmpls.push(t);
            classes.extend(c);
        }
        Ok((tmpls, classes))
    }

    // The entry of a C callee with a row.
    pub unsafe fn vector(&mut self, f: *mut PyObject, ins: &Instr, all: &[*mut PyObject], kwnames: *mut PyObject) -> Result<Call, String> {
        unsafe {
            if !matches!(ins.op, Op::Call { .. }) {
                return Err("a star call of C code: its operands are not on the instruction".into());
            }
            if !tr_partial_func(f).is_null() {
                return Err("a call through a functools.partial: its own arguments are no operands".into());
            }
            if ffi::Py_TYPE(f) == &raw mut ffi::PyCFunction_Type {
                let bound = (*(f as *mut ffi::PyCFunctionObject)).m_self;
                if !bound.is_null() && ffi::PyModule_Check(bound) == 0 && ffi::PyType_Check(bound) == 0 {
                    return Err("a builtin bound to its receiver: the receiver is no operand of the instruction".into());
                }
            }
            let first = if all[0].is_null() { 2 } else { 1 };
            let values: Vec<u32> = ins.args[first..].iter().map(|x| x.0).collect();
            let (args, classes) = self.operands(&all[first..], &values)?;
            let label = rules::torch::name(f).map_or_else(|| hist::callee_name(f), str::to_string);
            let entry = Entry {
                args: args.into(),
                classes: classes.into(),
                kwnames: (!kwnames.is_null()).then(|| Py::<pyo3::types::PyTuple>::from_borrowed_ptr(Python::assume_attached(), kwnames)),
                // Pure only if every row of the callee says, of every form of
                // its call, that it writes no operand (the table has no row
                // for a callee that draws or changes a key): a list with any
                // other entry is never run again.
                flags: Flags { bits: if rules::torch::only_computes(f) { PURE } else { 0 }, ..Flags::default() },
                ..blank(Callee::Vector(Py::from_borrowed_ptr(Python::assume_attached(), f), label.clone().into()), self.at)
            };
            Ok(Call { entry, label, ids: Vec::new(), operands: all[first..].to_vec(), returns: None, undeclared: Vec::new() })
        }
    }

    // The entry of an operator, a subscript or an attribute instruction on a
    // plain tensor. None: `ins` is no such instruction, or a key fact.
    pub unsafe fn slot(&mut self, ins: &Instr, site: &Site) -> Option<Result<Call, String>> {
        unsafe {
            let all = site.operands;
            // The callee, and the instruction's operands in the entry's order.
            let (callee, order): (Callee, &[usize]) = match ins.op {
                Op::Binary(k) => (Callee::Binary(k as u32), &[0, 1]),
                Op::Compare(k) => (Callee::Compare(k as u32), &[0, 1]),
                Op::Bytecode { opcode: UNARY_NEGATIVE | UNARY_INVERT, .. } | Op::Bytecode { opcode: CALL_INTRINSIC_1, arg: 5 } => {
                    let Op::Bytecode { opcode, arg } = ins.op else { return None };
                    (Callee::Unary(opcode, if opcode == CALL_INTRINSIC_1 { arg } else { 0 }), &[0])
                }
                Op::Bytecode { opcode: BINARY_SUBSCR, .. } => (Callee::GetItem, &[0, 1]),
                Op::Bytecode { opcode: BINARY_SLICE, .. } => (Callee::GetItem, &[0, 1, 2]),
                Op::Bytecode { opcode: STORE_SUBSCR, .. } => (Callee::SetItem, &[0, 1, 2]),
                Op::LoadAttr(n) => {
                    let name = *site.names.add(n as usize);
                    if FACTS.contains(&text(name).as_str()) {
                        *self.facts += plain_tensor(all[0]) as u64;
                        return None;
                    }
                    (Callee::GetAttr(Py::<pyo3::types::PyString>::from_borrowed_ptr(Python::assume_attached(), name)), &[0])
                }
                _ => return None,
            };
            let governs = if matches!(callee, Callee::SetItem) { 1 } else { 0 };
            if !(plain_tensor(all[governs]) || matches!(callee, Callee::Binary(_) | Callee::Compare(_)) && plain_tensor(all[1])) {
                return None;
            }
            let objects: Vec<*mut PyObject> = order.iter().map(|i| all[*i]).collect();
            let values: Vec<u32> = order.iter().map(|i| ins.args[*i].0).collect();
            let label = hist::kind(&ins.op).to_string();
            Some(self.operands(&objects, &values).and_then(|(mut args, mut classes)| {
                if matches!(ins.op, Op::Bytecode { opcode: BINARY_SLICE, .. }) {
                    let (stop, start) = (args.pop().unwrap(), args.pop().unwrap());
                    args.push(Tmpl::Slice(Box::new([start, stop, Tmpl::Null])));
                    classes.push(None);
                }
                let writes: Box<[u32]> = match callee {
                    Callee::Binary(k) if k >= 13 => Box::new([0]),
                    Callee::SetItem => Box::new([1]),
                    _ => Box::default(),
                };
                let flags = Flags { bits: if writes.is_empty() { PURE } else { 0 }, writes, ..Flags::default() };
                let entry = Entry { args: args.into(), classes: classes.into(), flags, ..blank(callee, self.at) };
                slots::license(&entry.callee, &check::shapes(&entry, &self.origins.consts))?;
                Ok(Call { entry, label, ids: Vec::new(), operands: objects, returns: None, undeclared: Vec::new() })
            }))
        }
    }

}

// Makes the leaf of `t` that is the whole result of slot j its move.
pub fn moves(t: &mut Tmpl, j: u32) -> bool {
    match t {
        Tmpl::Res(s) if *s == j => {
            *t = Tmpl::Move(j);
            true
        }
        Tmpl::Seq(_, parts) | Tmpl::Map(_, parts) => parts.iter_mut().any(|p| moves(p, j)),
        Tmpl::Slice(parts) => parts.iter_mut().any(|p| moves(p, j)),
        _ => false,
    }
}
