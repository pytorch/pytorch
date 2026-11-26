// The one way a list comes to be. List::new refuses every list it does not
// fully understand, with the reason, so that the interpreter has no case to
// guess at: an operand that is not there when its entry runs, a callee that
// could enter a Python frame by its type, flags that contradict each other.

use std::ptr::addr_of_mut;
use std::sync::atomic::AtomicU64;

use pyo3::ffi::{self, PyObject};
use pyo3::prelude::*;

use super::super::ffi::tr_type_heap;
use super::{Callee, Entry, List, Tmpl, DRAWS, EFFECT, FLAG_NAMES, FRESH, KEY_FIELDS, OPAQUE, PURE, STATE, UNARY_OPS};
use crate::cfg::ir::{BINARY_OPS, COMPARE_OPS};

// The exact types of a callee that is called as C code: a builtin function or
// method, a method descriptor, a slot wrapper, and a static type whose
// metatype is `type`. A Python function, a bound method, an instance with
// __call__ and a class made by Python code are frames, not entries.
unsafe fn c_callable(f: *mut PyObject) -> bool {
    unsafe {
        let t = ffi::Py_TYPE(f);
        t == addr_of_mut!(ffi::PyCFunction_Type)
            || t == addr_of_mut!(ffi::PyCMethod_Type)
            || t == addr_of_mut!(ffi::PyMethodDescr_Type)
            || t == addr_of_mut!(ffi::PyWrapperDescr_Type)
            || (t == addr_of_mut!(ffi::PyType_Type) && tr_type_heap(f as *mut ffi::PyTypeObject) == 0)
    }
}

// The values a dict the body made may have as keys: their hash and equality
// are C code.
unsafe fn plain_key(k: *mut PyObject) -> bool {
    unsafe {
        ffi::PyUnicode_CheckExact(k) != 0
            || ffi::PyLong_CheckExact(k) != 0
            || ffi::PyFloat_CheckExact(k) != 0
            || ffi::PyBytes_CheckExact(k) != 0
            || ffi::PyBool_Check(k) != 0
            || k == ffi::Py_None()
    }
}

struct Scope {
    ninputs: u32,
    npins: u32,
    nconsts: u32,
    // By entry so far: whether its slot holds a result.
    live: Vec<bool>,
}

impl Scope {
    fn held(&self, j: u32) -> Result<(), String> {
        match self.live.get(j as usize) {
            None => Err(format!("r{j} is the result of this or a later entry")),
            Some(false) => Err(format!("r{j} is not held: its entry keeps no result, or an earlier entry killed it")),
            Some(true) => Ok(()),
        }
    }

    fn tmpl(&self, t: &Tmpl, in_slice: bool) -> Result<(), String> {
        let within = |i: u32, n: u32, what: &str| if i < n { Ok(()) } else { Err(format!("{what} {i} of {n}")) };
        match t {
            Tmpl::In(k) => within(*k, self.ninputs, "input"),
            Tmpl::Pin(i) => within(*i, self.npins, "pin"),
            Tmpl::Const(i) => within(*i, self.nconsts, "constant"),
            Tmpl::Res(j) => self.held(*j),
            Tmpl::Part(j, path) if path.is_empty() => Err(format!("a part of r{j} with an empty path")),
            Tmpl::Part(j, _) => self.held(*j),
            Tmpl::Seq(_, parts) => parts.iter().try_for_each(|p| self.tmpl(p, false)),
            Tmpl::Map(keys, parts) => {
                if keys.len() != parts.len() {
                    return Err(format!("a map of {} keys and {} values", keys.len(), parts.len()));
                }
                if !keys.iter().all(|k| unsafe { plain_key(k.as_ptr()) }) {
                    return Err("a map key that is no exact str, int, float, bytes, bool or None".into());
                }
                parts.iter().try_for_each(|p| self.tmpl(p, false))
            }
            Tmpl::Slice(p) => p.iter().try_for_each(|p| self.tmpl(p, true)),
            Tmpl::Null if in_slice => Ok(()),
            Tmpl::Null => Err("a null operand that is no bound of a slice".into()),
        }
    }

    fn entry(&self, k: usize, e: &Entry) -> Result<(), String> {
        // (the count of operands or None for any, whether there is a result)
        let (arity, result) = match &e.callee {
            Callee::Vector(f, label) => {
                if !unsafe { c_callable(f.as_ptr()) } {
                    return Err(format!("the callee {label} is no builtin function or method, method descriptor, slot wrapper or static type"));
                }
                if label.is_empty() {
                    return Err("a callee with no name".into());
                }
                (None, true)
            }
            Callee::Binary(op) if (*op as usize) < BINARY_OPS.len() => (Some(2), true),
            Callee::Binary(op) => return Err(format!("no binary operator {op}")),
            Callee::Compare(op) if (*op as usize) < COMPARE_OPS.len() => (Some(2), true),
            Callee::Compare(op) => return Err(format!("no comparison {op}")),
            Callee::Unary(op, oparg) if UNARY_OPS.iter().any(|u| (u.1, u.2) == (*op, *oparg)) => (Some(1), true),
            Callee::Unary(op, oparg) => return Err(format!("no unary instruction {op} {oparg}")),
            Callee::GetItem => (Some(2), true),
            Callee::SetItem => (Some(3), false),
            Callee::DelItem => (Some(2), false),
            Callee::GetAttr(_) => (Some(1), true),
            Callee::SetAttr(_) => (Some(2), false),
            Callee::State(_) => return Err("a state setter: unimplemented".into()),
        };
        let n = e.args.len();
        if arity.is_some_and(|a| a != n) {
            return Err(format!("{n} operands where the callee takes {}", arity.unwrap_or(0)));
        }
        if let Some(kw) = &e.kwnames {
            let kw = kw.as_ptr();
            let exact = unsafe { ffi::PyTuple_CheckExact(kw) != 0 && (0..ffi::PyTuple_GET_SIZE(kw)).all(|i| ffi::PyUnicode_CheckExact(ffi::PyTuple_GET_ITEM(kw, i)) != 0) };
            if arity.is_some() || !exact || unsafe { ffi::PyTuple_GET_SIZE(kw) } as usize > n {
                return Err("keyword names that are no exact tuple of strs, more than the operands, or not on a vector".into());
            }
        }
        e.args.iter().try_for_each(|t| self.tmpl(t, false))?;
        match e.out {
            None => {}
            Some(j) if result && j as usize == k => {}
            Some(j) => return Err(format!("out {j}: the slot of an entry is its own index, and only where the callee has a result")),
        }
        let f = &e.flags;
        if f.bits >> FLAG_NAMES.len() != 0 || f.writes.iter().chain(f.rekeys.iter()).any(|i| *i as usize >= n) {
            return Err("an unknown flag, or a written or rekeyed operand that is not there".into());
        }
        if f.bits & PURE != 0 && (f.bits & (DRAWS | EFFECT | STATE | OPAQUE) != 0 || !f.writes.is_empty() || !f.rekeys.is_empty()) {
            return Err("pure beside writes, rekeys, draws, effect, state or opaque".into());
        }
        if f.bits & STATE != 0 {
            return Err("the state flag on a callee that is no state setter".into());
        }
        if let Some(a) = &e.assert {
            if !result || a.mask >> KEY_FIELDS.len() != 0 || a.same.is_some_and(|i| i as usize >= n) {
                return Err("an assert where there is no result, of an unknown field, or on an operand that is not there".into());
            }
        }
        Ok(())
    }
}

// An entry that can be taken back: it is pure, or all it writes is the result
// of an earlier entry whose storage the run made (fresh). A written part of a
// result or a written container of results is not taken as fresh: the trace
// says fresh of a whole result only.
fn reversible(e: &Entry, entries: &[Entry]) -> bool {
    let f = &e.flags;
    if f.bits & PURE != 0 {
        return true;
    }
    if f.writes.is_empty() || !f.rekeys.is_empty() || f.bits & (DRAWS | EFFECT | STATE | OPAQUE) != 0 {
        return false;
    }
    f.writes.iter().all(|i| match &e.args[*i as usize] {
        Tmpl::Res(j) => entries[*j as usize].flags.bits & FRESH != 0,
        Tmpl::In(_) | Tmpl::Pin(_) | Tmpl::Part(..) | Tmpl::Const(_) | Tmpl::Seq(..) | Tmpl::Map(..) | Tmpl::Slice(_) | Tmpl::Null => false,
    })
}

impl List {
    pub fn new(ninputs: u32, npins: u32, consts: Vec<Py<PyAny>>, entries: Vec<Entry>, outputs: Vec<Tmpl>) -> Result<List, String> {
        let mut scope = Scope { ninputs, npins, nconsts: consts.len() as u32, live: Vec::with_capacity(entries.len()) };
        for (k, e) in entries.iter().enumerate() {
            scope.entry(k, e).map_err(|why| format!("entry {k}: {why}"))?;
            scope.live.push(e.out.is_some());
            for j in e.kills.iter() {
                if *j as usize >= k || !scope.live[*j as usize] {
                    return Err(format!("entry {k}: kills r{j}, which is not an earlier entry's held result"));
                }
                scope.live[*j as usize] = false;
            }
        }
        for (i, t) in outputs.iter().enumerate() {
            scope.tmpl(t, false).map_err(|why| format!("output {i}: {why}"))?;
        }
        let first_irreversible = entries.iter().position(|e| !reversible(e, &entries)).unwrap_or(entries.len());
        let max_args = entries.iter().map(|e| e.args.len()).max().unwrap_or(0);
        Ok(List {
            entries: entries.into(),
            consts: consts.into(),
            ninputs,
            npins,
            outputs: outputs.into(),
            first_irreversible,
            max_args,
            stopped: AtomicU64::new(0),
        })
    }
}
