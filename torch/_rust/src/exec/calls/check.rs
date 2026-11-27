// The one way a list comes to be. List::new refuses every list it does not
// fully understand, with the reason, so that the interpreter has no case to
// guess at: an operand that is not there when its entry runs, a slot with no
// stated release or with two, a callee that could enter a Python frame by
// its type or (a slot callee) by the pinned classes of its operands, an
// operand whose class nothing establishes, flags that contradict each other.

use std::ptr::addr_of_mut;
use std::sync::Mutex;

use pyo3::ffi::{self, PyObject, PyTypeObject};
use pyo3::prelude::*;

use super::super::ffi::{tr_type_heap, tr_type_version};
use super::{plain, slots, Callee, Entry, List, Release, Tmpl, DRAWS, EFFECT, FLAG_NAMES, FRESH, OPAQUE, PURE, STATE, UNARY_OPS};
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

type Decided = Box<[(Py<PyAny>, u32)]>;

struct Scope<'a> {
    ninputs: u32,
    npins: u32,
    consts: &'a [Py<PyAny>],
    // By entry so far: whether its slot holds a result now, and the class its
    // assert gives that result (a type, by address).
    live: Vec<bool>,
    class: Vec<Option<usize>>,
}

// What the operand at each position is pinned to: the class of a leaf, the
// type a container is built as; null for a null.
unsafe fn operand_classes(e: &Entry) -> Vec<*mut PyTypeObject> {
    let mut at = 0;
    let mut leaves = Vec::new();
    e.args
        .iter()
        .map(|t| {
            let first = at;
            leaves.clear();
            t.leaves(&mut leaves);
            at += leaves.len();
            match t {
                Tmpl::Seq(true, _) => addr_of_mut!(ffi::PyList_Type),
                Tmpl::Seq(false, _) => addr_of_mut!(ffi::PyTuple_Type),
                Tmpl::Map(..) => addr_of_mut!(ffi::PyDict_Type),
                Tmpl::Slice(_) => addr_of_mut!(ffi::PySlice_Type),
                Tmpl::In(_) | Tmpl::Pin(_) | Tmpl::Res(_) | Tmpl::Move(_) | Tmpl::Part(..) | Tmpl::Const(_) | Tmpl::Null => {
                    e.classes.get(first).and_then(|c| c.as_ref()).map_or(std::ptr::null_mut(), |c| c.as_ptr() as *mut PyTypeObject)
                }
            }
        })
        .collect()
}

// The slots an entry moves and the containers it builds, as the call
// releases them: by argument, a container behind the slots moved into it.
fn releases(e: &Entry) -> Box<[Release]> {
    let mut out = Vec::new();
    for (pos, t) in e.args.iter().enumerate() {
        let mut leaves = Vec::new();
        t.leaves(&mut leaves);
        out.extend(leaves.iter().filter_map(|l| if let Tmpl::Move(j) = l { Some(Release::Slot(*j)) } else { None }));
        if t.made() {
            out.push(Release::Arg(pos as u32));
        }
    }
    out.into()
}

impl Scope<'_> {
    fn held(&self, j: u32) -> Result<(), String> {
        match self.live.get(j as usize) {
            None => Err(format!("r{j} is the result of this or a later entry")),
            Some(false) => Err(format!("r{j} is not held: its entry keeps no result, or it was moved or dropped before")),
            Some(true) => Ok(()),
        }
    }

    fn tmpl(&self, t: &Tmpl, in_slice: bool) -> Result<(), String> {
        let within = |i: u32, n: u32, what: &str| if i < n { Ok(()) } else { Err(format!("{what} {i} of {n}")) };
        match t {
            Tmpl::In(k) => within(*k, self.ninputs, "input"),
            Tmpl::Pin(i) => within(*i, self.npins, "pin"),
            Tmpl::Const(i) => within(*i, self.consts.len() as u32, "constant"),
            Tmpl::Res(j) | Tmpl::Move(j) => self.held(*j),
            Tmpl::Part(j, path) if path.is_empty() => Err(format!("a part of r{j} with an empty path")),
            Tmpl::Part(j, _) => self.held(*j),
            Tmpl::Seq(_, parts) => parts.iter().try_for_each(|p| self.tmpl(p, false)),
            Tmpl::Map(keys, parts) => {
                if keys.len() != parts.len() {
                    return Err(format!("a map of {} keys and {} values", keys.len(), parts.len()));
                }
                if !keys.iter().all(|k| unsafe { plain(k.as_ptr()) }) {
                    return Err("a map key that is no exact str, int, float, bytes, bool or None".into());
                }
                parts.iter().try_for_each(|p| self.tmpl(p, false))
            }
            Tmpl::Slice(p) => p.iter().try_for_each(|p| self.tmpl(p, true)),
            Tmpl::Null if in_slice => Ok(()),
            Tmpl::Null => Err("a null operand that is no bound of a slice".into()),
        }
    }

    // Every leaf of the operands against its pinned class.
    fn pins(&self, e: &Entry) -> Result<(), String> {
        let mut leaves = Vec::new();
        e.args.iter().for_each(|t| t.leaves(&mut leaves));
        if leaves.len() != e.classes.len() {
            return Err(format!("{} classes for {} operands", e.classes.len(), leaves.len()));
        }
        let slot = |t: &Tmpl| if let Tmpl::Res(j) | Tmpl::Move(j) | Tmpl::Part(j, _) = t { Some(*j) } else { None };
        for (t, class) in leaves.iter().zip(e.classes.iter()) {
            let pinned = class.as_ref().map(|c| c.as_ptr() as usize);
            if pinned.is_some() == matches!(t, Tmpl::Null) {
                return Err("an operand with no class, or a null with one".into());
            }
            match t {
                Tmpl::Const(i) if pinned != Some(unsafe { ffi::Py_TYPE(self.consts[*i as usize].as_ptr()) } as usize) => {
                    return Err(format!("constant {i} is not of the class it is pinned to"));
                }
                Tmpl::Res(j) | Tmpl::Move(j) => match self.class[*j as usize] {
                    None => return Err(format!("r{j} is an operand and its entry asserts no class of it")),
                    asserted if asserted != pinned => return Err(format!("r{j} is pinned to another class than its entry asserts")),
                    _ => {}
                },
                Tmpl::Part(j, _) if !self.class[*j as usize].is_some_and(|c| unsafe { slots::tuple_class(c as *mut PyTypeObject) }) => {
                    return Err(format!("a part of r{j}, which its entry does not assert to be a tuple or a struct sequence"));
                }
                // An input and a pin are compared when the entry runs.
                Tmpl::In(_) | Tmpl::Pin(_) | Tmpl::Const(_) | Tmpl::Part(..) | Tmpl::Null | Tmpl::Seq(..) | Tmpl::Map(..) | Tmpl::Slice(_) => {}
            }
            if let Tmpl::Move(j) = t {
                if leaves.iter().filter(|l| slot(l) == Some(*j)).count() != 1 {
                    return Err(format!("r{j} is moved and named again by the same entry"));
                }
            }
        }
        Ok(())
    }

    // Ok: the classes a slot callee was resolved on, with their versions.
    fn entry(&self, k: usize, e: &Entry) -> Result<Decided, String> {
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
            Callee::Drop(_) => return Err("a drop is no call".into()),
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
            let names = unsafe { ffi::PyTuple_GET_SIZE(kw) };
            if (0..names).any(|i| (0..i).any(|j| unsafe { ffi::PyUnicode_Compare(ffi::PyTuple_GET_ITEM(kw, i), ffi::PyTuple_GET_ITEM(kw, j)) } == 0)) {
                return Err("keyword names of which two are the same".into());
            }
        }
        e.args.iter().try_for_each(|t| self.tmpl(t, false))?;
        self.pins(e)?;
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
            if !result || a.same.is_some_and(|i| i as usize >= n) {
                return Err("an assert where there is no result, or on an operand that is not there".into());
            }
        }
        let Some((first, second)) = slots::deciding(&e.callee) else { return Ok(Box::default()) };
        let classes = unsafe { operand_classes(e) };
        let (a, b) = (classes[first], second.map_or(std::ptr::null_mut(), |i| classes[i]));
        unsafe { slots::resolve(&e.callee, a, b) }.map_err(|why| format!("the callee's slot would run Python, or is not covered: {why}"))?;
        let mut decided = Vec::new();
        for class in [a, b] {
            if !class.is_null() && !decided.iter().any(|(c, _): &(Py<PyAny>, u32)| c.as_ptr() == class as *mut PyObject) {
                let version = unsafe { tr_type_version(class) };
                if version == 0 {
                    return Err("a class of an operand has no version: a change to it could not be seen".into());
                }
                decided.push((unsafe { Py::from_borrowed_ptr(Python::assume_attached(), class as *mut PyObject) }, version));
            }
        }
        Ok(decided.into())
    }
}

// An entry that can be taken back: it is pure, or all it writes is the result
// of an earlier entry whose storage the run made (fresh). A written part of a
// result or a written container of results is not taken as fresh: the trace
// says fresh of a whole result only.
fn reversible(e: &Entry, entries: &[Entry]) -> bool {
    let f = &e.flags;
    if matches!(e.callee, Callee::Drop(_)) || f.bits & PURE != 0 {
        return true;
    }
    if f.writes.is_empty() || !f.rekeys.is_empty() || f.bits & (DRAWS | EFFECT | STATE | OPAQUE) != 0 {
        return false;
    }
    f.writes.iter().all(|i| match &e.args[*i as usize] {
        Tmpl::Res(j) | Tmpl::Move(j) => entries[*j as usize].flags.bits & FRESH != 0,
        Tmpl::In(_) | Tmpl::Pin(_) | Tmpl::Part(..) | Tmpl::Const(_) | Tmpl::Seq(..) | Tmpl::Map(..) | Tmpl::Slice(_) | Tmpl::Null => false,
    })
}

impl List {
    pub fn new(ninputs: u32, npins: u32, consts: Vec<Py<PyAny>>, mut entries: Vec<Entry>, outputs: Vec<Tmpl>) -> Result<List, String> {
        let mut scope = Scope { ninputs, npins, consts: &consts, live: Vec::with_capacity(entries.len()), class: Vec::with_capacity(entries.len()) };
        for k in 0..entries.len() {
            let e = &entries[k];
            if let Callee::Drop(slots) = &e.callee {
                for j in slots.iter() {
                    if !scope.live.get(*j as usize).is_some_and(|held| *held) {
                        return Err(format!("entry {k}: drops r{j}, which is not an earlier entry's held result"));
                    }
                    scope.live[*j as usize] = false;
                }
                scope.live.push(false);
                scope.class.push(None);
                continue;
            }
            let decided = scope.entry(k, e).map_err(|why| format!("entry {k}: {why}"))?;
            let release = releases(e);
            for r in release.iter() {
                if let Release::Slot(j) = r {
                    scope.live[*j as usize] = false;
                }
            }
            scope.live.push(e.out.is_some());
            scope.class.push(e.assert.as_ref().and_then(|a| a.class.as_ref()).map(|c| c.as_ptr() as usize));
            (entries[k].decided, entries[k].release) = (decided, release);
        }
        let mut named = Vec::new();
        for (i, t) in outputs.iter().enumerate() {
            scope.tmpl(t, false).map_err(|why| format!("output {i}: {why}"))?;
            let first = named.len();
            t.leaves(&mut named);
            if named[first..].iter().any(|l| matches!(l, Tmpl::Move(_))) {
                return Err(format!("output {i}: an output moves nothing"));
            }
        }
        // Every slot has one stated release: a move, a drop, or (it is an
        // output) the end of the run.
        let kept = |j: usize| named.iter().any(|l| matches!(l, Tmpl::Res(s) | Tmpl::Part(s, _) if *s as usize == j));
        if let Some(j) = (0..entries.len()).find(|j| scope.live[*j] && !kept(*j)) {
            return Err(format!("r{j} is never released: it is not moved, not dropped and no output"));
        }
        let first_irreversible = entries.iter().position(|e| !reversible(e, &entries)).unwrap_or(entries.len());
        let max_args = entries.iter().map(|e| e.args.len()).max().unwrap_or(0);
        Ok(List { entries: entries.into(), consts: consts.into(), ninputs, npins, outputs: outputs.into(), first_irreversible, max_args, stopped: Mutex::new(None) })
    }
}
