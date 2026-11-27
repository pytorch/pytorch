// Call lists: the body's own C calls in program order, each the callee by
// identity (or the slot an instruction runs) with templates of its operands.
// Running a list (run.rs) makes the same calls again on operands built from
// the templates. Nothing in the executor makes or runs a list yet:
// `_calls_make`, `_calls_run` and `_calls_text` are for tests of hand-built
// lists.
//
// Slot j holds the result of entry j, from that entry (if it keeps a result:
// `out`) to the ONE place the list states for its release. That place also
// says what the slot is in the body, a temporary on its stack or a local:
//   ("move", j)  as an operand: the slot's last use, and the slot is a
//                TEMPORARY. The reference goes when the callee returns, with
//                the other operands and what was built for the call, in the
//                order the callee's instruction releases them (the table
//                ORDER of slots.rs: a call its arguments in order, a slice of
//                two bounds before its container, ...), before the result is
//                stored or dropped. A move inside a seq, map or slice belongs
//                to that container.
//   a drop entry releases its slots, in the order given, at its place among
//                the entries (a local overwritten, `del`, the end of the
//                body): the slot is a LOCAL.
//   an output    a slot that an output names stays to the end of the run: a
//                local.
// A result with `out` None is released after the operands (the POP_TOP that
// follows). A slot with none of the three, a use after a move or drop, and a
// second release are refused.
// A list that stops lets go of its temporaries before it returns or raises,
// as the unwind of the body's stack does (run.rs), and of no local: the live
// locals go to the caller with the stop (the body's frame keeps them while a
// handler runs).
//
// The spec `_calls_make` takes, of exact tuples, lists, strs and ints only
// (spec.rs decodes it; check.rs refuses what it does not understand):
//   (ninputs, npins, consts, entries, outputs)   outputs: templates, no move
//   entry     (callee, args, kwnames, out, at, flags, asserts, classes)
//             | ("drop", (slot, ...), at)
//   callee    ("vector", obj, label) | ("binary", "+=") | ("compare", "<")
//             | ("unary", "-") | ("getitem",) | ("setitem",) | ("delitem",)
//             | ("getattr", name) | ("setattr", name) | ("state", setter)
//   args      templates in the order of the instruction's operands: (lhs, rhs),
//             (container, key), setitem (value, container, key), setattr
//             (value, owner); a vector's positionals, then its keyword values
//   kwnames   None, or the tuple of keyword names, no two the same (vector only)
//   out       the entry's own index, or None: the result is not kept
//   at        the step of the body the call ran under
//   flags     "pure" | "draws" | "effect" | "state" | "opaque" | "covered"
//             | "fresh" | ("writes", (operand, ...)) | ("rekeys", (operand, ...))
//   asserts   ("key", tensor): the result has the whole key of this tensor
//             (exec/tensor_key.rs; a tensor with no key is refused)
//             | ("class", type): the result's exact class | ("count", n): an
//             exact tuple or struct sequence of n | ("is", operand)
//   classes   parallel to args: the exact class each operand had at the
//             trace. A type for in, pin, res, move, part, const; None for
//             null; for a seq, map or slice the tuple of its parts' classes.
//   template  ("in", k) | ("pin", i) | ("res", j) | ("move", j)
//             | ("part", j, (i, ...)) | ("const", i)
//             | ("seq", "list" | "tuple", [template, ...])
//             | ("map", (key, ...), [template, ...])
//             | ("slice", template, template, template) | ("null",)
//
// Classes. Before an entry's call every operand's class is compared with its
// pin; another class is the stop "class" and the entry has not run. So an
// operand that is a result needs a class its producer asserts: ("class", t)
// or a key. A part needs its producer asserted to be a tuple or a struct
// sequence type, and is read off exactly those. A slot callee (binary,
// compare, unary, the subscripts, the attributes) is taken only if the
// pinned classes of ALL its operands are a line of the license table
// (slots.rs: the table, why each line runs C code only, and what is in no
// line: a subclass of Tensor, a class written in Python, an nn.Module, a
// dict as a container, a store to an attribute; rows of the rule table will
// widen it), and if what csrc/calls_slots.c resolves from the same classes
// agrees. Every class an operand of such an entry is pinned to is compared
// by version at every run, so a class changed since is the stop "class" too.
//
// NOT decided here: which C callables may be vector entries at all
// (`type.__call__`, `exec`, `sorted`, `operator.add` are C callables by
// their type), and what such a callee does with operands of the pinned
// classes. Both are the license of a rule (the callee's row of the table),
// which the recorder will ask; until then lists are made by tests only.

mod again;
mod check;
mod entries;
mod finish;
mod known;
mod origins;
mod reading;
pub mod record;
mod run;
pub mod shadow;
mod slots;
mod spec;
pub mod witness;

use std::sync::Mutex;

use pyo3::ffi::{self, PyObject};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString, PyTuple};

use crate::cfg::ir::{BINARY_OPS, COMPARE_OPS};
use crate::cfg::opcodes;
use crate::exec::tensor_key::TensorKey;

pub use run::{_calls_run, _witness, Stop};
pub use spec::_calls_make;

// The unary instructions the executor runs through tr_generic: (text, opcode, oparg).
pub const UNARY_OPS: [(&str, u8, u32); 3] =
    [("-", opcodes::UNARY_NEGATIVE, 0), ("~", opcodes::UNARY_INVERT, 0), ("+", opcodes::CALL_INTRINSIC_1, 5)];

pub enum Callee {
    // A C callable and the name it is printed by, given by who makes the list.
    Vector(Py<PyAny>, Box<str>),
    // BINARY_OP's oparg; COMPARE_OP's kind; a row of UNARY_OPS.
    Binary(u32),
    Compare(u32),
    Unary(u8, u32),
    GetItem,
    SetItem,
    DelItem,
    GetAttr(Py<PyString>),
    SetAttr(Py<PyString>),
    // A setter of ambient state. Not run yet: refused when a list is made.
    State(Py<PyAny>),
    // No call: releases these slots, in this order.
    Drop(Box<[u32]>),
}

pub enum Tmpl {
    In(u32),
    Pin(u32),
    Res(u32),
    // The slot's last use: released when the callee returns.
    Move(u32),
    Part(u32, Box<[u32]>),
    Const(u32),
    // An exact list (true) or tuple, a dict, a slice: built anew per call.
    Seq(bool, Box<[Tmpl]>),
    Map(Box<[Py<PyAny>]>, Box<[Tmpl]>),
    Slice(Box<[Tmpl; 3]>),
    // An absent bound of a slice.
    Null,
}

impl Tmpl {
    // Built for the call: a new reference.
    pub fn made(&self) -> bool {
        matches!(self, Tmpl::Seq(..) | Tmpl::Map(..) | Tmpl::Slice(_))
    }

    // Every template under this one that is no container, in the order a run
    // reads them (the order of an entry's flat `classes`).
    pub fn leaves<'a>(&'a self, into: &mut Vec<&'a Tmpl>) {
        match self {
            Tmpl::Seq(_, parts) | Tmpl::Map(_, parts) => parts.iter().for_each(|p| p.leaves(into)),
            Tmpl::Slice(p) => p.iter().for_each(|p| p.leaves(into)),
            Tmpl::In(_) | Tmpl::Pin(_) | Tmpl::Res(_) | Tmpl::Move(_) | Tmpl::Part(..) | Tmpl::Const(_) | Tmpl::Null => into.push(self),
        }
    }
}

// Bit i of Flags::bits is FLAG_NAMES[i]; the ones the engine acts on by name:
pub const FLAG_NAMES: [&str; 7] = ["pure", "draws", "effect", "state", "opaque", "covered", "fresh"];
pub const PURE: u16 = 1;
pub const DRAWS: u16 = 2;
pub const EFFECT: u16 = 4;
pub const STATE: u16 = 8;
pub const OPAQUE: u16 = 16;
pub const FRESH: u16 = 64;

#[derive(Default)]
pub struct Flags {
    pub bits: u16,
    // Operand positions.
    pub writes: Box<[u32]>,
    pub rekeys: Box<[u32]>,
}

// What Python read of the result at the trace: its key (with `class` the
// class the key names), its exact class, the length of a result tuple, the
// result being an operand.
#[derive(Default)]
pub struct Assert {
    pub key: Option<TensorKey>,
    pub class: Option<Py<PyAny>>,
    pub count: Option<usize>,
    pub same: Option<u32>,
}

// What an entry releases when its callee has returned, in this order.
pub enum Release {
    Slot(u32),
    // The argument at this position: a container built for the call.
    Arg(u32),
}

pub struct Entry {
    pub callee: Callee,
    pub args: Box<[Tmpl]>,
    pub kwnames: Option<Py<PyTuple>>,
    pub out: Option<u32>,
    pub at: u32,
    pub flags: Flags,
    pub assert: Option<Assert>,
    // One per leaf of `args` (Tmpl::leaves): its pinned class, None for Null.
    pub classes: Box<[Option<Py<PyAny>>]>,
    // Filled by List::new: the classes the operands of a slot callee are
    // pinned to, each with its version then; the releases behind the call.
    pub decided: Box<[(Py<PyAny>, u32)]>,
    pub release: Box<[Release]>,
}

// Made by List::new only (check.rs).
#[pyclass(frozen, module = "torch._rust", name = "CallList")]
pub struct List {
    entries: Box<[Entry]>,
    consts: Box<[Py<PyAny>]>,
    ninputs: u32,
    npins: u32,
    outputs: Box<[Tmpl]>,
    // By slot: the entry and the operand of it that move the slot (it is a
    // temporary of the body); None for a local and for no slot.
    moved: Box<[Option<(u32, u32)>]>,
    // The first entry that cannot be taken back; the count of entries if none.
    first_irreversible: usize,
    max_args: usize,
    // How the last `_calls_run` stopped, with the locals that were live, for
    // tests. Never locked while a reference is released: a stop is taken out
    // under the lock and dropped outside it.
    stopped: Mutex<Option<Stop>>,
}

#[pymethods]
impl List {
    /// None after a run that returned its outputs, else (entry, "raised" |
    /// "assert" | "part" | "class" | "refused", restartable, {slot: object})
    /// of the last `_calls_run`: the list keeps the locals that were live
    /// (the temporaries went with the stop) until the next run or `release()`.
    #[getter]
    fn stopped<'py>(&self, py: Python<'py>) -> PyResult<Option<(usize, &'static str, bool, Bound<'py, PyDict>)>> {
        let held = self.stopped.lock().unwrap().as_ref().map(|s| (s.entry, s.what.text(), s.restartable, s.live.iter().map(|(j, x)| (*j, x.clone_ref(py))).collect::<Vec<_>>()));
        let Some((entry, what, restartable, slots)) = held else { return Ok(None) };
        let live = PyDict::new(py);
        for (j, x) in slots {
            live.set_item(j, x)?;
        }
        Ok(Some((entry, what, restartable, live)))
    }

    /// Releases the locals a stopped run left, in slot order.
    fn release(&self) {
        let live = self.stopped.lock().unwrap().as_mut().map(|s| std::mem::take(&mut s.live));
        live.into_iter().flatten().for_each(drop);
    }
}

// One line per entry.
#[pyfunction]
pub fn _calls_text(list: &Bound<'_, List>) -> String {
    let (py, list) = (list.py(), list.get());
    (0..list.entries.len()).map(|k| list.line(py, k)).collect::<Vec<_>>().join("\n")
}

/// For tests and for who writes a recorder: the two tables of slots.rs.
/// (license, order, groups): license lines (name, kind, operators or names,
/// the operands' groups, why); order lines (kind, whether the key is a slice
/// of two bounds, instruction, operand positions in the order of their
/// release, in words); (group, what it is).
#[pyfunction]
#[allow(clippy::type_complexity)]
pub fn _calls_tables() -> (Vec<(&'static str, &'static str, Vec<&'static str>, Vec<String>, &'static str)>, Vec<(&'static str, bool, &'static str, Vec<usize>, &'static str)>, Vec<(String, &'static str)>) {
    let license = slots::LICENSE.iter().map(|l| (l.name, l.kind, l.ops.to_vec(), l.operands.iter().map(|is| format!("{is:?}")).collect(), l.why));
    let order = slots::ORDER.iter().map(|o| (o.kind, o.two_bounds, o.instruction, o.operands.to_vec(), o.how));
    let groups = slots::EVERY_IS.iter().map(|is| (format!("{is:?}"), is.sentence()));
    (license.collect(), order.collect(), groups.collect())
}

// Whether `x` is an exact int, float, bool, str or bytes, or None: the values
// whose hash, equality and repr are C code that runs nothing of the program.
// The one definition, for the keys of a map (check.rs) and for what a line
// prints by value.
pub unsafe fn plain(x: *mut PyObject) -> bool {
    unsafe {
        ffi::PyLong_CheckExact(x) != 0
            || ffi::PyFloat_CheckExact(x) != 0
            || ffi::PyBool_Check(x) != 0
            || ffi::PyUnicode_CheckExact(x) != 0
            || ffi::PyBytes_CheckExact(x) != 0
            || x == ffi::Py_None()
    }
}

// Whether `x` is a value a list may hold as a constant: of the rule table's
// closed list of immutable classes, a constant class of torch, or an exact
// tuple of those (at most four deep; a deeper one has no origin). The one
// definition; `plain` above is the part of it whose hash and repr a list
// itself runs.
pub unsafe fn immutable(x: *mut PyObject, depth: u8) -> bool {
    unsafe {
        let tp = ffi::Py_TYPE(x);
        if crate::exec::rules::Group::Immutable.holds(tp) || crate::exec::rules::Group::TorchConstant.holds(tp) {
            return true;
        }
        tp == &raw mut ffi::PyTuple_Type && depth < 4 && (0..ffi::PyTuple_GET_SIZE(x)).all(|i| immutable(ffi::PyTuple_GET_ITEM(x, i), depth + 1))
    }
}

// The repr of a plain value or of an exact tuple of those.
unsafe fn plain_text(x: *mut PyObject) -> Option<String> {
    unsafe {
        if ffi::PyTuple_CheckExact(x) != 0 {
            let n = ffi::PyTuple_GET_SIZE(x);
            let parts = (0..n).map(|i| plain_text(ffi::PyTuple_GET_ITEM(x, i))).collect::<Option<Vec<_>>>()?;
            return Some(format!("({}{})", parts.join(", "), if n == 1 { "," } else { "" }));
        }
        if !plain(x) {
            return None;
        }
        let r = Bound::from_owned_ptr_or_opt(Python::assume_attached(), ffi::PyObject_Repr(x))?;
        r.cast::<PyString>().ok()?.to_str().ok().map(str::to_owned)
    }
}

// A class by the name its type object carries (the last part of tp_name).
fn class_name(class: &Py<PyAny>) -> String {
    let name = unsafe { std::ffi::CStr::from_ptr((*(class.as_ptr() as *mut ffi::PyTypeObject)).tp_name) }.to_string_lossy();
    name.rsplit('.').next().unwrap_or("?").to_string()
}

impl List {
    fn tmpl(&self, t: &Tmpl) -> String {
        let all = |ts: &[Tmpl]| ts.iter().map(|t| self.tmpl(t)).collect::<Vec<_>>();
        let plain = |x: &Py<PyAny>| unsafe { plain_text(x.as_ptr()) };
        match t {
            Tmpl::In(k) => format!("in{k}"),
            Tmpl::Pin(i) => format!("pin{i}"),
            Tmpl::Res(j) => format!("r{j}"),
            Tmpl::Move(j) => format!("move r{j}"),
            Tmpl::Part(j, path) => format!("r{j}{}", path.iter().map(|i| format!(".{i}")).collect::<String>()),
            Tmpl::Const(i) => plain(&self.consts[*i as usize]).unwrap_or_else(|| format!("c{i}")),
            Tmpl::Seq(true, parts) => format!("[{}]", all(parts).join(", ")),
            Tmpl::Seq(false, parts) => format!("({}{})", all(parts).join(", "), if parts.len() == 1 { "," } else { "" }),
            Tmpl::Map(keys, parts) => {
                let pairs = keys.iter().zip(all(parts)).map(|(k, v)| format!("{}: {v}", plain(k).unwrap_or_else(|| "?".into())));
                format!("{{{}}}", pairs.collect::<Vec<_>>().join(", "))
            }
            Tmpl::Slice(p) => format!("{}:{}:{}", self.tmpl(&p[0]), self.tmpl(&p[1]), self.tmpl(&p[2])),
            Tmpl::Null => String::new(),
        }
    }

    pub fn line(&self, py: Python<'_>, k: usize) -> String {
        let e = &self.entries[k];
        let a = e.args.iter().map(|t| self.tmpl(t)).collect::<Vec<_>>();
        let arg = |i: usize| a.get(i).map_or("?", |s| s.as_str());
        let name = |n: &Py<PyString>| n.bind(py).to_string_lossy().into_owned();
        let call = match &e.callee {
            Callee::Vector(_, label) => {
                let names = e.kwnames.as_ref().map_or(vec![], |kw| kw.bind(py).iter().map(|n| n.to_string()).collect());
                let first = a.len().saturating_sub(names.len());
                let kw = names.iter().zip(&a[first..]).map(|(n, v)| format!("{n}={v}"));
                format!("{label}({})", a[..first].iter().cloned().chain(kw).collect::<Vec<_>>().join(", "))
            }
            Callee::Binary(op) => format!("{} {} {}", arg(0), BINARY_OPS.get(*op as usize).copied().unwrap_or("?"), arg(1)),
            Callee::Compare(op) => format!("{} {} {}", arg(0), COMPARE_OPS.get(*op as usize).copied().unwrap_or("?"), arg(1)),
            Callee::Unary(op, oparg) => format!("{}{}", UNARY_OPS.iter().find(|u| (u.1, u.2) == (*op, *oparg)).map_or("?", |u| u.0), arg(0)),
            Callee::GetItem => format!("{}[{}]", arg(0), arg(1)),
            Callee::SetItem => format!("{}[{}] = {}", arg(1), arg(2), arg(0)),
            Callee::DelItem => format!("del {}[{}]", arg(0), arg(1)),
            Callee::GetAttr(n) => format!("{}.{}", arg(0), name(n)),
            Callee::SetAttr(n) => format!("{}.{} = {}", arg(1), name(n), arg(0)),
            Callee::State(_) => format!("state({})", a.join(", ")),
            Callee::Drop(slots) => return format!("{k:>3}  drop{}  @{}", slots.iter().map(|j| format!(" r{j}")).collect::<String>(), e.at),
        };
        let classes = e.classes.iter().map(|c| c.as_ref().map_or("-".into(), class_name)).collect::<Vec<_>>().join(", ");
        let mut s = format!("{k:>3}  {}{call} of ({classes})  @{}", e.out.map_or(String::new(), |j| format!("r{j} = ")), e.at);
        let mut flags: Vec<String> = FLAG_NAMES.iter().enumerate().filter(|(i, _)| e.flags.bits & (1 << i) != 0).map(|(_, n)| n.to_string()).collect();
        for (what, operands) in [("writes", &e.flags.writes), ("rekeys", &e.flags.rekeys)] {
            if !operands.is_empty() {
                flags.push(format!("{what}({})", operands.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(", ")));
            }
        }
        if !flags.is_empty() {
            s += &format!(" [{}]", flags.join(" "));
        }
        if let Some(x) = &e.assert {
            s += &format!(" assert{}", x.text());
        }
        s
    }
}

impl Assert {
    pub fn text(&self) -> String {
        let class = match (&self.key, &self.class) {
            (Some(key), _) => format!(" key={key}"),
            (None, Some(class)) => format!(" class={}", class_name(class)),
            (None, None) => String::new(),
        };
        class + &self.count.map_or(String::new(), |n| format!(" count={n}")) + &self.same.map_or(String::new(), |i| format!(" is(operand {i})"))
    }
}
