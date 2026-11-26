// Call lists: the body's own C calls in program order, each the callee by
// identity (or the slot an instruction runs) with templates of its operands.
// Running a list (run.rs) makes the same calls again on operands built from
// the templates. Nothing in the executor makes or runs a list yet:
// `_calls_make`, `_calls_run` and `_calls_text` are for tests of hand-built
// lists.
//
// Slot j holds the result of entry j, from that entry (if it keeps a result:
// `out`) to the entry that names j in its `kills`.
//
// The spec `_calls_make` takes, of exact tuples, lists, strs and ints only
// (spec.rs decodes it; check.rs refuses what it does not understand):
//   (ninputs, npins, consts, entries, outputs)   outputs: templates
//   entry     (callee, args, kwnames, out, at, flags, asserts, kills)
//   callee    ("vector", obj, label) | ("binary", "+=") | ("compare", "<")
//             | ("unary", "-") | ("getitem",) | ("setitem",) | ("delitem",)
//             | ("getattr", name) | ("setattr", name) | ("state", setter)
//   args      templates in the order of the instruction's operands: (lhs, rhs),
//             (container, key), setitem (value, container, key), setattr
//             (value, owner); a vector's positionals, then its keyword values
//   kwnames   None, or the tuple of keyword names (vector only)
//   out       the entry's own index, or None: the result is not kept
//   at        the step of the body the call ran under
//   flags     "pure" | "draws" | "effect" | "state" | "opaque" | "covered"
//             | "fresh" | ("writes", (operand, ...)) | ("rekeys", (operand, ...))
//   asserts   ("key", (field, ...), tensor) with fields "sizes", "dtype",
//             "device", "requires_grad" read off the tensor | ("count", n)
//             | ("is", operand)
//   kills     indices of earlier entries
//   template  ("in", k) | ("pin", i) | ("res", j) | ("part", j, (i, ...))
//             | ("const", i) | ("seq", "list" | "tuple", [template, ...])
//             | ("map", (key, ...), [template, ...])
//             | ("slice", template, template, template) | ("null",)

mod check;
mod run;
mod spec;
pub mod witness;

use std::sync::atomic::{AtomicU64, Ordering};

use pyo3::ffi::{self, PyObject};
use pyo3::prelude::*;
use pyo3::types::{PyString, PyTuple};

use crate::bindings::ffi::{device_name, dtype_name, TensorMeta};
use crate::cfg::ir::{BINARY_OPS, COMPARE_OPS};
use crate::cfg::opcodes;

pub use run::{_calls_run, _witness};
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
}

pub enum Tmpl {
    In(u32),
    Pin(u32),
    Res(u32),
    Part(u32, Box<[u32]>),
    Const(u32),
    // An exact list (true) or tuple, a dict, a slice: built anew per call.
    Seq(bool, Box<[Tmpl]>),
    Map(Box<[Py<PyAny>]>, Box<[Tmpl]>),
    Slice(Box<[Tmpl; 3]>),
    // An absent bound of a slice.
    Null,
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

pub const KEY_FIELDS: [&str; 4] = ["sizes", "dtype", "device", "requires_grad"];

// What Python read of the result at the trace: fields of its key (bit i of
// `mask` is KEY_FIELDS[i], the traced values are in `key`), the length of a
// result tuple, the result being an operand.
#[derive(Default)]
pub struct Assert {
    pub mask: u8,
    pub key: TensorMeta,
    pub count: Option<usize>,
    pub same: Option<u32>,
}

pub struct Entry {
    pub callee: Callee,
    pub args: Box<[Tmpl]>,
    pub kwnames: Option<Py<PyTuple>>,
    pub out: Option<u32>,
    pub at: u32,
    pub flags: Flags,
    pub assert: Option<Assert>,
    pub kills: Box<[u32]>,
}

// Made by List::new only (check.rs).
#[pyclass(frozen, module = "torch._rust", name = "CallList")]
pub struct List {
    entries: Box<[Entry]>,
    consts: Box<[Py<PyAny>]>,
    ninputs: u32,
    npins: u32,
    outputs: Box<[Tmpl]>,
    // The first entry that cannot be taken back; the count of entries if none.
    first_irreversible: usize,
    max_args: usize,
    // How the last `_calls_run` ended, for tests (see Stop::code).
    stopped: AtomicU64,
}

#[pymethods]
impl List {
    // None after a run that returned its outputs, else (entry, "raised" |
    // "assert" | "part", restartable) of the last `_calls_run`.
    #[getter]
    fn stopped(&self) -> Option<(usize, &'static str, bool)> {
        let code = self.stopped.load(Ordering::Relaxed);
        (code != 0).then(|| ((code >> 8) as usize - 1, ["raised", "assert", "part"][(code >> 1 & 3) as usize], code & 1 != 0))
    }
}

// One line per entry.
#[pyfunction]
pub fn _calls_text(list: &Bound<'_, List>) -> String {
    let (py, list) = (list.py(), list.get());
    (0..list.entries.len()).map(|k| list.line(py, k)).collect::<Vec<_>>().join("\n")
}

// The repr of an exact int, float, bool, str or None, or of an exact tuple of
// those: values whose repr runs no code of the program.
unsafe fn plain(x: *mut PyObject) -> Option<String> {
    unsafe {
        if ffi::PyTuple_CheckExact(x) != 0 {
            let n = ffi::PyTuple_GET_SIZE(x);
            let parts = (0..n).map(|i| plain(ffi::PyTuple_GET_ITEM(x, i))).collect::<Option<Vec<_>>>()?;
            return Some(format!("({}{})", parts.join(", "), if n == 1 { "," } else { "" }));
        }
        let exact = ffi::PyLong_CheckExact(x) != 0 || ffi::PyFloat_CheckExact(x) != 0 || ffi::PyBool_Check(x) != 0;
        if !exact && ffi::PyUnicode_CheckExact(x) == 0 && x != ffi::Py_None() {
            return None;
        }
        let r = Bound::from_owned_ptr_or_opt(Python::assume_attached(), ffi::PyObject_Repr(x))?;
        r.cast::<PyString>().ok()?.to_str().ok().map(str::to_owned)
    }
}

impl List {
    fn tmpl(&self, t: &Tmpl) -> String {
        let all = |ts: &[Tmpl]| ts.iter().map(|t| self.tmpl(t)).collect::<Vec<_>>();
        let plain = |x: &Py<PyAny>| unsafe { plain(x.as_ptr()) };
        match t {
            Tmpl::In(k) => format!("in{k}"),
            Tmpl::Pin(i) => format!("pin{i}"),
            Tmpl::Res(j) => format!("r{j}"),
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
        };
        let mut s = format!("{k:>3}  {}{call}  @{}", e.out.map_or(String::new(), |j| format!("r{j} = ")), e.at);
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
        if !e.kills.is_empty() {
            s += &format!(" kills{}", e.kills.iter().map(|j| format!(" r{j}")).collect::<String>());
        }
        s
    }
}

impl Assert {
    pub fn text(&self) -> String {
        let m = &self.key;
        let fields = [format!("{:?}", m.sizes), dtype_name(m.dtype), device_name(m.device_type, m.device_index), m.requires_grad.to_string()];
        let key: String = (0..4).filter(|i| self.mask & (1 << i) != 0).map(|i| format!(" {}={}", KEY_FIELDS[i], fields[i])).collect();
        key + &self.count.map_or(String::new(), |n| format!(" count={n}")) + &self.same.map_or(String::new(), |i| format!(" is(operand {i})"))
    }
}
