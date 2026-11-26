// What the instructions under a trace are, counted (TORCH_RUST_GATE_HIST).
//
// A measurement, off unless the variable is set (read once; unset, empty or
// 0: off): which instructions a traced body and the frames under it run, on
// operands of which classes. It says how many rules a table keyed by
// (instruction, class) needs before one is written.
//
// What it is given
//   - by region.rs, when a trace begins on this thread: begin();
//   - by run.rs, before an instruction of a traced body or of a frame under
//     it runs: step(), with the instruction, its operands as the objects
//     they are (NULL where a value is unset), what it calls if it is a call,
//     whether the frame is one under the body, and whether the executor's
//     inline path ran it (run_blocks' Fast steps, which never come to
//     guarded()).
// What it guarantees
//   - a row per key: (the body's own or a frame under it; what; the exact
//     class of each operand in order). `what` is the Op's kind, with the
//     operator for Binary and Compare; the opcode's name for a generic
//     instruction; for a call "call py", "call c", "call class" or "call
//     instance of" (a Python function, C code, a class, an object that is
//     called) and the callee's qualified name. Per row: how many traces met
//     it, how many instructions, how many of those on the inline path.
//     _gate_report() gives the rows of this thread and the number of its
//     traces that ran an instruction;
//   - with TORCH_RUST_GATE_HIST_DIR naming a directory, the process leaves
//     the report of its main thread there as <pid>.json when the interpreter
//     is finalized (Py_AtExit: no Python runs for it);
//   - it runs no code of the program, and none of Python's: a class is
//     named from its type object (tp_name; the qualified name and the
//     `__module__` in the dict of a class Python code made), a callee by
//     exact type from its own fields (a function, a builtin, a bound
//     method, a descriptor, a class), anything else by its class;
//   - with the variable unset, on() is false and nothing else here runs.
// What it costs while it is on
//   - every class it has named stays alive (the table is keyed by address).

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::CStr;
use std::sync::LazyLock;

use pyo3::ffi;
use pyo3::ffi::PyObject;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::cfg::ir::Op;
use crate::cfg::opcodes;

static ON: LazyLock<bool> = LazyLock::new(|| std::env::var_os("TORCH_RUST_GATE_HIST").is_some_and(|v| !v.is_empty() && v != "0"));

#[inline]
pub fn on() -> bool {
    *ON
}

// BINARY_OP's operators, by oparg.
const BINARY: [&str; 26] = [
    "Binary(+)", "Binary(&)", "Binary(//)", "Binary(<<)", "Binary(@)", "Binary(*)", "Binary(%)", "Binary(|)", "Binary(**)", "Binary(>>)", "Binary(-)", "Binary(/)",
    "Binary(^)", "Binary(+=)", "Binary(&=)", "Binary(//=)", "Binary(<<=)", "Binary(@=)", "Binary(*=)", "Binary(%=)", "Binary(|=)", "Binary(**=)", "Binary(>>=)",
    "Binary(-=)", "Binary(/=)", "Binary(^=)",
];
// Rich comparisons, by Py_LT .. Py_GE.
const COMPARE: [&str; 6] = ["Compare(<)", "Compare(<=)", "Compare(==)", "Compare(!=)", "Compare(>)", "Compare(>=)"];

#[derive(Default)]
struct Table {
    // Interned texts (what an instruction is, a class's name), and their ids.
    texts: Vec<String>,
    ids: HashMap<String, u32>,
    // A class by address: its text's id, and the class, kept alive.
    classes: HashMap<usize, (u32, Py<PyAny>)>,
    // [under, what, classes...] -> [traces, instructions, inline, the last trace that met it].
    rows: HashMap<Vec<u32>, [u64; 4]>,
    key: Vec<u32>,
    // Traces that ran an instruction; whether the one running has yet.
    traces: u64,
    fresh: bool,
}

thread_local! {
    static TABLE: RefCell<Table> = RefCell::new(Table::default());
}

impl Table {
    fn text(&mut self, s: &str) -> u32 {
        if let Some(id) = self.ids.get(s) {
            return *id;
        }
        let id = self.texts.len() as u32;
        self.texts.push(s.to_string());
        self.ids.insert(s.to_string(), id);
        id
    }

    unsafe fn class(&mut self, tp: *mut ffi::PyTypeObject) -> u32 {
        if let Some((id, _)) = self.classes.get(&(tp as usize)) {
            return *id;
        }
        let id = self.text(&class_name(tp));
        let kept = Bound::from_borrowed_ptr(Python::assume_attached(), tp as *mut PyObject).unbind();
        self.classes.insert(tp as usize, (id, kept));
        id
    }
}

// "module.Qualified.Name" of a class Python code made; tp_name of any other
// (it carries the module where the class has one).
unsafe fn class_name(tp: *mut ffi::PyTypeObject) -> String {
    let plain = CStr::from_ptr((*tp).tp_name).to_string_lossy().into_owned();
    if ffi::PyType_HasFeature(tp, ffi::Py_TPFLAGS_HEAPTYPE) == 0 {
        return plain;
    }
    let qualname = ffi::PyType_GetQualName(tp);
    // The class's own `__module__`: its dict's keys are exact strs.
    let mut module = None;
    let (mut at, mut key, mut value) = (0, std::ptr::null_mut(), std::ptr::null_mut());
    while module.is_none() && !(*tp).tp_dict.is_null() && ffi::PyDict_Next((*tp).tp_dict, &mut at, &mut key, &mut value) != 0 {
        if text(key).as_deref() == Some("__module__") {
            module = text(value);
        }
    }
    let name = match (module, text(qualname)) {
        (Some(m), Some(q)) => format!("{m}.{q}"),
        (None, Some(q)) => q,
        _ => plain,
    };
    ffi::Py_XDECREF(qualname);
    ffi::PyErr_Clear();
    name
}

// The text of an exact str (borrowed, NULL allowed).
unsafe fn text(s: *mut PyObject) -> Option<String> {
    if s.is_null() || ffi::PyUnicode_CheckExact(s) == 0 {
        return None;
    }
    let mut n = 0;
    let p = ffi::PyUnicode_AsUTF8AndSize(s, &mut n);
    (!p.is_null()).then(|| String::from_utf8_lossy(std::slice::from_raw_parts(p as *const u8, n as usize)).into_owned())
}

// What a call calls, by kind and name: a function ("py") or a builtin ("c")
// by its module and qualified name, a builtin bound to an object and a
// descriptor ("c") by the class and the name, a bound method by its
// function, a class by its own name; anything else (an instance with
// __call__, a partial) by its class.
unsafe fn callee_name(f: *mut PyObject) -> String {
    if f.is_null() {
        return "?".to_string();
    }
    let tp = ffi::Py_TYPE(f);
    if tp == &raw mut super::ffi::PyMethod_Type {
        return callee_name(super::ffi::tr_bound_function(f));
    }
    if ffi::PyType_Check(f) != 0 {
        return format!("class {}", class_name(f as *mut ffi::PyTypeObject));
    }
    if tp == &raw mut ffi::PyMethodDescr_Type || tp == &raw mut ffi::PyWrapperDescr_Type || tp == &raw mut ffi::PyClassMethodDescr_Type {
        let d = f as *mut ffi::PyDescrObject;
        return format!("c {}.{}", class_name((*d).d_type), text((*d).d_name).unwrap_or_else(|| "?".to_string()));
    }
    let (kind, module, name) = if tp == &raw mut ffi::PyCFunction_Type {
        let c = f as *mut ffi::PyCFunctionObject;
        let name = CStr::from_ptr((*(*c).m_ml).ml_name).to_string_lossy().into_owned();
        let owner = (*c).m_self;
        if !owner.is_null() && ffi::PyModule_Check(owner) == 0 {
            let class = if ffi::PyType_Check(owner) != 0 { owner as *mut ffi::PyTypeObject } else { ffi::Py_TYPE(owner) };
            return format!("c {}.{name}", class_name(class));
        }
        ("c", text((*c).m_module), name)
    } else if tp == &raw mut ffi::PyFunction_Type {
        let function = f as *mut ffi::PyFunctionObject;
        ("py", text((*function).func_module), text((*function).func_qualname).unwrap_or_else(|| "?".to_string()))
    } else {
        return format!("instance of {}", class_name(tp));
    };
    match module {
        Some(m) => format!("{kind} {m}.{name}"),
        None => format!("{kind} {name}"),
    }
}

// What an instruction is, but for a call.
fn kind(op: &Op) -> &'static str {
    match op {
        Op::Null => "Null",
        Op::Unknown => "Unknown",
        Op::Param(_) => "Param",
        Op::Const(_) => "Const",
        Op::Project(_) => "Project",
        Op::ToBool => "ToBool",
        Op::IsNone => "IsNone",
        Op::LoadLocal { .. } => "LoadLocal",
        Op::LoadLocalChecked { .. } => "LoadLocalChecked",
        Op::StoreLocal { .. } => "StoreLocal",
        Op::DeleteLocal { .. } => "DeleteLocal",
        Op::LoadGlobal(_) => "LoadGlobal",
        Op::LoadGlobalIn { .. } => "LoadGlobalIn",
        Op::LoadAttr(_) => "LoadAttr",
        Op::LoadMethod(_) => "LoadMethod",
        Op::StoreAttr(_) => "StoreAttr",
        Op::Call { .. } => "Call",
        Op::Binary(k) => BINARY.get(*k as usize).copied().unwrap_or("Binary(?)"),
        Op::Compare(k) => COMPARE.get(*k as usize).copied().unwrap_or("Compare(?)"),
        Op::BuildTuple(_) => "BuildTuple",
        Op::GetIter => "GetIter",
        Op::MakeFunction(_) => "MakeFunction",
        Op::Yield => "Yield",
        Op::Resume => "Resume",
        Op::IterNext => "IterNext",
        Op::Send => "Send",
        Op::EvalBreaker => "EvalBreaker",
        Op::MakeCell { .. } => "MakeCell",
        Op::CopyFreeVars(_) => "CopyFreeVars",
        Op::LoadDeref { .. } => "LoadDeref",
        Op::StoreDeref { .. } => "StoreDeref",
        Op::DeleteDeref { .. } => "DeleteDeref",
        Op::LoadFromDictOrDeref { .. } => "LoadFromDictOrDeref",
        Op::Bytecode { opcode, .. } => opcodes::name(*opcode),
        Op::Aten { .. } => "Aten",
        Op::Lit(_) => "Lit",
        Op::Pinned(_) => "Pinned",
        Op::Expect(_) => "Expect",
        Op::Autograd(_) => "Autograd",
        Op::Region(_) => "Region",
        Op::Removed => "Removed",
    }
}

// A trace begins on this thread.
pub fn begin() {
    if on() {
        TABLE.with_borrow_mut(|t| t.fresh = true);
        static AT_EXIT: std::sync::Once = std::sync::Once::new();
        AT_EXIT.call_once(|| unsafe {
            ffi::Py_AtExit(Some(dump));
        });
    }
}

// Leaves this thread's report in TORCH_RUST_GATE_HIST_DIR, if that is set.
// Called when the interpreter is gone: it touches the table's texts only.
extern "C" fn dump() {
    let Some(dir) = std::env::var_os("TORCH_RUST_GATE_HIST_DIR") else { return };
    let quoted = |s: &str| {
        let mut out = String::from("\"");
        for c in s.chars() {
            match c {
                '"' | '\\' => out.extend(['\\', c]),
                c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                c => out.push(c),
            }
        }
        out + "\""
    };
    let _ = TABLE.try_with(|t| {
        let t = t.borrow();
        let rows: Vec<String> = t
            .rows
            .iter()
            .map(|(key, row)| {
                let classes: Vec<String> = key[2..].iter().map(|id| quoted(&t.texts[*id as usize])).collect();
                format!("[{}, {}, [{}], {}, {}, {}]", key[0] != 0, quoted(&t.texts[key[1] as usize]), classes.join(", "), row[0], row[1], row[2])
            })
            .collect();
        let text = format!("{{\"traces\": {}, \"rows\": [{}]}}", t.traces, rows.join(", "));
        let _ = std::fs::write(std::path::Path::new(&dir).join(format!("{}.json", std::process::id())), text);
    });
}

// An instruction of a traced body (`under`: of a frame under it) is about
// to run on `operands` (borrowed, NULL allowed; for a call: what follows
// the callable); `callee`: what it calls, if it is a call; `inline`: the
// executor's inline path runs it.
pub unsafe fn step(under: bool, op: &Op, operands: &[*mut PyObject], callee: Option<*mut PyObject>, inline: bool) {
    TABLE.with_borrow_mut(|t| {
        if std::mem::take(&mut t.fresh) {
            t.traces += 1;
        }
        let what = match callee {
            Some(f) => {
                let name = format!("call {}", callee_name(f));
                t.text(&name)
            }
            None => t.text(kind(op)),
        };
        let mut key = std::mem::take(&mut t.key);
        key.clear();
        key.push(under as u32);
        key.push(what);
        for o in operands {
            let id = if o.is_null() { t.text("NULL") } else { t.class(ffi::Py_TYPE(*o)) };
            key.push(id);
        }
        let now = t.traces;
        let row = match t.rows.get_mut(key.as_slice()) {
            Some(row) => row,
            None => t.rows.entry(key.clone()).or_insert([0; 4]),
        };
        if row[3] != now {
            row[3] = now;
            row[0] += 1;
        }
        row[1] += 1;
        row[2] += inline as u64;
        t.key = key;
    });
}

/// What the instructions under this thread's traces were, counted
/// (``TORCH_RUST_GATE_HIST=1``; empty without it): ``{"traces": n, "rows":
/// [(under, what, classes, traces, instructions, inline), ...]}``. ``under``:
/// in a frame under a traced body, not in the body itself; ``what``: the
/// kind of instruction, or ``"call "`` and the callee's qualified name;
/// ``classes``: the exact class of each operand; ``traces``: how many traces
/// met the row; ``instructions``: how many instructions; ``inline``: how
/// many of those the executor ran without asking anything.
#[pyfunction]
pub fn _gate_report(py: Python<'_>) -> PyResult<Py<PyDict>> {
    TABLE.with_borrow(|t| {
        let rows = PyList::empty(py);
        for (key, row) in &t.rows {
            let classes = PyTuple::new(py, key[2..].iter().map(|id| t.texts[*id as usize].as_str()))?;
            rows.append((key[0] != 0, t.texts[key[1] as usize].as_str(), classes, row[0], row[1], row[2]))?;
        }
        let out = PyDict::new(py);
        out.set_item("traces", t.traces)?;
        out.set_item("rows", rows)?;
        Ok(out.unbind())
    })
}
