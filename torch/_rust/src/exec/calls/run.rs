// Runs a call list. Per entry: the operands are built from the templates into
// one argument array, the callee is called as the executor calls it for the
// instruction (the same helpers), the result goes to the entry's slot, the
// entry's assert is evaluated, and the slots the entry kills are released. No
// frame, no gate, no dispatch hook.
//
// A run ends with the tuple of its outputs, or with a Stop:
//   Raised  An entry raised: the callee's own exception is set. What entries
//           before it did stays done. `restartable` says whether every entry up
//           to and with it can be taken back: the caller's license to make it
//           a miss instead.
//   Assert  A result does not have what the trace read of it.
//   Part    A part of a result is not there: no tuple, or too short. The entry
//           that wanted it has not run.
// For Assert and Part: restartable -> no exception is set (a miss); else a
// RuntimeError naming the entry is set, never a value.
// In every case the slots are released before the run returns.

use std::os::raw::c_int;
use std::sync::atomic::Ordering;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::ffi::{self, PyObject};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use super::super::ffi::{tr_binary_op, tr_call, tr_generic};
use super::{witness, Assert, Callee, Entry, List, Tmpl};
use crate::bindings::ffi::op_hook::{tensor_ident, TensorIdent};
use crate::bindings::ffi::{tensor_by_type, tensor_meta, TensorMeta};
use crate::cfg::opcodes;

#[derive(Clone, Copy, PartialEq)]
pub enum What {
    Raised,
    Assert,
    Part,
}

pub struct Stop {
    // The entry; the count of entries for a part an output wanted.
    pub entry: usize,
    pub what: What,
    pub restartable: bool,
}

impl Stop {
    // For List::stopped: never 0.
    fn code(&self) -> u64 {
        ((self.entry as u64 + 1) << 8) | ((self.what as u64) << 1) | self.restartable as u64
    }
}

struct Run<'a> {
    list: &'a List,
    inputs: &'a [*mut PyObject],
    pins: &'a [*mut PyObject],
    slots: Vec<*mut PyObject>,
    // What was built for the call in work: released behind it.
    made: Vec<*mut PyObject>,
}

unsafe fn made(x: *mut PyObject, into: &mut Vec<*mut PyObject>) -> Result<*mut PyObject, What> {
    if x.is_null() {
        return Err(What::Raised);
    }
    into.push(x);
    Ok(x)
}

impl Run<'_> {
    // A borrowed pointer (null for Tmpl::Null).
    unsafe fn operand(&mut self, t: &Tmpl) -> Result<*mut PyObject, What> {
        unsafe {
            Ok(match t {
                Tmpl::In(k) => self.inputs[*k as usize],
                Tmpl::Pin(i) => self.pins[*i as usize],
                Tmpl::Res(j) => self.slots[*j as usize],
                Tmpl::Const(i) => self.list.consts[*i as usize].as_ptr(),
                Tmpl::Null => std::ptr::null_mut(),
                Tmpl::Part(j, path) => {
                    let mut x = self.slots[*j as usize];
                    for i in path.iter() {
                        // A tuple or a struct sequence, read at the C level.
                        if ffi::PyTuple_Check(x) == 0 || *i as ffi::Py_ssize_t >= ffi::PyTuple_GET_SIZE(x) {
                            return Err(What::Part);
                        }
                        x = ffi::PyTuple_GET_ITEM(x, *i as ffi::Py_ssize_t);
                    }
                    x
                }
                Tmpl::Seq(list, parts) => {
                    let n = parts.len() as ffi::Py_ssize_t;
                    let seq = made(if *list { ffi::PyList_New(n) } else { ffi::PyTuple_New(n) }, &mut self.made)?;
                    for (k, p) in parts.iter().enumerate() {
                        let item = self.operand(p)?;
                        ffi::Py_INCREF(item);
                        if *list {
                            ffi::PyList_SET_ITEM(seq, k as ffi::Py_ssize_t, item);
                        } else {
                            ffi::PyTuple_SET_ITEM(seq, k as ffi::Py_ssize_t, item);
                        }
                    }
                    seq
                }
                Tmpl::Map(keys, parts) => {
                    let map = made(ffi::PyDict_New(), &mut self.made)?;
                    for (key, p) in keys.iter().zip(parts.iter()) {
                        let value = self.operand(p)?;
                        if ffi::PyDict_SetItem(map, key.as_ptr(), value) < 0 {
                            return Err(What::Raised);
                        }
                    }
                    map
                }
                Tmpl::Slice(p) => {
                    let (start, stop, step) = (self.operand(&p[0])?, self.operand(&p[1])?, self.operand(&p[2])?);
                    made(ffi::PySlice_New(start, stop, step), &mut self.made)?
                }
            })
        }
    }

    unsafe fn release_made(&mut self) {
        for m in self.made.drain(..) {
            unsafe { ffi::Py_DECREF(m) };
        }
    }

    // The call of entry `e` on `argv` (for a vector: null, the callee, then the
    // operands, the form tr_call takes). Null with no exception set where the
    // instruction has no result.
    unsafe fn call(e: &Entry, argv: &[*mut PyObject]) -> Result<*mut PyObject, What> {
        unsafe {
            let generic = |opcode: u8, oparg: u32| {
                let mut out = std::ptr::null_mut();
                let rc = tr_generic(std::ptr::null_mut(), opcode as c_int, oparg as c_int, argv.as_ptr(), &mut out);
                if rc == 0 { Ok(out) } else { Err(What::Raised) }
            };
            let result = |r: *mut PyObject| if r.is_null() { Err(What::Raised) } else { Ok(r) };
            match &e.callee {
                Callee::Vector(..) => {
                    let kwnames = e.kwnames.as_ref().map_or(std::ptr::null_mut(), |k| k.as_ptr());
                    result(tr_call(argv.as_ptr(), (argv.len() - 2) as c_int, kwnames))
                }
                Callee::Binary(op) => result(tr_binary_op(*op as c_int, argv[0], argv[1])),
                Callee::Compare(op) => result(ffi::PyObject_RichCompare(argv[0], argv[1], *op as c_int)),
                Callee::Unary(opcode, oparg) => generic(*opcode, *oparg),
                Callee::GetItem => generic(opcodes::BINARY_SUBSCR, 0),
                Callee::SetItem => generic(opcodes::STORE_SUBSCR, 0),
                Callee::DelItem => generic(opcodes::DELETE_SUBSCR, 0),
                Callee::GetAttr(name) => result(ffi::PyObject_GetAttr(argv[0], name.as_ptr())),
                Callee::SetAttr(name) => {
                    if ffi::PyObject_SetAttr(argv[1], name.as_ptr(), argv[0]) < 0 { Err(What::Raised) } else { Ok(std::ptr::null_mut()) }
                }
                Callee::State(_) => unreachable!("refused when the list was made"),
            }
        }
    }

    unsafe fn entry(&mut self, k: usize, e: &Entry, argv: &mut Vec<*mut PyObject>, meta: &mut TensorMeta) -> Result<(), What> {
        unsafe {
            argv.clear();
            let first = if let Callee::Vector(f, _) = &e.callee {
                argv.extend([std::ptr::null_mut(), f.as_ptr()]);
                2
            } else {
                0
            };
            let mut built = Ok(());
            for t in e.args.iter() {
                match self.operand(t) {
                    Ok(x) => argv.push(x),
                    Err(what) => {
                        built = Err(what);
                        break;
                    }
                }
            }
            let r = built.and_then(|()| Self::call(e, argv));
            self.release_made();
            let r = r?;
            let holds = e.assert.as_ref().is_none_or(|a| a.holds(r, &argv[first..], meta));
            if e.out.is_some() {
                self.slots[k] = r;
            } else {
                ffi::Py_XDECREF(r);
            }
            if !holds {
                return Err(What::Assert);
            }
            for j in e.kills.iter() {
                ffi::Py_DECREF(std::mem::replace(&mut self.slots[*j as usize], std::ptr::null_mut()));
            }
            Ok(())
        }
    }

    unsafe fn outputs(&mut self) -> Result<*mut PyObject, What> {
        unsafe {
            let outputs = &self.list.outputs;
            let tuple = ffi::PyTuple_New(outputs.len() as ffi::Py_ssize_t);
            if tuple.is_null() {
                return Err(What::Raised);
            }
            for (k, t) in outputs.iter().enumerate() {
                match self.operand(t) {
                    Ok(x) => {
                        ffi::Py_INCREF(x);
                        ffi::PyTuple_SET_ITEM(tuple, k as ffi::Py_ssize_t, x);
                    }
                    Err(what) => {
                        ffi::Py_DECREF(tuple);
                        self.release_made();
                        return Err(what);
                    }
                }
            }
            self.release_made();
            Ok(tuple)
        }
    }
}

impl Assert {
    unsafe fn holds(&self, r: *mut PyObject, operands: &[*mut PyObject], meta: &mut TensorMeta) -> bool {
        unsafe {
            if self.count.is_some_and(|n| ffi::PyTuple_Check(r) == 0 || ffi::PyTuple_GET_SIZE(r) as usize != n) {
                return false;
            }
            if self.same.is_some_and(|i| operands[i as usize] != r) {
                return false;
            }
            if self.mask == 0 {
                return true;
            }
            let Some(t) = tensor_by_type(r as usize).as_ref() else { return false };
            tensor_meta(t, meta);
            let (m, key) = (&*meta, &self.key);
            m.defined
                && (self.mask & 1 == 0 || (m.has_shape && m.sizes == key.sizes))
                && (self.mask & 2 == 0 || m.dtype == key.dtype)
                && (self.mask & 4 == 0 || (m.has_device && m.device_type == key.device_type && m.device_index == key.device_index))
                && (self.mask & 8 == 0 || m.requires_grad == key.requires_grad)
        }
    }
}

impl List {
    // Needs the GIL. `inputs` and `pins` are borrowed, as many as the list was
    // made for. Ok: a new reference to the tuple of the outputs.
    pub unsafe fn run(&self, py: Python<'_>, inputs: &[*mut PyObject], pins: &[*mut PyObject]) -> Result<*mut PyObject, Stop> {
        unsafe {
            let n = self.entries.len();
            let mut run = Run { list: self, inputs, pins, slots: vec![std::ptr::null_mut(); n], made: Vec::new() };
            let mut argv = Vec::with_capacity(self.max_args + 2);
            let mut meta = TensorMeta::default();
            let mut ended = Ok(std::ptr::null_mut());
            for (k, e) in self.entries.iter().enumerate() {
                if let Err(what) = run.entry(k, e, &mut argv, &mut meta) {
                    ended = Err((k, what));
                    break;
                }
            }
            if ended.is_ok() {
                ended = run.outputs().map_err(|what| (n, what));
            }
            for s in run.slots {
                ffi::Py_XDECREF(s);
            }
            ended.map_err(|(entry, what)| {
                // A part that is not there was found before its entry ran.
                let restartable = if what == What::Part { entry <= self.first_irreversible } else { entry < self.first_irreversible };
                if what != What::Raised && !restartable {
                    let (why, line) = match what {
                        What::Part if entry == n => ("a part of a result that an output is made of is not there".to_string(), String::new()),
                        What::Part => ("a part of a result that the entry is called on is not there".to_string(), self.line(py, entry)),
                        What::Assert | What::Raised => {
                            let read = self.entries[entry].assert.as_ref().map_or(String::new(), |a| a.text());
                            (format!("the result is not what the traced call read of it ({})", read.trim_start()), self.line(py, entry))
                        }
                    };
                    PyRuntimeError::new_err(format!("torch.vm: call list entry {entry}: {why}, and what ran before cannot be taken back: {}", line.trim_start())).restore(py);
                }
                Stop { entry, what, restartable }
            })
        }
    }
}

/// Runs a list made by `_calls_make` on a tuple of inputs and a tuple of pins.
/// Returns the tuple of the outputs, or None for a miss (nothing that cannot
/// be taken back was done); an entry's exception is raised as it is. How a run
/// ended otherwise is on the list's `stopped`.
#[pyfunction]
pub fn _calls_run<'py>(list: &Bound<'py, List>, inputs: &Bound<'py, PyAny>, pins: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let (py, l) = (list.py(), list.get());
    unsafe {
        let given = |o: &Bound<'py, PyAny>, n: u32, what: &str| {
            let o = o.as_ptr();
            if ffi::PyTuple_CheckExact(o) == 0 || ffi::PyTuple_GET_SIZE(o) as u32 != n {
                return Err(PyValueError::new_err(format!("torch.vm call list: {what} must be a tuple of {n}")));
            }
            Ok((0..n as ffi::Py_ssize_t).map(|i| ffi::PyTuple_GET_ITEM(o, i)).collect::<Vec<_>>())
        };
        let (inputs, pins) = (given(inputs, l.ninputs, "inputs")?, given(pins, l.npins, "pins")?);
        l.stopped.store(0, Ordering::Relaxed);
        match l.run(py, &inputs, &pins) {
            Ok(outputs) => Ok(Bound::from_owned_ptr(py, outputs)),
            Err(stop) => {
                l.stopped.store(stop.code(), Ordering::Relaxed);
                PyErr::take(py).map_or_else(|| Ok(py.None().into_bound(py)), Err)
            }
        }
    }
}

// The rest is the tests' entry to the witness (witness.rs), here because it
// calls its argument. The tensors among a call's operands and the tests'
// names of them: position or keyword, or (that, index) for a member of an
// exact list or tuple.
fn operands<'py>(args: &Bound<'py, PyTuple>, kwargs: Option<&Bound<'py, PyDict>>) -> PyResult<(Vec<TensorIdent>, Vec<Py<PyAny>>)> {
    let py = args.py();
    let named = args.iter().enumerate().map(|(k, v)| (k.into_pyobject(py).unwrap().into_any(), v));
    let (mut ids, mut names) = (vec![], vec![]);
    let mut add = |obj: &Bound<'py, PyAny>, name: Py<PyAny>| {
        let mut id = TensorIdent::default();
        tensor_ident(obj.as_ptr() as usize, &mut id);
        ids.push(id);
        names.push(name);
    };
    for (name, value) in named.chain(kwargs.into_iter().flat_map(|d| d.iter())) {
        let members = value.cast_exact::<PyList>().map(|l| l.iter().collect::<Vec<_>>());
        match members.or_else(|_| value.cast_exact::<PyTuple>().map(|t| t.iter().collect())) {
            Ok(members) => {
                for (i, member) in members.iter().enumerate() {
                    add(member, (&name, i).into_pyobject(py)?.into_any().unbind());
                }
            }
            Err(_) => add(&value, name.unbind()),
        }
    }
    Ok((ids, names))
}

/// For tests: `func(*args, **kwargs)` under a witness, on CPython. Returns
/// (result, what was witnessed): "ops", "below", "wrote" as (op, argument,
/// operand or None, nested), "opaque", "why", "drew", "read_value",
/// "read_value_top", "entered_python", and "pure": the call is licensed with
/// nothing allowed.
#[pyfunction]
#[pyo3(signature = (func, /, *args, **kwargs))]
pub fn _witness<'py>(
    func: &Bound<'py, PyAny>,
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyDict>)> {
    let py = func.py();
    let (ids, names) = operands(args, kwargs)?;
    let _frames = crate::exec::FramesHooked::new();
    witness::begin();
    let result = func.call(args, kwargs);
    let seen = witness::end();
    let wrote = seen.wrote.iter().map(|w| (w.op, w.arg, w.operand(&ids).map(|k| names[k].clone_ref(py)), w.nested));
    let (opaque, drew, read_value, read_value_top, entered_python) = seen.flags();
    let d = PyDict::new(py);
    d.set_item("ops", &seen.ops)?;
    d.set_item("below", seen.below)?;
    d.set_item("wrote", wrote.collect::<Vec<_>>())?;
    d.set_item("opaque", opaque)?;
    d.set_item("drew", drew)?;
    d.set_item("read_value", read_value)?;
    d.set_item("read_value_top", read_value_top)?;
    d.set_item("why", &seen.why)?;
    d.set_item("entered_python", entered_python)?;
    d.set_item("pure", seen.licenses(&ids, &witness::Allowed::default()))?;
    Ok((result?, d))
}
