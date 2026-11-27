// Runs a call list. Per entry: every operand is built from its template and
// its class compared with the pin, the callee is called as the executor
// calls it for the instruction (the same helpers), the arguments are
// released as CALL releases them (in their order: the slots the entry moves,
// what was built for the call), the entry's assert is evaluated and the
// result goes to the entry's slot or is dropped. A drop entry releases its
// slots. No frame, no gate, no dispatch hook.
//
// A run ends with the tuple of its outputs (the slots that are left, the
// outputs', are released behind it), or with a Stop:
//   Raised   An entry raised: the callee's own exception is set (a callee
//            that returned nothing and set none: a SystemError). What entries
//            before it did stays done. `restartable` says whether every entry
//            up to and with it can be taken back: the caller's license to
//            make it a miss instead.
//   Assert   A result does not have what the trace read of it.
//   Part     A part of a result is not there: no exact tuple or struct
//            sequence, or too short. The entry that wanted it has not run.
//   Class    An operand is not of the class it is pinned to, or a class a
//            slot callee was resolved on has changed. The entry has not run.
//   Refused  Nothing ran: this thread is already running this list
//            (RuntimeError), or lists are nested deeper than MAX_DEPTH
//            (RecursionError).
// For Assert, Part and Class: restartable -> no exception is set (a miss);
// else a RuntimeError naming the entry is set, never a value.
// A stop releases no slot: `live` hands every slot that holds a result to the
// caller, who puts them where the program's own run has them (the frame a
// traceback holds keeps its locals until the handler is done).

use std::cell::RefCell;
use std::os::raw::c_int;

use pyo3::exceptions::{PyRecursionError, PyRuntimeError, PySystemError, PyValueError};
use pyo3::ffi::{self, PyObject};
use pyo3::prelude::*;
use pyo3::types::{PyCFunction, PyDict, PyList, PyModule, PyTuple};

use super::super::ffi::{tr_binary_op, tr_call, tr_generic, tr_type_version_is};
use super::slots::tuple_like;
use super::{witness, Assert, Callee, Entry, List, Release, Tmpl};
use crate::bindings::ffi::op_hook::{tensor_ident, TensorIdent};
use crate::cfg::opcodes;

// How many lists may run inside one another on a thread.
const MAX_DEPTH: usize = 32;

thread_local! {
    // The lists this thread is running, outermost first, by address.
    static RUNNING: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
}

struct Entered;

impl Drop for Entered {
    fn drop(&mut self) {
        RUNNING.with_borrow_mut(|r| r.pop());
    }
}

#[derive(Clone, Copy, PartialEq)]
pub enum What {
    Raised,
    Assert,
    Part,
    Class,
    Refused,
}

impl What {
    pub fn text(self) -> &'static str {
        match self {
            What::Raised => "raised",
            What::Assert => "assert",
            What::Part => "part",
            What::Class => "class",
            What::Refused => "refused",
        }
    }
}

pub struct Stop {
    // The entry; the count of entries for a part an output wanted.
    pub entry: usize,
    pub what: What,
    pub restartable: bool,
    // (slot, its result) of every slot that held one, in slot order.
    pub live: Vec<(u32, Py<PyAny>)>,
}

struct Run<'a> {
    list: &'a List,
    inputs: &'a [*mut PyObject],
    pins: &'a [*mut PyObject],
    slots: Vec<*mut PyObject>,
}

type Pins<'a> = std::slice::Iter<'a, Option<Py<PyAny>>>;

impl Run<'_> {
    // The operand for the call: borrowed, or (Tmpl::made) a new reference.
    // Null for Tmpl::Null. `pins`: the classes of the leaves from here on; an
    // output has none.
    unsafe fn operand(&mut self, t: &Tmpl, mut pins: Option<&mut Pins<'_>>) -> Result<*mut PyObject, What> {
        unsafe {
            // A part of what is built: the container takes a reference.
            let owned = |t: &Tmpl, x: *mut PyObject| {
                if !t.made() {
                    ffi::Py_XINCREF(x);
                }
                x
            };
            let leaf = match t {
                Tmpl::In(k) => self.inputs[*k as usize],
                Tmpl::Pin(i) => self.pins[*i as usize],
                Tmpl::Res(j) | Tmpl::Move(j) => self.slots[*j as usize],
                Tmpl::Const(i) => self.list.consts[*i as usize].as_ptr(),
                Tmpl::Null => std::ptr::null_mut(),
                Tmpl::Part(j, path) => {
                    let mut x = self.slots[*j as usize];
                    for i in path.iter() {
                        if !tuple_like(x) || *i as ffi::Py_ssize_t >= ffi::PyTuple_GET_SIZE(x) {
                            return Err(What::Part);
                        }
                        x = ffi::PyTuple_GET_ITEM(x, *i as ffi::Py_ssize_t);
                    }
                    x
                }
                Tmpl::Seq(list, parts) => {
                    let n = parts.len() as ffi::Py_ssize_t;
                    let seq = if *list { ffi::PyList_New(n) } else { ffi::PyTuple_New(n) };
                    if seq.is_null() {
                        return Err(What::Raised);
                    }
                    for (k, p) in parts.iter().enumerate() {
                        let item = match self.operand(p, pins.as_deref_mut()) {
                            Ok(x) => owned(p, x),
                            Err(what) => {
                                ffi::Py_DECREF(seq);
                                return Err(what);
                            }
                        };
                        if *list {
                            ffi::PyList_SET_ITEM(seq, k as ffi::Py_ssize_t, item);
                        } else {
                            ffi::PyTuple_SET_ITEM(seq, k as ffi::Py_ssize_t, item);
                        }
                    }
                    return Ok(seq);
                }
                Tmpl::Map(keys, parts) => {
                    let map = ffi::PyDict_New();
                    if map.is_null() {
                        return Err(What::Raised);
                    }
                    for (key, p) in keys.iter().zip(parts.iter()) {
                        let stored = self.operand(p, pins.as_deref_mut()).and_then(|value| {
                            let rc = ffi::PyDict_SetItem(map, key.as_ptr(), value);
                            if p.made() {
                                ffi::Py_DECREF(value);
                            }
                            if rc < 0 { Err(What::Raised) } else { Ok(()) }
                        });
                        if let Err(what) = stored {
                            ffi::Py_DECREF(map);
                            return Err(what);
                        }
                    }
                    return Ok(map);
                }
                Tmpl::Slice(p) => {
                    let mut bounds = [std::ptr::null_mut(); 3];
                    let mut failed = None;
                    for (bound, t) in bounds.iter_mut().zip(p.iter()) {
                        match self.operand(t, pins.as_deref_mut()) {
                            Ok(x) => *bound = x,
                            Err(what) => {
                                failed = Some(what);
                                break;
                            }
                        }
                    }
                    let slice = if failed.is_none() { ffi::PySlice_New(bounds[0], bounds[1], bounds[2]) } else { std::ptr::null_mut() };
                    for (bound, t) in bounds.iter().zip(p.iter()) {
                        if t.made() {
                            ffi::Py_XDECREF(*bound);
                        }
                    }
                    return if slice.is_null() { Err(failed.unwrap_or(What::Raised)) } else { Ok(slice) };
                }
            };
            // An operand of an entry has a pin: the class, or none for a null.
            if let Some(pins) = pins {
                match pins.next() {
                    Some(None) if leaf.is_null() => {}
                    Some(Some(class)) if !leaf.is_null() && class.as_ptr() == ffi::Py_TYPE(leaf) as *mut PyObject => {}
                    Some(_) | None => return Err(What::Class),
                }
            }
            Ok(leaf)
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
                Callee::State(_) | Callee::Drop(_) => unreachable!("a state setter is refused when the list is made; a drop is no call"),
            }
        }
    }

    unsafe fn release(&mut self, j: u32) {
        unsafe { ffi::Py_XDECREF(std::mem::replace(&mut self.slots[j as usize], std::ptr::null_mut())) };
    }

    unsafe fn entry(&mut self, k: usize, e: &Entry, argv: &mut Vec<*mut PyObject>) -> Result<(), What> {
        unsafe {
            if let Callee::Drop(slots) = &e.callee {
                slots.iter().for_each(|j| self.release(*j));
                return Ok(());
            }
            argv.clear();
            let first = if let Callee::Vector(f, _) = &e.callee {
                argv.extend([std::ptr::null_mut(), f.as_ptr()]);
                2
            } else {
                0
            };
            let mut pins = e.classes.iter();
            let mut built = Ok(());
            for t in e.args.iter() {
                match self.operand(t, Some(&mut pins)) {
                    Ok(x) => argv.push(x),
                    Err(what) => {
                        built = Err(what);
                        break;
                    }
                }
            }
            if built.is_ok() && e.decided.iter().any(|(class, version)| tr_type_version_is(class.as_ptr() as *mut ffi::PyTypeObject, *version) == 0) {
                built = Err(What::Class);
            }
            if let Err(what) = built {
                // The entry has not run: what was built for it goes, every
                // slot stays.
                for (t, x) in e.args.iter().zip(&argv[first..]) {
                    if t.made() {
                        ffi::Py_DECREF(*x);
                    }
                }
                return Err(what);
            }
            let r = Self::call(e, argv);
            for release in e.release.iter() {
                match release {
                    Release::Slot(j) => self.release(*j),
                    Release::Arg(at) => ffi::Py_DECREF(argv[first + *at as usize]),
                }
            }
            let r = r?;
            let holds = e.assert.as_ref().is_none_or(|a| a.holds(r, &argv[first..]));
            if e.out.is_some() {
                self.slots[k] = r;
            } else {
                ffi::Py_XDECREF(r);
            }
            if holds { Ok(()) } else { Err(What::Assert) }
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
                match self.operand(t, None) {
                    Ok(x) => {
                        if !t.made() {
                            ffi::Py_INCREF(x);
                        }
                        ffi::PyTuple_SET_ITEM(tuple, k as ffi::Py_ssize_t, x);
                    }
                    Err(what) => {
                        ffi::Py_DECREF(tuple);
                        return Err(what);
                    }
                }
            }
            Ok(tuple)
        }
    }
}

impl Assert {
    // `operands` are compared by address only: a moved one may be gone.
    unsafe fn holds(&self, r: *mut PyObject, operands: &[*mut PyObject]) -> bool {
        unsafe {
            self.count.is_none_or(|n| tuple_like(r) && ffi::PyTuple_GET_SIZE(r) as usize == n)
                && self.same.is_none_or(|i| operands[i as usize] == r)
                && self.class.as_ref().is_none_or(|class| class.as_ptr() == ffi::Py_TYPE(r) as *mut PyObject)
                && self.key.as_ref().is_none_or(|key| key.holds(r))
        }
    }
}

impl List {
    // Needs the GIL. `inputs` and `pins` are borrowed, as many as the list was
    // made for. Ok: a new reference to the tuple of the outputs.
    pub unsafe fn run(&self, py: Python<'_>, inputs: &[*mut PyObject], pins: &[*mut PyObject]) -> Result<*mut PyObject, Stop> {
        let me = self as *const List as usize;
        let refused = RUNNING.with_borrow_mut(|running| {
            if running.contains(&me) {
                return Some(PyRuntimeError::new_err("torch.vm: a call list was entered while this thread runs it"));
            }
            if running.len() >= MAX_DEPTH {
                return Some(PyRecursionError::new_err(format!("torch.vm: call lists nested deeper than {MAX_DEPTH}")));
            }
            running.push(me);
            None
        });
        if let Some(error) = refused {
            error.restore(py);
            return Err(Stop { entry: 0, what: What::Refused, restartable: true, live: Vec::new() });
        }
        let _entered = Entered;
        unsafe {
            let n = self.entries.len();
            let mut run = Run { list: self, inputs, pins, slots: vec![std::ptr::null_mut(); n] };
            let mut argv = Vec::with_capacity(self.max_args + 2);
            let mut ended = Ok(std::ptr::null_mut());
            for (k, e) in self.entries.iter().enumerate() {
                if let Err(what) = run.entry(k, e, &mut argv) {
                    ended = Err((k, what));
                    break;
                }
            }
            if ended.is_ok() {
                ended = run.outputs().map_err(|what| (n, what));
            }
            let slots = std::mem::take(&mut run.slots);
            let (entry, what) = match ended {
                Ok(outputs) => {
                    slots.into_iter().for_each(|s| ffi::Py_XDECREF(s));
                    return Ok(outputs);
                }
                Err(stop) => stop,
            };
            // What was found before its entry ran leaves that entry undone.
            let restartable = match what {
                What::Raised | What::Assert => entry < self.first_irreversible,
                What::Part | What::Class | What::Refused => entry <= self.first_irreversible,
            };
            if what == What::Raised && ffi::PyErr_Occurred().is_null() {
                PySystemError::new_err(format!("torch.vm: call list entry {entry}: the callee returned no result and set no exception: {}", self.line(py, entry).trim_start())).restore(py);
            }
            if what != What::Raised && !restartable {
                let (why, line) = match what {
                    What::Part if entry == n => ("a part of a result that an output is made of is not there".to_string(), String::new()),
                    What::Part => ("a part of a result that the entry is called on is not there".to_string(), self.line(py, entry)),
                    What::Class => ("an operand is not of the class it had at the trace, or that class has changed".to_string(), self.line(py, entry)),
                    What::Assert | What::Raised | What::Refused => {
                        let read = self.entries[entry].assert.as_ref().map_or(String::new(), |a| a.text());
                        (format!("the result is not what the traced call read of it ({})", read.trim_start()), self.line(py, entry))
                    }
                };
                PyRuntimeError::new_err(format!("torch.vm: call list entry {entry}: {why}, and what ran before cannot be taken back: {}", line.trim_start())).restore(py);
            }
            let live = slots.into_iter().enumerate().filter(|(_, s)| !s.is_null()).map(|(j, s)| (j as u32, Py::from_owned_ptr(py, s))).collect();
            Err(Stop { entry, what, restartable, live })
        }
    }
}

/// Runs a list made by `_calls_make` on a tuple of inputs and a tuple of pins.
/// Returns the tuple of the outputs, or None for a miss (nothing that cannot
/// be taken back was done); an entry's exception is raised as it is. How a run
/// ended otherwise is on the list's `stopped`, with the slots that were live:
/// the list keeps them until the next run or `release()`.
#[pyfunction]
pub fn _calls_run<'py>(list: &Bound<'py, List>, inputs: &Bound<'py, PyAny>, pins: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let (py, l) = (list.py(), list.get());
    // Out of the lock first: releasing a slot can run code that reads `stopped`.
    let before = l.stopped.lock().unwrap().take();
    drop(before);
    unsafe {
        let given = |o: &Bound<'py, PyAny>, n: u32, what: &str| {
            let o = o.as_ptr();
            if ffi::PyTuple_CheckExact(o) == 0 || ffi::PyTuple_GET_SIZE(o) as u32 != n {
                return Err(PyValueError::new_err(format!("torch.vm call list: {what} must be a tuple of {n}")));
            }
            Ok((0..n as ffi::Py_ssize_t).map(|i| ffi::PyTuple_GET_ITEM(o, i)).collect::<Vec<_>>())
        };
        let (inputs, pins) = (given(inputs, l.ninputs, "inputs")?, given(pins, l.npins, "pins")?);
        match l.run(py, &inputs, &pins) {
            Ok(outputs) => Ok(Bound::from_owned_ptr(py, outputs)),
            Err(stop) => {
                let raised = stop.what == What::Raised || stop.what == What::Refused;
                *l.stopped.lock().unwrap() = Some(stop);
                match PyErr::take(py) {
                    Some(error) => Err(error),
                    None if raised => Err(PySystemError::new_err("torch.vm call list: a stop that raises set no exception")),
                    None => Ok(py.None().into_bound(py)),
                }
            }
        }
    }
}

// The rest is the tests' entry to the witness (witness.rs), here because it
// calls its argument. The tensors among a call's operands and the tests'
// names of them: position or keyword, or (that, index) for a member of an
// exact list or tuple. The object a builtin is bound to (`a.add_`) is the
// first operand, in front of the positional ones; a function of a module is
// bound to the module and a static one to nothing, which are no operands.
// What any other callable holds (a method-wrapper's object, a closure) is no
// operand: a write to it is a write outside.
fn operands<'py>(
    func: &Bound<'py, PyAny>,
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<(Vec<TensorIdent>, Vec<Py<PyAny>>)> {
    let py = args.py();
    let bound = func.cast_exact::<PyCFunction>().ok().and_then(|f| {
        // SAFETY: an exact builtin; its self is borrowed from it, or NULL.
        unsafe { Bound::from_borrowed_ptr_or_opt(py, ffi::PyCFunction_GetSelf(f.as_ptr())) }
    });
    let positional = bound.into_iter().filter(|s| s.cast::<PyModule>().is_err()).chain(args.iter());
    let named = positional.enumerate().map(|(k, v)| (k.into_pyobject(py).unwrap().into_any(), v));
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
/// operand or None, nested), "wrote_outside", "opaque", "why", "drew",
/// "read_value", "read_value_top", "data_sized", "entered_python", and
/// "saw_nothing": the call is licensed with nothing allowed. That is no
/// statement about the call (see the head of witness.rs).
#[pyfunction]
#[pyo3(signature = (func, /, *args, **kwargs))]
pub fn _witness<'py>(
    func: &Bound<'py, PyAny>,
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyDict>)> {
    let py = func.py();
    let (ids, names) = operands(func, args, kwargs)?;
    let _frames = crate::exec::FramesHooked::new();
    witness::begin();
    let result = func.call(args, kwargs);
    let seen = witness::end();
    let wrote = seen.wrote.iter().map(|w| (w.op, w.arg, w.operand(&ids).map(|k| names[k].clone_ref(py)), w.nested));
    let flags = seen.flags();
    let d = PyDict::new(py);
    d.set_item("ops", &seen.ops)?;
    d.set_item("below", seen.below)?;
    d.set_item("wrote", wrote.collect::<Vec<_>>())?;
    d.set_item("wrote_outside", seen.wrote_outside(&ids))?;
    d.set_item("opaque", flags.opaque)?;
    d.set_item("drew", flags.drew)?;
    d.set_item("read_value", flags.read_value)?;
    d.set_item("read_value_top", flags.read_value_top)?;
    d.set_item("data_sized", flags.data_sized)?;
    d.set_item("why", &seen.why)?;
    d.set_item("entered_python", flags.entered_python)?;
    d.set_item("saw_nothing", seen.licenses(&ids, &witness::Allowed::default()))?;
    Ok((result?, d))
}
