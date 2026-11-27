// What a block hands on, as a recorded run's list states it (finish.rs
// states, again.rs compares; nothing else asks).
//
// What it is given
//   - at the end of a recorded run: an output of the block with what the
//     recorder knows of its value where the body left (origins.rs Leaving);
//   - at a later arrival: what that arrival began on at the positions the
//     statements name (Kept::of), and, when its block has run, the block's
//     outputs (holds).
// What it guarantees
//   - an output is stated only POSITIVELY, by where its value is from:
//       Null     the block hands on nothing there (an unbound slot);
//       Tensor   a tensor by its origin (finish.rs): the list's output k;
//       Const    a value of an immutable class that known.rs knows AS AN
//                OBJECT (a constant of the code, an object a recorded read
//                expects by identity, None, True or False): that value. A
//                value known by class and value only (an input the list is
//                specialized on, a fact of a key) is no statement: the
//                block hands on an object, and which is not known;
//       Input    the value IS an input of the run by the flow of values (a
//                param of the block, a local of the body's frame loaded
//                before the run changed the slot) and the object the run
//                began on there: whatever the arrival has at that position;
//       Led      an object the recorded reads lead to from what was there
//                before the run, held weakly: that very object;
//       Seq, Map an exact list, tuple or dict whose BUILD the run recorded
//                and which no instruction has named since, with nothing in
//                it taken by value that is not known: a new one of these.
//     Nothing is stated by identity with an input or by the value an object
//     happens to have: True from a test is no param that is True;
//   - holds() compares a block's output with its statement and runs no code
//     of the program: Err(what differs), Ok(false) where it cannot say (the
//     block left nothing where a value is stated; an input of a class that
//     takes no weak reference and is no immutable is known by its address
//     only: another address differs, the same address says nothing),
//     Ok(true) only for what it compared;
//   - nothing kept here keeps an object of the program alive that has a
//     finalizer or a weak reference: a Const and a kept input value are of
//     the immutable classes, a Led and any other kept input is weak or an
//     address.
// What it may not assume
//   - that an object the reads lead to is the same on a later arrival: the
//     reads are the check, and until they run the list rests on them
//     (known.rs PENDING).

use pyo3::ffi::{self, PyObject};
use pyo3::prelude::*;

use super::origins::{class_text, plain_tensor};
use super::Tmpl;
use crate::exec::exact;
use crate::exec::pin::Pin;

pub enum Out {
    Null,
    Tensor(u32),
    Const(Py<PyAny>),
    Input(u32),
    Led(Pin),
    Seq(bool, Box<[Out]>),
    Map(Box<[Py<PyAny>]>, Box<[Out]>),
}

// What a later arrival began on at one position.
pub enum Kept {
    Weak(Pin),
    Value(Py<PyAny>),
    Address(usize),
}

impl Kept {
    pub unsafe fn of(o: *mut PyObject) -> Kept {
        unsafe {
            if o.is_null() {
                return Kept::Address(0);
            }
            if super::immutable(o, 0) {
                return Kept::Value(Py::from_borrowed_ptr(Python::assume_attached(), o));
            }
            // (The caller holds `o`: a pin that counts is dropped with
            // nothing released.)
            let pin = Pin::weak(o);
            if pin.counts().is_none() { Kept::Weak(pin) } else { Kept::Address(o as usize) }
        }
    }

    // Whether `now` is what was kept; None: it cannot be said.
    unsafe fn is(&self, now: *mut PyObject) -> Option<bool> {
        match self {
            Kept::Weak(pin) => Some(pin.is(now)),
            Kept::Value(v) => Some(v.as_ptr() == now),
            Kept::Address(a) => if *a == now as usize { None } else { Some(false) },
        }
    }
}

impl Out {
    // The statement of the container `o` that a BUILD of the run made as
    // `tmpl` (constants: `consts`). Each tensor in it becomes an output of
    // the list: its template goes to `outputs`, the tensor to `made`. None:
    // `o` is not what the template says (another class, another length), or
    // the template has a part that is no statement of an output (a slice).
    pub unsafe fn built(tmpl: &Tmpl, o: *mut PyObject, consts: &[Py<PyAny>], outputs: &mut Vec<Tmpl>, made: &mut Vec<Py<PyAny>>) -> Option<Out> {
        unsafe {
            let py = Python::assume_attached();
            if o.is_null() {
                return None;
            }
            Some(match tmpl {
                Tmpl::In(_) | Tmpl::Pin(_) | Tmpl::Res(_) | Tmpl::Part(..) if plain_tensor(o) => {
                    outputs.push(match tmpl {
                        Tmpl::In(k) => Tmpl::In(*k),
                        Tmpl::Pin(k) => Tmpl::Pin(*k),
                        Tmpl::Res(k) => Tmpl::Res(*k),
                        Tmpl::Part(k, path) => Tmpl::Part(*k, path.clone()),
                        _ => return None,
                    });
                    made.push(Py::from_borrowed_ptr(py, o));
                    Out::Tensor(outputs.len() as u32 - 1)
                }
                Tmpl::Const(i) if consts.get(*i as usize).is_some_and(|c| c.as_ptr() == o) => Out::Const(consts[*i as usize].clone_ref(py)),
                Tmpl::Seq(list, items) => {
                    let exact = if *list { ffi::PyList_CheckExact(o) } else { ffi::PyTuple_CheckExact(o) } != 0;
                    if !exact || ffi::Py_SIZE(o) != items.len() as ffi::Py_ssize_t {
                        return None;
                    }
                    let item = |i: usize| if *list { ffi::PyList_GET_ITEM(o, i as ffi::Py_ssize_t) } else { ffi::PyTuple_GET_ITEM(o, i as ffi::Py_ssize_t) };
                    Out::Seq(*list, items.iter().enumerate().map(|(i, t)| Out::built(t, item(i), consts, outputs, made)).collect::<Option<Vec<_>>>()?.into())
                }
                Tmpl::Map(keys, items) => {
                    let found = pairs(o).filter(|found| found.len() == keys.len() && found.iter().zip(keys.iter()).all(|((k, _), key)| exact::same(*k, key.as_ptr())))?;
                    let items = items.iter().zip(&found).map(|(t, (_, v))| Out::built(t, *v, consts, outputs, made)).collect::<Option<Vec<_>>>()?;
                    Out::Map(keys.iter().map(|k| k.clone_ref(py)).collect(), items.into())
                }
                _ => return None,
            })
        }
    }

    // The positions of the inputs this statement names.
    pub fn inputs(&self, into: &mut Vec<u32>) {
        match self {
            Out::Input(p) => into.push(*p),
            Out::Seq(_, items) | Out::Map(_, items) => items.iter().for_each(|x| x.inputs(into)),
            _ => {}
        }
    }

    // Whether `o` (borrowed, NULL: nothing), an output of the block on an
    // arrival that began on `kept` (by position), is what is stated. The
    // tensors it holds where the list has outputs go to `tensors`, by the
    // list's output: the caller compares them with what the list made.
    pub unsafe fn holds(&self, o: *mut PyObject, kept: &[(u32, Kept)], tensors: &mut [*mut PyObject]) -> Result<bool, String> {
        unsafe {
            let differs = |stated: &str| Err(format!("a {} where the list states {stated}", class_text(o)));
            match self {
                Out::Null => return if o.is_null() { Ok(true) } else { differs("nothing") },
                // (A version of the block that the optimizer made may leave
                // a value nobody reads unset: nothing to compare with.)
                _ if o.is_null() => return Ok(false),
                _ => {}
            }
            match self {
                Out::Null => Ok(true),
                Out::Tensor(k) => match tensors.get_mut(*k as usize) {
                    Some(slot) if plain_tensor(o) => {
                        *slot = o;
                        Ok(true)
                    }
                    _ => differs("a tensor"),
                },
                Out::Const(v) => if exact::same(o, v.as_ptr()) { Ok(true) } else { differs("a value it knows") },
                Out::Input(p) => match kept.iter().find(|(q, _)| q == p).and_then(|(_, kept)| kept.is(o)) {
                    Some(true) => Ok(true),
                    Some(false) => differs(&format!("what the arrival began on at position {p}")),
                    None => Ok(false),
                },
                Out::Led(pin) if pin.dead() => Ok(false),
                Out::Led(pin) => if pin.is(o) { Ok(true) } else { differs("the object its reads lead to") },
                Out::Seq(list, items) => {
                    let exact = if *list { ffi::PyList_CheckExact(o) } else { ffi::PyTuple_CheckExact(o) } != 0;
                    if !exact || ffi::Py_SIZE(o) != items.len() as ffi::Py_ssize_t {
                        return differs(&format!("a new {} of {} item(s)", if *list { "list" } else { "tuple" }, items.len()));
                    }
                    let item = |i: usize| if *list { ffi::PyList_GET_ITEM(o, i as ffi::Py_ssize_t) } else { ffi::PyTuple_GET_ITEM(o, i as ffi::Py_ssize_t) };
                    items.iter().enumerate().try_fold(true, |all, (i, x)| Ok(x.holds(item(i), kept, tensors)? && all))
                }
                Out::Map(keys, items) => {
                    let Some(found) = pairs(o).filter(|found| found.len() == keys.len() && found.iter().zip(keys.iter()).all(|((k, _), key)| exact::same(*k, key.as_ptr()))) else {
                        return differs(&format!("a new dict of {} key(s)", keys.len()));
                    };
                    items.iter().zip(&found).try_fold(true, |all, (x, (_, v))| Ok(x.holds(*v, kept, tensors)? && all))
                }
            }
        }
    }
}

// The items of the exact dict `o`, in its order; None for anything else.
unsafe fn pairs(o: *mut PyObject) -> Option<Vec<(*mut PyObject, *mut PyObject)>> {
    unsafe {
        if ffi::PyDict_CheckExact(o) == 0 {
            return None;
        }
        let (mut at, mut k, mut v, mut found) = (0, std::ptr::null_mut(), std::ptr::null_mut(), Vec::new());
        while ffi::PyDict_Next(o, &mut at, &mut k, &mut v) != 0 {
            found.push((k, v));
        }
        Some(found)
    }
}
