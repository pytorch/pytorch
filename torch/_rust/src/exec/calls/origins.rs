// Where the operands of a recorded run are from (record.rs asks; nothing
// else does). An origin is set where a value is PRODUCED, never worked out
// from the object afterwards.
//
// What it is given
//   - at the start of a recorded run: the block's params and the body
//     frame's locals (input);
//   - behind each entry: its tensor result, or the tensor items of its tuple
//     result (mark);
//   - behind an instruction that made no entry and whose C code produced a
//     tensor: that tensor, with what made it (mark, Origin::Unknown);
//   - at each BUILD_LIST / BUILD_TUPLE / BUILD_MAP / BUILD_CONST_KEY_MAP /
//     BUILD_SLICE of the body or of a frame under it: the container's
//     template, under the instruction's result value in its activation
//     (built); before any instruction: the values it names (used).
// What it guarantees
//   - a tensor is known by identity through a weak reference with a C
//     callback: the table counts no reference, an address stands for a
//     tensor only while that tensor lives, and deaths() hands out, in the
//     order they happened, the slots whose tensor died (no Python runs: the
//     callback is a C function that notes the weak reference's address);
//   - a container is a Seq / Map / Slice template only as the value its
//     BUILD instruction produced, handed straight to the instruction that
//     consumes it: any other instruction that names the value first forgets
//     it (it could store it, append to it, hand it on), and a container that
//     comes any other way has no origin;
//   - an operand of the closed immutable list is a constant BY VALUE at the
//     recorded run (by_value counts them); unless known.rs knows it for a
//     constant of the code, it is told as baked: the list is not the body;
//   - a plain tensor with no origin is LOOSE: something that was there
//     before, found by reads nobody recorded here (the executor's read hooks
//     work under today's trace only). It is taken as a pin WITHOUT a check
//     and counted apart (loose);
//   - anything else has no origin: Err says what, and the recording stops.
// What it may not assume
//   - that a tensor with no mark was there before the run: C code that makes
//     a tensor without an operator and hides it in a container is not seen.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::OnceLock;

use pyo3::ffi::{self, PyObject};
use pyo3::prelude::*;

use super::Tmpl;
use crate::exec::pin::Pin;
use crate::exec::rules::Group;

#[derive(Clone)]
pub enum Origin {
    In(u32),
    Pin(u32),
    Res(u32),
    Part(u32, u32),
    // Made by an instruction that is no entry: what it was.
    Unknown(String),
}

thread_local! {
    // The weak references whose tensor died, in order, by address.
    static DEATHS: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
    static LOST: Cell<bool> = const { Cell::new(false) };
}

unsafe extern "C" fn died(_module: *mut PyObject, weak: *mut PyObject) -> *mut PyObject {
    let noted = DEATHS.try_with(|d| d.try_borrow_mut().map(|mut d| d.push(weak as usize)).is_ok());
    if noted != Ok(true) {
        LOST.set(true);
    }
    unsafe { ffi::Py_NewRef(ffi::Py_None()) }
}

// The C function every weak reference of a table calls back, made once.
unsafe fn callback() -> *mut PyObject {
    static MADE: OnceLock<usize> = OnceLock::new();
    *MADE.get_or_init(|| unsafe {
        let def = Box::leak(Box::new(ffi::PyMethodDef {
            ml_name: c"_calls_died".as_ptr(),
            ml_meth: ffi::PyMethodDefPointer { PyCFunction: died },
            ml_flags: ffi::METH_O,
            ml_doc: std::ptr::null(),
        }));
        ffi::PyCFunction_NewEx(def, std::ptr::null_mut(), std::ptr::null_mut()) as usize
    }) as *mut PyObject
}

type Classes = Vec<Option<Py<PyAny>>>;
// A template with the class of each of its leaves, or why there is none.
pub type Made = Result<(Tmpl, Classes), String>;

#[derive(Default)]
pub struct Origins {
    // A tensor's address -> (its weak reference, where it is from).
    held: HashMap<usize, (Py<PyAny>, Origin)>,
    // A weak reference's address -> its tensor's address.
    weak: HashMap<usize, usize>,
    // (activation, value) -> the container that value's BUILD made, and the
    // operands of it taken by value that are no constants of the code.
    built: HashMap<(u32, u32), (Made, Vec<String>)>,
    // What is known of the values that are no tensors (known.rs).
    pub known: super::known::Known,
    // Operands taken by value, since record.rs last looked, that are no
    // constants of the code: the class and what made the value.
    pub baked: Vec<String>,
    pub consts: Vec<Py<PyAny>>,
    pub ninputs: u32,
    // Inputs, then pins (all loose), held weakly for the validation.
    pub inputs: Vec<Pin>,
    pub pins: Vec<Pin>,
    // Immutable operands taken by value, by class.
    pub by_value: HashMap<&'static str, u64>,
}

pub unsafe fn plain_tensor(o: *mut PyObject) -> bool {
    unsafe { !o.is_null() && Group::PlainTensor.holds(ffi::Py_TYPE(o)) }
}

// Every plain tensor a call can reach through `o`: `o` itself, or the
// members of exact lists and dicts and of tuples (a struct sequence too),
// as deep as REACH. The ONE walk of containers: what the witness is told a
// call's operands are, and what an output is asked to hold. A tensor deeper
// than REACH is not reached: a write to it is a write outside (refused), and
// an output that holds it is an output the list does not state anyway.
pub const REACH: u8 = 4;
pub unsafe fn reach(o: *mut PyObject, depth: u8, f: &mut dyn FnMut(*mut PyObject)) {
    unsafe {
        if o.is_null() {
            return;
        }
        if plain_tensor(o) {
            return f(o);
        }
        if depth >= REACH {
            return;
        }
        let tp = ffi::Py_TYPE(o);
        if super::slots::tuple_like(o) {
            (0..ffi::PyTuple_GET_SIZE(o)).for_each(|i| reach(ffi::PyTuple_GET_ITEM(o, i), depth + 1, f));
        } else if tp == &raw mut ffi::PyList_Type {
            (0..ffi::PyList_GET_SIZE(o)).for_each(|i| reach(ffi::PyList_GET_ITEM(o, i), depth + 1, f));
        } else if tp == &raw mut ffi::PyDict_Type {
            let (mut at, mut k, mut v) = (0, std::ptr::null_mut(), std::ptr::null_mut());
            while ffi::PyDict_Next(o, &mut at, &mut k, &mut v) != 0 {
                reach(v, depth + 1, f);
            }
        }
    }
}

unsafe fn class_of(o: *mut PyObject) -> Option<Py<PyAny>> {
    unsafe { Some(Py::from_borrowed_ptr(Python::assume_attached(), ffi::Py_TYPE(o) as *mut PyObject)) }
}

pub unsafe fn class_text(o: *mut PyObject) -> String {
    if o.is_null() { "NULL".into() } else { unsafe { crate::exec::hist::class_name(ffi::Py_TYPE(o)) } }
}

impl Origins {
    pub fn begin() -> Origins {
        DEATHS.with_borrow_mut(|d| d.clear());
        LOST.set(false);
        Origins::default()
    }

    // Whether a death could not be noted: the releases are not known.
    pub fn lost() -> bool {
        LOST.get()
    }

    // A plain tensor that is live when the run starts.
    pub unsafe fn input(&mut self, o: *mut PyObject) {
        unsafe {
            if plain_tensor(o) && self.get(o).is_none() && self.mark(o, Origin::In(self.ninputs)) {
                self.inputs.push(Pin::weak(o));
                self.ninputs += 1;
            }
        }
    }

    // False: no weak reference to `o` could be made.
    pub unsafe fn mark(&mut self, o: *mut PyObject, origin: Origin) -> bool {
        unsafe {
            let weak = ffi::PyWeakref_NewRef(o, callback());
            if weak.is_null() {
                ffi::PyErr_Clear();
                return false;
            }
            self.weak.insert(weak as usize, o as usize);
            self.held.insert(o as usize, (Py::from_owned_ptr(Python::assume_attached(), weak), origin));
            true
        }
    }

    // The origin of the tensor `o`, if it has one: only while the object at
    // that address is the one that was marked.
    pub unsafe fn get(&self, o: *mut PyObject) -> Option<&Origin> {
        let (weak, origin) = self.held.get(&(o as usize))?;
        let referent = unsafe { (*(weak.as_ptr() as *mut ffi::PyWeakReference)).wr_object };
        (referent == o).then_some(origin)
    }

    // The slots whose result died since the last call, in order.
    pub fn deaths(&mut self) -> Vec<u32> {
        let dead = DEATHS.with_borrow_mut(std::mem::take);
        let gone = dead.into_iter().filter_map(|w| self.weak.remove(&w)).filter_map(|o| self.held.remove(&o));
        // Out of the table first: dropping a weak reference runs nothing.
        gone.filter_map(|(_, origin)| if let Origin::Res(j) = origin { Some(j) } else { None }).collect()
    }

    // The slots whose tensor result is alive.
    pub fn alive(&self) -> Vec<(u32, *mut PyObject)> {
        let slot = |(o, (_, origin)): (&usize, &(Py<PyAny>, Origin))| if let Origin::Res(j) = origin { Some((*j, *o as *mut PyObject)) } else { None };
        self.held.iter().filter_map(slot).collect()
    }

    // Whether `o` is, or holds (reach), a tensor that an entry made.
    pub unsafe fn holds(&self, o: *mut PyObject) -> bool {
        let mut found = false;
        unsafe { reach(o, 0, &mut |t| found |= matches!(self.get(t), Some(Origin::Res(_) | Origin::Part(..)))) };
        found
    }

    pub fn built(&mut self, at: (u32, u32), made: Made) {
        self.built.insert(at, (made, std::mem::take(&mut self.baked)));
    }

    // An instruction that is no entry and no BUILD names these values.
    pub fn used(&mut self, depth: u32, values: impl Iterator<Item = u32>) {
        if !self.built.is_empty() {
            values.for_each(|v| drop(self.built.remove(&(depth, v))));
        }
    }

    // An activation is over: what its values built is gone.
    pub fn left(&mut self, depth: u32) {
        self.built.retain(|(d, _), _| *d != depth);
        self.known.left(depth);
    }

    fn constant(&mut self, o: *mut PyObject) -> Tmpl {
        let at = self.consts.iter().position(|c| c.as_ptr() == o).unwrap_or_else(|| {
            self.consts.push(unsafe { Py::from_borrowed_ptr(Python::assume_attached(), o) });
            self.consts.len() - 1
        });
        Tmpl::Const(at as u32)
    }

    // The template of the operand `o`, which is value `v` of activation
    // `depth`; `loose`: counts the tensors taken with no origin.
    pub unsafe fn template(&mut self, o: *mut PyObject, at: (u32, u32), loose: &mut u32) -> Made {
        unsafe {
            if let Some((made, baked)) = self.built.remove(&at) {
                self.baked.extend(baked);
                return made;
            }
            if o.is_null() {
                return Err("an operand that is not there (NULL)".into());
            }
            if plain_tensor(o) {
                let tmpl = match self.get(o).cloned() {
                    Some(Origin::In(k)) => Tmpl::In(k),
                    Some(Origin::Pin(i)) => Tmpl::Pin(i),
                    Some(Origin::Res(j)) => Tmpl::Res(j),
                    Some(Origin::Part(j, i)) => Tmpl::Part(j, Box::new([i])),
                    Some(Origin::Unknown(by)) => return Err(format!("an operand of unknown origin: {} made by {by}", class_text(o))),
                    None => {
                        let i = self.pins.len() as u32;
                        if !self.mark(o, Origin::Pin(i)) {
                            return Err("an operand of unknown origin: a tensor no weak reference can be made to".into());
                        }
                        self.pins.push(Pin::weak(o));
                        *loose += 1;
                        Tmpl::Pin(i)
                    }
                };
                return Ok((tmpl, vec![class_of(o)]));
            }
            if super::immutable(o, 0) {
                if !self.known.lit(at) {
                    self.baked.push(format!("{} from {}", class_text(o), self.known.maker(at)));
                }
                *self.by_value.entry(std::ffi::CStr::from_ptr((*ffi::Py_TYPE(o)).tp_name).to_str().unwrap_or("?")).or_default() += 1;
                return Ok((self.constant(o), vec![class_of(o)]));
            }
            let tp = ffi::Py_TYPE(o);
            if [Group::ExactList, Group::ExactTuple, Group::ExactDict].iter().any(|g| g.holds(tp)) || tp == &raw mut ffi::PySlice_Type {
                return Err(format!("a container the body did not build for this instruction: {}", class_text(o)));
            }
            Err(format!("an operand of unknown origin: {}", class_text(o)))
        }
    }

    // The template of what a BUILD instruction makes of `parts` (object,
    // value): a list, a tuple, a dict of these plain keys, a slice.
    pub unsafe fn build(&mut self, kind: Built, parts: &[(*mut PyObject, u32)], depth: u32, loose: &mut u32) -> Made {
        unsafe {
            let (mut tmpls, mut classes) = (Vec::new(), Vec::new());
            for (o, v) in parts {
                let (t, c) = self.template(*o, (depth, *v), loose)?;
                tmpls.push(t);
                classes.extend(c);
            }
            Ok((
                match kind {
                    Built::List => Tmpl::Seq(true, tmpls.into()),
                    Built::Tuple => Tmpl::Seq(false, tmpls.into()),
                    Built::Map(keys) => {
                        if !keys.iter().all(|k| super::plain(*k)) {
                            return Err("a dict built with a key that is no exact str, int, float, bytes, bool or None".into());
                        }
                        Tmpl::Map(keys.into_iter().map(|k| Py::from_borrowed_ptr(Python::assume_attached(), k)).collect(), tmpls.into())
                    }
                    Built::Slice => {
                        if tmpls.len() == 2 {
                            classes.push(class_of(ffi::Py_None()));
                            tmpls.push(self.constant(ffi::Py_None()));
                        }
                        let [a, b, c] = <[Tmpl; 3]>::try_from(tmpls).map_err(|_| "a slice of other than two or three bounds".to_string())?;
                        Tmpl::Slice(Box::new([a, b, c]))
                    }
                },
                classes,
            ))
        }
    }
}

pub enum Built {
    List,
    Tuple,
    Map(Vec<*mut PyObject>),
    Slice,
}
