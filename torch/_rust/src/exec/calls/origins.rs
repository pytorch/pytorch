// Where the operands of a recorded run are from (record.rs asks; nothing
// else does). An origin is set where a value is PRODUCED, never worked out
// from the object afterwards.
//
// What it is given
//   - at the start of a recorded run: the block's params and the body
//     frame's locals (begin, which walks them: walk);
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
//   - an INPUT is a plain tensor found at the start of the run by walk():
//     a param or a local, or an item of an exact tuple or list, or a value
//     of an exact dict whose keys are all exact strs, among them (as deep
//     as REACH, no longer than WIDE), each tensor once, with the path it
//     was found by; for a step through a dict the key at that position is
//     kept too (keys). A later arrival is asked for its inputs by the same
//     walk, so an input is that arrival's tensor at the same path, under
//     the same keys. A container of any other class (a subclass of dict, a
//     dict with a key that is no exact str, a set) is not walked: a tensor
//     in it is no input;
//   - an immutable operand that is an INPUT (known.rs input(): a param of
//     the block, a local of the body's frame the run had not changed) is
//     taken by value and the list is SPECIALIZED on it: the position and
//     the value are kept (exact), a later arrival is asked for exactly that
//     value there (again.rs, exec/exact.rs), and the value is known from
//     then on. Only if the object at that position when the run began is
//     the operand;
//   - a plain tensor with no origin is a PIN only if the reads recorded so
//     far (reads.rs, which the recorded run's instructions tell as a trace's
//     do) lead to it from what was there before the run: a namespace a
//     global was loaded from, a cell the run did not make, or a ROOT, an
//     input object that takes a weak reference (a module, a function, a
//     class). The reads that lead to it are its check. A tensor that no
//     recorded read leads to (handed out by C code, found in a list or a
//     dict that is an input) has no origin: Err, and the recording stops;
//   - anything else has no origin: Err says what, and the recording stops.
// What it may not assume
//   - that a tensor with no mark was there before the run: C code that makes
//     a tensor without an operator and hides it in a container is not seen
//     (it is no pin either: no read leads to it);
//   - that a root lives: it is held weakly, and one that died leads nowhere.

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
    // Inputs, then pins, held weakly for the validation; the path each
    // input was found by (walk), and for each step of those paths through
    // a dict the path of the key (KEY) with the key.
    pub inputs: Vec<Pin>,
    pub paths: Vec<Box<[u32]>>,
    pub keys: Vec<(Box<[u32]>, Py<PyAny>)>,
    pub pins: Vec<Pin>,
    // What the run began on, by address: the params, then the locals.
    start: Vec<usize>,
    // The immutable inputs the list is specialized on: (position, value).
    pub exact: Vec<(u32, Py<PyAny>)>,
    // The input objects a recorded read can start from, held weakly.
    roots: Vec<Pin>,
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

// What a run starts from, in the one order every arrival is walked in: each
// of `all` (the block's params, then the frame's locals) and, through exact
// tuples and lists no longer than WIDE and as deep as REACH, their items,
// and through exact dicts of at most WIDE items whose keys are all exact
// strs their values in the dict's order, with the path ([position, index,
// ...]; the i-th value of a dict is the step DICT | i). `f` is told every
// object that is no such container and not NULL, and each key of a dict
// that is walked, by the path of its value with KEY | i for the last step;
// a container too long or too deep, and a dict with another key, is told as
// the object it is, and what it holds is not walked.
pub const WIDE: isize = 64;
pub const DICT: u32 = 1 << 31;
pub const KEY: u32 = 1 << 30;
pub unsafe fn walk(all: impl Iterator<Item = *mut PyObject>, f: &mut dyn FnMut(&[u32], *mut PyObject)) {
    // The items of an exact dict of exact str keys, if `o` is one.
    unsafe fn items(o: *mut PyObject) -> Option<Vec<(*mut PyObject, *mut PyObject)>> {
        unsafe {
            if ffi::PyDict_CheckExact(o) == 0 || ffi::PyDict_Size(o) > WIDE {
                return None;
            }
            let (mut at, mut k, mut v, mut found) = (0, std::ptr::null_mut(), std::ptr::null_mut(), Vec::new());
            while ffi::PyDict_Next(o, &mut at, &mut k, &mut v) != 0 {
                if ffi::PyUnicode_CheckExact(k) == 0 {
                    return None;
                }
                found.push((k, v));
            }
            Some(found)
        }
    }
    unsafe fn item(o: *mut PyObject, path: &mut Vec<u32>, f: &mut dyn FnMut(&[u32], *mut PyObject)) {
        unsafe {
            if path.len() <= REACH as usize && let Some(found) = items(o) {
                for (i, (k, v)) in found.into_iter().enumerate() {
                    path.push(KEY | i as u32);
                    f(path, k);
                    path.pop();
                    path.push(DICT | i as u32);
                    item(v, path, f);
                    path.pop();
                }
                return;
            }
            let (tuple, list) = (ffi::PyTuple_CheckExact(o) != 0, ffi::PyList_CheckExact(o) != 0);
            if !(tuple || list) || path.len() > REACH as usize || ffi::Py_SIZE(o) > WIDE {
                return f(path, o);
            }
            for i in 0..ffi::Py_SIZE(o) {
                // (Looked up again each time: `f` runs no code of the
                // program, and a list is as long as it was.)
                let x = if tuple { ffi::PyTuple_GET_ITEM(o, i) } else { ffi::PyList_GET_ITEM(o, i) };
                path.push(i as u32);
                if !x.is_null() {
                    item(x, path, f);
                }
                path.pop();
            }
        }
    }
    for (position, o) in all.enumerate().filter(|(_, o)| !o.is_null()) {
        unsafe { item(o, &mut vec![position as u32], f) };
    }
}

unsafe fn class_of(o: *mut PyObject) -> Option<Py<PyAny>> {
    unsafe { Some(Py::from_borrowed_ptr(Python::assume_attached(), ffi::Py_TYPE(o) as *mut PyObject)) }
}

pub unsafe fn class_text(o: *mut PyObject) -> String {
    if o.is_null() { "NULL".into() } else { unsafe { crate::exec::hist::class_name(ffi::Py_TYPE(o)) } }
}

impl Origins {
    // A run begins on `all`: the block's params, then the frame's locals.
    pub unsafe fn begin(all: impl Iterator<Item = *mut PyObject>, known: super::known::Known) -> Origins {
        DEATHS.with_borrow_mut(|d| d.clear());
        LOST.set(false);
        let all: Vec<*mut PyObject> = all.collect();
        let mut origins = Origins { known, start: all.iter().map(|o| *o as usize).collect(), ..Origins::default() };
        // Each key of a walked dict, by its path.
        let mut keys: HashMap<Box<[u32]>, *mut PyObject> = HashMap::new();
        unsafe {
            walk(all.into_iter(), &mut |path, o| {
                if path.last().is_some_and(|step| step & KEY != 0) {
                    keys.insert(path.into(), o);
                } else if plain_tensor(o) {
                    if origins.get(o).is_none() && origins.mark(o, Origin::In(origins.ninputs)) {
                        origins.inputs.push(Pin::weak(o));
                        origins.paths.push(path.into());
                        origins.ninputs += 1;
                    }
                } else if (*ffi::Py_TYPE(o)).tp_weaklistoffset != 0 {
                    // Weak, or no root: an input lives as long as the
                    // program keeps it.
                    let root = Pin::weak(o);
                    if root.counts().is_none() {
                        origins.roots.push(root);
                    }
                }
            });
            // The keys the inputs were found under.
            for path in &origins.paths {
                for (at, step) in path.iter().enumerate().filter(|(_, step)| *step & DICT != 0) {
                    let mut key: Vec<u32> = path[..at].to_vec();
                    key.push(KEY | (step & !DICT));
                    if !origins.keys.iter().any(|(p, _)| **p == *key) && let Some(k) = keys.get(key.as_slice()) {
                        origins.keys.push((key.into(), Py::from_borrowed_ptr(Python::assume_attached(), *k)));
                    }
                }
            }
        };
        origins
    }

    // Whether a death could not be noted: the releases are not known.
    pub fn lost() -> bool {
        LOST.get()
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
    // `depth`.
    pub unsafe fn template(&mut self, o: *mut PyObject, at: (u32, u32)) -> Made {
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
                        let roots: Vec<*mut PyObject> = self.roots.iter().filter_map(|r| r.live()).filter(|r| !r.is_null()).collect();
                        if !crate::exec::reads::leads_to(o, &roots) {
                            return Err("an operand of unknown origin: a tensor no recorded read leads to".into());
                        }
                        let i = self.pins.len() as u32;
                        if !self.mark(o, Origin::Pin(i)) {
                            return Err("an operand of unknown origin: a tensor no weak reference can be made to".into());
                        }
                        self.pins.push(Pin::weak(o));
                        Tmpl::Pin(i)
                    }
                };
                return Ok((tmpl, vec![class_of(o)]));
            }
            if super::immutable(o, 0) {
                if !self.known.lit(at) {
                    // An input: the list is for this value of it. Anything
                    // else is baked: the list is not the body.
                    match self.known.input(at).filter(|p| self.start.get(*p as usize) == Some(&(o as usize))) {
                        Some(position) => {
                            if !self.exact.iter().any(|(p, _)| *p == position) {
                                self.exact.push((position, Py::from_borrowed_ptr(Python::assume_attached(), o)));
                            }
                            self.known.set(at, true, String::new);
                        }
                        None => self.baked.push(format!("{} from {}", class_text(o), self.known.maker(at))),
                    }
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
    pub unsafe fn build(&mut self, kind: Built, parts: &[(*mut PyObject, u32)], depth: u32) -> Made {
        unsafe {
            let (mut tmpls, mut classes) = (Vec::new(), Vec::new());
            for (o, v) in parts {
                let (t, c) = self.template(*o, (depth, *v))?;
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
