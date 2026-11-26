//! How the tensors of a region's call share memory: what a lowering is made
//! for, besides their shapes.
//!
//! Given: the tensors of the traced call, the region's tensor inputs and then
//! its constants (the weights its graph reads); which of the inputs the
//! traced run also read from somewhere (`read`: a global, an attribute, an
//! item, a cell); and whether the region runs something made of its graph
//! (`whole`) or the graph as it was traced.
//!
//! Guarantees: `holds(inputs)` is true only on a call whose tensors (these
//! inputs, the constants as they are now) stand to one another as the
//! trace's did:
//!   - the inputs that were one object are one object, and an input that
//!     was also read is the object it was: a trace knows a tensor by its
//!     identity, so the graph has one input for all the ways it came by it;
//!   and with `whole`:
//!   - no other two are one object;
//!   - the tensors of a storage in which two overlapped are in one storage,
//!     each where it was in it, and every other tensor overlaps none;
//!   - no input has the conjugate or the negative bit: they say how its
//!     memory is to be read, and a kernel reads it as it is. (`marked`: a
//!     tensor of the trace has one; the tracer makes no such region.)
//! A graph run as traced needs the first only: it reads and writes the
//! call's own tensors in the program's order. What is made of a graph (by
//! AOT Autograd, by Inductor, in a CUDA graph, by a pass) has writes moved
//! and reads merged as the trace's tensors allowed.
//! A call costs one step per input, a lookup among the constants' storages.
//! All the tensors are gone through, once, only when an input is in a
//! storage with another tensor, and after `moved()`. No Python code runs.
//!
//! May not assume: that a call for which `holds` is true has the trace's
//! sizes and strides (it compares where tensors start; the region's other
//! checks compare the rest, before it is asked); that memory two storages
//! share is seen (memory handed to PyTorch twice: nothing in PyTorch knows
//! of it).

use std::cell::Cell;
use std::collections::{HashMap, HashSet};

use pyo3::ffi;
use pyo3::ffi::PyObject;
use pyo3::prelude::*;
use pyo3::types::PyList;

use super::ffi::GilCell;
use crate::bindings::ffi as tffi;

// A tensor of a call: which object it is, which storage its elements are in
// (0: none to look at), the bytes of the storage they span, and its
// conjugate and negative bits.
struct Place {
    obj: usize,
    storage: usize,
    low: usize,
    high: usize,
    marks: u8,
}

unsafe fn place(obj: *mut PyObject) -> Place {
    let mut span = tffi::TensorSpan::default();
    if let Some(t) = tffi::tensor_by_type(obj as usize).as_ref() {
        tffi::tensor_span(t, &mut span);
    }
    Place { obj: obj as usize, storage: span.storage, low: span.low, high: span.high, marks: span.marks }
}

// How the tensors of a call stand to one another, each named by its place in
// the call: equal for two calls that what was made of the graph can't tell
// apart.
#[derive(Default, PartialEq, Eq, Debug)]
struct Relation {
    // (tensor, the first tensor that is the same object), for the later ones.
    same: Vec<(u32, u32)>,
    // The tensors of each storage in which two overlap: (tensor, the first
    // of them, where it starts in the storage). Spans are compared: two
    // tensors whose elements interleave count as overlapping.
    shared: Vec<(u32, u32, usize)>,
    // (tensor, its conjugate and negative bits), for those that have one.
    marked: Vec<(u32, u8)>,
}

// One step per tensor, and a sort of the tensors of a storage that has
// several.
fn relation(places: &[Place]) -> Relation {
    let mut out = Relation::default();
    let mut first: HashMap<usize, u32> = HashMap::with_capacity(places.len());
    let mut storages: HashMap<usize, Vec<u32>> = HashMap::new();
    for (i, p) in places.iter().enumerate() {
        let i = i as u32;
        if p.marks != 0 {
            out.marked.push((i, p.marks));
        }
        let k = *first.entry(p.obj).or_insert(i);
        if k != i {
            out.same.push((i, k));
        } else if p.storage != 0 && p.low < p.high {
            storages.entry(p.storage).or_default().push(i);
        }
    }
    for members in storages.values_mut().filter(|m| m.len() > 1) {
        // A sweep through the storage: a tensor that starts before the end
        // of one before it overlaps it.
        members.sort_by_key(|&i| places[i as usize].low);
        let (mut end, mut overlap) = (0, false);
        for &i in members.iter() {
            let p = &places[i as usize];
            overlap |= p.low < end;
            end = end.max(p.high);
        }
        if overlap {
            let lead = members.iter().copied().min().unwrap_or(0);
            out.shared.extend(members.iter().map(|&i| (i, lead, places[i as usize].low)));
        }
    }
    out.shared.sort_unstable();
    out
}

// The constants at the last count, for the look-up of a call's inputs.
#[derive(Default)]
struct Constants {
    // Their objects and their storages (no two live things have one address).
    known: HashSet<usize>,
    // Whether a call stands as the trace did if each of its inputs is alone:
    // no object twice, no storage with another tensor in it.
    alone: bool,
    // The objects and storages of the inputs of the call being looked at.
    seen: HashSet<usize>,
}

/// See the top of exec/overlap.rs. Made by torch._rust_inductor.trace_region.
#[pyclass(frozen, module = "torch._rust")]
pub struct Overlap {
    inputs: usize,
    consts: Vec<Py<PyAny>>,
    whole: bool,
    traced: Relation,
    // The inputs the traced run also read from somewhere, by place.
    read: Vec<(usize, Py<PyAny>)>,
    // The bits of the trace's inputs.
    marks: Vec<u8>,
    constants: GilCell<Constants>,
    // The constants are to be counted again: a constant's data is somewhere
    // else, or nothing says when one is (`told`).
    stale: Cell<bool>,
    told: Cell<bool>,
}

unsafe impl Sync for Overlap {}

impl Overlap {
    unsafe fn places(&self, input: impl Fn(usize) -> Place) -> Vec<Place> {
        (0..self.inputs).map(input).chain(self.consts.iter().map(|c| place(c.as_ptr()))).collect()
    }

    // Has the constants counted again before the next call is looked at: for
    // what watches where their data is (exec/reads.rs), which then says so
    // from now on.
    pub fn moved(&self) {
        self.told.set(true);
        self.stale.set(true);
    }
}

#[pymethods]
impl Overlap {
    #[new]
    #[pyo3(signature = (inputs, consts, whole, read=vec![]))]
    fn new(inputs: Vec<Bound<'_, PyAny>>, consts: Vec<Bound<'_, PyAny>>, whole: bool, read: Vec<usize>) -> Self {
        let n = inputs.len();
        let mut out = Overlap {
            inputs: n,
            consts: consts.into_iter().map(Bound::unbind).collect(),
            whole,
            traced: Relation::default(),
            read: read.into_iter().filter(|&i| i < n).map(|i| (i, inputs[i].clone().unbind())).collect(),
            marks: inputs.iter().map(|x| unsafe { place(x.as_ptr()) }.marks).collect(),
            constants: GilCell::new(Constants::default()),
            stale: Cell::new(true),
            told: Cell::new(false),
        };
        out.traced = relation(&unsafe { out.places(|i| place(inputs[i].as_ptr())) });
        if !whole {
            out.traced.same.retain(|&(i, _)| (i as usize) < n);
            out.traced.shared.clear();
            out.traced.marked.clear();
        }
        out
    }

    /// Whether there is anything to ask ``holds`` for.
    #[getter]
    fn needed(&self) -> bool {
        self.whole || !self.traced.same.is_empty() || !self.read.is_empty()
    }

    /// Whether a tensor of the trace has the conjugate or the negative bit
    /// and the region runs something made of its graph: a kernel reads the
    /// memory as it is, so such a region is not made.
    #[getter]
    fn marked(&self) -> bool {
        !self.traced.marked.is_empty()
    }

    /// The trace's relation, as a key: what is made of a graph is for one.
    fn key(&self) -> String {
        format!("{:?}", self.traced)
    }

    /// Whether the tensors of this call stand to one another as the trace's
    /// did. ``inputs`` are the region's inputs, the tensors first.
    fn holds(&self, inputs: &Bound<'_, PyList>) -> bool {
        if inputs.len() < self.inputs {
            return false;
        }
        let item = |i: usize| unsafe { ffi::PyList_GET_ITEM(inputs.as_ptr(), i as ffi::Py_ssize_t) };
        if !self.read.iter().all(|(i, was)| item(*i) == was.as_ptr()) {
            return false;
        }
        if !self.whole {
            return self.traced.same.iter().all(|&(i, k)| item(i as usize) == item(k as usize));
        }
        let c = unsafe { &mut *self.constants.get() };
        if self.stale.replace(!self.told.get()) {
            // The inputs stand in as tensors that are alone (their numbers
            // are no addresses).
            let places = unsafe { self.places(|i| Place { obj: i + 1, storage: 0, low: 0, high: 0, marks: self.marks[i] }) };
            c.alone = relation(&places) == self.traced;
            c.known.clear();
            for p in &places[self.inputs..] {
                c.known.insert(p.obj);
                c.known.insert(p.storage);
            }
            c.known.remove(&0);
        }
        c.seen.clear();
        let mut alone = true;
        for i in 0..self.inputs {
            let p = unsafe { place(item(i)) };
            if p.marks != self.marks[i] {
                return false;
            }
            alone &= !c.known.contains(&p.obj) && c.seen.insert(p.obj);
            alone &= p.storage == 0 || (!c.known.contains(&p.storage) && c.seen.insert(p.storage));
        }
        if alone {
            return c.alone;
        }
        relation(&unsafe { self.places(|i| place(item(i))) }) == self.traced
    }
}
