//! How the tensors a region is entered with stand to one another in memory:
//! a region is recorded for one such relation and entered only with it.
//!
//! Given: the tensors of the traced call, the region's tensor inputs and then
//! its constants (the weights its graph reads), and, once the trace's reads
//! are settled, which of the inputs the traced run also read from somewhere
//! (`also_read`: a global, an attribute, an item, a cell).
//!
//! The place of a tensor is the bytes from its first element to its last,
//! by address, in the memory of its device type; a tensor without elements
//! is nowhere. Two tensors overlap if their places meet. Elements that
//! interleave (`x[::2]` and `x[1::2]`, two columns) count as overlapping
//! though they share none: with the sizes and strides the region's other
//! checks compare, the same distance between two starts is the same shared
//! elements, or none again, so nothing is taken for apart that is not.
//!
//! Guarantees:
//!   - `unplaced` says so if the place of a tensor of the trace cannot be
//!     said: it is no tensor, or one without a storage of its own to point
//!     at (not strided, a Python-dispatch tensor, one whose storage gives no
//!     address), or it is on a device type other than the CPU and CUDA,
//!     whose addresses are not known to compare. No call holds then: the
//!     tracer makes no such region. A region of one tensor asks for no place.
//!   - `holds(inputs)` is true only on a call whose tensors (these inputs,
//!     the constants as they are now) can each be placed and stand to one
//!     another as the trace's did, for every pair of them: two that were
//!     one object are one object and two that were not are not; an input
//!     that was also read is the object it was (a trace knows a tensor by
//!     its identity, so the graph has one input for all the ways it came by
//!     it); two that were apart are apart; and two that overlapped, or were
//!     joined by others that did, are as far from one another as they were,
//!     each where it was in its storage, the same ones in one storage.
//!   - `two_storages`: two tensors of the trace overlap that are in two
//!     storages (memory handed to PyTorch twice). What is made of a graph
//!     knows shared memory by its storage: the tracer makes no such region
//!     for a lowering.
//! A call costs one step per input, a search among the constants' places
//! and a sort of the inputs' own. All the tensors are gone through, once,
//! only when an input is one of the other tensors or overlaps one, and
//! after `moved()`; without anyone to say when a constant moved, on every
//! call. No Python code runs.
//!
//! May not assume: that a call for which `holds` is true has the trace's
//! sizes and strides, or its conjugate and negative bits (the region's
//! other checks compare the tensor key, before it is asked); that one
//! piece of memory at two addresses (a file mapped twice) is seen as one:
//! nothing in PyTorch knows of it.

use std::cell::Cell;
use std::collections::HashMap;

use pyo3::ffi;
use pyo3::ffi::PyObject;
use pyo3::prelude::*;
use pyo3::types::PyList;

use super::ffi::GilCell;
use crate::bindings::ffi as tffi;

// A tensor of a call: which object it is, which storage its elements are in
// and how many bytes into it they start, the memory of which device type
// they are in, and their addresses, [low, high).
#[derive(Clone, Copy, PartialEq, Eq)]
struct Place {
    obj: usize,
    storage: usize,
    at: usize,
    space: u8,
    low: usize,
    high: usize,
}

// None: its place cannot be said (see the top of the file).
unsafe fn place(obj: *mut PyObject) -> Option<Place> {
    let t = tffi::tensor_by_type(obj as usize).as_ref()?;
    let mut span = tffi::TensorSpan::default();
    tffi::tensor_span(t, &mut span).then_some(Place { obj: obj as usize, storage: span.storage, at: span.at, space: span.space, low: span.low, high: span.high })
}

// A tensor of a run of memory in which tensors overlap: the first tensor of
// the run, the first of the run that is in this one's storage, how far this
// one starts from the first, where it starts in its storage, and its bytes.
#[derive(PartialEq, Eq, PartialOrd, Ord, Debug)]
struct Joined {
    tensor: u32,
    lead: u32,
    mate: u32,
    from_lead: isize,
    at: usize,
    bytes: usize,
}

// How the tensors of a call stand to one another, each named by its place in
// the call: equal for two calls exactly if every pair of tensors stands in
// one as in the other.
#[derive(Default, PartialEq, Eq, Debug)]
struct Relation {
    // (tensor, the first tensor that is the same object), for the later ones.
    same: Vec<(u32, u32)>,
    // The tensors of every run of two or more. A tensor in none overlaps
    // none, and two runs do not meet.
    joined: Vec<Joined>,
}

// One step per tensor and a sort by address.
fn relation(places: &[Place]) -> Relation {
    let mut out = Relation::default();
    let mut first: HashMap<usize, u32> = HashMap::with_capacity(places.len());
    let mut order: Vec<u32> = Vec::with_capacity(places.len());
    for (i, p) in places.iter().enumerate() {
        let i = i as u32;
        let k = *first.entry(p.obj).or_insert(i);
        if k != i {
            out.same.push((i, k));
        } else if p.low < p.high {
            order.push(i);
        }
    }
    let at = |i: u32| &places[i as usize];
    order.sort_unstable_by_key(|&i| (at(i).space, at(i).low, i));
    // A sweep through memory: a tensor that starts before the end of those
    // before it is in their run.
    let mut from = 0;
    while from < order.len() {
        let (space, mut end, mut to) = (at(order[from]).space, at(order[from]).high, from + 1);
        while to < order.len() && at(order[to]).space == space && at(order[to]).low < end {
            end = end.max(at(order[to]).high);
            to += 1;
        }
        let run = &order[from..to];
        if run.len() > 1 {
            let lead = run.iter().copied().fold(run[0], u32::min);
            let mut mates: HashMap<usize, u32> = HashMap::with_capacity(run.len());
            for &i in run {
                let mate = mates.entry(at(i).storage).or_insert(i);
                *mate = (*mate).min(i);
            }
            out.joined.extend(run.iter().map(|&i| {
                let p = at(i);
                Joined { tensor: i, lead, mate: mates[&p.storage], from_lead: p.low as isize - at(lead).low as isize, at: p.at, bytes: p.high - p.low }
            }));
        }
        from = to;
    }
    out.joined.sort_unstable();
    out
}

// The constants at the last count, for the look-up of a call's inputs.
#[derive(Default)]
struct Constants {
    // Each could be placed: no call holds otherwise.
    placed: bool,
    // Their objects, sorted.
    objs: Vec<usize>,
    // Their places, as runs of memory that do not meet, sorted:
    // (space, low, high).
    spans: Vec<(u8, usize, usize)>,
    // The places of those the trace's relation names (`Overlap::tied`).
    tied: Vec<Place>,
    // Whether a call stands as the trace did if each of its inputs is alone:
    // no other tensor's object, and overlapping none.
    alone: bool,
    // The objects and the places of the inputs of the call being looked at.
    ids: Vec<usize>,
    ranges: Vec<(u8, usize, usize)>,
}

impl Constants {
    unsafe fn count(&mut self, of: &Overlap, traced: &Relation) {
        let consts: Option<Vec<Place>> = of.consts.iter().map(|c| place(c.as_ptr())).collect();
        self.placed = consts.is_some();
        let Some(consts) = consts else { return };
        // The inputs stand in as tensors that are alone and nowhere (their
        // numbers are no addresses).
        let alone = (0..of.inputs).map(|i| Place { obj: i + 1, storage: 0, at: 0, space: 0, low: 0, high: 0 });
        self.alone = relation(&alone.chain(consts.iter().copied()).collect::<Vec<_>>()) == *traced;
        self.tied = of.tied.iter().map(|&j| consts[j]).collect();
        self.objs = consts.iter().map(|p| p.obj).collect();
        self.objs.sort_unstable();
        let mut spans: Vec<_> = consts.iter().filter(|p| p.low < p.high).map(|p| (p.space, p.low, p.high)).collect();
        spans.sort_unstable();
        self.spans.clear();
        for s in spans {
            match self.spans.last_mut() {
                Some(last) if last.0 == s.0 && s.1 <= last.2 => last.2 = last.2.max(s.2),
                _ => self.spans.push(s),
            }
        }
    }

    // Whether `p` overlaps a constant.
    fn meets(&self, p: &Place) -> bool {
        if p.low == p.high {
            return false;
        }
        // The first run that ends behind where `p` starts.
        let k = self.spans.partition_point(|s| (s.0, s.2) <= (p.space, p.low));
        self.spans.get(k).is_some_and(|s| s.0 == p.space && s.1 < p.high)
    }
}

/// See the top of exec/overlap.rs. Made by torch._rust_inductor.trace_region.
#[pyclass(frozen, module = "torch._rust")]
pub struct Overlap {
    inputs: usize,
    consts: Vec<Py<PyAny>>,
    // None: a tensor of the trace could not be placed (`unplaced` names it).
    traced: Option<Relation>,
    unplaced: Option<String>,
    // The constants the trace's relation names, by their place among the
    // constants: nothing says when the storage of one is another.
    tied: Vec<usize>,
    // The inputs the traced run also read from somewhere, by place.
    read: GilCell<Vec<(usize, Py<PyAny>)>>,
    constants: GilCell<Constants>,
    // The constants are to be counted again: a constant's data is somewhere
    // else, or nothing says when one is (`told`).
    stale: Cell<bool>,
    told: Cell<bool>,
}

unsafe impl Sync for Overlap {}

impl Overlap {
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
    fn new(inputs: Vec<Bound<'_, PyAny>>, consts: Vec<Bound<'_, PyAny>>) -> Self {
        let n = inputs.len();
        let mut places = Vec::with_capacity(n + consts.len());
        let mut unplaced = None;
        // One tensor stands to no other: it is not asked where it is.
        let asked = if n + consts.len() > 1 { n + consts.len() } else { 0 };
        for (i, x) in inputs.iter().chain(&consts).take(asked).enumerate() {
            match unsafe { place(x.as_ptr()) } {
                Some(p) => places.push(p),
                None => {
                    let which = if i < n { format!("input {i}") } else { format!("constant {}", i - n) };
                    unplaced.get_or_insert(format!("where {which} of its tensors is in memory cannot be said (it has no storage of its own, or is on a device whose addresses do not compare)"));
                }
            }
        }
        let traced = unplaced.is_none().then(|| relation(&places));
        let named = |j: &Joined| [j.tensor, j.lead, j.mate];
        let mut tied: Vec<usize> = traced.iter().flat_map(|r| r.joined.iter().flat_map(named)).filter_map(|i| (i as usize).checked_sub(n)).collect();
        tied.sort_unstable();
        tied.dedup();
        Overlap {
            inputs: n,
            consts: consts.into_iter().map(Bound::unbind).collect(),
            traced,
            unplaced,
            tied,
            read: GilCell::new(vec![]),
            constants: GilCell::new(Constants::default()),
            stale: Cell::new(true),
            told: Cell::new(false),
        }
    }

    /// ``read``: which of the trace's ``inputs``, by place, the traced run
    /// also read from somewhere. A call holds only if they are those objects.
    fn also_read(&self, inputs: Vec<Bound<'_, PyAny>>, read: Vec<usize>) -> PyResult<()> {
        if inputs.len() != self.inputs || read.iter().any(|&i| i >= self.inputs) {
            return Err(pyo3::exceptions::PyIndexError::new_err("also_read: not the trace's inputs"));
        }
        unsafe { *self.read.get() = read.into_iter().map(|i| (i, inputs[i].clone().unbind())).collect() };
        Ok(())
    }

    /// Whether there is anything to ask ``holds`` for: two tensors, or an
    /// input the traced run also read.
    #[getter]
    fn needed(&self) -> bool {
        self.inputs + self.consts.len() > 1 || !unsafe { &*self.read.get() }.is_empty()
    }

    /// Why no call holds, if none does: the tensor of the trace whose place
    /// in memory cannot be said. No region is made of such a trace.
    #[getter]
    fn unplaced(&self) -> Option<&str> {
        self.unplaced.as_deref()
    }

    /// Whether two tensors of the trace overlap that are in two storages:
    /// no region is made of such a trace for a lowering.
    #[getter]
    fn two_storages(&self) -> bool {
        self.traced.iter().flat_map(|r| &r.joined).any(|j| j.mate != j.lead)
    }

    /// The trace's relation, as a key: what is made of a graph is for one.
    fn key(&self) -> String {
        format!("{:?}", self.traced)
    }

    /// Whether the tensors of this call stand to one another as the trace's
    /// did. ``inputs`` are the region's inputs, the tensors first.
    fn holds(&self, inputs: &Bound<'_, PyList>) -> bool {
        let Some(traced) = &self.traced else { return false };
        if inputs.len() < self.inputs {
            return false;
        }
        let item = |i: usize| unsafe { ffi::PyList_GET_ITEM(inputs.as_ptr(), i as ffi::Py_ssize_t) };
        if !unsafe { &*self.read.get() }.iter().all(|(i, was)| item(*i) == was.as_ptr()) {
            return false;
        }
        if self.inputs + self.consts.len() < 2 {
            return true;
        }
        let c = unsafe { &mut *self.constants.get() };
        let tied = |c: &Constants| self.tied.iter().zip(&c.tied).all(|(&j, was)| unsafe { place(self.consts[j].as_ptr()) } == Some(*was));
        if self.stale.replace(!self.told.get()) || !tied(c) {
            unsafe { c.count(self, traced) };
        }
        if !c.placed {
            return false;
        }
        c.ids.clear();
        c.ranges.clear();
        let mut alone = true;
        for i in 0..self.inputs {
            let Some(p) = (unsafe { place(item(i)) }) else { return false };
            alone &= c.objs.binary_search(&p.obj).is_err() && !c.meets(&p);
            c.ids.push(p.obj);
            if p.low < p.high {
                c.ranges.push((p.space, p.low, p.high));
            }
        }
        if alone {
            c.ids.sort_unstable();
            c.ranges.sort_unstable();
            alone = c.ids.windows(2).all(|w| w[0] != w[1]) && c.ranges.windows(2).all(|w| w[0].0 != w[1].0 || w[0].2 <= w[1].1);
        }
        if alone {
            return c.alone;
        }
        let all = (0..self.inputs).map(item).chain(self.consts.iter().map(|c| c.as_ptr()));
        let places: Option<Vec<Place>> = all.map(|x| unsafe { place(x) }).collect();
        places.is_some_and(|p| relation(&p) == *traced)
    }
}
