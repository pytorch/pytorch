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
//! has a first address and no bytes. Two tensors overlap if their places
//! meet. Elements that interleave (`x[::2]` and `x[1::2]`, two columns)
//! count as overlapping though they share none: with the sizes and strides
//! the region's other checks compare, the same distance between two starts
//! is the same shared elements, or none again, so nothing is taken for
//! apart that is not.
//!
//! The relation of a call's tensors is:
//!   - which of them are one object;
//!   - which of them are in one storage, whether their places meet or not
//!     (a view made of one inside the region, `as_strided`, reaches all of
//!     its storage), and for those the bytes from the first element of one
//!     to the first element of the other;
//!   - which of them overlap, in one storage or in two (memory handed to
//!     PyTorch twice), and for those again the bytes between their first
//!     elements.
//! It is relative: where in its storage a tensor starts is no part of it.
//! Two views that slide through one buffer at the same distance stand the
//! same on every call; moving one against the other is another relation.
//! That is enough for a region that runs its graph as traced: every
//! operator runs again on the call's own tensors, wherever they start.
//! (What a body makes of the offset itself, `x.storage_offset()`, is a
//! number it reads out of a tensor: the tracer's matter, not this one's.)
//! It is not enough for what a lowering makes of a graph that writes to a
//! tensor: AOT Autograd puts overlapping inputs of which one is written
//! under one base that spans their storage and builds each of them from it
//! at the offset the traced tensor had. For such a region (`fixed`) the
//! relation also has, for every tensor that overlaps another, where it
//! starts in its storage.
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
//!     it); two that were in two storages and apart are so again; and two
//!     that were in one storage, or overlapped, or were joined by others
//!     that did, are as far from one another as they were.
//!   - `two_storages`: two tensors of the trace overlap that are in two
//!     storages. What is made of a graph knows shared memory by its storage:
//!     the tracer makes no such region for a lowering.
//! A call costs one step per input, two searches among the constants (their
//! storages, their places) and a sort of the inputs' own. The relation is
//! made again, of the inputs and the constants as last counted, only when
//! an input is one of the other tensors, is in the storage of one or
//! overlaps one. The constants are counted again after `moved()`; without
//! anyone to say when a constant moved, on every call. No Python code runs.
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
// they are in, and their addresses, [low, high); `low` is where the first
// element is, or would be.
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
// the run, how far this one starts from that one, its bytes and, in the
// relation of a region made for the offsets of its tensors (`fixed`), where
// it starts in its storage (None otherwise).
#[derive(PartialEq, Eq, PartialOrd, Ord, Debug)]
struct Joined {
    tensor: u32,
    lead: u32,
    from_lead: isize,
    bytes: usize,
    at: Option<usize>,
}

// How the tensors of a call stand to one another, each named by its place in
// the call: equal for two calls exactly if every pair of tensors stands in
// one as in the other.
#[derive(Default, PartialEq, Eq, Debug)]
struct Relation {
    // (tensor, the first tensor that is the same object), for the later ones.
    same: Vec<(u32, u32)>,
    // (tensor, the first tensor in its storage, the bytes from that one's
    // first element to this one's), for the later ones of every storage.
    // A tensor in none is alone in its storage.
    shared: Vec<(u32, u32, isize)>,
    // The tensors of every run of two or more. A tensor in none overlaps
    // none, and two runs do not meet.
    joined: Vec<Joined>,
    // A run has tensors of two storages.
    split: bool,
    // Two tensors of one storage are in no run together.
    apart: bool,
}

// One step per tensor and two sorts, by storage and by address. `fixed`:
// with where in its storage each tensor of a run starts.
fn relation(places: &[Place], fixed: bool) -> Relation {
    let mut out = Relation::default();
    let mut first: HashMap<usize, u32> = HashMap::with_capacity(places.len());
    let mut objects: Vec<u32> = Vec::with_capacity(places.len());
    for (i, p) in places.iter().enumerate() {
        let i = i as u32;
        let k = *first.entry(p.obj).or_insert(i);
        if k != i {
            out.same.push((i, k));
        } else {
            objects.push(i);
        }
    }
    let at = |i: u32| &places[i as usize];
    // The first tensor of the run each tensor is in (itself: in none).
    let mut run_of: Vec<u32> = (0..places.len() as u32).collect();
    let mut order = objects.clone();
    order.sort_unstable_by_key(|&i| (at(i).storage, i));
    for group in order.chunk_by(|&a, &b| at(a).storage == at(b).storage) {
        let mate = group[0];
        out.shared.extend(group[1..].iter().map(|&i| (i, mate, at(i).low as isize - at(mate).low as isize)));
    }
    out.shared.sort_unstable();
    order.clear();
    order.extend(objects.iter().filter(|&&i| at(i).low < at(i).high));
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
            out.split |= run.iter().any(|&i| at(i).storage != at(lead).storage);
            out.joined.extend(run.iter().map(|&i| {
                let p = at(i);
                run_of[i as usize] = lead;
                Joined { tensor: i, lead, from_lead: p.low as isize - at(lead).low as isize, bytes: p.high - p.low, at: fixed.then_some(p.at) }
            }));
        }
        from = to;
    }
    out.joined.sort_unstable();
    out.apart = out.shared.iter().any(|&(i, mate, _)| run_of[i as usize] != run_of[mate as usize]);
    out
}

// The constants at the last count, for the look-up of a call's inputs.
#[derive(Default)]
struct Constants {
    // Each could be placed (`places`): no call holds otherwise.
    placed: bool,
    places: Vec<Place>,
    // Their objects and their storages, sorted.
    objs: Vec<usize>,
    storages: Vec<usize>,
    // Their places, as runs of memory that do not meet, sorted:
    // (space, low, high).
    spans: Vec<(u8, usize, usize)>,
    // Whether a call stands as the trace did if each of its inputs is alone:
    // no other tensor's object, in no other tensor's storage, and
    // overlapping none.
    alone: bool,
    // The tensors of the call being looked at: the inputs' places, with the
    // constants' behind them when the relation is made; the inputs' objects,
    // storages and places for the look among themselves.
    all: Vec<Place>,
    ids: Vec<usize>,
    stores: Vec<usize>,
    ranges: Vec<(u8, usize, usize)>,
}

impl Constants {
    unsafe fn count(&mut self, of: &Overlap, traced: &Relation) {
        let consts: Option<Vec<Place>> = of.consts.iter().map(|c| place(c.as_ptr())).collect();
        self.placed = consts.is_some();
        let Some(consts) = consts else { return };
        // The inputs stand in as tensors that are alone (their numbers are
        // no addresses).
        let alone = (0..of.inputs).map(|i| Place { obj: i + 1, storage: i + 1, at: 0, space: 0, low: 0, high: 0 });
        self.alone = relation(&alone.chain(consts.iter().copied()).collect::<Vec<_>>(), of.fixed) == *traced;
        self.objs = consts.iter().map(|p| p.obj).collect();
        self.objs.sort_unstable();
        self.storages = consts.iter().map(|p| p.storage).collect();
        self.storages.sort_unstable();
        let mut spans: Vec<_> = consts.iter().filter(|p| p.low < p.high).map(|p| (p.space, p.low, p.high)).collect();
        spans.sort_unstable();
        self.spans.clear();
        for s in spans {
            match self.spans.last_mut() {
                Some(last) if last.0 == s.0 && s.1 <= last.2 => last.2 = last.2.max(s.2),
                _ => self.spans.push(s),
            }
        }
        self.places = consts;
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
    // What runs for the region's graph is made for where in its storage
    // each of the traced tensors that overlap another started.
    fixed: bool,
    // None: a tensor of the trace could not be placed (`unplaced` names it).
    traced: Option<Relation>,
    unplaced: Option<String>,
    // The inputs the traced run also read from somewhere, by place.
    read: GilCell<Vec<(usize, Py<PyAny>)>>,
    constants: GilCell<Constants>,
    // The constants are to be counted again: a constant's first element is
    // somewhere else or in another storage, or nothing says when one is
    // (`told`).
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
    /// ``fixed``: what runs for the region's graph may build overlapping
    /// tensors from one base, at the offsets the traced ones had (a
    /// lowering of a graph that writes to a tensor).
    #[new]
    #[pyo3(signature = (inputs, consts, fixed=false))]
    fn new(inputs: Vec<Bound<'_, PyAny>>, consts: Vec<Bound<'_, PyAny>>, fixed: bool) -> Self {
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
        let traced = unplaced.is_none().then(|| relation(&places, fixed));
        Overlap {
            inputs: n,
            consts: consts.into_iter().map(Bound::unbind).collect(),
            fixed,
            traced,
            unplaced,
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
        self.traced.as_ref().is_some_and(|r| r.split)
    }

    /// Whether two tensors of the trace are in one storage without
    /// overlapping, directly or through others: a lowering takes such two
    /// for independent, which they are while every view the graph makes
    /// stays inside the tensor it is made of.
    #[getter]
    fn apart(&self) -> bool {
        self.traced.as_ref().is_some_and(|r| r.apart)
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
        if self.stale.replace(!self.told.get()) {
            unsafe { c.count(self, traced) };
        }
        if !c.placed {
            return false;
        }
        c.all.clear();
        let mut alone = true;
        for i in 0..self.inputs {
            let Some(p) = (unsafe { place(item(i)) }) else { return false };
            alone &= c.objs.binary_search(&p.obj).is_err() && c.storages.binary_search(&p.storage).is_err() && !c.meets(&p);
            c.all.push(p);
        }
        if alone {
            c.ids.clear();
            c.ids.extend(c.all.iter().map(|p| p.obj));
            c.ids.sort_unstable();
            c.stores.clear();
            c.stores.extend(c.all.iter().map(|p| p.storage));
            c.stores.sort_unstable();
            c.ranges.clear();
            c.ranges.extend(c.all.iter().filter(|p| p.low < p.high).map(|p| (p.space, p.low, p.high)));
            c.ranges.sort_unstable();
            alone = c.ids.windows(2).all(|w| w[0] != w[1]) && c.stores.windows(2).all(|w| w[0] != w[1]) && c.ranges.windows(2).all(|w| w[0].0 != w[1].0 || w[0].2 <= w[1].1);
        }
        if alone {
            return c.alone;
        }
        c.all.extend_from_slice(&c.places);
        relation(&c.all, self.fixed) == *traced
    }
}
