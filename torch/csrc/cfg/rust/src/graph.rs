//! Graphs: the tensor work of a specialized block as a unit of its own.
//!
//! A graph is a run of the instructions of one specialized block: kernel
//! call sites, its nodes, with what sits between them (reads of globals and
//! attributes, constants, arithmetic on sizes). The graph pass finds the runs
//! of a block once the other passes are done with it and describes each as a
//! `Path`. The VM runs a graph as it runs the block, instruction by
//! instruction and each node through its site, on eager's autograd tape; the
//! path is what `torch.vm.regions` reports.
//!
//! When the list of passes has a pass over graphs, a path is also what such
//! a pass is given, as a `torch.fx.GraphModule`. The instruction `Op::Graph`
//! then stands in front of the run. A call that finds no exported form for
//! its values runs the instructions and takes down what they did (the site
//! each node hit, the values the nodes took from outside the graph): from
//! that one run the graph is built in Python, the passes are applied, and
//! what they return is kept with the conditions it was made under
//! (`Exported`). A later call that meets them runs the callable in place of
//! the nodes.

use crate::analysis::BitSet;
use crate::capi::{self, TcfgApi, TcfgSite, TcfgTensorInfo, SITE_RANDOM, SITE_SCALAR, SITE_WRITES};
use crate::ir::*;
use crate::pyffi::{self, new_ref, xdecref, Obj, PyObj, Shared};
use crate::shapes::{key_of, ShapeGuard, KIND_BYTES};
use crate::spec::{CalleeFact, CfgPass, InstFacts, Kernels, PassCx, UInst, Unit};
use pyo3::ffi;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

/// A value of a path. The path of one block numbers its values as the
/// function does.
pub type PathValue = V;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    /// A kernel call site: an operation of the graph.
    Node,
    /// Makes a value from its operands alone: a constant, a size, arithmetic
    /// on numbers, a check.
    Pure,
    /// Reads a global, a cell, or an attribute or item of an object, which
    /// entered no frame and ran no kernel when it was profiled.
    Read,
    /// Keeps values alive, or binds a variable of the frame.
    Residual,
}

/// When an instruction of a path runs once a callable stands for its nodes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Stage {
    /// In front of the callable: it depends on no node.
    Before,
    /// Not at all: the callable does its work.
    Inside,
    /// Behind the callable, on what it returned.
    After,
    /// Not at all: nothing behind the path needs what it does.
    Skipped,
}

pub struct PathInst {
    pub op: Op,
    pub dst: PathValue,
    pub kills: Vec<PathValue>,
    pub kind: Kind,
    pub bc: u32,
    pub stage: Stage,
    /// An integer the exported graph computes itself: a size of one of its
    /// tensors, or arithmetic on such sizes.
    pub in_graph: bool,
}

/// What a call has to find for an exported form to be run for it.
pub struct Check {
    pub target: PathValue,
    pub cond: Cond,
}

pub enum Cond {
    /// The very object.
    Is(PyObj),
    /// A value of the same exact type that is equal (see `same_value`).
    ValueEq(PyObj),
    /// A plain tensor like the one the graph was exported for.
    Tensor(Row),
    /// An object of this very type: of a form without a callable, for the
    /// operand that kept the call from being exported. The type is kept:
    /// one that is made per call would otherwise be gone, and its address
    /// that of another.
    Type(PyObj),
}

/// What a callable made for a tensor depends on, but for its sizes and
/// strides: those are numbers of the form (`Exported::sizes`).
#[derive(PartialEq)]
pub struct Row {
    /// The type of its Python object, and that type's version.
    pub ty: usize,
    pub type_version: u32,
    /// Its rank, dtype, device, `requires_grad`, dispatch keys and whether
    /// it has strides, as `shapes::key_of` gives them.
    pub kind: [u8; KIND_BYTES],
    /// It carries a forward-mode gradient: nothing is exported for one.
    pub forward: bool,
    /// The first of the tensors that are checked that is this same tensor.
    pub same: usize,
}

/// What the passes over a graph made of one path, for calls like the one it
/// was recorded on.
pub struct Exported {
    /// The modes and the local dispatch keys of the thread, and what else
    /// of its state a compiled graph depends on (`thread_state`).
    pub modes: u64,
    pub included: u64,
    pub excluded: u64,
    pub state: u64,
    /// On the values the nodes take from outside the graph, evaluated once
    /// the instructions in front of the callable have run.
    pub checks: Vec<Check>,
    /// The sizes and strides of the tensors that are checked, one after the
    /// other, as the form was made on them: it leaves none of them open.
    pub sizes: ShapeGuard,
    /// The tensors the callable is called with.
    pub args: Vec<PathValue>,
    /// What the passes asked of an argument (`torch.vm.guard`): its position
    /// and an object with `holds(tensor)`.
    pub guards: Vec<(usize, PyObj)>,
    /// What runs in place of the nodes. None for a kind of call that is not
    /// exported, for the reason `closed` gives: such a call runs the
    /// instructions, and is not recorded again.
    pub callable: Option<PyObj>,
    /// For a form that was closed because numbers were new at every call
    /// (`GraphSite::ever_new`): where those numbers are, what they were at
    /// the last call of the kind, and how many calls in a row had them.
    pub varying: Vec<PathValue>,
    pub last: Shared<Vec<PyObj>>,
    pub repeats: AtomicU32,
    pub closed: Option<String>,
    /// It has returned once. Until then, a call of it that raises is taken
    /// back: the instructions run instead.
    pub proven: AtomicBool,
    pub hits: AtomicU64,
}

pub struct Path {
    /// Where it is entered: the block, and the generic instruction its first
    /// node stands for.
    pub root: (usize, usize),
    /// Where the generic code goes on behind it.
    pub land: (usize, usize),
    /// The values it takes from before it, in the order it first reads them.
    pub inputs: Vec<V>,
    pub body: Vec<PathInst>,
    /// The values its nodes take that the graph does not compute: inputs,
    /// and what the instructions in front of the callable read. In the order
    /// they are first taken.
    pub feeds: Vec<PathValue>,
    /// What is read behind it of what the graph computes, in the order it
    /// is computed: results of nodes, and integers. A callable returns
    /// these, and then one integer per entry of `exits`.
    pub outputs: Vec<PathValue>,
    /// The guards of the block on results of nodes that a callable does not
    /// return: it returns what each looks at instead, the rank of the tensor
    /// or the size of the dimension.
    pub exits: Vec<GuardCond>,
    /// The variables of the frame its instructions bind, each with the value
    /// it holds at the end, or UNDEF where a callable does not give it.
    pub bound: Vec<(u32, PathValue)>,
    /// The operators of its nodes, as the dispatcher names them.
    pub operators: Vec<String>,
    /// Why it ends where it does: what the instruction behind it is, or how
    /// the block ends.
    pub ends: String,
    pub exported: Shared<Vec<Arc<Exported>>>,
    pub hits: AtomicU64,
    pub misses: AtomicU64,
}

impl Path {
    /// The values that die in the path: released when a callable has run in
    /// place of its instructions.
    pub fn dying(&self) -> impl Iterator<Item = PathValue> + '_ {
        self.body.iter().flat_map(|inst| inst.kills.iter().copied())
    }

    /// How many kinds of calls have something that runs in place of the
    /// nodes.
    pub fn compiled(&self) -> usize {
        self.exported.borrow().iter().filter(|exported| exported.callable.is_some()).count()
    }

    /// How many kinds of calls it was recorded for and is not exported for.
    pub fn closed(&self) -> usize {
        self.exported.borrow().iter().filter(|exported| exported.callable.is_none()).count()
    }
}

pub enum GraphState {
    /// The VM runs the instructions.
    Built,
    /// As `Built`, and the next call that no exported form is for is exported.
    Waiting,
    /// As `Built`: exporting it failed, for this reason.
    GivenUp(String),
}

/// Instructions of a block that are in no graph: the generic instructions
/// lo..hi, and why.
#[derive(Clone)]
pub struct Left {
    pub lo: u32,
    pub hi: u32,
    pub reason: String,
}

/// How many calls in a row may find that an instruction in front of the
/// callable or behind it could run Python code before the graph is left to
/// the VM.
const MAX_UNSURE: u32 = 3;

/// How many exported forms that differ in nothing but numbers, and of
/// which none ever ran again, say that the numbers are new on every call.
/// Three: the layers of a model that call one function with two or three
/// settings each keep their forms.
const EVER_NEW: usize = 3;

pub const EVER_NEW_REASON: &str = "its inputs take ever new values";

/// How many calls in a row with the same numbers reopen a kind of call that
/// was closed for numbers that are new every time.
pub const SETTLED: u32 = 3;

/// A graph of one version of a block.
pub struct GraphSite {
    /// The function, the block, and the generic instructions from the first
    /// node to the last.
    pub name: String,
    /// One path: that of the version it was built from.
    pub variants: Shared<Vec<Arc<Path>>>,
    pub state: Shared<GraphState>,
    /// The version of the block.
    pub version: AtomicU32,
    /// What makes an exported form of a description (`torch.vm._export`),
    /// if the list of passes has a pass over graphs, and how many forms a
    /// path may have.
    pub exporter: Option<PyObj>,
    pub max_variants: usize,
    unsure: AtomicU32,
    /// How many times it was handed to the exporter, and how many calls
    /// came to it and ran its instructions.
    pub exports: AtomicU64,
    pub misses: AtomicU64,
}

impl GraphSite {
    pub fn install(&self, path: Path, version: u32) {
        self.variants.borrow_mut().push(Arc::new(path));
        self.version.store(version, Ordering::Relaxed);
    }

    pub fn path(&self) -> Option<Arc<Path>> {
        self.variants.borrow().first().cloned()
    }

    pub fn describe(&self) -> String {
        match self.path() {
            Some(path) => format!("{}: {} operations", self.name, path.operators.len()),
            None => self.name.clone(),
        }
    }

    /// Whether a call that no exported form is for may be recorded for one:
    /// while there are fewer than `max_variants` forms with a callable, and
    /// no more than that without one.
    pub fn records(&self, path: &Path) -> bool {
        matches!(*self.state.borrow(), GraphState::Waiting)
            && path.compiled() < self.max_variants
            && path.closed() <= self.max_variants
    }

    /// Whether `Op::Graph` has anything to do for it.
    pub fn runs(&self, path: &Path) -> bool {
        matches!(*self.state.borrow(), GraphState::Waiting) || !path.exported.borrow().is_empty()
    }

    pub fn give_up(&self, reason: String) {
        *self.state.borrow_mut() = GraphState::GivenUp(reason);
    }

    /// An instruction that would run out of its place for an exported form
    /// could not be shown to run no Python code. Where that goes on, the
    /// forms are dropped and the graph is left to the VM.
    pub fn unsure(&self, path: &Path, what: &str) {
        if self.unsure.fetch_add(1, Ordering::Relaxed) + 1 >= MAX_UNSURE {
            path.exported.borrow_mut().clear();
            self.give_up(format!("{} could run Python code", what));
        }
    }

    pub fn sure(&self) {
        self.unsure.store(0, Ordering::Relaxed);
    }

    /// Whether a run that is about to be exported is one more of a kind of
    /// call whose numbers are new every time: `EVER_NEW` forms differ from
    /// it in nothing but numbers, and none of them ever ran again. Those
    /// forms are then dropped, and the checks that had the numbers are
    /// returned: the form made of the others has every call of the kind
    /// run the instructions.
    pub unsafe fn ever_new(
        &self,
        path: &Path,
        world: (u64, u64, u64, u64),
        checks: &[Check],
        sizes: &[i64],
    ) -> Option<Vec<usize>> {
        let mut forms = path.exported.borrow_mut();
        let alike: Vec<(usize, Vec<usize>)> = forms
            .iter()
            .enumerate()
            .filter(|(_, form)| form.callable.is_some())
            .filter_map(|(k, form)| Some((k, form.other_numbers(world, checks, sizes)?)))
            .filter(|(_, numbers)| !numbers.is_empty())
            .collect();
        let ran = alike.iter().any(|(k, _)| forms[*k].hits.load(Ordering::Relaxed) > 0);
        if alike.len() < EVER_NEW || ran {
            return None;
        }
        let mut varying: Vec<usize> = alike.iter().flat_map(|(_, numbers)| numbers.iter().copied()).collect();
        varying.sort_unstable();
        varying.dedup();
        let mut at = 0;
        forms.retain(|_| {
            at += 1;
            !alike.iter().any(|(k, _)| *k == at - 1)
        });
        Some(varying)
    }

    /// Hands a description to the exporter. It returns the callable and the
    /// guards of the passes, or the reason there is none as a str. An error
    /// that is no Exception is left set.
    pub unsafe fn export(&self, description: Obj) -> Result<(PyObj, Vec<(usize, PyObj)>), String> {
        let Some(exporter) = &self.exporter else {
            return Err("nothing exports graphs".to_string());
        };
        let made = ffi::PyObject_CallOneArg(exporter.as_ptr(), description);
        if made.is_null() {
            // What is no Exception (an interrupt) stays raised.
            if ffi::PyErr_ExceptionMatches(ffi::PyExc_Exception) == 0 {
                return Err(String::new());
            }
            return Err(format!("the export failed: {}", raised_text()));
        }
        let made = PyObj::from_owned(made);
        if ffi::PyUnicode_Check(made.as_ptr()) != 0 {
            return Err(pyffi::str_to_string(made.as_ptr()));
        }
        let malformed = || "the exporter returns (callable, guards) or a reason".to_string();
        if ffi::PyTuple_Check(made.as_ptr()) == 0 || ffi::PyTuple_GET_SIZE(made.as_ptr()) != 2 {
            return Err(malformed());
        }
        let callable = ffi::PyTuple_GET_ITEM(made.as_ptr(), 0);
        let listed = ffi::PyTuple_GET_ITEM(made.as_ptr(), 1);
        if ffi::PyList_Check(listed) == 0 {
            return Err(malformed());
        }
        let mut guards = Vec::new();
        for i in 0..ffi::PyList_GET_SIZE(listed) {
            let pair = ffi::PyList_GET_ITEM(listed, i);
            if ffi::PyTuple_Check(pair) == 0 || ffi::PyTuple_GET_SIZE(pair) != 2 {
                return Err(malformed());
            }
            let position = ffi::PyLong_AsSsize_t(ffi::PyTuple_GET_ITEM(pair, 0));
            if position < 0 {
                ffi::PyErr_Clear();
                return Err(malformed());
            }
            guards.push((position as usize, PyObj::from_borrowed(ffi::PyTuple_GET_ITEM(pair, 1))));
        }
        Ok((PyObj::from_borrowed(callable), guards))
    }
}

/// Operators that write to arguments without their schema saying so and
/// without the version counters of the tensors moving: a normalization
/// updates the running statistics it is given when the argument named here
/// is true.
const UPDATES_STATISTICS: [(&str, &str); 5] = [
    ("aten::batch_norm", "training"),
    ("aten::native_batch_norm", "training"),
    ("aten::cudnn_batch_norm", "training"),
    ("aten::_batch_norm_impl_index", "training"),
    ("aten::instance_norm", "use_input_stats"),
];

/// Whether the call of a site, as `site_export` describes it and with the
/// operands it had, is one of `UPDATES_STATISTICS` that updates statistics.
unsafe fn updates_statistics(exported: Obj, operands: &[Obj]) -> bool {
    let name = ffi::PyDict_GetItemString(exported, c"op".as_ptr());
    let args = ffi::PyDict_GetItemString(exported, c"args".as_ptr());
    if name.is_null() || args.is_null() {
        return false;
    }
    let name = pyffi::str_to_string(name);
    let Some((_, flag)) = UPDATES_STATISTICS.iter().find(|(op, _)| *op == name) else {
        return false;
    };
    // The value a schema argument had: a constant of the site, or the
    // operand it comes from. Null where that cannot be told.
    let value = |wanted: &str| -> Obj {
        for i in 0..ffi::PyList_GET_SIZE(args) {
            let arg = ffi::PyList_GET_ITEM(args, i);
            if pyffi::str_to_string(ffi::PyTuple_GET_ITEM(arg, 0)) != wanted {
                continue;
            }
            let payload = ffi::PyTuple_GET_ITEM(arg, 3);
            return match pyffi::str_to_string(ffi::PyTuple_GET_ITEM(arg, 2)).as_str() {
                "const" => payload,
                "operand" => {
                    let k = ffi::PyLong_AsSsize_t(ffi::PyTuple_GET_ITEM(payload, 0));
                    operands.get(k as usize).copied().unwrap_or(null_mut())
                }
                _ => null_mut(),
            };
        }
        null_mut()
    };
    let (flag, statistics) = (value(flag), value("running_mean"));
    let off = !flag.is_null() && ffi::PyObject_IsTrue(flag) == 0;
    !(off || statistics == ffi::Py_None())
}

/// The error that is set, as one line of text; clears it.
pub unsafe fn raised_text() -> String {
    let raised = ffi::PyErr_GetRaisedException();
    if raised.is_null() {
        return "unknown error".to_string();
    }
    let text = ffi::PyObject_Str(raised);
    let message = if text.is_null() {
        ffi::PyErr_Clear();
        String::new()
    } else {
        let message = pyffi::str_to_string(text);
        ffi::Py_DECREF(text);
        message
    };
    let kind = pyffi::type_name(raised);
    ffi::Py_DECREF(raised);
    let line = message.lines().next().unwrap_or("");
    format!("{}: {}", kind, line.chars().take(200).collect::<String>())
}

/// Whether two values are the same as far as a kernel that is given them
/// can tell: the same exact type, and equal numbers, texts, slices, dtypes
/// and devices, or tuples or lists of such. 0.0 is not -0.0. Runs no Python
/// code. A list, or what holds one, is the same as another by what is in
/// it now, not by being that object: it can be changed in place (see
/// `frozen`).
pub unsafe fn same_value(a: Obj, b: Obj) -> bool {
    same_within(a, b, 0)
}

/// `same_value`, this many lists, tuples and slices deep in both values.
/// Two values that go on further than `NESTED` are not the same: a list
/// that holds itself is the same as nothing, itself included.
unsafe fn same_within(a: Obj, b: Obj, depth: usize) -> bool {
    if ffi::Py_TYPE(a) != ffi::Py_TYPE(b) || depth > NESTED {
        return false;
    }
    if ffi::PyFloat_CheckExact(a) != 0 {
        return ffi::PyFloat_AS_DOUBLE(a).to_bits() == ffi::PyFloat_AS_DOUBLE(b).to_bits();
    }
    if a == b && ffi::PyTuple_CheckExact(a) == 0 && ffi::PyList_CheckExact(a) == 0 && ffi::PySlice_Check(a) == 0 {
        return true;
    }
    if ffi::PyLong_CheckExact(a) != 0 {
        return ffi::PyObject_RichCompareBool(a, b, ffi::Py_EQ) == 1;
    }
    if ffi::PyUnicode_CheckExact(a) != 0 {
        return ffi::PyUnicode_Compare(a, b) == 0;
    }
    if ffi::PyComplex_CheckExact(a) != 0 {
        let bits = |o: Obj| (ffi::PyComplex_RealAsDouble(o).to_bits(), ffi::PyComplex_ImagAsDouble(o).to_bits());
        return bits(a) == bits(b);
    }
    if ffi::PySlice_Check(a) != 0 {
        return [c"start", c"stop", c"step"].iter().all(|name| {
            let (x, y) = (ffi::PyObject_GetAttrString(a, name.as_ptr()), ffi::PyObject_GetAttrString(b, name.as_ptr()));
            let same = !x.is_null() && !y.is_null() && same_within(x, y, depth + 1);
            xdecref(x);
            xdecref(y);
            same
        });
    }
    if is_token(a) {
        return ffi::PyObject_RichCompareBool(a, b, ffi::Py_EQ) == 1;
    }
    let sequence = ffi::PyTuple_CheckExact(a) != 0 || ffi::PyList_CheckExact(a) != 0;
    if !sequence || ffi::PyObject_Size(a) != ffi::PyObject_Size(b) {
        return false;
    }
    (0..ffi::PyObject_Size(a)).all(|i| same_within(sequence_item(a, i), sequence_item(b, i), depth + 1))
}

/// How many lists, tuples and slices inside one another `same_value` goes
/// into.
const NESTED: usize = 64;

/// Whether a list, a tuple or a slice holds itself, directly or through
/// the lists, tuples and slices it holds: going into what it holds, and
/// into what that holds, does not end. Each of them is gone into once
/// (those the walk is inside of, and those it is done with, are kept),
/// so this ends for any value. Runs no Python code.
pub unsafe fn holds_itself(o: Obj) -> bool {
    let holds = |o: Obj| !o.is_null() && !capi::is_unboxed(o) && (ffi::PyList_Check(o) != 0 || ffi::PyTuple_Check(o) != 0 || ffi::PySlice_Check(o) != 0);
    let count = |o: Obj| {
        if ffi::PyList_Check(o) != 0 {
            ffi::PyList_GET_SIZE(o)
        } else if ffi::PyTuple_Check(o) != 0 {
            ffi::PyTuple_GET_SIZE(o)
        } else {
            3
        }
    };
    let item = |o: Obj, i: isize| {
        if ffi::PyList_Check(o) != 0 {
            ffi::PyList_GET_ITEM(o, i)
        } else if ffi::PyTuple_Check(o) != 0 {
            ffi::PyTuple_GET_ITEM(o, i)
        } else {
            let slice = o as *mut ffi::PySliceObject;
            [(*slice).start, (*slice).stop, (*slice).step][i as usize]
        }
    };
    if !holds(o) {
        return false;
    }
    // Depth first: what the walk is inside of, each with the item it has
    // got to, and what it has been through.
    let mut inside: Vec<(Obj, isize)> = vec![(o, 0)];
    let mut open: std::collections::HashSet<usize> = std::collections::HashSet::from([o as usize]);
    let mut done: std::collections::HashSet<usize> = std::collections::HashSet::new();
    while let Some((held, at)) = inside.last_mut() {
        let held = *held;
        if *at >= count(held) {
            open.remove(&(held as usize));
            done.insert(held as usize);
            inside.pop();
            continue;
        }
        let part = item(held, *at);
        *at += 1;
        if !holds(part) || done.contains(&(part as usize)) {
            continue;
        }
        if !open.insert(part as usize) {
            return true;
        }
        inside.push((part, 0));
    }
    false
}

/// What a check keeps of a value to tell the value of a later call from
/// it (`same_value`), and what a form has as a constant: the value itself
/// or, for what can be changed in place (a list, and a tuple or a slice
/// that holds one), a copy that nothing else has. A list that lives
/// outside the function is the same object at every call whatever is in
/// it.
pub unsafe fn frozen(o: Obj) -> PyObj {
    // What holds itself has no copy that ends, and is nothing a check is
    // made of (`checkable`).
    if capi::is_unboxed(o) || holds_itself(o) {
        return PyObj::from_borrowed(o);
    }
    frozen_row(o)
}

unsafe fn frozen_row(o: Obj) -> PyObj {
    if ffi::PySlice_Check(o) != 0 {
        let parts: Vec<Obj> = [c"start", c"stop", c"step"].iter().map(|name| ffi::PyObject_GetAttrString(o, name.as_ptr())).collect();
        let kept: Vec<PyObj> = parts.iter().map(|part| if part.is_null() { PyObj::from_borrowed(ffi::Py_None()) } else { frozen_row(*part) }).collect();
        let same = parts.iter().zip(&kept).all(|(part, kept)| *part == kept.as_ptr());
        parts.into_iter().for_each(|part| xdecref(part));
        ffi::PyErr_Clear();
        if same {
            return PyObj::from_borrowed(o);
        }
        return PyObj::from_owned(ffi::PySlice_New(kept[0].as_ptr(), kept[1].as_ptr(), kept[2].as_ptr()));
    }
    if ffi::PyTuple_CheckExact(o) == 0 && ffi::PyList_CheckExact(o) == 0 {
        return PyObj::from_borrowed(o);
    }
    let count = ffi::PyObject_Size(o);
    let items: Vec<PyObj> = (0..count).map(|i| frozen_row(sequence_item(o, i))).collect();
    let list = ffi::PyList_CheckExact(o) != 0;
    if !list && (0..count).all(|i| items[i as usize].as_ptr() == sequence_item(o, i)) {
        return PyObj::from_borrowed(o);
    }
    let copy = if list { ffi::PyList_New(count) } else { ffi::PyTuple_New(count) };
    for (i, item) in items.iter().enumerate() {
        if list {
            ffi::PyList_SET_ITEM(copy, i as isize, new_ref(item.as_ptr()));
        } else {
            ffi::PyTuple_SET_ITEM(copy, i as isize, new_ref(item.as_ptr()));
        }
    }
    PyObj::from_owned(copy)
}

unsafe fn sequence_item(s: Obj, i: isize) -> Obj {
    if ffi::PyTuple_CheckExact(s) != 0 {
        ffi::PyTuple_GET_ITEM(s, i)
    } else {
        ffi::PyList_GET_ITEM(s, i)
    }
}

/// A dtype, a device, a layout or a memory format: objects of types that
/// torch defines in C, which compare by what they stand for.
unsafe fn is_token(o: Obj) -> bool {
    let name = std::ffi::CStr::from_ptr((*ffi::Py_TYPE(o)).tp_name).to_bytes();
    [&b"torch.dtype"[..], b"torch.device", b"torch.layout", b"torch.memory_format"].contains(&name)
}

/// Whether `same_value` can tell a later value from this one: what an
/// exported form may have as a constant under a check.
pub unsafe fn checkable(o: Obj) -> bool {
    // Not a list or a tuple that holds itself: it has no copy to keep
    // (`frozen`), and the walk into it would not end.
    !holds_itself(o) && plainly_checkable(o)
}

unsafe fn plainly_checkable(o: Obj) -> bool {
    if capi::is_unboxed(o) {
        return false;
    }
    if is_number(o)
        || o == ffi::Py_None()
        || o == ffi::Py_Ellipsis()
        || ffi::PyUnicode_CheckExact(o) != 0
        || ffi::PyComplex_CheckExact(o) != 0
        || is_token(o)
    {
        return true;
    }
    if ffi::PySlice_Check(o) != 0 {
        return [c"start", c"stop", c"step"].iter().all(|name| {
            let part = ffi::PyObject_GetAttrString(o, name.as_ptr());
            let fine = !part.is_null() && plainly_checkable(part);
            xdecref(part);
            fine
        });
    }
    let sequence = ffi::PyTuple_CheckExact(o) != 0 || ffi::PyList_CheckExact(o) != 0;
    sequence && (0..ffi::PyObject_Size(o)).all(|i| plainly_checkable(sequence_item(o, i)))
}

/// An exact int, float or bool.
pub unsafe fn is_number(o: Obj) -> bool {
    !capi::is_unboxed(o)
        && (ffi::PyLong_CheckExact(o) != 0 || ffi::PyFloat_CheckExact(o) != 0 || ffi::PyBool_Check(o) != 0)
}

/// A number, a text or None: what compares, hashes and counts in C code.
unsafe fn is_plain(o: Obj) -> bool {
    is_number(o) || o == ffi::Py_None() || ffi::PyUnicode_CheckExact(o) != 0
}

/// What indexes a list, a tuple or a dict in C code: a plain value, a slice
/// of plain values, or a tuple of them.
unsafe fn is_plain_key(o: Obj) -> bool {
    if capi::is_unboxed(o) {
        return false;
    }
    if ffi::PyTuple_CheckExact(o) != 0 {
        return (0..ffi::PyTuple_GET_SIZE(o)).all(|i| is_plain(ffi::PyTuple_GET_ITEM(o, i)));
    }
    is_plain(o) || (ffi::PySlice_Check(o) != 0 && checkable(o))
}

unsafe fn is_static(ty: *mut ffi::PyTypeObject) -> bool {
    ffi::PyType_HasFeature(ty, ffi::Py_TPFLAGS_HEAPTYPE) == 0
}

/// A dict of a type that is defined in C and has no `__missing__` (an
/// `OrderedDict`; not a `defaultdict`, which makes what it lacks).
unsafe fn is_plain_dict(o: Obj) -> bool {
    static MISSING: std::sync::OnceLock<PyObj> = std::sync::OnceLock::new();
    if ffi::PyDict_CheckExact(o) != 0 {
        return true;
    }
    let ty = ffi::Py_TYPE(o);
    let missing = MISSING.get_or_init(|| PyObj::from_owned(ffi::PyUnicode_InternFromString(c"__missing__".as_ptr())));
    ffi::PyDict_Check(o) != 0 && is_static(ty) && pyffi::_PyType_Lookup(ty, missing.as_ptr()).is_null()
}

/// A list, a dict or a str, or a tuple of a type that is defined in C (a
/// `torch.Size`): what is indexed, counted and unpacked by C code.
unsafe fn is_plain_container(o: Obj) -> bool {
    !capi::is_unboxed(o)
        && (ffi::PyList_CheckExact(o) != 0
            || is_plain_dict(o)
            || ffi::PyUnicode_CheckExact(o) != 0
            || (ffi::PyTuple_Check(o) != 0 && is_static(ffi::Py_TYPE(o))))
}

/// Whether a plain key is looked up in a dict, or put into it, without a
/// key of the dict being compared with it in Python code (a key of a class
/// with an `__eq__`, whose hash is that of the plain one): every key the
/// dict has is plain. A dict that has taken nothing but texts says so
/// itself (`pyffi::text_keys`); any other is gone through, if it is small.
pub unsafe fn quiet_keys(dict: Obj) -> bool {
    const SCANNED: isize = 32;
    if pyffi::text_keys(dict) {
        return true;
    }
    if ffi::PyDict_Size(dict) > SCANNED {
        return false;
    }
    let (mut at, mut key, mut value) = (0, null_mut(), null_mut());
    while ffi::PyDict_Next(dict, &mut at, &mut key, &mut value) != 0 {
        if !is_plain_key(key) {
            return false;
        }
    }
    true
}

/// A weak reference as the interpreter lays it out: only looked into on the
/// versions whose layout this is.
#[cfg(not(Py_3_13))]
#[repr(C)]
struct WeakRef {
    ob_base: ffi::PyObject,
    wr_object: Obj,
    wr_callback: Obj,
    hash: ffi::Py_hash_t,
    wr_prev: *mut WeakRef,
    wr_next: *mut WeakRef,
}

/// The function by which a weak container takes an entry out when what
/// a weak reference of it refers to ends, known by what it is: its code
/// (the code object that the class's `__init__`, or `__setitem__`, has
/// among its constants), the class whose objects make it, the classes of
/// the weak references such a container makes, and, where the function
/// calls a method of the container, the code of that method.
#[cfg(not(Py_3_13))]
#[derive(Clone)]
struct Remover {
    code: usize,
    /// What the container keeps by the reference: 0 a dictionary whose
    /// keys are the references, 1 one whose values are, 2 a set, 3 a
    /// dictionary that keeps its values under the identities of its keys.
    kind: u8,
    class: usize,
    references: Vec<usize>,
    method: usize,
}

/// The weak containers whose own callbacks are let pass, found once by
/// looking, without running anything: `weakref.WeakKeyDictionary`,
/// `weakref.WeakValueDictionary`, `weakref.WeakSet`, torch's
/// `WeakIdKeyDictionary` (a tracer's memo of the tensors it has seen) and
/// `ExactWeakKeyDictionary` of `torch._dynamo.utils` (in which every module
/// is noted that is made after `torch.compile` compiled a function). The
/// names only say where to look. What is found is taken only if it is
/// what is expected: a function among the constants of the method that
/// uses no names but those of the container's own attributes. A class
/// whose module was not imported is looked for again later.
#[cfg(not(Py_3_13))]
unsafe fn weak_containers() -> Vec<Remover> {
    type Class = (&'static std::ffi::CStr, &'static std::ffi::CStr);
    const WEAKREF: Class = (c"weakref", c"ref");
    const OWN: [&str; 7] = ["_iterating", "_pending_removals", "append", "data", "KeyError", "key", "discard"];
    const CLASSES: [(Class, u8, &[Class]); 5] = [
        ((c"torch.utils.weak", c"WeakIdKeyDictionary"), 0, &[(c"torch.utils.weak", c"WeakIdRef"), (c"torch.utils.weak", c"_WeakHashRef")]),
        ((c"weakref", c"WeakKeyDictionary"), 0, &[WEAKREF]),
        ((c"weakref", c"WeakValueDictionary"), 1, &[(c"weakref", c"KeyedRef")]),
        ((c"_weakrefset", c"WeakSet"), 2, &[WEAKREF]),
        ((c"torch._dynamo.utils", c"ExactWeakKeyDictionary"), 3, &[WEAKREF]),
    ];
    static FOUND: std::sync::OnceLock<Vec<Remover>> = std::sync::OnceLock::new();
    if let Some(found) = FOUND.get() {
        return found.clone();
    }
    let class_of = |(module, class): Class| {
        let module = ffi::PyDict_GetItemString(ffi::PyImport_GetModuleDict(), module.as_ptr());
        let found = if module.is_null() || ffi::PyModule_Check(module) == 0 {
            null_mut()
        } else {
            ffi::PyDict_GetItemString(ffi::PyModule_GetDict(module), class.as_ptr())
        };
        if found.is_null() || ffi::PyType_Check(found) == 0 { null_mut() } else { found }
    };
    // The names a code object uses, and whether it has code of its own
    // among its constants.
    let names_of = |code: Obj| {
        let names = ffi::PyObject_GetAttrString(code, c"co_names".as_ptr());
        let mut found = Vec::new();
        if !names.is_null() && ffi::PyTuple_Check(names) != 0 {
            found.extend((0..ffi::PyTuple_GET_SIZE(names)).map(|i| pyffi::str_to_string(ffi::PyTuple_GET_ITEM(names, i))));
        }
        xdecref(names);
        found
    };
    let consts_of = |code: Obj| {
        let consts = ffi::PyObject_GetAttrString(code, c"co_consts".as_ptr());
        let mut found = Vec::new();
        if !consts.is_null() && ffi::PyTuple_Check(consts) != 0 {
            // The code keeps its constants for as long as it lives.
            found.extend((0..ffi::PyTuple_GET_SIZE(consts)).map(|i| ffi::PyTuple_GET_ITEM(consts, i)).filter(|held| ffi::PyCode_Check(*held) != 0));
        }
        xdecref(consts);
        found
    };
    let method_of = |class: Obj, name: &std::ffi::CStr| {
        let name = PyObj::from_owned(ffi::PyUnicode_InternFromString(name.as_ptr()));
        let found = pyffi::_PyType_Lookup(class as *mut ffi::PyTypeObject, name.as_ptr());
        if found.is_null() || ffi::PyFunction_Check(found) == 0 { null_mut() } else { ffi::PyFunction_GetCode(found) }
    };
    let (mut found, mut all) = (Vec::new(), true);
    for (named, kind, made) in CLASSES {
        let class = class_of(named);
        let references: Vec<usize> = made.iter().map(|named| class_of(*named) as usize).collect();
        let within = if class.is_null() { null_mut() } else { method_of(class, if kind == 3 { c"__setitem__" } else { c"__init__" }) };
        if within.is_null() || references.contains(&0) {
            all = false;
            continue;
        }
        let inner = consts_of(within);
        // The method that the function of the last kind calls: it may
        // name nothing but what the container keeps its entries in.
        let method = if kind == 3 { method_of(class, c"_remove_id") } else { null_mut() };
        let plain = kind != 3 || (!method.is_null() && consts_of(method).is_empty() && names_of(method).iter().all(|name| name == "values" || name == "refs"));
        let fits = |code: Obj| {
            let names = names_of(code);
            let own = if kind == 3 { names == ["_remove_id"] } else { names.iter().all(|name| OWN.contains(&name.as_str())) };
            own && consts_of(code).is_empty()
        };
        if let (true, [code], true) = (plain, inner.as_slice(), inner.iter().all(|code| fits(*code))) {
            // Kept for good: they are compared by where they are.
            ffi::Py_INCREF(*code);
            ffi::Py_XINCREF(method);
            found.push(Remover { code: *code as usize, kind, class: class as usize, references, method: method as usize });
        }
    }
    ffi::PyErr_Clear();
    if all {
        let _ = FOUND.set(found.clone());
    }
    found
}

/// Whether an object is hashed without Python code running: a number, a
/// text, None, bytes; an object whose class has the hash of `object` (the
/// slot is `object`'s: a class with a `__hash__` of its own has another);
/// an exact tuple of such.
#[cfg(not(Py_3_13))]
unsafe fn hashed_in_c(o: Obj, depth: usize) -> bool {
    if o.is_null() || capi::is_unboxed(o) {
        return false;
    }
    if is_plain(o) || ffi::PyBytes_CheckExact(o) != 0 {
        return true;
    }
    if ffi::PyTuple_CheckExact(o) != 0 {
        return depth < 4 && (0..ffi::PyTuple_GET_SIZE(o)).all(|i| hashed_in_c(ffi::PyTuple_GET_ITEM(o, i), depth + 1));
    }
    let object = std::ptr::addr_of!(ffi::PyBaseObject_Type);
    (*ffi::Py_TYPE(o)).tp_hash.map(|own| own as usize) == (*object).tp_hash.map(|own| own as usize)
}

/// Whether a key is one of the keys of a dict, by identity, and the dict
/// keeps no other key under the same hash: looking the key up, or taking
/// its entry out, compares it with no other key.
#[cfg(not(Py_3_13))]
unsafe fn alone_under_its_hash(dict: Obj, key: Obj) -> bool {
    let (mut at, mut other, mut value, mut hash) = (0, null_mut(), null_mut(), 0);
    let mut hashes: Vec<ffi::Py_hash_t> = Vec::new();
    let mut own = None;
    while pyffi::_PyDict_Next(dict, &mut at, &mut other, &mut value, &mut hash) != 0 {
        if other == key {
            own = Some(hash);
        } else {
            hashes.push(hash);
        }
    }
    own.is_some_and(|own| !hashes.contains(&own))
}

/// For the function by which a dictionary that keeps its values under the
/// identities of its keys takes an entry out: the dictionary and the
/// identity, which are the variables `self` and `idx` of the method the
/// function was made in. Borrowed.
#[cfg(not(Py_3_13))]
unsafe fn enclosed(callback: Obj) -> Option<(Obj, Obj)> {
    let closure = ffi::PyFunction_GetClosure(callback);
    let names = ffi::PyObject_GetAttrString(ffi::PyFunction_GetCode(callback), c"co_freevars".as_ptr());
    if names.is_null() {
        ffi::PyErr_Clear();
        return None;
    }
    let mut found = (null_mut(), null_mut());
    let fits = !closure.is_null() && ffi::PyTuple_Check(names) != 0 && ffi::PyTuple_GET_SIZE(names) == ffi::PyTuple_GET_SIZE(closure);
    if fits {
        for i in 0..ffi::PyTuple_GET_SIZE(names) {
            let held = pyffi::PyCell_Get(ffi::PyTuple_GET_ITEM(closure, i));
            match pyffi::str_to_string(ffi::PyTuple_GET_ITEM(names, i)).as_str() {
                "self" => found.0 = held,
                "idx" => found.1 = held,
                _ => {}
            }
            // The cells keep what they hold.
            xdecref(held);
        }
    }
    xdecref(names);
    (!found.0.is_null() && !found.1.is_null()).then_some(found)
}

/// What ends with an object besides itself because of the weak references
/// to it, added to `more`; true where a weak reference has something to
/// call when the object ends that may run code of the user's.
///
/// A callback is let pass only if all of this holds, each by identity or
/// by exact type, nothing by a name: it is a function whose code is the
/// very code object of a known weak container's own function
/// (`weak_containers`); the weak reference is of exactly a class such a
/// container makes its references of; the container that the function
/// was made for is of exactly the class (no subclass, whose attributes
/// and methods could be anything); and what the function reaches of the
/// container is what the class put there: the dict or set of the
/// entries, the set of who goes over it, the list of what is to be
/// taken out later, each of the interpreter's own exact type, so that
/// taking the entry out compares and removes in C. For the dictionary
/// by identity, whose function calls a method: the container has no
/// attribute of its own under the method's name, and the class has the
/// method it had when it was found. What goes with the entry that the
/// container takes out (the value under the reference, the key of a
/// value) is added to `more` and looked at like what the object refers
/// to.
#[cfg(not(Py_3_13))]
unsafe fn weakly_watched(o: Obj, more: &mut Vec<PyObj>) -> bool {
    let offset = (*ffi::Py_TYPE(o)).tp_weaklistoffset;
    if offset == 0 {
        return false;
    }
    let mut known: Option<Vec<Remover>> = None;
    let mut at = *((o as *mut u8).offset(offset) as *mut *mut WeakRef);
    while !at.is_null() {
        let callback = (*at).wr_callback;
        let reference = at as Obj;
        at = (*at).wr_next;
        if callback.is_null() {
            continue;
        }
        if ffi::PyFunction_Check(callback) == 0 {
            return true;
        }
        let code = ffi::PyFunction_GetCode(callback) as usize;
        let known = known.get_or_insert_with(|| weak_containers());
        let Some(remover) = known.iter().find(|other| other.code == code) else {
            return true;
        };
        if !remover.references.contains(&(ffi::Py_TYPE(reference) as usize)) {
            return true;
        }
        // The container: a variable of the method that made the function,
        // or behind the weak reference to it that the function has as a
        // default.
        let (container, identity) = if remover.kind == 3 {
            let Some(found) = enclosed(callback) else {
                return true;
            };
            found
        } else {
            let defaults = ffi::PyFunction_GetDefaults(callback);
            let to_it = if defaults.is_null() || ffi::PyTuple_GET_SIZE(defaults) == 0 { null_mut() } else { ffi::PyTuple_GET_ITEM(defaults, 0) };
            if to_it.is_null() || ffi::PyWeakref_CheckRefExact(to_it) == 0 {
                return true;
            }
            (ffi::PyWeakref_GetObject(to_it), null_mut())
        };
        if container.is_null() || container == ffi::Py_None() {
            // The container is gone: the function finds that and returns.
            continue;
        }
        if ffi::Py_TYPE(container) as usize != remover.class {
            return true;
        }
        let slot = pyffi::_PyObject_GetDictPtr(container);
        if slot.is_null() || (*slot).is_null() {
            return true;
        }
        let own = |name: &std::ffi::CStr| ffi::PyDict_GetItemString(*slot, name.as_ptr());
        let dict = |held: Obj| !held.is_null() && ffi::PyDict_CheckExact(held) != 0;
        let set = |held: Obj| !held.is_null() && ffi::PySet_CheckExact(held) != 0;
        let list = |held: Obj| !held.is_null() && ffi::PyList_CheckExact(held) != 0;
        if remover.kind == 3 {
            let method = PyObj::from_owned(ffi::PyUnicode_InternFromString(c"_remove_id".as_ptr()));
            let found = pyffi::_PyType_Lookup(ffi::Py_TYPE(container), method.as_ptr());
            let same = !found.is_null() && ffi::PyFunction_Check(found) != 0 && ffi::PyFunction_GetCode(found) as usize == remover.method;
            let (values, references) = (own(c"values"), own(c"refs"));
            if !same || !own(c"_remove_id").is_null() || !dict(values) || !dict(references) || ffi::PyLong_CheckExact(identity) == 0 {
                return true;
            }
            let value = ffi::PyDict_GetItemWithError(values, identity);
            if value.is_null() {
                ffi::PyErr_Clear();
            } else {
                more.push(PyObj::from_borrowed(value));
            }
            continue;
        }
        let (data, waiting) = (own(c"data"), own(c"_pending_removals"));
        if !set(own(c"_iterating")) || !(list(waiting) || set(waiting)) {
            return true;
        }
        match remover.kind {
            0 => {
                if !dict(data) || ffi::PyDict_Size(data) > 65536 {
                    return true;
                }
                let (mut place, mut key, mut value) = (0, null_mut(), null_mut());
                while ffi::PyDict_Next(data, &mut place, &mut key, &mut value) != 0 {
                    if key == reference {
                        more.push(PyObj::from_borrowed(value));
                    }
                }
            }
            1 => {
                if !dict(data) || ffi::PyDict_Size(data) > 65536 {
                    return true;
                }
                let key = ffi::PyObject_GetAttrString(reference, c"key".as_ptr());
                if key.is_null() {
                    ffi::PyErr_Clear();
                    return true;
                }
                let key = PyObj::from_owned(key);
                // The entry is taken out by its key, which is hashed for
                // that and compared with every key the dict keeps under
                // the same hash: in C, or the user's code runs.
                if !hashed_in_c(key.as_ptr(), 0) || !alone_under_its_hash(data, key.as_ptr()) {
                    return true;
                }
                // What goes with the entry is every reference it holds:
                // the key, and the weak reference, which holds the key
                // once more (`KeyedRef.key`) and ends with the entry where
                // nothing else holds it.
                more.push(key);
                more.push(PyObj::from_borrowed(reference));
            }
            _ => {
                if !set(data) {
                    return true;
                }
            }
        }
    }
    false
}

/// Whether there is a weak reference to the object: one may have
/// something to call when the object dies.
#[cfg(Py_3_13)]
unsafe fn weakly_watched(o: Obj, _more: &mut Vec<PyObj>) -> bool {
    let offset = (*ffi::Py_TYPE(o)).tp_weaklistoffset;
    offset != 0 && !(*((o as *mut u8).offset(offset) as *mut Obj)).is_null()
}

unsafe extern "C" fn every_referent(o: Obj, arg: *mut std::ffi::c_void) -> std::ffi::c_int {
    let seen = &mut *(arg as *mut Vec<Obj>);
    if !o.is_null() {
        seen.push(o);
    }
    0
}

/// Whether an object is a tensor whose end may let go of what cannot be
/// looked at from here: one that requires a gradient. Such a tensor has a
/// history, or is a leaf that gradients are accumulated in, and what
/// that holds (the nodes of the history with the hooks on them and the
/// Python objects that functions written in Python saved, the hooks of
/// the tensor) is held from C++ and ends with the tensor where nothing
/// else holds it.
pub unsafe fn holds_unseen(o: Obj) -> bool {
    let mut info = TcfgTensorInfo::empty();
    capi::api().is_some_and(|api| (api.tensor_info)(o, &mut info) != 0 && info.requires_grad != 0)
}

/// A class by where it is, found once: `module.name`. Null where the
/// module was not imported or has no such class.
unsafe fn class_named(module: &std::ffi::CStr, name: &std::ffi::CStr) -> Obj {
    let module = ffi::PyDict_GetItemString(ffi::PyImport_GetModuleDict(), module.as_ptr());
    let found = if module.is_null() || ffi::PyModule_Check(module) == 0 {
        null_mut()
    } else {
        ffi::PyDict_GetItemString(ffi::PyModule_GetDict(module), name.as_ptr())
    };
    if found.is_null() || ffi::PyType_Check(found) == 0 { null_mut() } else { found }
}

/// What a tensor that ends holds besides what it shows the garbage
/// collector, as far as that can be asked without running Python code.
/// Err where it may hold something that cannot be looked at from here,
/// so that its end has to be taken to run Python code:
///
/// - it requires a gradient (`holds_unseen`);
/// - it is of a class other than `torch.Tensor` itself (what is asked
///   below is answered by `__torch_function__` for such a tensor, which
///   is Python code);
/// - its memory is not known to be torch's own allocation: a storage
///   that cannot be resized was made around memory that something else
///   owns (a numpy array, a buffer, a DLPack capsule, a file that is
///   mapped), and what gives that memory back when the storage ends is
///   out of sight. Whether the storage ends with the tensor (another
///   tensor may share it) cannot be asked from here either.
///
/// - a level of forward-mode differentiation is open (a tangent is held
///   for the tensor from C++, and cannot be asked for without the level);
/// - it is not laid out plainly (`plainly_laid_out`): sparse, quantized,
///   nested, of another library's layout, or a wrapper around another
///   tensor, each of which has parts that C++ holds for it.
///
/// Ok with what C++ holds for the tensor that can be handed over (new
/// references): the Python object of its storage, which C++ keeps for
/// the storage and which ends with it; its `.grad` where it has one; and,
/// for a view, the tensor it is a view of, whose Python object C++ keeps
/// for as long as a view needs it. They are looked at like what the
/// tensor refers to, whatever their counts of references say.
unsafe fn unseen(o: Obj) -> Result<Vec<PyObj>, ()> {
    static NAMES: std::sync::OnceLock<(PyObj, PyObj)> = std::sync::OnceLock::new();
    if holds_unseen(o) {
        return Err(());
    }
    if ffi::Py_TYPE(o) as Obj != class_named(c"torch", c"Tensor") {
        return Err(());
    }
    let (storage, resizable) = NAMES.get_or_init(|| {
        let name = |text: &std::ffi::CStr| PyObj::from_owned(ffi::PyUnicode_InternFromString(text.as_ptr()));
        (name(c"untyped_storage"), name(c"resizable"))
    });
    let memory = ffi::PyObject_CallMethodNoArgs(o, storage.as_ptr());
    // A tensor without a storage of this kind says so by raising.
    if memory.is_null() {
        ffi::PyErr_Clear();
        return Err(());
    }
    let memory = PyObj::from_owned(memory);
    let said = ffi::PyObject_CallMethodNoArgs(memory.as_ptr(), resizable.as_ptr());
    let own = said == ffi::Py_True();
    xdecref(said);
    ffi::PyErr_Clear();
    if !own || forward_level() || !plainly_laid_out(o) {
        return Err(());
    }
    // The Python object of the storage ends with the storage: it is
    // looked at like any object that ends (what it has of its own, a
    // weak reference on it that is told, a finalizer).
    let mut held = vec![memory];
    for name in [c"grad", c"_base"] {
        let found = ffi::PyObject_GetAttrString(o, name.as_ptr());
        if found.is_null() {
            ffi::PyErr_Clear();
            return Err(());
        }
        let found = PyObj::from_owned(found);
        if found.as_ptr() != ffi::Py_None() {
            held.push(found);
        }
    }
    Ok(held)
}

/// Whether a tensor is one whose parts are all in its one storage: dense
/// and strided, not quantized (the scales and zero points per channel are
/// tensors that C++ holds for it), not nested, not of another library's
/// layout, and no wrapper around another tensor (a transform of
/// `torch.func`, a functional tensor), which C++ holds for it. Each is
/// asked of the tensor through an attribute or a function that torch has
/// in C; false where one cannot be asked.
unsafe fn plainly_laid_out(o: Obj) -> bool {
    static PARTS: std::sync::OnceLock<Option<(PyObj, PyObj, PyObj)>> = std::sync::OnceLock::new();
    let parts = PARTS.get_or_init(|| {
        let of = |module: &std::ffi::CStr, name: &std::ffi::CStr| {
            let module = ffi::PyDict_GetItemString(ffi::PyImport_GetModuleDict(), module.as_ptr());
            let found = if module.is_null() { null_mut() } else { ffi::PyObject_GetAttrString(module, name.as_ptr()) };
            ffi::PyErr_Clear();
            (!found.is_null()).then(|| PyObj::from_owned(found))
        };
        Some((of(c"torch", c"strided")?, of(c"torch._C._functorch", c"is_functorch_wrapped_tensor")?, of(c"torch", c"_is_functional_tensor")?))
    });
    let Some((strided, wrapped, functional)) = parts else {
        return false;
    };
    let layout = ffi::PyObject_GetAttrString(o, c"layout".as_ptr());
    let dense = layout == strided.as_ptr();
    xdecref(layout);
    let says = |name: &std::ffi::CStr| {
        let said = ffi::PyObject_GetAttrString(o, name.as_ptr());
        xdecref(said);
        said
    };
    let asked = |function: &PyObj| {
        let said = ffi::PyObject_CallOneArg(function.as_ptr(), o);
        xdecref(said);
        said
    };
    let no = ffi::Py_False();
    let plain = dense && says(c"is_quantized") == no && says(c"is_nested") == no && says(c"is_mkldnn") == no && asked(wrapped) == no && asked(functional) == no;
    ffi::PyErr_Clear();
    plain
}

/// Whether a level of forward-mode differentiation is open on this thread
/// (`torch.autograd.forward_ad`), or cannot be shown not to be.
unsafe fn forward_level() -> bool {
    let module = ffi::PyDict_GetItemString(ffi::PyImport_GetModuleDict(), c"torch.autograd.forward_ad".as_ptr());
    if module.is_null() || ffi::PyModule_Check(module) == 0 {
        return false;
    }
    let level = ffi::PyDict_GetItemString(ffi::PyModule_GetDict(module), c"_current_level".as_ptr());
    level.is_null() || ffi::PyLong_CheckExact(level) == 0 || ffi::PyLong_AsLong(level) >= 0
}

/// Whether an object that shows the garbage collector nothing (its class
/// has no `tp_traverse`) is known to hold nothing whose end could run
/// Python code: an object of one of the interpreter's own built-in
/// classes (numbers, texts, bytes, ranges and what goes over them, code,
/// None), other than a capsule (which owns what its destructor lets go
/// of) and what goes over a text or bytes (which holds it, and it may be
/// of a class of the user's); a dtype, a device, a layout or a memory
/// format of torch's; a number of numpy's. An object of any other such
/// class may hold references that cannot be seen (a numpy array and the
/// buffer it is over, a storage), and its end is taken to run Python
/// code.
unsafe fn holds_nothing(o: Obj) -> bool {
    // The interpreter marks its own built-in classes.
    const BUILT_IN: std::ffi::c_ulong = 1 << 1;
    if is_plain(o) {
        return true;
    }
    let ty = ffi::Py_TYPE(o);
    let class = ty as Obj;
    if ffi::PyType_GetFlags(ty) & BUILT_IN != 0 {
        static RANGES: std::sync::OnceLock<[usize; 2]> = std::sync::OnceLock::new();
        let ranges = RANGES.get_or_init(|| {
            // What goes over a range, a short one and a long one.
            let over = |stop: Obj| {
                let range = ffi::PyObject_CallOneArg(std::ptr::addr_of_mut!(ffi::PyRange_Type) as Obj, stop);
                let made = if range.is_null() { null_mut() } else { ffi::PyObject_GetIter(range) };
                let class = if made.is_null() { 0 } else { ffi::Py_TYPE(made) as usize };
                xdecref(made);
                xdecref(range);
                xdecref(stop);
                ffi::PyErr_Clear();
                class
            };
            [over(ffi::PyLong_FromLong(1)), over(ffi::PyLong_FromDouble(1e30))]
        });
        let goes_over = (*ty).tp_iternext.is_some() && !ranges.contains(&(class as usize));
        return ffi::PyCapsule_CheckExact(o) == 0 && !goes_over;
    }
    const OWN: [&std::ffi::CStr; 4] = [c"dtype", c"device", c"layout", c"memory_format"];
    if OWN.iter().any(|name| class == class_named(c"torch", name)) {
        return true;
    }
    // A number of numpy's own kinds: a class defined in C under `number`
    // or `bool_`.
    let (number, flag) = (class_named(c"numpy", c"number"), class_named(c"numpy", c"bool_"));
    let under = |base: Obj| !base.is_null() && ffi::PyType_IsSubtype(ty, base as *mut ffi::PyTypeObject) != 0;
    is_static(ty) && (under(number) || under(flag))
}

/// Whether the end of an object runs code of its class: a `__del__`, or
/// what the interpreter's own classes do at the end of their objects (a
/// file is closed, a generator that waits at a `yield` is closed, which
/// runs what it has left to run). A generator that was never started, or
/// that ran to its end, has nothing to run.
unsafe fn finalized(o: Obj) -> bool {
    let ty = ffi::Py_TYPE(o);
    if (*ty).tp_del.is_some() {
        return true;
    }
    if (*ty).tp_finalize.is_none() {
        return false;
    }
    if ffi::PyGen_CheckExact(o) == 0 {
        return true;
    }
    let rests = |name: &std::ffi::CStr| {
        let said = ffi::PyObject_GetAttrString(o, name.as_ptr());
        let rests = said == ffi::Py_False();
        if said.is_null() {
            ffi::PyErr_Clear();
        }
        xdecref(said);
        rests
    };
    !(rests(c"gi_suspended") && rests(c"gi_running"))
}

/// Whether an object is a tensor (of any class of tensors).
unsafe fn is_tensor(o: Obj) -> bool {
    let mut info = TcfgTensorInfo::empty();
    capi::api().is_some_and(|api| (api.tensor_info)(o, &mut info) != 0)
}

/// Whether the objects that a path lets go of where a callable has run in
/// its place (`gone`, each as often as the path lets go of a reference to
/// it; put in order here) end without Python code running, as far as they
/// end at all. Asked before the path has taken anything of the call, so
/// that every reference is somebody else's or one that is let go of.
/// Looked at as the garbage collector looks at a set of objects: an
/// object ends if its count of references is no larger than the
/// references to it that the path lets go of and those from objects that
/// end; else one of its references is somebody else's and it stays, with
/// everything it refers to, and is not looked into. `stay` are objects
/// that are known to stay. What ends must be of a class without a
/// finalizer and have no weak reference with a callback other than a weak
/// container's own (`weakly_watched`); what it refers to, and what a weak
/// container lets go of with its entry, is looked at in the same way.
/// Nothing of this is remembered from one call to the next: which
/// objects are let go of, how many references each has, what class it is
/// of and what that class does at the end of an object, the weak
/// references to it and what it refers to are all read at the call. Err
/// with the object whose end may run Python code, or with null where
/// there is too much to look at (4096 objects that end).
pub unsafe fn end_quietly(gone: &mut [Obj], stay: &[Obj]) -> Result<(), Obj> {
    const ENDING: usize = 4096;
    gone.sort_unstable();
    // The distinct objects, each with how often it is let go of.
    let runs = |gone: &[Obj]| {
        let mut runs: Vec<(Obj, isize)> = Vec::new();
        for o in gone {
            match runs.last_mut() {
                Some((last, count)) if *last == *o => *count += 1,
                _ => runs.push((*o, 1)),
            }
        }
        runs
    };
    // A number, a text or None refers to nothing and ends quietly.
    let ends_alone = |o: Obj| is_plain(o) || ffi::PyBytes_CheckExact(o) != 0;
    // The common case, without keeping anything: every object has a
    // reference besides those that are let go of, so nothing ends.
    let (mut at, mut all) = (0, true);
    while at < gone.len() && all {
        let o = gone[at];
        let count = gone[at..].iter().take_while(|other| **other == o).count();
        all = ffi::Py_REFCNT(o) > count as isize || ends_alone(o) || stay.contains(&o);
        at += count;
    }
    if all {
        return Ok(());
    }
    // Per object that was reached: the references to it that are not let
    // go of by the path (counted once, when it is reached),
    // how many of those are from objects that end, and whether it ends.
    struct Reached {
        others: isize,
        inside: isize,
        ends: bool,
        /// It goes with a tensor that ends, whatever its count says.
        with: bool,
    }
    let mut reached: HashMap<usize, Reached> = HashMap::new();
    // Those that were reached and not counted yet, and those whose count
    // is to be compared again.
    let mut fresh: Vec<(Obj, isize)> = Vec::new();
    let mut touched: Vec<Obj> = Vec::new();
    for (o, count) in runs(gone) {
        if !ends_alone(o) && !stay.contains(&o) && !reached.contains_key(&(o as usize)) {
            reached.insert(o as usize, Reached { others: 0, inside: 0, ends: false, with: false });
            fresh.push((o, count));
            touched.push(o);
        }
    }
    // What a weak container lets go of with its entry is kept here while
    // it is looked at: those references are not anybody's.
    let mut more: Vec<PyObj> = Vec::new();
    let mut found: Vec<Obj> = Vec::new();
    let mut ended = 0;
    loop {
        for (o, dropped) in fresh.drain(..) {
            let kept_here = more.iter().filter(|other| other.as_ptr() == o).count() as isize;
            let entry = reached.get_mut(&(o as usize)).expect("it was reached");
            entry.others = ffi::Py_REFCNT(o) - kept_here - dropped;
        }
        let mut ending: Vec<Obj> = Vec::new();
        for o in touched.drain(..) {
            let entry = reached.get_mut(&(o as usize)).expect("it was reached");
            if !entry.ends && (entry.with || entry.others - entry.inside < 1) {
                entry.ends = true;
                ending.push(o);
            }
        }
        if ending.is_empty() {
            return Ok(());
        }
        for o in ending {
            ended += 1;
            if ended > ENDING {
                return Err(null_mut());
            }
            let ty = ffi::Py_TYPE(o);
            found.clear();
            let before = more.len();
            if finalized(o) || weakly_watched(o, &mut more) {
                return Err(o);
            }
            // A tensor ends with what C++ holds for it: its history
            // and the hooks on it, what owns its memory, its `.grad`.
            // Where that cannot be looked at, its end is taken to run
            // Python code.
            let tensor = is_tensor(o);
            if tensor {
                match unseen(o) {
                    Ok(gradient) => more.extend(gradient),
                    Err(()) => return Err(o),
                }
            }
            found.extend(more[before..].iter().map(|other| other.as_ptr()));
            match (*ty).tp_traverse {
                Some(traverse) => {
                    traverse(o, every_referent, &mut found as *mut Vec<Obj> as *mut std::ffi::c_void);
                }
                // What shows nothing may hold what cannot be seen.
                None if !holds_nothing(o) => return Err(o),
                None => {}
            }
            for other in &found {
                // A class does not end where a reference to it is let go of.
                if ffi::PyType_Check(*other) != 0 || ends_alone(*other) || stay.contains(other) {
                    continue;
                }
                let entry = reached.entry(*other as usize).or_insert_with(|| {
                    let dropped = gone.partition_point(|o| *o <= *other) - gone.partition_point(|o| *o < *other);
                    fresh.push((*other, dropped as isize));
                    Reached { others: 0, inside: 0, ends: false, with: false }
                });
                entry.inside += 1;
                // What a tensor shows of itself (its attributes, the
                // hooks on it) may be held from C++ as well, by what
                // ends with the tensor: it is looked at whatever its
                // count of references says.
                entry.with |= tensor;
                touched.push(*other);
            }
        }
    }
}

/// The same for a set: a small one that holds nothing but plain values.
unsafe fn quiet_members(set: Obj) -> bool {
    const SCANNED: isize = 32;
    if ffi::PySet_Size(set) > SCANNED {
        return false;
    }
    let members = ffi::PyObject_GetIter(set);
    if members.is_null() {
        ffi::PyErr_Clear();
        return false;
    }
    let mut plain = true;
    loop {
        let member = ffi::PyIter_Next(members);
        if member.is_null() {
            break;
        }
        plain &= is_plain_key(member);
        ffi::Py_DECREF(member);
    }
    ffi::PyErr_Clear();
    ffi::Py_DECREF(members);
    plain
}

/// Whether `container[key]` runs no Python code.
pub unsafe fn quiet_item(container: Obj, key: Obj) -> bool {
    is_plain_container(container) && is_plain_key(key) && (ffi::PyDict_Check(container) == 0 || quiet_keys(container))
}

/// Whether `len(container)`, or unpacking it, runs no Python code.
pub unsafe fn quiet_sized(container: Obj) -> bool {
    is_plain_container(container) || (!capi::is_unboxed(container) && ffi::PyAnySet_CheckExact(container) != 0)
}

/// Whether `item in container` runs no Python code: the item is compared
/// with what the container holds.
pub unsafe fn quiet_contains(item: Obj, container: Obj) -> bool {
    if capi::is_unboxed(item) || capi::is_unboxed(container) || !is_plain(item) {
        return false;
    }
    if is_plain_dict(container) {
        return quiet_keys(container);
    }
    if ffi::PyAnySet_CheckExact(container) != 0 {
        return quiet_members(container);
    }
    if ffi::PyUnicode_CheckExact(container) != 0 {
        return ffi::PyUnicode_CheckExact(item) != 0;
    }
    let sequence = ffi::PyList_CheckExact(container) != 0 || ffi::PyTuple_CheckExact(container) != 0;
    sequence && (0..ffi::PyObject_Size(container)).all(|i| is_plain(sequence_item(container, i)))
}

/// Whether what a type has under a name is given out by Python code: a
/// property whose getter is a Python function, or a descriptor of a class
/// that is defined in Python.
unsafe fn python_descriptor(descr: Obj) -> bool {
    let ty = ffi::Py_TYPE(descr);
    if (*ty).tp_descr_get.is_none() {
        return false;
    }
    if ty == std::ptr::addr_of_mut!(ffi::PyProperty_Type) {
        let getter = ffi::PyObject_GetAttrString(descr, c"fget".as_ptr());
        let python = getter.is_null() || ffi::PyCFunction_Check(getter) == 0;
        ffi::PyErr_Clear();
        xdecref(getter);
        return python;
    }
    !is_static(ty) && ffi::PyFunction_Check(descr) == 0
}

/// Whether `getattr(obj, name)` runs no Python code, as far as that can be
/// told without running it: what the lookup of the object's type would do
/// is followed up to where it finds the attribute. False where it would
/// call a property, a descriptor or a `__getattr__` written in Python,
/// where the type has a lookup of its own that is not known here, and where
/// the attribute is not there.
pub unsafe fn quiet_attr(obj: Obj, name: Obj, api: &TcfgApi) -> bool {
    if capi::is_unboxed(obj) {
        return true;
    }
    let ty = ffi::Py_TYPE(obj);
    if ffi::PyModule_CheckExact(obj) != 0 {
        let dict = ffi::PyModule_GetDict(obj);
        return !dict.is_null() && !ffi::PyDict_GetItemWithError(dict, name).is_null();
    }
    if ffi::PyType_Check(obj) != 0 {
        // A class: what it has itself, or else what `type` has for it.
        if ty != std::ptr::addr_of_mut!(ffi::PyType_Type) {
            return false;
        }
        let own = pyffi::_PyType_Lookup(obj as *mut ffi::PyTypeObject, name);
        let meta = pyffi::_PyType_Lookup(ty, name);
        return if own.is_null() { !meta.is_null() } else { !python_descriptor(own) };
    }
    // The lookup of `object`, or that and then a `__getattr__` for what it
    // does not find: of the latter, only one that has a shortcut is known.
    let mut info = TcfgTensorInfo::empty();
    let object = std::ptr::addr_of!(ffi::PyBaseObject_Type);
    let generic = (*ty).tp_getattro.map(|own| own as usize) == (*object).tp_getattro.map(|own| own as usize);
    let hooked = !generic && crate::shortcuts::plain_getattribute(ty);
    let shortcut = hooked && crate::shortcuts::has_getattr(ty);
    if !(generic || hooked || (api.tensor_info)(obj, &mut info) != 0) {
        return false;
    }
    let descr = pyffi::_PyType_Lookup(ty, name);
    let python = !descr.is_null() && python_descriptor(descr);
    let data = !descr.is_null() && (*ffi::Py_TYPE(descr)).tp_descr_set.is_some();
    if data {
        return !python;
    }
    let slot = pyffi::_PyObject_GetDictPtr(obj);
    let own = !slot.is_null() && !(*slot).is_null() && !ffi::PyDict_GetItemWithError(*slot, name).is_null();
    if own {
        return true;
    }
    if !descr.is_null() {
        return !python;
    }
    shortcut && crate::shortcuts::finds(obj, name)
}

/// What of the state of a thread changes what a compiled graph computes and
/// is neither among the modes nor among the local dispatch keys that a form
/// is for: the dtype that autocast casts to on each kind of device, while
/// autocast is on for any, and whether operations have to be deterministic.
/// As one number, to compare. A graph compiler builds these into what it
/// makes; the kernels of eager mode look them up when they run.
pub unsafe fn thread_state() -> u64 {
    const DEVICES: [&std::ffi::CStr; 8] = [c"cpu", c"cuda", c"mps", c"xpu", c"ipu", c"hpu", c"xla", c"mtia"];
    const ASKED: [&std::ffi::CStr; 4] = [
        c"_is_any_autocast_enabled",
        c"get_autocast_dtype",
        c"_get_deterministic_algorithms",
        c"_get_deterministic_algorithms_warn_only",
    ];
    static FUNCTIONS: std::sync::OnceLock<(Vec<Option<PyObj>>, Vec<PyObj>)> = std::sync::OnceLock::new();
    let (functions, devices) = FUNCTIONS.get_or_init(|| {
        let binding = ffi::PyImport_ImportModule(c"torch._C".as_ptr());
        let functions = ASKED
            .iter()
            .map(|name| {
                let found = if binding.is_null() { null_mut() } else { ffi::PyObject_GetAttrString(binding, name.as_ptr()) };
                (!found.is_null()).then(|| PyObj::from_owned(found))
            })
            .collect();
        ffi::PyErr_Clear();
        xdecref(binding);
        let devices = DEVICES.iter().map(|name| PyObj::from_owned(ffi::PyUnicode_FromString(name.as_ptr()))).collect();
        (functions, devices)
    });
    let mut state: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |v: u64| state = (state ^ v).wrapping_mul(0x0000_0100_0000_01b3);
    let yes = |which: usize| match &functions[which] {
        Some(function) => {
            let said = ffi::PyObject_CallNoArgs(function.as_ptr());
            let yes = said == ffi::Py_True();
            xdecref(said);
            // One that cannot be asked is taken to be on.
            yes || said.is_null()
        }
        None => true,
    };
    if yes(0) {
        for device in devices {
            let dtype = match &functions[1] {
                Some(function) => ffi::PyObject_CallOneArg(function.as_ptr(), device.as_ptr()),
                None => null_mut(),
            };
            // A dtype is one object for as long as the process runs.
            mix(dtype as u64);
            xdecref(dtype);
        }
    }
    mix(yes(2) as u64);
    mix(yes(3) as u64);
    ffi::PyErr_Clear();
    state
}

/// What stands between the operations of the thread and their kernels and
/// has to see each one as the function wrote it, if anything does: a
/// dispatch mode, or a transform of functorch (a torch function mode shows
/// in `call_modes`). A run under one is never exported: what was made of a
/// graph would hide its operations, or show others.
pub unsafe fn interposed() -> Option<&'static str> {
    static ASKED: std::sync::OnceLock<[Option<PyObj>; 2]> = std::sync::OnceLock::new();
    let asked = ASKED.get_or_init(|| {
        let found = |module: &std::ffi::CStr, name: &std::ffi::CStr| {
            let module = ffi::PyImport_ImportModule(module.as_ptr());
            let function = if module.is_null() { null_mut() } else { ffi::PyObject_GetAttrString(module, name.as_ptr()) };
            ffi::PyErr_Clear();
            xdecref(module);
            (!function.is_null()).then(|| PyObj::from_owned(function))
        };
        [found(c"torch._C", c"_len_torch_dispatch_stack"), found(c"torch._C._functorch", c"maybe_current_level")]
    });
    let said = |which: usize| asked[which].as_ref().map_or(null_mut(), |f| ffi::PyObject_CallNoArgs(f.as_ptr()));
    let (modes, level) = (said(0), said(1));
    // What cannot be asked is taken to be there.
    let dispatch = modes.is_null() || ffi::PyLong_AsLong(modes) != 0;
    let transform = level.is_null() || level != ffi::Py_None();
    ffi::PyErr_Clear();
    xdecref(modes);
    xdecref(level);
    if dispatch {
        Some("a dispatch mode is to see the operations")
    } else if transform {
        Some("a transform of torch.func is on")
    } else {
        None
    }
}

/// What a run of a path on the values of a call found, for an exported form
/// to be made from: the values its nodes took from outside the graph (owned,
/// null until one is taken), and per node the site that ran and what it
/// returned (dtype, device, requires_grad, sizes: an owned tuple).
pub struct Recording {
    pub feeds: Vec<Obj>,
    pub nodes: Vec<(*mut TcfgSite, Obj)>,
    /// Why the run cannot be described, once something says so.
    pub failed: Option<String>,
    /// An operation wrote to a tensor it was given.
    pub wrote: bool,
}

impl Recording {
    pub fn new(path: &Path) -> Recording {
        let feeds = vec![null_mut(); path.feeds.len()];
        Recording { feeds, nodes: Vec::new(), failed: None, wrote: false }
    }
}

impl Drop for Recording {
    fn drop(&mut self) {
        unsafe {
            for feed in &self.feeds {
                xdecref(*feed);
            }
            for (_, meta) in &self.nodes {
                xdecref(*meta);
            }
        }
    }
}

/// (dtype, device, requires_grad, sizes) of a tensor that has its Python
/// object; null if it cannot be read.
pub unsafe fn tensor_meta(tensor: Obj) -> Obj {
    let dtype = ffi::PyObject_GetAttrString(tensor, c"dtype".as_ptr());
    let device = ffi::PyObject_GetAttrString(tensor, c"device".as_ptr());
    let grad = ffi::PyObject_GetAttrString(tensor, c"requires_grad".as_ptr());
    let shape = ffi::PyObject_GetAttrString(tensor, c"shape".as_ptr());
    let sizes = if shape.is_null() { null_mut() } else { ffi::PySequence_Tuple(shape) };
    let all = [dtype, device, grad, sizes];
    let meta = if all.iter().any(|o| o.is_null()) {
        ffi::PyErr_Clear();
        null_mut()
    } else {
        ffi::PyTuple_Pack(4, dtype, device, grad, sizes)
    };
    for o in all.into_iter().chain([shape]) {
        xdecref(o);
    }
    meta
}

unsafe fn tuple_of(items: &[Obj]) -> Obj {
    let tuple = ffi::PyTuple_New(items.len() as isize);
    for (i, item) in items.iter().enumerate() {
        ffi::PyTuple_SET_ITEM(tuple, i as isize, *item);
    }
    tuple
}

unsafe fn text(s: &str) -> Obj {
    ffi::PyUnicode_FromStringAndSize(s.as_ptr().cast(), s.len() as isize)
}

/// ("v", value): a value the graph has.
unsafe fn value_ref(v: PathValue) -> Obj {
    tuple_of(&[text("v"), ffi::PyLong_FromLong(v as _)])
}

/// ("c", object): a constant of the graph.
unsafe fn const_ref(o: Obj) -> Obj {
    tuple_of(&[text("c"), new_ref(o)])
}

/// A list of these objects, whose references it takes.
unsafe fn list_of(items: Vec<Obj>) -> Obj {
    let list = ffi::PyList_New(items.len() as isize);
    for (i, item) in items.into_iter().enumerate() {
        ffi::PyList_SET_ITEM(list, i as isize, item);
    }
    list
}

/// What `Path::describe` makes of a recorded run: an exported form is made
/// from it.
pub struct Described {
    /// What the exporter is handed. None for a call that is not to be
    /// exported: `closed` says why.
    pub description: Option<PyObj>,
    pub closed: Option<String>,
    /// What a call has to find for the form to be for it, and the sizes
    /// and strides of the tensors among that.
    pub checks: Vec<Check>,
    pub numbers: Vec<i64>,
    /// The tensors the callable is called with.
    pub args: Vec<PathValue>,
}

impl Path {
    /// What the exporter is handed for a recorded run, with the checks the
    /// exported form holds under and the values the callable is called
    /// with. A dict:
    ///   where: the name of the graph;
    ///   inputs, input_ids, input_names: the tensors the graph takes, as
    ///     they are in this call, with the value each is and a name for it;
    ///   nodes: in order, ("aten", what `site_export` gives, operands, value,
    ///     meta) per kernel call, and ("size", tensor, dim, value), ("dim",
    ///     tensor, value), ("binary", symbol, a, b, value), ("neg", a, value)
    ///     or ("tuple", items, value) per integer the graph computes; an
    ///     operand is ("v", value) or ("c", object), a constant;
    ///   outputs: the values the callable returns, as a tuple in this order;
    ///   exits: integers it returns behind them, ("dim", tensor) or ("size",
    ///     tensor, dim) each.
    /// A run that cannot be exported is described by its checks alone: the
    /// form made of them has the calls like it run the instructions.
    pub unsafe fn describe(&self, name: &str, func: &Function, rec: &Recording, api: &TcfgApi) -> Described {
        let mut checks: Vec<Check> = Vec::new();
        let mut tensors = vec![false; self.feeds.len()];
        let mut closed = rec.failed.clone();
        let mut wrote = rec.wrote;
        let nodes = match &closed {
            None => self.nodes(rec, api, &mut checks, &mut tensors, &mut wrote),
            Some(_) => Ok(PyObj::from_owned(ffi::PyList_New(0))),
        };
        let nodes = nodes.unwrap_or_else(|why| {
            closed = Some(why);
            PyObj::from_owned(ffi::PyList_New(0))
        });
        // The tensors the run was given, each checked to be like the one of
        // this call; those the graph takes are its arguments.
        let list = || PyObj::from_owned(ffi::PyList_New(0));
        let (inputs, ids, names) = (list(), list(), list());
        let mut args: Vec<PathValue> = Vec::new();
        let mut given: Vec<Obj> = Vec::new();
        let mut numbers: Vec<i64> = Vec::new();
        // A run whose operations could not be followed says nothing of the
        // values they took: the form is then for calls that have a value of
        // the same class wherever this one has something that is no plain
        // tensor, and for no other.
        let constants: Vec<PathValue> =
            self.body.iter().filter(|inst| matches!(inst.op, Op::Const(_))).map(|inst| inst.dst).collect();
        for (k, v) in self.feeds.iter().enumerate() {
            let (tensor, taken) = (rec.feeds[k], tensors[k]);
            let same = given.iter().position(|o| *o == tensor).unwrap_or(given.len());
            let Some(row) = (if tensor.is_null() { None } else { Row::of(tensor, same, &mut numbers, api) }) else {
                let unsaid = rec.failed.is_some()
                    && !tensor.is_null()
                    && !capi::is_unboxed(tensor)
                    && !constants.contains(v)
                    && !checks.iter().any(|check| check.target == *v);
                if unsaid {
                    checks.push(Check { target: *v, cond: Cond::Type(PyObj::from_borrowed(ffi::Py_TYPE(tensor) as Obj)) });
                }
                continue;
            };
            if row.forward && taken && closed.is_none() {
                closed = Some("a tensor of the call carries a forward-mode gradient".to_string());
            }
            checks.push(Check { target: *v, cond: Cond::Tensor(row) });
            given.push(tensor);
            if !taken {
                continue;
            }
            args.push(*v);
            ffi::PyList_Append(inputs.as_ptr(), tensor);
            let id = ffi::PyLong_FromLong(*v as _);
            ffi::PyList_Append(ids.as_ptr(), id);
            ffi::Py_DECREF(id);
            let label = match func.value_names.get(*v as usize).and_then(|n| n.as_ref()) {
                Some(label) => label.clone(),
                None => format!("v{}", v),
            };
            let label = text(&label);
            ffi::PyList_Append(names.as_ptr(), label);
            ffi::Py_DECREF(label);
        }
        if wrote && closed.is_none() {
            // What is made of a graph runs its operations again where it
            // fails the first time, so a call whose operations write is
            // not exported.
            closed = Some("its operations write to what they are given".to_string());
        }
        let outputs = list_of(self.outputs.iter().map(|v| ffi::PyLong_FromLong(*v as _)).collect());
        let exits = list_of(
            self.exits
                .iter()
                .map(|cond| match cond {
                    GuardCond::TensorRank { v, .. } => tuple_of(&[text("dim"), value_ref(*v)]),
                    GuardCond::DimRange { v, dim, .. } => {
                        tuple_of(&[text("size"), value_ref(*v), ffi::PyLong_FromLongLong(*dim)])
                    }
                })
                .collect(),
        );
        let description = PyObj::from_owned(ffi::PyDict_New());
        let label = text(name);
        let entries: [(&std::ffi::CStr, Obj); 7] = [
            (c"where", label),
            (c"inputs", inputs.as_ptr()),
            (c"input_ids", ids.as_ptr()),
            (c"input_names", names.as_ptr()),
            (c"nodes", nodes.as_ptr()),
            (c"outputs", outputs),
            (c"exits", exits),
        ];
        for (key, value) in entries {
            ffi::PyDict_SetItemString(description.as_ptr(), key.as_ptr(), value);
        }
        for made in [label, outputs, exits] {
            ffi::Py_DECREF(made);
        }
        if !ffi::PyErr_Occurred().is_null() && closed.is_none() {
            closed = Some(format!("the graph cannot be described: {}", raised_text()));
        }
        ffi::PyErr_Clear();
        let description = if closed.is_none() { Some(description) } else { None };
        Described { description, closed, checks, numbers, args }
    }

    /// The nodes of the description of a recorded run, as a list. Adds the
    /// checks on the values the graph has as constants, marks the feeds it
    /// takes as tensors, and finds whether a normalization updated its
    /// statistics. The reason, if the run cannot be exported.
    unsafe fn nodes(
        &self,
        rec: &Recording,
        api: &TcfgApi,
        checks: &mut Vec<Check>,
        tensors: &mut [bool],
        wrote: &mut bool,
    ) -> Result<PyObj, String> {
        let inside: Vec<PathValue> = self
            .body
            .iter()
            .filter(|inst| inst.kind == Kind::Node || inst.in_graph)
            .map(|inst| inst.dst)
            .collect();
        // An operand as the graph has it: a value of its own, a tensor it is
        // given, or a constant, which the exported form is then checked for.
        // How it is taken is what its call site says.
        let mut operand = |v: PathValue, how: &str| -> Result<Obj, String> {
            if inside.contains(&v) {
                return Ok(value_ref(v));
            }
            let feed = self.feeds.iter().position(|f| *f == v);
            let value = feed.map_or(null_mut(), |k| rec.feeds[k]);
            if value.is_null() {
                return Err("an operand of the graph has no value".to_string());
            }
            let mut info = TcfgTensorInfo::empty();
            let tensor = (api.tensor_info)(value, &mut info) != 0;
            let fits = match how {
                "tensor" => tensor,
                "int" => ffi::PyLong_CheckExact(value) != 0,
                "exact" => checkable(value),
                _ => true,
            };
            if !fits {
                checks.push(Check { target: v, cond: Cond::Type(PyObj::from_borrowed(ffi::Py_TYPE(value) as Obj)) });
                return Err(format!("an operand of the graph is a {}", pyffi::type_name(value)));
            }
            let cond = match how {
                "tensor" => {
                    tensors[feed.unwrap()] = true;
                    return Ok(value_ref(v));
                }
                "none" => return Ok(const_ref(value)),
                "int" | "exact" => Cond::ValueEq(frozen(value)),
                _ => Cond::Is(PyObj::from_borrowed(value)),
            };
            // What the graph has as the constant is what the check keeps:
            // a list that is changed in place later is another value.
            let constant = match &cond {
                Cond::ValueEq(kept) => const_ref(kept.as_ptr()),
                _ => const_ref(value),
            };
            let known = checks.iter().any(|c| {
                c.target == v && std::mem::discriminant(&c.cond) == std::mem::discriminant(&cond)
            });
            if !known {
                checks.push(Check { target: v, cond });
            }
            Ok(constant)
        };
        // Some operands, as a list.
        type Taken<'a> = [(PathValue, &'a str)];
        fn gather(
            operand: &mut impl FnMut(PathValue, &str) -> Result<Obj, String>,
            vs: &Taken,
        ) -> Result<Obj, String> {
            let mut items: Vec<Obj> = Vec::new();
            for (v, how) in vs {
                match operand(*v, *how) {
                    Ok(o) => items.push(o),
                    Err(why) => {
                        items.into_iter().for_each(|o| unsafe { ffi::Py_DECREF(o) });
                        return Err(why);
                    }
                }
            }
            Ok(unsafe { list_of(items) })
        }
        let nodes = PyObj::from_owned(ffi::PyList_New(0));
        let add = |node: Obj| {
            ffi::PyList_Append(nodes.as_ptr(), node);
            ffi::Py_DECREF(node);
        };
        let number = |n: i64| ffi::PyLong_FromLongLong(n);
        let mut recorded = rec.nodes.iter();
        let did_not_run = || "an operation of the graph did not run".to_string();
        for inst in &self.body {
            let value = inst.dst as i64;
            match &inst.op {
                Op::Dispatch(d) if inst.kind == Kind::Node => {
                    let (site, meta) = recorded.next().ok_or_else(did_not_run)?;
                    let exported = (api.site_export)(*site);
                    if exported.is_null() {
                        return Err(format!("a call site cannot be exported: {}", raised_text()));
                    }
                    let exported = PyObj::from_owned(exported);
                    let given: Vec<Obj> = d
                        .operands
                        .iter()
                        .map(|v| self.feeds.iter().position(|f| f == v).map_or(null_mut(), |k| rec.feeds[k]))
                        .collect();
                    *wrote |= updates_statistics(exported.as_ptr(), &given);
                    let guards = ffi::PyDict_GetItemString(exported.as_ptr(), c"guards".as_ptr());
                    let hows: Vec<String> = (0..d.operands.len())
                        .map(|k| {
                            if guards.is_null() || k as isize >= ffi::PyList_GET_SIZE(guards) {
                                "identity".to_string()
                            } else {
                                pyffi::str_to_string(ffi::PyList_GET_ITEM(guards, k as isize))
                            }
                        })
                        .collect();
                    let taken: Vec<(PathValue, &str)> =
                        d.operands.iter().zip(&hows).map(|(v, how)| (*v, how.as_str())).collect();
                    let taken = gather(&mut operand, &taken)?;
                    let meta = if meta.is_null() { ffi::Py_None() } else { *meta };
                    add(tuple_of(&[text("aten"), new_ref(exported.as_ptr()), taken, number(value), new_ref(meta)]));
                }
                op if inst.in_graph => {
                    let node = match op {
                        Op::TensorSize { t, dim } => {
                            let taken = operand(*t, "tensor")?;
                            vec![text("size"), taken, number(*dim)]
                        }
                        Op::TensorDim(t) => {
                            let taken = operand(*t, "tensor")?;
                            vec![text("dim"), taken]
                        }
                        Op::Binary(k, a, b) => {
                            let taken = gather(&mut operand, &[(*a, "int"), (*b, "int")])?;
                            let (a, b) = (ffi::PyList_GET_ITEM(taken, 0), ffi::PyList_GET_ITEM(taken, 1));
                            let node = vec![text("binary"), text(BINARY_SYMBOLS[*k as usize]), new_ref(a), new_ref(b)];
                            ffi::Py_DECREF(taken);
                            node
                        }
                        Op::Unary(_, a) => {
                            let taken = operand(*a, "int")?;
                            vec![text("neg"), taken]
                        }
                        Op::BuildTuple(vs) => {
                            let items: Vec<(PathValue, &str)> = vs.iter().map(|v| (*v, "int")).collect();
                            let taken = gather(&mut operand, &items)?;
                            vec![text("tuple"), taken]
                        }
                        _ => return Err("a value of the graph has no node".to_string()),
                    };
                    let mut node = node;
                    node.push(number(value));
                    add(tuple_of(&node));
                }
                _ => {}
            }
        }
        Ok(nodes)
    }
}

impl Exported {
    /// For a form of the same kind of call as these checks describe that
    /// differs from them in nothing but numbers: which of the checks have
    /// another number. None for a form of another kind.
    unsafe fn other_numbers(&self, world: (u64, u64, u64, u64), checks: &[Check], sizes: &[i64]) -> Option<Vec<usize>> {
        let same = (self.modes, self.included, self.excluded, self.state) == world && self.sizes.holds_numbers(sizes);
        if !same || self.checks.len() != checks.len() {
            return None;
        }
        let number = |o: Obj| ffi::PyFloat_CheckExact(o) != 0 || ffi::PyLong_CheckExact(o) != 0;
        let mut others = Vec::new();
        for (i, (mine, theirs)) in self.checks.iter().zip(checks).enumerate() {
            if mine.target != theirs.target {
                return None;
            }
            match (&mine.cond, &theirs.cond) {
                (Cond::Is(a), Cond::Is(b)) if a.as_ptr() == b.as_ptr() => {}
                (Cond::Type(a), Cond::Type(b)) if a.as_ptr() == b.as_ptr() => {}
                (Cond::Tensor(a), Cond::Tensor(b)) if a == b => {}
                (Cond::ValueEq(a), Cond::ValueEq(b)) if same_value(a.as_ptr(), b.as_ptr()) => {}
                (Cond::ValueEq(a), Cond::ValueEq(b))
                    if number(a.as_ptr()) && ffi::Py_TYPE(a.as_ptr()) == ffi::Py_TYPE(b.as_ptr()) =>
                {
                    others.push(i)
                }
                _ => return None,
            }
        }
        Some(others)
    }

    /// What its checks ask of a call, as text.
    pub unsafe fn conditions(&self, func: &Function) -> Vec<String> {
        const DTYPES: [&str; 12] = [
            "uint8", "int8", "int16", "int32", "int64", "float16", "float32", "float64", "complex32",
            "complex64", "complex128", "bool",
        ];
        let short = |o: &PyObj| named(o.as_ptr());
        let numbers = self.sizes.made_on();
        let mut at = 0;
        self.checks
            .iter()
            .map(|check| {
                let value = func.value_name(check.target);
                match &check.cond {
                    Cond::Is(expected) => format!("{} is {}", value, short(expected)),
                    Cond::ValueEq(expected) => format!("{} == {}", value, short(expected)),
                    Cond::Type(ty) => format!("type({}) is {}", value, pyffi::class_name(ty.as_ptr())),
                    Cond::Tensor(row) => {
                        let field = |i: usize| i32::from_ne_bytes(row.kind[4 * i..4 * i + 4].try_into().unwrap());
                        let (rank, dtype, flags) = (field(0) as usize, field(1), field(3));
                        let dtype = DTYPES.get(dtype as usize).map_or_else(|| format!("dtype{}", dtype), |d| d.to_string());
                        let sizes: Vec<String> = numbers[at..at + rank].iter().map(|n| n.to_string()).collect();
                        at += rank * (1 + (flags >> 1 & 1) as usize);
                        let grad = if flags & 1 != 0 { ", requires_grad" } else { "" };
                        let forward = if row.forward { ", forward-mode gradient" } else { "" };
                        format!("{}: {}[{}]{}{}", value, dtype, sizes.join(", "), grad, forward)
                    }
                }
            })
            .collect()
    }
}

/// What an object is, in the text of a check: the value of what is compared
/// by value, the name of a function, a class or a module, and of anything
/// else the name of its class. No address is part of it, so the text is the
/// same in every run.
unsafe fn named(o: Obj) -> String {
    const LONGEST: usize = 40;
    if checkable(o) {
        let text = pyffi::repr(o);
        return match text.char_indices().nth(LONGEST) {
            Some((cut, _)) => format!("{}...", &text[..cut]),
            None => text,
        };
    }
    let string = |s: Obj| (!s.is_null() && ffi::PyUnicode_Check(s) != 0).then(|| pyffi::str_to_string(s));
    let of_module = |module: Obj| {
        let name = ffi::PyModule_GetNameObject(module);
        ffi::PyErr_Clear();
        let text = string(name);
        xdecref(name);
        text
    };
    if ffi::PyType_Check(o) != 0 {
        return pyffi::class_name(o);
    }
    if ffi::PyModule_Check(o) != 0 {
        return format!("module {}", of_module(o).unwrap_or_default());
    }
    if ffi::PyFunction_Check(o) != 0 {
        let function = o as *mut ffi::PyFunctionObject;
        let name = string((*function).func_qualname).unwrap_or_default();
        return match string((*function).func_module) {
            Some(module) => format!("{}.{}", module, name),
            None => name,
        };
    }
    if ffi::PyCFunction_Check(o) != 0 {
        let function = o as *mut ffi::PyCFunctionObject;
        let name = std::ffi::CStr::from_ptr((*(*function).m_ml).ml_name).to_string_lossy().into_owned();
        let bound = (*function).m_self;
        let owner = match string((*function).m_module) {
            Some(module) => Some(module),
            None if bound.is_null() => None,
            None if ffi::PyModule_Check(bound) != 0 => of_module(bound),
            None if ffi::PyType_Check(bound) != 0 => Some(pyffi::class_name(bound)),
            None => Some(format!("a {}", pyffi::type_name(bound))),
        };
        return match owner {
            Some(owner) => format!("{}.{}", owner, name),
            None => name,
        };
    }
    format!("a {}", pyffi::type_name(o))
}

impl Row {
    /// The row of a tensor that has its Python object, whose sizes and
    /// strides are appended to `numbers`. `same` is the first of the
    /// tensors that are checked that is this tensor. None for what is no
    /// plain tensor.
    pub unsafe fn of(tensor: Obj, same: usize, numbers: &mut Vec<i64>, api: &TcfgApi) -> Option<Row> {
        let (kind, _) = key_of(api, tensor, numbers)?;
        let ty = ffi::Py_TYPE(tensor);
        Some(Row {
            ty: ty as usize,
            type_version: (*ty).tp_version_tag,
            kind,
            forward: (api.tensor_forward_grad)(tensor) != 0,
            same,
        })
    }

    /// Whether a tensor, which may be unboxed, is of the kind of the row.
    /// Its sizes and strides are appended to `numbers`, for the form to
    /// compare.
    pub unsafe fn fits(&self, tensor: Obj, numbers: &mut Vec<i64>, api: &TcfgApi) -> bool {
        key_of(api, tensor, numbers).is_some_and(|(kind, _)| kind == self.kind)
            && ((api.tensor_forward_grad)(tensor) != 0) == self.forward
    }
}

// ----------------------------------------------------------------------
// The pass
// ----------------------------------------------------------------------

/// The graph pass: finds the graphs of a block, once the other passes are
/// done with it. With `export`, it puts `Op::Graph` in front of each.
pub struct Graphs {
    /// A run with fewer nodes than this is no graph.
    pub min_ops: usize,
    pub max_variants: usize,
    pub export: Option<PyObj>,
}

impl Default for Graphs {
    fn default() -> Graphs {
        Graphs { min_ops: 4, max_variants: 8, export: None }
    }
}

impl CfgPass for Graphs {
    fn name(&self) -> String {
        "graphs".to_string()
    }

    fn finishes(&self) -> bool {
        true
    }

    fn run(&self, unit: &mut Unit, cx: &PassCx) -> PyResult<bool> {
        Ok(self.build(unit, cx))
    }
}

/// What an instruction is to a graph: a part of one, or what ends it, with
/// the reason.
enum Role {
    Part(Kind),
    Break(String),
}

/// What the run so far knows of a value one of its instructions defines.
#[derive(Clone, Copy)]
struct Def {
    node: bool,
    /// It depends on the result of a node.
    late: bool,
    /// An integer the graph can compute itself.
    graph: bool,
}

/// The binary operators a graph computes on integers: + - * // %.
const INT_BINARY: [u8; 5] = [0, 10, 5, 2, 6];

impl Graphs {
    fn build(&self, unit: &mut Unit, cx: &PassCx) -> bool {
        let n = unit.insts.len();
        let roles: Vec<Role> = (0..n).map(|i| role(unit, cx, i)).collect();
        // The runs, each from its first node to its last.
        let mut runs: Vec<(usize, usize, String)> = Vec::new();
        let mut at = 0;
        while at < n {
            if matches!(roles[at], Role::Break(_)) {
                at += 1;
                continue;
            }
            let end = stretch(unit, cx, &roles, at);
            let is_node = |i: &usize| matches!(roles[*i], Role::Part(Kind::Node));
            if let (Some(first), Some(last)) = ((at..end).find(is_node), (at..end).rev().find(is_node)) {
                let ends = match roles.get(end) {
                    Some(Role::Break(reason)) => reason.clone(),
                    Some(Role::Part(_)) => "an operation takes what the graph cannot compute".to_string(),
                    None => match &unit.term {
                        Term::Return(_) => "the function returns".to_string(),
                        Term::Jump(_) => "the block ends".to_string(),
                        Term::Br { .. } | Term::BrNull { .. } => "the block ends in a branch".to_string(),
                        Term::Raise { .. } | Term::Reraise(..) => "the block raises".to_string(),
                    },
                };
                runs.push((first, last + 1, ends));
            }
            at = end;
        }
        let origins: Vec<u32> = unit.insts.iter().map(|u| u.origin).collect();
        // What is reported as left out: the instructions that end a run
        // where the block has tensor work on both sides, and the runs that
        // are too short to be graphs.
        let works = |i: &usize| {
            let facts = cx.facts.get(origins[*i] as usize);
            matches!(unit.insts[*i].inst.op, Op::Dispatch(_))
                || facts.is_some_and(|f| matches!(f.kernels, Kernels::One | Kernels::Many))
        };
        let mut left: Vec<Left> = Vec::new();
        if let (Some(first), Some(last)) = ((0..n).find(works), (0..n).rev().find(works)) {
            for i in first..=last {
                if let (Role::Break(reason), true) = (&roles[i], origins[i] != NONE) {
                    left.push(Left { lo: origins[i], hi: origins[i] + 1, reason: reason.clone() });
                }
            }
        }
        let mut found: Vec<(usize, Path)> = Vec::new();
        for (first, end, ends) in runs {
            let nodes = (first..end).filter(|i| matches!(roles[*i], Role::Part(Kind::Node))).count();
            if nodes < self.min_ops {
                let reason = format!("{} operations, fewer than min_ops ({})", nodes, self.min_ops);
                left.push(Left { lo: origins[first], hi: origins[end - 1] + 1, reason });
                continue;
            }
            found.push((first, path(unit, cx, &roles, first, end, ends)));
        }
        unit.left = left;
        let export = self.export.is_some() && !found.is_empty();
        let firsts: Vec<usize> = found.iter().map(|(first, _)| *first).collect();
        for (k, (first, path)) in found.into_iter().enumerate() {
            let name = format!("{} (block {})[{}]", cx.func().qualname, unit.block, path.root.1);
            let site = Arc::new(GraphSite {
                name,
                variants: Shared::new(Vec::new()),
                state: Shared::new(if export { GraphState::Waiting } else { GraphState::Built }),
                version: AtomicU32::new(0),
                exporter: self.export.clone(),
                max_variants: self.max_variants,
                unsure: AtomicU32::new(0),
                exports: AtomicU64::new(0),
                misses: AtomicU64::new(0),
            });
            // With `Graph` in front of this run and of each one before it,
            // the first instruction of the path moves down by as many.
            unit.graphs.push((site, path, if export { first + k + 1 } else { first }));
        }
        if export {
            for (k, first) in firsts.iter().enumerate().rev() {
                let inst = Inst {
                    dst: NONE,
                    op: Op::Graph(unit.graphs[k].0.clone()),
                    bc: unit.insts[*first].inst.bc,
                    kills: Vec::new(),
                };
                unit.insts.insert(*first, UInst { inst, origin: NONE, removable: false });
            }
        }
        export
    }
}

/// An exact number among the constants of the function.
fn const_number(cx: &PassCx, unit: &Unit, v: V, int: bool) -> bool {
    let Some(Op::Const(c)) = unit.def_op(v) else {
        return false;
    };
    let obj = cx.func().consts[*c as usize].as_ptr();
    unsafe { if int { ffi::PyLong_CheckExact(obj) != 0 } else { is_number(obj) } }
}

/// Whether a value was an exact number whenever it was looked at: a constant
/// that is one, a size, or what the profile saw.
fn number(cx: &PassCx, unit: &Unit, v: V) -> bool {
    let sized = matches!(unit.def_op(v), Some(Op::TensorSize { .. } | Op::TensorDim(_)));
    sized || const_number(cx, unit, v, false) || cx.obs.get(v as usize).is_some_and(|o| o.is_number())
}

fn int_like(cx: &PassCx, unit: &Unit, v: V) -> bool {
    let sized = matches!(unit.def_op(v), Some(Op::TensorSize { .. } | Op::TensorDim(_)));
    let seen = matches!(cx.obs.get(v as usize), Some(crate::spec::ValueObs::Int { .. }));
    sized || const_number(cx, unit, v, true) || seen
}

fn facts<'a>(cx: &'a PassCx, unit: &Unit, i: usize) -> Option<&'a InstFacts> {
    cx.facts.get(unit.insts[i].origin as usize)
}

fn role(unit: &Unit, cx: &PassCx, i: usize) -> Role {
    let quiet = facts(cx, unit, i).is_some_and(|f| f.quiet());
    let wrote = facts(cx, unit, i).is_some_and(|f| f.wrote);
    let numbers = |vs: &[V]| vs.iter().all(|v| number(cx, unit, *v));
    let kind = match &unit.insts[i].inst.op {
        Op::Dispatch(d) if d.slot.traits() == 0 && d.slot.len() > 0 && !wrote => Some(Kind::Node),
        Op::Const(_)
        | Op::TensorSize { .. }
        | Op::TensorDim(_)
        | Op::TupleGet(..)
        | Op::IsNone(..)
        | Op::Is(..)
        | Op::IsFunction { .. }
        | Op::BuildTuple(_)
        | Op::CheckBound(..)
        | Op::Guard(_) => Some(Kind::Pure),
        Op::Binary(_, a, b) | Op::Compare(_, a, b) if numbers(&[*a, *b]) => Some(Kind::Pure),
        Op::Unary(_, a) | Op::ToBool(a) if numbers(&[*a]) => Some(Kind::Pure),
        Op::LoadGlobal(_) | Op::LoadDeref(_) | Op::CellGet(..) => Some(Kind::Read),
        Op::LoadAttr(..) | Op::GetItem(..) | Op::Contains(..) | Op::GetLen(_) | Op::Unpack(..) if quiet => {
            Some(Kind::Read)
        }
        Op::Hold(_) | Op::StoreLocal(..) => Some(Kind::Residual),
        _ => None,
    };
    match kind {
        Some(kind) => Role::Part(kind),
        None => Role::Break(why(unit, cx, i)),
    }
}

/// Why an instruction is in no graph.
fn why(unit: &Unit, cx: &PassCx, i: usize) -> String {
    let u = &unit.insts[i];
    let found = facts(cx, unit, i);
    let no_site = || match cx.reasons.get(u.origin as usize).and_then(|r| r.as_ref()) {
        Some(reason) => format!("runs kernels and is no call site: {}", reason),
        None => "runs kernels and is no call site".to_string(),
    };
    let kernels = found.is_some_and(|f| matches!(f.kernels, Kernels::One | Kernels::Many));
    match &u.inst.op {
        Op::Dispatch(d) => {
            let operator = unsafe { d.slot.operator() }.unwrap_or_else(|| "the kernel".to_string());
            let traits = d.slot.traits();
            if traits & SITE_WRITES != 0 || found.is_some_and(|f| f.wrote) {
                format!("{} writes to an argument", operator)
            } else if traits & SITE_RANDOM != 0 {
                format!("{} draws random numbers", operator)
            } else if traits & SITE_SCALAR != 0 {
                format!("{} gives a number read from a tensor", operator)
            } else {
                format!("{} has no call site", operator)
            }
        }
        Op::Call { .. } | Op::CallMethod { .. } | Op::CallEx { .. } => match found.map(|f| &f.callee) {
            Some(CalleeFact::One(code, _)) => {
                let name = unsafe {
                    let name = ffi::PyObject_GetAttrString(code.as_ptr(), c"co_qualname".as_ptr());
                    if name.is_null() {
                        ffi::PyErr_Clear();
                        "a function".to_string()
                    } else {
                        let text = pyffi::str_to_string(name);
                        ffi::Py_DECREF(name);
                        text
                    }
                };
                format!("calls {}", name)
            }
            Some(CalleeFact::Many) => "calls more than one function".to_string(),
            _ if kernels => no_site(),
            _ if found.is_some_and(|f| f.frames) => "calls Python code".to_string(),
            _ if found.is_some_and(|f| f.runs > 0) => "a call that runs no kernel".to_string(),
            _ => "a call that was not profiled".to_string(),
        },
        Op::StoreAttr { .. }
        | Op::SetItem { .. }
        | Op::DelItem(..)
        | Op::StoreGlobal(..)
        | Op::StoreDeref(..)
        | Op::CellSet(..) => {
            if found.is_some_and(|f| f.wrote) {
                "writes to a tensor".to_string()
            } else if kernels {
                no_site()
            } else {
                "stores into an object".to_string()
            }
        }
        op => {
            let text = cx.func().op_repr(op);
            let name = text.split_whitespace().next().unwrap_or("it");
            if kernels {
                no_site()
            } else if found.is_some_and(|f| f.frames) {
                format!("{} enters Python code", name)
            } else if found.is_some_and(|f| f.raised) {
                format!("{} raised", name)
            } else {
                format!("{} is no part of a graph", name)
            }
        }
    }
}

/// What an instruction of a run reads, for where it stands in the run: the
/// operands of a site, not what its fallback would take.
pub fn reads(op: &Op, mut f: impl FnMut(V)) {
    match op {
        Op::Dispatch(d) => d.operands.iter().for_each(|v| f(*v)),
        Op::Hold(_) | Op::StoreLocal(..) => {}
        op => op.for_each_use(f),
    }
}

/// Whether the graph can compute what a late instruction gives: a size of a
/// tensor of its own, or arithmetic on such sizes and other integers.
fn computes(op: &Op, defs: &HashMap<V, Def>, cx: &PassCx, unit: &Unit) -> bool {
    let fits = |v: &V| match defs.get(v) {
        Some(def) if def.late => def.graph,
        _ => int_like(cx, unit, *v),
    };
    match op {
        Op::TensorSize { t, .. } | Op::TensorDim(t) => defs.get(t).is_some_and(|def| def.node),
        Op::Binary(k, a, b) => INT_BINARY.contains(k) && fits(a) && fits(b),
        Op::Unary(UnaryKind::Neg, a) => fits(a),
        Op::BuildTuple(vs) => vs.iter().all(fits),
        _ => false,
    }
}

/// The definitions of a run as far as it has got, with the instruction at
/// `i` added. False if the instruction cannot be in the run: a node that
/// takes something that was computed from a node and that the graph cannot
/// compute itself, or a guard on such a thing.
fn define(unit: &Unit, cx: &PassCx, roles: &[Role], defs: &mut HashMap<V, Def>, i: usize) -> bool {
    let inst = &unit.insts[i].inst;
    let node = matches!(roles[i], Role::Part(Kind::Node));
    let mut late = node;
    let mut fits = true;
    reads(&inst.op, |v| {
        if let Some(def) = defs.get(&v) {
            late |= def.late;
            fits &= !def.late || def.node || def.graph;
        }
    });
    let guarded = match &inst.op {
        Op::Guard(g) => defs.get(&g.cond.value()).is_some_and(|def| def.late && !def.node),
        _ => false,
    };
    if (node && !fits) || guarded {
        return false;
    }
    if inst.dst != NONE {
        let graph = late && !node && computes(&inst.op, defs, cx, unit);
        defs.insert(inst.dst, Def { node, late, graph });
    }
    true
}

/// How far a run that starts at `at` goes: up to the first instruction that
/// ends it.
fn stretch(unit: &Unit, cx: &PassCx, roles: &[Role], at: usize) -> usize {
    let mut defs: HashMap<V, Def> = HashMap::new();
    let mut i = at;
    while i < unit.insts.len() {
        if matches!(roles[i], Role::Break(_)) || !define(unit, cx, roles, &mut defs, i) {
            break;
        }
        i += 1;
    }
    // An instruction that cannot start a run either would never be passed.
    i.max(at + 1)
}

/// Whether nothing is lost when an instruction that depends on a node does
/// not run because nothing needs its result.
fn droppable(op: &Op, kind: Kind) -> bool {
    kind == Kind::Read
        || op.is_pure()
        || matches!(op, Op::Binary(0 | 5 | 10, ..) | Op::Compare(..) | Op::Unary(UnaryKind::Neg, _))
}

/// The path of the instructions first..end of a unit, which start and end
/// with a node.
fn path(unit: &Unit, cx: &PassCx, roles: &[Role], first: usize, end: usize, ends: String) -> Path {
    let cf = cx.cf;
    let block = unit.block;
    let mut defs: HashMap<V, Def> = HashMap::new();
    let mut body: Vec<PathInst> = Vec::with_capacity(end - first);
    let mut operators = Vec::new();
    for i in first..end {
        define(unit, cx, roles, &mut defs, i);
        let inst = &unit.insts[i].inst;
        let Role::Part(kind) = &roles[i] else {
            unreachable!("a run has no instruction that ends it");
        };
        if let Op::Dispatch(d) = &inst.op {
            operators.push(unsafe { d.slot.operator() }.unwrap_or_else(|| "?".to_string()));
        }
        body.push(PathInst {
            op: inst.op.clone(),
            dst: inst.dst,
            kills: Vec::new(),
            kind: *kind,
            bc: inst.bc,
            stage: Stage::Before,
            in_graph: false,
        });
    }
    let late = |inst: &PathInst| {
        let mut late = inst.kind == Kind::Node;
        reads(&inst.op, |v| late |= defs.get(&v).is_some_and(|def| def.late));
        late
    };
    // What is read behind the path: by the rest of the block, by the blocks
    // after it, and by the generic code a guard behind it resumes.
    let n = cx.func().nvalues as usize;
    let mut need = BitSet::new(n);
    for u in &unit.insts[end..] {
        if !matches!(u.inst.op, Op::Hold(_)) {
            u.inst.op.for_each_use(|v| {
                need.insert(v);
            });
        }
        if let Op::Guard(g) = &u.inst.op {
            need.union_with(&cf.read_at(block, g.resume as usize));
        }
    }
    unit.term.for_each_use(|v| {
        need.insert(v);
    });
    need.union_with(&cf.read_out(block));
    let mut outputs: Vec<V> = Vec::new();
    let mut exits: Vec<GuardCond> = Vec::new();
    for inst in body.iter_mut().rev() {
        let needed = inst.dst != NONE && need.contains(inst.dst);
        let computed = defs.get(&inst.dst).is_some_and(|def| def.graph);
        inst.stage = match (&inst.op, inst.kind) {
            (_, Kind::Node) => {
                if needed {
                    outputs.push(inst.dst);
                }
                Stage::Inside
            }
            (_, Kind::Residual) => Stage::Skipped,
            _ if !late(inst) => Stage::Before,
            // A guard on what a node returns looks at what the graph says
            // of it.
            (Op::Guard(g), _) => {
                exits.push(g.cond.clone());
                Stage::Inside
            }
            // An integer the graph computes is one it can give.
            _ if needed && computed => {
                outputs.push(inst.dst);
                Stage::Inside
            }
            (op, kind) if needed || !droppable(op, kind) => {
                op.for_each_use(|v| {
                    need.insert(v);
                });
                Stage::After
            }
            _ => Stage::Skipped,
        };
    }
    outputs.reverse();
    exits.reverse();
    // The integers the graph computes: what its nodes take of them, what it
    // gives of them, and what those are computed from.
    let mut wanted: Vec<V> = Vec::new();
    for inst in body.iter().rev() {
        let given = inst.kind != Kind::Node && outputs.contains(&inst.dst);
        if given {
            wanted.push(inst.dst);
        }
        if inst.kind == Kind::Node || wanted.contains(&inst.dst) {
            reads(&inst.op, |v| {
                if defs.get(&v).is_some_and(|def| def.late && !def.node) {
                    wanted.push(v);
                }
            });
        }
    }
    for inst in body.iter_mut() {
        inst.in_graph = inst.dst != NONE && wanted.contains(&inst.dst);
        if inst.in_graph && inst.stage == Stage::Skipped {
            inst.stage = Stage::Inside;
        }
    }
    let made: Vec<V> = body.iter().filter(|inst| inst.dst != NONE).map(|inst| inst.dst).collect();
    let mut inputs: Vec<V> = Vec::new();
    let mut feeds: Vec<V> = Vec::new();
    for inst in &body {
        inst.op.for_each_use(|v| {
            if v != UNDEF && !made.contains(&v) && !inputs.contains(&v) && inst.kind != Kind::Residual {
                inputs.push(v);
            }
        });
        if inst.kind == Kind::Node || inst.in_graph {
            reads(&inst.op, |v| {
                let own = defs.get(&v).is_some_and(|def| def.node) || wanted.contains(&v);
                if !own && !feeds.contains(&v) {
                    feeds.push(v);
                }
            });
        }
    }
    // The variables the instructions bind are bound at the end: each to what
    // it was bound to last, if that has a register then.
    let given = |v: V| {
        !made.contains(&v)
            || outputs.contains(&v)
            || body.iter().any(|inst| inst.dst == v && matches!(inst.stage, Stage::Before | Stage::After))
    };
    let mut bound: Vec<(u32, V)> = Vec::new();
    for inst in &body {
        if let Op::StoreLocal(slot, v) = &inst.op {
            bound.retain(|(other, _)| other != slot);
            bound.push((*slot, if given(*v) { *v } else { UNDEF }));
        }
    }
    Path {
        root: (block, unit.insts[first].origin as usize),
        land: (block, unit.insts[end - 1].origin as usize + 1),
        inputs,
        body,
        feeds,
        outputs,
        exits,
        bound,
        operators,
        ends,
        exported: Shared::new(Vec::new()),
        hits: AtomicU64::new(0),
        misses: AtomicU64::new(0),
    }
}
