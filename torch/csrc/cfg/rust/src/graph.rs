//! Graphs: the tensor work of a function along a path through its blocks,
//! as a unit of its own.
//!
//! A graph starts at an operation of a specialized block: a kernel call
//! site, or a call that runs kernels and is none. The graph pass finds the
//! runs of operations of a block once the other passes are done with it,
//! with what sits between them (reads of globals and attributes, constants,
//! arithmetic on sizes), and makes a `GraphSite` for the first operation of
//! each. From there a `Path` is taken down while the function runs
//! (`Compiled::record`): the instructions as they came, through the
//! branches as they went and the loops as often as they went round, up to
//! the first instruction that can be no part of one. Each way a branch went
//! is a check of the path (`Assert`). The VM runs a graph as it runs the
//! blocks, instruction by instruction and each node through its site, on
//! eager's autograd tape; the path is what `torch.vm.regions` reports.
//!
//! When the list of passes has a pass over graphs, a path is also what such
//! a pass is given, as a `torch.fx.GraphModule`: from the run that was
//! taken down the graph is built in Python, the passes are applied, and
//! what they return is kept with the conditions it was made under
//! (`Exported`). A later call that comes to the first operation runs the
//! instructions of the path that depend on no node, with their checks, on
//! registers of the path's own; if they hold and the values are those of an
//! exported form, the callable runs in place of the nodes, the rest of the
//! instructions run on what it returned, and the function goes on where
//! the path ended. A call that fails a check has changed nothing: the
//! function goes on at the first operation.

use crate::analysis::BitSet;
use crate::capi::{self, TcfgApi, TcfgSite, TcfgTensorInfo, SITE_RANDOM, SITE_SCALAR, SITE_WRITES};
use crate::ir::*;
use crate::pyffi::{self, new_ref, xdecref, Obj, PyObj, Shared};
use crate::shapes::{key_of, ShapeGuard, KIND_BYTES};
use crate::spec::{CalleeFact, CfgPass, InstFacts, Kernels, PassCx, Unit, ValueObs};
use crate::vm::Compiled;
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
    /// An operation of the graph: a kernel call site, or a call of a C
    /// function, a method of a tensor or a subscript of one that runs
    /// kernels and nothing else and is no call site (a list of tensors
    /// among its arguments, an index).
    Node,
    /// Makes a value from its operands alone: a constant, a size, arithmetic
    /// on numbers, a check.
    Pure,
    /// Reads a global, a cell, or an attribute or item of an object, which
    /// entered no frame and ran no kernel when it was profiled.
    Read,
    /// Keeps values alive, binds a variable of the frame, or works on a
    /// list, a dict or an iterator that the path made itself.
    Residual,
    /// What the path took for granted where it went on (`Assert`).
    Check,
    /// A store into an object from outside the path: an attribute, an item
    /// of a list or a dict, an append. It is done when the path lands, in
    /// the order of the function, behind every check.
    Effect,
}

/// What a path takes for granted of a value where it goes on.
#[derive(Clone)]
pub enum Assert {
    /// A branch went this way: its condition is this bool.
    Truth(PathValue, bool),
    /// An iterator was exhausted (true), or gave an item.
    Null(PathValue, bool),
    /// A guard of the code the path was taken down from.
    Guard(GuardCond),
    /// A Python function that runs as this compiled function when it is
    /// called: what a call that the path goes into enters.
    Enters(PathValue, *const Compiled),
    /// What an `__iter__` returned is an iterator, what a `__len__`
    /// returned is a count: the interpreter raises for anything else.
    Iterator(PathValue),
    Count(PathValue),
    /// The object has no attribute of the name (`absent`): reading it
    /// comes down to the `__getattr__` of its class.
    Absent(PathValue, PyObj),
    /// The class of the object has nothing under the name: no `__len__`
    /// and no `__length_hint__`, which `list` and `tuple` would ask.
    Lacks(PathValue, PyObj),
    /// What a call hands on with `*`: exactly of this class, a tuple or a
    /// list, with this many items. With `**`: an exact dict with these
    /// names, in this order.
    Sequence(PathValue, usize, usize),
    Keys(PathValue, Vec<PyObj>),
    /// The number is this one: the place of an item that the path takes
    /// out of a list it filled itself.
    Equals(PathValue, PyObj),
    /// What an `__instancecheck__` returned is True or False.
    Flag(PathValue),
    /// The value is this very object: a function, a class or a module
    /// that something computed behind the callable is computed with, and
    /// is to come out the same.
    Is(PathValue, PyObj),
    /// Two values are one object: a read behind a store of the path, of
    /// the same object by another way, is given what was stored.
    Same(PathValue, PathValue),
    /// What a read behind a store of the path goes through is not what the
    /// store changes: the first value (the list or dict that is read, or,
    /// with the flag, the object an attribute is read of) against the
    /// object that was stored into and where it keeps what was stored
    /// (`apart`).
    Apart(PathValue, bool, PathValue, Keeps),
}

impl Assert {
    pub fn value(&self) -> PathValue {
        match self {
            Assert::Truth(v, _)
            | Assert::Null(v, _)
            | Assert::Enters(v, _)
            | Assert::Iterator(v)
            | Assert::Count(v)
            | Assert::Absent(v, _)
            | Assert::Lacks(v, _)
            | Assert::Sequence(v, ..)
            | Assert::Keys(v, _)
            | Assert::Equals(v, _)
            | Assert::Flag(v)
            | Assert::Is(v, _)
            | Assert::Same(v, _)
            | Assert::Apart(v, ..) => *v,
            Assert::Guard(cond) => cond.value(),
        }
    }
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
    /// Where the path lands, behind everything that can fail: an effect.
    Landing,
}

pub struct PathInst {
    pub op: Op,
    pub dst: PathValue,
    pub kind: Kind,
    pub bc: u32,
    pub stage: Stage,
    /// A value the exported graph computes itself that is no tensor: a size
    /// of one of its tensors, arithmetic on such sizes, a list of them.
    pub in_graph: bool,
    /// For `Kind::Check`.
    pub assert: Option<Assert>,
    /// Which function it is an instruction of (`Path::owners`): the one
    /// whose constants and names it refers to.
    pub owner: u32,
}

/// What a check looks at and a callable is handed: a value, or one of the
/// items of a list or tuple that a value is.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Place {
    pub value: PathValue,
    /// Which item, or NONE for the value itself.
    pub item: u32,
}

impl Place {
    pub fn of(value: PathValue) -> Place {
        Place { value, item: NONE }
    }

    /// What is there: borrowed, and possibly an unboxed tensor; null if
    /// nothing is.
    pub unsafe fn at(&self, regs: &[Obj]) -> Obj {
        let held = regs[self.value as usize];
        if self.item == NONE || held.is_null() {
            return held;
        }
        let sequence = !capi::is_unboxed(held)
            && (ffi::PyList_CheckExact(held) != 0 || ffi::PyTuple_CheckExact(held) != 0)
            && (self.item as isize) < ffi::PyObject_Size(held);
        if sequence { sequence_item(held, self.item as isize) } else { null_mut() }
    }

    /// A name for it: that of the value of the function it stands for.
    pub unsafe fn text(&self, path: &Path) -> String {
        let value = path.label(self.value);
        match self.item {
            NONE => value,
            item => format!("{}[{}]", value, item),
        }
    }
}

/// What a call has to find for an exported form to be run for it.
pub struct Check {
    pub target: Place,
    pub cond: Cond,
    /// What it is of is computed behind the callable, from the sizes of
    /// what the graph gives (`Path::constants`): the check is made there.
    pub late: bool,
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
    /// An object that has this very value under a name, and gives it out
    /// without running Python code: the module a function is called
    /// through.
    Has(PyObj, PyObj),
    /// A list or a tuple, by its exact type, of this many items: one whose
    /// items the graph takes.
    Sequence(usize, usize),
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
    pub args: Vec<Place>,
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
    pub varying: Vec<Place>,
    pub last: Shared<Vec<PyObj>>,
    pub repeats: AtomicU32,
    /// See `Described::aliases`. Empty for a path that writes to nothing
    /// from outside.
    pub aliases: Vec<(u32, i64)>,
    /// See `Described::bumps`: what is made of the graph may write to a
    /// tensor fewer times than the operations do (AOTAutograd writes an
    /// input back once), and the counters are moved on to where the
    /// operations leave them.
    pub bumps: Vec<(u32, u64)>,
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
    /// Where the function goes on behind it: the block and the generic
    /// instruction, and the code and the instruction of it to go on with.
    /// Without the latter, the block is entered as from a block before it.
    /// The code is a version of a block of the function the path is of, or
    /// its generic code: the function keeps both for as long as it lives.
    pub land: (usize, usize),
    pub resume: Option<(*const BlockCode, usize)>,
    /// How many registers it runs on. A path that was taken down numbers
    /// its values itself, from 1; the one the graph pass makes of a run of
    /// a block (`GraphSite::seed`), which is only reported, as the function
    /// does.
    pub nregs: usize,
    /// The values of the function it takes from before it, in the order it
    /// first reads them, each with the value of its own that it is.
    pub inputs: Vec<(V, PathValue)>,
    pub body: Vec<PathInst>,
    /// The values its nodes take that the graph does not compute: inputs,
    /// and what the instructions in front of the callable read. In the order
    /// they are first taken.
    pub feeds: Vec<PathValue>,
    /// What is read behind it of what the graph computes, in the order it
    /// is computed: results of nodes, and integers. A callable returns
    /// these, and then one integer per entry of `exits`.
    pub outputs: Vec<PathValue>,
    /// The guards of the code on results of nodes that a callable does not
    /// return: it returns what each looks at instead, the rank of the tensor
    /// or the size of the dimension.
    pub exits: Vec<GuardCond>,
    /// Where it ends: the values of the function that hold one of its
    /// values then and are read from there on, and those that hold nothing
    /// anymore.
    pub writes: Vec<(PathValue, V)>,
    pub clears: Vec<V>,
    /// The variables of the frame its instructions bind, each with the value
    /// it holds at the end, or UNDEF where a callable does not give it.
    pub bound: Vec<(u32, PathValue)>,
    /// The operators of its nodes, as the dispatcher names them.
    pub operators: Vec<String>,
    /// Why it ends where it does: what the instruction behind it is, or how
    /// the block ends.
    pub ends: String,
    /// Which instructions it is, of which code: two runs with the same
    /// numbers are the same path.
    pub shape: Vec<u32>,
    /// The functions it goes through: the one it starts and ends in first,
    /// then those its calls enter, in the order they are first entered. The
    /// function it is a path of keeps them all for as long as it lives.
    pub owners: Vec<*const Compiled>,
    /// Per value: the function and the value of it that it stood for when
    /// it was made (NONE for one that is no value of a function).
    pub source: Vec<(u32, V)>,
    /// How many branches and loop tests it goes through.
    pub branches: usize,
    /// How many calls of Python functions it goes into.
    pub calls: usize,
    /// The block and the instruction of its first operation, where that
    /// is one of the function it starts in: what the graph is named by.
    pub first: Option<(usize, u32)>,
    /// It ended at a block that had not been profiled yet: the next run
    /// gets further.
    pub provisional: bool,
    /// It has stores into objects from outside (`Kind::Effect`), which are
    /// done when it lands.
    pub effects: bool,
    /// An operation of it writes to a tensor from outside (or draws random
    /// numbers): the callable cannot be run a second time for a call, and
    /// what it did cannot be taken back.
    pub mutates: bool,
    /// See `Trace::pinned`.
    pub pinned: Vec<PathValue>,
    /// What an operation of the graph takes that is computed behind the
    /// callable, from the sizes of what the graph gives: the same in
    /// every call a form is for (`Def::fixed`), so a constant of the
    /// graph, which the form checks where it is computed.
    pub constants: Vec<PathValue>,
    pub exported: Shared<Vec<Arc<Exported>>>,
    /// Register files for the calls that run it, kept from one to the next.
    pub registers: Shared<Vec<Vec<Obj>>>,
}

// The registers are empty between calls, and only touched with the GIL held.
unsafe impl Send for Path {}
unsafe impl Sync for Path {}

impl Path {
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
    /// The path from its first operation has this many operations, fewer
    /// than a graph has.
    Small(usize),
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

/// Whether `TORCH_VM_GRAPH_DEBUG` asks for a line on stderr per path that
/// is taken down: how long it is, where it ends and why, and why a run was
/// not exported.
pub fn debug() -> bool {
    static ASKED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ASKED.get_or_init(|| std::env::var_os("TORCH_VM_GRAPH_DEBUG").is_some_and(|v| v != "0"))
}

/// Whether `TORCH_VM_GRAPH_TIMES` asks for the time of the calls that run
/// an exported form to be taken: what runs in front of the callable, the
/// callable, and what runs behind it (`GraphSite::times`).
pub fn timed() -> bool {
    static ASKED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ASKED.get_or_init(|| std::env::var_os("TORCH_VM_GRAPH_TIMES").is_some_and(|v| v != "0"))
}

/// How many times a graph is taken down again because its path ended at a
/// block that had not been profiled.
const MAX_PROVISIONAL: u32 = 3;

/// How many operations with several results a graph ends its paths in front
/// of because what they gave was needed whole (`GraphSite::shunned`).
const MAX_SHUNNED: usize = 8;

/// A graph: the paths from one operation of a block on.
pub struct GraphSite {
    /// The function, the block, and the generic instructions of the block
    /// from the first operation to the last of its run there.
    pub name: String,
    /// The block, and the generic instruction of it, that it starts at:
    /// the first operation of a run, or (`entry`) the first instruction of
    /// the function, from where a path has the lists the function makes
    /// before its first operation.
    pub root: (usize, usize),
    pub entry: bool,
    /// What the graph pass made of the run of the block the graph starts
    /// with: reported until a path was taken down.
    pub seed: Shared<Arc<Path>>,
    /// The paths that were taken down, the one that ran last first.
    pub variants: Shared<Vec<Arc<Path>>>,
    pub state: Shared<GraphState>,
    /// What makes an exported form of a description (`torch.vm._export`),
    /// if the list of passes has a pass over graphs; the fewest operations
    /// a graph has; and how many forms one may have.
    pub exporter: Option<PyObj>,
    pub min_ops: usize,
    pub max_variants: usize,
    unsure: AtomicU32,
    provisional: AtomicU32,
    /// How many times it was handed to the exporter, and how many calls
    /// ran what was made of it and how many ran its instructions.
    pub exports: AtomicU64,
    pub hits: AtomicU64,
    pub misses: AtomicU64,
    /// The graph whose path went through the first operation of this one:
    /// while that one is there, this one is part of it.
    pub within: Shared<std::sync::Weak<GraphSite>>,
    /// The call of the function that last came here and ran the
    /// instructions: it does not try again.
    pub missed: AtomicU64,
    /// With `TORCH_VM_GRAPH_TIMES`: the nanoseconds the calls that ran an
    /// exported form spent in front of its callable, in it and behind it,
    /// and how many such calls there were.
    pub times: [AtomicU64; 4],
    /// Where graphs are not exported: a graph that starts further on has
    /// asked for another path of this one (`asked`), and how many times.
    again: AtomicBool,
    asks: AtomicU32,
    /// Operations, each by its function and its place in the code of it,
    /// that a path of this graph took and that turned out to be none of
    /// a graph when they had run, with the reason: one with several
    /// results whose value was needed whole behind the path (a graph gives
    /// back the items), one that changed the sizes of a tensor from
    /// outside or whether it requires a gradient (a form is for the
    /// tensors as a call hands them over). The path was taken down again
    /// and ends in front of such an operation.
    pub shunned: Shared<Vec<(*const crate::vm::Compiled, u32, &'static str)>>,
}

impl GraphSite {
    pub fn new(name: String, seed: Path, entry: bool, graphs: &Graphs) -> GraphSite {
        let exported = graphs.export.is_some();
        GraphSite {
            name,
            root: seed.root,
            entry,
            again: AtomicBool::new(false),
            asks: AtomicU32::new(0),
            seed: Shared::new(Arc::new(seed)),
            variants: Shared::new(Vec::new()),
            state: Shared::new(if exported { GraphState::Waiting } else { GraphState::Built }),
            exporter: graphs.export.clone(),
            min_ops: graphs.min_ops,
            max_variants: graphs.max_variants,
            unsure: AtomicU32::new(0),
            provisional: AtomicU32::new(0),
            exports: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            within: Shared::new(std::sync::Weak::new()),
            missed: AtomicU64::new(u64::MAX),
            times: Default::default(),
            shunned: Shared::new(Vec::new()),
        }
    }

    /// Has the paths of the graph end in front of an operation from now
    /// on; false if they do already, or there are too many such.
    pub fn shun(&self, place: (*const crate::vm::Compiled, u32), why: &'static str) -> bool {
        let mut shunned = self.shunned.borrow_mut();
        if shunned.len() >= MAX_SHUNNED || shunned.iter().any(|(of, at, _)| (*of, *at) == place) {
            return false;
        }
        shunned.push((place.0, place.1, why));
        true
    }

    /// Why the paths of the graph end in front of an operation, if they do.
    pub fn shuns(&self, place: (*const crate::vm::Compiled, u32)) -> Option<&'static str> {
        self.shunned.borrow().iter().find(|(of, at, _)| (*of, *at) == place).map(|(_, _, why)| *why)
    }

    /// The path that is reported: the one that ran last, or the run of the
    /// block before any was taken down.
    pub fn path(&self) -> Arc<Path> {
        self.variants.borrow().first().cloned().unwrap_or_else(|| self.seed.borrow().clone())
    }

    pub fn paths(&self) -> Vec<Arc<Path>> {
        self.variants.borrow().clone()
    }

    pub fn describe(&self) -> String {
        format!("{}: {} operations", self.name, self.path().operators.len())
    }

    /// How many forms with a callable, and how many without, its paths have.
    pub fn forms(&self) -> (usize, usize) {
        let paths = self.variants.borrow();
        (paths.iter().map(|path| path.compiled()).sum(), paths.iter().map(|path| path.closed()).sum())
    }

    /// Whether a call that no path or no exported form is for may be taken
    /// down: for a graph that is exported, while there are fewer than
    /// `max_variants` forms with a callable and no more than that without
    /// one; for one that is not, until it has its path.
    pub fn records(&self) -> bool {
        match &*self.state.borrow() {
            GraphState::Waiting => {
                let (compiled, closed) = self.forms();
                compiled < self.max_variants && closed <= self.max_variants
            }
            GraphState::Built => {
                self.again.load(Ordering::Relaxed)
                    || self.variants.borrow().first().is_none_or(|path| path.provisional)
            }
            _ => false,
        }
    }

    /// For a graph that is not exported and has its path: has it take
    /// another one down the next time the function comes to it, if it has
    /// not been asked twice already. A graph that starts further on asks,
    /// when a call comes to it on a way that the path of this one does not
    /// go: the two would otherwise both have the operations behind it.
    pub fn asked(&self) -> bool {
        let has = matches!(*self.state.borrow(), GraphState::Built)
            && self.variants.borrow().first().is_some_and(|path| !path.provisional);
        if !has || self.asks.load(Ordering::Relaxed) >= 2 {
            return false;
        }
        self.asks.fetch_add(1, Ordering::Relaxed);
        self.again.store(true, Ordering::Relaxed);
        true
    }

    /// For what is no graph because it has too few operations: how many.
    /// The path that was taken down from here, or, before one was, the
    /// run of the block.
    pub fn short(&self) -> Option<usize> {
        match &*self.state.borrow() {
            GraphState::Small(count) => Some(*count),
            GraphState::Built | GraphState::Waiting if self.variants.borrow().is_empty() => {
                let count = self.seed.borrow().operators.len();
                (count < self.min_ops).then_some(count)
            }
            _ => None,
        }
    }

    /// The graph, if there is one, that this one is part of. It stays part
    /// of one that was given up: what was made of the larger graph failed,
    /// and is not tried again on a part of it.
    pub fn outer(&self) -> Option<Arc<GraphSite>> {
        let outer = self.within.borrow().upgrade()?;
        let there = !matches!(*outer.state.borrow(), GraphState::Small(_));
        there.then_some(outer)
    }

    /// Whether `Op::Graph` has anything to do for it.
    pub fn runs(&self) -> bool {
        self.outer().is_none() && (self.records() || self.forms() != (0, 0))
    }

    pub fn give_up(&self, reason: String) {
        *self.state.borrow_mut() = GraphState::GivenUp(reason);
    }

    /// Takes a path that was taken down, with the form that was made of the
    /// run if one was: a path it has already gets the form. A path that is
    /// not final (see `Path::provisional`) is taken a few times at most.
    pub fn install(&self, path: Path, form: Option<Arc<Exported>>) {
        self.again.store(false, Ordering::Relaxed);
        if path.provisional && self.provisional.fetch_add(1, Ordering::Relaxed) + 1 < MAX_PROVISIONAL {
            return;
        }
        if path.operators.len() < self.min_ops {
            let small = matches!(*self.state.borrow(), GraphState::Built | GraphState::Waiting)
                && self.variants.borrow().is_empty();
            if small {
                *self.state.borrow_mut() = GraphState::Small(path.operators.len());
            }
            return;
        }
        let mut paths = self.variants.borrow_mut();
        let known = paths.iter().position(|other| other.shape == path.shape && other.provisional == path.provisional);
        let at = known.unwrap_or_else(|| {
            // A path that was not final makes way for the ones that are.
            paths.retain(|other| !other.provisional);
            paths.insert(0, Arc::new(path));
            0
        });
        if let Some(form) = form {
            paths[at].exported.borrow_mut().insert(0, form);
        }
        let path = paths.remove(at);
        paths.insert(0, path);
    }

    /// Whether the graph is being taken down again because its path ended
    /// where a later one will go on: the graphs that start behind it, in
    /// the function and in what it calls, wait for that.
    pub fn pending(&self) -> bool {
        let taken = self.provisional.load(Ordering::Relaxed);
        let open = matches!(*self.state.borrow(), GraphState::Built | GraphState::Waiting);
        open && taken > 0 && taken < MAX_PROVISIONAL && self.variants.borrow().is_empty()
    }

    /// An instruction that would run out of its place for an exported form
    /// could not be shown to run no Python code. Where that goes on, the
    /// forms are dropped and the graph is left to the VM.
    pub fn unsure(&self, what: &str) {
        if self.unsure.fetch_add(1, Ordering::Relaxed) + 1 >= MAX_UNSURE {
            for path in self.variants.borrow().iter() {
                path.exported.borrow_mut().clear();
            }
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
        shape: &[u32],
        world: (u64, u64, u64, u64),
        checks: &[Check],
        sizes: &[i64],
    ) -> Option<Vec<usize>> {
        let paths = self.variants.borrow();
        let path = paths.iter().find(|path| path.shape == shape)?;
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

/// Whether an operator is one of `UPDATES_STATISTICS`.
pub fn normalization(name: &str) -> bool {
    UPDATES_STATISTICS.iter().any(|(op, _)| *op == overload_of(name).0)
}

/// Whether the call of a site, as `site_export` describes it and with the
/// operands it had, is one of `UPDATES_STATISTICS` that updates statistics.
pub unsafe fn updates_statistics(exported: Obj, operands: &[Obj]) -> bool {
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

/// `raised_text` of an exception that stays raised.
pub unsafe fn raised_text_kept() -> String {
    let raised = ffi::PyErr_GetRaisedException();
    if raised.is_null() {
        return "unknown error".to_string();
    }
    ffi::Py_INCREF(raised);
    ffi::PyErr_SetRaisedException(raised);
    let text = raised_text();
    ffi::PyErr_SetRaisedException(raised);
    text
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
    if ffi::Py_TYPE(a) != ffi::Py_TYPE(b) {
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
            let same = !x.is_null() && !y.is_null() && same_value(x, y);
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
    (0..ffi::PyObject_Size(a)).all(|i| same_value(sequence_item(a, i), sequence_item(b, i)))
}

/// What a check keeps of a value to tell the value of a later call from
/// it (`same_value`), and what a form has as a constant: the value itself
/// or, for what can be changed in place (a list, and a tuple or a slice
/// that holds one), a copy that nothing else has. A list that lives
/// outside the function is the same object at every call whatever is in
/// it.
pub unsafe fn frozen(o: Obj) -> PyObj {
    if capi::is_unboxed(o) {
        return PyObj::from_borrowed(o);
    }
    if ffi::PySlice_Check(o) != 0 {
        let parts: Vec<Obj> = [c"start", c"stop", c"step"].iter().map(|name| ffi::PyObject_GetAttrString(o, name.as_ptr())).collect();
        let kept: Vec<PyObj> = parts.iter().map(|part| if part.is_null() { PyObj::from_borrowed(ffi::Py_None()) } else { frozen(*part) }).collect();
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
    let items: Vec<PyObj> = (0..count).map(|i| frozen(sequence_item(o, i))).collect();
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

pub unsafe fn sequence_item(s: Obj, i: isize) -> Obj {
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
            let fine = !part.is_null() && checkable(part);
            xdecref(part);
            fine
        });
    }
    let sequence = ffi::PyTuple_CheckExact(o) != 0 || ffi::PyList_CheckExact(o) != 0;
    sequence && (0..ffi::PyObject_Size(o)).all(|i| checkable(sequence_item(o, i)))
}

/// What `len` may be given by a `__len__`: an exact int that is not
/// negative.
pub unsafe fn is_count(o: Obj) -> bool {
    if capi::is_unboxed(o) || ffi::PyLong_CheckExact(o) == 0 {
        return false;
    }
    let count = ffi::PyLong_AsSsize_t(o);
    if count == -1 && !ffi::PyErr_Occurred().is_null() {
        ffi::PyErr_Clear();
        return false;
    }
    count >= 0
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
pub unsafe fn is_plain_key(o: Obj) -> bool {
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

/// The code of the functions that the weak containers give their weak
/// references to call when what they refer to ends, each with what the
/// container keeps by the reference: 0 for a dictionary whose keys are the
/// references (`weakref.WeakKeyDictionary`, torch's `WeakIdKeyDictionary`),
/// 1 for one whose values are (`weakref.WeakValueDictionary`), 2 for a set
/// (`weakref.WeakSet`). Such a function takes the entry out of its
/// container and does nothing else. Found by looking (the functions are
/// defined inside `__init__` of the classes), without running anything.
#[cfg(not(Py_3_13))]
unsafe fn weak_containers() -> Vec<(usize, u8)> {
    const CLASSES: [(&std::ffi::CStr, &std::ffi::CStr, u8); 4] = [
        (c"torch.utils.weak", c"WeakIdKeyDictionary", 0),
        (c"weakref", c"WeakKeyDictionary", 0),
        (c"weakref", c"WeakValueDictionary", 1),
        (c"_weakrefset", c"WeakSet", 2),
    ];
    static FOUND: std::sync::OnceLock<Vec<(usize, u8)>> = std::sync::OnceLock::new();
    if let Some(found) = FOUND.get() {
        return found.clone();
    }
    let init = PyObj::from_owned(ffi::PyUnicode_InternFromString(c"__init__".as_ptr()));
    let (mut found, mut all) = (Vec::new(), true);
    for (module, class, kind) in CLASSES {
        let module = ffi::PyDict_GetItemString(ffi::PyImport_GetModuleDict(), module.as_ptr());
        let class = if module.is_null() || ffi::PyModule_Check(module) == 0 {
            null_mut()
        } else {
            ffi::PyDict_GetItemString(ffi::PyModule_GetDict(module), class.as_ptr())
        };
        let made = if class.is_null() || ffi::PyType_Check(class) == 0 {
            null_mut()
        } else {
            pyffi::_PyType_Lookup(class as *mut ffi::PyTypeObject, init.as_ptr())
        };
        if made.is_null() || ffi::PyFunction_Check(made) == 0 {
            all = false;
            continue;
        }
        let consts = ffi::PyObject_GetAttrString(ffi::PyFunction_GetCode(made), c"co_consts".as_ptr());
        let count = if consts.is_null() || ffi::PyTuple_Check(consts) == 0 { 0 } else { ffi::PyTuple_GET_SIZE(consts) };
        for i in 0..count {
            let inner = ffi::PyTuple_GET_ITEM(consts, i);
            if ffi::PyCode_Check(inner) == 0 {
                continue;
            }
            let name = ffi::PyObject_GetAttrString(inner, c"co_name".as_ptr());
            let text = if name.is_null() { String::new() } else { pyffi::str_to_string(name) };
            xdecref(name);
            if text == "remove" || text == "_remove" {
                // Kept for good: it is compared by where it is.
                ffi::Py_INCREF(inner);
                found.push((inner as usize, kind));
            }
        }
        xdecref(consts);
    }
    ffi::PyErr_Clear();
    if all {
        let _ = FOUND.set(found.clone());
    }
    found
}

/// What ends with an object besides itself because of the weak references
/// to it, added to `more`: for a weak reference that a weak container
/// keeps (`weak_containers`), what goes with the entry that the container
/// takes out (the value under the reference, the key of a value). False
/// where a weak reference has anything else to call when the object ends.
#[cfg(not(Py_3_13))]
unsafe fn weakly_watched(o: Obj, more: &mut Vec<PyObj>) -> bool {
    let offset = (*ffi::Py_TYPE(o)).tp_weaklistoffset;
    if offset == 0 {
        return false;
    }
    let mut known: Option<Vec<(usize, u8)>> = None;
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
        let Some((_, kind)) = known.iter().find(|(other, _)| *other == code) else {
            return true;
        };
        // The container, through the weak reference to it that the
        // function has as a default, and what it keeps its entries in.
        let defaults = ffi::PyFunction_GetDefaults(callback);
        let to_it = if defaults.is_null() || ffi::PyTuple_GET_SIZE(defaults) == 0 { null_mut() } else { ffi::PyTuple_GET_ITEM(defaults, 0) };
        if to_it.is_null() || ffi::PyWeakref_Check(to_it) == 0 {
            return true;
        }
        let container = ffi::PyWeakref_GetObject(to_it);
        if container.is_null() || container == ffi::Py_None() {
            continue;
        }
        match kind {
            0 => {
                let slot = pyffi::_PyObject_GetDictPtr(container);
                let data = if slot.is_null() || (*slot).is_null() { null_mut() } else { ffi::PyDict_GetItemString(*slot, c"data".as_ptr()) };
                if data.is_null() || ffi::PyDict_CheckExact(data) == 0 || ffi::PyDict_Size(data) > 65536 {
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
                let key = ffi::PyObject_GetAttrString(reference, c"key".as_ptr());
                if key.is_null() {
                    ffi::PyErr_Clear();
                    return true;
                }
                more.push(PyObj::from_owned(key));
            }
            _ => {}
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

/// Whether letting go of a reference to an object runs no Python code even
/// if it is the last one, as far as that can be shown by looking: neither
/// the object nor anything it refers to, and so may take with it, is of a
/// class with a finalizer (`__del__`, a generator's, a file's) or has a
/// weak reference with a callback, other than the one by which a weak
/// dictionary or a weak set takes its entry out (what goes with that
/// entry is looked at like what the object refers to). At most 64 objects
/// are looked at; with more, the answer is no. A class is not looked into: it does not die
/// where a reference to it is let go of (its own order of classes refers to
/// it). What an object holds outside of what it shows the garbage
/// collector (the history of a tensor in C++) is not seen.
pub unsafe fn dies_quietly(o: Obj) -> bool {
    const LOOKED_AT: usize = 64;
    if o.is_null() || capi::is_unboxed(o) {
        return true;
    }
    let mut seen: Vec<Obj> = vec![o];
    let mut found: Vec<Obj> = Vec::new();
    // What a weak container lets go of with its entry: kept while looked at.
    let mut more: Vec<PyObj> = Vec::new();
    let mut at = 0;
    while at < seen.len() {
        let held = seen[at];
        at += 1;
        if ffi::PyType_Check(held) != 0 || is_plain(held) || ffi::PyBytes_CheckExact(held) != 0 {
            continue;
        }
        let ty = ffi::Py_TYPE(held);
        found.clear();
        let before = more.len();
        if (*ty).tp_finalize.is_some() || (*ty).tp_del.is_some() || weakly_watched(held, &mut more) {
            return false;
        }
        found.extend(more[before..].iter().map(|other| other.as_ptr()));
        if let Some(traverse) = (*ty).tp_traverse {
            traverse(held, every_referent, &mut found as *mut Vec<Obj> as *mut std::ffi::c_void);
        }
        for other in &found {
            if !seen.contains(other) {
                if seen.len() >= LOOKED_AT {
                    return false;
                }
                seen.push(*other);
            }
        }
    }
    true
}

/// Whether an object is of a class of the interpreter itself that holds
/// references to what it is made of or goes over: a tuple, a list, a dict,
/// a set, a slice, a cell, a bound method, an iterator, a view. Not a
/// function (it holds the globals of its module), a class or a module.
unsafe fn holds_others(o: Obj) -> bool {
    let ty = ffi::Py_TYPE(o);
    (*ty).tp_traverse.is_some()
        && is_static(ty)
        && ffi::PyType_Check(o) == 0
        && ffi::PyModule_Check(o) == 0
        && ffi::PyFunction_Check(o) == 0
        && ffi::PyCode_Check(o) == 0
}

/// Whether an object stays alive when a path lands and leaves: `dropped`
/// of the references to it that were there before the path ran are let go
/// of where the path lands, and when it leaves, so is every reference the
/// path itself holds. Those are counted, with whatever else may be one:
/// the registers of the path that hold the object (`own`), and what the
/// values in those registers refer to where a value may be something the
/// path made (a container or an iterator of the interpreter's own
/// classes, and what such a one holds inside), but not what the function
/// had before the path (`given`). If the object has more references than
/// that, one of them is somebody else's, and it stays. False where there
/// is too much to look at, or containers are more than four deep inside
/// one another.
pub unsafe fn outlives(o: Obj, own: &[Obj], given: &[Obj], dropped: isize) -> bool {
    const LOOKED_AT: usize = 4096;
    let mut slots: Vec<Obj> = own.iter().copied().filter(|held| !held.is_null() && !capi::is_unboxed(*held)).collect();
    let mut held = slots.iter().filter(|other| **other == o).count() as isize;
    slots.sort_unstable();
    slots.dedup();
    let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut work: Vec<(Obj, u32)> = Vec::new();
    for other in &slots {
        if *other != o && !given.contains(other) && holds_others(*other) {
            work.push((*other, 0));
        }
    }
    let mut found: Vec<Obj> = Vec::new();
    while let Some((at, depth)) = work.pop() {
        if seen.len() >= LOOKED_AT {
            return false;
        }
        seen.insert(at as usize);
        let Some(traverse) = (*ffi::Py_TYPE(at)).tp_traverse else {
            continue;
        };
        found.clear();
        traverse(at, every_referent, &mut found as *mut Vec<Obj> as *mut std::ffi::c_void);
        for other in &found {
            if *other == o {
                held += 1;
            } else if slots.binary_search(other).is_err() && holds_others(*other) && !seen.contains(&(*other as usize)) {
                if depth >= 4 {
                    return false;
                }
                work.push((*other, depth + 1));
            }
        }
    }
    ffi::Py_REFCNT(o) - held - dropped >= 1
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

/// Whether the entries of one dict go into another without Python code
/// running: exact dicts, the names of a call as the keys.
pub unsafe fn quiet_merge(into: Obj, more: Obj) -> bool {
    let exact = |o: Obj| !o.is_null() && !capi::is_unboxed(o) && ffi::PyDict_CheckExact(o) != 0;
    if !exact(into) || !exact(more) {
        return false;
    }
    let (mut at, mut key, mut value) = (0, null_mut(), null_mut());
    while ffi::PyDict_Next(more, &mut at, &mut key, &mut value) != 0 {
        if ffi::PyUnicode_CheckExact(key) == 0 {
            return false;
        }
    }
    true
}

/// Whether `import name` runs no Python code: the `__import__` of the
/// interpreter is asked for a module that has no package around it and
/// that was imported before, which it then only looks up.
pub unsafe fn quiet_import(builtins: Obj, name: Obj, fromlist: Obj, level: Obj) -> bool {
    let import = ffi::PyDict_GetItemString(builtins, c"__import__".as_ptr());
    let own = !import.is_null() && ffi::PyCFunction_Check(import) != 0 && {
        let function = import as *mut ffi::PyCFunctionObject;
        let of = (*function).m_self;
        !of.is_null() && ffi::PyModule_CheckExact(of) != 0 && ffi::PyModule_GetDict(of) == builtins
    };
    let whole = fromlist == ffi::Py_None()
        && !level.is_null()
        && ffi::PyLong_CheckExact(level) != 0
        && ffi::PyLong_AsLong(level) == 0;
    if !own || !whole || ffi::PyUnicode_CheckExact(name) == 0 || pyffi::str_to_string(name).contains('.') {
        return false;
    }
    let module = ffi::PyDict_GetItemWithError(ffi::PyImport_GetModuleDict(), name);
    if module.is_null() || ffi::PyModule_CheckExact(module) == 0 {
        ffi::PyErr_Clear();
        return false;
    }
    // One that is still being imported is waited for.
    let spec = ffi::PyDict_GetItemString(ffi::PyModule_GetDict(module), c"__spec__".as_ptr());
    let slot = if spec.is_null() { std::ptr::null_mut() } else { pyffi::_PyObject_GetDictPtr(spec) };
    let done = !slot.is_null()
        && !(*slot).is_null()
        && ffi::PyDict_GetItemString(*slot, c"_initializing".as_ptr()) == ffi::Py_False();
    ffi::PyErr_Clear();
    done
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

/// Whether `bool(o)` runs no Python code.
pub unsafe fn quiet_truth(o: Obj) -> bool {
    !capi::is_unboxed(o) && (is_plain(o) || quiet_sized(o))
}

/// Moves the version counter of a tensor on by `count`, as that many writes
/// to it in place would (`torch._C._increment_version`, which takes the
/// tensors to do it for).
pub unsafe fn move_version(tensor: Obj, count: u64) {
    static BUMP: std::sync::OnceLock<Option<PyObj>> = std::sync::OnceLock::new();
    let bump = BUMP.get_or_init(|| {
        let module = ffi::PyImport_ImportModule(c"torch._C".as_ptr());
        let found = if module.is_null() { null_mut() } else { ffi::PyObject_GetAttrString(module, c"_increment_version".as_ptr()) };
        xdecref(module);
        ffi::PyErr_Clear();
        (!found.is_null()).then(|| PyObj::from_owned(found))
    });
    let Some(bump) = bump else {
        return;
    };
    let times = ffi::PyTuple_New(count as isize);
    if !times.is_null() {
        for i in 0..count as isize {
            ffi::PyTuple_SET_ITEM(times, i, new_ref(tensor));
        }
        xdecref(ffi::PyObject_CallOneArg(bump.as_ptr(), times));
        ffi::Py_DECREF(times);
    }
    ffi::PyErr_Clear();
}

/// Whether two values compare in C code: numbers, texts, or dtypes and
/// devices.
pub unsafe fn quiet_compare(a: Obj, b: Obj) -> bool {
    let text = |o: Obj| !capi::is_unboxed(o) && ffi::PyUnicode_CheckExact(o) != 0;
    let token = |o: Obj| !capi::is_unboxed(o) && is_token(o);
    (is_number(a) && is_number(b)) || (text(a) && text(b)) || (token(a) && token(b))
}

/// An iterator over, or a view of, a list, a tuple, a dict or a range: what
/// is read of it is read of what it was made of.
pub unsafe fn goes_over_another(o: Obj) -> bool {
    let plain = ffi::PyList_Check(o) != 0
        || ffi::PyTuple_Check(o) != 0
        || ffi::PyDict_Check(o) != 0
        || ffi::PyUnicode_Check(o) != 0
        || ffi::Py_TYPE(o) == std::ptr::addr_of_mut!(ffi::PyRange_Type);
    !plain && quiet_iter(o).is_some()
}

/// Whether iterating over a value runs no Python code: Some(false) for a
/// list, a tuple, a dict, a view of a dict, a str or a range; Some(true)
/// for an iterator over one of those, an `enumerate` or a `zip`, which is
/// used up by it and of which that is only known where a path made it (see
/// `pure_call`).
pub unsafe fn quiet_iter(o: Obj) -> Option<bool> {
    const VIEWS: [&[u8]; 6] = [b"dict_keys", b"dict_values", b"dict_items", b"odict_keys", b"odict_values", b"odict_items"];
    const ITERATORS: [&[u8]; 14] = [
        b"list_iterator",
        b"list_reverseiterator",
        b"tuple_iterator",
        b"range_iterator",
        b"longrange_iterator",
        b"dict_keyiterator",
        b"dict_valueiterator",
        b"dict_itemiterator",
        b"odict_iterator",
        b"str_ascii_iterator",
        b"str_iterator",
        b"enumerate",
        b"zip",
        b"reversed",
    ];
    if capi::is_unboxed(o) {
        return None;
    }
    let ty = ffi::Py_TYPE(o);
    let plain = ffi::PyList_CheckExact(o) != 0
        || ffi::PyTuple_CheckExact(o) != 0
        || is_plain_dict(o)
        || ffi::PyUnicode_CheckExact(o) != 0
        || ty == std::ptr::addr_of_mut!(ffi::PyRange_Type);
    if plain {
        return Some(false);
    }
    if !is_static(ty) {
        return None;
    }
    let name = std::ffi::CStr::from_ptr((*ty).tp_name).to_bytes();
    if VIEWS.contains(&name) {
        return Some(false);
    }
    ITERATORS.contains(&name).then_some(true)
}

/// The functions of the interpreter that a path may call: they run no
/// Python code on arguments of the kinds `pure_call` asks for, and change
/// nothing.
#[derive(Clone, Copy, PartialEq)]
enum Builtin {
    Len,
    Range,
    Enumerate,
    Zip,
    IsInstance,
    Iter,
    Int,
    Float,
    Bool,
    Tuple,
    List,
    Type,
    Callable,
    GetAttr,
    HasAttr,
    Abs,
    Min,
    Max,
    Str,
}

const BUILTINS: [(&std::ffi::CStr, Builtin); 19] = [
    (c"len", Builtin::Len),
    (c"range", Builtin::Range),
    (c"enumerate", Builtin::Enumerate),
    (c"zip", Builtin::Zip),
    (c"isinstance", Builtin::IsInstance),
    (c"iter", Builtin::Iter),
    (c"int", Builtin::Int),
    (c"float", Builtin::Float),
    (c"bool", Builtin::Bool),
    (c"tuple", Builtin::Tuple),
    (c"list", Builtin::List),
    (c"type", Builtin::Type),
    (c"callable", Builtin::Callable),
    (c"getattr", Builtin::GetAttr),
    (c"hasattr", Builtin::HasAttr),
    (c"abs", Builtin::Abs),
    (c"min", Builtin::Min),
    (c"max", Builtin::Max),
    (c"str", Builtin::Str),
];

/// The address of one of them.
unsafe fn builtin_named(which: Builtin) -> usize {
    static FOUND: std::sync::OnceLock<Vec<usize>> = std::sync::OnceLock::new();
    let found = FOUND.get_or_init(|| {
        let module = ffi::PyImport_ImportModule(c"builtins".as_ptr());
        let found = BUILTINS
            .iter()
            .map(|(name, _)| {
                let function = if module.is_null() { null_mut() } else { ffi::PyObject_GetAttrString(module, name.as_ptr()) };
                // The module keeps it.
                xdecref(function);
                function as usize
            })
            .collect();
        ffi::PyErr_Clear();
        xdecref(module);
        found
    });
    found[BUILTINS.iter().position(|(_, other)| *other == which).unwrap()]
}

/// Whether a callee is the builtin `zip`.
pub unsafe fn is_zip(callee: Obj) -> bool {
    !callee.is_null() && builtin_named(Builtin::Zip) == callee as usize
}

/// Whether a callee is `getattr` or `hasattr`: a read of an attribute by
/// its name.
pub unsafe fn reads_attribute(callee: Obj) -> bool {
    builtin_named(Builtin::GetAttr) == callee as usize || builtin_named(Builtin::HasAttr) == callee as usize
}

/// Whether a callee is the builtin `len`.
pub unsafe fn is_len(callee: Obj) -> bool {
    builtin_named(Builtin::Len) == callee as usize
}

/// For a call of a builtin that asks its arguments for an iterator and for
/// nothing else, in their order, before it does anything with them: how
/// many of the arguments it asks, from the first on; 0 for any other call.
/// `iter(x)` (not `iter(x, sentinel)`, which calls `x`), `enumerate(x)` and
/// `enumerate(x, n)` with an int (which is asked for its index first),
/// `zip(...)`, and `list(x)` and `tuple(x)` of an object whose class has
/// neither `__len__` nor `__length_hint__`, which those ask of the object
/// itself.
pub unsafe fn goes_over(callee: Obj, args: &[Obj]) -> usize {
    const TAKE: [Builtin; 5] = [Builtin::Enumerate, Builtin::Zip, Builtin::Iter, Builtin::List, Builtin::Tuple];
    static HINT: std::sync::OnceLock<PyObj> = std::sync::OnceLock::new();
    let Some(which) = TAKE.iter().copied().find(|which| builtin_named(*which) == callee as usize) else {
        return 0;
    };
    if args.is_empty() || args.iter().any(|o| o.is_null()) {
        return 0;
    }
    let one = args.len() == 1;
    let plain = |o: Obj| {
        let hint = HINT.get_or_init(|| PyObj::from_owned(ffi::PyUnicode_InternFromString(c"__length_hint__".as_ptr())));
        let ty = ffi::Py_TYPE(o);
        !capi::is_unboxed(o)
            && pyffi::_PyType_Lookup(ty, specials().len.as_ptr()).is_null()
            && pyffi::_PyType_Lookup(ty, hint.as_ptr()).is_null()
    };
    match which {
        Builtin::Iter => one as usize,
        Builtin::Enumerate => {
            (one || (args.len() == 2 && !capi::is_unboxed(args[1]) && ffi::PyLong_CheckExact(args[1]) != 0)) as usize
        }
        Builtin::List | Builtin::Tuple => (one && plain(args[0])) as usize,
        _ => args.len(),
    }
}

/// The name `__length_hint__`, for a check that a class has none.
pub unsafe fn length_hint() -> PyObj {
    PyObj::from_owned(ffi::PyUnicode_InternFromString(c"__length_hint__".as_ptr()))
}

/// Whether a callee is the builtin `list` or `tuple`.
pub unsafe fn makes_sequence(callee: Obj) -> bool {
    builtin_named(Builtin::List) == callee as usize || builtin_named(Builtin::Tuple) == callee as usize
}

/// The modules whose functions a path may call on any arguments (`torch._C`:
/// those that ask about the state of the thread or of what they are given)
/// or on numbers (`math`).
unsafe fn quiet_modules() -> (usize, usize, usize) {
    static FOUND: std::sync::OnceLock<(usize, usize, usize)> = std::sync::OnceLock::new();
    *FOUND.get_or_init(|| {
        let [state, math, operator] = [c"torch._C", c"math", c"_operator"].map(|name| {
            let module = ffi::PyImport_ImportModule(name.as_ptr());
            // The interpreter keeps it.
            xdecref(module);
            module as usize
        });
        ffi::PyErr_Clear();
        (state, math, operator)
    })
}

/// Whether a call is one of a function a path may call, with arguments on
/// which it runs no Python code and changes nothing: Some, with whether it
/// makes something that is the path's own to change or to use up (a list,
/// an iterator, a `range`, an `enumerate`, a `zip`). `own` says of an
/// argument whether the path made it: an iterator is only gone over then.
pub unsafe fn pure_call(callee: Obj, args: &[Obj], own: impl Fn(usize) -> bool, api: &TcfgApi) -> Option<bool> {
    const ASKS: [&[u8]; 6] = [b"_has_torch_function", b"_get_", b"_is_", b"is_", b"get_", b"_has_"];
    if callee.is_null() || capi::is_unboxed(callee) || args.iter().any(|o| o.is_null()) {
        return None;
    }
    if ffi::PyCFunction_Check(callee) != 0 {
        let function = callee as *mut ffi::PyCFunctionObject;
        let (state, math, operator) = quiet_modules();
        let of = (*function).m_self as usize;
        if of != 0 && of == state {
            let name = std::ffi::CStr::from_ptr((*(*function).m_ml).ml_name).to_bytes();
            return ASKS.iter().any(|asks| name.starts_with(asks)).then_some(false);
        }
        if of != 0 && of == math {
            return args.iter().all(|o| is_number(*o)).then_some(false);
        }
        if of != 0 && of == operator {
            let name = std::ffi::CStr::from_ptr((*(*function).m_ml).ml_name).to_bytes();
            let ints = args.len() == 1 && !capi::is_unboxed(args[0]) && ffi::PyLong_CheckExact(args[0]) != 0;
            return (name == b"index" && ints).then_some(false);
        }
    }
    let which = BUILTINS.iter().map(|(_, which)| *which).find(|which| builtin_named(*which) == callee as usize)?;
    let numbers = || args.iter().all(|o| is_number(*o));
    // What is gone over without anything but the path's own being used up.
    let lasting = |i: usize| match quiet_iter(args[i]) {
        Some(used_up) => !used_up || own(i),
        None => false,
    };
    let all = || (0..args.len()).all(lasting);
    let fine = match which {
        Builtin::Len => {
            let mut info = TcfgTensorInfo::empty();
            args.len() == 1 && (quiet_sized(args[0]) || (api.tensor_info)(args[0], &mut info) != 0)
        }
        Builtin::Range => {
            (1..=3).contains(&args.len()) && args.iter().all(|o| !capi::is_unboxed(*o) && ffi::PyLong_CheckExact(*o) != 0)
        }
        Builtin::Enumerate | Builtin::Iter => args.len() == 1 && all(),
        Builtin::Zip => !args.is_empty() && all(),
        Builtin::Tuple | Builtin::List => args.len() <= 1 && all(),
        Builtin::Int | Builtin::Float | Builtin::Abs => args.len() == 1 && numbers(),
        Builtin::Str => args.len() == 1 && (numbers() || ffi::PyUnicode_CheckExact(args[0]) != 0),
        Builtin::Min | Builtin::Max => args.len() >= 2 && numbers(),
        Builtin::Bool => args.len() == 1 && quiet_truth(args[0]),
        Builtin::Type | Builtin::Callable => args.len() == 1,
        Builtin::GetAttr | Builtin::HasAttr => {
            let found = which == Builtin::GetAttr && args.len() == 3;
            (args.len() == 2 || found)
                && !capi::is_unboxed(args[1])
                && ffi::PyUnicode_CheckExact(args[1]) != 0
                && quiet_attr(args[0], args[1], api)
        }
        Builtin::IsInstance => {
            static CLASS: std::sync::OnceLock<PyObj> = std::sync::OnceLock::new();
            let name = CLASS.get_or_init(|| PyObj::from_owned(ffi::PyUnicode_InternFromString(c"__class__".as_ptr())));
            // A class whose instances `type` itself tells; any class tells
            // what is exactly of it without being asked.
            // (The class of a class may be its own, in C, and still ask as
            // `type` does: `torch.Tensor`.)
            let asks = |meta: *mut ffi::PyTypeObject| pyffi::_PyType_Lookup(meta, specials().instancecheck.as_ptr());
            let class = |o: Obj| {
                !capi::is_unboxed(o)
                    && ffi::PyType_Check(o) != 0
                    && (asks(ffi::Py_TYPE(o)) == asks(std::ptr::addr_of_mut!(ffi::PyType_Type))
                        || (!capi::is_unboxed(args[0]) && ffi::Py_TYPE(args[0]) as Obj == o))
            };
            let classes = |o: Obj| {
                class(o)
                    || (!capi::is_unboxed(o)
                        && ffi::PyTuple_CheckExact(o) != 0
                        && (0..ffi::PyTuple_GET_SIZE(o)).all(|i| class(ffi::PyTuple_GET_ITEM(o, i))))
            };
            args.len() == 2 && classes(args[1]) && quiet_attr(args[0], name.as_ptr(), api)
        }
    };
    fine.then_some(matches!(
        which,
        Builtin::Range | Builtin::Enumerate | Builtin::Zip | Builtin::Iter | Builtin::List
    ))
}

/// Whether a call of a method runs no Python code, no kernel, and changes
/// nothing: what a dict has, as a view or under a key; what a tensor says
/// of itself.
pub unsafe fn pure_method(recv: Obj, name: Obj, args: &[Obj], api: &TcfgApi) -> bool {
    const SAYS: [&str; 8] =
        ["size", "dim", "ndimension", "numel", "nelement", "is_floating_point", "is_complex", "element_size"];
    if recv.is_null() || args.iter().any(|o| o.is_null()) {
        return false;
    }
    let name = pyffi::str_to_string(name);
    let mut info = TcfgTensorInfo::empty();
    if (api.tensor_info)(recv, &mut info) != 0 {
        let ints = args.iter().all(|o| !capi::is_unboxed(*o) && ffi::PyLong_CheckExact(*o) != 0);
        return SAYS.contains(&name.as_str()) && args.len() <= 1 && ints;
    }
    if capi::is_unboxed(recv) || !is_plain_dict(recv) {
        return false;
    }
    match name.as_str() {
        "keys" | "values" | "items" => args.is_empty(),
        "get" => (1..=2).contains(&args.len()) && is_plain_key(args[0]) && quiet_keys(recv),
        _ => false,
    }
}

/// The names of the functions that a class has for what the interpreter
/// does with its instances without a call being written.
pub struct Specials {
    pub iter: PyObj,
    pub len: PyObj,
    pub getitem: PyObj,
    pub get: PyObj,
    pub getattr: PyObj,
    pub setattr: PyObj,
    pub instancecheck: PyObj,
    pub getattribute: PyObj,
}

pub unsafe fn specials() -> &'static Specials {
    static NAMES: std::sync::OnceLock<Specials> = std::sync::OnceLock::new();
    let name = |name: &std::ffi::CStr| PyObj::from_owned(ffi::PyUnicode_InternFromString(name.as_ptr()));
    NAMES.get_or_init(|| Specials {
        iter: name(c"__iter__"),
        len: name(c"__len__"),
        getitem: name(c"__getitem__"),
        get: name(c"__get__"),
        getattr: name(c"__getattr__"),
        setattr: name(c"__setattr__"),
        instancecheck: name(c"__instancecheck__"),
        getattribute: name(c"__getattribute__"),
    })
}

/// The Python function that the class of an object has under a name, for
/// a class that is defined in Python: what the interpreter calls where it
/// iterates over the object, counts it, or takes an item of it. Null for
/// anything else. Borrowed.
pub unsafe fn special(o: Obj, name: &PyObj) -> Obj {
    if o.is_null() || capi::is_unboxed(o) || is_static(ffi::Py_TYPE(o)) || ffi::PyType_Check(o) != 0 {
        return null_mut();
    }
    let found = pyffi::_PyType_Lookup(ffi::Py_TYPE(o), name.as_ptr());
    if !found.is_null() && ffi::PyFunction_Check(found) != 0 { found } else { null_mut() }
}

/// The Python function that `isinstance(obj, class)` comes down to: the
/// `__instancecheck__` of the class of the class, where that is a class
/// defined in Python (`isinstance(3, torch.Tensor)`) and the object is not
/// exactly of the class, which the interpreter answers itself. Null for
/// anything else. Borrowed.
pub unsafe fn instance_check(object: Obj, class: Obj) -> Obj {
    let plain = |o: Obj| !o.is_null() && !capi::is_unboxed(o);
    if !plain(object) || !plain(class) || ffi::PyType_Check(class) == 0 || ffi::Py_TYPE(object) as Obj == class {
        return null_mut();
    }
    let meta = ffi::Py_TYPE(class);
    if is_static(meta) {
        return null_mut();
    }
    let found = pyffi::_PyType_Lookup(meta, specials().instancecheck.as_ptr());
    if !found.is_null() && ffi::PyFunction_Check(found) != 0 { found } else { null_mut() }
}

/// Whether a callee is the builtin `isinstance`.
pub unsafe fn is_isinstance(callee: Obj) -> bool {
    !callee.is_null() && builtin_named(Builtin::IsInstance) == callee as usize
}

/// Whether the attributes of an object are looked up as `object` or a
/// module does it, before any `__getattr__` of its class.
unsafe fn plain_lookup(o: Obj) -> bool {
    if capi::is_unboxed(o) || ffi::PyType_Check(o) != 0 {
        return false;
    }
    let names = specials();
    let found = pyffi::_PyType_Lookup(ffi::Py_TYPE(o), names.getattribute.as_ptr());
    let of = |ty: *mut ffi::PyTypeObject| pyffi::_PyType_Lookup(ty, names.getattribute.as_ptr());
    found == of(std::ptr::addr_of_mut!(ffi::PyBaseObject_Type)) || found == of(std::ptr::addr_of_mut!(ffi::PyModule_Type))
}

/// What the class of an object has under a name, if that is what reading
/// the attribute comes down to whatever the object holds itself: a
/// descriptor that also takes stores. Null for anything else. Borrowed.
pub unsafe fn data_descriptor(o: Obj, name: Obj) -> Obj {
    if o.is_null() || !plain_lookup(o) {
        return null_mut();
    }
    let descr = pyffi::_PyType_Lookup(ffi::Py_TYPE(o), name);
    let data = !descr.is_null() && (*ffi::Py_TYPE(descr)).tp_descr_set.is_some();
    if data { descr } else { null_mut() }
}

/// Whether an object has no attribute of a name, so that reading it comes
/// down to the `__getattr__` of its class: neither the class nor the
/// object has it, and a module has no `__getattr__` of its own.
pub unsafe fn absent(o: Obj, name: Obj) -> bool {
    if o.is_null() || !plain_lookup(o) || !pyffi::_PyType_Lookup(ffi::Py_TYPE(o), name).is_null() {
        return false;
    }
    let slot = pyffi::_PyObject_GetDictPtr(o);
    let dict = if slot.is_null() { null_mut() } else { *slot };
    let has = |name: Obj| !dict.is_null() && !ffi::PyDict_GetItemWithError(dict, name).is_null();
    !has(name) && !(ffi::PyModule_Check(o) != 0 && has(specials().getattr.as_ptr()))
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
    let hooked = !generic && plain_lookup(obj);
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

/// The name and the overload of an operator as the dispatcher names it:
/// (`aten::add_`, `Tensor`) of `aten::add_.Tensor`.
fn overload_of(name: &str) -> (&str, &str) {
    match name.split_once("::").and_then(|(_, rest)| rest.find('.')) {
        Some(dot) => {
            let at = name.len() - (name.split_once("::").unwrap().1.len() - dot);
            (&name[..at], &name[at + 1..])
        }
        None => (name, ""),
    }
}

/// Whether an operator writes to its first argument and to nothing else,
/// and draws no random numbers (`aten::relu_`, `aten::add_.Tensor`,
/// `aten::copy_`; not `aten::add.out`, not `aten::uniform_`): its schema
/// and its tags are asked once.
pub fn writes_first(name: &str) -> bool {
    static KNOWN: std::sync::OnceLock<Shared<HashMap<String, bool>>> = std::sync::OnceLock::new();
    let known = KNOWN.get_or_init(|| Shared::new(HashMap::new()));
    if let Some(writes) = known.borrow().get(name) {
        return *writes;
    }
    let (qualified, overload) = overload_of(name);
    let asked = |py: Python<'_>| -> PyResult<bool> {
        let binding = py.import("torch._C")?;
        let schema = binding.call_method1("_get_schema", (qualified, overload))?;
        let mut first = true;
        for argument in schema.getattr("arguments")?.try_iter()? {
            let alias = argument?.getattr("alias_info")?;
            let written = !alias.is_none() && alias.getattr("is_write")?.is_truthy()?;
            if written != first {
                return Ok(false);
            }
            first = false;
        }
        let found = binding.call_method1("_get_operation_overload", (qualified, overload))?;
        if first || found.is_none() {
            return Ok(false);
        }
        let seeded = py.import("torch")?.getattr("Tag")?.getattr("nondeterministic_seeded")?;
        Ok(!found.get_item(2)?.contains(seeded)?)
    };
    let quiet = !UPDATES_STATISTICS.iter().any(|(op, _)| *op == qualified);
    let writes = quiet && unsafe { asked(Python::assume_attached()).unwrap_or(false) };
    unsafe { ffi::PyErr_Clear() };
    known.borrow_mut().insert(name.to_string(), writes);
    writes
}

/// Whether the sizes of what an operator gives follow from the sizes of
/// what it is given and its other arguments, and not from what the tensors
/// hold (`aten::nonzero`, `aten::masked_select`, an index by a mask): its
/// tags are asked once. They say it of an operator with a kernel of its
/// own; one that is made of others (`composite_operator`) is asked by
/// running it on tensors that hold nothing (`Compiled::sized`).
pub fn shaped_operator(name: &str) -> bool {
    static KNOWN: std::sync::OnceLock<Shared<HashMap<String, bool>>> = std::sync::OnceLock::new();
    let known = KNOWN.get_or_init(|| Shared::new(HashMap::new()));
    if let Some(shaped) = known.borrow().get(name) {
        return *shaped;
    }
    let (qualified, overload) = overload_of(name);
    let asked = |py: Python<'_>| -> PyResult<bool> {
        let binding = py.import("torch._C")?;
        let found = binding.call_method1("_get_operation_overload", (qualified, overload))?;
        if found.is_none() {
            return Ok(false);
        }
        let tags = py.import("torch")?.getattr("Tag")?;
        let (dynamic, data) = (tags.getattr("dynamic_output_shape")?, tags.getattr("data_dependent_output")?);
        Ok(!found.get_item(2)?.contains(dynamic)? && !found.get_item(2)?.contains(data)?)
    };
    let shaped = unsafe { asked(Python::assume_attached()).unwrap_or(false) };
    unsafe { ffi::PyErr_Clear() };
    known.borrow_mut().insert(name.to_string(), shaped);
    shaped
}

/// Whether an operator has no kernel of its own and is made of other
/// operators (`aten::conv2d`; `aten::argwhere`, which is `aten::nonzero`):
/// its tags do not say what those do. What cannot be asked is taken to be
/// one.
pub fn composite_operator(name: &str) -> bool {
    static KNOWN: std::sync::OnceLock<Shared<HashMap<String, bool>>> = std::sync::OnceLock::new();
    let known = KNOWN.get_or_init(|| Shared::new(HashMap::new()));
    if let Some(composite) = known.borrow().get(name) {
        return *composite;
    }
    let asked = |py: Python<'_>| -> PyResult<bool> {
        let has = py.import("torch._C")?.getattr("_dispatch_has_kernel_for_dispatch_key")?;
        has.call1((name, "CompositeImplicitAutograd"))?.is_truthy()
    };
    let composite = unsafe { asked(Python::assume_attached()).unwrap_or(true) };
    unsafe { ffi::PyErr_Clear() };
    known.borrow_mut().insert(name.to_string(), composite);
    composite
}

/// Whether an operator can fail for what its tensors hold, once it has run
/// on tensors of the same sizes and dtypes with the same other arguments.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Total {
    /// It cannot.
    Always,
    /// It cannot on floating point tensors (an integer division can).
    Floats,
    /// It can, or it is not known.
    Never,
}

/// The kernels of layers and of putting tensors together that look at no
/// value to decide whether to run: by the name of the operator, without
/// the overload and the `_` of the variant that writes.
const TOTAL: [&str; 137] = [
    "conv1d", "conv2d", "conv3d", "conv_transpose1d", "conv_transpose2d", "conv_transpose3d", "convolution",
    "_convolution", "linear", "matmul", "mm", "bmm", "addmm", "addbmm", "baddbmm", "addmv", "mv", "dot", "outer",
    "einsum", "batch_norm", "layer_norm", "group_norm", "instance_norm", "native_batch_norm", "native_layer_norm",
    "native_group_norm", "cudnn_batch_norm", "_batch_norm_impl_index", "max_pool1d", "max_pool2d", "max_pool3d",
    "max_pool2d_with_indices", "avg_pool1d", "avg_pool2d", "avg_pool3d", "adaptive_avg_pool1d",
    "adaptive_avg_pool2d", "adaptive_avg_pool3d", "adaptive_max_pool1d", "adaptive_max_pool2d",
    "adaptive_max_pool3d", "_adaptive_avg_pool2d", "upsample_nearest1d", "upsample_nearest2d",
    "upsample_nearest3d", "upsample_bilinear2d", "upsample_bicubic2d", "upsample_linear1d",
    "upsample_trilinear3d", "pad", "constant_pad_nd", "dropout", "feature_dropout", "alpha_dropout",
    "feature_alpha_dropout", "scaled_dot_product_attention", "softmax", "log_softmax", "_softmax", "_log_softmax",
    "cat", "concat", "concatenate", "stack", "hstack", "vstack", "dstack", "clone", "contiguous", "to", "type_as",
    "_to_copy", "copy", "detach", "alias", "zeros", "ones", "full", "empty", "zeros_like", "ones_like",
    "full_like", "empty_like", "arange", "linspace", "eye", "scalar_tensor", "lift_fresh", "lift_fresh_copy",
    "rand", "randn", "rand_like", "randn_like", "uniform", "expand", "expand_as", "repeat", "tile", "flip",
    "roll", "tril", "triu", "flatten", "unflatten", "reshape", "view_as", "reshape_as", "chunk", "split",
    "split_with_sizes", "unbind", "meshgrid", "broadcast_to", "fill", "zero", "masked_fill", "where", "clamp",
    "clamp_min", "clamp_max", "lerp", "cumsum", "sort", "argsort", "topk", "max", "min", "relu6", "selu", "celu",
    "prelu", "mse_loss", "l1_loss", "smooth_l1_loss", "huber_loss", "binary_cross_entropy_with_logits",
];

/// Of the operators that work on each element by itself: those that
/// cannot fail on integers either.
const WHOLE_NUMBERS: [&str; 23] = [
    "add", "sub", "rsub", "mul", "neg", "abs", "eq", "ne", "lt", "le", "gt", "ge", "maximum", "minimum",
    "logical_and", "logical_or", "logical_not", "logical_xor", "bitwise_and", "bitwise_or", "bitwise_xor",
    "bitwise_not", "sign",
];

/// See `Total`. A view gives its argument seen another way and reads
/// nothing of it (its schema says that it is one, or its tag that it
/// makes its argument one); the operators that work element by element,
/// and those that reduce, say so in their tags;
/// the rest is the list above. `aten::as_strided` is a view that can fail
/// for how much memory there is, which no form checks.
pub fn total_operator(name: &str) -> Total {
    static KNOWN: std::sync::OnceLock<Shared<HashMap<String, Total>>> = std::sync::OnceLock::new();
    let known = KNOWN.get_or_init(|| Shared::new(HashMap::new()));
    if let Some(total) = known.borrow().get(name) {
        return *total;
    }
    let (qualified, overload) = overload_of(name);
    let base = qualified.rsplit("::").next().unwrap_or(qualified);
    let base = base.strip_suffix('_').filter(|rest| !rest.is_empty()).unwrap_or(base);
    let asked = |py: Python<'_>| -> PyResult<Total> {
        if base == "as_strided" {
            return Ok(Total::Never);
        }
        if TOTAL.contains(&base) {
            return Ok(Total::Always);
        }
        let binding = py.import("torch._C")?;
        let schema = binding.call_method1("_get_schema", (qualified, overload))?;
        for given in schema.getattr("returns")?.try_iter()? {
            let alias = given?.getattr("alias_info")?;
            if !alias.is_none() && !alias.getattr("is_write")?.is_truthy()? {
                return Ok(Total::Always);
            }
        }
        let found = binding.call_method1("_get_operation_overload", (qualified, overload))?;
        if found.is_none() {
            return Ok(Total::Never);
        }
        let (tags, tag) = (found.get_item(2)?, py.import("torch")?.getattr("Tag")?);
        if tags.contains(tag.getattr("reduction")?)? || tags.contains(tag.getattr("inplace_view")?)? {
            return Ok(Total::Always);
        }
        if tags.contains(tag.getattr("pointwise")?)? {
            return Ok(if WHOLE_NUMBERS.contains(&base) { Total::Always } else { Total::Floats });
        }
        Ok(Total::Never)
    };
    let total = unsafe { asked(Python::assume_attached()).unwrap_or(Total::Never) };
    unsafe { ffi::PyErr_Clear() };
    known.borrow_mut().insert(name.to_string(), total);
    total
}

/// Whether an operator, as the dispatcher names it (`aten::cat`,
/// `aten::add_.Tensor`), neither writes to an argument nor draws random
/// numbers: its schema and its tags are asked once.
pub fn pure_operator(name: &str) -> bool {
    static KNOWN: std::sync::OnceLock<Shared<HashMap<String, bool>>> = std::sync::OnceLock::new();
    let known = KNOWN.get_or_init(|| Shared::new(HashMap::new()));
    if let Some(pure) = known.borrow().get(name) {
        return *pure;
    }
    let (qualified, overload) = overload_of(name);
    let asked = |py: Python<'_>| -> PyResult<bool> {
        let binding = py.import("torch._C")?;
        let schema = binding.call_method1("_get_schema", (qualified, overload))?;
        if schema.getattr("is_mutable")?.is_truthy()? {
            return Ok(false);
        }
        let found = binding.call_method1("_get_operation_overload", (qualified, overload))?;
        if found.is_none() {
            return Ok(false);
        }
        let seeded = py.import("torch")?.getattr("Tag")?.getattr("nondeterministic_seeded")?;
        Ok(!found.get_item(2)?.contains(seeded)?)
    };
    let quiet = !UPDATES_STATISTICS.iter().any(|(op, _)| *op == qualified);
    let pure = quiet && unsafe { asked(Python::assume_attached()).unwrap_or(false) };
    unsafe { ffi::PyErr_Clear() };
    known.borrow_mut().insert(name.to_string(), pure);
    pure
}

/// What a run of a path on the values of a call found, for an exported form
/// to be made from: the values of the path as they were (owned; null where
/// one was not kept), and per node the site that ran and what it returned
/// (dtype, device, requires_grad, sizes: an owned tuple).
pub struct Recording {
    pub values: Vec<Obj>,
    /// Per value: the version counter of the tensor it was when the path
    /// came to it (`TCFG_NO_VERSION` for what has none).
    pub born: Vec<u64>,
    /// The site is null for an operation that is no call site.
    pub nodes: Vec<(*mut TcfgSite, Obj)>,
    /// Why the run cannot be described, once something says so.
    pub failed: Option<String>,
    /// An operation wrote to a tensor it was given.
    pub wrote: bool,
}

impl Recording {
    fn value(&self, v: PathValue) -> Obj {
        self.values.get(v as usize).copied().unwrap_or(null_mut())
    }
}

impl Drop for Recording {
    fn drop(&mut self) {
        unsafe {
            for value in &self.values {
                xdecref(*value);
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

/// The value that the k-th tensor a graph takes out of a list or tuple is
/// in its description: a number no value of a function has.
fn item_id(k: usize) -> i64 {
    (1i64 << 32) + k as i64
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
    pub args: Vec<Place>,
    /// For a path that writes to tensors from outside: per tensor that is
    /// checked, the first of them that has the same memory, and where in
    /// that memory it starts if another has it too (`UNSHARED` if none
    /// has). What a compiler makes of writes to its inputs is for inputs
    /// that lie over one another exactly as these do (the sizes and
    /// strides are the form's anyway), and for no others.
    pub aliases: Vec<(u32, i64)>,
    /// For such a path: the tensors the callable is called with (by their
    /// place among them) that the operations wrote to in place, and by how
    /// much the version counter of each moved in this run.
    pub bumps: Vec<(u32, u64)>,
}

impl Path {
    /// A name for a value of the path that a reader can find in the code:
    /// the variable it is, in the function the path starts in; for a value
    /// of a function the path goes into, that function with it; for what
    /// was read off an object, the object and the attribute
    /// (`self.conv1.weight`), as far back as that goes.
    pub unsafe fn label(&self, value: PathValue) -> String {
        if let Some(name) = self.named(value).filter(|name| name.contains('.')) {
            // Read off an object: by the object and the attributes, with
            // the function the object is a variable of if it is a called one.
            let mut base = value;
            while let Some(PathInst { op: Op::LoadAttr(object, _), .. }) = self.body.iter().find(|inst| inst.dst == base) {
                base = *object;
            }
            return match self.source.get(base as usize) {
                Some((owner, _)) if *owner != 0 => format!("{}:{}", (*self.owners[*owner as usize]).func.qualname, name),
                _ => name,
            };
        }
        match self.source.get(value as usize) {
            Some((0, v)) if *v != NONE => (*self.owners[0]).func.value_name(*v),
            Some((owner, v)) if *v != NONE => {
                let func = &(*self.owners[*owner as usize]).func;
                format!("{}:{}", func.qualname, func.value_name(*v))
            }
            Some(_) => format!("%{}", value),
            None => (*self.owners[0]).func.value_name(value),
        }
    }

    /// The name of a value as the code has it: the variable, or the object
    /// and the attribute that was read off it. None for what has no name.
    unsafe fn named(&self, value: PathValue) -> Option<String> {
        let read = self.body.iter().find(|inst| inst.dst == value && value != NONE);
        if let Some(PathInst { op: Op::LoadAttr(object, name), owner, .. }) = read {
            let func = &(*self.owners[*owner as usize]).func;
            return Some(format!("{}.{}", self.named(*object)?, pyffi::str_to_string(func.consts[*name as usize].as_ptr())));
        }
        let (owner, v) = self.source.get(value as usize).copied()?;
        let func = &(*self.owners[owner as usize]).func;
        func.value_names.get(v as usize).and_then(|name| name.clone())
    }

    /// What the exporter is handed for a recorded run, with the checks the
    /// exported form holds under and the values the callable is called
    /// with. A dict:
    ///   where: the name of the graph;
    ///   inputs, input_ids, input_names: the tensors the graph takes, as
    ///     they are in this call, with the value each is and a name for it;
    ///   nodes: in order, per operation ("aten", what `site_export` gives,
    ///     operands, value, meta) for a call site, and for what is none
    ///     ("call", callee, operands, kwnames, value, meta), ("method",
    ///     name, operands, kwnames, value, meta) or ("getitem", None,
    ///     [tensor, index], None, value, meta); and ("size", tensor, dim,
    ///     value), ("dim", tensor, value), ("binary", symbol, a, b, value),
    ///     ("neg", a, value), ("tuple", items, value), ("list", items,
    ///     value) or ("slice", [start, stop, step], value) per integer and
    ///     container the graph makes; an operand is ("v", value) or ("c",
    ///     object), a constant;
    ///   outputs: the values the callable returns, as a tuple in this order;
    ///   exits: integers it returns behind them, ("dim", tensor) or ("size",
    ///     tensor, dim) each.
    /// A run that cannot be exported is described by its checks alone: the
    /// form made of them has the calls like it run the instructions.
    pub unsafe fn describe(&self, name: &str, rec: &Recording, api: &TcfgApi) -> Described {
        let mut checks: Vec<Check> = Vec::new();
        let mut tensors = vec![false; self.feeds.len()];
        let mut items: Vec<Place> = Vec::new();
        let mut closed = rec.failed.clone();
        let mut wrote = rec.wrote;
        let nodes = match &closed {
            None => self.nodes(rec, api, &mut checks, &mut tensors, &mut items, &mut wrote),
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
        let mut args: Vec<Place> = Vec::new();
        let mut given: Vec<Obj> = Vec::new();
        let mut numbers: Vec<i64> = Vec::new();
        // The tensors among the values the nodes take, and then those the
        // graph takes out of lists and tuples.
        let feeds = self.feeds.iter().enumerate().map(|(k, v)| (Place::of(*v), rec.value(*v), tensors[k]));
        let held = items.iter().map(|place| (*place, place.at(&rec.values), true));
        // And those that are only looked at behind the callable.
        let pinned = self.pinned.iter().filter(|v| !self.feeds.contains(v)).map(|v| (Place::of(*v), rec.value(*v), false));
        let places: Vec<(Place, Obj, bool)> = feeds.chain(held).chain(pinned).collect();
        // A run whose operations could not be followed says nothing of the
        // values they took: the form is then for calls that have a value of
        // the same class wherever this one has something that is no plain
        // tensor, and for no other.
        let constants: Vec<PathValue> =
            self.body.iter().filter(|inst| matches!(inst.op, Op::Const(_))).map(|inst| inst.dst).collect();
        for (k, (place, tensor, taken)) in places.into_iter().enumerate() {
            let same = given.iter().position(|o| *o == tensor).unwrap_or(given.len());
            let Some(row) = (if tensor.is_null() { None } else { Row::of(tensor, same, &mut numbers, api) }) else {
                let unsaid = rec.failed.is_some()
                    && !tensor.is_null()
                    && !capi::is_unboxed(tensor)
                    && !constants.contains(&place.value)
                    && !checks.iter().any(|check| check.target == place);
                if unsaid {
                    checks.push(Check { target: place, cond: Cond::Type(PyObj::from_borrowed(ffi::Py_TYPE(tensor) as Obj)), late: false });
                }
                continue;
            };
            if row.forward && taken && closed.is_none() {
                closed = Some("a tensor of the call carries a forward-mode gradient".to_string());
            }
            checks.push(Check { target: place, cond: Cond::Tensor(row), late: false });
            given.push(tensor);
            if !taken {
                continue;
            }
            args.push(place);
            ffi::PyList_Append(inputs.as_ptr(), tensor);
            let id = match place.item {
                NONE => ffi::PyLong_FromLong(place.value as _),
                _ => ffi::PyLong_FromLongLong(item_id(k - self.feeds.len())),
            };
            ffi::PyList_Append(ids.as_ptr(), id);
            ffi::Py_DECREF(id);
            // By the variable, or by where it was read from: the function
            // and the attributes (`self.conv1.weight`), as an identifier.
            let label = match self.named(place.value) {
                Some(name) => name.chars().map(|c| if c.is_alphanumeric() || c == '_' { c } else { '_' }).collect(),
                None => format!("v{}", place.value),
            };
            let label = if place.item == NONE { text(&label) } else { text(&format!("{}_{}", label, place.item)) };
            ffi::PyList_Append(names.as_ptr(), label);
            ffi::Py_DECREF(label);
        }
        if wrote && closed.is_none() {
            // What is made of a graph runs its operations again where it
            // fails the first time, so a call whose operations write is
            // not exported.
            closed = Some("its operations write to what they are given".to_string());
        }
        // What an operation with several results gave is taken apart in
        // the graph: whole, it is not what a compiler would give back.
        let whole = |v: &PathValue| {
            let held = rec.value(*v);
            !held.is_null() && !capi::is_unboxed(held) && (ffi::PyTuple_Check(held) != 0 || ffi::PyList_Check(held) != 0)
        };
        if closed.is_none() && self.outputs.iter().any(whole) {
            closed = Some("what an operation with several results gave leaves the graph whole".to_string());
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
        let aliases = if self.mutates { overlaps(&given, api) } else { Vec::new() };
        let mut bumps: Vec<(u32, u64)> = Vec::new();
        if self.mutates {
            for (k, place) in args.iter().enumerate().filter(|(_, place)| place.item == NONE) {
                let (held, born) = (rec.value(place.value), rec.born.get(place.value as usize).copied().unwrap_or(capi::TCFG_NO_VERSION));
                let mut info = TcfgTensorInfo::empty();
                let counted = !held.is_null() && (api.tensor_info)(held, &mut info) != 0 && info.version != capi::TCFG_NO_VERSION;
                if counted && born != capi::TCFG_NO_VERSION && info.version > born {
                    bumps.push((k as u32, info.version - born));
                }
            }
        }
        Described { description, closed, checks, numbers, args, aliases, bumps }
    }

    /// For a method call whose receiver the graph is given and that is no
    /// tensor: the function the receiver has under the name, as a constant
    /// of the graph, which the form is then checked for. None for a
    /// receiver that is a tensor or a value of the graph.
    unsafe fn function_of(
        &self,
        recv: PathValue,
        name: Obj,
        rec: &Recording,
        api: &TcfgApi,
        checks: &mut Vec<Check>,
    ) -> Option<Result<Obj, String>> {
        self.feeds.iter().position(|f| *f == recv)?;
        let value = rec.value(recv);
        let mut info = TcfgTensorInfo::empty();
        if value.is_null() || (api.tensor_info)(value, &mut info) != 0 {
            return None;
        }
        let function = if quiet_attr(value, name, api) { ffi::PyObject_GetAttr(value, name) } else { null_mut() };
        ffi::PyErr_Clear();
        let bound = !function.is_null()
            && ffi::PyCFunction_Check(function) != 0
            && !ffi::PyCFunction_GetSelf(function).is_null()
            && (api.tensor_info)(ffi::PyCFunction_GetSelf(function), &mut info) != 0;
        if function.is_null() || ffi::PyCFunction_Check(function) == 0 || bound {
            xdecref(function);
            checks.push(Check { target: Place::of(recv), cond: Cond::Type(PyObj::from_borrowed(ffi::Py_TYPE(value) as Obj)), late: false });
            return Some(Err(format!("an operation of the graph is a method of a {}", pyffi::type_name(value))));
        }
        let function = PyObj::from_owned(function);
        let found = const_ref(function.as_ptr());
        checks.push(Check { target: Place::of(recv), cond: Cond::Has(PyObj::from_borrowed(name), function), late: false });
        Some(Ok(found))
    }

    /// The nodes of the description of a recorded run, as a list. Adds the
    /// checks on the values the graph has as constants, marks the feeds it
    /// takes as tensors, lists the tensors it takes out of lists and tuples
    /// (`items`: the k-th is the value `item_id(k)` of the graph), and finds
    /// whether a normalization updated its statistics. The reason, if the run cannot be exported.
    unsafe fn nodes(
        &self,
        rec: &Recording,
        api: &TcfgApi,
        checks: &mut Vec<Check>,
        tensors: &mut [bool],
        items: &mut Vec<Place>,
        wrote: &mut bool,
    ) -> Result<PyObj, String> {
        let inside: Vec<PathValue> = self
            .body
            .iter()
            .filter(|inst| inst.kind == Kind::Node || inst.in_graph)
            .map(|inst| inst.dst)
            .collect();
        // The closure below and what is said of a receiver both add checks.
        let checks = std::cell::RefCell::new(checks);
        let constants: Vec<PathValue> =
            self.body.iter().filter(|inst| matches!(inst.op, Op::Const(_))).map(|inst| inst.dst).collect();
        // An operand as the graph has it: a value of its own, a tensor it is
        // given, or a constant, which the exported form is then checked for.
        // How it is taken is what its call site says, or, without one, what
        // the value is.
        let mut operand = |v: PathValue, how: Option<&str>| -> Result<Obj, String> {
            if inside.contains(&v) {
                return Ok(value_ref(v));
            }
            let feed = self.feeds.iter().position(|f| *f == v);
            let late = self.constants.contains(&v);
            let value = if feed.is_some() || late { rec.value(v) } else { null_mut() };
            if value.is_null() {
                return Err("an operand of the graph has no value".to_string());
            }
            let mut info = TcfgTensorInfo::empty();
            let tensor = (api.tensor_info)(value, &mut info) != 0;
            let function = !tensor && ffi::PyCFunction_Check(value) != 0;
            let how = match how {
                Some(how) => how,
                // A constant of the function is what it is.
                None if constants.contains(&v) => "none",
                None if tensor => "tensor",
                // A method bound to a tensor is another object at every call.
                None if function => {
                    let bound = ffi::PyCFunction_GetSelf(value);
                    let free = bound.is_null() || (api.tensor_info)(bound, &mut info) == 0;
                    if free { "identity" } else { "bound" }
                }
                None => "exact",
            };
            let fits = match how {
                "tensor" => tensor,
                "int" => ffi::PyLong_CheckExact(value) != 0,
                "exact" => checkable(value),
                "bound" => false,
                _ => true,
            };
            let sequence = ffi::PyList_CheckExact(value) != 0 || ffi::PyTuple_CheckExact(value) != 0;
            if how == "exact" && !fits && sequence {
                // A list or tuple with tensors in it: the graph takes its
                // items, and is for one of this length.
                let count = ffi::PyObject_Size(value);
                let mut parts: Vec<Obj> = Vec::new();
                let mut held = checks.borrow_mut();
                held.push(Check { target: Place::of(v), cond: Cond::Sequence(ffi::Py_TYPE(value) as usize, count as usize), late: false });
                for i in 0..count {
                    let item = sequence_item(value, i);
                    let place = Place { value: v, item: i as u32 };
                    if (api.tensor_info)(item, &mut info) != 0 {
                        parts.push(tuple_of(&[text("v"), ffi::PyLong_FromLongLong(item_id(items.len()))]));
                        items.push(place);
                    } else if checkable(item) {
                        let kept = frozen(item);
                        parts.push(const_ref(kept.as_ptr()));
                        held.push(Check { target: place, cond: Cond::ValueEq(kept), late: false });
                    } else {
                        parts.into_iter().for_each(|o| ffi::Py_DECREF(o));
                        held.push(Check { target: place, cond: Cond::Type(PyObj::from_borrowed(ffi::Py_TYPE(item) as Obj)), late: false });
                        return Err(format!("an operand of the graph holds a {}", pyffi::type_name(item)));
                    }
                }
                let kind = if ffi::PyList_CheckExact(value) != 0 { "l" } else { "t" };
                return Ok(tuple_of(&[text(kind), list_of(parts)]));
            }
            let v = Place::of(v);
            if !fits {
                checks.borrow_mut().push(Check { target: v, cond: Cond::Type(PyObj::from_borrowed(ffi::Py_TYPE(value) as Obj)), late: false });
                return Err(format!("an operand of the graph is a {}", pyffi::type_name(value)));
            }
            let cond = match how {
                "tensor" => {
                    // One that is made behind the callable is nothing the
                    // graph can be handed.
                    let Some(feed) = feed else {
                        return Err("an operand of the graph is a tensor that is made behind it".to_string());
                    };
                    tensors[feed] = true;
                    return Ok(value_ref(v.value));
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
            let known = checks.borrow().iter().any(|c| {
                c.target == v && std::mem::discriminant(&c.cond) == std::mem::discriminant(&cond)
            });
            if !known {
                checks.borrow_mut().push(Check { target: v, cond, late });
            }
            Ok(constant)
        };
        // Some operands, as a list.
        type Taken<'a> = [(PathValue, Option<&'a str>)];
        fn gather(
            operand: &mut impl FnMut(PathValue, Option<&str>) -> Result<Obj, String>,
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
        let free = |vs: &[V]| -> Vec<(PathValue, Option<&str>)> { vs.iter().map(|v| (*v, None)).collect() };
        let nodes = PyObj::from_owned(ffi::PyList_New(0));
        let add = |node: Obj| {
            ffi::PyList_Append(nodes.as_ptr(), node);
            ffi::Py_DECREF(node);
        };
        let number = |n: i64| ffi::PyLong_FromLongLong(n);
        let none = || new_ref(ffi::Py_None());
        let mut recorded = rec.nodes.iter();
        let did_not_run = || "an operation of the graph did not run".to_string();
        for inst in &self.body {
            let value = inst.dst as i64;
            let func = &(*self.owners[inst.owner as usize]).func;
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
                        .map(|v| self.feeds.iter().position(|f| f == v).map_or(null_mut(), |_| rec.value(*v)))
                        .collect();
                    // The statistics of a normalization are tensors from
                    // outside: a path that may write to those has it.
                    *wrote |= !self.mutates && updates_statistics(exported.as_ptr(), &given);
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
                    let taken: Vec<(PathValue, Option<&str>)> =
                        d.operands.iter().zip(&hows).map(|(v, how)| (*v, Some(how.as_str()))).collect();
                    let taken = gather(&mut operand, &taken)?;
                    let meta = if meta.is_null() { ffi::Py_None() } else { *meta };
                    add(tuple_of(&[text("aten"), new_ref(exported.as_ptr()), taken, number(value), new_ref(meta)]));
                }
                op if inst.kind == Kind::Node => {
                    let (_, meta) = recorded.next().ok_or_else(did_not_run)?;
                    let names = |kwnames: u32| {
                        if kwnames == NONE {
                            none()
                        } else {
                            new_ref(func.consts[kwnames as usize].as_ptr())
                        }
                    };
                    let (kind, head, taken, keywords) = match op {
                        Op::Call { callee, args, kwnames } => {
                            ("call", operand(*callee, None)?, free(args), names(*kwnames))
                        }
                        Op::CallMethod { recv, name, args, kwnames } => {
                            let name = func.consts[*name as usize].as_ptr();
                            match self.function_of(*recv, name, rec, api, &mut checks.borrow_mut()) {
                                // A function of a module: a call of what the
                                // module has under the name.
                                Some(function) => ("call", function?, free(args), names(*kwnames)),
                                None => {
                                    let mut taken = free(&[*recv]);
                                    taken.extend(free(args));
                                    ("method", new_ref(name), taken, names(*kwnames))
                                }
                            }
                        }
                        Op::GetItem(a, k) => ("getitem", none(), free(&[*a, *k]), none()),
                        Op::SetItem { obj, key, val } => ("setitem", none(), free(&[*obj, *key, *val]), none()),
                        _ => return Err("an operation of the graph has no node".to_string()),
                    };
                    let taken = match gather(&mut operand, &taken) {
                        Ok(taken) => taken,
                        Err(why) => {
                            ffi::Py_DECREF(head);
                            ffi::Py_DECREF(keywords);
                            return Err(why);
                        }
                    };
                    let meta = if meta.is_null() { ffi::Py_None() } else { *meta };
                    add(tuple_of(&[text(kind), head, taken, keywords, number(value), new_ref(meta)]));
                }
                op if inst.in_graph => {
                    let tensor: Option<&str> = Some("tensor");
                    let int: Option<&str> = Some("int");
                    let node = match op {
                        Op::TensorSize { t, dim } => {
                            let taken = operand(*t, tensor)?;
                            vec![text("size"), taken, number(*dim)]
                        }
                        Op::TensorDim(t) => {
                            let taken = operand(*t, tensor)?;
                            vec![text("dim"), taken]
                        }
                        Op::Binary(k, a, b) => {
                            let taken = gather(&mut operand, &[(*a, int), (*b, int)])?;
                            let (a, b) = (ffi::PyList_GET_ITEM(taken, 0), ffi::PyList_GET_ITEM(taken, 1));
                            let node = vec![text("binary"), text(BINARY_SYMBOLS[*k as usize]), new_ref(a), new_ref(b)];
                            ffi::Py_DECREF(taken);
                            node
                        }
                        Op::Unary(_, a) => {
                            let taken = operand(*a, int)?;
                            vec![text("neg"), taken]
                        }
                        Op::BuildTuple(vs) | Op::BuildList(vs) => {
                            let taken = gather(&mut operand, &free(vs))?;
                            vec![text(if matches!(op, Op::BuildTuple(_)) { "tuple" } else { "list" }), taken]
                        }
                        Op::Unpack(v, _) => vec![text("same"), operand(*v, None)?],
                        Op::TupleGet(t, i) => vec![text("item"), operand(*t, None)?, const_ref(number(*i as i64))],
                        Op::GetItem(v, at) => {
                            let taken = gather(&mut operand, &[(*v, None), (*at, int)])?;
                            let (v, at) = (ffi::PyList_GET_ITEM(taken, 0), ffi::PyList_GET_ITEM(taken, 1));
                            let node = vec![text("item"), new_ref(v), new_ref(at)];
                            ffi::Py_DECREF(taken);
                            node
                        }
                        Op::LoadAttr(v, name) => {
                            vec![text("attr"), operand(*v, None)?, new_ref(func.consts[*name as usize].as_ptr())]
                        }
                        Op::BuildSlice(a, b, c) => {
                            let parts = gather(&mut operand, &free(&[*a, *b]))?;
                            let step = if *c == NONE { Ok(const_ref(ffi::Py_None())) } else { operand(*c, None) };
                            let step = step.inspect_err(|_| ffi::Py_DECREF(parts))?;
                            ffi::PyList_Append(parts, step);
                            ffi::Py_DECREF(step);
                            vec![text("slice"), parts]
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
                (Cond::Has(name, a), Cond::Has(other, b))
                    if name.as_ptr() == other.as_ptr() && a.as_ptr() == b.as_ptr() => {}
                (Cond::Sequence(a, n), Cond::Sequence(b, m)) if (a, n) == (b, m) => {}
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
    pub unsafe fn conditions(&self, path: &Path) -> Vec<String> {
        const DTYPES: [&str; 12] = [
            "uint8", "int8", "int16", "int32", "int64", "float16", "float32", "float64", "complex32",
            "complex64", "complex128", "bool",
        ];
        let short = |o: &PyObj| named(o.as_ptr());
        let numbers = self.sizes.made_on();
        let mut at = 0;
        // For a form of a path that writes to tensors from outside: which
        // of its tensors share memory, and where each starts in it.
        let tensors: Vec<&Check> = self.checks.iter().filter(|check| matches!(check.cond, Cond::Tensor(_))).collect();
        let mut shared: Vec<String> = Vec::new();
        for (k, (first, start)) in self.aliases.iter().enumerate() {
            if *first as usize == k && *start != UNSHARED {
                let lie = |(other, (of, start)): (usize, &(u32, i64))| {
                    (*of as usize == k).then(|| format!("{} at {}", tensors.get(other).map_or(String::new(), |check| check.target.text(path)), start))
                };
                let all: Vec<String> = self.aliases.iter().enumerate().filter_map(lie).collect();
                shared.push(format!("share memory: {}", all.join(", ")));
            }
        }
        if !self.aliases.is_empty() && shared.is_empty() {
            shared.push("no two tensors share memory".to_string());
        }
        self.checks
            .iter()
            .map(|check| {
                let value = check.target.text(path);
                match &check.cond {
                    Cond::Is(expected) => format!("{} is {}", value, short(expected)),
                    Cond::ValueEq(expected) => format!("{} == {}", value, short(expected)),
                    Cond::Type(ty) => format!("type({}) is {}", value, pyffi::class_name(ty.as_ptr())),
                    Cond::Has(name, expected) => {
                        format!("{}.{} is {}", value, pyffi::str_to_string(name.as_ptr()), short(expected))
                    }
                    Cond::Sequence(ty, count) => {
                        format!("{} is a {} of {}", value, pyffi::class_name(*ty as Obj), count)
                    }
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
            .chain(shared)
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

/// The graph pass: finds where the graphs of a block start, once the other
/// passes are done with it: at the first operation of every run of
/// operations, and at the first instruction of the function. `finalize`
/// puts `Op::Graph` there in a copy of the code, which runs while one of
/// the graphs has something to do; with `export`, what the paths from
/// there are made of is handed to the passes over graphs.
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

/// What is known of a value that an instruction of a run or a path
/// defines.
#[derive(Clone, Copy, Default)]
pub struct Def {
    pub node: bool,
    /// It depends on the result of a node.
    pub late: bool,
    /// An integer, or a list, tuple or slice, that the graph can compute
    /// itself.
    pub graph: bool,
    /// A list, a dict or an iterator that an instruction of the path made:
    /// the path may change it.
    pub made: bool,
    /// A tensor whose memory an operation of the path made (it, or one it
    /// is a view of): nothing outside the path has it, and an operation
    /// that writes to it is an operation of the graph.
    pub own: bool,
    /// It is the same in every call that a form of the path is for: a
    /// constant; a tensor an operation gave, as far as its sizes go (the
    /// sizes of what a graph computes follow from those of what it is
    /// given, which a form checks, unless an operator says otherwise:
    /// `shaped_operator`); what is computed from the sizes of such
    /// tensors and from constants. A check of such a value behind the
    /// callable cannot fail.
    pub fixed: bool,
}

/// The binary operators a graph computes on integers: + - * // %.
const INT_BINARY: [u8; 5] = [0, 10, 5, 2, 6];

impl Graphs {
    fn build(&self, unit: &mut Unit, cx: &PassCx) -> bool {
        if !starts_graphs(&cx.func().filename) {
            return false;
        }
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
        // where the block has tensor work on both sides.
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
        unit.left = left;
        // A graph starts at the first operation of each run, and one at the
        // first instruction of the function. The versions of a block share
        // the graphs that start at the same instruction.
        let mut seeds: Vec<(usize, Path, bool)> = Vec::new();
        for (first, end, ends) in runs {
            seeds.push((first, path(unit, cx, &roles, first, end, ends), false));
        }
        let starts = seeds.first().is_some_and(|(_, seed, _)| seed.root.1 == 0);
        if unit.block == 0 && n > 0 && !starts {
            let run = seeds.first().map_or(0, |(_, seed, _)| seed.land.1);
            seeds.insert(0, (0, Path::empty((0, 0), (0, run)), true));
        }
        for (first, seed, entry) in seeds {
            let root = seed.root.1 as u32;
            let mut state = cx.cf.state.borrow_mut();
            let known = &mut state.blocks[unit.block].graphs;
            let site = match known.iter().find(|(at, _)| *at == root) {
                Some((_, site)) => {
                    *site.seed.borrow_mut() = Arc::new(seed);
                    site.clone()
                }
                None => {
                    let name =
                        format!("{} (block {})[{}]", cx.func().qualname, unit.block, seed.root.1);
                    let site = Arc::new(GraphSite::new(name, seed, entry, self));
                    known.push((root, site.clone()));
                    site
                }
            };
            unit.graphs.push((site, first));
        }
        !unit.graphs.is_empty()
    }
}

/// The parts of torch whose functions graphs start in: the code of models.
/// Its other functions are library code around tensors: a graph gets into
/// them only along a path from outside.
const MODEL_CODE: [&str; 7] = ["nn/", "functional.py", "_tensor.py", "distributions/", "_functorch/", "func/", "optim/"];

/// Whether graphs start in the functions of a file.
fn starts_graphs(filename: &str) -> bool {
    static TORCH: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    let torch = TORCH.get_or_init(|| {
        let asked = |py: Python<'_>| -> PyResult<String> { py.import("torch")?.getattr("__file__")?.extract() };
        let file = unsafe { asked(Python::assume_attached()) }.unwrap_or_default();
        unsafe { ffi::PyErr_Clear() };
        file.rsplit_once('/').map_or(String::new(), |(dir, _)| format!("{}/", dir))
    });
    match filename.strip_prefix(torch.as_str()) {
        Some(rest) if !torch.is_empty() => MODEL_CODE.iter().any(|place| rest.starts_with(place)),
        _ => true,
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
        Op::Call { .. } | Op::CallMethod { .. } | Op::GetItem(..) if generic(unit, cx, i) => Some(Kind::Node),
        Op::Const(_)
        | Op::TensorSize { .. }
        | Op::TensorDim(_)
        | Op::TupleGet(..)
        | Op::IsNone(..)
        | Op::Is(..)
        | Op::IsFunction { .. }
        | Op::BuildTuple(_)
        | Op::BuildList(_)
        | Op::BuildSlice(..)
        | Op::CheckBound(..)
        | Op::Guard(_) => Some(Kind::Pure),
        Op::Binary(_, a, b) | Op::Compare(_, a, b) if numbers(&[*a, *b]) => Some(Kind::Pure),
        Op::Unary(_, a) | Op::ToBool(a) if numbers(&[*a]) => Some(Kind::Pure),
        Op::LoadGlobal(_) | Op::LoadDeref(_) | Op::CellGet(..) => Some(Kind::Read),
        Op::LoadAttr(..) | Op::GetItem(..) | Op::Contains(..) | Op::GetLen(_) | Op::Unpack(..) if quiet => {
            Some(Kind::Read)
        }
        Op::Hold(_) | Op::StoreLocal(..) | Op::LoadSlot(_) | Op::StoreSlot(..) => Some(Kind::Residual),
        _ => None,
    };
    match kind {
        Some(kind) => Role::Part(kind),
        None => Role::Break(why(unit, cx, i)),
    }
}

/// Whether an instruction that is no call site is an operation of a graph
/// all the same: a call of a C function, a method of a tensor or a subscript
/// of one that, whenever it was profiled, gave a tensor, ran kernels that
/// neither write to an argument nor draw random numbers, and entered no
/// frame.
fn generic(unit: &Unit, cx: &PassCx, i: usize) -> bool {
    let inst = &unit.insts[i].inst;
    let Some(found) = facts(cx, unit, i) else {
        return false;
    };
    let tensor = |v: V| matches!(cx.obs.get(v as usize), Some(ValueObs::Tensor { .. }));
    let fits = match &inst.op {
        // A method that was bound earlier has its receiver in no operand.
        Op::Call { callee, .. } => !matches!(unit.def_op(*callee), Some(Op::LoadAttr(recv, _)) if tensor(*recv)),
        Op::CallMethod { .. } => true,
        Op::GetItem(a, _) => tensor(*a),
        _ => false,
    };
    fits && inst.dst != NONE
        && tensor(inst.dst)
        && found.runs > 0
        && matches!(found.kernels, Kernels::One | Kernels::Many)
        && !(found.frames || found.raised || found.wrote)
        && matches!(found.callee, CalleeFact::Unseen)
        && !found.operators.is_empty()
        && found.operators.iter().all(|name| pure_operator(name))
}

/// What cannot fail and runs no Python code whatever it is given: behind
/// the callable of a path that wrote to a tensor from outside, nothing
/// else may run.
pub fn infallible(op: &Op) -> bool {
    matches!(
        op,
        Op::Const(_)
            | Op::Object(_)
            | Op::Hold(_)
            | Op::BuildTuple(_)
            | Op::BuildList(_)
            | Op::BuildSlice(..)
            | Op::ListAppend(..)
            | Op::TupleGet(..)
            | Op::IsNone(..)
            | Op::Is(..)
            | Op::TensorSize { .. }
            | Op::TensorDim(_)
    )
}

/// A list, a tuple or a slice that the instructions make.
fn container(op: &Op) -> bool {
    matches!(op, Op::BuildTuple(_) | Op::BuildList(_) | Op::BuildSlice(..))
}

/// Why an instruction is in no graph.
fn why(unit: &Unit, cx: &PassCx, i: usize) -> String {
    let u = &unit.insts[i];
    let reason = cx.reasons.get(u.origin as usize).and_then(|r| r.as_ref());
    why_not(cx.func(), &u.inst.op, facts(cx, unit, i), reason)
}

/// Why an instruction is no part of a graph, from what the profiling tier
/// saw it do (`found`) and why it is no call site (`reason`).
pub fn why_not(func: &Function, op: &Op, found: Option<&InstFacts>, reason: Option<&String>) -> String {
    let no_site = || match reason {
        Some(reason) => format!("runs kernels and is no call site: {}", reason),
        None => "runs kernels and is no call site".to_string(),
    };
    let kernels = found.is_some_and(|f| matches!(f.kernels, Kernels::One | Kernels::Many));
    match op {
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
            let text = func.op_repr(op);
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
    // What a container may hold: what the graph has, and whatever is there
    // in front of it, which is looked at when a run is exported.
    let held = |v: &V| defs.get(v).is_none_or(|def| !def.late || def.node || def.graph);
    match op {
        Op::TensorSize { t, .. } | Op::TensorDim(t) => defs.get(t).is_some_and(|def| def.node),
        Op::Binary(k, a, b) => INT_BINARY.contains(k) && fits(a) && fits(b),
        Op::Unary(UnaryKind::Neg, a) => fits(a),
        Op::BuildTuple(vs) | Op::BuildList(vs) => vs.iter().all(held),
        Op::BuildSlice(a, b, c) => [a, b, c].into_iter().all(|v| *v == NONE || held(v)),
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
        defs.insert(inst.dst, Def { node, late, graph, made: false, own: false, fixed: false });
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
        || container(op)
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
        } else if *kind == Kind::Node {
            // What is no call site may call more than one operator.
            operators.push(facts(cx, unit, i).map_or_else(|| "?".to_string(), |f| f.operators.join("+")));
        }
        body.push(PathInst {
            op: inst.op.clone(),
            dst: inst.dst,
            kind: *kind,
            bc: inst.bc,
            stage: Stage::Before,
            in_graph: false,
            assert: None,
            owner: 0,
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
            _ if needed && computed && !container(&inst.op) => {
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
    let containers: Vec<V> = body.iter().filter(|inst| container(&inst.op)).map(|inst| inst.dst).collect();
    let mut wanted: Vec<V> = Vec::new();
    for inst in body.iter().rev() {
        let given = inst.kind != Kind::Node && outputs.contains(&inst.dst);
        if given {
            wanted.push(inst.dst);
        }
        // An operation that is no call site is described with the lists,
        // tuples and slices it is given, item by item.
        let unfolds = (inst.kind == Kind::Node && !matches!(inst.op, Op::Dispatch(_)))
            || (container(&inst.op) && wanted.contains(&inst.dst));
        if inst.kind == Kind::Node || wanted.contains(&inst.dst) {
            reads(&inst.op, |v| {
                let made = unfolds && containers.contains(&v);
                if made || defs.get(&v).is_some_and(|def| def.late && !def.node) {
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
    let mut inputs: Vec<(V, PathValue)> = Vec::new();
    let mut feeds: Vec<V> = Vec::new();
    for inst in &body {
        inst.op.for_each_use(|v| {
            if v != UNDEF && !made.contains(&v) && !inputs.contains(&(v, v)) && inst.kind != Kind::Residual {
                inputs.push((v, v));
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
        resume: None,
        nregs: 0,
        inputs,
        body,
        feeds,
        outputs,
        exits,
        writes: Vec::new(),
        clears: Vec::new(),
        bound,
        operators,
        ends,
        shape: Vec::new(),
        owners: vec![cx.cf as *const Compiled],
        source: Vec::new(),
        branches: 0,
        calls: 0,
        first: None,
        provisional: false,
        effects: false,
        mutates: false,
        pinned: Vec::new(),
        constants: Vec::new(),
        exported: Shared::new(Vec::new()),
        registers: Shared::new(Vec::new()),
    }
}

// ----------------------------------------------------------------------
// Paths that are taken down
// ----------------------------------------------------------------------

/// What became of a value of the function along a path.
#[derive(Clone, Copy, PartialEq)]
pub enum Held {
    /// The path neither made it nor let go of it.
    Untouched,
    /// It holds this value of the path.
    Value(PathValue),
    /// It holds nothing anymore.
    Empty,
}

/// Where a store of a path puts what it stores: into the list or the dict
/// itself, into the dict the object keeps its attributes in, or into the
/// dict a module keeps its buffers in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Keeps {
    Itself,
    Attributes,
    Buffers,
}

/// The list or dict of an object that a store of that kind changes; null
/// where there is none.
pub unsafe fn kept(o: Obj, keeps: Keeps) -> Obj {
    if o.is_null() || capi::is_unboxed(o) {
        return null_mut();
    }
    if keeps == Keeps::Itself {
        return o;
    }
    let slot = pyffi::_PyObject_GetDictPtr(o);
    if slot.is_null() || (*slot).is_null() {
        return null_mut();
    }
    if keeps == Keeps::Attributes { *slot } else { ffi::PyDict_GetItemString(*slot, c"_buffers".as_ptr()) }
}

/// Whether what a read goes through is not what a store changes: `read`
/// is the list or dict that is read, or with `attributes` the object an
/// attribute is read of, whose dict and, for a module, the dicts of its
/// parameters, buffers and modules are looked into. Two objects may keep
/// their attributes in one dict, and a dict that is handed about may be
/// the one an object keeps its attributes in: what is compared is the
/// dict or list itself, and the objects only where one has no dict yet.
pub unsafe fn apart(read: Obj, attributes: bool, target: Obj, keeps: Keeps) -> bool {
    if read.is_null() || target.is_null() || read == target || capi::is_unboxed(read) {
        return false;
    }
    let changed = kept(target, keeps);
    if changed.is_null() {
        return true;
    }
    if !attributes {
        return read != changed;
    }
    let own = kept(read, Keeps::Attributes);
    let part = |name: &std::ffi::CStr| if own.is_null() { null_mut() } else { ffi::PyDict_GetItemString(own, name.as_ptr()) };
    ![own, part(c"_parameters"), part(c"_buffers"), part(c"_modules")].contains(&changed)
}

/// Whether what a tensor says of itself under a name is the same in front
/// of a write to it and behind: its sizes, its dtype, its device, and
/// whether it requires a gradient (a write that changes that, or the
/// sizes, is no operation of a path). Its version counter, the function
/// of autograd it came from and whatever else its class computes for a
/// name are not; a method and what the object keeps under a name itself
/// are nothing a write changes.
pub unsafe fn settled_attribute(tensor: Obj, name: Obj) -> bool {
    const SETTLED: [&str; 6] = ["shape", "ndim", "dtype", "device", "layout", "requires_grad"];
    let text = pyffi::str_to_string(name);
    if SETTLED.contains(&text.as_str()) || (text.starts_with("is_") && text != "is_leaf" && text != "is_inference") {
        return true;
    }
    static TENSOR: std::sync::OnceLock<Option<PyObj>> = std::sync::OnceLock::new();
    let class = if capi::is_unboxed(tensor) {
        let found = TENSOR.get_or_init(|| {
            let module = ffi::PyImport_ImportModule(c"torch".as_ptr());
            let class = if module.is_null() { null_mut() } else { ffi::PyObject_GetAttrString(module, c"Tensor".as_ptr()) };
            xdecref(module);
            ffi::PyErr_Clear();
            (!class.is_null()).then(|| PyObj::from_owned(class))
        });
        match found {
            Some(class) => class.as_ptr() as *mut ffi::PyTypeObject,
            None => return false,
        }
    } else {
        ffi::Py_TYPE(tensor)
    };
    let descr = pyffi::_PyType_Lookup(class, name);
    descr.is_null() || (*ffi::Py_TYPE(descr)).tp_descr_set.is_none()
}

/// For a tensor that shares its memory with no other of those looked at.
pub const UNSHARED: i64 = i64::MIN;

/// Which of some tensors share memory, and how they lie in it: per tensor
/// the first of them with the same memory, and where in it the tensor
/// starts if another of them has that memory too.
pub unsafe fn overlaps(tensors: &[Obj], api: &TcfgApi) -> Vec<(u32, i64)> {
    let mut memory: Vec<(*const std::ffi::c_void, i64)> = Vec::new();
    for tensor in tensors {
        let mut info = TcfgTensorInfo::empty();
        (api.tensor_info)(*tensor, &mut info);
        memory.push((info.data, info.storage_offset));
    }
    let first = |at: usize| memory.iter().position(|other| !memory[at].0.is_null() && other.0 == memory[at].0).unwrap_or(at);
    let firsts: Vec<usize> = (0..memory.len()).map(first).collect();
    (0..memory.len())
        .map(|at| {
            let shared = firsts.iter().filter(|other| **other == firsts[at]).count() > 1;
            (firsts[at] as u32, if shared { memory[at].1 } else { UNSHARED })
        })
        .collect()
}

/// A store that a path has taken down: the object as it was in the run
/// that was taken down and where it keeps what was stored, the name of the
/// attribute or the key of the dict (none for what changes a list), and
/// the values of the path that the object and what was stored are.
pub struct Stored {
    pub object: Obj,
    pub keeps: Keeps,
    pub key: Option<PyObj>,
    pub attribute: bool,
    pub target: PathValue,
    pub value: PathValue,
}

/// How an attribute store that runs no Python code is done.
#[derive(Clone, Copy, PartialEq)]
pub enum Store {
    /// As `object.__setattr__` does it.
    Plain,
    /// `nn.Module.__setattr__` of the name of a buffer: into that dict.
    Buffer(Obj),
}

/// What `nn.Module.__setattr__` is, the dict of the hooks it calls for a
/// buffer, and the classes it tells apart: `nn.Module` and `torch.Tensor`.
unsafe fn module_store() -> (usize, usize, usize, usize) {
    static FOUND: std::sync::OnceLock<(usize, usize, usize, usize)> = std::sync::OnceLock::new();
    *FOUND.get_or_init(|| {
        let torch = ffi::PyImport_ImportModule(c"torch".as_ptr());
        let tensor = if torch.is_null() { null_mut() } else { ffi::PyObject_GetAttrString(torch, c"Tensor".as_ptr()) };
        let module = ffi::PyImport_ImportModule(c"torch.nn.modules.module".as_ptr());
        let class = if module.is_null() { null_mut() } else { ffi::PyObject_GetAttrString(module, c"Module".as_ptr()) };
        let dict = if class.is_null() { null_mut() } else { ffi::PyObject_GetAttrString(class, c"__dict__".as_ptr()) };
        let setattr = if dict.is_null() { null_mut() } else { ffi::PyObject_GetItem(dict, specials().setattr.as_ptr()) };
        let hooks =
            if module.is_null() { null_mut() } else { ffi::PyObject_GetAttrString(module, c"_global_buffer_registration_hooks".as_ptr()) };
        ffi::PyErr_Clear();
        // The module keeps them.
        for held in [module, class, dict, setattr, hooks, torch, tensor] {
            xdecref(held);
        }
        (setattr as usize, hooks as usize, class as usize, tensor as usize)
    })
}

/// How `obj.name = value` is done if it runs no Python code and cannot
/// fail: on an object whose class stores attributes as `object` does and
/// has no descriptor of its own for the name, or on an `nn.Module` whose
/// class has `Module.__setattr__`, for the cases of that function that
/// come down to one store: a value that is a plain tensor or of a class
/// that is neither a tensor's nor a module's (so no parameter, module or
/// buffer object), under a name that is no parameter and no module, which
/// goes to the object, or, under the name of a buffer and being a plain
/// tensor or None with no hook registered for buffers, into the dict of
/// buffers.
/// None for everything else: a path ends in front of such a store.
pub unsafe fn store_kind(obj: Obj, name: Obj, value: Obj, api: &TcfgApi) -> Option<Store> {
    const KEPT: [&[u8]; 4] = [b"__dict__", b"__class__", b"_forward", b"_backward"];
    if obj.is_null() || value.is_null() || capi::is_unboxed(obj) || ffi::PyType_Check(obj) != 0 {
        return None;
    }
    let text = pyffi::str_to_string(name);
    if KEPT.iter().any(|kept| text.as_bytes().starts_with(kept)) || text.starts_with("_compiled") {
        return None;
    }
    let ty = ffi::Py_TYPE(obj);
    let slot = pyffi::_PyObject_GetDictPtr(obj);
    let descr = pyffi::_PyType_Lookup(ty, name);
    let data = !descr.is_null() && (*ffi::Py_TYPE(descr)).tp_descr_set.is_some();
    if slot.is_null() || data {
        return None;
    }
    let names = specials();
    let object = std::ptr::addr_of_mut!(ffi::PyBaseObject_Type);
    let setattr = pyffi::_PyType_Lookup(ty, names.setattr.as_ptr());
    if setattr == pyffi::_PyType_Lookup(object, names.setattr.as_ptr()) {
        return Some(Store::Plain);
    }
    let (module, hooks, layer, tensors) = module_store();
    if setattr.is_null() || setattr as usize != module || (*slot).is_null() || layer == 0 || tensors == 0 {
        return None;
    }
    // What the value is: a tensor that is no parameter and no buffer
    // object, or something that is neither a tensor nor a module.
    let mut info = TcfgTensorInfo::empty();
    let tensor = (api.tensor_info)(value, &mut info) != 0;
    let exact = tensor
        && (capi::is_unboxed(value) || {
            let own = pyffi::_PyObject_GetDictPtr(value);
            ffi::Py_TYPE(value) as usize == tensors && (own.is_null() || (*own).is_null() || ffi::PyDict_Size(*own) == 0)
        });
    let of_class = |class: usize| ffi::PyType_IsSubtype(ffi::Py_TYPE(value), class as *mut ffi::PyTypeObject) != 0;
    if !(exact || (!tensor && !capi::is_unboxed(value) && !of_class(layer) && !of_class(tensors))) {
        return None;
    }
    let of = |kind: &std::ffi::CStr| ffi::PyDict_GetItemString(*slot, kind.as_ptr());
    let holds = |dict: Obj| !dict.is_null() && ffi::PyDict_CheckExact(dict) != 0 && !ffi::PyDict_GetItemWithError(dict, name).is_null();
    let (parameters, modules, buffers) = (of(c"_parameters"), of(c"_modules"), of(c"_buffers"));
    let known = |dict: Obj| dict.is_null() || dict == ffi::Py_None() || ffi::PyDict_CheckExact(dict) != 0;
    if !known(parameters) || !known(modules) || !known(buffers) || holds(parameters) || holds(modules) {
        return None;
    }
    if !holds(buffers) {
        return Some(Store::Plain);
    }
    let hooked = hooks == 0 || ffi::PyObject_Size(hooks as Obj) != 0;
    ffi::PyErr_Clear();
    ((exact || value == ffi::Py_None()) && !hooked).then_some(Store::Buffer(buffers))
}

/// One activation of a function along a trace: the function, the value of
/// the path that the function object is (UNDEF for the first, whose
/// function the registers have), and per value of the function the value of
/// the path it is, or NONE.
pub struct Act {
    pub owner: u32,
    pub function: PathValue,
    pub map: Vec<PathValue>,
}

/// Where a trace was when its first function called a function that the
/// trace went into, or came to a call that is inlined in it: what it is
/// cut back to if the path ends in there.
pub struct Mark {
    /// The instruction: its block, the code, and its place in the code.
    pub at: (usize, *const BlockCode, usize),
    /// How many inlined calls the trace is in since then; 0 for the mark of
    /// a call it went into.
    pub frames: usize,
    stores: usize,
    flags: (bool, bool, bool),
    pinned: usize,
    body: usize,
    values: usize,
    nodes: usize,
    operators: usize,
    bound: usize,
    shape: usize,
    inputs: usize,
    wrote: bool,
    failed: Option<String>,
    map: Vec<PathValue>,
    held: Vec<Held>,
}

/// The instructions of a function from the first instruction of a graph on,
/// and of the functions it calls, as they are taken down while they run
/// (`Compiled::record`): renamed to the values of the path, which are
/// numbered from 1 in the order they are made or first read (0 is the
/// undefined value).
pub struct Trace {
    pub body: Vec<PathInst>,
    /// Per value of the path.
    pub defs: Vec<Def>,
    pub source: Vec<(u32, V)>,
    pub inputs: Vec<(V, PathValue)>,
    /// The activations the trace is in, the first function's first.
    pub acts: Vec<Act>,
    pub owners: Vec<*const Compiled>,
    /// Per value of the first function: what it holds where the trace has
    /// got to.
    pub held: Vec<Held>,
    /// The variables of the first function that were bound, in order.
    pub bound: Vec<(u32, PathValue)>,
    pub operators: Vec<String>,
    pub shape: Vec<u32>,
    pub rec: Recording,
    pub mark: Option<Mark>,
    /// The stores of the path so far (`Kind::Effect`), in the order of the
    /// function, which a read behind them has to see or be shown to go
    /// past; and what was already checked to be apart (`Assert::Apart`).
    pub stores: Vec<Stored>,
    pub apart: Vec<(PathValue, bool, PathValue, Keeps)>,
    /// The run is no run to make a form of (an operation that is no part
    /// of the path changed what kind of tensor one of its tensors is): the
    /// path is taken down again at the next call.
    pub again: bool,
    /// An operation of the path writes to a tensor from outside it (or
    /// draws random numbers): from there on nothing may be taken down
    /// that can fail behind the callable, since the write cannot be taken
    /// back. And the other way round: something that can fail behind the
    /// callable is on the trace, so no such write may follow.
    pub writes: bool,
    pub fallible: bool,
    /// An operation of the path writes to a tensor in place.
    pub inplace: bool,
    /// See `Path::first`.
    pub first: Option<(usize, u32)>,
    /// Tensors from outside the path that something computed behind the
    /// callable depends on the kind of: a form checks them as it checks
    /// the tensors its graph takes.
    pub pinned: Vec<PathValue>,
    /// Per inlined call the trace is in, in its first function: how many
    /// variables were bound when it began. Those bound in it are of its
    /// frame, which is gone when it ends.
    pub inlined: Vec<usize>,
    /// A called function read a value that the path does not have: the
    /// trace is of no use from there on.
    pub lost: bool,
    /// The path ends where a later one will go on: at a block, or a call
    /// of a function, that the tiers have not decided on yet.
    pub provisional: bool,
}

impl Trace {
    pub fn new(root: *const Compiled, nvalues: usize) -> Trace {
        Trace {
            body: Vec::new(),
            defs: vec![Def::default()],
            source: vec![(0, UNDEF)],
            inputs: Vec::new(),
            acts: vec![Act { owner: 0, function: UNDEF, map: vec![NONE; nvalues] }],
            owners: vec![root],
            held: vec![Held::Untouched; nvalues],
            bound: Vec::new(),
            operators: Vec::new(),
            shape: Vec::new(),
            rec: Recording { values: vec![null_mut()], born: vec![capi::TCFG_NO_VERSION], nodes: Vec::new(), failed: None, wrote: false },
            mark: None,
            stores: Vec::new(),
            apart: Vec::new(),
            again: false,
            writes: false,
            fallible: false,
            inplace: false,
            first: None,
            pinned: Vec::new(),
            inlined: Vec::new(),
            lost: false,
            provisional: false,
        }
    }

    fn act(&mut self) -> &mut Act {
        self.acts.last_mut().expect("a trace is in a function")
    }

    /// Whether the trace is in the function it started in.
    pub fn outermost(&self) -> bool {
        self.acts.len() == 1
    }

    pub fn owner(&self) -> u32 {
        self.acts.last().map_or(0, |act| act.owner)
    }

    /// The value of the path that the function it is in is: UNDEF in the
    /// function it started in.
    pub fn function(&self) -> PathValue {
        self.acts.last().map_or(UNDEF, |act| act.function)
    }

    /// A value that the path reads from before it, in the function it
    /// started in: `kept` is what it is in this run.
    pub fn input(&mut self, v: V, kept: Obj) -> PathValue {
        let made = self.value(Def::default(), kept);
        self.source[made as usize] = (0, v);
        self.acts[0].map[v as usize] = made;
        self.inputs.push((v, made));
        made
    }

    /// The value of the path that a value of the function it is in is, or
    /// NONE.
    pub fn mapped(&self, v: V) -> PathValue {
        self.acts.last().map_or(NONE, |act| act.map[v as usize])
    }

    /// A new value of the path that is no value of a function; `kept` is
    /// what it is in this run (a reference of its own, or null).
    pub fn value(&mut self, def: Def, kept: Obj) -> PathValue {
        let made = self.defs.len() as PathValue;
        self.defs.push(def);
        self.source.push((self.owner(), NONE));
        self.rec.values.push(kept);
        let mut info = TcfgTensorInfo::empty();
        let tensor = !kept.is_null() && capi::api().is_some_and(|api| unsafe { (api.tensor_info)(kept, &mut info) } != 0);
        self.rec.born.push(if tensor { info.version } else { capi::TCFG_NO_VERSION });
        made
    }

    /// A new value of the path, which a value of the function it is in is
    /// from here on.
    pub fn define(&mut self, v: V, def: Def, kept: Obj) -> PathValue {
        let made = self.value(def, kept);
        self.source[made as usize] = (self.owner(), v);
        self.rename(v, made);
        made
    }

    /// A value of the function becomes the one another is: a variable of an
    /// edge, a slot, a parameter, the result of a call.
    pub fn rename(&mut self, v: V, to: PathValue) {
        self.act().map[v as usize] = to;
        if self.outermost() {
            self.held[v as usize] = if to == UNDEF { Held::Empty } else { Held::Value(to) };
        }
    }

    /// The function lets go of a value.
    pub fn kill(&mut self, v: V) {
        if self.outermost() {
            self.held[v as usize] = Held::Empty;
        }
    }

    pub fn push(&mut self, op: Op, dst: PathValue, kind: Kind, bc: u32) {
        let owner = self.owner();
        self.body.push(PathInst { op, dst, kind, bc, stage: Stage::Before, in_graph: false, assert: None, owner });
    }

    pub fn check(&mut self, assert: Assert, bc: u32) {
        // What the instructions that follow were made from tells the paths
        // apart: how many items were taken out of a list, which names out
        // of a dict.
        match &assert {
            Assert::Sequence(_, class, count) => self.shape.extend([u32::MAX - 3, *class as u32, *count as u32]),
            Assert::Keys(_, names) => {
                self.shape.push(u32::MAX - 4);
                self.shape.extend(names.iter().map(|name| name.as_ptr() as usize as u32));
            }
            Assert::Same(a, b) => self.shape.extend([u32::MAX - 5, *a, *b]),
            Assert::Apart(a, attributes, b, keeps) => {
                self.shape.extend([u32::MAX - 6, *a, *attributes as u32, *b, *keeps as u32])
            }
            _ => {}
        }
        // A check of what the graph gives is made behind the callable,
        // and can fail there unless what it is of is the same whenever the
        // form is for the call.
        let unsure = |def: Def| def.late && !def.fixed;
        let mut late = unsure(self.defs[assert.value() as usize]);
        if let Assert::Same(_, other) | Assert::Apart(_, _, other, _) = &assert {
            late |= unsure(self.defs[*other as usize]);
        }
        if late {
            self.fails_late();
        }
        self.push(Op::Hold(Vec::new()), NONE, Kind::Check, bc);
        self.body.last_mut().expect("just pushed").assert = Some(assert);
    }

    /// Something that can fail behind the callable is taken down. Where
    /// the path has written to a tensor from outside by then, the run is
    /// not exported: those who take such things down ask `writes` first,
    /// and end the path in front of them.
    pub fn fails_late(&mut self) {
        self.fallible = true;
        if self.writes && self.rec.failed.is_none() {
            self.rec.failed = Some("a check behind a write to a tensor from outside the path".to_string());
        }
    }

    /// What an instruction of the path gives, from what it takes: a value
    /// that depends on a node if one of those does, and that the graph
    /// computes if `computed` says so.
    pub fn derive(&mut self, op: Op, kind: Kind, made: bool, kept: Obj, bc: u32) -> PathValue {
        self.derive_fixed(op, kind, made, kept, bc, false)
    }

    /// `derive` for what is the same in every call a form is for where
    /// `fixed` says so (`Def::fixed`): a method of a tensor the graph gives.
    pub fn derive_fixed(&mut self, op: Op, kind: Kind, made: bool, kept: Obj, bc: u32, fixed: bool) -> PathValue {
        let mut late = false;
        reads(&op, |v| late |= self.defs[v as usize].late);
        let graph = late && computed(&op, &self.defs, |_| false);
        let fixed = fixed || matches!(op, Op::Const(_) | Op::Object(_));
        if late && !graph && !fixed && !infallible(&op) {
            self.fails_late();
        }
        let dst = self.value(Def { node: false, late, graph, made, own: false, fixed }, kept);
        self.push(op, dst, kind, bc);
        dst
    }

    /// The trace goes into a function that a call enters.
    pub fn enter(&mut self, compiled: *const Compiled, nvalues: usize, function: PathValue) {
        let known = self.owners.iter().position(|other| *other == compiled);
        let owner = known.unwrap_or_else(|| {
            self.owners.push(compiled);
            self.owners.len() - 1
        }) as u32;
        self.acts.push(Act { owner, function, map: vec![NONE; nvalues] });
        // Which function it is tells the paths apart, not the place it has
        // among those of this trace.
        let which = compiled as usize as u64;
        self.shape.extend([u32::MAX - 1, which as u32, (which >> 32) as u32]);
    }

    pub fn leave(&mut self) {
        self.acts.pop();
        self.shape.push(u32::MAX - 2);
    }

    /// Takes down where the trace is, in its first function, in front of a
    /// call it goes into or of one that is inlined (`frames`: 1).
    pub fn mark(&mut self, at: (usize, *const BlockCode, usize), frames: usize) {
        self.mark = Some(Mark {
            at,
            frames,
            stores: self.stores.len(),
            flags: (self.writes, self.fallible, self.inplace),
            pinned: self.pinned.len(),
            body: self.body.len(),
            values: self.defs.len(),
            nodes: self.rec.nodes.len(),
            operators: self.operators.len(),
            bound: self.bound.len(),
            shape: self.shape.len(),
            inputs: self.inputs.len(),
            wrote: self.rec.wrote,
            failed: self.rec.failed.clone(),
            map: self.acts[0].map.clone(),
            held: self.held.clone(),
        });
    }

    /// Lets go of the mark of a call that the trace has come back from,
    /// into the function it started in.
    pub fn unmark(&mut self) {
        if self.outermost() && self.mark.as_ref().is_some_and(|mark| mark.frames == 0) {
            self.mark = None;
        }
    }

    /// Cuts the trace back to its mark: the path ends in front of the call.
    pub unsafe fn back(&mut self) {
        let Some(mark) = self.mark.take() else {
            return;
        };
        self.body.truncate(mark.body);
        self.stores.truncate(mark.stores);
        self.apart.clear();
        (self.writes, self.fallible, self.inplace) = mark.flags;
        if self.operators.is_empty() {
            self.first = None;
        }
        self.pinned.truncate(mark.pinned);
        self.defs.truncate(mark.values);
        self.source.truncate(mark.values);
        for kept in self.rec.values.drain(mark.values..) {
            xdecref(kept);
        }
        for (_, meta) in self.rec.nodes.drain(mark.nodes..) {
            xdecref(meta);
        }
        self.operators.truncate(mark.operators);
        self.bound.truncate(mark.bound);
        self.shape.truncate(mark.shape);
        self.inputs.truncate(mark.inputs);
        self.rec.wrote = mark.wrote;
        self.rec.failed = mark.failed;
        self.acts.truncate(1);
        self.acts[0].map = mark.map;
        self.held = mark.held;
        self.inlined.clear();
        self.lost = false;
    }
}

/// A copy of an instruction with its operands renamed, for the instructions
/// a path can hold; None for any other. A call site is renamed without its
/// fallback, which no path runs.
pub fn renamed(op: &Op, f: &mut impl FnMut(V) -> V) -> Option<Op> {
    let mut all = |vs: &[V]| vs.iter().map(|v| f(*v)).collect::<Vec<V>>();
    Some(match op {
        Op::Const(_) | Op::Object(_) | Op::LoadGlobal(_) | Op::LoadDeref(_) => op.clone(),
        Op::Part(part, v) => Op::Part(part.clone(), f(*v)),
        Op::Hold(_) => Op::Hold(Vec::new()),
        Op::BuildTuple(vs) => Op::BuildTuple(all(vs)),
        Op::BuildList(vs) => Op::BuildList(all(vs)),
        Op::BuildMap(vs) => Op::BuildMap(all(vs)),
        Op::Dispatch(d) => {
            let operands = all(&d.operands);
            Op::Dispatch(Box::new(DispatchOp { slot: d.slot.clone(), operands, fallback: Op::Hold(Vec::new()) }))
        }
        Op::Call { callee, args, kwnames } => {
            let callee = all(&[*callee])[0];
            Op::Call { callee, args: all(args), kwnames: *kwnames }
        }
        Op::CallMethod { recv, name, args, kwnames } => {
            let recv = all(&[*recv])[0];
            Op::CallMethod { recv, name: *name, args: all(args), kwnames: *kwnames }
        }
        Op::BuildSlice(a, b, c) => {
            let parts = all(&[*a, *b]);
            Op::BuildSlice(parts[0], parts[1], if *c == NONE { NONE } else { all(&[*c])[0] })
        }
        Op::CellGet(a, n) => Op::CellGet(f(*a), *n),
        Op::LoadAttr(a, n) => Op::LoadAttr(f(*a), *n),
        Op::Binary(k, a, b) => Op::Binary(*k, f(*a), f(*b)),
        Op::Unary(k, a) => Op::Unary(*k, f(*a)),
        Op::Compare(k, a, b) => Op::Compare(*k, f(*a), f(*b)),
        Op::Is(a, b, invert) => Op::Is(f(*a), f(*b), *invert),
        Op::Contains(a, b, invert) => Op::Contains(f(*a), f(*b), *invert),
        Op::ToBool(a) => Op::ToBool(f(*a)),
        Op::IsNone(a, invert) => Op::IsNone(f(*a), *invert),
        Op::GetItem(a, b) => Op::GetItem(f(*a), f(*b)),
        Op::ImportName { name, fromlist, level } => {
            Op::ImportName { name: *name, fromlist: f(*fromlist), level: f(*level) }
        }
        Op::SetItem { obj, key, val } => Op::SetItem { obj: f(*obj), key: f(*key), val: f(*val) },
        Op::StoreAttr { obj, name, val } => Op::StoreAttr { obj: f(*obj), name: *name, val: f(*val) },
        Op::ListAppend(a, b) => Op::ListAppend(f(*a), f(*b)),
        Op::ListExtend(a, b) => Op::ListExtend(f(*a), f(*b)),
        Op::DictMerge(a, b, c) => Op::DictMerge(f(*a), f(*b), f(*c)),
        Op::MapAdd(a, b, c) => Op::MapAdd(f(*a), f(*b), f(*c)),
        Op::ListToTuple(a) => Op::ListToTuple(f(*a)),
        Op::Unpack(a, n) => Op::Unpack(f(*a), *n),
        Op::TupleGet(a, i) => Op::TupleGet(f(*a), *i),
        Op::GetLen(a) => Op::GetLen(f(*a)),
        Op::GetIter(a) => Op::GetIter(f(*a)),
        Op::IterNext(a) => Op::IterNext(f(*a)),
        Op::CheckBound(a, n) => Op::CheckBound(f(*a), *n),
        Op::IsFunction { v, func, code } => Op::IsFunction { v: f(*v), func: *func, code: *code },
        Op::TensorSize { t, dim } => Op::TensorSize { t: f(*t), dim: *dim },
        Op::TensorDim(t) => Op::TensorDim(f(*t)),
        _ => return None,
    })
}

pub fn renamed_guard(cond: &GuardCond, f: &mut impl FnMut(V) -> V) -> GuardCond {
    match cond {
        GuardCond::TensorRank { v, min_rank } => GuardCond::TensorRank { v: f(*v), min_rank: *min_rank },
        GuardCond::DimRange { v, dim, lo, hi } => GuardCond::DimRange { v: f(*v), dim: *dim, lo: *lo, hi: *hi },
    }
}

/// Whether the graph computes what an instruction of a path gives from
/// values that depend on its nodes: a size of a tensor of its own,
/// arithmetic on such sizes and other integers (`int` says of a value from
/// in front of the nodes whether it is one), a list, tuple or slice of what
/// it has.
pub fn computed(op: &Op, defs: &[Def], int: impl Fn(PathValue) -> bool) -> bool {
    let fits = |v: &PathValue| match defs[*v as usize] {
        def if def.late => def.graph,
        _ => int(*v),
    };
    let held = |v: &PathValue| {
        let def = defs[*v as usize];
        !def.late || def.node || def.graph
    };
    match op {
        Op::TensorSize { t, .. } | Op::TensorDim(t) => defs[*t as usize].node,
        Op::Binary(k, a, b) => INT_BINARY.contains(k) && fits(a) && fits(b),
        Op::Unary(UnaryKind::Neg, a) => fits(a),
        Op::BuildTuple(vs) | Op::BuildList(vs) => vs.iter().all(held),
        Op::BuildSlice(a, b, c) => [a, b, c].into_iter().all(|v| *v == NONE || held(v)),
        // What an operation with several results gave, taken apart: the
        // tuple it is, and an item of it.
        Op::Unpack(v, _) => defs[*v as usize].node,
        Op::TupleGet(t, _) => defs[*t as usize].node || defs[*t as usize].graph,
        Op::GetItem(v, at) => defs[*v as usize].node && !defs[*at as usize].late && int(*at),
        _ => false,
    }
}

impl Path {
    /// A path without instructions: what a graph that starts with the
    /// function is before a path was taken down.
    pub fn empty(root: (usize, usize), land: (usize, usize)) -> Path {
        Path {
            root,
            land,
            resume: None,
            nregs: 0,
            inputs: Vec::new(),
            body: Vec::new(),
            feeds: Vec::new(),
            outputs: Vec::new(),
            exits: Vec::new(),
            writes: Vec::new(),
            clears: Vec::new(),
            bound: Vec::new(),
            operators: Vec::new(),
            ends: String::new(),
            shape: Vec::new(),
            owners: Vec::new(),
            source: Vec::new(),
            branches: 0,
            calls: 0,
            first: None,
            provisional: false,
            effects: false,
            mutates: false,
            pinned: Vec::new(),
            constants: Vec::new(),
            exported: Shared::new(Vec::new()),
            registers: Shared::new(Vec::new()),
        }
    }

    /// The path of a trace that ended in front of an instruction of a block
    /// (or of a block as a whole): `live` are the values of the function
    /// that are read from there on. Decides when each instruction runs once
    /// a callable stands for the nodes (`Stage`), and with that what the
    /// callable is handed and what it gives.
    #[allow(clippy::too_many_arguments)]
    pub fn of(
        trace: &mut Trace,
        root: (usize, usize),
        land: (usize, usize),
        resume: Option<(*const BlockCode, usize)>,
        live: &BitSet,
        ends: String,
        provisional: bool,
    ) -> Path {
        let defs = &trace.defs;
        let nregs = defs.len();
        // What is read behind the path of what it made, and so has to be
        // there when it ends.
        let mut need = vec![false; nregs];
        let (mut writes, mut clears) = (Vec::new(), Vec::new());
        for (v, held) in trace.held.iter().enumerate() {
            match held {
                Held::Untouched => {}
                Held::Value(made) if live.contains(v as V) => {
                    need[*made as usize] = true;
                    writes.push((*made, v as V));
                }
                _ => clears.push(v as V),
            }
        }
        // The lists, tuples and slices that the path builds.
        let mut built = vec![false; nregs];
        for inst in trace.body.iter().filter(|inst| inst.dst != NONE && container(&inst.op)) {
            built[inst.dst as usize] = true;
        }
        // The values that are no tensors and that the graph computes.
        let mut wanted = vec![false; nregs];
        let (mut outputs, mut exits) = (Vec::new(), Vec::new());
        let (mut branches, mut calls) = (0, 0);
        let constants: std::cell::RefCell<Vec<PathValue>> = std::cell::RefCell::new(Vec::new());
        for inst in trace.body.iter_mut().rev() {
            let dst = inst.dst;
            let needed = dst != NONE && need[dst as usize];
            let def = if dst == NONE { Def::default() } else { defs[dst as usize] };
            // What an operation of the graph, or a value the graph computes,
            // takes: of what depends on a node, the graph computes it too,
            // and it has the lists the path builds as what they hold; the
            // rest is there in front of the callable.
            let takes = |op: &Op, wanted: &mut [bool], need: &mut [bool]| {
                reads(op, |v| match defs[v as usize] {
                    // Computed behind the callable and the same whenever
                    // the form is for the call: a constant of the graph.
                    taken if taken.late && taken.fixed && !taken.node && !taken.graph => {
                        need[v as usize] = true;
                        if !constants.borrow().contains(&v) {
                            constants.borrow_mut().push(v);
                        }
                    }
                    taken if (taken.late && !taken.node) || built[v as usize] => wanted[v as usize] = true,
                    taken if !taken.late => need[v as usize] = true,
                    _ => {}
                })
            };
            let uses = |op: &Op, need: &mut [bool]| {
                let mut late = false;
                op.for_each_use(|v| {
                    need[v as usize] = true;
                    late |= defs[v as usize].late;
                });
                if late { Stage::After } else { Stage::Before }
            };
            inst.stage = match inst.kind {
                Kind::Node => {
                    if needed {
                        outputs.push(dst);
                    }
                    takes(&inst.op, &mut wanted, &mut need);
                    Stage::Inside
                }
                Kind::Check => {
                    let assert = inst.assert.as_ref().expect("a check has what it checks");
                    let v = assert.value() as usize;
                    match assert {
                        // What is the same in every call a form is for was
                        // checked when the path was taken down: the tensor
                        // it is computed from need not be given back for it.
                        Assert::Truth(..) | Assert::Null(..) | Assert::Guard(_) if defs[v].late && defs[v].fixed => {
                            branches += matches!(assert, Assert::Truth(..) | Assert::Null(..)) as usize;
                            Stage::Skipped
                        }
                        // A guard on what a node returns looks at what the
                        // graph says of it.
                        Assert::Guard(cond) if defs[v].node => {
                            exits.push(cond.clone());
                            Stage::Inside
                        }
                        _ => {
                            branches += matches!(assert, Assert::Truth(..) | Assert::Null(..)) as usize;
                            calls += matches!(assert, Assert::Enters(..)) as usize;
                            need[v] = true;
                            let mut late = defs[v].late;
                            if let Assert::Same(_, other) | Assert::Apart(_, _, other, _) = assert {
                                need[*other as usize] = true;
                                late |= defs[*other as usize].late;
                            }
                            if late { Stage::After } else { Stage::Before }
                        }
                    }
                }
                // What it stores is there when the path lands: what the
                // graph computed is one of its results.
                Kind::Effect => {
                    inst.op.for_each_use(|v| need[v as usize] = true);
                    Stage::Landing
                }
                Kind::Residual => {
                    // What changes something the path made goes with it.
                    let mut target = NONE;
                    inst.op.for_each_use(|v| {
                        if target == NONE {
                            target = v;
                        }
                    });
                    let kept = needed || (dst == NONE && target != NONE && need[target as usize]);
                    if kept && !matches!(inst.op, Op::Hold(_) | Op::StoreLocal(..)) {
                        uses(&inst.op, &mut need)
                    } else {
                        Stage::Skipped
                    }
                }
                _ if dst != NONE && wanted[dst as usize] => {
                    inst.in_graph = true;
                    takes(&inst.op, &mut wanted, &mut need);
                    if needed && container(&inst.op) {
                        // What is read behind the path is made there.
                        uses(&inst.op, &mut need)
                    } else {
                        if needed {
                            outputs.push(dst);
                        }
                        Stage::Inside
                    }
                }
                // An integer the graph computes is one it can give.
                _ if needed && def.late && def.graph && !container(&inst.op) => {
                    inst.in_graph = true;
                    outputs.push(dst);
                    takes(&inst.op, &mut wanted, &mut need);
                    Stage::Inside
                }
                _ if needed => uses(&inst.op, &mut need),
                _ => Stage::Skipped,
            };
        }
        outputs.reverse();
        exits.reverse();
        let effects = trace.body.iter().any(|inst| inst.stage == Stage::Landing);
        let constants = constants.into_inner();
        let mut feeds: Vec<PathValue> = Vec::new();
        for inst in trace.body.iter().filter(|inst| inst.kind == Kind::Node || inst.in_graph) {
            reads(&inst.op, |v| {
                if !defs[v as usize].late && !wanted[v as usize] && !feeds.contains(&v) {
                    feeds.push(v);
                }
            });
        }
        // The variables the instructions bind are bound at the end: each to
        // what it was bound to last, if that is there then.
        let mut there = vec![false; nregs];
        trace.inputs.iter().for_each(|(_, v)| there[*v as usize] = true);
        outputs.iter().for_each(|v| there[*v as usize] = true);
        for inst in trace.body.iter().filter(|inst| matches!(inst.stage, Stage::Before | Stage::After)) {
            if inst.dst != NONE {
                there[inst.dst as usize] = true;
            }
        }
        let mut bound: Vec<(u32, PathValue)> = Vec::new();
        for (slot, v) in &trace.bound {
            bound.retain(|(other, _)| other != slot);
            bound.push((*slot, if there[*v as usize] { *v } else { UNDEF }));
        }
        Path {
            root,
            land,
            resume,
            nregs,
            inputs: trace.inputs.clone(),
            body: std::mem::take(&mut trace.body),
            feeds,
            outputs,
            exits,
            writes,
            clears,
            bound,
            operators: std::mem::take(&mut trace.operators),
            ends,
            shape: std::mem::take(&mut trace.shape),
            owners: trace.owners.clone(),
            source: trace.source.clone(),
            branches,
            calls,
            first: trace.first,
            provisional,
            effects,
            mutates: trace.writes,
            pinned: std::mem::take(&mut trace.pinned),
            constants,
            exported: Shared::new(Vec::new()),
            registers: Shared::new(Vec::new()),
        }
    }
}
