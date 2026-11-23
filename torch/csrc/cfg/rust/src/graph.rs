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
}

impl Assert {
    pub fn value(&self) -> PathValue {
        match self {
            Assert::Truth(v, _) | Assert::Null(v, _) => *v,
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
    pub fn text(&self, func: &Function, path: &Path) -> String {
        let value = func.value_name(path.source.get(self.value as usize).copied().unwrap_or(self.value));
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
    /// Per value: the value of the function it stood for when it was made.
    pub source: Vec<V>,
    /// How many branches and loop tests it goes through.
    pub branches: usize,
    /// The block and the instruction of its first operation, where that
    /// is one of the function it starts in: what the graph is named by.
    pub first: Option<(usize, u32)>,
    /// It ended at a block that had not been profiled yet: the next run
    /// gets further.
    pub provisional: bool,
    pub exported: Shared<Vec<Arc<Exported>>>,
    /// Register files for the calls that run it, kept from one to the next.
    pub registers: Shared<Vec<Vec<Obj>>>,
    /// Which instructions of `body` run in front of the callable and
    /// behind it, in order (`Path::staged`).
    pub stages: [Vec<u32>; 2],
    /// The values of the function that the path takes and hands on: it
    /// writes them to a value of the function or binds a variable to
    /// them, as they are or as an item of a tuple it builds. What such a
    /// value holds is held by the function when the path has landed.
    pub handed: Vec<V>,
    /// Per register: an instruction of the path made what it holds
    /// (`Def::made`), which is the path's own to change and to use up.
    pub made: Vec<bool>,
}

// The registers are empty between calls, and only touched with the GIL held.
unsafe impl Send for Path {}
unsafe impl Sync for Path {}

/// The instructions of a path by the stage they run in (`Path::stages`).
pub fn stages_of(body: &[PathInst]) -> [Vec<u32>; 2] {
    let mut stages = [Vec::new(), Vec::new()];
    for (at, inst) in body.iter().enumerate() {
        match inst.stage {
            Stage::Before => stages[0].push(at as u32),
            Stage::After => stages[1].push(at as u32),
            Stage::Inside | Stage::Skipped => {}
        }
    }
    stages
}

/// The values of the function that a path hands on (`Path::handed`).
fn handed_of(inputs: &[(V, PathValue)], body: &[PathInst], writes: &[(PathValue, V)], bound: &[(u32, PathValue)]) -> Vec<V> {
    let mut handed: std::collections::HashSet<PathValue> = writes.iter().map(|(made, _)| *made).collect();
    handed.extend(bound.iter().map(|(_, made)| *made));
    for inst in body.iter().rev() {
        if let (Op::BuildTuple(items), true) = (&inst.op, handed.contains(&inst.dst)) {
            handed.extend(items.iter().copied());
        }
    }
    inputs.iter().filter(|(_, made)| handed.contains(made)).map(|(v, _)| *v).collect()
}

impl Path {
    /// The instructions that run in one stage: in front of the callable or
    /// behind it.
    pub fn staged(&self, stage: Stage) -> impl Iterator<Item = &PathInst> + '_ {
        let at = match stage {
            Stage::Before => 0,
            Stage::After => 1,
            Stage::Inside | Stage::Skipped => unreachable!("no instruction runs in this stage"),
        };
        self.stages[at].iter().map(|k| &self.body[*k as usize])
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
/// How many calls in a row may find that a graph cannot take them for
/// what its path lets go of (a path that is turned down where it would
/// land, a path that is not taken down to its end) before the graph
/// gives way to the graphs inside it. A call that runs a path of the
/// graph starts the count anew; one that takes a path down does not.
const MAX_TURNED: u32 = 3;

/// How many exported forms that differ in nothing but numbers, and of
/// which none ever ran again, say that the numbers are new on every call.
/// Three: the layers of a model that call one function with two or three
/// settings each keep their forms.
const EVER_NEW: usize = 3;

pub const EVER_NEW_REASON: &str = "its inputs take ever new values";

/// What is said of a graph that gave way (`GraphSite::gave_way`).
pub const GAVE_WAY_REASON: &str = "where its path lands it would let go of an object whose end may run Python code";

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

/// How many times a graph is taken down again because its path ended at a
/// block that had not been profiled.
const MAX_PROVISIONAL: u32 = 3;

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
    /// How many calls in a row a path of the graph was turned down for
    /// what it would let go of where it lands; from `MAX_TURNED` on the
    /// graph has given way (`gave_way`).
    turned: AtomicU32,
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
    /// Where graphs are not exported: a graph that starts further on has
    /// asked for another path of this one (`asked`), and how many times.
    again: AtomicBool,
    asks: AtomicU32,
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
            turned: AtomicU32::new(0),
            provisional: AtomicU32::new(0),
            exports: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            within: Shared::new(std::sync::Weak::new()),
            missed: AtomicU64::new(u64::MAX),
        }
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
        let there = !matches!(*outer.state.borrow(), GraphState::Small(_)) && !outer.gave_way();
        there.then_some(outer)
    }

    /// Whether `Op::Graph` has anything to do for it.
    pub fn runs(&self) -> bool {
        self.outer().is_none() && !self.gave_way() && (self.records() || self.forms() != (0, 0))
    }

    /// A path of the graph was turned down for what it would let go of
    /// where it lands (an object whose end may run Python code), or was
    /// not taken down to its end for such an object. Where that happens
    /// `MAX_TURNED` calls in a row, the graph gives way: it is not
    /// entered again, and the graphs that start further on, which were
    /// part of it, are their own again.
    pub fn turned_down(&self) {
        self.turned.fetch_add(1, Ordering::Relaxed);
    }

    /// Whether the graph has given way (`turned_down`).
    pub fn gave_way(&self) -> bool {
        self.turned.load(Ordering::Relaxed) >= MAX_TURNED
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
        if !self.gave_way() {
            self.turned.store(0, Ordering::Relaxed);
        }
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

/// Whether letting go of a reference to an object runs no Python code even
/// if it is the last one, as far as that can be shown by looking: neither
/// the object nor anything it refers to, and so may take with it, is of a
/// class with a finalizer (`__del__`, a generator's, a file's), is a
/// tensor that requires a gradient (`holds_unseen`), or has a
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
        if finalized(held) || weakly_watched(held, &mut more) {
            return false;
        }
        if is_tensor(held) {
            match unseen(held) {
                Ok(gradient) => more.extend(gradient),
                Err(()) => return false,
            }
        }
        found.extend(more[before..].iter().map(|other| other.as_ptr()));
        match (*ty).tp_traverse {
            Some(traverse) => {
                traverse(held, every_referent, &mut found as *mut Vec<Obj> as *mut std::ffi::c_void);
            }
            None if !holds_nothing(held) => return false,
            None => {}
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

/// Who holds references that are not the function's while a path is run
/// or taken down, and so go when that is over: for each of `asked`, no
/// fewer than those references (None where that cannot be told); and
/// whether an object is one that the path made: a tensor an operation of
/// it gave, a list, a dict, an iterator or an object of a class that an
/// instruction of it made.
pub trait Holds {
    unsafe fn held(&mut self, asked: &[Obj]) -> Option<Vec<isize>>;
    unsafe fn made(&mut self, o: Obj) -> bool;
}

impl Holds for Scan<'_> {
    unsafe fn held(&mut self, asked: &[Obj]) -> Option<Vec<isize>> {
        Scan::held(self, asked)
    }

    unsafe fn made(&mut self, _o: Obj) -> bool {
        false
    }
}

/// What a trace keeps of a run (`Recording::kept`), with the values of
/// the path that an operation gave.
pub struct Kept<'a> {
    pub kept: &'a HashMap<usize, (isize, PathValue)>,
    pub defs: &'a [Def],
}

impl Holds for Kept<'_> {
    unsafe fn held(&mut self, asked: &[Obj]) -> Option<Vec<isize>> {
        Some(asked.iter().map(|o| self.kept.get(&(*o as usize)).map_or(0, |(count, _)| *count)).collect())
    }

    unsafe fn made(&mut self, o: Obj) -> bool {
        self.kept.get(&(o as usize)).is_some_and(|(_, first)| self.defs.get(*first as usize).is_some_and(|def| def.node || def.made))
    }
}

/// The values of a path, for counting the references the path itself may
/// hold to an object: the registers of the path (`own`), and what the
/// function had before the path (`given`: such a value is not the path's).
pub struct Scan<'a> {
    own: &'a [Obj],
    given: &'a [Obj],
    /// What the registers hold, in order of where it is, once looked at.
    slots: Option<Vec<Obj>>,
    /// The values that may be something the path made and that hold
    /// references: containers, iterators and bound methods of the
    /// interpreter's own classes.
    temps: Vec<Obj>,
}

impl<'a> Scan<'a> {
    pub fn new(own: &'a [Obj], given: &'a [Obj]) -> Scan<'a> {
        Scan { own, given, slots: None, temps: Vec::new() }
    }

    /// For each of `asked`, no fewer than the references the path itself
    /// holds to it: the registers that hold it, and every reference to it
    /// from a value that may be something the path made, or from what
    /// such a value holds inside. None where there is too much to look at
    /// (4096 objects, or containers more than four deep inside one
    /// another).
    pub unsafe fn held(&mut self, asked: &[Obj]) -> Option<Vec<isize>> {
        const LOOKED_AT: usize = 4096;
        if self.slots.is_none() {
            let mut slots: Vec<Obj> = self.own.iter().copied().filter(|held| !held.is_null() && !capi::is_unboxed(*held)).collect();
            slots.sort_unstable();
            let mut temps = slots.clone();
            temps.dedup();
            temps.retain(|other| !self.given.contains(other) && holds_others(*other));
            self.temps = temps;
            self.slots = Some(slots);
        }
        let slots = self.slots.as_ref().expect("just made");
        let mut counts: Vec<isize> = asked
            .iter()
            .map(|o| (slots.partition_point(|other| *other <= *o) - slots.partition_point(|other| *other < *o)) as isize)
            .collect();
        let index: HashMap<usize, usize> = asked.iter().enumerate().map(|(i, o)| (*o as usize, i)).collect();
        let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
        let mut work: Vec<(Obj, u32)> = self.temps.iter().map(|other| (*other, 0)).collect();
        let mut found: Vec<Obj> = Vec::new();
        while let Some((at, depth)) = work.pop() {
            if seen.len() >= LOOKED_AT {
                return None;
            }
            seen.insert(at as usize);
            let Some(traverse) = (*ffi::Py_TYPE(at)).tp_traverse else {
                continue;
            };
            found.clear();
            traverse(at, every_referent, &mut found as *mut Vec<Obj> as *mut std::ffi::c_void);
            for other in &found {
                if let Some(i) = index.get(&(*other as usize)) {
                    counts[*i] += 1;
                } else if slots.binary_search(other).is_err() && holds_others(*other) && !seen.contains(&(*other as usize)) {
                    if depth >= 4 {
                        return None;
                    }
                    work.push((*other, depth + 1));
                }
            }
        }
        Some(counts)
    }
}

/// Whether the objects that a path lets go of where it lands (`gone`, each
/// as often as the path lets go of a reference to it; put in order here)
/// end without Python code running, as far as they end at all. Looked at
/// as the garbage collector looks at a set of objects: an object ends if
/// its count of references is no larger than the references to it that
/// the path lets go of, those from objects that end, and those the path
/// itself holds (`scan`; none where the path has taken nothing of the
/// call yet: every reference is then somebody else's or one that is let
/// go of); else one of its references is somebody else's and it stays,
/// with everything it refers to, and is not looked into. `stay` are
/// objects that are known to stay, `more` further objects to look at, of
/// which no reference is let go of. What ends must be of a class without a
/// finalizer and have no weak reference with a callback other than a weak
/// container's own (`weakly_watched`); what it refers to, and what a weak
/// container lets go of with its entry, is looked at in the same way.
/// Nothing of this is remembered from one call to the next: which
/// objects are let go of, how many references each has, what class it is
/// of and what that class does at the end of an object, the weak
/// references to it and what it refers to are all read at the call. Err
/// with the object whose end may run Python code, or with null where
/// there is too much to look at (4096 objects that end, or `Scan::held`
/// says so).
pub unsafe fn end_quietly(gone: &mut [Obj], stay: &[Obj], more: &[Obj], mut scan: Option<&mut dyn Holds>) -> Result<(), Obj> {
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
    if scan.is_none() {
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
    }
    // Per object that was reached: the references to it that are neither
    // the path's nor let go of by it (counted once, when it is reached),
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
    let nothing = more.iter().filter(|o| gone.binary_search(o).is_err()).map(|o| (*o, 0));
    for (o, count) in runs(gone).into_iter().chain(nothing) {
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
        if !fresh.is_empty() {
            let asked: Vec<Obj> = fresh.iter().map(|(o, _)| *o).collect();
            let held = match scan.as_deref_mut() {
                Some(scan) => scan.held(&asked).ok_or(null_mut::<ffi::PyObject>())?,
                None => vec![0; asked.len()],
            };
            for ((o, dropped), held) in fresh.drain(..).zip(held) {
                let kept_here = more.iter().filter(|other| other.as_ptr() == o).count() as isize;
                let entry = reached.get_mut(&(o as usize)).expect("it was reached");
                entry.others = ffi::Py_REFCNT(o) - kept_here - held - dropped;
            }
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
            // Python code: unless the operations of the path made it,
            // whose history and memory are theirs.
            let tensor = is_tensor(o);
            let made = scan.as_deref_mut().is_some_and(|scan| scan.made(o));
            if tensor && !made {
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
                // What shows nothing may hold what cannot be seen: unless
                // the path made it, of what its instructions are known
                // to go over.
                None if !made && !holds_nothing(o) => return Err(o),
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

/// Whether `bool(o)` runs no Python code.
pub unsafe fn quiet_truth(o: Obj) -> bool {
    !capi::is_unboxed(o) && (is_plain(o) || quiet_sized(o))
}

/// Whether two values compare in C code: numbers, texts, or dtypes and
/// devices.
pub unsafe fn quiet_compare(a: Obj, b: Obj) -> bool {
    let text = |o: Obj| !capi::is_unboxed(o) && ffi::PyUnicode_CheckExact(o) != 0;
    let token = |o: Obj| !capi::is_unboxed(o) && is_token(o);
    (is_number(a) && is_number(b)) || (text(a) && text(b)) || (token(a) && token(b))
}

/// Whether iterating over a value runs no Python code: Some(false) for a
/// list, a tuple, a dict, a str or a range; Some(true) for an `enumerate`
/// or a `zip`, of which that is only known where a path made it (see
/// `pure_call`).
pub unsafe fn quiet_iter(o: Obj) -> Option<bool> {
    if capi::is_unboxed(o) {
        return None;
    }
    let ty = ffi::Py_TYPE(o);
    let plain = ffi::PyList_CheckExact(o) != 0
        || ffi::PyTuple_CheckExact(o) != 0
        || ffi::PyDict_CheckExact(o) != 0
        || ffi::PyUnicode_CheckExact(o) != 0
        || ty == std::ptr::addr_of_mut!(ffi::PyRange_Type);
    if plain {
        return Some(false);
    }
    let made = [Builtin::Enumerate, Builtin::Zip].iter().any(|which| builtin_named(*which) == ty as usize);
    made.then_some(true)
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
}

const BUILTINS: [(&std::ffi::CStr, Builtin); 5] = [
    (c"len", Builtin::Len),
    (c"range", Builtin::Range),
    (c"enumerate", Builtin::Enumerate),
    (c"zip", Builtin::Zip),
    (c"isinstance", Builtin::IsInstance),
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

/// Whether a call is one of a function a path may call, with arguments on
/// which it runs no Python code: Some, with whether it makes something to
/// iterate over that is the path's own (a `range`, an `enumerate`, a
/// `zip`).
pub unsafe fn pure_call(callee: Obj, args: &[Obj], api: &TcfgApi) -> Option<bool> {
    if callee.is_null() || args.iter().any(|o| o.is_null()) {
        return None;
    }
    let which = BUILTINS.iter().map(|(_, which)| *which).find(|which| builtin_named(*which) == callee as usize)?;
    let sequence = |o: Obj| !capi::is_unboxed(o) && (ffi::PyList_CheckExact(o) != 0 || ffi::PyTuple_CheckExact(o) != 0);
    let fine = match which {
        Builtin::Len => {
            let mut info = TcfgTensorInfo::empty();
            args.len() == 1 && (quiet_sized(args[0]) || (api.tensor_info)(args[0], &mut info) != 0)
        }
        Builtin::Range => {
            (1..=3).contains(&args.len()) && args.iter().all(|o| !capi::is_unboxed(*o) && ffi::PyLong_CheckExact(*o) != 0)
        }
        Builtin::Enumerate => args.len() == 1 && sequence(args[0]),
        Builtin::Zip => !args.is_empty() && args.iter().all(|o| sequence(*o)),
        Builtin::IsInstance => {
            static CLASS: std::sync::OnceLock<PyObj> = std::sync::OnceLock::new();
            let name = CLASS.get_or_init(|| PyObj::from_owned(ffi::PyUnicode_InternFromString(c"__class__".as_ptr())));
            // A class whose instances `type` itself tells.
            let class = |o: Obj| {
                !capi::is_unboxed(o) && ffi::Py_TYPE(o) == std::ptr::addr_of_mut!(ffi::PyType_Type)
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
    fine.then_some(matches!(which, Builtin::Range | Builtin::Enumerate | Builtin::Zip))
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

/// Whether the lookup of `object` finds nothing under a name of an object,
/// and finds that out without running Python code: the class has nothing
/// under the name, and the object keeps what it has under texts. What
/// `object.__getattribute__(obj, name)` then does is raise AttributeError.
pub unsafe fn generic_absent(obj: Obj, name: Obj) -> bool {
    let plain = |o: Obj| !o.is_null() && !capi::is_unboxed(o);
    if !plain(obj) || ffi::PyType_Check(obj) != 0 || !plain(name) || ffi::PyUnicode_CheckExact(name) == 0 {
        return false;
    }
    if !pyffi::_PyType_Lookup(ffi::Py_TYPE(obj), name).is_null() {
        return false;
    }
    let slot = pyffi::_PyObject_GetDictPtr(obj);
    slot.is_null() || (*slot).is_null() || (pyffi::text_keys(*slot) && ffi::PyDict_GetItemWithError(*slot, name).is_null())
}

/// Whether an object whose attributes are read as `object` reads them,
/// with no function of its class asked behind that, has nothing under a
/// name: reading it raises AttributeError and does nothing else.
pub unsafe fn plainly_absent(obj: Obj, name: Obj) -> bool {
    if obj.is_null() || capi::is_unboxed(obj) || ffi::PyModule_Check(obj) != 0 {
        return false;
    }
    let object = std::ptr::addr_of!(ffi::PyBaseObject_Type);
    let generic = (*ffi::Py_TYPE(obj)).tp_getattro.map(|own| own as usize) == (*object).tp_getattro.map(|own| own as usize);
    generic && generic_absent(obj, name)
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

/// Whether an operator, as the dispatcher names it (`aten::cat`,
/// `aten::add_.Tensor`), neither writes to an argument nor draws random
/// numbers: its schema and its tags are asked once.
pub fn pure_operator(name: &str) -> bool {
    static KNOWN: std::sync::OnceLock<Shared<HashMap<String, bool>>> = std::sync::OnceLock::new();
    let known = KNOWN.get_or_init(|| Shared::new(HashMap::new()));
    if let Some(pure) = known.borrow().get(name) {
        return *pure;
    }
    let (qualified, overload) = match name.split_once("::").and_then(|(_, rest)| rest.find('.')) {
        Some(dot) => {
            let at = name.len() - (name.split_once("::").unwrap().1.len() - dot);
            (&name[..at], &name[at + 1..])
        }
        None => (name, ""),
    };
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
    /// Per object among `values`: how many of them it is (each is a
    /// reference the trace holds beyond those of the function), and the
    /// first value it is.
    pub kept: HashMap<usize, (isize, PathValue)>,
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
}

impl Path {
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
    pub unsafe fn describe(&self, name: &str, func: &Function, rec: &Recording, api: &TcfgApi) -> Described {
        let mut checks: Vec<Check> = Vec::new();
        let mut tensors = vec![false; self.feeds.len()];
        let mut items: Vec<Place> = Vec::new();
        let mut closed = rec.failed.clone();
        let mut wrote = rec.wrote;
        let nodes = match &closed {
            None => self.nodes(func, rec, api, &mut checks, &mut tensors, &mut items, &mut wrote),
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
        let places: Vec<(Place, Obj, bool)> = feeds.chain(held).collect();
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
                    checks.push(Check { target: place, cond: Cond::Type(PyObj::from_borrowed(ffi::Py_TYPE(tensor) as Obj)) });
                }
                continue;
            };
            if row.forward && taken && closed.is_none() {
                closed = Some("a tensor of the call carries a forward-mode gradient".to_string());
            }
            checks.push(Check { target: place, cond: Cond::Tensor(row) });
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
            let source = self.source.get(place.value as usize).copied().unwrap_or(place.value);
            let label = match func.value_names.get(source as usize).and_then(|n| n.as_ref()) {
                Some(label) => label.clone(),
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
            checks.push(Check { target: Place::of(recv), cond: Cond::Type(PyObj::from_borrowed(ffi::Py_TYPE(value) as Obj)) });
            return Some(Err(format!("an operation of the graph is a method of a {}", pyffi::type_name(value))));
        }
        let function = PyObj::from_owned(function);
        let found = const_ref(function.as_ptr());
        checks.push(Check { target: Place::of(recv), cond: Cond::Has(PyObj::from_borrowed(name), function) });
        Some(Ok(found))
    }

    /// The nodes of the description of a recorded run, as a list. Adds the
    /// checks on the values the graph has as constants, marks the feeds it
    /// takes as tensors, lists the tensors it takes out of lists and tuples
    /// (`items`: the k-th is the value `item_id(k)` of the graph), and finds
    /// whether a normalization updated its statistics. The reason, if the run cannot be exported.
    unsafe fn nodes(
        &self,
        func: &Function,
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
            let value = feed.map_or(null_mut(), |_| rec.value(v));
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
                held.push(Check { target: Place::of(v), cond: Cond::Sequence(ffi::Py_TYPE(value) as usize, count as usize) });
                for i in 0..count {
                    let item = sequence_item(value, i);
                    let place = Place { value: v, item: i as u32 };
                    if (api.tensor_info)(item, &mut info) != 0 {
                        parts.push(tuple_of(&[text("v"), ffi::PyLong_FromLongLong(item_id(items.len()))]));
                        items.push(place);
                    } else if checkable(item) {
                        let kept = frozen(item);
                        parts.push(const_ref(kept.as_ptr()));
                        held.push(Check { target: place, cond: Cond::ValueEq(kept) });
                    } else {
                        parts.into_iter().for_each(|o| ffi::Py_DECREF(o));
                        held.push(Check { target: place, cond: Cond::Type(PyObj::from_borrowed(ffi::Py_TYPE(item) as Obj)) });
                        return Err(format!("an operand of the graph holds a {}", pyffi::type_name(item)));
                    }
                }
                let kind = if ffi::PyList_CheckExact(value) != 0 { "l" } else { "t" };
                return Ok(tuple_of(&[text(kind), list_of(parts)]));
            }
            let v = Place::of(v);
            if !fits {
                checks.borrow_mut().push(Check { target: v, cond: Cond::Type(PyObj::from_borrowed(ffi::Py_TYPE(value) as Obj)) });
                return Err(format!("an operand of the graph is a {}", pyffi::type_name(value)));
            }
            let cond = match how {
                "tensor" => {
                    tensors[feed.unwrap()] = true;
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
                checks.borrow_mut().push(Check { target: v, cond });
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
    pub unsafe fn conditions(&self, func: &Function, path: &Path) -> Vec<String> {
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
                let value = check.target.text(func, path);
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
        Op::Hold(_) | Op::StoreLocal(..) => Some(Kind::Residual),
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

/// What cannot fail and runs no Python code whatever it is given.
pub fn infallible(op: &Op) -> bool {
    matches!(
        op,
        Op::Const(_)
            | Op::Hold(_)
            | Op::BuildTuple(_)
            | Op::BuildList(_)
            | Op::BuildSlice(..)
            | Op::ListAppend(..)
            | Op::MakeCell(_)
            | Op::CellSet(..)
            | Op::TupleGet(..)
            | Op::IsNone(..)
            | Op::Is(..)
            | Op::TensorSize { .. }
            | Op::TensorDim(_)
    )
}

/// Whether a change of something the path made can be left out of a run
/// together with the thing: only one that cannot fail and runs no Python
/// code whatever it is given. An append is such. An entry that is put into
/// a dict is one only under a key that is a constant of the function: any
/// other key may be one that cannot be hashed at another call, or that is
/// hashed in Python.
fn leaves_quietly(op: &Op, constant: &[bool]) -> bool {
    match op {
        Op::MapAdd(_, key, _) => constant[*key as usize],
        _ => true,
    }
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
        defs.insert(inst.dst, Def { node, late, graph, made: false });
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
        stages: stages_of(&body),
        handed: Vec::new(),
        made: Vec::new(),
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
        source: Vec::new(),
        branches: 0,
        first: None,
        provisional: false,
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

/// The instructions of a function from the first operation of a graph on,
/// as they are taken down while they run (`Compiled::record`): renamed to
/// the values of the path, which are numbered from 1 in the order they are
/// made or first read (0 is the undefined value).
pub struct Trace {
    pub body: Vec<PathInst>,
    /// Per value of the path.
    pub defs: Vec<Def>,
    pub source: Vec<V>,
    pub inputs: Vec<(V, PathValue)>,
    /// Per value of the function: the value of the path it is, or NONE; and
    /// what it holds where the trace has got to.
    pub map: Vec<PathValue>,
    pub held: Vec<Held>,
    /// The variables that were bound, in order.
    pub bound: Vec<(u32, PathValue)>,
    pub operators: Vec<String>,
    pub shape: Vec<u32>,
    pub rec: Recording,
    /// The trace was let go of (`abandon`), and why.
    pub abandoned: Option<String>,
    /// See `Path::first`.
    pub first: Option<(usize, u32)>,
    /// The things the path made and what may be them or hold them, in
    /// families (see `Family`).
    pub things: Family,
}

/// The things a path made (a list, a dict, an iterator) and the values of
/// the path that may be one of them or may hold one, at this call or at
/// another, in families: two things are of one family once one may hold
/// the other, or one value may be or hold both. A value can be a thing, or
/// hold one, in four ways only: an instruction of the path made it; it was
/// taken out of a value of a family in which something holds a thing (an
/// item, what an iterator gives); it is an attribute of one (a method
/// bound to it); or an instruction that is no operation of the graph was
/// handed one and may give it back or keep it (a call, a tuple that is
/// built). What comes from in front of the path, and what is read of an
/// object from outside, is none: nothing of the path's is stored there
/// before the path lands. Kept as the trace goes, so a value is what it
/// may be where its instruction stands.
///
/// A family is the unit of two things: which changes run in a run
/// (`Path::of`: those of a family that something that runs takes a value
/// of), and on which side of what stands for the graph its instructions
/// run (`since`).
#[derive(Default)]
pub struct Family {
    /// Per value of the path: a value of the same family that was made
    /// before it, itself for the first one, NONE for a value of no family.
    first: Vec<PathValue>,
    /// Per first value: the values of the family; whether something of it
    /// may hold a thing, so that what is taken out of a value of it may be
    /// one; whether it runs behind the callable from some instruction on.
    values: HashMap<PathValue, Vec<PathValue>>,
    nested: HashMap<PathValue, bool>,
    behind: HashMap<PathValue, bool>,
    /// Per value of the path: the instruction of the path from which it is
    /// late (computed behind the callable, or of a family that runs there
    /// from that instruction on); u32::MAX for one that is not.
    since: Vec<u32>,
}

impl Family {
    /// The first value of the family of a value, or NONE.
    pub fn of(&self, v: PathValue) -> PathValue {
        let mut at = self.first.get(v as usize).copied().unwrap_or(NONE);
        while at != NONE && self.first[at as usize] != at {
            at = self.first[at as usize];
        }
        at
    }

    /// Whether a value may be a thing the path made, or hold one.
    pub fn has(&self, v: PathValue) -> bool {
        v != NONE && v != UNDEF && self.first.get(v as usize).is_some_and(|first| *first != NONE)
    }

    /// Whether a value is late for the instruction `at` of the path.
    pub fn late(&self, v: PathValue, at: usize) -> bool {
        self.since.get(v as usize).is_some_and(|since| (*since as usize) <= at)
    }
}

impl Trace {
    pub fn new(nvalues: usize) -> Trace {
        Trace {
            body: Vec::new(),
            defs: vec![Def::default()],
            source: vec![UNDEF],
            inputs: Vec::new(),
            map: vec![NONE; nvalues],
            held: vec![Held::Untouched; nvalues],
            bound: Vec::new(),
            operators: Vec::new(),
            shape: Vec::new(),
            rec: Recording { values: vec![null_mut()], kept: HashMap::new(), nodes: Vec::new(), failed: None, wrote: false },
            abandoned: None,
            first: None,
            things: Family { first: vec![NONE], since: vec![u32::MAX], ..Family::default() },
        }
    }

    /// The family of a value becomes late from the instruction `from` on:
    /// every value of it that is not late yet is, from there.
    fn runs_behind(&mut self, first: PathValue, from: u32) {
        if self.things.behind.insert(first, true) == Some(true) {
            return;
        }
        for v in self.things.values.get(&first).cloned().unwrap_or_default() {
            if self.things.since[v as usize] == u32::MAX {
                self.things.since[v as usize] = from;
                self.defs[v as usize].late = true;
            }
        }
    }

    /// Two values are of one family from here on (`a` is of one). The
    /// family is kept under its first value. It runs behind the callable
    /// only if both did: the values of the one that does not are not late,
    /// and become so with the first instruction that makes them
    /// (`runs_behind`).
    fn joins(&mut self, a: PathValue, b: PathValue) {
        let (a, b) = (self.things.of(a), self.things.of(b));
        if a == b || a == NONE || b == NONE {
            return;
        }
        let (first, other) = (a.min(b), a.max(b));
        let moved = self.things.values.remove(&other).unwrap_or_default();
        self.things.values.entry(first).or_default().extend(moved);
        self.things.first[other as usize] = first;
        let nested = self.things.nested.remove(&other).unwrap_or(false);
        *self.things.nested.entry(first).or_insert(false) |= nested;
        let behind = self.things.behind.remove(&other).unwrap_or(false);
        *self.things.behind.entry(first).or_insert(false) &= behind;
    }

    /// A value is of the family of `of` from here on.
    fn belongs(&mut self, v: PathValue, of: PathValue) {
        let first = self.things.of(of);
        if first == NONE || self.things.first[v as usize] != NONE {
            return;
        }
        self.things.first[v as usize] = first;
        self.things.values.entry(first).or_default().push(v);
    }

    /// What an instruction that is about to be added to the path means
    /// for the things the path made (`Family`): which family what it gives
    /// is of, which families it makes one, and on which side of the
    /// callable they run from here on.
    ///
    /// The side. An instruction runs behind the callable if it takes a
    /// late value. One that does, and that reads or changes a thing (it
    /// takes a value of the thing's family as more than a reference to
    /// keep), puts the family behind the callable from itself on: every
    /// later instruction that takes a value of the family is late, and so
    /// runs behind it, in the order of the function; every earlier one
    /// was not, and ran in front of it, in that order. So what reads and
    /// what changes one thing run in the order the function has them. A
    /// tuple, a list, a dict or a cell that is built keeps a reference to
    /// what it is built of and reads nothing of it, so it moves no
    /// family. An operation of the graph runs between the two sides, and
    /// puts the families of the lists and other things it is handed
    /// behind it.
    pub fn note(&mut self, op: &Op, dst: PathValue, kind: Kind) {
        let at = self.body.len() as u32;
        self.things.first.resize(self.defs.len(), NONE);
        self.things.since.resize(self.defs.len(), u32::MAX);
        if dst != NONE && self.defs[dst as usize].late && self.things.since[dst as usize] == u32::MAX {
            self.things.since[dst as usize] = at;
        }
        if kind == Kind::Check {
            return;
        }
        let made = dst != NONE && self.defs[dst as usize].made;
        let (mut late, mut any) = (kind == Kind::Node, false);
        reads(op, |v| {
            late |= v != NONE && self.defs[v as usize].late;
            any |= self.things.has(v);
        });
        if !any && !made {
            return;
        }
        let mut taken: Vec<PathValue> = Vec::new();
        reads(op, |v| {
            if self.things.has(v) {
                taken.push(v);
            }
        });
        // What only keeps a reference to what it is given reads nothing
        // of it.
        let keeps = matches!(op, Op::BuildTuple(_) | Op::BuildList(_) | Op::BuildMap(_) | Op::BuildSlice(..) | Op::MakeCell(_) | Op::Is(..) | Op::IsNone(..));
        if kind == Kind::Node {
            // A form is for calls that hand its operations what this one
            // handed them: a number, a text or a tensor is one of the
            // same kind at each (the checks of the form), and nothing
            // of a thing is read through it.
            let holds_nothing = |trace: &Trace, v: PathValue| {
                let held = trace.rec.values[v as usize];
                let mut info = TcfgTensorInfo::empty();
                !held.is_null() && unsafe { is_plain(held) || is_token(held) || capi::api().is_some_and(|api| (api.tensor_info)(held, &mut info) != 0) }
            };
            taken.retain(|v| !holds_nothing(self, *v));
            for v in taken {
                let first = self.things.of(v);
                self.runs_behind(first, at + 1);
            }
            return;
        }
        if late && !keeps {
            for v in &taken {
                let first = self.things.of(*v);
                self.runs_behind(first, at);
            }
        }
        let nested = |things: &Family, v: PathValue| things.nested.get(&things.of(v)).copied().unwrap_or(false);
        let holds = |trace: &mut Trace, v: PathValue| {
            let first = trace.things.of(v);
            trace.things.nested.insert(first, true);
        };
        if made {
            // A thing: of a family of its own, and of that of what it was
            // made of. An iterator, or a copy, of a thing holds what the
            // thing holds; anything else holds what it was made of.
            self.things.first[dst as usize] = dst;
            self.things.values.insert(dst, vec![dst]);
            if late {
                self.things.behind.insert(dst, true);
            }
            let over = matches!(op, Op::GetIter(_) | Op::Call { .. } | Op::CallMethod { .. });
            let inner = taken.iter().any(|v| nested(&self.things, *v));
            for v in &taken {
                self.joins(dst, *v);
            }
            if !taken.is_empty() && (!over || inner) {
                holds(self, dst);
            }
            return;
        }
        match op {
            // What a change puts into a thing.
            Op::ListAppend(into, item) | Op::CellSet(into, item) | Op::StoreAttr { obj: into, val: item, .. } => {
                if self.things.has(*item) {
                    self.joins(*into, *item);
                    holds(self, *into);
                }
            }
            Op::MapAdd(into, key, item) => {
                for v in [*key, *item] {
                    if self.things.has(v) {
                        self.joins(*into, v);
                        holds(self, *into);
                    }
                }
            }
            // The items of another: things only where that one holds some.
            Op::ListExtend(into, more) | Op::DictMerge(into, more, _) => {
                if self.things.has(*more) && nested(&self.things, *more) {
                    self.joins(*into, *more);
                }
            }
            // What is taken out of something is a thing only where the
            // family has one that is held.
            Op::GetItem(of, _) | Op::IterNext(of) | Op::TupleGet(of, _) | Op::CellGet(of, _) | Op::Unpack(of, _) | Op::ListToTuple(of) => {
                if dst != NONE && self.things.has(*of) && nested(&self.things, *of) {
                    self.belongs(dst, *of);
                }
            }
            // An attribute may be a part of the thing (the dict of an
            // object) or hold it (a bound method), and what a method is
            // bound to may be the thing.
            Op::LoadAttr(of, _) => {
                if dst != NONE {
                    self.belongs(dst, *of);
                }
            }
            // What cannot hold anything: a number, a text, a flag.
            Op::Compare(..) | Op::Is(..) | Op::IsNone(..) | Op::ToBool(_) | Op::Contains(..) | Op::GetLen(_) | Op::Unary(..) | Op::Format { .. } | Op::BuildString(_) | Op::CheckBound(..) | Op::IsFunction { .. } => {}
            // A tuple or a slice that is built holds what it is built of.
            Op::BuildTuple(_) | Op::BuildSlice(..) => {
                for v in &taken {
                    self.belongs(dst, *v);
                    self.joins(dst, *v);
                }
                holds(self, dst);
            }
            // Anything else may give back what it was handed (the
            // default of a `get`), go over it (a view of a dict) or be
            // made of what it holds (two lists that are joined): of its
            // family, and nothing is held that was not.
            _ => {
                if dst != NONE {
                    for v in &taken {
                        self.belongs(dst, *v);
                        self.joins(dst, *v);
                    }
                }
            }
        }
    }

    /// A new value of the path, which a value of the function is from here
    /// on; `kept` is what it is in this run (a reference of its own, or
    /// null).
    pub fn define(&mut self, v: V, def: Def, kept: Obj) -> PathValue {
        let made = self.defs.len() as PathValue;
        self.defs.push(def);
        self.things.first.resize(self.defs.len(), NONE);
        self.things.since.resize(self.defs.len(), u32::MAX);
        self.source.push(v);
        self.rec.values.push(kept);
        if !kept.is_null() && !capi::is_unboxed(kept) {
            self.rec.kept.entry(kept as usize).or_insert((0, made)).0 += 1;
        }
        self.map[v as usize] = made;
        self.held[v as usize] = Held::Value(made);
        made
    }

    /// Lets go of every reference the trace holds to the values of the
    /// run. A trace keeps what its instructions gave and took, so that
    /// the run can be described when it has ended; an object that the
    /// function has let go of is alive by that alone. Where the end of
    /// such an object may run Python code, it has to end where the
    /// function let go of it: the trace gives up what it holds, there,
    /// and nothing is made of this run.
    pub unsafe fn abandon(&mut self, why: String) {
        // What the trace reads of its values from here on finds None.
        for kept in self.rec.values.iter_mut().filter(|kept| !kept.is_null()) {
            let held = std::mem::replace(kept, new_ref(ffi::Py_None()));
            if capi::is_unboxed(held) { capi::release(held) } else { xdecref(held) }
        }
        self.rec.kept.clear();
        self.abandoned = Some(why);
    }

    /// A value of the function becomes the one another is: a variable of an
    /// edge.
    pub fn rename(&mut self, v: V, to: PathValue) {
        self.map[v as usize] = to;
        self.held[v as usize] = if to == UNDEF { Held::Empty } else { Held::Value(to) };
    }

    pub fn check(&mut self, assert: Assert, bc: u32) {
        let inst = PathInst {
            op: Op::Hold(Vec::new()),
            dst: NONE,
            kind: Kind::Check,
            bc,
            stage: Stage::Before,
            in_graph: false,
            assert: Some(assert),
        };
        self.body.push(inst);
    }
}

/// A copy of an instruction with its operands renamed, for the instructions
/// a path can hold; None for any other. A call site is renamed without its
/// fallback, which no path runs.
pub fn renamed(op: &Op, f: &mut impl FnMut(V) -> V) -> Option<Op> {
    let mut all = |vs: &[V]| vs.iter().map(|v| f(*v)).collect::<Vec<V>>();
    Some(match op {
        Op::Const(_) | Op::LoadGlobal(_) | Op::LoadDeref(_) => op.clone(),
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
        Op::ListAppend(a, b) => Op::ListAppend(f(*a), f(*b)),
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
            source: Vec::new(),
            branches: 0,
            first: None,
            provisional: false,
            exported: Shared::new(Vec::new()),
            registers: Shared::new(Vec::new()),
            stages: Default::default(),
            handed: Vec::new(),
            made: Vec::new(),
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
        // A value that is late is so from some instruction on: from the
        // first, where nothing says from which.
        trace.things.since.resize(trace.defs.len(), u32::MAX);
        for v in 0..trace.defs.len() {
            if trace.defs[v].late && trace.things.since[v] == u32::MAX {
                trace.things.since[v] = 0;
            }
        }
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
        // The constants of the function among the values of the path.
        let mut constant = vec![false; nregs];
        for inst in trace.body.iter().filter(|inst| inst.dst != NONE && matches!(inst.op, Op::Const(_))) {
            constant[inst.dst as usize] = true;
        }
        let values = &trace.rec.values;
        // A method of the interpreter's, bound to what it was read off: a
        // tensor that an operation of the graph gave, or a list, a dict or
        // a set that the path made, of exactly that class. Nothing in such
        // a read can fail or run code, at any call.
        let bound_method = |op: &Op, dst: PathValue| match op {
            Op::LoadAttr(of, _) if dst != NONE => {
                let (held, of, def) = (values[dst as usize], values[*of as usize], defs[*of as usize]);
                let builtin = |class: *mut ffi::PyTypeObject| unsafe { ffi::Py_TYPE(of) == class };
                let plain = !of.is_null()
                    && (builtin(std::ptr::addr_of_mut!(ffi::PyList_Type))
                        || builtin(std::ptr::addr_of_mut!(ffi::PyDict_Type))
                        || builtin(std::ptr::addr_of_mut!(ffi::PySet_Type)));
                (def.node || (def.made && plain))
                    && !held.is_null()
                    && unsafe { ffi::PyCFunction_Check(held) != 0 && (*(held as *mut ffi::PyCFunctionObject)).m_self == of }
            }
            _ => false,
        };
        // Which thing a value of the path is, or may be, or may hold: its
        // family (`Family`), or itself for a value of none.
        let things = &trace.things;
        let thing = |v: PathValue| match things.of(v) {
            NONE => v,
            first => first,
        };
        // Whether a value is late for the instruction `at`: computed
        // behind the callable, or of a family that runs behind it by then.
        let late = |v: PathValue, at: usize| things.late(v, at);
        // What an instruction changes of what the path made: the list it
        // appends to or extends, the dict it stores into, the iterator it
        // takes the next item of.
        let changes = |inst: &PathInst| {
            let mut target = NONE;
            if inst.kind == Kind::Residual && !matches!(inst.op, Op::Hold(_) | Op::StoreLocal(..) | Op::GetIter(_)) {
                inst.op.for_each_use(|v| {
                    if target == NONE {
                        target = v;
                    }
                });
            }
            target
        };
        // The things that something that runs takes, wherever on the path
        // it stands: a change to one of them runs too. An object is held
        // by reference, so what took it in front of the change (another list
        // of the path) has the change,
        // and what takes the next item of an iterator needs every item in
        // front of it taken. The instructions are gone over from the last
        // to the first, which finds a use behind a change and not one in
        // front of it: they are gone over again with what the last time
        // found, until that finds no more.
        let mut alive = vec![false; nregs];
        let first_need = need;
        let mut need = first_need.clone();
        // The values that are no tensors and that the graph computes.
        let mut wanted = vec![false; nregs];
        let (mut outputs, mut exits) = (Vec::new(), Vec::new());
        let mut branches = 0;
        loop {
            for (at, inst) in trace.body.iter_mut().enumerate().rev() {
                inst.in_graph = false;
                let dst = inst.dst;
                let needed = dst != NONE && need[dst as usize];
                let def = if dst == NONE { Def::default() } else { Def { late: late(dst, at), ..defs[dst as usize] } };
                // What an operation of the graph, or a value the graph computes,
                // takes: of what depends on a node, the graph computes it too,
                // and it has the lists the path builds as what they hold; the
                // rest is there in front of the callable.
                let takes = |op: &Op, wanted: &mut [bool], need: &mut [bool]| {
                    reads(op, |v| match (Def { late: late(v, at), ..defs[v as usize] }) {
                        taken if (taken.late && !taken.node) || built[v as usize] => wanted[v as usize] = true,
                        taken if !taken.late => need[v as usize] = true,
                        _ => {}
                    })
                };
                let uses = |op: &Op, need: &mut [bool]| {
                    let mut behind = false;
                    op.for_each_use(|v| {
                        need[v as usize] = true;
                        behind |= late(v, at);
                    });
                    if behind { Stage::After } else { Stage::Before }
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
                            // A guard on what a node returns looks at what the
                            // graph says of it.
                            Assert::Guard(cond) if defs[v].node => {
                                exits.push(cond.clone());
                                Stage::Inside
                            }
                            _ => {
                                branches += matches!(assert, Assert::Truth(..) | Assert::Null(..)) as usize;
                                need[v] = true;
                                if late(v as PathValue, at) { Stage::After } else { Stage::Before }
                            }
                        }
                    }
                    Kind::Residual => {
                        // What changes something the path made goes with it,
                        // unless it can fail: then it runs, and the thing
                        // with it, so that a call in which it fails is not
                        // the path's.
                        let target = changes(inst);
                        let taken = target != NONE && (need[target as usize] || alive[thing(target) as usize]);
                        let fails = !leaves_quietly(&inst.op, &constant);
                        if (needed || taken || fails) && !matches!(inst.op, Op::Hold(_) | Op::StoreLocal(..)) {
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
                    // What nothing uses is left out of a run only where it
                    // cannot fail: anything else is what shows that the call
                    // gets this far (an attribute that is read for nothing
                    // and may be gone, a call whose answer is dropped), and
                    // a call in which it raises is not the path's.
                    // Nor is a method that was read off a tensor the graph
                    // gives, which the operation that calls it stands for,
                    // or off a list, a dict or a set the path made (the
                    // `append` of a list that a loop fills).
                    _ if !infallible(&inst.op) && !bound_method(&inst.op, dst) => uses(&inst.op, &mut need),
                    _ => Stage::Skipped,
                };
            }
            let mut found = alive.clone();
            (0..nregs).filter(|v| need[*v]).for_each(|v| found[thing(v as PathValue) as usize] = true);
            let left_out = |inst: &PathInst| inst.stage == Stage::Skipped && changes(inst) != NONE && found[thing(changes(inst)) as usize];
            if !trace.body.iter().any(left_out) {
                break;
            }
            alive = found;
            need = first_need.clone();
            wanted.iter_mut().for_each(|wanted| *wanted = false);
            (outputs.clear(), exits.clear());
            branches = 0;
        }
        outputs.reverse();
        exits.reverse();
        let mut feeds: Vec<PathValue> = Vec::new();
        for (at, inst) in trace.body.iter().enumerate().filter(|(_, inst)| inst.kind == Kind::Node || inst.in_graph) {
            reads(&inst.op, |v| {
                if !late(v, at) && !wanted[v as usize] && !feeds.contains(&v) {
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
            stages: stages_of(&trace.body),
            handed: handed_of(&trace.inputs, &trace.body, &writes, &bound),
            made: trace.defs.iter().map(|def| def.made).collect(),
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
            source: trace.source.clone(),
            branches,
            first: trace.first,
            provisional,
            exported: Shared::new(Vec::new()),
            registers: Shared::new(Vec::new()),
        }
    }
}
