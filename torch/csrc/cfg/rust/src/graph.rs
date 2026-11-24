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
    /// called, and has this many defaults for its last parameters: what a
    /// call that the path goes into enters, and how it binds what it is
    /// not handed.
    Enters(PathValue, *const Compiled, usize),
    /// What an `__iter__` returned is an iterator, what a `__len__`
    /// returned is a count: the interpreter raises for anything else.
    Iterator(PathValue),
    Count(PathValue),
    /// The object has no attribute of the name (`absent`): reading it
    /// comes down to the `__getattr__` of its class.
    Absent(PathValue, PyObj),
    /// The class of the object has nothing under the name: no
    /// `__getattr__` behind a `__getattribute__` that the path went into.
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
    /// An object of the class keeps what is stored under the name in its
    /// own dict, as `object` has it (`stores_plainly`): a store into an
    /// object the path made cannot fail and runs no Python code.
    Plain(PathValue, PyObj),
    /// What `object` reads of the object this is bound to finds nothing
    /// under the name (`reads_nothing`): the read raises AttributeError
    /// and does nothing else.
    Missing(PathValue, PyObj),
}

impl Assert {
    pub fn value(&self) -> PathValue {
        match self {
            Assert::Truth(v, _)
            | Assert::Null(v, _)
            | Assert::Enters(v, ..)
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
            | Assert::Apart(v, ..)
            | Assert::Plain(v, _)
            | Assert::Missing(v, _) => *v,
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
    /// For an operation that was handed a dict with `**`: the names, a
    /// tuple, of its last operands, which no constant of the function is.
    pub named: Option<PyObj>,
    /// For a read in front of the callable (a global, an attribute of a
    /// module) that an earlier instruction in front of it makes as well:
    /// the value that one gives, or NONE. Nothing changes between the
    /// instructions in front of the callable, so the second read finds
    /// the object the first found (`Path::of`, `run_stage`).
    pub again: PathValue,
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
    /// It ends in front of an instruction that raised when it was taken
    /// down, or of a call it went into that did. Its checks say nothing
    /// of the raise: it lands in front of the instruction, which the
    /// function then runs. It is tried behind the paths that are not so
    /// (`Compiled::enter_graph`), and stands back for a call when the
    /// raise did not come (`GraphSite::passes`).
    pub raised: bool,
    /// It leaves a variable of the frame unbound that the instructions
    /// bind: one that is bound to what the graph computes and does not
    /// give back, where the path lands at the return of the function and
    /// the frame is gone. Not run for a call whose frame has a frame
    /// object, which may outlive the call (`Compiled::run_path`).
    pub unbinds: bool,
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
    /// Per output: whether something can still hold it when the path has
    /// landed (a variable or a value of the function, an object it was
    /// stored into, a list or a tuple that is held). One that is only
    /// looked at behind the callable, for its sizes or its kind, is gone
    /// by then.
    pub held: Vec<bool>,
    pub exported: Shared<Vec<Arc<Exported>>>,
    /// Register files for the calls that run it, kept from one to the next.
    pub registers: Shared<Vec<Vec<Obj>>>,
    /// Which instructions of `body` run in front of the callable, behind
    /// it, and where the path lands, in order (`Path::staged`).
    pub stages: [Vec<u32>; 3],
    /// The values of the function that the path takes and hands on: it
    /// writes them to a value of the function or binds a variable to
    /// them, as they are or as an item of a tuple it builds. What such a
    /// value holds is held by the function when the path has landed.
    pub handed: Vec<V>,
    /// Per register: an instruction of the path made what it holds
    /// (`Def::made`), which is the path's own to change and to use up.
    pub made: Vec<bool>,
    /// What the checks of the path pin: a value of the path with the
    /// value it has to have (`Assert::Equals`, and the name of
    /// `Assert::Missing`) or, with the flag unset, the object it has to
    /// be (`Assert::Is`). The path was taken down with these, so they are
    /// part of which path it is (`pinned_as`), as its instructions are.
    pub pins: Vec<(PathValue, PyObj, bool)>,
}

// The registers are empty between calls, and only touched with the GIL held.
unsafe impl Send for Path {}
unsafe impl Sync for Path {}

/// The instructions of a path by the stage they run in (`Path::stages`).
pub fn stages_of(body: &[PathInst]) -> [Vec<u32>; 3] {
    let mut stages = [Vec::new(), Vec::new(), Vec::new()];
    for (at, inst) in body.iter().enumerate() {
        match inst.stage {
            Stage::Before => stages[0].push(at as u32),
            Stage::After => stages[1].push(at as u32),
            Stage::Landing => stages[2].push(at as u32),
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
    /// The instructions that run in one stage: in front of the callable,
    /// behind it, or where the path lands.
    pub fn staged(&self, stage: Stage) -> impl Iterator<Item = &PathInst> + '_ {
        let at = match stage {
            Stage::Before => 0,
            Stage::After => 1,
            Stage::Landing => 2,
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

    /// Whether two paths of the same instructions were taken down with
    /// the same pinned values: only then are they one path, and a form
    /// of the one may run under the checks of the other.
    pub fn pinned_as(&self, other: &Path) -> bool {
        self.pins.len() == other.pins.len()
            && self.pins.iter().zip(&other.pins).all(|((v, a, value), (w, b, _))| {
                v == w && if *value { unsafe { same_value(a.as_ptr(), b.as_ptr()) } } else { a.as_ptr() == b.as_ptr() }
            })
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
    /// than a graph has; or the graph starts behind a call and its path
    /// is that of a graph that starts further on (`GraphSite::twin`).
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
    static ASKED: pyffi::Once<bool> = pyffi::Once::new();
    *ASKED.get_or_init(|| std::env::var_os("TORCH_VM_GRAPH_DEBUG").is_some_and(|v| v != "0"))
}

/// Whether `TORCH_VM_GRAPH_TIMES` asks for the time of the calls that run
/// an exported form to be taken: what runs in front of the callable, the
/// callable, and what runs behind it (`GraphSite::times`).
pub fn timed() -> bool {
    static ASKED: pyffi::Once<bool> = pyffi::Once::new();
    *ASKED.get_or_init(|| std::env::var_os("TORCH_VM_GRAPH_TIMES").is_some_and(|v| v != "0"))
}

/// How many times a graph is taken down again because its path ended at a
/// block that had not been profiled.
const MAX_PROVISIONAL: u32 = 3;
/// How often a path of a graph is taken down again to land in a new
/// version of a block.
const MAX_MOVES: u32 = 8;

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
    /// The graph is exported for nothing but its backward
    /// (`Graphs::derives`), and whether a call whose tensors need a
    /// gradient has come to it. Until one has, the graph is one the VM
    /// runs itself.
    pub derives: bool,
    pub grads: AtomicBool,
    pub min_ops: usize,
    pub max_variants: usize,
    unsure: AtomicU32,
    /// How many calls in a row a path of the graph was turned down for
    /// what it would let go of where it lands; from `MAX_TURNED` on the
    /// graph has given way (`gave_way`).
    turned: AtomicU32,
    provisional: AtomicU32,
    /// How often a path was taken down again to land in a new version of
    /// a block, or with the paths that end at a raise passed over
    /// (`retakes`).
    moves: AtomicU32,
    /// The call of the function in which a path that ends in front of an
    /// instruction that raised (`Path::raised`) was last the path of the
    /// call; u64::MAX for none. If nothing raised in that call, such
    /// paths are passed over (`passes`) until a call that is taken down
    /// gets a path of its own.
    awaits: AtomicU64,
    passing: AtomicBool,
    /// How many calls were taken down and ended at what raised, or left
    /// such a path the one that is reported: a graph that is not exported
    /// is taken down `MAX_MOVES` times more for it.
    raises: AtomicU32,
    /// How many calls took a path of the graph down.
    pub taken: AtomicU64,
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
            state: Shared::new(if exported && !graphs.derives { GraphState::Waiting } else { GraphState::Built }),
            exporter: graphs.export.clone(),
            derives: graphs.derives,
            grads: AtomicBool::new(false),
            min_ops: graphs.min_ops,
            max_variants: graphs.max_variants,
            unsure: AtomicU32::new(0),
            turned: AtomicU32::new(0),
            provisional: AtomicU32::new(0),
            moves: AtomicU32::new(0),
            awaits: AtomicU64::new(u64::MAX),
            passing: AtomicBool::new(false),
            raises: AtomicU32::new(0),
            taken: AtomicU64::new(0),
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
                // A path that ends in front of what raised is the report
                // until a call gets further, for a few calls more.
                let open = |path: &Arc<Path>| path.provisional || (path.raised && self.raises.load(Ordering::Relaxed) < MAX_MOVES);
                self.again.load(Ordering::Relaxed) || self.variants.borrow().first().is_none_or(open)
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

    /// For a graph that starts behind a call (`Graphs::build`): the graph
    /// among `over`, those its path went through, that has a path with
    /// the same operations up to the same place. That one was there
    /// first and has what was made of the operations: this graph stands
    /// back for it (`stand_back`).
    pub fn twin(&self, path: &Path, over: &[Arc<GraphSite>]) -> bool {
        let behind_a_call = self.entry && self.root != (0, 0);
        behind_a_call
            && !path.provisional
            && self.variants.borrow().is_empty()
            && over.iter().any(|other| {
                let theirs = other.variants.borrow();
                let known = matches!(*other.state.borrow(), GraphState::Built | GraphState::Waiting);
                known && theirs.iter().any(|had| !had.provisional && had.land == path.land && had.operators == path.operators)
            })
    }

    /// The graph is none: the graphs that start further on are their own.
    pub fn stand_back(&self, count: usize) {
        *self.state.borrow_mut() = GraphState::Small(count);
    }

    /// Whether the graph has given way (`turned_down`).
    pub fn gave_way(&self) -> bool {
        self.turned.load(Ordering::Relaxed) >= MAX_TURNED
    }

    pub fn give_up(&self, reason: String) {
        *self.state.borrow_mut() = GraphState::GivenUp(reason);
    }

    /// Whether the paths of the graph are taken down as what a callable can
    /// stand for: there is an exporter, or the engine derives backwards and
    /// a call whose tensors need a gradient has come.
    pub fn exports(&self) -> bool {
        if self.derives {
            self.grads.load(Ordering::Relaxed)
        } else {
            self.exporter.is_some()
        }
    }

    /// A call whose tensors need a gradient has come to a graph that is
    /// exported for nothing but its backward: from here on its paths are
    /// taken down for that.
    pub fn needs_gradients(&self) {
        self.grads.store(true, Ordering::Relaxed);
        let mut state = self.state.borrow_mut();
        if matches!(*state, GraphState::Built) {
            *state = GraphState::Waiting;
        }
    }

    /// Whether the graph is exported for nothing but its backward and has
    /// its path from calls none of whose tensors needed a gradient: it is
    /// exported once a call that needs one is taken down.
    pub fn dormant(&self) -> bool {
        self.derives
            && !self.grads.load(Ordering::Relaxed)
            && matches!(*self.state.borrow(), GraphState::Built)
            && self.variants.borrow().first().is_some_and(|path| !path.provisional)
    }

    /// Has such a graph take the next call that comes to it down again.
    pub fn wake(&self) -> bool {
        let dormant = self.dormant();
        if dormant {
            self.again.store(true, Ordering::Relaxed);
        }
        dormant
    }

    /// Takes a path that was taken down, with the form that was made of the
    /// run if one was: a path it has already gets the form. A path that is
    /// not final (see `Path::provisional`) is taken a few times at most.
    pub fn install(&self, path: Path, form: Option<Arc<Exported>>) {
        self.again.store(false, Ordering::Relaxed);
        if path.provisional && self.provisional.fetch_add(1, Ordering::Relaxed) + 1 < MAX_PROVISIONAL {
            return;
        }
        if path.operators.len() < self.min_ops && form.is_none() {
            // Too few operations for a graph, and no other path: the graph
            // is none. Not yet where the path ended at what raised, which
            // the next call may get past: only when that has been so a
            // few times.
            let small = matches!(*self.state.borrow(), GraphState::Built | GraphState::Waiting)
                && self.variants.borrow().is_empty()
                && !(path.raised && self.raises.load(Ordering::Relaxed) < MAX_MOVES);
            if small {
                *self.state.borrow_mut() = GraphState::Small(path.operators.len());
            }
            return;
        }
        let known = self.known(&path);
        // What makes way is let go of behind the borrow: the end of what a
        // path holds can run code, which may come to this graph.
        let (mut gone, mut unused) = (Vec::new(), None);
        {
            let mut paths = self.variants.borrow_mut();
            let at = match known {
                Some(at) => {
                    unused = Some(path);
                    at
                }
                None => {
                    // A path that was not final makes way for the ones that are.
                    let (kept, provisional) = std::mem::take(&mut *paths).into_iter().partition(|other| !other.provisional);
                    (*paths, gone) = (kept, provisional);
                    paths.insert(0, Arc::new(path));
                    self.passing.store(false, Ordering::Relaxed);
                    0
                }
            };
            if let Some(form) = form {
                paths[at].exported.borrow_mut().insert(0, form);
            }
            let path = paths.remove(at);
            paths.insert(0, path);
        }
        drop((gone, unused));
    }

    /// Where the graph has a path already: the same instructions, taken
    /// down with the same pinned values, and writing to a tensor from
    /// outside or not as this one does. A form runs under the checks of
    /// the path it is kept with, and as what that path is (`Path::mutates`:
    /// what wrote is not run a second time), so the form of a trace with
    /// other pinned values, or of one that writes where the other did not
    /// (a batch norm that trains at one call and not at another), is kept
    /// with a path of its own.
    pub fn known(&self, path: &Path) -> Option<usize> {
        let same = |other: &Arc<Path>| {
            other.shape == path.shape && other.provisional == path.provisional && other.mutates == path.mutates && other.pinned_as(path)
        };
        self.variants.borrow().iter().position(same)
    }

    /// Whether a path of the graph may be dropped and taken down again at
    /// the next call, so that it lands in a new version of a block, or a
    /// call be taken down although a path that ends in front of what
    /// raised is for it (`passes`): `MAX_MOVES` times per graph. Counts
    /// the time it says yes.
    pub fn retakes(&self) -> bool {
        self.moves.fetch_add(1, Ordering::Relaxed) < MAX_MOVES
    }

    /// A path that ends in front of an instruction that raised when it was
    /// taken down is the path of this call of the function.
    pub fn awaits(&self, call: u64) {
        self.awaits.store(call, Ordering::Relaxed);
    }

    /// Whether the paths that end in front of an instruction that raised
    /// are passed over for this call. One of them was the path of an
    /// earlier call, and nothing raised in that call (`raised` is the
    /// call of the function in which something last did): a call that
    /// gets further finds the checks of such a path holding too, and
    /// would never be taken down. From there on they are passed over,
    /// so that a call that no other path is for is taken down, until one
    /// gets a path of its own (`install`) or the graph `retakes` no more
    /// (`passed`).
    pub fn passes(&self, raised: u64) -> bool {
        let awaited = self.awaits.swap(u64::MAX, Ordering::Relaxed);
        if awaited != u64::MAX && awaited != raised && self.moves.load(Ordering::Relaxed) < MAX_MOVES {
            self.passing.store(true, Ordering::Relaxed);
        }
        self.passing.load(Ordering::Relaxed)
    }

    /// Whether such paths may still be passed over for a call (`passed`).
    pub fn skips(&self) -> bool {
        self.moves.load(Ordering::Relaxed) < MAX_MOVES
    }

    /// A call was taken down with such paths passed over, and for
    /// nothing: it ended where one of them does. One of the graph's
    /// retakes; with those used up they are passed over no more.
    pub fn passed(&self) {
        if !self.retakes() || !self.skips() {
            self.passing.store(false, Ordering::Relaxed);
        }
    }

    /// A call was taken down and ended at what raised, or left such a
    /// path the one that is reported.
    pub fn raise(&self) {
        self.raises.fetch_add(1, Ordering::Relaxed);
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
            // The forms end with nothing borrowed: their end can run code.
            let paths: Vec<Arc<Path>> = self.variants.borrow().iter().cloned().collect();
            for path in paths {
                let forms = std::mem::take(&mut *path.exported.borrow_mut());
                drop(forms);
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
        let (gone, kept): (Vec<_>, Vec<_>) = std::mem::take(&mut *forms).into_iter().partition(|_| {
            at += 1;
            alike.iter().any(|(k, _)| *k == at - 1)
        });
        *forms = kept;
        // The forms end with nothing borrowed: their end can run code.
        drop(forms);
        drop(paths);
        drop(gone);
        Some(varying)
    }

    /// Hands a description to the exporter. It returns the callable and the
    /// guards of the passes, or the reason there is none as a str. An error
    /// that the exporter did not raise itself (`interrupted`) is left set.
    pub unsafe fn export(&self, description: Obj) -> Result<(PyObj, Vec<(usize, PyObj)>), String> {
        if self.derives {
            // Nothing of Python is asked: the engine runs the graph itself.
            return crate::backward::staged(description).map(|callable| (callable, Vec::new()));
        }
        let Some(exporter) = &self.exporter else {
            return Err("nothing exports graphs".to_string());
        };
        let made = ffi::PyObject_CallOneArg(exporter.as_ptr(), description);
        if made.is_null() {
            // What came between the instructions of the exporter and was
            // raised by none of them (an interrupt, what the handler of
            // a signal raised) stays raised.
            if interrupted() {
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

/// The error that is set, as one line of text (`said`); clears it.
pub unsafe fn raised_text() -> String {
    let raised = ffi::PyErr_GetRaisedException();
    if raised.is_null() {
        return "unknown error".to_string();
    }
    let text = said(raised);
    ffi::Py_DECREF(raised);
    text
}

/// What an exception says, where C code alone says it: its class has the
/// base exception's own `__str__`, `KeyError`'s or `OSError`'s, which read
/// the arguments (and of an `OSError` its number, its text and its file
/// names), and those are numbers, texts or None. A `__str__` or a
/// `__repr__` of anybody's never runs. None otherwise.
unsafe fn spoken(raised: Obj) -> Option<String> {
    if raised.is_null() || ffi::PyExceptionInstance_Check(raised) == 0 {
        return None;
    }
    let slot = |class: Obj| (*(class as *mut ffi::PyTypeObject)).tp_str.map(|says| says as usize);
    let own = slot(ffi::Py_TYPE(raised) as Obj);
    let (keyed, filed) = (own == slot(ffi::PyExc_KeyError), own == slot(ffi::PyExc_OSError));
    if own != slot(ffi::PyExc_BaseException) && !keyed && !filed {
        return None;
    }
    let args = (*(raised as *mut ffi::PyBaseExceptionObject)).args;
    if args.is_null() || ffi::PyTuple_CheckExact(args) == 0 {
        return None;
    }
    let count = ffi::PyTuple_GET_SIZE(args);
    if !(0..count).all(|i| is_plain(ffi::PyTuple_GET_ITEM(args, i))) {
        return None;
    }
    if filed {
        let error = raised as *mut ffi::PyOSErrorObject;
        let read = [(*error).myerrno, (*error).strerror, (*error).filename, (*error).filename2];
        if !read.iter().all(|part| part.is_null() || is_plain(*part)) {
            return None;
        }
        // The class's own code, over what was just looked at.
        let text = ffi::PyObject_Str(raised);
        if text.is_null() {
            ffi::PyErr_Clear();
            return None;
        }
        let said = pyffi::str_to_string(text);
        ffi::Py_DECREF(text);
        return Some(said);
    }
    let only = if count == 1 { ffi::PyTuple_GET_ITEM(args, 0) } else { null_mut() };
    Some(match count {
        0 => String::new(),
        1 if !keyed && ffi::PyUnicode_CheckExact(only) != 0 => pyffi::str_to_string(only),
        1 => pyffi::repr(only),
        _ => pyffi::repr(args),
    })
}

/// An exception in a reason: its class, and behind it the first line of
/// what it says where that can be asked without running code (`spoken`).
pub unsafe fn said(raised: Obj) -> String {
    let kind = pyffi::type_name(raised);
    match spoken(raised) {
        Some(text) => format!("{}: {}", kind, text.lines().next().unwrap_or("").chars().take(200).collect::<String>()),
        None => kind,
    }
}

/// What an exception says (`spoken`), or its class where that cannot be
/// asked.
pub unsafe fn message(raised: Obj) -> String {
    spoken(raised).unwrap_or_else(|| pyffi::type_name(raised))
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
    if a == b && !is_row(a) && ffi::PySlice_Check(a) == 0 {
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
    if !is_row(a) || ffi::PyObject_Size(a) != ffi::PyObject_Size(b) {
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
    if !is_row(o) {
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

/// A list or a tuple, or a tuple of a class that is defined in C and
/// compares as a tuple does (a `torch.Size`): what holds its items as a
/// tuple or a list does, and runs no Python code to give them out.
pub unsafe fn is_row(o: Obj) -> bool {
    if capi::is_unboxed(o) {
        return false;
    }
    if ffi::PyTuple_CheckExact(o) != 0 || ffi::PyList_CheckExact(o) != 0 {
        return true;
    }
    let (ty, tuple) = (ffi::Py_TYPE(o), std::ptr::addr_of!(ffi::PyTuple_Type));
    ffi::PyTuple_Check(o) != 0
        && is_static(ty)
        && (*ty).tp_richcompare.map(|f| f as usize) == (*tuple).tp_richcompare.map(|f| f as usize)
}

/// Whether a value is turned into text by C code alone (`str`, `repr`,
/// `format`): a number, a text, None, a dtype or a device, and a list or
/// tuple of such.
pub unsafe fn quiet_text(o: Obj) -> bool {
    // Not what holds itself: the walk into it would not end.
    !o.is_null() && !holds_itself(o) && plain_text(o)
}

unsafe fn plain_text(o: Obj) -> bool {
    if o.is_null() || capi::is_unboxed(o) {
        return false;
    }
    is_plain(o) || is_token(o) || (is_row(o) && (0..ffi::PyObject_Size(o)).all(|i| plain_text(sequence_item(o, i))))
}

/// Whether `a op b` runs no Python code: on numbers; on two texts; on two
/// lists or two tuples (one after the other); on a list, a tuple or a text
/// and a whole number (so many times); on a text and what it formats
/// (`"%d" % n`). Not `a op= b` on a list, which changes the list.
pub unsafe fn quiet_binary(kind: u8, a: Obj, b: Obj) -> bool {
    // The kinds of `op=`, as the interpreter numbers them.
    const IN_PLACE: u8 = 13;
    if a.is_null() || b.is_null() || capi::is_unboxed(a) || capi::is_unboxed(b) {
        return false;
    }
    if kind >= IN_PLACE && ffi::PyList_Check(a) != 0 {
        return false;
    }
    let text = |o: Obj| ffi::PyUnicode_CheckExact(o) != 0;
    let whole = |o: Obj| ffi::PyLong_CheckExact(o) != 0;
    (is_number(a) && is_number(b))
        || (text(a) && (text(b) || quiet_text(b)))
        || (is_row(a) && is_row(b) && (ffi::PyList_Check(a) != 0) == (ffi::PyList_Check(b) != 0))
        || ((is_row(a) || text(a)) && whole(b))
        || (whole(a) && (is_row(b) || text(b)))
}

pub unsafe fn sequence_item(s: Obj, i: isize) -> Obj {
    if ffi::PyTuple_Check(s) != 0 {
        ffi::PyTuple_GET_ITEM(s, i)
    } else {
        ffi::PyList_GET_ITEM(s, i)
    }
}

/// A dtype, a device, a layout or a memory format: objects of types that
/// torch defines in C, which compare by what they stand for. The class
/// itself is asked for, not its name: a class of anybody's can be named
/// `torch.dtype`, and compares and shows itself with code of its own.
unsafe fn is_token(o: Obj) -> bool {
    let class = ffi::Py_TYPE(o);
    is_static(class) && torch_classes()[3..7].contains(&(class as usize))
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
    is_row(o) && (0..ffi::PyObject_Size(o)).all(|i| plainly_checkable(sequence_item(o, i)))
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
    static MISSING: pyffi::Once<PyObj> = pyffi::Once::new();
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

/// What a dict has under a name (borrowed; null for nothing), wherever
/// the engine itself looks into a dict that it did not make: in front of
/// an instruction, to show what the instruction will do, or in its
/// place. `Err` where the name is not plain or a key of the dict is not
/// (`quiet_keys`): comparing the two could run Python code, once more
/// than the instruction runs it. Nothing of the engine looks into such
/// a dict: what asked is not shown, and the instruction runs as it is.
pub unsafe fn entry(dict: Obj, name: Obj) -> Result<Obj, ()> {
    if !is_plain_key(name) || !quiet_keys(dict) {
        return Err(());
    }
    let held = ffi::PyDict_GetItemWithError(dict, name);
    if held.is_null() {
        ffi::PyErr_Clear();
    }
    Ok(held)
}

/// `entry` under a name of the engine's own (`_buffers`).
unsafe fn entry_of(dict: Obj, name: &std::ffi::CStr) -> Result<Obj, ()> {
    if !quiet_keys(dict) {
        return Err(());
    }
    Ok(ffi::PyDict_GetItemString(dict, name.as_ptr()))
}

/// The class of what `functools.lru_cache` makes of a function, or null
/// where `functools` was not imported.
unsafe fn cache_wrapper() -> *mut ffi::PyTypeObject {
    static WRAPPER: pyffi::Once<usize> = pyffi::Once::new();
    let found = *WRAPPER.get_or_init(|| {
        let class = pyffi::in_module(pyffi::module(c"functools"), c"_lru_cache_wrapper");
        if class.is_null() || ffi::PyType_Check(class) == 0 { 0 } else { class as usize }
    });
    found as *mut ffi::PyTypeObject
}

/// Whether an object is what `functools.lru_cache` made of a function.
/// Read off an object as an attribute its class has, it is bound to the
/// object by C code, as a function is.
pub unsafe fn is_cache_wrapper(o: Obj) -> bool {
    !o.is_null() && !capi::is_unboxed(o) && !cache_wrapper().is_null() && ffi::Py_TYPE(o) == cache_wrapper()
}

/// A value that is hashed and compared by C code alone, as a key or as an
/// item of a tuple that is one: a number, a text, None, an object whose
/// class hashes and compares as `object` does (by which object it is), or
/// a tuple of such.
unsafe fn plain_member(o: Obj, depth: usize) -> bool {
    if o.is_null() || capi::is_unboxed(o) {
        return false;
    }
    if is_plain(o) {
        return true;
    }
    if ffi::PyTuple_CheckExact(o) != 0 {
        return depth < 4 && (0..ffi::PyTuple_GET_SIZE(o)).all(|i| plain_member(ffi::PyTuple_GET_ITEM(o, i), depth + 1));
    }
    let (ty, object) = (ffi::Py_TYPE(o), std::ptr::addr_of!(ffi::PyBaseObject_Type));
    let same = |own: usize, of_object: usize| own != 0 && own == of_object;
    same((*ty).tp_hash.map_or(0, |f| f as usize), (*object).tp_hash.map_or(0, |f| f as usize))
        && same((*ty).tp_richcompare.map_or(0, |f| f as usize), (*object).tp_richcompare.map_or(0, |f| f as usize))
}

/// What the garbage collector is shown of an object: the exact dicts and
/// the classes among what it refers to.
struct Referents {
    dicts: Vec<Obj>,
    classes: Vec<Obj>,
}

unsafe extern "C" fn referent(o: Obj, arg: *mut std::ffi::c_void) -> std::ffi::c_int {
    let seen = &mut *(arg as *mut Referents);
    if !o.is_null() && ffi::PyDict_CheckExact(o) != 0 && !seen.dicts.contains(&o) {
        seen.dicts.push(o);
    }
    if !o.is_null() && ffi::PyType_Check(o) != 0 {
        seen.classes.push(o);
    }
    0
}

/// What a call of a `functools.lru_cache` wrapper (or of one that is bound
/// to an object as a method) with these positional arguments returns
/// without calling the function it wraps: the entry its cache has for
/// them. Borrowed; None where that cannot be shown by looking, in which
/// case the call may run the function. Nothing here runs Python code.
///
/// The cache is the one exact dict the wrapper refers to besides the dict
/// of its own attributes (what it shows the garbage collector). Its keys
/// are made of the arguments in one of two ways, by whether the wrapper
/// tells arguments of different types apart, which cannot be asked: the
/// arguments as a tuple (a lone text or whole number as itself), or that
/// tuple with the types of the arguments behind them. A key of the first
/// kind that is found in a cache of the second kind would end in a type,
/// so one whose last argument is no type and that is found is the key of
/// this call. Every key of the cache and every argument is hashed and
/// compared by C code (`plain_member`; a cache of more than 256 entries
/// is not gone through). A wrapper with a bound on its size keeps its
/// results inside objects of a class of its own, which it refers to: what
/// is found is the result only where it is of none of the classes the
/// wrapper refers to.
pub unsafe fn cached(callee: Obj, args: &[Obj]) -> Option<Obj> {
    const SCANNED: isize = 256;
    if callee.is_null() || capi::is_unboxed(callee) || cache_wrapper().is_null() {
        return None;
    }
    let mut given: Vec<Obj> = Vec::with_capacity(args.len() + 1);
    let wrapper = if ffi::Py_TYPE(callee) == std::ptr::addr_of_mut!(pyffi::PyMethod_Type) {
        given.push(pyffi::PyMethod_Self(callee));
        pyffi::PyMethod_Function(callee)
    } else {
        callee
    };
    if !is_cache_wrapper(wrapper) {
        return None;
    }
    given.extend_from_slice(args);
    if !given.iter().all(|o| plain_member(*o, 0)) {
        return None;
    }
    let traverse = (*ffi::Py_TYPE(wrapper)).tp_traverse?;
    let mut seen = Referents { dicts: Vec::new(), classes: Vec::new() };
    traverse(wrapper, referent, &mut seen as *mut Referents as *mut std::ffi::c_void);
    let slot = pyffi::_PyObject_GetDictPtr(wrapper);
    let own = if slot.is_null() { null_mut() } else { *slot };
    seen.dicts.retain(|dict| *dict != own);
    let [cache] = seen.dicts[..] else {
        return None;
    };
    if ffi::PyDict_Size(cache) > SCANNED {
        return None;
    }
    let (mut at, mut key, mut value) = (0, null_mut(), null_mut());
    while ffi::PyDict_Next(cache, &mut at, &mut key, &mut value) != 0 {
        if !plain_member(key, 0) {
            return None;
        }
    }
    let lone = given.len() == 1 && (ffi::PyUnicode_CheckExact(given[0]) != 0 || ffi::PyLong_CheckExact(given[0]) != 0);
    let found = if lone {
        ffi::PyDict_GetItemWithError(cache, given[0])
    } else {
        if given.last().is_some_and(|last| ffi::PyType_Check(*last) != 0) {
            return None;
        }
        let key = ffi::PyTuple_New(given.len() as isize);
        if key.is_null() {
            ffi::PyErr_Clear();
            return None;
        }
        for (i, o) in given.iter().enumerate() {
            ffi::PyTuple_SET_ITEM(key, i as isize, new_ref(*o));
        }
        let found = ffi::PyDict_GetItemWithError(cache, key);
        ffi::Py_DECREF(key);
        found
    };
    if found.is_null() {
        ffi::PyErr_Clear();
        return None;
    }
    let class = ffi::Py_TYPE(found) as Obj;
    (!seen.classes.contains(&class)).then_some(found)
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
    static FOUND: pyffi::Once<Vec<Remover>> = pyffi::Once::new();
    if let Some(found) = FOUND.get() {
        return found.clone();
    }
    let class_of = |(module, class): Class| {
        let found = pyffi::in_module(pyffi::module(module), class);
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
        if slot.is_null() || (*slot).is_null() || !quiet_keys(*slot) {
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
            let Ok(value) = entry(values, identity) else {
                return true;
            };
            if !value.is_null() {
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
    let found = pyffi::in_module(pyffi::module(module), name);
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
    if holds_unseen(o) {
        return Err(());
    }
    if ffi::Py_TYPE(o) as Obj != class_named(c"torch", c"Tensor") {
        return Err(());
    }
    let Some(memory) = own_storage(o) else {
        return Err(());
    };
    if forward_level() || !plainly_laid_out(o) {
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

/// The Python object of the storage of a tensor, where the memory is
/// torch's own allocation (the storage can be resized); None for memory
/// that something else owns, and for a tensor without such a storage.
/// Both are asked through the method of the class under every tensor and
/// under every storage, found once: what somebody put on a class of
/// tensors or of storages under these names is not called for the engine.
unsafe fn own_storage(o: Obj) -> Option<PyObj> {
    static METHODS: pyffi::Once<Option<(PyObj, PyObj)>> = pyffi::Once::new();
    let methods = METHODS.get_or_init(|| {
        let of = |class: &std::ffi::CStr, name: &std::ffi::CStr| {
            let class = pyffi::in_module(pyffi::module(c"torch._C"), class);
            let found = if class.is_null() { null_mut() } else { ffi::PyObject_GetAttrString(class, name.as_ptr()) };
            ffi::PyErr_Clear();
            (!found.is_null()).then(|| PyObj::from_owned(found))
        };
        Some((of(c"TensorBase", c"untyped_storage")?, of(c"StorageBase", c"resizable")?))
    });
    let (storage, resizable) = methods.as_ref()?;
    if ffi::PyObject_TypeCheck(o, torch_classes()[2] as *mut ffi::PyTypeObject) == 0 {
        return None;
    }
    let memory = ffi::PyObject_CallOneArg(storage.as_ptr(), o);
    // A tensor without a storage of this kind says so by raising.
    if memory.is_null() {
        ffi::PyErr_Clear();
        return None;
    }
    let memory = PyObj::from_owned(memory);
    let said = ffi::PyObject_CallOneArg(resizable.as_ptr(), memory.as_ptr());
    let own = said == ffi::Py_True();
    xdecref(said);
    ffi::PyErr_Clear();
    own.then_some(memory)
}

/// Whether nothing is told when the storage of a tensor ends: its memory
/// is torch's own allocation, and the Python object of the storage, which
/// ends with it, ends quietly (`dies_quietly`: no finalizer, no weak
/// reference that is told, nothing of its own whose end runs code). What
/// holds another tensor on the storage for longer than the tensor lives
/// asks this first.
pub unsafe fn storage_ends_quietly(o: Obj) -> bool {
    own_storage(o).is_some_and(|memory| dies(memory.as_ptr(), true))
}

/// Whether a tensor is one whose parts are all in its one storage: dense
/// and strided, not quantized (the scales and zero points per channel are
/// tensors that C++ holds for it), not nested, not of another library's
/// layout, and no wrapper around another tensor (a transform of
/// `torch.func`, a functional tensor), which C++ holds for it. Each is
/// asked of the tensor through an attribute or a function that torch has
/// in C; false where one cannot be asked.
unsafe fn plainly_laid_out(o: Obj) -> bool {
    static PARTS: pyffi::Once<Option<(PyObj, PyObj, PyObj)>> = pyffi::Once::new();
    let parts = PARTS.get_or_init(|| {
        let of = |module: &std::ffi::CStr, name: &std::ffi::CStr| {
            let found = pyffi::in_module(pyffi::module(module), name);
            (!found.is_null()).then(|| PyObj::from_borrowed(found))
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
    let module = pyffi::module(c"torch.autograd.forward_ad");
    if module.is_null() {
        return false;
    }
    let level = pyffi::in_module(module, c"_current_level");
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
        static RANGES: pyffi::Once<[usize; 2]> = pyffi::Once::new();
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
    dies(o, true)
}

/// `dies_quietly`; with `unseen` false, what a tensor holds outside of
/// what it shows the garbage collector is not asked for, as it is not for
/// a tensor without a Python object.
unsafe fn dies(o: Obj, unseen_too: bool) -> bool {
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
        if unseen_too && is_tensor(held) {
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

/// Whether letting go of one reference to an object runs no Python code
/// where it is let go of: there is no object; something else refers to
/// it, so it does not end here; it is a number, a text or None; or its end
/// is shown to run none (`dies_quietly`). What C++ alone holds on to (the
/// history of a tensor that requires a gradient, what a storage frees) is
/// not seen from here, for a tensor with a Python object as for one
/// without: Python code that such an end gets to is met where its frame
/// starts.
pub unsafe fn lets_go_quietly(o: Obj) -> bool {
    o.is_null() || capi::is_unboxed(o) || ffi::Py_REFCNT(o) > 1 || ends_quietly(o)
}

/// `lets_go_quietly` for an object that this is the last reference to.
pub unsafe fn ends_quietly(o: Obj) -> bool {
    is_plain(o) || dies(o, false)
}

/// Whether drawing the next item from an iterator runs no Python code,
/// whoever made the iterator. One of the interpreter's own over a list, a
/// tuple, a range, a dict or a text reads the item out of what it goes
/// over, and lets go of that at its end: so what it goes over is held by
/// something else too, or is let go of quietly (`lets_go_quietly`). An
/// `enumerate` or a `zip` draws on the iterators it was given, which are
/// asked the same, and lets go of the items it gave last. Not a `map` or
/// a `filter` (they call a function), what goes over an `OrderedDict` (it
/// asks a class made of it for each item), or anything else.
pub unsafe fn draws_quietly(o: Obj) -> bool {
    draws_in_c(o, 0)
}

unsafe fn draws_in_c(o: Obj, depth: usize) -> bool {
    // What an iterator shows the garbage collector, in the order it shows
    // it: of an `enumerate` the iterator it draws on first, of a `zip` the
    // tuple of them.
    struct Shown {
        held: [Obj; 4],
        count: usize,
    }
    unsafe extern "C" fn shown(held: Obj, into: *mut std::ffi::c_void) -> std::ffi::c_int {
        let into = &mut *(into as *mut Shown);
        if into.count == into.held.len() {
            return 1;
        }
        into.held[into.count] = held;
        into.count += 1;
        0
    }
    static OWN: pyffi::Once<Vec<usize>> = pyffi::Once::new();
    if o.is_null() || capi::is_unboxed(o) {
        return false;
    }
    let own = OWN.get_or_init(|| {
        // Made of the interpreter's own containers, by its functions of C:
        // no name is looked up where the user could have put another.
        let class_of = |made: Obj| {
            let class = if made.is_null() { 0 } else { ffi::Py_TYPE(made) as usize };
            xdecref(made);
            class
        };
        let over = |held: Obj| {
            let class = if held.is_null() { 0 } else { class_of(ffi::PyObject_GetIter(held)) };
            xdecref(held);
            class
        };
        let called = |class: *mut ffi::PyTypeObject, with: Obj| {
            let made = if with.is_null() { null_mut() } else { ffi::PyObject_CallOneArg(class as Obj, with) };
            xdecref(with);
            made
        };
        let view = |name: &std::ffi::CStr| {
            let (dict, name) = (ffi::PyDict_New(), ffi::PyUnicode_FromString(name.as_ptr()));
            let made = if dict.is_null() || name.is_null() { null_mut() } else { ffi::PyObject_CallMethodNoArgs(dict, name) };
            xdecref(dict);
            xdecref(name);
            made
        };
        let range = std::ptr::addr_of_mut!(ffi::PyRange_Type);
        let own = vec![
            std::ptr::addr_of_mut!(ffi::PyEnum_Type) as usize,
            std::ptr::addr_of_mut!(ffi::PyZip_Type) as usize,
            over(ffi::PyList_New(0)),
            over(ffi::PyTuple_New(0)),
            over(called(range, ffi::PyLong_FromLong(1))),
            over(called(range, ffi::PyLong_FromDouble(1e30))),
            over(ffi::PyDict_New()),
            over(view(c"keys")),
            over(view(c"values")),
            over(view(c"items")),
            over(ffi::PyUnicode_FromString(c"".as_ptr())),
            over(ffi::PyUnicode_FromString(c"\u{20ac}".as_ptr())),
            class_of(called(std::ptr::addr_of_mut!(ffi::PyReversed_Type), ffi::PyList_New(0))),
        ];
        // The classes are the interpreter's, and stay.
        ffi::PyErr_Clear();
        own.into_iter().filter(|class| *class != 0).collect()
    });
    let class = ffi::Py_TYPE(o);
    let Some(which) = own.iter().position(|other| *other == class as usize) else {
        return false;
    };
    let mut refers = Shown { held: [null_mut(); 4], count: 0 };
    if let Some(traverse) = (*class).tp_traverse {
        if traverse(o, shown, &mut refers as *mut Shown as *mut std::ffi::c_void) != 0 {
            return false;
        }
    }
    let held = &refers.held[..refers.count];
    if which >= 2 {
        return held.iter().all(|other| lets_go_quietly(*other));
    }
    // An `enumerate` (0) or a `zip` (1).
    let Some((drawn, rest)) = held.split_first() else {
        return false;
    };
    let draws = if which == 0 {
        depth < 2 && draws_in_c(*drawn, depth + 1)
    } else {
        depth < 2 && ffi::PyTuple_CheckExact(*drawn) != 0 && (0..ffi::PyTuple_GET_SIZE(*drawn)).all(|i| draws_in_c(ffi::PyTuple_GET_ITEM(*drawn, i), depth + 1))
    };
    // What it gave last (a tuple it uses again), and the count.
    draws
        && rest.iter().all(|other| {
            if ffi::PyTuple_CheckExact(*other) != 0 {
                (0..ffi::PyTuple_GET_SIZE(*other)).all(|i| lets_go_quietly(ffi::PyTuple_GET_ITEM(*other, i)))
            } else {
                is_plain(*other)
            }
        })
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
/// hold to an object: the registers of the path (`own`), what the function
/// had before the path (`given`: such a value is not the path's), and the
/// objects of classes that the path made (`made`).
pub struct Scan<'a> {
    own: &'a [Obj],
    given: &'a [Obj],
    made: &'a [Obj],
    /// What the registers hold, in order of where it is, once looked at.
    slots: Option<Vec<Obj>>,
    /// The values that may be something the path made and that hold
    /// references: containers, iterators and bound methods of the
    /// interpreter's own classes, and the objects in `made`.
    temps: Vec<Obj>,
}

impl<'a> Scan<'a> {
    pub fn new(own: &'a [Obj], given: &'a [Obj], made: &'a [Obj]) -> Scan<'a> {
        Scan { own, given, made, slots: None, temps: Vec::new() }
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
            temps.retain(|other| !self.given.contains(other) && (holds_others(*other) || self.made.contains(other)));
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
/// interpreter is asked for a module that was imported before, which it
/// then only looks up. For a module in a package (`import a.b`) it looks
/// up the module and then the outermost package, which is what the
/// statement gives.
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
    if !own || !whole || ffi::PyUnicode_CheckExact(name) == 0 {
        return false;
    }
    // One that is still being imported is waited for.
    let imported = |name: &str| {
        let Ok(name) = std::ffi::CString::new(name) else {
            return false;
        };
        let module = pyffi::module(&name);
        if module.is_null() || ffi::PyModule_CheckExact(module) == 0 {
            return false;
        }
        let spec = pyffi::in_module(module, c"__spec__");
        let slot = if spec.is_null() { std::ptr::null_mut() } else { pyffi::_PyObject_GetDictPtr(spec) };
        let done = !slot.is_null()
            && !(*slot).is_null()
            && ffi::PyDict_GetItemString(*slot, c"_initializing".as_ptr()) == ffi::Py_False();
        ffi::PyErr_Clear();
        done
    };
    let name = pyffi::str_to_string(name);
    imported(&name) && name.split('.').next().is_some_and(imported)
}

/// Whether `len(container)`, or unpacking it, runs no Python code.
pub unsafe fn quiet_sized(container: Obj) -> bool {
    is_plain_container(container) || (!capi::is_unboxed(container) && ffi::PyAnySet_CheckExact(container) != 0)
}

/// Whether `item in container` runs no Python code: the item is compared
/// with what the container holds.
pub unsafe fn quiet_contains(item: Obj, container: Obj) -> bool {
    if capi::is_unboxed(item) || capi::is_unboxed(container) {
        return false;
    }
    // A list or a tuple is gone over, and each item is compared with the
    // value (the item's class is asked first).
    if ffi::PyList_CheckExact(container) != 0 || ffi::PyTuple_CheckExact(container) != 0 {
        return (0..ffi::PyObject_Size(container)).all(|i| compares_in_c(item) && compares_in_c(sequence_item(container, i)));
    }
    // A dict, a set and a text hash the value or search for it.
    if !is_plain(item) {
        return false;
    }
    if is_plain_dict(container) {
        return quiet_keys(container);
    }
    if ffi::PyAnySet_CheckExact(container) != 0 {
        return quiet_members(container);
    }
    ffi::PyUnicode_CheckExact(container) != 0 && ffi::PyUnicode_CheckExact(item) != 0
}

/// Whether `bool(o)` runs no Python code.
pub unsafe fn quiet_truth(o: Obj) -> bool {
    !capi::is_unboxed(o) && (is_plain(o) || quiet_sized(o))
}

/// Moves the version counter of a tensor on by `count`, as that many writes
/// to it in place would (`torch._C._increment_version`, which takes the
/// tensors to do it for).
pub unsafe fn move_version(tensor: Obj, count: u64) {
    static BUMP: pyffi::Once<Option<PyObj>> = pyffi::Once::new();
    let bump = BUMP.get_or_init(|| {
        let found = pyffi::in_module(pyffi::module(c"torch._C"), c"_increment_version");
        (!found.is_null()).then(|| PyObj::from_borrowed(found))
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

/// Whether the class of a value compares it with the function of `str`,
/// `int`, `float` or `object`: the class is one of them, or one written
/// in Python that has no comparison of its own anywhere above it (a
/// member of an enum, an object that is equal to itself only, None).
/// The interpreter keeps what a class compares with in this slot, and
/// puts its own function there when the class or one above it gets an
/// `__eq__` or an `__lt__`. Each of the four looks at what it is handed,
/// answers or says that it has no answer, and calls nothing. So does
/// the class of None, which no class can be made of; a class with
/// nothing in the slot is not asked at all.
unsafe fn compares_in_c(o: Obj) -> bool {
    if o.is_null() || capi::is_unboxed(o) {
        return false;
    }
    let slot = |ty: *const ffi::PyTypeObject| (*ty).tp_richcompare.map(|f| f as usize);
    let own = slot(ffi::Py_TYPE(o));
    own.is_none()
        || o == ffi::Py_None()
        || [
            std::ptr::addr_of!(ffi::PyUnicode_Type),
            std::ptr::addr_of!(ffi::PyLong_Type),
            std::ptr::addr_of!(ffi::PyFloat_Type),
            std::ptr::addr_of!(ffi::PyBaseObject_Type),
        ]
        .iter()
        .any(|ty| slot(*ty) == own)
}

/// Whether `str` of a value is the text `type` makes of a class: the value
/// is a class, and its class turns it into text as `type` does (it has no
/// `__str__` and no `__repr__` of its own, which the interpreter keeps in
/// these two slots). `type` reads the module and the name of the class in
/// the class's own dict, whose keys are texts here, and calls nothing.
unsafe fn class_text(o: Obj) -> bool {
    if o.is_null() || capi::is_unboxed(o) || ffi::PyType_Check(o) == 0 {
        return false;
    }
    let (meta, plain, object) = (ffi::Py_TYPE(o), std::ptr::addr_of!(ffi::PyType_Type), std::ptr::addr_of!(ffi::PyBaseObject_Type));
    let says = (*meta).tp_str.map(|f| f as usize) == (*object).tp_str.map(|f| f as usize)
        && (*meta).tp_repr.map(|f| f as usize) == (*plain).tp_repr.map(|f| f as usize);
    let dict = (*(o as *mut ffi::PyTypeObject)).tp_dict;
    says && (is_static(o as *mut ffi::PyTypeObject) || (!dict.is_null() && quiet_keys(dict)))
}

/// Whether two values compare in C code: two values whose classes
/// compare with the function of `str`, `int`, `float` or `object`
/// (`compares_in_c`: the interpreter asks the class of one, then that of
/// the other, and with no answer says which object each is, or raises
/// TypeError for an order); dtypes and devices; or tuples of numbers,
/// texts and None (the sizes of a tensor with those of another, or with
/// a tuple).
pub unsafe fn quiet_compare(a: Obj, b: Obj) -> bool {
    if compares_in_c(a) && compares_in_c(b) {
        return true;
    }
    let text = |o: Obj| !capi::is_unboxed(o) && ffi::PyUnicode_CheckExact(o) != 0;
    let token = |o: Obj| !capi::is_unboxed(o) && is_token(o);
    // A tuple, or one of a class that is defined in C and compares as a
    // tuple does (a `torch.Size`).
    let row = |o: Obj| {
        if capi::is_unboxed(o) || ffi::PyTuple_Check(o) == 0 {
            return false;
        }
        let (ty, tuple) = (ffi::Py_TYPE(o), std::ptr::addr_of!(ffi::PyTuple_Type));
        let compares = (*ty).tp_richcompare.map(|f| f as usize) == (*tuple).tp_richcompare.map(|f| f as usize);
        (ffi::PyTuple_CheckExact(o) != 0 || (is_static(ty) && compares))
            && (0..ffi::PyTuple_GET_SIZE(o)).all(|i| is_plain(ffi::PyTuple_GET_ITEM(o, i)))
    };
    (is_number(a) && is_number(b)) || (text(a) && text(b)) || (token(a) && token(b)) || (row(a) && row(b))
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
    // A tuple of a class that is defined in C and goes over itself as a
    // tuple does (a `torch.Size`).
    let tuple = std::ptr::addr_of!(ffi::PyTuple_Type);
    if ffi::PyTuple_Check(o) != 0 && (*ty).tp_iter.map(|f| f as usize) == (*tuple).tp_iter.map(|f| f as usize) {
        return Some(false);
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

/// The address of one of them: of a class, where the interpreter has it;
/// of a function, what the module `builtins` has under the name if that is
/// the function of C that the module was made with (not one that was put
/// there in its place), and 0 otherwise. Kept once every one is found.
unsafe fn builtin_named(which: Builtin) -> usize {
    static FOUND: pyffi::Once<Vec<usize>> = pyffi::Once::new();
    let at = BUILTINS.iter().position(|(_, other)| *other == which).unwrap();
    if let Some(found) = FOUND.get() {
        return found[at];
    }
    let class = |which: Builtin| -> Option<*mut ffi::PyTypeObject> {
        Some(match which {
            Builtin::Range => std::ptr::addr_of_mut!(ffi::PyRange_Type),
            Builtin::Enumerate => std::ptr::addr_of_mut!(ffi::PyEnum_Type),
            Builtin::Zip => std::ptr::addr_of_mut!(ffi::PyZip_Type),
            Builtin::Int => std::ptr::addr_of_mut!(ffi::PyLong_Type),
            Builtin::Float => std::ptr::addr_of_mut!(ffi::PyFloat_Type),
            Builtin::Bool => std::ptr::addr_of_mut!(ffi::PyBool_Type),
            Builtin::Tuple => std::ptr::addr_of_mut!(ffi::PyTuple_Type),
            Builtin::List => std::ptr::addr_of_mut!(ffi::PyList_Type),
            Builtin::Type => std::ptr::addr_of_mut!(ffi::PyType_Type),
            Builtin::Str => std::ptr::addr_of_mut!(ffi::PyUnicode_Type),
            _ => return None,
        })
    };
    let found: Vec<usize> = BUILTINS.iter().map(|(name, which)| class(*which).map_or_else(|| pyffi::builtin(name) as usize, |class| class as usize)).collect();
    let asked = found[at];
    if found.iter().all(|address| *address != 0) {
        FOUND.set(found);
    }
    asked
}

/// Whether an object is code of a generator (and of no coroutine, nor of
/// an asynchronous generator).
pub unsafe fn generator_code(code: Obj) -> bool {
    const CO_GENERATOR_LIKE: std::os::raw::c_long = 0x20 | 0x80 | 0x100 | 0x200;
    if code.is_null() || capi::is_unboxed(code) || ffi::Py_TYPE(code) != std::ptr::addr_of_mut!(ffi::PyCode_Type) {
        return false;
    }
    let flags = ffi::PyObject_GetAttrString(code, c"co_flags".as_ptr());
    let found = if flags.is_null() { -1 } else { ffi::PyLong_AsLong(flags) };
    xdecref(flags);
    ffi::PyErr_Clear();
    found >= 0 && found & CO_GENERATOR_LIKE == 0x20
}

/// Whether an object is a Python function whose code is a generator's.
pub unsafe fn generator_function(callee: Obj) -> bool {
    !callee.is_null() && !capi::is_unboxed(callee) && ffi::PyFunction_Check(callee) != 0 && generator_code(ffi::PyFunction_GetCode(callee))
}

/// Whether a callee is the builtin `tuple` (true) or `list` (false), which
/// go over what they are given to its end and keep what it gives.
pub unsafe fn keeps_all(callee: Obj) -> Option<bool> {
    if callee.is_null() {
        return None;
    }
    if builtin_named(Builtin::Tuple) == callee as usize {
        return Some(true);
    }
    (builtin_named(Builtin::List) == callee as usize).then_some(false)
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

/// Whether a callee is the builtin `hasattr`.
pub unsafe fn is_hasattr(callee: Obj) -> bool {
    builtin_named(Builtin::HasAttr) == callee as usize
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
    static HINT: pyffi::Once<PyObj> = pyffi::Once::new();
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

/// What an abstract base class keeps its answers in, as CPython 3.12 lays
/// it out (`_abc_data` in `Modules/_abc.c`).
#[cfg(not(Py_3_13))]
#[repr(C)]
struct AbcData {
    ob_base: ffi::PyObject,
    registry: Obj,
    cache: Obj,
    negative: Obj,
    version: std::os::raw::c_ulonglong,
}

/// The builtin that `ABCMeta.__instancecheck__` calls, the builtin that
/// gives the token of the answers that are kept, and the class of what an
/// abstract base class keeps them in: found once, by where they are (the
/// last as the class of what `abc.ABC` has). None while `abc` is not
/// imported.
#[cfg(not(Py_3_13))]
unsafe fn abc_parts() -> Option<(usize, usize, usize)> {
    static FOUND: pyffi::Once<(usize, usize, usize)> = pyffi::Once::new();
    if let Some(found) = FOUND.get() {
        return Some(*found);
    }
    let of = |module: &std::ffi::CStr, name: &std::ffi::CStr| pyffi::in_module(pyffi::module(module), name);
    let (check, token, base) = (of(c"_abc", c"_abc_instancecheck"), of(c"_abc", c"get_cache_token"), of(c"abc", c"ABC"));
    let builtin = |o: Obj| !o.is_null() && ffi::PyCFunction_Check(o) != 0;
    if !builtin(check) || !builtin(token) || base.is_null() || ffi::PyType_Check(base) == 0 {
        return None;
    }
    let name = PyObj::from_owned(ffi::PyUnicode_InternFromString(c"_abc_impl".as_ptr()));
    let kept = pyffi::_PyType_Lookup(base as *mut ffi::PyTypeObject, name.as_ptr());
    if kept.is_null() {
        return None;
    }
    // The modules keep them.
    let found = (check as usize, token as usize, ffi::Py_TYPE(kept) as usize);
    let _ = FOUND.set(found);
    Some(found)
}

/// For a call of the builtin behind `isinstance(instance, cls)` for an
/// abstract base class (`_abc._abc_instancecheck(cls, instance)`): the
/// answer, where the class has kept one for the class of the instance and
/// the builtin gives it without calling anything. That is so when what
/// the class keeps its answers in is found as `type` finds an attribute;
/// the instance says it is of the class it is of, and saying so runs no
/// Python code; classes are hashed and compared as `type` does it (the
/// class of the instance, and every class an answer is kept for); and the
/// class of the instance is among those the answer is yes for, or among
/// those it is no for while no class has been registered since. None for
/// any other call: the builtin then asks `__subclasscheck__`.
#[cfg(not(Py_3_13))]
pub unsafe fn abc_kept(callee: Obj, args: &[Obj], api: &TcfgApi) -> Option<bool> {
    static NAMES: pyffi::Once<(PyObj, PyObj)> = pyffi::Once::new();
    let (check, token, data) = abc_parts()?;
    if callee as usize != check || args.len() != 2 || args.iter().any(|o| o.is_null() || capi::is_unboxed(*o)) {
        return None;
    }
    let (cls, instance) = (args[0], args[1]);
    if ffi::PyType_Check(cls) == 0 {
        return None;
    }
    let (kept, class) = NAMES.get_or_init(|| {
        let name = |text: &std::ffi::CStr| PyObj::from_owned(ffi::PyUnicode_InternFromString(text.as_ptr()));
        (name(c"_abc_impl"), name(c"__class__"))
    });
    let meta = ffi::Py_TYPE(cls);
    let reads = (*meta).tp_getattro.map(|own| own as usize) == (*std::ptr::addr_of!(ffi::PyType_Type)).tp_getattro.map(|own| own as usize);
    let held = pyffi::_PyType_Lookup(cls as *mut ffi::PyTypeObject, kept.as_ptr());
    if !reads || !pyffi::_PyType_Lookup(meta, kept.as_ptr()).is_null() || held.is_null() || ffi::Py_TYPE(held) as usize != data {
        return None;
    }
    if !quiet_attr(instance, class.as_ptr(), api) {
        return None;
    }
    let said = ffi::PyObject_GetAttr(instance, class.as_ptr());
    ffi::PyErr_Clear();
    xdecref(said);
    let of = ffi::Py_TYPE(instance) as Obj;
    if said != of {
        return None;
    }
    let object = std::ptr::addr_of!(ffi::PyBaseObject_Type);
    let as_type = |class: Obj| {
        let meta = ffi::Py_TYPE(class);
        (*meta).tp_hash.map(|own| own as usize) == (*object).tp_hash.map(|own| own as usize)
            && (*meta).tp_richcompare.map(|own| own as usize) == (*object).tp_richcompare.map(|own| own as usize)
    };
    if !as_type(of) {
        return None;
    }
    // Whether the class is among those a set of weak references is of;
    // None where one of them is of a class that is hashed or compared by
    // code of its own.
    let within = |set: Obj| -> Option<bool> {
        if set.is_null() {
            return Some(false);
        }
        if ffi::PySet_CheckExact(set) == 0 {
            return None;
        }
        let over = ffi::PyObject_GetIter(set);
        if over.is_null() {
            ffi::PyErr_Clear();
            return None;
        }
        let (mut found, mut plain) = (false, true);
        loop {
            let reference = ffi::PyIter_Next(over);
            if reference.is_null() {
                break;
            }
            let to = if ffi::PyWeakref_Check(reference) != 0 { ffi::PyWeakref_GetObject(reference) } else { null_mut() };
            found |= to == of;
            plain &= !to.is_null() && (to == ffi::Py_None() || (ffi::PyType_Check(to) != 0 && as_type(to)));
            ffi::Py_DECREF(reference);
        }
        ffi::PyErr_Clear();
        ffi::Py_DECREF(over);
        plain.then_some(found)
    };
    let answers = held as *const AbcData;
    if within((*answers).cache)? {
        return Some(true);
    }
    let now = ffi::PyObject_CallNoArgs(token as Obj);
    let current = if now.is_null() { None } else { Some(ffi::PyLong_AsUnsignedLongLong(now)) };
    xdecref(now);
    ffi::PyErr_Clear();
    (current == Some((*answers).version) && within((*answers).negative)?).then_some(false)
}

/// The modules whose functions a path may call on any arguments (`torch._C`:
/// those that ask about the state of the thread or of what they are given)
/// or on numbers (`math`).
unsafe fn quiet_modules() -> (usize, usize, usize) {
    static FOUND: pyffi::Once<(usize, usize, usize)> = pyffi::Once::new();
    *FOUND.get_or_init(|| {
        // The interpreter keeps them.
        let [state, math, operator] = [c"torch._C", c"math", c"_operator"].map(|name| pyffi::module(name) as usize);
        (state, math, operator)
    })
}

/// Whether a call is one of a function a path may call, with arguments on
/// which it runs no Python code and changes nothing: Some, with whether it
/// makes something that is the path's own to change or to use up (a list,
/// an iterator, a `range`, an `enumerate`, a `zip`). `own` says of an
/// argument whether the path made it: an iterator is only gone over then.
pub unsafe fn pure_call(callee: Obj, args: &[Obj], own: impl Fn(usize) -> bool, api: &TcfgApi) -> Option<bool> {
    const ASKS: [&[u8]; 7] = [b"_has_torch_function", b"_get_", b"_is_", b"is_", b"get_", b"_has_", b"_cuda_is"];
    if callee.is_null() || capi::is_unboxed(callee) || args.iter().any(|o| o.is_null()) {
        return None;
    }
    // What `object` reads of the object it is bound to, where that runs
    // no Python code. Of a tensor nothing is read this way.
    let reader = generic_reader(callee);
    if !reader.is_null() {
        let mut info = TcfgTensorInfo::empty();
        let named = args.len() == 1 && !capi::is_unboxed(args[0]) && ffi::PyUnicode_CheckExact(args[0]) != 0;
        let fine = named && (api.tensor_info)(reader, &mut info) == 0 && quiet_generic(reader, args[0]);
        xdecref(reader);
        return fine.then_some(false);
    }
    // A call of a generator function makes a generator, the path's own,
    // and runs nothing of the function.
    if generator_function(callee) {
        return Some(true);
    }
    if ffi::PyCFunction_Check(callee) != 0 {
        // An abstract base class that has its answer for the class of an
        // instance: the builtin gives it, and calls nothing.
        #[cfg(not(Py_3_13))]
        if abc_kept(callee, args, api).is_some() {
            return Some(false);
        }
        let function = callee as *mut ffi::PyCFunctionObject;
        let (state, math, operator) = quiet_modules();
        let of = (*function).m_self as usize;
        let name = std::ffi::CStr::from_ptr((*(*function).m_ml).ml_name);
        // One that `torch._C` has under its name, however it is bound.
        let asks = ASKS.iter().any(|asks| name.to_bytes().starts_with(asks));
        if asks && of != state && state != 0 {
            let found = ffi::PyObject_GetAttrString(state as Obj, name.as_ptr());
            ffi::PyErr_Clear();
            xdecref(found);
            if found == callee {
                return Some(false);
            }
        }
        if of != 0 && of == state {
            return asks.then_some(false);
        }
        // A method of a number, a text or a tuple of such, on such values:
        // they cannot change, and what their classes do is written in C.
        if of != 0 && unchanging(of as Obj) {
            return args.iter().all(|o| quiet_text(*o)).then_some(false);
        }
        // What a dict has, asked of a method that was bound to it earlier.
        if of != 0 && is_plain_dict(of as Obj) {
            let name = ffi::PyUnicode_FromString((*(*function).m_ml).ml_name);
            let fine = !name.is_null() && pure_method(of as Obj, name, args, api);
            ffi::PyErr_Clear();
            xdecref(name);
            return fine.then_some(false);
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
        Builtin::Str => args.len() == 1 && (numbers() || ffi::PyUnicode_CheckExact(args[0]) != 0 || class_text(args[0])),
        Builtin::Min | Builtin::Max => args.len() >= 2 && numbers(),
        Builtin::Bool => args.len() == 1 && quiet_truth(args[0]),
        Builtin::Type | Builtin::Callable => args.len() == 1,
        Builtin::GetAttr | Builtin::HasAttr => {
            let found = which == Builtin::GetAttr && args.len() == 3;
            let named = (args.len() == 2 || found) && !capi::is_unboxed(args[1]) && ffi::PyUnicode_CheckExact(args[1]) != 0;
            // What is not there is answered too where nothing is raised
            // out of the call.
            let caught = which == Builtin::HasAttr || found;
            let absent = || quiet_absent(args[0], args[1]) || plainly_absent(args[0], args[1]);
            named && (quiet_attr(args[0], args[1], api) || (caught && absent()))
        }
        Builtin::IsInstance => {
            static CLASS: pyffi::Once<PyObj> = pyffi::Once::new();
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

/// A number, a text, bytes, or a tuple of numbers and texts: a value that
/// cannot change, whose methods are written in C.
unsafe fn unchanging(o: Obj) -> bool {
    !capi::is_unboxed(o)
        && (is_number(o)
            || ffi::PyUnicode_CheckExact(o) != 0
            || ffi::PyBytes_CheckExact(o) != 0
            || (ffi::PyTuple_CheckExact(o) != 0 && quiet_text(o)))
}

/// Whether a call of a method takes an entry out of a dict and gives what
/// was under it (or the default), running no Python code: `pop` of an
/// exact dict with a plain key, whose keys are all plain. Without a
/// default it raises KeyError where the key is not there, and does
/// nothing else. For a dict the path made: it is changed.
pub unsafe fn quiet_pop(recv: Obj, name: Obj, args: &[Obj]) -> bool {
    !recv.is_null()
        && !capi::is_unboxed(recv)
        && ffi::PyDict_CheckExact(recv) != 0
        && (1..=2).contains(&args.len())
        && args.iter().all(|o| !o.is_null())
        && ffi::PyUnicode_CompareWithASCIIString(name, c"pop".as_ptr()) == 0
        && is_plain_key(args[0])
        && quiet_keys(recv)
}

/// Whether a call of a method runs no Python code, no kernel, and changes
/// nothing: what a dict has, as a view or under a key; what a tensor says
/// of itself; any method of a number, a text or a tuple of such on such
/// values (`unchanging`).
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
    if capi::is_unboxed(recv) {
        return false;
    }
    if unchanging(recv) {
        return args.iter().all(|o| quiet_text(*o));
    }
    if !is_plain_dict(recv) {
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
    pub new: PyObj,
    pub init: PyObj,
    pub iter: PyObj,
    pub len: PyObj,
    pub getitem: PyObj,
    pub get: PyObj,
    pub getattr: PyObj,
    pub setattr: PyObj,
    pub instancecheck: PyObj,
    pub getattribute: PyObj,
    pub contains: PyObj,
    pub truth: PyObj,
}

pub unsafe fn specials() -> &'static Specials {
    static NAMES: pyffi::Once<Specials> = pyffi::Once::new();
    let name = |name: &std::ffi::CStr| PyObj::from_owned(ffi::PyUnicode_InternFromString(name.as_ptr()));
    NAMES.get_or_init(|| Specials {
        new: name(c"__new__"),
        init: name(c"__init__"),
        iter: name(c"__iter__"),
        len: name(c"__len__"),
        getitem: name(c"__getitem__"),
        get: name(c"__get__"),
        getattr: name(c"__getattr__"),
        setattr: name(c"__setattr__"),
        instancecheck: name(c"__instancecheck__"),
        getattribute: name(c"__getattribute__"),
        contains: name(c"__contains__"),
        truth: name(c"__bool__"),
    })
}

/// Whether the class of an object has nothing under `__bool__`, as the
/// interpreter's own truth test finds it: its truth is whether its
/// `__len__` gives what is not zero, where it has one. A path that takes
/// the truth that way asks this again at every call (`Assert::Lacks`).
pub unsafe fn lacks_truth(o: Obj) -> bool {
    !o.is_null() && !capi::is_unboxed(o) && pyffi::_PyType_Lookup(ffi::Py_TYPE(o), specials().truth.as_ptr()).is_null()
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

/// The C function that makes an object of a class where the class is
/// called: the `__new__` of `object`, which the class has unless it or a
/// class it is made of defines one. It gives a new, empty object of the
/// class whatever else it is handed (the arguments of the call are for
/// `__init__`), runs no Python code, and what it does depends on nothing but
/// the class. Only for a class defined in Python whose own class is `type`
/// (calling it is `__new__` and then `__init__`), that can have objects
/// (an abstract class raises) and whose objects go without a `__del__`
/// being called (an object that a path made and that a later check of the
/// path gives up is let go of). Null for any other. Borrowed.
pub unsafe fn fresh_new(class: Obj) -> Obj {
    const IS_ABSTRACT: std::os::raw::c_ulong = 1 << 20;
    if class.is_null() || capi::is_unboxed(class) || ffi::Py_TYPE(class) != std::ptr::addr_of_mut!(ffi::PyType_Type) {
        return null_mut();
    }
    let ty = class as *mut ffi::PyTypeObject;
    if is_static(ty) || ffi::PyType_HasFeature(ty, IS_ABSTRACT) != 0 || (*ty).tp_finalize.is_some() || (*ty).tp_del.is_some() {
        return null_mut();
    }
    let object = std::ptr::addr_of_mut!(ffi::PyBaseObject_Type);
    let found = pyffi::_PyType_Lookup(ty, specials().new.as_ptr());
    if !found.is_null() && found == pyffi::_PyType_Lookup(object, specials().new.as_ptr()) { found } else { null_mut() }
}

/// Whether an object of a class keeps what is stored under a name in its
/// own dict, by C code that cannot fail: the class stores attributes as
/// `object` does, has nothing of its own that takes a store under the name
/// (a property, a slot), and its objects have a dict. Not for the names the
/// interpreter keeps for itself.
pub unsafe fn stores_plainly(class: Obj, name: Obj) -> bool {
    if class.is_null() || capi::is_unboxed(class) || ffi::PyType_Check(class) == 0 || ffi::PyUnicode_CheckExact(name) == 0 {
        return false;
    }
    let ty = class as *mut ffi::PyTypeObject;
    let text = pyffi::str_to_string(name);
    if is_static(ty) || (*ty).tp_dictoffset == 0 || text.starts_with("__") {
        return false;
    }
    let object = std::ptr::addr_of_mut!(ffi::PyBaseObject_Type);
    let setattr = specials().setattr.as_ptr();
    let descr = pyffi::_PyType_Lookup(ty, name);
    let data = !descr.is_null() && (*ffi::Py_TYPE(descr)).tp_descr_set.is_some();
    !data && pyffi::_PyType_Lookup(ty, setattr) == pyffi::_PyType_Lookup(object, setattr)
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
    // Not shown for a dict that is not looked into (`entry`).
    let lacks = |name: Obj| dict.is_null() || entry(dict, name).is_ok_and(|held| held.is_null());
    lacks(name) && (ffi::PyModule_Check(o) == 0 || lacks(specials().getattr.as_ptr()))
}

/// Whether an object has no attribute of a name and says so without
/// changing anything: neither its class nor the object has it, and the
/// `__getattr__` of the class is one with a shortcut (that of `nn.Module`)
/// none of whose dicts has it, which then raises AttributeError and does
/// nothing else. `hasattr` is False then, and `getattr` gives its default;
/// both call the function, at every call.
pub unsafe fn quiet_absent(obj: Obj, name: Obj) -> bool {
    absent(obj, name)
        && ffi::PyModule_Check(obj) == 0
        && crate::shortcuts::has_getattr(ffi::Py_TYPE(obj))
        && crate::shortcuts::among(obj, name) == Ok(null_mut())
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

/// Whether a callee is `object.__getattribute__` bound to an object that
/// is no tensor and has nothing under a name (`generic_absent`): calling
/// it with the name raises AttributeError and does nothing else.
pub unsafe fn reads_nothing(callee: Obj, name: Obj, api: &TcfgApi) -> bool {
    let reader = generic_reader(callee);
    if reader.is_null() {
        return false;
    }
    let mut info = TcfgTensorInfo::empty();
    let nothing = (api.tensor_info)(reader, &mut info) == 0 && generic_absent(reader, name);
    ffi::Py_DECREF(reader);
    nothing
}

/// Whether an object whose attributes are read as `object` reads them,
/// with no function of its class asked behind that, has nothing under a
/// name: reading it raises AttributeError and does nothing else, and
/// `getattr` with a default gives the default, `hasattr` False.
pub unsafe fn plainly_absent(obj: Obj, name: Obj) -> bool {
    if obj.is_null() || capi::is_unboxed(obj) || ffi::PyModule_Check(obj) != 0 {
        return false;
    }
    let object = std::ptr::addr_of!(ffi::PyBaseObject_Type);
    let generic = (*ffi::Py_TYPE(obj)).tp_getattro.map(|own| own as usize) == (*object).tp_getattro.map(|own| own as usize);
    generic && generic_absent(obj, name)
}

/// The class of a slot wrapper that is bound to an object, and the wrapper
/// that `object` has for `__getattribute__`, where a bound wrapper has its
/// two fields where `pyffi::WrapperObject` says (shown once, on
/// `None.__getattribute__`, against what its attributes say); (0, 0)
/// where it has not.
unsafe fn generic_wrapper() -> (usize, usize) {
    static FOUND: pyffi::Once<(usize, usize)> = pyffi::Once::new();
    *FOUND.get_or_init(|| {
        let names = specials();
        let bound = ffi::PyObject_GetAttr(ffi::Py_None(), names.getattribute.as_ptr());
        let of_object = pyffi::_PyType_Lookup(std::ptr::addr_of_mut!(ffi::PyBaseObject_Type), names.getattribute.as_ptr());
        let mut found = (0, 0);
        if !bound.is_null() && !of_object.is_null() {
            let fields = bound as *mut pyffi::WrapperObject;
            let said = ffi::PyObject_GetAttrString(bound, c"__self__".as_ptr());
            if (*fields).descr == of_object && (*fields).bound == ffi::Py_None() && said == ffi::Py_None() {
                found = (ffi::Py_TYPE(bound) as usize, of_object as usize);
            }
            xdecref(said);
        }
        ffi::PyErr_Clear();
        xdecref(bound);
        found
    })
}

/// No answer of `generic_bound`: the fields of a bound wrapper are not
/// known to be where they are read.
const NOT_READ: Obj = 1 as Obj;

/// `generic_reader`, read off the fields of the callee: the object
/// (borrowed), null for a callee that is something else, `NOT_READ` where
/// the fields are not known.
unsafe fn generic_bound(callee: Obj) -> Obj {
    let (wrapper, of_object) = generic_wrapper();
    if wrapper == 0 {
        return NOT_READ;
    }
    if callee.is_null() || capi::is_unboxed(callee) || ffi::Py_TYPE(callee) as usize != wrapper {
        return null_mut();
    }
    let fields = callee as *mut pyffi::WrapperObject;
    if (*fields).descr as usize == of_object { (*fields).bound } else { null_mut() }
}

/// What `object.__getattribute__(obj, name)` gives (a new reference) for
/// a callee that is that function bound to an object and a name that is a
/// text, where the lookup comes down to finding the value: in the dict of
/// the object, with nothing in its class that takes the lookup first, or
/// in the class as a value that is no descriptor. False for every other
/// callee and lookup: `pure_call` says which run no Python code, and the
/// call is made as it is.
pub unsafe fn holds_generic(callee: Obj, name: Obj, found: &mut Obj, api: &TcfgApi) -> bool {
    let obj = generic_bound(callee);
    let mut info = TcfgTensorInfo::empty();
    if obj.is_null() || obj == NOT_READ || name.is_null() || capi::is_unboxed(name) || ffi::PyUnicode_CheckExact(name) == 0 {
        return false;
    }
    if capi::is_unboxed(obj) || ffi::PyType_Check(obj) != 0 || (api.tensor_info)(obj, &mut info) != 0 {
        return false;
    }
    let descr = pyffi::_PyType_Lookup(ffi::Py_TYPE(obj), name);
    if !descr.is_null() && (*ffi::Py_TYPE(descr)).tp_descr_set.is_some() {
        return false;
    }
    let slot = pyffi::_PyObject_GetDictPtr(obj);
    if !slot.is_null() && !(*slot).is_null() {
        match entry(*slot, name) {
            Err(()) => return false,
            Ok(held) if !held.is_null() => {
                *found = new_ref(held);
                return true;
            }
            Ok(_) => {}
        }
    }
    if descr.is_null() || (*ffi::Py_TYPE(descr)).tp_descr_get.is_some() {
        return false;
    }
    *found = new_ref(descr);
    true
}

/// What `d.get(key)` or `d.get(key, default)` gives (a new reference)
/// for a callee that is the `get` of an exact dict, bound to it, a plain
/// key and a dict whose keys are all plain: the entry, or the default.
/// False for every other callee and call.
pub unsafe fn holds_get(callee: Obj, key: Obj, default: Obj, found: &mut Obj) -> bool {
    static GET: pyffi::Once<usize> = pyffi::Once::new();
    if callee.is_null() || capi::is_unboxed(callee) || ffi::PyCFunction_Check(callee) == 0 {
        return false;
    }
    let get = *GET.get_or_init(|| {
        let dict = ffi::PyDict_New();
        let bound = if dict.is_null() { null_mut() } else { ffi::PyObject_GetAttrString(dict, c"get".as_ptr()) };
        let method = if bound.is_null() || ffi::PyCFunction_Check(bound) == 0 { 0 } else { (*(bound as *mut ffi::PyCFunctionObject)).m_ml as usize };
        ffi::PyErr_Clear();
        xdecref(bound);
        xdecref(dict);
        method
    });
    let function = callee as *mut ffi::PyCFunctionObject;
    let dict = (*function).m_self;
    if get == 0 || (*function).m_ml as usize != get || dict.is_null() || ffi::PyDict_CheckExact(dict) == 0 {
        return false;
    }
    if key.is_null() || !is_plain_key(key) || !quiet_keys(dict) {
        return false;
    }
    let held = ffi::PyDict_GetItemWithError(dict, key);
    if held.is_null() && !ffi::PyErr_Occurred().is_null() {
        ffi::PyErr_Clear();
        return false;
    }
    *found = new_ref(if !held.is_null() {
        held
    } else if default.is_null() {
        ffi::Py_None()
    } else {
        default
    });
    true
}

/// For what `object.__getattribute__` is when it is bound to an object
/// (what `super().__getattribute__(name)` comes down to in a class that
/// reads its attributes in a function of its own): the object. Owned;
/// null for anything else.
pub unsafe fn generic_reader(callee: Obj) -> Obj {
    let held = generic_bound(callee);
    if held != NOT_READ {
        ffi::Py_XINCREF(held);
        return held;
    }
    static WRAPPER: pyffi::Once<usize> = pyffi::Once::new();
    if callee.is_null() || capi::is_unboxed(callee) {
        return null_mut();
    }
    let names = specials();
    let wrapper = *WRAPPER.get_or_init(|| {
        let bound = ffi::PyObject_GetAttr(ffi::Py_None(), names.getattribute.as_ptr());
        ffi::PyErr_Clear();
        let class = if bound.is_null() { 0 } else { ffi::Py_TYPE(bound) as usize };
        xdecref(bound);
        class
    });
    if wrapper == 0 || ffi::Py_TYPE(callee) as usize != wrapper {
        return null_mut();
    }
    let class = ffi::PyObject_GetAttrString(callee, c"__objclass__".as_ptr());
    let name = ffi::PyObject_GetAttrString(callee, c"__name__".as_ptr());
    let lookup = class == std::ptr::addr_of_mut!(ffi::PyBaseObject_Type) as Obj
        && !name.is_null()
        && ffi::PyUnicode_Check(name) != 0
        && ffi::PyUnicode_Compare(name, names.getattribute.as_ptr()) == 0;
    ffi::PyErr_Clear();
    xdecref(class);
    xdecref(name);
    if !lookup {
        return null_mut();
    }
    let bound = ffi::PyObject_GetAttrString(callee, c"__self__".as_ptr());
    ffi::PyErr_Clear();
    bound
}

/// Whether the lookup of `object` finds an attribute of an object without
/// running Python code, whatever the class of the object does when an
/// attribute is read: what `object.__getattribute__(obj, name)` does.
/// False for a name that is not there.
pub unsafe fn quiet_generic(obj: Obj, name: Obj) -> bool {
    if obj.is_null() || capi::is_unboxed(obj) || ffi::PyType_Check(obj) != 0 {
        return false;
    }
    let descr = pyffi::_PyType_Lookup(ffi::Py_TYPE(obj), name);
    let python = !descr.is_null() && python_descriptor(descr, obj);
    let data = !descr.is_null() && (*ffi::Py_TYPE(descr)).tp_descr_set.is_some();
    if data {
        return !python;
    }
    let slot = pyffi::_PyObject_GetDictPtr(obj);
    let own = if slot.is_null() || (*slot).is_null() { Ok(null_mut()) } else { entry(*slot, name) };
    own.is_ok_and(|held| !held.is_null() || (!descr.is_null() && !python))
}

/// The property that the lookup of `object` comes down to for a name of
/// an object, whatever the object holds itself. Null for anything else.
/// Borrowed.
pub unsafe fn generic_property(obj: Obj, name: Obj) -> Obj {
    if obj.is_null() || capi::is_unboxed(obj) || ffi::PyType_Check(obj) != 0 {
        return null_mut();
    }
    let descr = pyffi::_PyType_Lookup(ffi::Py_TYPE(obj), name);
    let property = !descr.is_null() && ffi::Py_TYPE(descr) == std::ptr::addr_of_mut!(ffi::PyProperty_Type);
    if property { descr } else { null_mut() }
}

/// Whether what `super(class, obj)` gives has its fields where
/// `pyffi::SuperObject` says: shown once, on `super(int, True)`, against
/// what its attributes say.
unsafe fn super_fields() -> bool {
    static SHOWN: pyffi::Once<bool> = pyffi::Once::new();
    *SHOWN.get_or_init(|| {
        let (int, flag) = (std::ptr::addr_of_mut!(ffi::PyLong_Type) as Obj, ffi::Py_True());
        let made = ffi::PyObject_CallFunctionObjArgs(std::ptr::addr_of_mut!(ffi::PySuper_Type) as Obj, int, flag, null_mut::<ffi::PyObject>());
        let shown = !made.is_null() && {
            let fields = made as *mut pyffi::SuperObject;
            let (this, class) = (ffi::PyObject_GetAttrString(made, c"__thisclass__".as_ptr()), ffi::PyObject_GetAttrString(made, c"__self_class__".as_ptr()));
            let same = this == int && (*fields).this as Obj == int && (*fields).obj == flag && !class.is_null() && (*fields).obj_type as Obj == class;
            xdecref(this);
            xdecref(class);
            same
        };
        ffi::PyErr_Clear();
        xdecref(made);
        shown
    })
}

unsafe fn new_ref_or_null(o: Obj) -> Obj {
    ffi::Py_XINCREF(o);
    o
}

/// Whether reading an attribute of what `super(class, obj)` gave runs no
/// Python code: the classes behind `class`, in the order of the class of
/// `obj`, are gone through for the name, and what the first that has it
/// holds is bound by C code (a function, a method of a class written in
/// C) or is no descriptor. False for a name none of them has.
unsafe fn quiet_super(bound: Obj, name: Obj) -> bool {
    if ffi::PyUnicode_CompareWithASCIIString(name, c"__class__".as_ptr()) == 0 {
        return false;
    }
    let (this, class) = if super_fields() {
        let fields = bound as *mut pyffi::SuperObject;
        (new_ref_or_null((*fields).this as Obj), new_ref_or_null((*fields).obj_type as Obj))
    } else {
        (ffi::PyObject_GetAttrString(bound, c"__thisclass__".as_ptr()), ffi::PyObject_GetAttrString(bound, c"__self_class__".as_ptr()))
    };
    // What a getter is asked of: the object, or nothing where `super` is
    // bound to the class itself.
    let held = if super_fields() {
        new_ref_or_null((*(bound as *mut pyffi::SuperObject)).obj)
    } else {
        ffi::PyObject_GetAttrString(bound, c"__self__".as_ptr())
    };
    ffi::PyErr_Clear();
    let of = if held == class { null_mut() } else { held };
    let mut quiet = false;
    if !this.is_null() && !class.is_null() && ffi::PyType_Check(class) != 0 {
        let order = (*(class as *mut ffi::PyTypeObject)).tp_mro;
        let count = if order.is_null() { 0 } else { ffi::PyTuple_GET_SIZE(order) };
        if let Some(start) = (0..count).find(|i| ffi::PyTuple_GET_ITEM(order, *i) == this) {
            for i in start + 1..count {
                let dict = pyffi::PyType_GetDict(ffi::PyTuple_GET_ITEM(order, i) as *mut ffi::PyTypeObject);
                let found = if dict.is_null() { Ok(null_mut()) } else { entry(dict, name) };
                xdecref(dict);
                match found {
                    Ok(found) if found.is_null() => {}
                    Ok(found) => {
                        quiet = !python_descriptor(found, of);
                        break;
                    }
                    Err(()) => break,
                }
            }
        }
    }
    ffi::PyErr_Clear();
    xdecref(held);
    xdecref(this);
    xdecref(class);
    quiet
}

/// Whether what a type has under a name is given out by code that is not
/// shown to read the object itself and nothing else. A property: its
/// getter is called with the object, and a getter that is a function of C
/// (`len`, `dir`) asks the object in turn. A descriptor of a class that is
/// defined in Python. A getter of C (`getset`) that is not among those
/// whose doing is known (`plain_getters`): that of a frame's `f_locals`
/// reads the variables of the frame. Any other descriptor of C that takes
/// stores too, but a member, which is a field of the object at a fixed
/// place. What is left is given out by C code that reads the object: no
/// descriptor at all (the value is what the class holds), a function or a
/// method of C, which is bound to the object, a member, a known getter.
/// `of` is the object the descriptor is asked of; null where it is read
/// off the class that has it, which hands out a getter and runs none.
unsafe fn python_descriptor(descr: Obj, of: Obj) -> bool {
    let ty = ffi::Py_TYPE(descr);
    if (*ty).tp_descr_get.is_none() {
        return false;
    }
    if ty == std::ptr::addr_of_mut!(ffi::PyProperty_Type) {
        return true;
    }
    if ty == std::ptr::addr_of_mut!(ffi::PyGetSetDescr_Type) {
        let fields = descr as *mut ffi::PyDescrObject;
        return !plain_getters((*fields).d_type, (*fields).d_name, of);
    }
    if is_static(ty) {
        return (*ty).tp_descr_set.is_some() && ty != std::ptr::addr_of_mut!(ffi::PyMemberDescr_Type);
    }
    // What `functools.lru_cache` made is bound to the object by C code.
    ffi::PyFunction_Check(descr) == 0 && !is_cache_wrapper(descr)
}

/// The getters of C that read fields of the object they are asked of, run
/// no code and lead to no frame's variables, by the class that defines
/// them: `object` and `type` (not the `__annotations__` of a class, which
/// may be computed, nor its `__doc__` where the class has a descriptor
/// under that name, which is asked); the numbers, texts and bytes; tuple,
/// list, dict, set, frozenset, slice and range; a function (from 3.14 on
/// not its `__annotations__`, which may be computed; the reads of its
/// `__code__`, `__defaults__` and `__kwdefaults__` tell the audit hooks,
/// tools that watch what runs, which are not asked for here: whether one
/// is installed cannot be asked from this side); a function of C
/// (not its `__qualname__`, which asks a class for an attribute); a
/// cell; the exception classes that are defined in C (what an exception
/// carries: its arguments, its traceback, the exceptions it came from);
/// of torch the class under every tensor, for the tensors `tensor_getter`
/// says, dtype, device, layout and memory format, and what `finfo` and
/// `iinfo` give (the limits of a dtype, as numbers). Of a code object the
/// names of its variables and its bytecode, which are made of its own
/// tables (`co_varnames`, `co_cellvars`, `co_freevars`, `co_code`; not
/// `co_lnotab`, which warns, and a warning is shown by Python code). A
/// class that is written in Python has two getters, which the
/// interpreter gives it: the dict of its object and the list of weak
/// references to it. No other class: not a frame (`f_locals` is the
/// variables of a function that is running), nor a traceback, a
/// generator or a coroutine, each of which leads to a frame; not a bound
/// method, whose `__doc__` is an attribute of what it is bound over; not
/// a module, whose one getter is `__annotations__`.
unsafe fn plain_getters(class: *mut ffi::PyTypeObject, name: Obj, of: Obj) -> bool {
    if class.is_null() {
        return false;
    }
    let named = |names: &[&std::ffi::CStr]| !name.is_null() && names.iter().any(|own| ffi::PyUnicode_CompareWithASCIIString(name, own.as_ptr()) == 0);
    if !is_static(class) {
        return named(&[c"__dict__", c"__weakref__"]);
    }
    if class == std::ptr::addr_of_mut!(ffi::PyCode_Type) {
        return named(&[c"co_varnames", c"co_cellvars", c"co_freevars", c"co_code"]);
    }
    if class == std::ptr::addr_of_mut!(ffi::PyType_Type) {
        if named(&[c"__doc__"]) && !of.is_null() && ffi::PyType_Check(of) != 0 {
            let dict = pyffi::PyType_GetDict(of as *mut ffi::PyTypeObject);
            let held = if dict.is_null() { Ok(null_mut()) } else { entry(dict, name) };
            xdecref(dict);
            return held.is_ok_and(|held| held.is_null() || (*ffi::Py_TYPE(held)).tp_descr_get.is_none());
        }
        return !named(&[c"__annotations__"]);
    }
    if class == std::ptr::addr_of_mut!(ffi::PyFunction_Type) {
        return !(cfg!(Py_3_14) && named(&[c"__annotations__"]));
    }
    if class == std::ptr::addr_of_mut!(ffi::PyCFunction_Type) {
        return !named(&[c"__qualname__"]);
    }
    let known = [
        std::ptr::addr_of!(ffi::PyBaseObject_Type),
        std::ptr::addr_of!(ffi::PyLong_Type),
        std::ptr::addr_of!(ffi::PyBool_Type),
        std::ptr::addr_of!(ffi::PyFloat_Type),
        std::ptr::addr_of!(ffi::PyComplex_Type),
        std::ptr::addr_of!(ffi::PyUnicode_Type),
        std::ptr::addr_of!(ffi::PyBytes_Type),
        std::ptr::addr_of!(ffi::PyTuple_Type),
        std::ptr::addr_of!(ffi::PyList_Type),
        std::ptr::addr_of!(ffi::PyDict_Type),
        std::ptr::addr_of!(ffi::PySet_Type),
        std::ptr::addr_of!(ffi::PyFrozenSet_Type),
        std::ptr::addr_of!(ffi::PySlice_Type),
        std::ptr::addr_of!(ffi::PyRange_Type),
        std::ptr::addr_of!(pyffi::PyCell_Type),
    ];
    if known.contains(&(class as *const ffi::PyTypeObject)) {
        return true;
    }
    if ffi::PyType_IsSubtype(class, ffi::PyExc_BaseException as *mut ffi::PyTypeObject) != 0 {
        return true;
    }
    let torch = torch_classes();
    if class as usize == torch[2] {
        return of.is_null() || tensor_getter(of, name);
    }
    torch[3..].contains(&(class as usize))
}

/// Whether a getter of the class under every tensor reads the tensor it
/// is asked of and nothing else. Each of them looks for an override
/// first (`__torch_function__`), which is Python code. None is looked
/// for where no torch function mode is on and the tensor has no Python
/// object or is of exactly `Tensor` or `Parameter`; none is found for a
/// class that says so (`overrides_nothing`). And a getter that warns is
/// no such read, since a warning is shown by Python code: `volatile`
/// always does, `grad` for a tensor that is no leaf, `T` the first time
/// it is asked of a tensor with other than 0 or 2 dimensions.
unsafe fn tensor_getter(of: Obj, name: Obj) -> bool {
    let Some(api) = capi::api() else {
        return false;
    };
    if (api.call_modes)() >> 16 != 0 {
        return false;
    }
    let boxed = !capi::is_unboxed(of);
    if boxed && !torch_classes()[..2].contains(&(ffi::Py_TYPE(of) as usize)) && !overrides_nothing(of) {
        return false;
    }
    // The name is read once, as bytes.
    let mut size: ffi::Py_ssize_t = 0;
    let text = ffi::PyUnicode_AsUTF8AndSize(name, &mut size);
    if text.is_null() {
        ffi::PyErr_Clear();
        return false;
    }
    let name = std::slice::from_raw_parts(text as *const u8, size as usize);
    let mut info = TcfgTensorInfo::empty();
    match name {
        b"volatile" => false,
        // A tensor that requires no gradient is a leaf; one without an
        // object that requires one was made by an operation.
        b"grad" | b"_grad" if boxed => is_leaf(of),
        b"grad" | b"_grad" => (api.tensor_info)(of, &mut info) != 0 && info.requires_grad == 0,
        b"T" => (api.tensor_info)(of, &mut info) != 0 && (info.ndim == 0 || info.ndim == 2),
        _ => true,
    }
}

/// What a getter of the class under every tensor gives for a tensor of
/// `Tensor` or `Parameter` (a new reference; null where it cannot be
/// asked): the getter is called as the lookup of `object` would call it,
/// and for those classes it reads the tensor. Found once the class is
/// there, in the cell of the name.
unsafe fn tensor_get(getter: &pyffi::Once<(usize, usize)>, name: &std::ffi::CStr, of: Obj) -> Obj {
    let found = match getter.get() {
        Some(found) => *found,
        None => {
            let base = torch_classes()[2] as *mut ffi::PyTypeObject;
            let name = ffi::PyUnicode_InternFromString(name.as_ptr());
            let descr = if base.is_null() || name.is_null() { null_mut() } else { pyffi::_PyType_Lookup(base, name) };
            ffi::PyErr_Clear();
            xdecref(name);
            if descr.is_null() || ffi::Py_TYPE(descr) != std::ptr::addr_of_mut!(ffi::PyGetSetDescr_Type) {
                return null_mut();
            }
            let fields = (*(descr as *mut ffi::PyGetSetDescrObject)).d_getset;
            let Some(get) = (*fields).get else {
                return null_mut();
            };
            let found = (get as usize, (*fields).closure as usize);
            getter.set(found);
            found
        }
    };
    if ffi::PyObject_TypeCheck(of, torch_classes()[2] as *mut ffi::PyTypeObject) == 0 {
        return null_mut();
    }
    let get: ffi::getter = std::mem::transmute(found.0);
    let said = get(of, found.1 as *mut std::ffi::c_void);
    ffi::PyErr_Clear();
    said
}

/// Whether a tensor of `Tensor` or `Parameter` is a leaf (`is_leaf`, a
/// flag).
pub unsafe fn is_leaf(of: Obj) -> bool {
    static GETTER: pyffi::Once<(usize, usize)> = pyffi::Once::new();
    let said = tensor_get(&GETTER, c"is_leaf", of);
    xdecref(said);
    said == ffi::Py_True()
}

/// What a tensor is a view of (`_base`: a new reference, to None where
/// it is no view; null where it cannot be asked).
pub unsafe fn base_of(of: Obj) -> Obj {
    static GETTER: pyffi::Once<(usize, usize)> = pyffi::Once::new();
    tensor_get(&GETTER, c"_base", of)
}

/// The node of autograd that made a tensor (`grad_fn`: a new reference,
/// to None where none did; null where it cannot be asked, as of what is
/// no tensor).
pub unsafe fn node_of(of: Obj) -> Obj {
    static GETTER: pyffi::Once<(usize, usize)> = pyffi::Once::new();
    tensor_get(&GETTER, c"grad_fn", of)
}

/// Whether a tensor of `Tensor` or `Parameter` is a view of another
/// (`_base` gives one).
pub unsafe fn is_view(of: Obj) -> bool {
    let base = base_of(of);
    xdecref(base);
    !base.is_null() && base != ffi::Py_None()
}

/// Whether torch finds no `__torch_function__` to call for an object of
/// a class made of the tensor class: the lookup its getters make
/// (`getattr(obj, "__torch_function__")`), followed without running it.
/// The class reads attributes as `object` does and has under the name
/// the function of `torch._C` that stands for "none"
/// (`_disabled_torch_function_impl`, as `Parameter` has it), which is no
/// descriptor, and the object has nothing under the name itself.
unsafe fn overrides_nothing(of: Obj) -> bool {
    let ty = ffi::Py_TYPE(of);
    let object = std::ptr::addr_of!(ffi::PyBaseObject_Type);
    if (*ty).tp_getattro.map(|own| own as usize) != (*object).tp_getattro.map(|own| own as usize) {
        return false;
    }
    let binding = pyffi::module(c"torch._C");
    let none = pyffi::in_module(binding, c"_disabled_torch_function_impl");
    if none.is_null() || ffi::PyCFunction_Check(none) == 0 || (*(none as *mut ffi::PyCFunctionObject)).m_self != binding {
        return false;
    }
    let name = ffi::PyUnicode_InternFromString(c"__torch_function__".as_ptr());
    if name.is_null() {
        ffi::PyErr_Clear();
        return false;
    }
    let slot = pyffi::_PyObject_GetDictPtr(of);
    let own = if slot.is_null() || (*slot).is_null() { Ok(null_mut()) } else { entry(*slot, name) };
    let nothing = pyffi::_PyType_Lookup(ty, name) == none && own.is_ok_and(|held| held.is_null());
    xdecref(name);
    nothing
}

/// The classes of torch that are asked for at an instruction: `Tensor`
/// and `Parameter` first, then the class under every tensor and those of
/// a dtype, a device, a layout, a memory format and the limits of a
/// dtype. Looked up once: they stay.
unsafe fn torch_classes() -> &'static [usize; 9] {
    static FOUND: pyffi::Once<[usize; 9]> = pyffi::Once::new();
    FOUND.get_or_init(|| {
        let named = |module: &std::ffi::CStr, name: &std::ffi::CStr| class_named(module, name) as usize;
        [
            named(c"torch", c"Tensor"),
            named(c"torch.nn.parameter", c"Parameter"),
            named(c"torch._C", c"TensorBase"),
            named(c"torch", c"dtype"),
            named(c"torch", c"device"),
            named(c"torch", c"layout"),
            named(c"torch", c"memory_format"),
            named(c"torch", c"finfo"),
            named(c"torch", c"iinfo"),
        ]
    })
}

/// Whether the classes of the tensors that `own_tensor` says read
/// `shape`, `ndim`, `size` and `dim` through the getters and the methods
/// that the class under every tensor defines in C: nobody has put
/// anything under one of the names on `Tensor`, on `Parameter` or on that
/// class, where the interpreter would find it and run it. (A tensor
/// without a Python object would get one of `Tensor`.) The four names
/// are looked up once per state of a class (`classes_hold`). And the
/// classes read their attributes as `object` does (`reads_are_plain`): a
/// `__getattribute__` of somebody's is called for every name.
pub unsafe fn sizes_are_own() -> bool {
    static SEEN: [AtomicU32; 2] = [AtomicU32::new(0), AtomicU32::new(0)];
    let base = torch_classes()[2] as *mut ffi::PyTypeObject;
    let defined = [std::ptr::addr_of_mut!(ffi::PyGetSetDescr_Type), std::ptr::addr_of_mut!(ffi::PyMethodDescr_Type)];
    let named = |class: *mut ffi::PyTypeObject| {
        [c"shape", c"ndim", c"size", c"dim"].iter().all(|name| {
            let name = ffi::PyUnicode_InternFromString(name.as_ptr());
            let found = if name.is_null() { null_mut() } else { pyffi::_PyType_Lookup(class, name) };
            xdecref(name);
            !found.is_null() && defined.contains(&ffi::Py_TYPE(found)) && (*(found as *mut ffi::PyDescrObject)).d_type == base
        })
    };
    reads_are_plain() && !base.is_null() && classes_hold(&SEEN, named)
}

/// Whether `Tensor` and `Parameter` read their attributes as `object`
/// does: nobody has put a `__getattribute__` on one of them, which the
/// interpreter would call at every read of an attribute of a tensor,
/// whatever the name. Asked wherever something stands for such a read: a
/// read of a size, a read that a path keeps, a call site whose
/// instruction takes a method or a getter off a tensor.
pub unsafe fn reads_are_plain() -> bool {
    static SEEN: [AtomicU32; 2] = [AtomicU32::new(0), AtomicU32::new(0)];
    let reads = |class: *mut ffi::PyTypeObject| pyffi::_PyType_Lookup(class, specials().getattribute.as_ptr());
    let plain = reads(std::ptr::addr_of_mut!(ffi::PyBaseObject_Type));
    classes_hold(&SEEN, |class| reads(class) == plain)
}

/// Whether something holds of `Tensor` and of `Parameter` that can change
/// only where one of them, or a class under it, is assigned to: asked
/// once per state of a class. Its version tag is a number that no other
/// state of this or any other class has, and the interpreter takes it
/// away whenever the class or a class under it is assigned to. `seen` has
/// the tags it held for last.
unsafe fn classes_hold(seen: &[AtomicU32; 2], holds: impl Fn(*mut ffi::PyTypeObject) -> bool) -> bool {
    let classes = torch_classes();
    let tag = |which: usize| {
        let class = classes[which] as *mut ffi::PyTypeObject;
        if class.is_null() { 0 } else { (*class).tp_version_tag }
    };
    if (0..2).all(|which| tag(which) != 0 && seen[which].load(Ordering::Relaxed) == tag(which)) {
        return true;
    }
    let held = classes[..2].iter().all(|class| {
        let class = *class as *mut ffi::PyTypeObject;
        !class.is_null() && holds(class)
    });
    ffi::PyErr_Clear();
    // The lookups gave each class its tag, if it had none.
    if held && tag(0) != 0 && tag(1) != 0 {
        seen[0].store(tag(0), Ordering::Relaxed);
        seen[1].store(tag(1), Ordering::Relaxed);
    }
    held
}

/// A tensor of one of the two classes that the kernels' own code treats
/// as a tensor and nothing else (`Tensor`, `Parameter`), or one without a
/// Python object: asked whether it is true, or indexed by plain values or
/// by another such tensor, it runs kernels and no Python code of its
/// class.
pub unsafe fn own_tensor(o: Obj) -> bool {
    !o.is_null() && (capi::is_unboxed(o) || torch_classes()[..2].contains(&(ffi::Py_TYPE(o) as usize)))
}

/// What `getattr(obj, name)` gives (a new reference), where the lookup
/// runs no Python code and comes down to finding the object in a dict: in
/// that of a module; in that of an object whose attributes are looked up
/// as `object` does it and whose class has nothing under the name that
/// takes the lookup (no descriptor that also takes stores); in one of the
/// dicts that a `__getattr__` with a shortcut looks into, where the class
/// has nothing under the name at all. None for every other lookup, quiet
/// or not: `quiet_attr` says which, and the lookup is made as it is.
pub unsafe fn read_attr(obj: Obj, name: Obj) -> Option<Obj> {
    if obj.is_null() || capi::is_unboxed(obj) {
        return None;
    }
    let found = |held: Obj| (!held.is_null()).then(|| new_ref(held));
    let ty = ffi::Py_TYPE(obj);
    if ffi::PyModule_CheckExact(obj) != 0 {
        let dict = ffi::PyModule_GetDict(obj);
        return if dict.is_null() { None } else { found(entry(dict, name).ok()?) };
    }
    if ty == std::ptr::addr_of_mut!(ffi::PySuper_Type) || ffi::PyType_Check(obj) != 0 {
        return None;
    }
    let object = std::ptr::addr_of!(ffi::PyBaseObject_Type);
    let generic = (*ty).tp_getattro.map(|own| own as usize) == (*object).tp_getattro.map(|own| own as usize);
    if !generic && !plain_lookup(obj) {
        return None;
    }
    let descr = pyffi::_PyType_Lookup(ty, name);
    if !descr.is_null() && (*ffi::Py_TYPE(descr)).tp_descr_set.is_some() {
        return None;
    }
    let slot = pyffi::_PyObject_GetDictPtr(obj);
    if !slot.is_null() && !(*slot).is_null() {
        if let Some(held) = found(entry(*slot, name).ok()?) {
            return Some(held);
        }
    }
    if !descr.is_null() || generic || !crate::shortcuts::has_getattr(ty) {
        return None;
    }
    found(crate::shortcuts::among(obj, name).ok()?)
}

/// `read_attr`, with what was found put into `found`.
pub unsafe fn holds_attr(obj: Obj, name: Obj, found: &mut Obj) -> bool {
    match read_attr(obj, name) {
        Some(held) => {
            *found = held;
            true
        }
        None => false,
    }
}

/// Whether `getattr(obj, name)` runs no Python code, as far as that can be
/// told without running it: what the lookup of the object's type would do
/// is followed up to where it finds the attribute. False where it would
/// call a property, a descriptor or a `__getattr__` written in Python,
/// where the type has a lookup of its own that is not known here, and where
/// the attribute is not there.
pub unsafe fn quiet_attr(obj: Obj, name: Obj, _api: &TcfgApi) -> bool {
    if capi::is_unboxed(obj) {
        // A tensor without a Python object: of the class of tensors, with
        // nothing of its own, and read as that class reads.
        let class = torch_classes()[0] as Obj;
        let descr = if class.is_null() { null_mut() } else { pyffi::_PyType_Lookup(class as *mut ffi::PyTypeObject, name) };
        return !descr.is_null() && !python_descriptor(descr, obj) && reads_are_plain();
    }
    let ty = ffi::Py_TYPE(obj);
    if ffi::PyModule_CheckExact(obj) != 0 {
        let dict = ffi::PyModule_GetDict(obj);
        return !dict.is_null() && entry(dict, name).is_ok_and(|held| !held.is_null());
    }
    if ty == std::ptr::addr_of_mut!(ffi::PySuper_Type) {
        return quiet_super(obj, name);
    }
    if ffi::PyType_Check(obj) != 0 {
        // A class whose class reads attributes as `type` does (`type`
        // itself, or a class made of it without a `__getattribute__` or
        // a `__getattr__` of its own: an enum class, an abstract one).
        // As `type` reads: what the class of the class has under the
        // name if that takes stores too; else what the class has
        // itself; else what the class of the class has. Each is given
        // out by C code, or the read is none of a path.
        let reads = (*ty).tp_getattro.map(|own| own as usize) == (*std::ptr::addr_of!(ffi::PyType_Type)).tp_getattro.map(|own| own as usize);
        if !reads {
            return false;
        }
        let meta = pyffi::_PyType_Lookup(ty, name);
        if !meta.is_null() && (*ffi::Py_TYPE(meta)).tp_descr_set.is_some() {
            return !python_descriptor(meta, obj);
        }
        let own = pyffi::_PyType_Lookup(obj as *mut ffi::PyTypeObject, name);
        return if own.is_null() { !meta.is_null() && !python_descriptor(meta, obj) } else { !python_descriptor(own, null_mut()) };
    }
    // The lookup of `object`, or that and then a `__getattr__` for what it
    // does not find: of the latter, only one that has a shortcut is known.
    // A tensor is read like any other object: a `__getattribute__` that
    // somebody put on its class is called for every name.
    let object = std::ptr::addr_of!(ffi::PyBaseObject_Type);
    let generic = (*ty).tp_getattro.map(|own| own as usize) == (*object).tp_getattro.map(|own| own as usize);
    let hooked = !generic && plain_lookup(obj);
    let shortcut = hooked && crate::shortcuts::has_getattr(ty);
    if !(generic || hooked) {
        return false;
    }
    let descr = pyffi::_PyType_Lookup(ty, name);
    let python = !descr.is_null() && python_descriptor(descr, obj);
    let data = !descr.is_null() && (*ffi::Py_TYPE(descr)).tp_descr_set.is_some();
    if data {
        return !python;
    }
    let slot = pyffi::_PyObject_GetDictPtr(obj);
    let Ok(own) = (if slot.is_null() || (*slot).is_null() { Ok(null_mut()) } else { entry(*slot, name) }) else {
        return false;
    };
    if !own.is_null() {
        return true;
    }
    if !descr.is_null() {
        return !python;
    }
    shortcut && crate::shortcuts::among(obj, name).is_ok_and(|held| !held.is_null())
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
    static FUNCTIONS: pyffi::Once<(Vec<Option<PyObj>>, Vec<PyObj>)> = pyffi::Once::new();
    let (functions, devices) = FUNCTIONS.get_or_init(|| {
        let binding = pyffi::module(c"torch._C");
        let functions = ASKED
            .iter()
            .map(|name| {
                let found = pyffi::in_module(binding, name);
                (!found.is_null()).then(|| PyObj::from_borrowed(found))
            })
            .collect();
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

/// Whether the exception that is set came between the instructions of a
/// function and was raised by none of them: one that is no `Exception`
/// (a `KeyboardInterrupt`, which the interpreter's own handler of an
/// interrupt raises from C; a `SystemExit`), or one whose traceback goes
/// through the handler of a signal (`through_handler`). A handler runs
/// wherever the interpreter or the VM looks for signals, behind whichever
/// instruction was running: what it raises says nothing about that
/// instruction. The exception stays set.
pub unsafe fn interrupted() -> bool {
    let raised = ffi::PyErr_GetRaisedException();
    if raised.is_null() {
        return false;
    }
    let came = came_between(raised);
    ffi::PyErr_SetRaisedException(raised);
    came
}

/// `interrupted`, of an exception that was caught.
pub unsafe fn came_between(raised: Obj) -> bool {
    ffi::PyErr_GivenExceptionMatches(raised, ffi::PyExc_Exception) == 0 || through_handler(raised)
}

/// Whether the traceback of an exception goes through the handler of a
/// signal. Two marks, either of which says so. The frame is that of a
/// function that is installed as the handler of a signal now: a
/// function, a method of one, or an object whose class has a function
/// for calls (asked of `_signal.getsignal`, the function of C that reads
/// the table). Or the frame was called as the interpreter calls a
/// handler, `handler(number, frame)` (`called_as_handler`): with the
/// number of a signal and the frame that was running, which is the frame
/// of the entry above it in the traceback, or for the outermost entry
/// the frame that is running now; that mark sees a handler that takes
/// more in front (a `functools.partial`, a method) and one that is the
/// signal's no longer. Neither sees a handler of C, which has no frame.
unsafe fn through_handler(raised: Obj) -> bool {
    let module = pyffi::module(c"_signal");
    let asks = pyffi::in_module(module, c"getsignal");
    if asks.is_null() || ffi::PyCFunction_Check(asks) == 0 || (*(asks as *mut ffi::PyCFunctionObject)).m_self != module {
        return false;
    }
    let count = pyffi::in_module(module, c"NSIG");
    let count = if count.is_null() { 65 } else { ffi::PyLong_AsLong(count).clamp(0, 1024) };
    let mut handlers: Vec<PyObj> = Vec::new();
    for signal in 1..count {
        let number = ffi::PyLong_FromLong(signal);
        let handler = ffi::PyObject_CallOneArg(asks, number);
        xdecref(number);
        if !handler.is_null() {
            handlers.push(PyObj::from_owned(handler));
        }
    }
    ffi::PyErr_Clear();
    let code_of = |handler: Obj| -> Obj {
        let method = ffi::Py_TYPE(handler) == std::ptr::addr_of_mut!(pyffi::PyMethod_Type);
        let function = if method { pyffi::PyMethod_Function(handler) } else { handler };
        if ffi::PyFunction_Check(function) != 0 {
            return ffi::PyFunction_GetCode(function);
        }
        let name = ffi::PyUnicode_InternFromString(c"__call__".as_ptr());
        let call = if name.is_null() { null_mut() } else { pyffi::_PyType_Lookup(ffi::Py_TYPE(handler), name) };
        xdecref(name);
        if !call.is_null() && ffi::PyFunction_Check(call) != 0 { ffi::PyFunction_GetCode(call) } else { null_mut() }
    };
    let codes: Vec<Obj> = handlers.iter().map(|handler| code_of(handler.as_ptr())).filter(|code| !code.is_null()).collect();
    ffi::PyErr_Clear();
    let first = ffi::PyException_GetTraceback(raised);
    let mut at = first as *mut ffi::PyTracebackObject;
    let mut above: *mut ffi::PyFrameObject = null_mut();
    let mut found = false;
    while !at.is_null() && !found {
        let frame = (*at).tb_frame;
        if !frame.is_null() {
            let code = ffi::PyFrame_GetCode(frame) as Obj;
            found = codes.contains(&code) || called_as_handler(frame, code, above, count);
            xdecref(code);
        }
        above = frame;
        at = (*at).tb_next;
    }
    xdecref(first);
    found
}

/// Whether a frame was called as the interpreter calls the handler of a
/// signal: among what it was called with (its arguments, and what a
/// `*args` holds) are a whole number that names a signal and the frame
/// that was running when the interpreter looked for signals: `above`,
/// the frame of the traceback entry above this one, or where this is the
/// outermost entry (null) the frame that is running now, which the
/// exception has not gone back through. Reads the arguments as the frame
/// has them: one that the function has bound to something else or
/// deleted is not seen.
unsafe fn called_as_handler(frame: *mut ffi::PyFrameObject, code: Obj, above: *mut ffi::PyFrameObject, signals: std::ffi::c_long) -> bool {
    let number = |name: &std::ffi::CStr| {
        let held = ffi::PyObject_GetAttrString(code, name.as_ptr());
        let count = if held.is_null() { 0 } else { ffi::PyLong_AsLong(held) };
        xdecref(held);
        count.max(0) as isize
    };
    // CO_VARARGS and CO_VARKEYWORDS.
    let flags = number(c"co_flags");
    let taken = number(c"co_argcount") + number(c"co_kwonlyargcount") + ((flags >> 2) & 1) + ((flags >> 3) & 1);
    let names = ffi::PyObject_GetAttrString(code, c"co_varnames".as_ptr());
    let (mut named, mut running): (bool, Vec<Obj>) = (false, Vec::new());
    if !names.is_null() && ffi::PyTuple_CheckExact(names) != 0 {
        let mut look = |value: Obj| {
            if ffi::PyLong_CheckExact(value) != 0 {
                let signal = ffi::PyLong_AsLong(value);
                named |= signal >= 1 && signal < signals;
            } else if ffi::Py_TYPE(value) == std::ptr::addr_of_mut!(ffi::PyFrame_Type) {
                running.push(value);
            }
        };
        for i in 0..taken.min(ffi::PyTuple_GET_SIZE(names)) {
            let value = ffi::PyFrame_GetVar(frame, ffi::PyTuple_GET_ITEM(names, i));
            if value.is_null() {
                continue;
            }
            if ffi::PyTuple_CheckExact(value) != 0 {
                (0..ffi::PyTuple_GET_SIZE(value)).for_each(|k| look(ffi::PyTuple_GET_ITEM(value, k)));
            } else {
                look(value);
            }
            // The frame holds it.
            ffi::Py_DECREF(value);
        }
    }
    xdecref(names);
    ffi::PyErr_Clear();
    if !named || running.is_empty() {
        return false;
    }
    // Asked only of a frame that was handed a signal's number and a frame:
    // the frame that is running gets its object here, if it has none.
    let above = if above.is_null() { ffi::PyEval_GetFrame() } else { above };
    !above.is_null() && running.contains(&(above as Obj))
}

/// What stands between the operations of the thread and their kernels and
/// has to see each one as the function wrote it, if anything does: a
/// dispatch mode, or a transform of functorch (a torch function mode shows
/// in `call_modes`, and no graph is entered under one). A run under one is
/// never exported: what was made of a graph would hide its operations, or
/// show others.
pub unsafe fn interposed() -> Option<&'static str> {
    static ASKED: pyffi::Once<[Option<PyObj>; 2]> = pyffi::Once::new();
    let asked = ASKED.get_or_init(|| {
        let found = |module: &std::ffi::CStr, name: &std::ffi::CStr| {
            let function = pyffi::in_module(pyffi::module(module), name);
            (!function.is_null()).then(|| PyObj::from_borrowed(function))
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
    static KNOWN: pyffi::Once<Shared<HashMap<String, bool>>> = pyffi::Once::new();
    let known = KNOWN.get_or_init(|| Shared::new(HashMap::new()));
    if let Some(writes) = known.borrow().get(name) {
        return *writes;
    }
    let (qualified, overload) = overload_of(name);
    let asked = |py: Python<'_>| -> PyResult<bool> {
        let schema = pyffi::of_module(py, c"torch._C", c"_get_schema")?.call1((qualified, overload))?;
        let mut first = true;
        for argument in schema.getattr("arguments")?.try_iter()? {
            let alias = argument?.getattr("alias_info")?;
            let written = !alias.is_none() && alias.getattr("is_write")?.is_truthy()?;
            if written != first {
                return Ok(false);
            }
            first = false;
        }
        let found = pyffi::of_module(py, c"torch._C", c"_get_operation_overload")?.call1((qualified, overload))?;
        if first || found.is_none() {
            return Ok(false);
        }
        let seeded = pyffi::of_module(py, c"torch", c"Tag")?.getattr("nondeterministic_seeded")?;
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
    static KNOWN: pyffi::Once<Shared<HashMap<String, bool>>> = pyffi::Once::new();
    let known = KNOWN.get_or_init(|| Shared::new(HashMap::new()));
    if let Some(shaped) = known.borrow().get(name) {
        return *shaped;
    }
    let (qualified, overload) = overload_of(name);
    let asked = |py: Python<'_>| -> PyResult<bool> {
        let found = pyffi::of_module(py, c"torch._C", c"_get_operation_overload")?.call1((qualified, overload))?;
        if found.is_none() {
            return Ok(false);
        }
        let tags = pyffi::of_module(py, c"torch", c"Tag")?;
        let (dynamic, data) = (tags.getattr("dynamic_output_shape")?, tags.getattr("data_dependent_output")?);
        Ok(!found.get_item(2)?.contains(dynamic)? && !found.get_item(2)?.contains(data)?)
    };
    let shaped = unsafe { asked(Python::assume_attached()).unwrap_or(false) };
    unsafe { ffi::PyErr_Clear() };
    known.borrow_mut().insert(name.to_string(), shaped);
    shaped
}

/// The operators that drop elements at random where they are told to, and
/// give back what they were handed where they are not (their last
/// argument).
pub const DROPOUTS: [&str; 4] = ["aten::dropout", "aten::feature_dropout", "aten::alpha_dropout", "aten::feature_alpha_dropout"];

/// The operators that draw random numbers only where an argument is not
/// zero, which is what it is where it is not given: the argument, by
/// place and by name.
pub const DRAWS_AT_A_RATE: [(&str, usize, &std::ffi::CStr); 1] = [("aten::scaled_dot_product_attention", 4, c"dropout_p")];

/// Whether all that keeps an operator from being a function of its
/// arguments is that it draws random numbers: it writes to none of them,
/// and its tags say that it draws from the generator.
pub fn seeded_operator(name: &str) -> bool {
    static KNOWN: pyffi::Once<Shared<HashMap<String, bool>>> = pyffi::Once::new();
    let known = KNOWN.get_or_init(|| Shared::new(HashMap::new()));
    if let Some(seeded) = known.borrow().get(name) {
        return *seeded;
    }
    let (qualified, overload) = overload_of(name);
    let asked = |py: Python<'_>| -> PyResult<bool> {
        let schema = pyffi::of_module(py, c"torch._C", c"_get_schema")?.call1((qualified, overload))?;
        if schema.getattr("is_mutable")?.is_truthy()? {
            return Ok(false);
        }
        let found = pyffi::of_module(py, c"torch._C", c"_get_operation_overload")?.call1((qualified, overload))?;
        if found.is_none() {
            return Ok(false);
        }
        let seeded = pyffi::of_module(py, c"torch", c"Tag")?.getattr("nondeterministic_seeded")?;
        found.get_item(2)?.contains(seeded)
    };
    let quiet = !UPDATES_STATISTICS.iter().any(|(op, _)| *op == qualified);
    let seeded = quiet && unsafe { asked(Python::assume_attached()).unwrap_or(false) };
    unsafe { ffi::PyErr_Clear() };
    known.borrow_mut().insert(name.to_string(), seeded);
    seeded
}

/// Whether an operator has no kernel of its own and is made of other
/// operators (`aten::conv2d`; `aten::argwhere`, which is `aten::nonzero`):
/// its tags do not say what those do. What cannot be asked is taken to be
/// one.
pub fn composite_operator(name: &str) -> bool {
    static KNOWN: pyffi::Once<Shared<HashMap<String, bool>>> = pyffi::Once::new();
    let known = KNOWN.get_or_init(|| Shared::new(HashMap::new()));
    if let Some(composite) = known.borrow().get(name) {
        return *composite;
    }
    let asked = |py: Python<'_>| -> PyResult<bool> {
        let has = pyffi::of_module(py, c"torch._C", c"_dispatch_has_kernel_for_dispatch_key")?;
        // The key itself: named by a text, torch looks for `builtins` by
        // an import at every call.
        let key = pyffi::of_module(py, c"torch._C", c"DispatchKey")?.getattr("CompositeImplicitAutograd")?;
        has.call1((name, key))?.is_truthy()
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
    /// It cannot where its tensors are on one device.
    OneDevice,
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

/// Operators that the schema or a tag would have unable to fail, and that
/// fail for what no form checks: how much memory a tensor stands in
/// (`aten::as_strided`), and whether its storage can grow or be
/// exchanged (`aten::resize_`, `aten::resize_as_`, `aten::set_`). By the
/// name without the overload and the `_`.
const MEMORY_BOUND: [&str; 4] = ["as_strided", "resize", "resize_as", "set"];

/// The overloads that read a number off a tensor they are handed, by their
/// full names: the number may not fit the dtype it is put into (a value to
/// fill with, the ends of a range), or be no place of the tensor it counts
/// into (where a view starts, where a tensor is split). The overloads that
/// take the number itself are handed one that the form checks.
const READS_A_NUMBER: [&str; 7] = [
    "aten::masked_fill.Tensor",
    "aten::masked_fill_.Tensor",
    "aten::linspace.Tensor_Tensor",
    "aten::linspace.Tensor_Scalar",
    "aten::linspace.Scalar_Tensor",
    "aten::narrow.Tensor",
    "aten::tensor_split.tensor_indices_or_sections",
];

/// The overloads that read such a number only off a tensor on another
/// device than the one they write to, and copy what is on the same one
/// (a copy converts without a check).
const READS_ACROSS_DEVICES: [&str; 2] = ["aten::fill.Tensor", "aten::fill_.Tensor"];

/// See `Total`. A view gives its argument seen another way and reads
/// nothing of it (its schema says that it is one, or its tag that it
/// makes its argument one); the operators that work element by element,
/// and those that reduce, say so in their tags;
/// the rest is the list above. In front of all of them stand the
/// operators that can fail all the same (`MEMORY_BOUND`,
/// `READS_A_NUMBER`, `READS_ACROSS_DEVICES`).
pub fn total_operator(name: &str) -> Total {
    static KNOWN: pyffi::Once<Shared<HashMap<String, Total>>> = pyffi::Once::new();
    let known = KNOWN.get_or_init(|| Shared::new(HashMap::new()));
    if let Some(total) = known.borrow().get(name) {
        return *total;
    }
    let (qualified, overload) = overload_of(name);
    let base = qualified.rsplit("::").next().unwrap_or(qualified);
    let base = base.strip_suffix('_').filter(|rest| !rest.is_empty()).unwrap_or(base);
    let asked = |py: Python<'_>| -> PyResult<Total> {
        if MEMORY_BOUND.contains(&base) || READS_A_NUMBER.contains(&name) {
            return Ok(Total::Never);
        }
        if READS_ACROSS_DEVICES.contains(&name) {
            return Ok(Total::OneDevice);
        }
        if TOTAL.contains(&base) {
            return Ok(Total::Always);
        }
        let schema = pyffi::of_module(py, c"torch._C", c"_get_schema")?.call1((qualified, overload))?;
        for given in schema.getattr("returns")?.try_iter()? {
            let alias = given?.getattr("alias_info")?;
            if !alias.is_none() && !alias.getattr("is_write")?.is_truthy()? {
                return Ok(Total::Always);
            }
        }
        let found = pyffi::of_module(py, c"torch._C", c"_get_operation_overload")?.call1((qualified, overload))?;
        if found.is_none() {
            return Ok(Total::Never);
        }
        let (tags, tag) = (found.get_item(2)?, pyffi::of_module(py, c"torch", c"Tag")?);
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
    static KNOWN: pyffi::Once<Shared<HashMap<String, bool>>> = pyffi::Once::new();
    let known = KNOWN.get_or_init(|| Shared::new(HashMap::new()));
    if let Some(pure) = known.borrow().get(name) {
        return *pure;
    }
    let (qualified, overload) = overload_of(name);
    let asked = |py: Python<'_>| -> PyResult<bool> {
        let schema = pyffi::of_module(py, c"torch._C", c"_get_schema")?.call1((qualified, overload))?;
        if schema.getattr("is_mutable")?.is_truthy()? {
            return Ok(false);
        }
        let found = pyffi::of_module(py, c"torch._C", c"_get_operation_overload")?.call1((qualified, overload))?;
        if found.is_none() {
            return Ok(false);
        }
        let seeded = pyffi::of_module(py, c"torch", c"Tag")?.getattr("nondeterministic_seeded")?;
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
    ///   held: per output, whether something can hold it when the path has
    ///     landed (`Path::held`);
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
        let held = list_of(self.held.iter().map(|held| new_ref(if *held { ffi::Py_True() } else { ffi::Py_False() })).collect());
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
        let entries: [(&std::ffi::CStr, Obj); 8] = [
            (c"where", label),
            (c"inputs", inputs.as_ptr()),
            (c"input_ids", ids.as_ptr()),
            (c"input_names", names.as_ptr()),
            (c"nodes", nodes.as_ptr()),
            (c"outputs", outputs),
            (c"held", held),
            (c"exits", exits),
        ];
        for (key, value) in entries {
            ffi::PyDict_SetItemString(description.as_ptr(), key.as_ptr(), value);
        }
        for made in [label, outputs, held, exits] {
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
                    let names = |kwnames: u32| match &inst.named {
                        Some(named) => new_ref(named.as_ptr()),
                        None if kwnames == NONE => none(),
                        None => new_ref(func.consts[kwnames as usize].as_ptr()),
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

/// Whether `repr` of a value is made by C code alone: what a check
/// compares by value, a text of bytes, a function, a function of C code, a
/// code object, and a tuple or a frozenset of such (what the constants of a
/// code object are).
unsafe fn quiet_repr(o: Obj) -> bool {
    if checkable(o) || ffi::PyBytes_CheckExact(o) != 0 || ffi::PyFunction_Check(o) != 0 || ffi::PyCFunction_CheckExact(o) != 0 {
        return true;
    }
    if ffi::PyCode_Check(o) != 0 {
        // Its text has `repr` of the file's name in it.
        let file = ffi::PyObject_GetAttrString(o, c"co_filename".as_ptr());
        let quiet = !file.is_null() && ffi::PyUnicode_CheckExact(file) != 0;
        xdecref(file);
        ffi::PyErr_Clear();
        return quiet;
    }
    if ffi::PyTuple_CheckExact(o) != 0 {
        return !holds_itself(o) && (0..ffi::PyTuple_GET_SIZE(o)).all(|i| quiet_repr(ffi::PyTuple_GET_ITEM(o, i)));
    }
    if ffi::PyFrozenSet_CheckExact(o) != 0 {
        let items = ffi::PyObject_GetIter(o);
        let mut quiet = !items.is_null();
        while quiet {
            let item = ffi::PyIter_Next(items);
            if item.is_null() {
                break;
            }
            quiet = quiet_repr(item);
            ffi::Py_DECREF(item);
        }
        xdecref(items);
        ffi::PyErr_Clear();
        return quiet;
    }
    false
}

/// A value in a report (`dump()`): `repr` of it where C code alone makes
/// that (`quiet_repr`), and otherwise what a check calls it (`named`):
/// `repr` of anything else can be somebody's code, which a report never
/// runs.
pub unsafe fn shown(o: Obj) -> String {
    if o.is_null() {
        "<null>".to_string()
    } else if quiet_repr(o) {
        pyffi::repr(o)
    } else {
        named(o)
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
    /// There is no pass over graphs, and the engine derives the backward of
    /// a graph whose tensors need a gradient: such a graph is exported to
    /// the engine itself (`backward::Staged`), and no other.
    pub derives: bool,
}

impl Default for Graphs {
    fn default() -> Graphs {
        Graphs { min_ops: 4, max_variants: 8, export: None, derives: false }
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
    /// A list, a dict, an iterator or an object of a class that an
    /// instruction of the path made: the path may change it.
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
    /// What an effect of the path gives (the token of a `set` of a context
    /// variable): it is made where the path lands, behind everything else
    /// the path runs. Only another effect takes it, a test for None (which
    /// it is not), a variable, and what the function gives back.
    pub lands: bool,
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
        // And one behind each instruction that a path of the function
        // ended in front of, where no graph starts there anyway.
        let wanted = cx.cf.state.borrow().blocks[unit.block].wanted.clone();
        for origin in wanted {
            let Some(at) = origins.iter().position(|other| *other == origin) else {
                continue;
            };
            if !seeds.iter().any(|(first, ..)| *first == at) {
                let place = (unit.block, origin as usize);
                seeds.push((at, Path::empty(place, place), true));
            }
        }
        seeds.sort_by_key(|(first, ..)| *first);
        for (first, seed, entry) in seeds {
            let root = seed.root.1 as u32;
            // The seed that makes way ends with nothing borrowed.
            let mut gone = None;
            let site = {
                let mut state = cx.cf.state.borrow_mut();
                let known = &mut state.blocks[unit.block].graphs;
                match known.iter().find(|(at, _)| *at == root) {
                    Some((_, site)) => {
                        gone = Some(std::mem::replace(&mut *site.seed.borrow_mut(), Arc::new(seed)));
                        site.clone()
                    }
                    None => {
                        let name =
                            format!("{} (block {})[{}]", cx.func().qualname, unit.block, seed.root.1);
                        let site = Arc::new(GraphSite::new(name, seed, entry, self));
                        known.push((root, site.clone()));
                        site
                    }
                }
            };
            drop(gone);
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
    static TORCH: pyffi::Once<String> = pyffi::Once::new();
    let torch = TORCH.get_or_init(|| {
        let asked = |py: Python<'_>| -> PyResult<String> { pyffi::of_module(py, c"torch", c"__file__")?.extract() };
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
    // A dropout that was told not to drop anything whenever it was looked
    // at (a model that is not training) draws nothing.
    let idle = |d: &DispatchOp| {
        let dropout = unsafe { d.slot.operator() }.is_some_and(|name| DROPOUTS.contains(&name.as_str()));
        let off = d.operands.last().is_some_and(|train| {
            let never = matches!(cx.obs.get(*train as usize), Some(crate::spec::ValueObs::Bool { seen_true: 0, .. }));
            let no = matches!(unit.def_op(*train), Some(Op::Const(c)) if cx.func().consts[*c as usize].as_ptr() == unsafe { ffi::Py_False() });
            never || no
        });
        dropout && off
    };
    let kind = match &unit.insts[i].inst.op {
        Op::Dispatch(d) if d.slot.traits() == 0 && d.slot.len() > 0 && !wrote => Some(Kind::Node),
        Op::Dispatch(d) if d.slot.traits() == capi::SITE_RANDOM && d.slot.len() > 0 && idle(d) => Some(Kind::Node),
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
            | Op::MakeCell(_)
            | Op::CellSet(..)
            | Op::TupleGet(..)
            | Op::IsNone(..)
            | Op::Is(..)
            | Op::TensorSize { .. }
            | Op::TensorDim(_)
    )
}

/// Whether an instruction of a path that gives what nothing uses can be
/// left out of a run: it cannot fail whatever it is given, or it is a part
/// of how a call binds its arguments that the checks of the call stand for
/// (`Assert::Enters`, `Assert::Sequence`, `Assert::Keys`). Not a default
/// by its name or a variable of an outer function, which may not be there.
fn idle(op: &Op) -> bool {
    match op {
        Op::Part(Part::Cell(_) | Part::KwDefault(_), _) => false,
        Op::Part(..) => true,
        op => infallible(op),
    }
}

/// Whether a change of something the path made can be left out of a run
/// together with the thing: like every instruction that is left out
/// (`idle`), only one that cannot fail and runs no Python code whatever it
/// is given. An append, a store into an object of the path and a variable
/// that is bound are such. An entry that is put into a dict is one only
/// under a key that is the same at every call (a constant): any other
/// key may be one that cannot be hashed at another call, or that is
/// hashed in Python. A `pop` is none (the key may not be there), nor is
/// what goes over a value from outside the path to extend a list or to
/// fill a dict. `own` says whether the path built or made a value.
fn leaves_quietly(op: &Op, defs: &[Def], own: impl Fn(PathValue) -> bool) -> bool {
    match op {
        Op::MapAdd(_, key, _) => defs[*key as usize].fixed,
        Op::ListExtend(_, more) | Op::DictMerge(_, more, _) => own(*more),
        Op::Call { .. } | Op::CallMethod { .. } => false,
        _ => true,
    }
}

/// A list, a tuple or a slice that the instructions make.
pub fn container(op: &Op) -> bool {
    matches!(op, Op::BuildTuple(_) | Op::BuildList(_) | Op::BuildSlice(..))
}

/// Why an instruction is in no graph.
fn why(unit: &Unit, cx: &PassCx, i: usize) -> String {
    let u = &unit.insts[i];
    let reason = cx.reasons.get(u.origin as usize).and_then(|r| r.as_ref());
    why_not(cx.func(), &u.inst.op, facts(cx, unit, i), reason, false)
}

/// Why an instruction is no part of a graph, from what the profiling tier
/// saw it do (`found`) and why it is no call site (`reason`). `behind`:
/// the path it would be on has taken down something that can fail behind
/// what stands for the graph, which is why an operation that draws random
/// numbers is none of it.
pub fn why_not(func: &Function, op: &Op, found: Option<&InstFacts>, reason: Option<&String>, behind: bool) -> String {
    let no_site = || match reason {
        Some(reason) => format!("runs kernels and is no call site: {}", reason),
        None => "runs kernels and is no call site".to_string(),
    };
    let kernels = found.is_some_and(|f| matches!(f.kernels, Kernels::One | Kernels::Many));
    let behind = if behind { " behind what can fail" } else { "" };
    // The operators of a call that is no call site, where all that keeps
    // it from being an operation is that it draws.
    let draws = found.filter(|_| kernels).and_then(|f| {
        let drawn: Vec<&str> = f.operators.iter().filter(|name| !pure_operator(name)).map(|name| name.as_str()).collect();
        (!drawn.is_empty() && drawn.iter().all(|name| seeded_operator(name))).then(|| drawn.join("+"))
    });
    match op {
        Op::Dispatch(d) => {
            let operator = unsafe { d.slot.operator() }.unwrap_or_else(|| "the kernel".to_string());
            let traits = d.slot.traits();
            if traits & SITE_WRITES != 0 || found.is_some_and(|f| f.wrote) {
                format!("{} writes to an argument", operator)
            } else if traits & SITE_RANDOM != 0 {
                format!("{} draws random numbers{}", operator, behind)
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
            _ if draws.is_some() => format!("{} draws random numbers{}", draws.unwrap_or_default(), behind),
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
        defs.insert(inst.dst, Def { node, late, graph, made: false, own: false, fixed: false, lands: false });
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
            named: None,
            again: NONE,
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
        pins: Vec::new(),
        body,
        feeds,
        held: vec![true; outputs.len()],
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
        raised: false,
        unbinds: false,
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
/// where there is none. `Err` where finding it would look into a dict
/// that is not looked into (`entry`).
pub unsafe fn kept(o: Obj, keeps: Keeps) -> Result<Obj, ()> {
    if o.is_null() || capi::is_unboxed(o) {
        return Ok(null_mut());
    }
    if keeps == Keeps::Itself {
        return Ok(o);
    }
    let slot = pyffi::_PyObject_GetDictPtr(o);
    if slot.is_null() || (*slot).is_null() {
        return Ok(null_mut());
    }
    if keeps == Keeps::Attributes { Ok(*slot) } else { entry_of(*slot, c"_buffers") }
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
    // Not shown where a dict would be looked into that is not (`entry`).
    let Ok(changed) = kept(target, keeps) else {
        return false;
    };
    if changed.is_null() {
        return true;
    }
    if !attributes {
        return read != changed;
    }
    let Ok(own) = kept(read, Keeps::Attributes) else {
        return false;
    };
    let part = |name: &std::ffi::CStr| if own.is_null() { Ok(null_mut()) } else { entry_of(own, name) };
    let (Ok(parameters), Ok(buffers), Ok(modules)) = (part(c"_parameters"), part(c"_buffers"), part(c"_modules")) else {
        return false;
    };
    ![own, parameters, buffers, modules].contains(&changed)
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
    static TENSOR: pyffi::Once<Option<PyObj>> = pyffi::Once::new();
    let class = if capi::is_unboxed(tensor) {
        let found = TENSOR.get_or_init(|| {
            let class = class_named(c"torch", c"Tensor");
            (!class.is_null()).then(|| PyObj::from_borrowed(class))
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
    /// The store is into an object the path made, and is done where the
    /// instruction is: a read behind it that is no read of that object by
    /// the value it is finds what it finds.
    pub own: bool,
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
    static FOUND: pyffi::Once<(usize, usize, usize, usize)> = pyffi::Once::new();
    *FOUND.get_or_init(|| {
        // The modules keep them.
        let tensor = class_named(c"torch", c"Tensor");
        let class = class_named(c"torch.nn.modules.module", c"Module");
        let own = if class.is_null() { null_mut() } else { pyffi::PyType_GetDict(class as *mut ffi::PyTypeObject) };
        let setattr = if own.is_null() { null_mut() } else { ffi::PyDict_GetItemWithError(own, specials().setattr.as_ptr()) };
        let hooks = pyffi::in_module(pyffi::module(c"torch.nn.modules.module"), c"_global_buffer_registration_hooks");
        ffi::PyErr_Clear();
        xdecref(own);
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
    // What the function of the class looks into, looked into in its
    // place: not where a key is not plain (`entry`).
    let of = |kind: &std::ffi::CStr| entry_of(*slot, kind).ok();
    let holds = |dict: Obj| {
        if dict.is_null() || ffi::PyDict_CheckExact(dict) == 0 { Some(false) } else { entry(dict, name).ok().map(|held| !held.is_null()) }
    };
    let (parameters, modules, buffers) = (of(c"_parameters")?, of(c"_modules")?, of(c"_buffers")?);
    let known = |dict: Obj| dict.is_null() || dict == ffi::Py_None() || ffi::PyDict_CheckExact(dict) != 0;
    if !known(parameters) || !known(modules) || !known(buffers) || holds(parameters)? || holds(modules)? {
        return None;
    }
    if !holds(buffers)? {
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
    classes: usize,
    pins: usize,
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
    /// The objects the path made by calling a class, each with the value
    /// that the class is: what the class has for its objects is read of
    /// the class, in front of whatever is stored into the object.
    pub classes: Vec<(PathValue, PathValue)>,
    /// What goes over the generators whose code the trace is in, one
    /// inside the other: the list of the path that keeps what the
    /// generator gives.
    pub drains: Vec<PathValue>,
    /// The run is no run to make a form of (an operation that is no part
    /// of the path changed what kind of tensor one of its tensors is): the
    /// path is taken down again at the next call.
    pub again: bool,
    /// The graphs of the first function whose first instruction the trace
    /// went through: they are part of the graph the trace is of.
    pub over: Vec<Arc<GraphSite>>,
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
    /// The trace was let go of (`abandon`), and why.
    pub abandoned: Option<String>,
    /// An instruction of a function the path went into raised because a
    /// lookup of `object` found nothing, as the path checks
    /// (`Assert::Missing`), and the raise is on its way out of the
    /// functions the path is in.
    pub absent: bool,
    /// The path ends where a later one will go on: at a block, or a call
    /// of a function, that the tiers have not decided on yet.
    pub provisional: bool,
    /// The things the path made and what may be them or hold them, in
    /// families (see `Family`).
    pub things: Family,
    /// See `Path::pins`.
    pub pins: Vec<(PathValue, PyObj, bool)>,
}

/// The things a path made (a list, a dict, an iterator, a cell, an object
/// of a class it called) and the values of the path that may be one of
/// them or may hold one, at this call or at another, in families: two
/// things are of one family once one may hold the other, or one value may
/// be or hold both. A value can be a thing, or hold one, in four ways
/// only: an instruction of the path made it; it was taken out of a value
/// of a family in which something holds a thing (an item, what an
/// iterator gives, what a cell holds); it is an attribute of one (the
/// dict of an object, a method bound to it); or an instruction that is no
/// operation of the graph was handed one and may give it back or keep it
/// (a call, a tuple that is built). What comes from in front of the path,
/// and what is read of an object from outside, is none: nothing of the
/// path's is stored there before the path lands. Kept as the trace goes,
/// so a value is what it may be where its instruction stands.
///
/// A family is the unit of three things: which changes run in a run
/// (`Path::of`: those of a family that something that runs takes a value
/// of); which object a store is into (`Compiled::record_effect`: none of a
/// family's); and on which side of what stands for the graph its
/// instructions run (`since`).
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
            rec: Recording { values: vec![null_mut()], kept: HashMap::new(), born: vec![capi::TCFG_NO_VERSION], nodes: Vec::new(), failed: None, wrote: false },
            mark: None,
            stores: Vec::new(),
            apart: Vec::new(),
            classes: Vec::new(),
            drains: Vec::new(),
            again: false,
            over: Vec::new(),
            abandoned: None,
            absent: false,
            writes: false,
            fallible: false,
            inplace: false,
            first: None,
            pinned: Vec::new(),
            inlined: Vec::new(),
            lost: false,
            provisional: false,
            things: Family { first: vec![NONE], since: vec![u32::MAX], ..Family::default() },
            pins: Vec::new(),
        }
    }

    fn act(&mut self) -> &mut Act {
        self.acts.last_mut().expect("a trace is in a function")
    }

    /// The value that the class of an object is, for an object the path
    /// made by calling the class.
    pub fn class_of(&self, object: PathValue) -> Option<PathValue> {
        self.classes.iter().find(|(made, _)| *made == object).map(|(_, class)| *class)
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

    /// For a call of a method of a dict the path made that gives an entry
    /// of the dict or what the call was handed (`get`, `pop`): the dict.
    /// The path made the dict, so it is of exactly that class at every
    /// call, and the method is read off it by its name.
    fn entry_of(&self, op: &Op, dst: PathValue) -> Option<PathValue> {
        let dict = match op {
            Op::CallMethod { recv, .. } => *recv,
            Op::Call { callee, .. } => match self.body.iter().rev().find(|inst| inst.dst == *callee).map(|inst| &inst.op) {
                Some(Op::LoadAttr(of, _)) => *of,
                _ => return None,
            },
            _ => return None,
        };
        let (held, gave) = (self.rec.values[dict as usize], self.rec.values[dst as usize]);
        let exact = self.defs[dict as usize].made && !held.is_null() && unsafe { ffi::PyDict_CheckExact(held) != 0 };
        // `keys()`, `values()` and `items()` give a view.
        let view = gave.is_null() || unsafe { std::ffi::CStr::from_ptr((*ffi::Py_TYPE(gave)).tp_name).to_bytes().starts_with(b"dict_") };
        (exact && !view).then_some(dict)
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
    fn note(&mut self, op: &Op, dst: PathValue, kind: Kind) {
        let at = self.body.len() as u32;
        if dst != NONE && self.defs[dst as usize].late && self.things.since[dst as usize] == u32::MAX {
            self.things.since[dst as usize] = at;
        }
        if matches!(kind, Kind::Effect | Kind::Check) {
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
            Op::GetItem(of, _) | Op::IterNext(of) | Op::TupleGet(of, _) | Op::CellGet(of, _) | Op::Unpack(of, _) | Op::ListToTuple(of) | Op::Part(Part::Index(_) | Part::Entry(_) | Part::Cell(_), of) => {
                if dst != NONE && self.things.has(*of) && nested(&self.things, *of) {
                    self.belongs(dst, *of);
                }
            }
            // An attribute may be a part of the thing (the dict of an
            // object) or hold it (a bound method), and what a method is
            // bound to may be the thing.
            Op::LoadAttr(of, _) | Op::Part(_, of) => {
                if dst != NONE {
                    self.belongs(dst, *of);
                }
            }
            // A method of a dict the path made (`get`, and `pop`, which
            // is a change of it): what it gives was in the dict, as
            // above, or is the default it was handed. Not a view of the
            // dict, which goes over it as an iterator does.
            Op::Call { args, .. } | Op::CallMethod { args, .. } if dst != NONE && self.entry_of(op, dst).is_some() => {
                let dict = self.entry_of(op, dst).expect("just asked");
                if nested(&self.things, dict) {
                    self.belongs(dst, dict);
                }
                for v in args.iter().filter(|v| self.things.has(**v)).copied().collect::<Vec<PathValue>>() {
                    self.belongs(dst, v);
                    self.joins(dst, v);
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
        self.things.first.push(NONE);
        self.things.since.push(u32::MAX);
        self.source.push((self.owner(), NONE));
        self.rec.values.push(kept);
        if !kept.is_null() && !capi::is_unboxed(kept) {
            self.rec.kept.entry(kept as usize).or_insert((0, made)).0 += 1;
        }
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
        self.again = true;
        self.abandoned = Some(why);
    }

    pub fn push(&mut self, op: Op, dst: PathValue, kind: Kind, bc: u32) {
        let owner = self.owner();
        // What is made where the path lands (`Def::lands`) is there for
        // the effects behind it and for nothing that runs in front of
        // them: `Compiled::record_inst` ends the path in front of an
        // instruction of a function that takes it, and a run in which
        // anything else took it is no run to make a form of.
        if kind != Kind::Effect {
            let mut lands = false;
            reads(&op, |v| lands |= v != NONE && self.defs[v as usize].lands);
            if lands && self.rec.failed.is_none() {
                self.rec.failed = Some("an instruction takes what is made where the path lands".to_string());
            }
        }
        self.note(&op, dst, kind);
        self.body.push(PathInst { op, dst, kind, bc, stage: Stage::Before, in_graph: false, assert: None, owner, named: None, again: NONE });
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
            // The function it is in is left where one that found the
            // attribute goes on.
            Assert::Missing(v, name) => {
                self.shape.extend([u32::MAX - 7, *v]);
                self.pins.push((*v, name.clone(), true));
            }
            // How a call binds what it is not handed.
            Assert::Enters(_, _, defaults) => self.shape.extend([u32::MAX - 8, *defaults as u32]),
            // What the path was taken down with: the place of an item, a
            // number or a name that was made a constant, an object.
            Assert::Equals(v, value) => self.pins.push((*v, value.clone(), true)),
            Assert::Is(v, object) => self.pins.push((*v, object.clone(), false)),
            _ => {}
        }
        // A check of what the graph gives is made behind the callable,
        // and can fail there unless what it is of is the same whenever the
        // form is for the call.
        let unsure = |def: Def| def.late && !def.fixed;
        let mut late = unsure(self.defs[assert.value() as usize]);
        if let Assert::Same(_, other) | Assert::Apart(_, _, other, _) = &assert {
            // Which object one that the path made is is known in front of
            // the callable.
            let own = matches!(assert, Assert::Apart(..)) && self.class_of(*other).is_some();
            late |= unsure(self.defs[*other as usize]) && !own;
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
        let dst = self.value(Def { node: false, late, graph, made, own: false, fixed, lands: false }, kept);
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
            classes: self.classes.len(),
            pins: self.pins.len(),
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
        self.classes.truncate(mark.classes);
        self.pins.truncate(mark.pins);
        self.apart.clear();
        (self.writes, self.fallible, self.inplace) = mark.flags;
        if self.operators.is_empty() {
            self.first = None;
        }
        self.pinned.truncate(mark.pinned);
        self.defs.truncate(mark.values);
        // A family is kept under its first value, which is still there
        // when a later one is not. What was joined or put behind the
        // callable since the mark stays so: nothing is taken down behind
        // this, and the path is staged by the instruction (`Family::late`).
        self.things.first.truncate(mark.values);
        self.things.since.truncate(mark.values);
        self.source.truncate(mark.values);
        for kept in self.rec.values.drain(mark.values..) {
            if let Some(count) = self.rec.kept.get_mut(&(kept as usize)) {
                count.0 -= 1;
                if count.0 == 0 {
                    self.rec.kept.remove(&(kept as usize));
                }
            }
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
        Op::MakeCell(a) => Op::MakeCell(if *a == UNDEF { UNDEF } else { f(*a) }),
        Op::MakeFunction { code, defaults, kwdefaults, annotations, annotate, closure } => {
            let mut of = |v: V| if v == NONE { NONE } else { f(v) };
            Op::MakeFunction {
                code: of(*code),
                defaults: of(*defaults),
                kwdefaults: of(*kwdefaults),
                annotations: of(*annotations),
                annotate: of(*annotate),
                closure: of(*closure),
            }
        }
        Op::CellSet(a, b) if *b != UNDEF => Op::CellSet(f(*a), f(*b)),
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
        Op::SuperObject { global_super, class, this, explicit } => {
            Op::SuperObject { global_super: f(*global_super), class: f(*class), this: f(*this), explicit: *explicit }
        }
        Op::Format { val, conv, spec } => Op::Format { val: f(*val), conv: *conv, spec: if *spec == NONE { NONE } else { f(*spec) } },
        Op::BuildString(vs) => Op::BuildString(all(vs)),
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
    /// Whether every check of the path is asked in front of the callable:
    /// none waits for what the operations give, so a form without a
    /// callable says of a call that it goes the way of the path.
    pub fn asked_in_front(&self) -> bool {
        let waits = |inst: &PathInst| inst.kind == Kind::Check && matches!(inst.stage, Stage::Inside | Stage::After);
        self.exits.is_empty() && !self.body.iter().any(waits)
    }

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
            held: Vec::new(),
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
            raised: false,
            unbinds: false,
            effects: false,
            mutates: false,
            pinned: Vec::new(),
            constants: Vec::new(),
            exported: Shared::new(Vec::new()),
            registers: Shared::new(Vec::new()),
            stages: Default::default(),
            handed: Vec::new(),
            made: Vec::new(),
            pins: Vec::new(),
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
        returns: bool,
    ) -> Path {
        // A value that is late is so from some instruction on: from the
        // first, where nothing says from which.
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
        // What can still be reached when the path has landed: what a value
        // of the function holds then, what a variable of the frame is bound
        // to last (unless the path lands where the function returns, and
        // the frame is gone), what an effect stores, and what something
        // that can be reached may hold of what it was made of.
        let mut kept = vec![false; nregs];
        writes.iter().for_each(|(made, _)| kept[*made as usize] = true);
        // The frame has every variable the instructions bound, wherever
        // something can look at it: what a variable is bound to last is
        // there when the path lands, read behind it or not. Where the path
        // lands at the return of the function the frame is gone, unless it
        // has a frame object by then (`Path::unbinds`).
        for (at, (slot, v)) in trace.bound.iter().enumerate() {
            if !returns && !trace.bound[at + 1..].iter().any(|(later, _)| later == slot) {
                kept[*v as usize] = true;
                need[*v as usize] |= *v != UNDEF;
            }
        }
        let values = &trace.rec.values;
        // What gives a number, a text, a size, a dtype or a class holds
        // nothing of what it was given.
        let plain = |v: PathValue| {
            let held = values[v as usize];
            !held.is_null() && unsafe { checkable(held) || (!capi::is_unboxed(held) && ffi::PyType_Check(held) != 0) }
        };
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
        // appends to or extends, the object or the dict it stores into,
        // the cell it puts into, the iterator it takes the next item of.
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
        // by reference, so what took it in front of the change (a store, a
        // list of the caller's, another list of the path) has the change,
        // and what takes the next item of an iterator needs every item in
        // front of it taken. The instructions are gone over from the last
        // to the first, which finds a use behind a change and not one in
        // front of it: they are gone over again with what the last time
        // found, until that finds no more.
        let mut alive = vec![false; nregs];
        let (first_need, first_kept) = (need, kept);
        let (mut need, mut kept) = (first_need.clone(), first_kept.clone());
        // The values that are no tensors and that the graph computes.
        let mut wanted = vec![false; nregs];
        let (mut outputs, mut exits) = (Vec::new(), Vec::new());
        let (mut branches, mut calls) = (0, 0);
        let constants: std::cell::RefCell<Vec<PathValue>> = std::cell::RefCell::new(Vec::new());
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
                            // What is the same in every call a form is for was
                            // checked when the path was taken down: the tensor
                            // it is computed from need not be given back for it.
                            Assert::Truth(..) | Assert::Null(..) | Assert::Guard(_) if late(v as PathValue, at) && defs[v].fixed => {
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
                                let mut behind = late(v as PathValue, at);
                                if let Assert::Same(_, other) | Assert::Apart(_, _, other, _) = assert {
                                    need[*other as usize] = true;
                                    // Which object one that the path made is
                                    // does not wait for what is put into it.
                                    let own = matches!(assert, Assert::Apart(..)) && trace.classes.iter().any(|(made, _)| made == other);
                                    behind |= late(*other, at) && !own;
                                }
                                if behind { Stage::After } else { Stage::Before }
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
                        // What changes something the path made goes with it,
                        // unless it can fail: then it runs, and the thing
                        // with it, so that a call in which it fails is not
                        // the path's.
                        let target = changes(inst);
                        let taken = target != NONE && (need[target as usize] || alive[thing(target) as usize]);
                        let fails = !leaves_quietly(&inst.op, defs, |v| defs[v as usize].made || built[v as usize]);
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
                    // a call in which it raises is not the path's. What is
                    // computed behind the graph and is the same in every call
                    // the form is for did not fail when the path was taken
                    // down, and does the same again.
                    // Nor is a method that was read off a tensor the graph
                    // gives, which the operation that calls it stands for,
                    // or off a list, a dict or a set the path made (the
                    // `append` of a list that a loop fills).
                    _ if !idle(&inst.op) && !(def.late && def.fixed) && !bound_method(&inst.op, dst) => uses(&inst.op, &mut need),
                    _ => Stage::Skipped,
                };
                // What is put into something the path made is held where that
                // is, which may be anywhere; a check holds nothing.
                let holds = match (inst.stage, inst.kind) {
                    (Stage::Landing, _) => true,
                    (Stage::Before | Stage::After, Kind::Residual) => true,
                    (Stage::Before | Stage::After, Kind::Check) => false,
                    (Stage::Before | Stage::After, _) => dst != NONE && kept[dst as usize] && !plain(dst),
                    _ => false,
                };
                if holds {
                    inst.op.for_each_use(|v| kept[v as usize] = true);
                }
            }
            let mut found = alive.clone();
            (0..nregs).filter(|v| need[*v]).for_each(|v| found[thing(v as PathValue) as usize] = true);
            let left_out = |inst: &PathInst| inst.stage == Stage::Skipped && changes(inst) != NONE && found[thing(changes(inst)) as usize];
            if !trace.body.iter().any(left_out) {
                break;
            }
            alive = found;
            (need, kept) = (first_need.clone(), first_kept.clone());
            wanted.iter_mut().for_each(|wanted| *wanted = false);
            (outputs.clear(), exits.clear(), constants.borrow_mut().clear());
            (branches, calls) = (0, 0);
        }
        outputs.reverse();
        exits.reverse();
        let held: Vec<bool> = outputs.iter().map(|v| kept[*v as usize]).collect();
        let effects = trace.body.iter().any(|inst| inst.stage == Stage::Landing);
        // The reads in front of the callable that an earlier one makes as
        // well: of a global under one name in one module's globals, of an
        // attribute under one name of one value. Which value a read is of
        // is followed through such reads.
        let mut same: Vec<PathValue> = (0..nregs as PathValue).collect();
        let mut globals: HashMap<(usize, usize, usize), PathValue> = HashMap::new();
        let mut attributes: HashMap<(PathValue, usize), PathValue> = HashMap::new();
        for inst in trace.body.iter_mut().filter(|inst| inst.stage == Stage::Before && inst.assert.is_none() && inst.dst != NONE) {
            let function = unsafe { &(*trace.owners[inst.owner as usize]).func };
            let first = match &inst.op {
                Op::LoadGlobal(name) => {
                    let name = function.consts[*name as usize].as_ptr() as usize;
                    let key = (function.globals.as_ptr() as usize, function.builtins.as_ptr() as usize, name);
                    Some(*globals.entry(key).or_insert(inst.dst))
                }
                Op::LoadAttr(of, name) => {
                    let key = (same[*of as usize], function.consts[*name as usize].as_ptr() as usize);
                    Some(*attributes.entry(key).or_insert(inst.dst))
                }
                _ => None,
            };
            if let Some(first) = first.filter(|first| *first != inst.dst) {
                inst.again = first;
                same[inst.dst as usize] = same[first as usize];
            }
        }
        let constants = constants.into_inner();
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
        for inst in trace.body.iter().filter(|inst| matches!(inst.stage, Stage::Before | Stage::After | Stage::Landing)) {
            if inst.dst != NONE {
                there[inst.dst as usize] = true;
            }
        }
        let mut bound: Vec<(u32, PathValue)> = Vec::new();
        // The variables that are left unbound though the instructions
        // bind them.
        let mut left: Vec<u32> = Vec::new();
        for (slot, v) in &trace.bound {
            bound.retain(|(other, _)| other != slot);
            left.retain(|other| other != slot);
            bound.push((*slot, if there[*v as usize] { *v } else { UNDEF }));
            if *v != UNDEF && !there[*v as usize] {
                left.push(*slot);
            }
        }
        let unbinds = !left.is_empty();
        Path {
            root,
            land,
            resume,
            nregs,
            unbinds,
            stages: stages_of(&trace.body),
            handed: handed_of(&trace.inputs, &trace.body, &writes, &bound),
            made: trace.defs.iter().map(|def| def.made).collect(),
            pins: std::mem::take(&mut trace.pins),
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
            raised: false,
            effects,
            mutates: trace.writes,
            pinned: std::mem::take(&mut trace.pinned),
            constants,
            held,
            exported: Shared::new(Vec::new()),
            registers: Shared::new(Vec::new()),
        }
    }
}
