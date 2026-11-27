// The rule table: what an instruction may do under a trace, by rows.
//
// A row is keyed on (what, the classes of its operands) and says what the
// instruction reads (and a region is therefore checked on), what it writes,
// where its result comes from and whether it enters Python code. The table
// has no row for "anything else": what a family of it has taken over and no
// row describes does not run under a trace, and sits in no region.
//
// What it is given
//   - by covered.rs, before an instruction of a traced body or of a frame
//     under it runs: what it is and the exact class of every operand
//     (license);
//   - by the optimizer's verdicts (mod.rs's pure_instr and observed_ops,
//     shown.rs's instr), which know no operand: the instruction (advice);
//   - by each family, a file of its own listed in FAMILIES: for any What,
//     whether it has taken it over, and then its rows or why it stops.
// What it guarantees
//   - license() is NoRow, with the reason, for a What that a family stops,
//     and for one that families have lines for of which none takes these
//     classes, or these values; Row for the first row that does; NotMine
//     where no family has a stop or lines for the What: the code that decided
//     before decides, and the What is left BY NAME (later()) to rules that
//     are not written: an instruction by its family's list, a callee no
//     family lists by its kind (callees.rs). No What is NotMine without a
//     name, and one that nothing names is a stop;
//   - A WHAT WHOSE ANSWER GOES BY THE CLASS OF AN OPERAND has LINES
//     (Taken::Lines), tried in their order: a row; a stop for operands of
//     the classes the line names, with its reason; or, for such operands,
//     the name of the rules they are left to (NotMine for those operands
//     only: later_of() says the name). Operands that no line takes are a
//     stop, as with rows;
//   - A ROW THAT ENTERS PYTHON CODE (Enters::Named) names the functions:
//     names() gives them, by identity, found by the lookup the instruction
//     itself does, in C. A frame of any other function under the
//     instruction is not the row's;
//   - A ROW WITH READS OF ITS OWN (Row::before). What an instruction reads
//     besides its operands is recorded, as checks of the region being traced,
//     by hooks of the executor for most kinds still. A row may carry that
//     itself: a function of its family's file that license() calls once the
//     classes fit, before the instruction runs, with the frame and the
//     operands. It tests what the classes do not say (the VALUES the row is
//     for) and, if they fit, hands every read the row's `reads` names to
//     exec/reads.rs with the value it found, by reads of C containers only.
//     That is before the instruction and not after it because such a row
//     writes nothing and enters nothing: what it finds is what the
//     instruction finds. Err is "no row for these values";
//   - A ROW THAT WRITES OPERANDS (Write::Operands) says which tensors a call
//     writes by what calls of the callee were measured to write, declared by
//     a schema or not. Saying is all the row does: who runs the call acts on
//     it (see torch.rs). A ROW THAT DRAWS (Write::Draws) says the same of
//     the default random generators: the call moves their state;
//   - UNDER A CALL TAKEN AT ANOTHER'S WORD (covered.rs's Trusted: a callable
//     the pass list declares pure, a call a cfg pass made movable, a
//     function of torch that another tracer keeps whole) the table is asked
//     about every instruction as anywhere else, and every answer is heard:
//     a stop of a family, an opcode that is no instruction, rows that do
//     not take the operands, a row's `before`. The word says that the call
//     only computes; it does not say that an instruction nobody can check
//     is fine there. What such a call may run UNHEARD is a list (UNHEARD,
//     through hear()): the families that are not written yet, each with
//     what stands in for its rules there. What is left to a family that is
//     not on the list is not run unheard: the trace stops;
//   - advice() is Err where license() is NoRow for every class: a stop, or
//     taken over with no row at all; advice_callee() is Err for a callee a
//     family stops. It never says more than that: a verdict
//     is advice on where a region may be, and license() is asked again;
//   - no answer here runs code of the program: a class is compared by
//     address, a callee is an address.
// What it may not assume
//   - that a Row is all the license an instruction has: the columns are not
//     yet what a trace records from, and the checks that were there before
//     the table still run behind a Row.

use pyo3::ffi::PyObject;

use super::ffi::Frame;
use crate::cfg::ir::Op;

pub mod callees;
pub mod dump;
pub mod groups;
pub mod imports;
pub mod ops;
pub mod torch;
pub mod truth;

pub use groups::{Class, Group};

macro_rules! kinds {
    ($($k:ident)*) => {
        // An Op without its fields, by the names hist.rs counts under.
        #[derive(Copy, Clone, PartialEq, Eq, Debug)]
        pub enum Kind { $($k),* }
        pub const KINDS: &[Kind] = &[$(Kind::$k),*];
    };
}
kinds!(Null Unknown Param Const Project ToBool IsNone LoadLocal LoadLocalChecked StoreLocal DeleteLocal LoadGlobal LoadGlobalIn LoadAttr LoadMethod StoreAttr Call Binary Compare
    BuildTuple GetIter MakeFunction Yield Resume IterNext Send EvalBreaker MakeCell CopyFreeVars LoadDeref StoreDeref DeleteDeref LoadFromDictOrDeref Aten Lit Pinned Expect Autograd Region Removed);

// What a rule is about: an instruction by its kind, a generic opcode with
// its oparg, a callee by identity (borrowed).
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum What {
    Op(Kind),
    Opcode(u8, u32),
    Callee(*mut PyObject),
}

impl What {
    pub fn name(&self) -> String {
        match *self {
            What::Op(k) => format!("{k:?}"),
            What::Opcode(op, arg) => ops::opcode_name(op, arg),
            What::Callee(f) => unsafe { torch::name(f) }.map_or_else(|| format!("the callee at {f:p}"), str::to_string),
        }
    }
}

// The columns, in the words of the table's design: some values have no row
// yet and are for the families to come (hence the allows). What is read besides which objects the operands are: a slot
// of the frame, a constant of the code, the item of a globals or builtins
// dict, the content of a cell, the closure of the frame's function, which
// object an operand is, an operand's class and its MRO, the characters of a
// str, the exception the thread handles, the interpreter's pending work, an
// item of the interpreter's table of loaded modules, an attribute of an
// object as its class resolves it, how many items a container holds.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Read { Slot, Constant, Global, Cell, Closure, Identity, Class, Text, Handled, Pending, Modules, Attribute, Length }
// What is written: a slot of the frame, the container at an operand (one
// the frame is building), the exception the thread handles, the memory of
// tensors among the operands of a call, the state of the default random
// generators (of the CPU and of the CUDA devices in use) and the memory of
// the operands named besides.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Write { None, Slot, Fills(u8), Handled, Operands(&'static [Written]), Draws(&'static [Written]) }
// An operand of a call whose memory the callee was measured to write: where
// it is among the operands (its position, the receiver of a method first, or
// the name of its keyword), and whether the schema of the operator the call
// dispatches declares that write.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Written {
    pub at: &'static str,
    pub declared: bool,
}
// Where the result is from: a value of the engine's, an entry of the
// region's list, an operand, what a slot holds, a part of an operand's
// result, a new object, an object a check pins, a frame's return.
// HOW A RESULT IS KNOWN (RESULTS below says it in the dump): Lit is a
// function of the operands and of what the row reads: it is known by its
// VALUE when those are, and says nothing of which object it is; Pinned is
// the OBJECT that a recorded read holds.
#[allow(dead_code)]
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Res { None, Lit, Entry, Operand(u8), Slot, Part, Made, Pinned, Return }
// The Python code it enters: none, the frames the row names, or only what
// nobody named (a handler, a finalizer: each a stop of its own).
#[allow(dead_code)]
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Enters { Never, Named, Foreign }
// A scalar among the operands: no row of its own, an operand of the entry,
// a part of what is made, or the region is for that value only.
#[allow(dead_code)]
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Scalars { None, Operand, Parts, Specialize }

// Where an instruction is about to run: its frame, the names of the frame's
// code (an oparg that is a name counts into them), its operands and, for a
// call with keywords, the tuple of their names (else NULL), all borrowed.
pub struct Site<'a> {
    pub frame: *mut Frame,
    pub names: *const *mut PyObject,
    pub operands: &'a [*mut PyObject],
    pub kwnames: *mut PyObject,
}

// See "a row with reads of its own" at the head of the file. It is handed
// the row it is the `before` of. WHAT A `before` MAY DO: look at the
// operands by identity, exact type and C-level reads of containers, and
// record what it reads through reads.rs. It is called by every asker of
// license() (the trace of today's engine, the recorder of call lists), so
// it runs no Python code, makes no table on first use (a family makes its
// tables at one moment before any trace: torch::ready) and tells nothing
// to one asker's machinery.
pub type Before = unsafe fn(&Row, &What, &Site) -> Result<(), String>;

pub struct Row {
    // Operands by position (below 0: from the last) and the class each is to have.
    pub operands: &'static [(i8, Class)],
    pub reads: &'static [Read],
    pub writes: Write,
    pub result: Res,
    pub enters: Enters,
    pub scalars: Scalars,
    pub why: &'static str,
    pub before: Option<Before>,
    // The one form of the call's operands the row is for, in the words of
    // its family (torch.rs); "": the row is not for one form.
    pub form: &'static str,
}

// Whether operands of the classes `classes` are what `operands` names.
unsafe fn takes(operands: &[(i8, Class)], classes: &[Class]) -> bool {
    operands.iter().all(|(at, want)| {
        let k = if *at < 0 { classes.len() as isize + *at as isize } else { *at as isize };
        usize::try_from(k).ok().and_then(|k| classes.get(k)).is_some_and(|have| want.takes(*have))
    })
}

pub enum License {
    // Nothing records from a row's columns yet.
    Row(&'static Row),
    NoRow(String),
    NotMine,
}

// One line of a What whose answer goes by the classes of its operands: a
// row; or, for operands of the classes named (as a row names them), a stop
// and why, or the name of the rules those operands are left to.
pub enum Line {
    Row(Row),
    Stop(&'static [(i8, Class)], &'static str),
    Later(&'static [(i8, Class)], &'static str),
}

// A family's answer for a What: not its own; its own when the family named
// is written; an opcode that is instructions of these kinds; a stop and why;
// its rows; its lines by the classes of the operands; an opcode no list of
// the family has.
#[derive(Copy, Clone)]
pub enum Taken {
    No,
    Later(&'static str),
    As(&'static [Kind]),
    Stop(&'static str),
    Rows(&'static [Row]),
    Lines(&'static [Line]),
    Unlisted,
}

// A line as license() goes through it.
#[derive(Copy, Clone)]
enum Entry {
    Row(&'static Row),
    Stop(&'static [(i8, Class)], &'static str),
    Later(&'static [(i8, Class)], &'static str),
}

pub struct Family {
    pub name: &'static str,
    pub rule: fn(&What) -> Taken,
    // Every What the family answers for, by name, for the dump.
    pub table: fn() -> Vec<(String, Taken)>,
}

static FAMILIES: &[Family] = &[ops::FAMILY, imports::FAMILY, torch::FAMILY];

// What the table says of `what` before any operand is looked at.
enum Said {
    // No trace runs it, and why.
    Stop(String),
    // The lines of every family that has rows or lines for it.
    Lines(Vec<Entry>),
    // No family has rows for it: it is left, by this name, to a family of
    // rules that is not written.
    Later(&'static str),
}

// The one walk of the families, for license(), later() and the advice. A
// stop of any family wins over the lines of another; the lines of all
// families that have some are tried together, in the order of FAMILIES and
// of each family's list; a family that only names who will take the What
// counts where nobody has lines. Where no family says anything: a callee is
// told by its kind (callees.rs: the name it is left under, or a stop), and
// anything else is a stop.
fn said(what: &What) -> Said {
    let mut found: Option<Vec<Entry>> = None;
    let mut later = None;
    for family in FAMILIES {
        match (family.rule)(what) {
            Taken::No => {}
            Taken::Later(name) => later = later.or(Some(name)),
            Taken::Stop(why) => return Said::Stop(format!("{} stops a trace: {why}", what.name())),
            Taken::As(_) | Taken::Unlisted => return Said::Stop(format!("{} is no instruction the rules list", what.name())),
            Taken::Rows(rows) => found.get_or_insert_default().extend(rows.iter().map(Entry::Row)),
            Taken::Lines(lines) => found.get_or_insert_default().extend(lines.iter().map(|line| match line {
                Line::Row(row) => Entry::Row(row),
                Line::Stop(operands, why) => Entry::Stop(operands, why),
                Line::Later(operands, name) => Entry::Later(operands, name),
            })),
        }
    }
    match (found, later, what) {
        (Some(lines), ..) => Said::Lines(lines),
        (None, Some(name), _) => Said::Later(name),
        (None, None, What::Callee(f)) => match unsafe { callees::of(*f) } {
            Taken::Later(name) => Said::Later(name),
            Taken::Stop(why) => Said::Stop(format!("{} stops a trace: {why}", what.name())),
            _ => Said::Stop(format!("{} is a callee of a kind the rules do not tell", what.name())),
        },
        (None, None, _) => Said::Stop(format!("{} is left to no family of rules", what.name())),
    }
}

// In which order license() goes through the lines, in words, for the dump:
// of the families, and of the rows.
pub const ORDER: [&str; 2] = [
    "A stop of any family is the answer whatever rows another family has. Else the rows of every family that has rows for the instruction are tried together: family by family in this order, and within a family in the order of its lines.",
    "The first line whose classes take the operands is the answer, and no line after it is asked: a stop for those operands, the name they are left under, or a row. A row whose before refuses the operands is no answer: the lines after it are asked.",
];

// How a result is known from the RESULT column of a row, for the dump.
pub const RESULTS: &str = "Lit: a function of the operands and of what the row reads, known by its value when those are known (which object it is, is not said; a bool is one object per value). Pinned: the object that a recorded read holds.";

// What is answered where no line of a family decides, for the dump: the
// case, the answer, and why. (A callee that no family lists
// is answered by its kind: the list of callees.rs. What a call taken at
// another's word runs unheard: UNHEARD.) The first eight are said() and
// license() and hear() below; the last two are the torch family's own
// (torch.rs: rule(), and the `before` of its rows for forms).
pub const DEFAULTS: &[(&str, &str, &str)] = &[
    ("an opcode number that no list of a family has", "stop", "A number that is no opcode of CPython 3.12, or an opcode the executor does not know, is no instruction a rule describes."),
    ("an opcode that is lowered to instruction kinds, met as the opcode", "stop", "A trace runs the kinds an opcode is lowered to, each by its own line: the opcode itself is no instruction of a plan."),
    ("rows, of which none takes the classes of the operands", "stop", "A row is for the classes it names: an operand of another class, a subclass of a member too, is a case no row describes."),
    ("rows that take the classes, each with a before that refuses the operands", "stop", "A row with a before is for the values it tests: where every such row refuses, no row describes the instruction on these operands."),
    ("an instruction or an opcode that no family answers for", "stop", "The first family answers for every kind and every opcode number: one that it does not answer for is new, and is not left to the code that decided before."),
    ("a callee that no family lists, whose kind answers neither a name nor a stop", "stop", "A kind of callee answers the name it is left under, or a stop: any other answer is a mistake in the list of kinds, and licenses nothing."),
    ("a stop of one family and rows of another for one instruction", "stop", "A stop says that no trace runs the instruction: a row of another family does not take that back."),
    ("under a call taken at another's word, left to a name that is not on the list of what runs unheard", "stop", "What such a call runs with no rule heard is that list and nothing else: a name that comes later is not unheard because nobody listed it."),
    ("of the torch family: any callee, while the table of torch's members is not made", "stop", "Without the table no callee can be told from a member that stops: it is made where a function goes onto the executor, before any trace."),
    ("of the torch family: a call of a member that has lines per form, in a form that no line has", "stop", "Such a member has rows for the forms of operands that were measured to only compute: another form was not measured, or was measured to follow the data."),
];

// The `before` functions of the families' rows, for the dump: by family,
// the function, its name, and what it refuses and what it records. A row's
// function that is on no family's list is shown as UNNAMED, and a test
// fails for it.
pub type Says = (Before, &'static str, &'static str);
pub const BEFORES: &[(&str, &[Says])] = &[("ops", truth::BEFORES), ("imports", imports::BEFORES), ("torch", torch::BEFORES)];

// The name of a row's `before` for the dump: "-" where it has none, UNNAMED
// for a function that is on no family's list.
pub fn before_name(row: &Row) -> &'static str {
    let Some(before) = row.before else { return "-" };
    BEFORES.iter().flat_map(|(_, list)| list.iter()).find(|(listed, ..)| *listed as usize == before as usize).map_or("UNNAMED", |(_, name, _)| name)
}

// Asked before an instruction runs under a trace; `classes`: of every
// operand, in order; `site`: where it runs. NotMine is never silent: later()
// names the family the What is left to.
pub unsafe fn license(what: &What, classes: &[Class], site: &Site) -> License {
    answer(what, classes, site).0
}

// license(), and with NotMine the name of the rules the instruction is left
// to: the What's, or that of the line that takes these operands.
unsafe fn answer(what: &What, classes: &[Class], site: &Site) -> (License, Option<&'static str>) {
    let lines = match said(what) {
        Said::Stop(why) => return (License::NoRow(why), None),
        Said::Later(name) => return (License::NotMine, Some(name)),
        Said::Lines(lines) => lines,
    };
    let of_classes = || classes.iter().map(|c| c.name()).collect::<Vec<_>>().join(", ");
    let mut why = format!("{} has no row for operands of the classes ({})", what.name(), of_classes());
    for line in lines {
        match line {
            Entry::Row(row) if takes(row.operands, classes) => match row.before.map_or(Ok(()), |before| before(row, what, site)) {
                Ok(()) => return (License::Row(row), None),
                Err(values) => why = format!("{} has no row for these operands: {values}", what.name()),
            },
            Entry::Stop(operands, reason) if takes(operands, classes) => return (License::NoRow(format!("{} stops a trace on operands of the classes ({}): {reason}", what.name(), of_classes())), None),
            Entry::Later(operands, name) if takes(operands, classes) => return (License::NotMine, Some(name)),
            Entry::Row(_) | Entry::Stop(..) | Entry::Later(..) => {}
        }
    }
    (License::NoRow(why), None)
}

// The name of the rules that an instruction on operands of these classes is
// left to, where license() is NotMine for them.
pub unsafe fn later_of(what: &What, classes: &[Class], site: &Site) -> Option<&'static str> {
    answer(what, classes, site).1
}

// The Python functions that the instruction enters by the row license()
// chose for it, borrowed, by identity (NULL: none in that place). A row
// that does not say Enters::Named names none; for one that does, the
// family that has the row says which, by the lookup the instruction itself
// does. Runs no code of the program.
pub unsafe fn names(row: &Row, what: &What, site: &Site) -> [*mut PyObject; 2] {
    let none = [std::ptr::null_mut(); 2];
    let (Enters::Named, Some(before)) = (row.enters, row.before) else { return none };
    NAMERS.iter().find(|(of, _)| *of as usize == before as usize).map_or(none, |(_, namer)| namer(row, what, site))
}

// For each `before` of a row that enters Python code it names: the function
// that says which.
pub type Namer = unsafe fn(&Row, &What, &Site) -> [*mut PyObject; 2];
const NAMERS: &[(Before, Namer)] = &[(truth::python_truth, truth::enters)];

// What a call taken at another's word may run without a rule being heard:
// the families of rules that are not written yet, by the name an entry of
// the table is left to them under, each with what stands in for its rules
// there. Outside such a call the code that decided before the table asks of
// these instructions whether the code they run only computes (a verdict on
// what was observed) and, of a call, whether that callee was seen there;
// under such a call it asks neither, and the word is all there is for it.
// What the instruction READS is recorded there as anywhere, by the hook
// named; where there is none, the entry says so.
pub const UNHEARD: &[(&str, &str)] = &[
    ("attributes", "an attribute load hands the object, the name and what it found to the executor's hook (reads::attr, reads::super_attr) wherever it runs"),
    ("subscripts", "a truth test through a Python __len__ (the one truth test that has no line of its own), a comparison, a subscript and a membership test hand what they read to the executor's hooks (reads::truth, used, subscript, contains) wherever they run; a display of a dict or a set reads no object but through the hash of its keys"),
    ("iteration", "what is iterated or unpacked is handed to the executor's hooks whole when the iteration begins (reads::iterated, used), wherever it runs; a step of an iterator that was there before the trace has no hook, here as outside such a call"),
    ("torch", "an operator hands both operands to the executor's hook (reads::used) wherever it runs, and on tensors it is an operator the trace sees"),
    ("functions", "a call with starred arguments hands the callee and what it is given to the executor's hooks (reads::called, starred, spread) wherever it runs, and the frames it enters are asked about one instruction at a time"),
    ("stores", "a store is taken by the journal of writes (writes.rs) wherever it runs: made again by the region, or the trace is refused"),
    ("descriptors", "a format and a unary plus hand their operand to the executor's hook (reads::used) wherever they run"),
    ("classes", "a raise ends a block and is run by the executor as the block's end: it is not an instruction that is asked about, here or outside such a call"),
    ("recorder", "what the optimizer put in place of instructions it judged reads what its plan's guard checked"),
    ("native", "a member of torch that enters torch's own Python kernels: frames that are asked about one instruction at a time"),
    ("ambient", "a member of torch that draws, makes a tensor of no tensor or follows ambient state runs as an operator the trace sees, and the ambient word is compared where a region is entered"),
    ("facts", "a member of torch that returns a fact or a value of a tensor: the trace's own refusal of a value read and the key check of a fact read from state work wherever the call runs"),
    ("callees: python functions", "its frame is asked about one instruction at a time, as any frame under the body"),
    ("callees: bound methods", "a call of the function it is bound from with the receiver in front: a Python function's frame is asked about one instruction at a time, a C function is as a builtin"),
    ("callees: instances", "the Python function its class has under __call__ runs in a frame that is asked about one instruction at a time"),
    ("callees: partials", "a call of the callable it holds with the arguments it holds in front: a Python function's frame is asked about one instruction at a time, a C function is as a builtin"),
    ("callees: classes", "a Python __new__ or __init__ runs in a frame that is asked about one instruction at a time; what C code makes of its operands goes to the executor's hook (reads::built)"),
    ("callees: classes of a metaclass", "as a class; the metaclass's own __call__, if it is Python, is a frame that is asked about"),
    ("callees: builtins", "what it is handed goes to the executor's hooks (reads::call, called, built); that it writes nothing and calls nothing back is said by the word alone"),
    ("callees: method descriptors", "as a builtin: its operands go to the executor's hooks, and the word alone says that it writes nothing"),
    ("callees: slot wrappers", "as a builtin: its operands go to the executor's hooks, and the word alone says that it writes nothing"),
    ("callees: C callables", "as a builtin: its operands go to the executor's hooks, and the word alone says what the C code behind the object does"),
    ("callees: torch, unswept", "a C callable of torch that was not measured runs as an operator the trace sees; nothing but the word says that it writes or draws nothing. Listed because it ran so before the list was written, not because that is right"),
];

// What license() answers as the trace of today's engine hears it
// (covered.rs), which runs its older checks behind the table: a stop, with
// the reason; a row; left to the code that decided before; or, under a call
// taken at another's word, where that code asks nothing, run unheard, with
// the family it is left to.
pub enum Heard {
    Stop(String),
    Row(&'static Row),
    Older,
    Unheard(&'static str),
}

// license(), for an instruction that runs under a call taken at another's
// word (`at_a_word`) or not. Every answer is heard there as anywhere; what
// no family has taken over is run unheard there only if UNHEARD lists the
// family it is left to, and is a stop if not.
pub unsafe fn hear(what: &What, classes: &[Class], site: &Site, at_a_word: bool) -> Heard {
    let (license, left_to) = answer(what, classes, site);
    match license {
        License::NoRow(why) => Heard::Stop(why),
        License::Row(row) => Heard::Row(row),
        License::NotMine if !at_a_word => Heard::Older,
        License::NotMine => match left_to {
            Some(family) if UNHEARD.iter().any(|(listed, _)| *listed == family) => Heard::Unheard(family),
            Some(family) => Heard::Stop(format!("{} is left to the rules of {family}, which are not written, and is not among what a call taken at another's word may run unheard", what.name())),
            None => Heard::Stop(format!("{} is left to no family of rules, and a call taken at another's word runs unheard only what the rules list", what.name())),
        },
    }
}

// The family of rules that `what` is left to whatever its operands are,
// where license() is NotMine for every operand: for a report. (Where only
// the operands of some classes are left to a name: later_of.)
pub fn later(what: &What) -> Option<&'static str> {
    match said(what) {
        Said::Later(name) => Some(name),
        _ => None,
    }
}

// Asked by a verdict about the callee its visits called: Err, with the
// reason, if the table stops that callee.
pub fn advice_callee(f: *mut PyObject) -> Result<(), String> {
    match said(&What::Callee(f)) {
        Said::Stop(why) => Err(why),
        _ => Ok(()),
    }
}

// Asked by a verdict, which knows no operand: Err, with the reason, if no
// trace could run the instruction whatever its operands are.
pub fn advice(op: &Op) -> Result<(), String> {
    let what = ops::what(op, None);
    match said(&what) {
        Said::Stop(why) => Err(why),
        Said::Lines(lines) if lines.iter().all(|line| matches!(line, Entry::Stop(..))) => Err(format!("{} has no row", what.name())),
        _ => Ok(()),
    }
}

// Makes what the families read under a trace and never make there: the
// table of torch's members, the classes of plain tensors, the kinds of
// callees. Runs Python code: called where a function goes onto the
// executor and where a recording begins, never under an instruction.
pub fn ready() {
    torch::ready();
    super::plain_tensor_types();
    callees::ready();
    groups::ready();
}
