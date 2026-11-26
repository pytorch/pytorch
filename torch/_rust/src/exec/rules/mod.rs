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
//     and for one that families have rows for of which none takes these
//     classes; Row for the first row that does; NotMine where no family has
//     taken the What over: the code that decided before decides;
//   - advice() is Err where license() is NoRow for every class: a stop, or
//     taken over with no row at all. It never says more than that: a verdict
//     is advice on where a region may be, and license() is asked again;
//   - no answer here runs code of the program: a class is compared by
//     address, a callee is an address.
// What it may not assume
//   - that a Row is all the license an instruction has: the columns are not
//     yet what a trace records from, and the checks that were there before
//     the table still run behind a Row.

use pyo3::ffi::PyObject;

use crate::cfg::ir::Op;

pub mod dump;
pub mod groups;
pub mod ops;

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
            What::Callee(f) => format!("the callee at {f:p}"),
        }
    }
}

// The columns, in the words of the table's design: some values have no row
// yet and are for the families to come (hence the allows). What is read besides which objects the operands are: a slot
// of the frame, a constant of the code, the item of a globals or builtins
// dict, the content of a cell, the closure of the frame's function, which
// object an operand is, an operand's class and its MRO, the characters of a
// str, the exception the thread handles, the interpreter's pending work.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Read { Slot, Constant, Global, Cell, Closure, Identity, Class, Text, Handled, Pending }
// What is written: a slot of the frame, the container at an operand (one
// the frame is building), the exception the thread handles.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Write { None, Slot, Fills(u8), Handled }
// Where the result is from: a value of the engine's, an entry of the
// region's list, an operand, what a slot holds, a part of an operand's
// result, a new object, an object a check pins, a frame's return.
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

pub struct Row {
    // Operands by position (below 0: from the last) and the class each is to have.
    pub operands: &'static [(i8, Class)],
    pub reads: &'static [Read],
    pub writes: Write,
    pub result: Res,
    pub enters: Enters,
    pub scalars: Scalars,
    pub why: &'static str,
}

impl Row {
    unsafe fn takes(&self, classes: &[Class]) -> bool {
        self.operands.iter().all(|(at, want)| {
            let k = if *at < 0 { classes.len() as isize + *at as isize } else { *at as isize };
            usize::try_from(k).ok().and_then(|k| classes.get(k)).is_some_and(|have| want.takes(*have))
        })
    }
}

pub enum License {
    // Nothing records from a row's columns yet.
    Row(#[allow(dead_code)] &'static Row),
    NoRow(String),
    NotMine,
}

// A family's answer for a What: not its own; its own when the family named
// is written; an opcode that is instructions of these kinds; a stop and why;
// its rows; an opcode no list of the family has.
#[derive(Copy, Clone)]
pub enum Taken {
    No,
    Later(&'static str),
    As(&'static [Kind]),
    Stop(&'static str),
    Rows(&'static [Row]),
    Unlisted,
}

pub struct Family {
    pub name: &'static str,
    pub rule: fn(&What) -> Taken,
    // Every What the family answers for, by name, for the dump.
    pub table: fn() -> Vec<(String, Taken)>,
}

static FAMILIES: &[Family] = &[ops::FAMILY];

// What the families say of `what`: Err, why it stops; None, nobody took it
// over; else the rows of every family that did.
fn rows(what: &What) -> Result<Option<Vec<&'static Row>>, String> {
    let mut found: Option<Vec<&'static Row>> = None;
    for family in FAMILIES {
        match (family.rule)(what) {
            Taken::No | Taken::Later(_) => {}
            Taken::Stop(why) => return Err(format!("{} stops a trace: {why}", what.name())),
            Taken::As(_) | Taken::Unlisted => return Err(format!("{} is no instruction the rules list", what.name())),
            Taken::Rows(rows) => found.get_or_insert_default().extend(rows),
        }
    }
    Ok(found)
}

// Asked before an instruction runs under a trace; `classes`: of every
// operand, in order.
pub unsafe fn license(what: &What, classes: &[Class]) -> License {
    match rows(what) {
        Err(why) => License::NoRow(why),
        Ok(None) => License::NotMine,
        Ok(Some(rows)) => rows.into_iter().find(|r| r.takes(classes)).map_or_else(
            || License::NoRow(format!("{} has no row for operands of the classes ({})", what.name(), classes.iter().map(|c| c.name()).collect::<Vec<_>>().join(", "))),
            License::Row,
        ),
    }
}

// Asked by a verdict, which knows no operand: Err, with the reason, if no
// trace could run the instruction whatever its operands are.
pub fn advice(op: &Op) -> Result<(), String> {
    let what = ops::what(op, None);
    match rows(&what)? {
        Some(rows) if rows.is_empty() => Err(format!("{} has no row", what.name())),
        _ => Ok(()),
    }
}
