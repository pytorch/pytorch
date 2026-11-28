// Where a tensor's value goes to Python: a region ends in front of it.
//
// A region is the torch ops of one traced run, made again. An instruction
// that hands a tensor's value to Python (`if t.any():`, `bool(t)`, `int(t)`,
// `float(t)`, `t.item()`, `t.tolist()`, `torch.equal(a, b)`) is where the
// program may go one way or another by the data: what the traced run did
// behind it is that call's. Such an instruction is not movable, as a store
// is not: the code in front of it is a region, and the code behind it,
// which has the value as a Python object, is judged on its own.
//
// One read is kept inside a region: `t.item()` of a float tensor on the
// CPU, which a trace makes a value of its graph (a stand-in that the
// operators it reaches take as an input, see torch._rust_inductor). That
// holds while the traced run hands the float only to what the graph
// records. A run that needs it in Python (a comparison, a branch, `int()`,
// a format) says so when its trace is over: the instruction of the body
// that read it is marked, and the code is planned again with the mark.
//
// What it is given
//   - by the optimizer (mod.rs's pure_instr and observed_ops): an
//     instruction of a code as observed: the ops its visits ran, the callee
//     recorded for it, the kinds of its result;
//   - by the executor (run.rs): the code whose block or span is about to be
//     traced (begin), the code unit of each instruction of the body itself
//     before it runs (at), and the frame of a code that is over (again);
//   - by the trace (torch._rust_inductor): that a float was read from a
//     tensor as a value of the graph (_region_value_read), and, when the
//     trace is over, that the run needed such a float in Python
//     (_region_value_needed).
// What it guarantees
//   - seen(): Some(why) for an instruction that was observed to read a
//     tensor's value other than as the float above, or that is marked;
//   - _region_value_needed() is true if it marked an instruction that was
//     not marked before: again() is then true once for the code's Compiled
//     that ran the trace, which plans the code again when its frame is
//     over. No new mark, no new plan: a place is given up for a read at
//     most once per instruction of its code that reads.
// What it may not assume
//   - that the observed calls took every way to a read: a read the traced
//     run is the first to make is refused by the trace itself;
//   - that a mark is honoured (a pass list may declare the call pure): the
//     next trace that needs the value marks nothing new and is refused as
//     any other.

use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use pyo3::prelude::*;

use super::effects::Effect;
use super::types::{Kind, Ty};
use super::{Compiled, Prepared};
use crate::cfg::ir::{Site, ValueId};

const READS: &str = "reads a tensor's value";
const NEEDED: &str = "reads a tensor's value, which the traced run needed in Python";

// The instructions marked, by (code, code unit), and whether there are any.
static MARKED: Mutex<Vec<(usize, u32)>> = Mutex::new(Vec::new());
static ANY: AtomicBool = AtomicBool::new(false);
// The Compileds to plan again, by address.
static PLAN: Mutex<Vec<usize>> = Mutex::new(Vec::new());

thread_local! {
    // The code and the Compiled of the trace running on this thread, the
    // code unit of the body's own instruction that runs, and the units at
    // which a float was read as a value of the graph.
    static TRACE: Cell<(usize, usize)> = const { Cell::new((0, 0)) };
    static AT: Cell<u32> = const { Cell::new(0) };
    static READ: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
}

// The torch ops that hand a tensor's value to Python.
fn hands_value(op: &str) -> bool {
    matches!(op, "aten::equal" | "aten::allclose") || op.starts_with("aten::_local_scalar_dense") || op.starts_with("aten::is_nonzero") || op.starts_with("aten::item")
}

// Tensor.item, and the methods that hand out a tensor's values with no
// torch op, by address.
fn readers(py: Python<'_>) -> &'static (usize, Vec<usize>) {
    static READERS: OnceLock<(usize, Vec<usize>)> = OnceLock::new();
    READERS.get_or_init(|| {
        let of = |name: &str| py.import("torch").and_then(|m| m.getattr("Tensor")?.getattr(name)).map_or(0, |f| f.as_ptr() as usize);
        (of("item"), ["tolist", "numpy", "__array__"].into_iter().map(of).collect())
    })
}

// Why instr v of `prep`'s Cfg, an instruction of `code`, is where a region
// ends for a read of a tensor's value, if it is.
pub(super) fn seen(py: Python<'_>, prep: &Prepared, code: usize, v: ValueId) -> Option<String> {
    let ins = &prep.cfg[v];
    if ANY.load(Ordering::Relaxed) && MARKED.lock().unwrap().contains(&(code, ins.origin)) {
        return Some(NEEDED.into());
    }
    let callee = if super::is_call(&ins.op) { unsafe { &*prep.callees.get() }.get(&v).and_then(|c| c.get(py)).map_or(0, |f| f.as_ptr() as usize) } else { 0 };
    let (item, out) = readers(py);
    if callee != 0 && out.contains(&callee) {
        return Some(READS.into());
    }
    let ops = unsafe { &*prep.effects.get() }.get(Site::Instr(v)).iter().flat_map(|(seq, _)| seq);
    let mut reads = ops.filter_map(|e| match e {
        Effect::TorchOp { op, args } if hands_value(op) => Some((op, args)),
        _ => None,
    });
    let (op, first) = reads.next()?;
    // The float a trace makes a value of its graph (see _ScalarArgs).
    let float = matches!(unsafe { &*prep.types.get() }.get(v.idx()).and_then(|t| t.ty.as_ref()).and_then(Ty::kinds), Some([Kind::Float]));
    let kept = super::SCALAR_READS.load(Ordering::Relaxed) && callee == *item && float && !first.contains('@') && reads.all(|(_, args)| !args.contains('@'));
    (!kept).then(|| format!("ran {op}, which {READS}"))
}

// A block or a span of `code`, run by the Compiled at `compiled`, is about
// to be traced on this thread.
pub(super) fn begin(code: usize, compiled: usize) {
    TRACE.set((code, compiled));
    READ.with_borrow_mut(|r| r.clear());
}

// The instruction at code unit `unit` of the traced body itself is about to
// run.
pub(super) fn at(unit: u32) {
    AT.set(unit);
}

/// The body traced on this thread read a float from a tensor as a value of
/// the graph, in the instruction of its own that is running.
#[pyfunction]
pub fn _region_value_read() {
    READ.with_borrow_mut(|r| r.push(AT.get()));
}

/// The run traced on this thread needed in Python a float it read from a
/// tensor: the instructions of the body that read one are not movable from
/// now on. Whether one of them is newly so: the code is then planned again.
#[pyfunction]
pub fn _region_value_needed() -> bool {
    let (code, compiled) = TRACE.get();
    let units = READ.take();
    if code == 0 || units.is_empty() {
        return false;
    }
    let mut marked = MARKED.lock().unwrap();
    let before = marked.len();
    for unit in units {
        if !marked.contains(&(code, unit)) {
            marked.push((code, unit));
        }
    }
    if marked.len() == before {
        return false;
    }
    ANY.store(true, Ordering::Relaxed);
    PLAN.lock().unwrap().push(compiled);
    true
}

// Whether `c`, whose frame is over, is to be planned again (asked once).
pub(super) fn again(c: &Compiled) -> bool {
    if !ANY.load(Ordering::Relaxed) {
        return false;
    }
    let mut plan = PLAN.lock().unwrap();
    let n = plan.len();
    plan.retain(|p| *p != c as *const Compiled as usize);
    plan.len() != n
}
