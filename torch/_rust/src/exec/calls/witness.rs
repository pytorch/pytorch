// The witness: what the dispatcher and the eval-frame hook saw on this thread
// while one C call of a traced body ran. A tripwire for the recorder, never
// its source: `begin()` before the call, `end()` behind it, and the caller
// refuses what `Witnessed::licenses` does not grant. Meant to be on only
// while a trace records. Nothing here calls the program (the tests' entry,
// `_witness`, is in run.rs for that reason).
//
// What a program can see of an open witness, and nothing else: the
// PreDispatch key is in the thread's include set
// (`torch._C._dispatch_tls_is_dispatch_key_included("PreDispatch")` is True).
// No mode is pushed, no kernel asks about that key, and every operator runs
// the kernel it runs without the witness (csrc/op_hook.cpp continues it boxed
// at the keys below PreDispatch).
//
// It sees: every aten operator dispatched on this thread, at any depth (the
// key stays included below a reported operator); what each schema says it
// writes; the tags nondeterministic_seeded and data_dependent_output; any
// Python frame entered (exec::eval_frame). It cannot see, so `false` is no
// proof: a write no schema declares (batch norm's running statistics), a
// write to memory no operand has, a read that is no operator (`tolist`,
// `numpy`, a data pointer read in a kernel), a draw by an untagged operator,
// operators on other threads. The draw tag goes by names (torchgen/model.py),
// so `drew` is also true for a dropout that is off.
//
// THE DEFAULT: what cannot be read or classified sets `opaque` (an operator
// outside aten, one without a readable schema, a result that is no tensor
// from an operator on neither list below, a witness that could not see when
// it began or ended: `no_dispatch()`, no eval-frame hook), and an opaque
// result licenses nothing.

use std::cell::{Cell, RefCell};

use pyo3::prelude::*;

use crate::bindings::ffi::op_hook::{op_hook_depth, op_hook_seeing, op_hook_witness, TensorIdent, WitnessWrite};

// As csrc/op_hook.cpp sets them.
const UNREADABLE: u32 = 1;
const DRAWS: u32 = 2;
const READS_VALUE: u32 = 4;
const VALUE_OUT: u32 = 8;

// Operators without the data_dependent_output tag whose result is computed
// from tensor data: is_nonzero (prim_native_functions.cpp: `self.item()`),
// _choose_qparams_per_tensor (quantized/QTensor.cpp: min and max),
// _use_cudnn_ctc_loss (cudnn/LossCTC.cpp: the lengths' elements).
const READS_BY_NAME: [&str; 3] = ["aten::is_nonzero", "aten::_choose_qparams_per_tensor", "aten::_use_cudnn_ctc_loss"];

// Operators with a result that is no tensor and holds no tensor data: it
// follows from sizes, strides, offset, dtype, device, the conj / neg /
// inference / zero-tensor bits, `grad_fn is None`, operands that are no
// tensors, and process settings (TensorProperties.cpp, TypeProperties.cpp,
// TensorImpl's accessors; _batch_norm_impl_index's int is the backend picked,
// Normalization.cpp). An operator on neither list is opaque.
const NO_DATA: [&str; 26] = [
    "aten::dim", "aten::numel", "aten::size", "aten::stride", "aten::storage_offset", "aten::sym_size", "aten::sym_stride",
    "aten::sym_numel", "aten::sym_storage_offset", "aten::sym_is_contiguous", "aten::is_contiguous", "aten::is_same_size",
    "aten::is_complex", "aten::is_floating_point", "aten::is_signed", "aten::is_conj", "aten::is_neg", "aten::is_inference",
    "aten::_is_zerotensor", "aten::is_leaf", "aten::get_device", "aten::result_type", "aten::can_cast", "aten::promote_types",
    "aten::_batch_norm_impl_index", "aten::cudnn_is_acceptable",
]; // fmt: skip

// One argument an operator's schema says it writes. `id` holds addresses
// (the TensorImpl, the storage): they name an operand of the C call only
// while that call's operands are alive. Map them at once (`operand`), never
// keep them.
pub struct Write {
    // Index into `ops` of the top-level operator that is the writer or
    // (`nested`) runs it; the writer's own argument.
    pub op: usize,
    pub arg: usize,
    pub nested: bool,
    pub id: TensorIdent,
}

impl Write {
    // The operand written: the same tensor, else one on the same storage (a
    // view made inside the call). None: memory no operand has.
    pub fn operand(&self, operands: &[TensorIdent]) -> Option<usize> {
        let same = operands.iter().position(|o| o.tensor != 0 && o.tensor == self.id.tensor);
        same.or_else(|| operands.iter().position(|o| o.storage != 0 && o.storage == self.id.storage))
    }
}

// What a rule lets one call do; `writes` are positions among its operands.
#[derive(Default)]
pub struct Allowed<'a> {
    pub writes: &'a [usize],
    pub draws: bool,
    pub reads_value: bool,
    pub python: bool,
}

#[derive(Default)]
pub struct Witnessed {
    // Top-level operators, "aten::add.Tensor", in order.
    pub ops: Vec<String>,
    // Operators run below those.
    pub below: u32,
    pub wrote: Vec<Write>,
    // The first thing that made it opaque, for a report.
    pub why: String,
    drew: bool,
    // By a top-level operator (its value is the call's to hand on), by one
    // below (the value stays in C: an error check, a size).
    read_top: bool,
    read_below: bool,
    entered_python: bool,
    opaque: bool,
}

impl Witnessed {
    // (opaque; an operator tagged nondeterministic_seeded ran, at any depth;
    // one that hands a value out of a tensor ran, at any depth; one of those
    // was top-level; a Python frame was entered). Opaque is all of them.
    pub fn flags(&self) -> (bool, bool, bool, bool, bool) {
        let o = self.opaque;
        (o, self.drew || o, self.read_top || self.read_below || o, self.read_top || o, self.entered_python || o)
    }
    fn refuse(&mut self, why: impl FnOnce() -> String) {
        if !self.opaque {
            (self.opaque, self.why) = (true, why());
        }
    }
    // Whether nothing was seen but what `allowed` names. A write to memory no
    // operand has needs no grant.
    pub fn licenses(&self, operands: &[TensorIdent], allowed: &Allowed) -> bool {
        !self.opaque
            && (allowed.draws || !self.drew)
            && (allowed.reads_value || !(self.read_top || self.read_below))
            && (allowed.python || !self.entered_python)
            && self.wrote.iter().all(|w| w.operand(operands).is_none_or(|k| allowed.writes.contains(&k)))
    }
}

struct Open {
    seen: Witnessed,
    // op_hook_depth() at begin: an operator at this depth is top-level.
    base: u32,
    paused: bool,
}

thread_local! {
    static STACK: RefCell<Vec<Open>> = const { RefCell::new(Vec::new()) };
    // Open witnesses that are not paused.
    static LIVE: Cell<usize> = const { Cell::new(0) };
}

// Tells the hook whether anybody collects; false if it then cannot see.
fn sync() -> bool {
    let live = STACK.with_borrow(|s| s.iter().filter(|o| !o.paused).count());
    LIVE.set(live);
    let seeing = op_hook_witness(live > 0);
    seeing && crate::exec::frames_hooked()
}

// Opens a witness on this thread, inside any already open: each open one
// that is not paused collects. Needs the GIL, as end / pause / resume do.
pub fn begin() {
    STACK.with_borrow_mut(|s| s.push(Open { seen: Witnessed::default(), base: op_hook_depth(), paused: false }));
    if !sync() {
        blind("begin");
    }
}

// The innermost witness could not see operators or frames at `at`.
fn blind(at: &str) {
    STACK.with_borrow_mut(|s| s.last_mut().map(|o| o.seen.refuse(|| format!("could not see at its {at}"))));
}

// Closes the innermost witness. Without one, the result is opaque.
pub fn end() -> Witnessed {
    if !(op_hook_seeing() && crate::exec::frames_hooked()) && STACK.with_borrow(|s| s.last().is_some_and(|o| !o.paused)) {
        blind("end");
    }
    let open = STACK.with_borrow_mut(|s| s.pop());
    sync();
    let mut seen = open.map(|o| o.seen).unwrap_or_else(|| Witnessed { opaque: true, ..Witnessed::default() });
    seen.why = if seen.opaque && seen.why.is_empty() { "end without begin".into() } else { seen.why };
    seen
}

// The innermost witness collects nothing from here (a stop of its recorder)
// until `resume`. Those around it go on.
pub fn pause() {
    STACK.with_borrow_mut(|s| s.last_mut().map(|o| o.paused = true));
    sync();
}

pub fn resume() {
    STACK.with_borrow_mut(|s| s.last_mut().map(|o| o.paused = false));
    if !sync() {
        blind("resume");
    }
}

// From exec::eval_frame, for every Python frame entered on this thread.
pub fn frame_entered() {
    if LIVE.get() != 0 {
        STACK.with_borrow_mut(|s| s.iter_mut().filter(|o| !o.paused).for_each(|o| o.seen.entered_python = true));
    }
}

// From the PreDispatch hook, maybe without the GIL: an operator starts,
// `depth` reported operators above it.
pub fn on_witness_op(name: &str, overload: &str, depth: u32, flags: u32, wrote: &[WitnessWrite]) {
    let reads = flags & READS_VALUE != 0 || READS_BY_NAME.contains(&name);
    let why = if flags & UNREADABLE != 0 {
        Some("no schema the witness can read")
    } else if !name.starts_with("aten::") {
        Some("no aten operator: its tags say nothing")
    } else if flags & VALUE_OUT != 0 && !reads && !NO_DATA.contains(&name) {
        Some("a result that is no tensor, on neither list")
    } else {
        None
    };
    STACK.with_borrow_mut(|s| {
        for open in s.iter_mut().filter(|o| !o.paused) {
            let (top, seen) = (depth <= open.base, &mut open.seen);
            if top {
                seen.ops.push(if overload.is_empty() { name.to_string() } else { format!("{name}.{overload}") });
            } else {
                seen.below += 1;
            }
            let op = seen.ops.len().saturating_sub(1);
            seen.wrote.extend(wrote.iter().map(|w| Write { op, arg: w.arg as usize, nested: !top, id: w.id }));
            seen.drew |= flags & DRAWS != 0;
            seen.read_top |= reads && top;
            seen.read_below |= reads && !top;
            if let Some(why) = why {
                seen.refuse(|| format!("{name}: {why}"));
            }
        }
    });
}

/// For tests: the operators classified by name, (reads a value, result holds
/// no tensor data).
#[pyfunction]
pub fn _witness_names() -> (Vec<&'static str>, Vec<&'static str>) {
    (READS_BY_NAME.to_vec(), NO_DATA.to_vec())
}

/// For tests: `pause()` (True) or `resume()` from a callback.
#[pyfunction]
pub fn _witness_pause(paused: bool) {
    if paused { pause() } else { resume() }
}
