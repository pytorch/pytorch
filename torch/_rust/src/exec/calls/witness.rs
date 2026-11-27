// The witness: what the dispatcher and the eval-frame hook saw on this thread
// while one C call of a traced body ran. A tripwire for the recorder, never
// its source: `begin()` before the call, `end()` behind it, and the caller
// refuses what `Witnessed::licenses` does not grant. Meant to be on only
// while a trace records. Nothing here calls the program (the tests' entry,
// `_witness`, is in run.rs for that reason).
//
// It can NEVER certify that a call changed nothing: what is no aten operator
// (`requires_grad_`, a generator seeded, `.grad` attached) and what no schema
// declares never reach it. The license is the rule's; this only takes away.
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
// writes; the tags nondeterministic_seeded, data_dependent_output and
// dynamic_output_shape; any Python frame entered (exec::eval_frame). It
// cannot see, so `false` is no proof: a write no schema declares (batch
// norm's running statistics), a read that is no operator (`tolist`, `numpy`,
// a data pointer read in a kernel), a draw by an untagged operator, operators
// on other threads (the autograd engine runs the backward of a CUDA graph on
// a thread of its own: nothing of it is seen). The tags go by names
// (torchgen/model.py), so `drew` is also true for a dropout that is off and
// `data_sized` for an index by integers.
//
// A declared write is to an operand of the call (the caller's list, by
// tensor, else by storage: a view made inside the call), or to a storage an
// operator MADE while this witness was open (it returned a new tensor on a
// storage none of its arguments held, and nothing wrote that storage before
// the operator began; csrc/op_hook.cpp), or it is a write outside, which no
// rule can allow. Memory an operator made needs no license.
//
// THE DEFAULT: what cannot be read or classified sets `opaque` (an operator
// outside aten, one without a readable schema, a result that is no tensor
// from an operator on neither list below, a witness found blind at any of
// its calls or at a Python frame: `no_dispatch()`, a program that clears the
// key, no eval-frame hook), `opaque` is sticky, and an opaque result licenses
// nothing.
//
// KNOWN HOLE: the key taken away and put back with no call of the witness and
// no Python frame entered in between (one C callee doing both, or C calls of
// one frame: `_dispatch_tls_set_dispatch_key_included`, `_DisableTorchDispatch`
// entered and left) hides the operators in between. Nothing short of a change
// to c10 or to those bindings shows it.

use std::cell::{Cell, RefCell};

use pyo3::prelude::*;

use crate::bindings::ffi::op_hook::{
    op_hook_depth, op_hook_innermost, op_hook_seeing, op_hook_serial, op_hook_witness, TensorIdent, WitnessFlag, WitnessWrite,
};

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
    // Its storage was made by an operator while this witness was open.
    pub made: bool,
    pub id: TensorIdent,
    // The writer's serial (csrc/op_hook.h).
    serial: u64,
}

impl Write {
    // The operand written: the same tensor, else one on the same storage (a
    // view made inside the call). None: no operand has that memory; it is
    // `made`, or the write is outside.
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
    pub data_sized: bool,
    pub python: bool,
}

// What `Witnessed::flags` answers. `opaque` is folded into each of the others.
pub struct Flags {
    pub opaque: bool,
    // An operator tagged nondeterministic_seeded ran, at any depth.
    pub drew: bool,
    // One that hands a value out of a tensor ran, at any depth; one of those
    // was top-level (its value is the call's to hand on).
    pub read_value: bool,
    pub read_value_top: bool,
    // One tagged dynamic_output_shape ran, at any depth.
    pub data_sized: bool,
    pub entered_python: bool,
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
    // By a top-level operator, by one below (the value stays in C: an error
    // check, a size).
    read_top: bool,
    read_below: bool,
    data_sized: bool,
    entered_python: bool,
    opaque: bool,
}

impl Witnessed {
    pub fn flags(&self) -> Flags {
        let o = self.opaque;
        Flags {
            opaque: o,
            drew: self.drew || o,
            read_value: self.read_top || self.read_below || o,
            read_value_top: self.read_top || o,
            data_sized: self.data_sized || o,
            entered_python: self.entered_python || o,
        }
    }
    fn refuse(&mut self, why: impl FnOnce() -> String) {
        if !self.opaque {
            (self.opaque, self.why) = (true, why());
        }
    }
    // A declared write to memory that is no operand's and that no operator
    // made under this witness. `licenses` never grants it.
    pub fn wrote_outside(&self, operands: &[TensorIdent]) -> bool {
        self.opaque || self.wrote.iter().any(|w| !w.made && w.operand(operands).is_none())
    }
    // Whether nothing was seen but what `allowed` names. A write to memory an
    // operator made under this witness needs no grant.
    pub fn licenses(&self, operands: &[TensorIdent], allowed: &Allowed) -> bool {
        !self.opaque
            && (allowed.draws || !self.drew)
            && (allowed.reads_value || !(self.read_top || self.read_below))
            && (allowed.data_sized || !self.data_sized)
            && (allowed.python || !self.entered_python)
            && self.wrote.iter().all(|w| w.operand(operands).map_or(w.made, |k| allowed.writes.contains(&k)))
    }
}

struct Open {
    seen: Witnessed,
    // op_hook_depth() at begin: an operator at this depth is top-level.
    base: u32,
    // op_hook_serial() at begin, and at the pause it is in.
    began: u64,
    paused: bool,
    paused_at: u64,
}

thread_local! {
    static STACK: RefCell<Vec<Open>> = const { RefCell::new(Vec::new()) };
    // Open witnesses that are not paused.
    static LIVE: Cell<usize> = const { Cell::new(0) };
}

// Every witness that collects could not see operators or frames at `at`.
fn blind(at: &str) {
    STACK.with_borrow_mut(|s| s.iter_mut().filter(|o| !o.paused).for_each(|o| o.seen.refuse(|| format!("could not see at {at}"))));
}

// Before anything of the witness changes: a program that took the key (or
// the eval-frame hook) away since the last look blinded those that collect,
// and stays found out when the key comes back.
fn look(at: &str) {
    if LIVE.get() != 0 && !(op_hook_seeing() && crate::exec::frames_hooked()) {
        blind(at);
    }
}

// Tells the hook whether anybody collects; false if it then cannot see.
fn sync() -> bool {
    let (live, open) = STACK.with_borrow(|s| (s.iter().filter(|o| !o.paused).count(), !s.is_empty()));
    LIVE.set(live);
    let seeing = op_hook_witness(live > 0, open);
    seeing && crate::exec::frames_hooked()
}

// Opens a witness on this thread, inside any already open: each open one
// that is not paused collects. Needs the GIL, as end / pause / resume do.
pub fn begin() {
    look("a begin");
    let open = Open { seen: Witnessed::default(), base: op_hook_depth(), began: op_hook_serial(), paused: false, paused_at: 0 };
    STACK.with_borrow_mut(|s| s.push(open));
    if !sync() {
        blind("a begin");
    }
}

// Closes the innermost witness. Without one, the result is opaque.
pub fn end() -> Witnessed {
    look("an end");
    let open = STACK.with_borrow_mut(|s| s.pop());
    sync();
    let mut seen = open.map(|o| o.seen).unwrap_or_else(|| Witnessed { opaque: true, ..Witnessed::default() });
    seen.why = if seen.opaque && seen.why.is_empty() { "end without begin".into() } else { seen.why };
    seen
}

// The innermost witness collects nothing from here (a stop of its recorder)
// until `resume`. Those around it go on.
pub fn pause() {
    look("a pause");
    let at = op_hook_serial();
    STACK.with_borrow_mut(|s| {
        if let Some(o) = s.last_mut().filter(|o| !o.paused) {
            (o.paused, o.paused_at) = (true, at);
        }
    });
    sync();
}

pub fn resume() {
    look("a resume");
    let innermost = op_hook_innermost();
    STACK.with_borrow_mut(|s| {
        if let Some(o) = s.last_mut().filter(|o| o.paused) {
            o.paused = false;
            // What runs below that operator would be taken for top-level, or
            // for a part of the last operator this witness saw.
            if innermost > o.paused_at {
                o.seen.refuse(|| "resumed inside an operator that began while it was paused".into());
            }
        }
    });
    if !sync() {
        blind("a resume");
    }
}

// From exec::eval_frame, for every Python frame entered on this thread.
pub fn frame_entered() {
    if LIVE.get() != 0 {
        let seeing = op_hook_seeing();
        STACK.with_borrow_mut(|s| {
            for o in s.iter_mut().filter(|o| !o.paused) {
                o.seen.entered_python = true;
                if !seeing {
                    o.seen.refuse(|| "could not see at a Python frame".into());
                }
            }
        });
    }
}

// From the PreDispatch hook, maybe without the GIL: an operator starts,
// `depth` reported operators above it.
pub fn on_witness_op(name: &str, overload: &str, depth: u32, serial: u64, flags: u32, wrote: &[WitnessWrite]) {
    let has = |flag: WitnessFlag| flags & flag.repr != 0;
    let reads = has(WitnessFlag::ReadsValue) || READS_BY_NAME.contains(&name);
    let why = if has(WitnessFlag::Unreadable) {
        Some("no schema the witness can read")
    } else if !name.starts_with("aten::") {
        Some("no aten operator: its tags say nothing")
    } else if has(WitnessFlag::ValueOut) && !reads && !NO_DATA.contains(&name) {
        Some("a result that is no tensor, on neither list")
    } else {
        None
    };
    STACK.with_borrow_mut(|s| {
        for open in s.iter_mut().filter(|o| !o.paused) {
            let (top, began, seen) = (depth <= open.base, open.began, &mut open.seen);
            if top {
                seen.ops.push(if overload.is_empty() { name.to_string() } else { format!("{name}.{overload}") });
            } else {
                seen.below += 1;
            }
            // A nested operator is a part of the top-level one in flight.
            let Some(op) = seen.ops.len().checked_sub(1) else {
                seen.refuse(|| format!("{name}: below an operator the witness did not see begin"));
                continue;
            };
            let write = |w: &WitnessWrite| Write { op, arg: w.arg as usize, nested: !top, made: w.made > began, id: w.id, serial };
            seen.wrote.extend(wrote.iter().map(write));
            seen.drew |= has(WitnessFlag::Draws);
            seen.read_top |= reads && top;
            seen.read_below |= reads && !top;
            seen.data_sized |= has(WitnessFlag::DataSized);
            if let Some(why) = why {
                seen.refuse(|| format!("{name}: {why}"));
            }
        }
    });
}

// From the PreDispatch hook: the operator `by` returned a new tensor on
// `storage`, which none of its arguments held; the operators from `since` on,
// all inside it, that wrote that storage wrote what it made.
pub fn on_witness_made(storage: usize, since: u64, by: u64) {
    STACK.with_borrow_mut(|s| {
        for open in s.iter_mut().filter(|o| !o.paused && by > o.began) {
            let inside = open.seen.wrote.iter_mut().rev().take_while(|w| w.serial >= since);
            inside.filter(|w| w.id.storage == storage).for_each(|w| w.made = true);
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
