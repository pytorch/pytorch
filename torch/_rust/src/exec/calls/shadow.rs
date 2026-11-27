// What the call-list recorder found (record.rs), kept for tests and tools,
// and the two things asked of a finished recording: whether the witness of
// an entry's call agrees with what the entry says, and whether the list,
// run on the recorded run's own operands, makes what the block made.
//
// What it is given
//   - by record.rs: one Shadow per recorded run, of plain texts and numbers
//     (keep); behind an entry's call, what its witness saw (disagreements);
//     behind a complete list of pure entries, the list, the run's inputs
//     and pins held weakly, and the block's outputs (validate);
//   - by run.rs, when today's engine traces a place: its key (traced).
// What it guarantees
//   - nothing of the program is kept: a Shadow holds no object;
//   - `torch._rust._calls_shadow()` gives this thread's Shadows as dicts,
//     the last recording of each place, and `_calls_shadow_missed()` how
//     many traces were of a place with no recording; with
//     TORCH_RUST_CALLS_DIR naming a directory the process leaves both there
//     as <pid>.json when the interpreter is finalized (no Python runs for
//     it);
//   - A STATUS IS FROM A CLOSED SET AND "replayable" IS POSITIVE: a Shadow
//     is "stopped" (the recording ended early), "refused" (the check of
//     lists refused it), "partial: <the first reason>" (it reached the end
//     and record.rs met at least one thing the list does not say), or
//     "replayable" (none of those). Nothing here decides a reason;
//   - validate() runs the list once, outside the executor, and compares bit
//     for bit (compare): each output must have the key of the block's, on
//     the CPU and contiguous its bytes, and the same memory relation: it
//     shares its storage with the same operands and earlier outputs as the
//     block's does, and where it shares an operand's it is the same memory.
//     It answers "not validated" for what it cannot compare; it never
//     answers "equal" for what it did not compare.

use std::cell::RefCell;

use pyo3::ffi::{self, PyObject};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::witness::Witnessed;
use super::List;
use crate::bindings::ffi as tffi;
use crate::bindings::ffi::op_hook::TensorIdent;
use crate::exec::pin::Pin;
use crate::exec::tensor_key::TensorKey;

#[derive(Default)]
pub struct Shadow {
    pub place: String,
    // The plan and block the run was of (run.rs's key of a place).
    pub key: (usize, u32),
    // Today's engine went on to trace the place.
    pub traced: bool,
    // Entries made (drops are not counted), instructions the run ran, and
    // how many of those ran behind the stop.
    pub entries: usize,
    pub seen: u64,
    pub left_behind: u64,
    // (the instruction, why): where the recording ended, if it did.
    pub stop: Option<(String, String)>,
    // The recording reached the end and the check of lists refused it.
    pub refused: Option<String>,
    pub text: String,
    // Tensors taken as pins with no check; slots whose release was not
    // seen (stated at the end of the body).
    pub loose: u32,
    pub late: u32,
    // Outputs that are results of entries; the block's other outputs.
    pub outputs: (usize, usize),
    // Outputs that hold results of entries and are none themselves.
    pub unstated: u32,
    pub by_value: Vec<(String, u64)>,
    pub left: Vec<(String, u64)>,
    pub handed: Vec<(String, u64)>,
    pub facts: u64,
    pub notes: Vec<String>,
    pub validated: String,
    // Why the recording, though it reached its end, is not the body: each
    // distinct reason with how often it was met, in the order met.
    pub reasons: Vec<(String, u64)>,
    // Conditional jumps taken under the body: on a known value, on any other.
    pub jumps: (u64, u64),
    // Python frames that ran under the body.
    pub frames: u64,
    // Later arrivals at the place the list was run on (again.rs): how many
    // gave the block's outputs, how many could not be compared, and the
    // first that differed.
    pub again: (u32, u32, Option<String>),
}

thread_local! {
    static KEPT: RefCell<Vec<Shadow>> = const { RefCell::new(Vec::new()) };
    // Traces of today's engine at a place no run was recorded of.
    static MISSED: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

// Every recording is kept, a stopped one too. Its index, for amend().
pub fn keep(shadow: Shadow) -> usize {
    KEPT.with_borrow_mut(|kept| {
        kept.push(shadow);
        kept.len() - 1
    })
}

pub fn amend(at: usize, f: impl FnOnce(&mut Shadow)) {
    KEPT.with_borrow_mut(|kept| kept.get_mut(at).map(f));
}

pub fn traced(key: (usize, u32)) {
    static AT_EXIT: std::sync::Once = std::sync::Once::new();
    AT_EXIT.call_once(|| unsafe {
        ffi::Py_AtExit(Some(dump));
    });
    KEPT.with_borrow_mut(|kept| {
        let mut found = false;
        for s in kept.iter_mut().filter(|s| s.key == key) {
            (s.traced, found) = (true, true);
        }
        MISSED.set(MISSED.get() + !found as u64);
    });
}

fn quoted(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' | '\\' => out.extend(['\\', c]),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out + "\""
}

impl Shadow {
    pub fn status(&self) -> String {
        match (&self.stop, &self.refused, self.reasons.first()) {
            (Some(_), ..) => "stopped".into(),
            (None, Some(_), _) => "refused".into(),
            (None, None, Some((why, _))) => format!("partial: {why}"),
            (None, None, None) => "replayable".into(),
        }
    }

    fn again_text(&self) -> String {
        match &self.again {
            (_, _, Some(different)) => different.clone(),
            (0, 0, None) => String::new(),
            (equal, skipped, None) => format!("equal on {equal} later arrival(s), {skipped} not comparable"),
        }
    }

    fn json(&self) -> String {
        let counts = |v: &[(String, u64)]| format!("{{{}}}", v.iter().map(|(k, n)| format!("{}: {n}", quoted(k))).collect::<Vec<_>>().join(", "));
        let stop = self.stop.as_ref().map_or("null".into(), |(at, why)| format!("[{}, {}]", quoted(at), quoted(why)));
        format!(
            "{{\"place\": {}, \"status\": {}, \"traced\": {}, \"entries\": {}, \"seen\": {}, \"left_behind\": {}, \"stop\": {stop}, \"refused\": {}, \"text\": {}, \"loose\": {}, \"late\": {}, \"outputs\": [{}, {}], \"unstated\": {}, \"by_value\": {}, \"left\": {}, \"handed\": {}, \"facts\": {}, \"notes\": [{}], \"validated\": {}, \"reasons\": {}, \"jumps\": [{}, {}], \"frames\": {}, \"again\": {}}}",
            quoted(&self.place), quoted(&self.status()), self.traced, self.entries, self.seen, self.left_behind, self.refused.as_deref().map_or("null".into(), quoted), quoted(&self.text),
            self.loose, self.late, self.outputs.0, self.outputs.1, self.unstated, counts(&self.by_value), counts(&self.left), counts(&self.handed), self.facts,
            self.notes.iter().map(|n| quoted(n)).collect::<Vec<_>>().join(", "), quoted(&self.validated), counts(&self.reasons), self.jumps.0, self.jumps.1, self.frames, quoted(&self.again_text())
        )
    }
}

// Called when the interpreter is gone: it touches the kept texts only.
extern "C" fn dump() {
    let Some(dir) = std::env::var_os("TORCH_RUST_CALLS_DIR") else { return };
    let _ = KEPT.try_with(|kept| {
        let text = format!("{{\"missed\": {}, \"places\": [{}]}}", MISSED.get(), kept.borrow().iter().map(|s| s.json()).collect::<Vec<_>>().join(",\n"));
        let _ = std::fs::write(std::path::Path::new(&dir).join(format!("{}.json", std::process::id())), text);
    });
}

/// What the call-list recorder found on this thread
/// (``TORCH_RUST_CALLS=shadow``; empty without it): a dict per place, the
/// recorded run of it. ``status``: "replayable" (``text`` is the list and the
/// recorder met nothing the list does not say), "partial: <first reason>"
/// (``text`` is a list that is not the body; ``reasons``: every reason with
/// its count), "stopped" (``stop`` = (instruction, why)) or "refused" (the
/// check of lists refused what was recorded: ``refused``). ``traced``: today's
/// engine went on to trace the place. ``validated``: "equal", "DIFFERENT: ..."
/// or "not validated: ...", on the recorded run's operands; ``again``: the same
/// on later arrivals at the place. ``loose``: tensors taken as pins with no
/// check. ``jumps``: conditional jumps under the body on a (known, other) value.
#[pyfunction]
pub fn _calls_shadow(py: Python<'_>) -> PyResult<Vec<Py<PyDict>>> {
    KEPT.with_borrow(|kept| {
        kept.iter()
            .map(|s| {
                let d = PyDict::new(py);
                d.set_item("place", &s.place)?;
                d.set_item("status", s.status())?;
                d.set_item("traced", s.traced)?;
                d.set_item("entries", s.entries)?;
                d.set_item("seen", s.seen)?;
                d.set_item("left_behind", s.left_behind)?;
                d.set_item("stop", s.stop.clone())?;
                d.set_item("refused", s.refused.clone())?;
                d.set_item("text", &s.text)?;
                d.set_item("loose", s.loose)?;
                d.set_item("late", s.late)?;
                d.set_item("outputs", s.outputs)?;
                d.set_item("unstated", s.unstated)?;
                d.set_item("by_value", s.by_value.clone())?;
                d.set_item("left", s.left.clone())?;
                d.set_item("handed", s.handed.clone())?;
                d.set_item("facts", s.facts)?;
                d.set_item("notes", s.notes.clone())?;
                d.set_item("validated", &s.validated)?;
                d.set_item("reasons", s.reasons.clone())?;
                d.set_item("jumps", s.jumps)?;
                d.set_item("frames", s.frames)?;
                d.set_item("again", s.again_text())?;
                Ok(d.unbind())
            })
            .collect()
    })
}

/// How many traces of today's engine on this thread were of a place that no
/// run was recorded of (``TORCH_RUST_CALLS=shadow``): the place was traced on
/// its first arrival, with no run before it.
#[pyfunction]
pub fn _calls_shadow_missed() -> u64 {
    MISSED.get()
}

// The tensors a call can reach through `o`, operand `position` of it
// (origins::reach, the one walk of containers).
pub unsafe fn tensors(o: *mut PyObject, position: usize, into: &mut Vec<(TensorIdent, usize)>) {
    unsafe {
        super::origins::reach(o, 0, &mut |t| {
            let mut id = TensorIdent::default();
            tffi::op_hook::tensor_ident(t as usize, &mut id);
            into.push((id, position));
        })
    }
}

// Where what the witness saw under an entry's call is not what the entry
// says: `ids`, the tensors among its operands with their positions;
// `writes`, the positions the entry says it writes. One line each.
pub fn disagreements(seen: &Witnessed, ids: &[(TensorIdent, usize)], writes: &[u32]) -> Vec<String> {
    let flags = seen.flags();
    if flags.opaque {
        return vec![format!("the witness could not judge the call: {}", seen.why)];
    }
    let operands: Vec<TensorIdent> = ids.iter().map(|(id, _)| *id).collect();
    let mut out = Vec::new();
    let mut written = Vec::new();
    for w in &seen.wrote {
        let op = seen.ops.get(w.op).map_or("?", |s| s.as_str());
        match w.operand(&operands).map(|k| ids[k].1 as u32) {
            Some(position) if writes.contains(&position) => written.push(position),
            Some(position) => out.push(format!("a declared write to operand {position} by {op}, where the entry says it writes none")),
            None if !w.made => out.push(format!("a declared write by {op} to memory that is no operand's")),
            None => {}
        }
    }
    out.extend(writes.iter().filter(|p| !written.contains(p)).map(|p| format!("the entry says it writes operand {p}, and no operator declared that write")));
    let said = [(flags.drew, "an operator that draws ran"), (flags.read_value, "an operator read a value out of a tensor"), (flags.data_sized, "an operator whose result is sized by data ran"), (flags.entered_python, "a Python frame was entered")];
    out.extend(said.iter().filter(|(on, _)| *on).map(|(_, what)| what.to_string()));
    out.sort();
    out.dedup();
    out
}

// Whether `a` and `b` are tensors of one key and, on the CPU and
// contiguous, of the same bytes. None: not comparable here.
unsafe fn same_tensor(a: *mut PyObject, b: *mut PyObject) -> Option<bool> {
    unsafe {
        let (ka, kb) = (TensorKey::of(a)?, TensorKey::of(b)?);
        if ka != kb {
            return Some(false);
        }
        let (ta, tb) = (tffi::tensor_by_type(a as usize).as_ref()?, tffi::tensor_by_type(b as usize).as_ref()?);
        let (mut sa, mut sb) = (tffi::TensorSpan::default(), tffi::TensorSpan::default());
        if !tffi::tensor_is_cpu(ta) || !tffi::tensor_is_contiguous(ta) || !tffi::tensor_span(ta, &mut sa) || !tffi::tensor_span(tb, &mut sb) {
            return None;
        }
        let n = sa.high - sa.low;
        Some(n == sb.high - sb.low && std::slice::from_raw_parts(sa.low as *const u8, n) == std::slice::from_raw_parts(sb.low as *const u8, n))
    }
}

// The storage of the tensor `o` by address, 0 if it has none.
pub unsafe fn storage(o: *mut PyObject) -> usize {
    let mut id = TensorIdent::default();
    tffi::op_hook::tensor_ident(o as usize, &mut id);
    id.storage
}

// What `o` shares its storage with: the first of `operands` (storages), and
// the first of `earlier`; and where in the storage its memory starts.
unsafe fn relation(o: *mut PyObject, operands: &[usize], earlier: &[*mut PyObject]) -> (Option<usize>, Option<usize>, usize) {
    unsafe {
        let mine = storage(o);
        let mut span = tffi::TensorSpan::default();
        let at = match tffi::tensor_by_type(o as usize).as_ref() {
            Some(t) if tffi::tensor_span(t, &mut span) => span.at,
            _ => 0,
        };
        (operands.iter().position(|s| *s != 0 && *s == mine), earlier.iter().position(|e| mine != 0 && storage(*e) == mine), at)
    }
}

// The outputs the list made (`made`, its tuple) against the block's (`outs`,
// borrowed), in order; `operands`: the storages of the inputs and pins both
// ran on.
pub unsafe fn compare(made: *mut PyObject, outs: &[*mut PyObject], operands: &[usize]) -> String {
    unsafe {
        let mine: Vec<*mut PyObject> = (0..outs.len()).map(|i| ffi::PyTuple_GET_ITEM(made, i as ffi::Py_ssize_t)).collect();
        let mut verdict = String::from("equal");
        for (i, want) in outs.iter().enumerate() {
            match same_tensor(mine[i], *want) {
                Some(true) => {}
                Some(false) => return format!("DIFFERENT: output {i} of {}", outs.len()),
                None => verdict = "not validated: an output is not a contiguous tensor on the CPU".into(),
            }
            let (a, b) = (relation(mine[i], operands, &mine[..i]), relation(*want, operands, &outs[..i]));
            if (a.0, a.1) != (b.0, b.1) || a.0.is_some() && a.2 != b.2 {
                return format!("DIFFERENT: output {i} of {} shares memory with {}, the block's with {}", outs.len(), shared(a), shared(b));
            }
        }
        verdict
    }
}

fn shared(r: (Option<usize>, Option<usize>, usize)) -> String {
    match (r.0, r.1) {
        (Some(k), _) => format!("operand {k}"),
        (None, Some(j)) => format!("output {j}"),
        (None, None) => "nothing".into(),
    }
}

// Runs `list` on the recorded run's inputs and pins and compares its
// outputs with `outs`, the block's (borrowed), in order.
pub unsafe fn validate(py: Python<'_>, list: &List, inputs: &[Pin], pins: &[Pin], outs: &[*mut PyObject]) -> String {
    unsafe {
        let live = |held: &[Pin]| held.iter().map(|p| p.live().filter(|o| !o.is_null()).map(|o| Py::<PyAny>::from_borrowed_ptr(py, o))).collect::<Option<Vec<_>>>();
        let (Some(inputs), Some(pins)) = (live(inputs), live(pins)) else { return "not validated: an input or a pin of the run did not live to its end".into() };
        let pointers = |held: &[Py<PyAny>]| held.iter().map(|o| o.as_ptr()).collect::<Vec<_>>();
        let made = match crate::exec::outside_executor(|| list.run(py, &pointers(&inputs), &pointers(&pins))) {
            Ok(made) => made,
            Err(stop) => {
                ffi::PyErr_Clear();
                return format!("DIFFERENT: the list stopped at entry {} ({})", stop.entry, stop.what.text());
            }
        };
        let operands: Vec<usize> = inputs.iter().chain(&pins).map(|o| storage(o.as_ptr())).collect();
        let verdict = compare(made, outs, &operands);
        ffi::Py_DECREF(made);
        verdict
    }
}

/// For tests: what the recorder's validation answers for the hand-built
/// ``list`` run on ``inputs`` and ``pins`` when the block made ``outs``.
#[pyfunction]
pub fn _calls_validate(list: &Bound<'_, List>, inputs: Vec<Py<PyAny>>, pins: Vec<Py<PyAny>>, outs: Vec<Py<PyAny>>) -> String {
    unsafe {
        let weak = |held: &[Py<PyAny>]| held.iter().map(|o| Pin::weak(o.as_ptr())).collect::<Vec<_>>();
        let outs: Vec<*mut PyObject> = outs.iter().map(|o| o.as_ptr()).collect();
        validate(list.py(), list.get(), &weak(&inputs), &weak(&pins), &outs)
    }
}
