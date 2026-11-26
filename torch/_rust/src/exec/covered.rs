// What the optimizer's verdict covered, asked again while a body is traced.
//
// The optimizer judges code by what it saw it do: a function by the blocks
// its observed calls entered, a call by the callee its visits called. A
// region's trace is one more run, and the first that may leave what was
// seen: take a branch no call took, call another function. What that code
// does besides torch ops (a store, a counter that C code advances) would be
// done once, by the trace, and never by the region.
//
// What it is given
//   - by the optimizer (mod.rs's pure_compiled): the instructions of a code
//     it judged to only compute, by their code units, and for a call the
//     callee its visits called (judged);
//   - by the executor (run.rs), before a place is traced: how many traces
//     of that place were refused here and left it open (place); and for
//     each instruction that a traced body or a frame under it is about to
//     run: its code and code unit, and for a call the callable and the
//     callee recorded there (step);
//   - by the frame hook (mod.rs's eval_frame): the frames of a code that is
//     observed again, as each starts (observes) and is over (observed).
// What it guarantees
//   - step() is Err, with the reason for torch.vm.regions, before a frame
//     under the body runs an instruction that no verdict covered, and
//     before a call of a callable that is neither one seen there (by the
//     frame's own records, or by a verdict on any copy of the code: the
//     frame a trace runs is often a call site's copy that no call ran), nor
//     judged where it runs (a Python function: by its instructions), nor
//     known to only compute (the engine's tables, a class whose objects
//     Python code makes);
//   - OBSERVED AGAIN. Code a trace is refused for is not lost for good. The
//     refusal marks the code (per code, whichever frame comes next: a call
//     site's clone or the code's own): its next frames, which the place's
//     run as written enters, run the plan that records (also code that is
//     left to CPython otherwise), each is judged when it is over, and what
//     a verdict covers then is covered for the next trace. This until a
//     frame has run the instruction the trace was refused for, FRAMES
//     frames at most. If that frame did more than compute, the code keeps
//     no place open any more;
//   - THE BOUND. later() says whether the place is left open ("not yet" in
//     torch.vm.regions, with which trace of LATER it was): the first
//     LATER - 1 traces of a place that are refused here, unless the code
//     is known to do more than compute. The place's owner counts;
//   - TORCH'S HAND-OVER TO A FUNCTION MODE is the trace's machinery, not
//     the program's: a block of torch's own library that no call entered,
//     whose only call is of `handle_torch_function`, and that writes
//     nothing outside its frame, runs because the trace's own mode is on
//     (`has_torch_function` is true for it alone). It is covered as it is
//     (mod.rs's hand_over): the mode calls the function again with itself
//     off, and those frames are asked about as any other;
//   - A TENSOR'S HASH, which C code runs for a key, is taken as the
//     optimizer takes it: by the rule of the instruction that needed it;
//   - OPERANDS OF A KIND NO VERDICT SAW. Before an instruction that takes
//     the items of an operand, or changes it in place (kinds.rs says which),
//     step() asks whether a verdict saw an operand of that class there, and
//     treats one it did not as code no call ran: the verdict is asked
//     again (of a frame under the body: on its code; of the body's own
//     instruction: as the optimizer judges it, on what was observed until
//     now), and unless it has seen the class by now and still passes the
//     instruction, the trace is refused before the instruction runs and
//     the code is observed again, within the same bound;
//   - FILLED CONTAINERS. filled() names, by address, every container that
//     an instruction of the traced body or of a frame under it was about
//     to fill in place since place() (a display or a call's arguments
//     under construction: kinds::filled says which instructions), whatever
//     step() answered. One that an earlier block made is among the trace's
//     args: a trace that is given up gives it back what it held (region.rs's
//     _region_given_up), as its code then fills it again;
//   - step() is Ok(true) for a call the optimizer takes at another's word
//     without looking inside (a callable the pass list calls pure, a call a
//     cfg pass made movable, a torch function another tracer keeps whole):
//     the caller holds a Trusted while it runs, and nothing under it is
//     asked about.
// What it may not assume
//   - that a covered instruction does the same in another frame: a verdict
//     is for what its frame owns (an __init__ on the object just made), and
//     what is kept is per code, not per verdict;
//   - that the operands an instruction only reads are of the kinds seen
//     (kinds.rs asks about those it takes from or changes);
//   - that a container filled() names was there before the trace: most are
//     the run's own, and an address may be that of one already gone (the
//     tracer looks only for the args it holds itself);
//   - that the run as written takes the way the trace took (state may have
//     changed in between): then nothing new is observed, and the bound is
//     what ends it.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use pyo3::ffi;
use pyo3::ffi::PyObject;
use pyo3::prelude::*;

use crate::cfg::ir::Instr;

use super::ffi::*;
use super::{Compiled, effects, kinds, shown};

// The instructions of a code that some verdict covered, by code unit. Kept
// in the code's own Compiled, for its clones too.
#[derive(Default)]
pub struct Units {
    seen: Mutex<Vec<bool>>,
    // The callables a verdict saw called at a code unit.
    called: Mutex<HashMap<u32, Vec<Py<PyAny>>>>,
    // The kinds of operands a verdict saw at a code unit.
    shown: kinds::Shown,
    // While frames of the code are observed again: one more than the code
    // unit a trace was refused for (0: none), and how many more frames.
    wanted: AtomicU32,
    frames: AtomicU32,
    // A frame observed again ran that instruction and did more than
    // compute: no place is left open for this code any more.
    more: AtomicBool,
}

impl Drop for Units {
    fn drop(&mut self) {
        if *self.wanted.get_mut() != 0 {
            WANTED.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

// How many traces of one place may be refused for code no call ran before
// the place is closed.
pub const LATER: u32 = 3;
// How many frames of a code are observed again for one refusal.
const FRAMES: u32 = 8;
// How many codes are being observed again (the frame hook asks at every
// frame).
static WANTED: AtomicU32 = AtomicU32::new(0);

// The code's own Compiled (`c` itself unless it is a clone), made if it is
// not there yet: the clones of a code come and go with their callers' plans.
unsafe fn own(c: &Compiled) -> &Compiled {
    if c.depth == 0 {
        return c;
    }
    match super::compiled(c.code as *mut PyObject) {
        super::Lookup::Compiled(o) => o,
        _ => {
            ffi::PyErr_Clear();
            c
        }
    }
}

// The optimizer judged the instructions at `units` of c's code to only
// compute, each with the kinds of its operands, if it is one kinds.rs asks
// about, and the callee its visits called, if it is a call that recorded
// one. Kept with `c` as well: its code's own Compiled may not be there yet.
pub fn judged(c: &Compiled, units: impl Iterator<Item = (u32, Option<kinds::Seen>, Option<Py<PyAny>>)>) {
    let py = unsafe { Python::assume_attached() };
    let units: Vec<(u32, Option<kinds::Seen>, Option<Py<PyAny>>)> = units.collect();
    let o = unsafe { own(c) };
    for at in [Some(c), (!std::ptr::eq(o, c)).then_some(o)].into_iter().flatten() {
        let mut seen = at.covered.seen.lock().unwrap();
        let mut called = at.covered.called.lock().unwrap();
        for (u, operands, callee) in &units {
            if seen.len() <= *u as usize {
                seen.resize(*u as usize + 1, false);
            }
            seen[*u as usize] = true;
            if let Some(operands) = operands {
                at.covered.shown.add(*u, operands.clone());
            }
            if let Some(f) = callee {
                let known = called.entry(*u).or_default();
                if !known.iter().any(|k| k.as_ptr() == f.as_ptr()) {
                    known.push(f.clone_ref(py));
                }
            }
        }
    }
}

// The optimizer's verdict on one instruction of the code `c` runs passes it
// with operands of these kinds; the instruction is the body's own, so what
// the frames under a body may run of the code is as it was.
pub fn shown(c: &Compiled, unit: u32, operands: kinds::Seen) {
    c.covered.shown.add(unit, operands);
}

// Whether a verdict saw `g` called at code unit `unit` of c's code.
fn called(c: &Compiled, unit: u32, g: *mut PyObject) -> bool {
    let at = |c: &Compiled| c.covered.called.lock().unwrap().get(&unit).is_some_and(|k| k.iter().any(|f| f.as_ptr() == g));
    at(c) || at(unsafe { own(c) })
}

thread_local! {
    // How many calls taken at another's word are running.
    static TRUSTED: Cell<u32> = const { Cell::new(0) };
    // Whether another tracer keeps a function whole, by its address (those
    // asked about are module-level functions of torch).
    static WHOLE: RefCell<HashMap<usize, bool>> = RefCell::new(HashMap::new());
    // Whether the last refusal of step() on this thread leaves its place
    // open (see later).
    static AGAIN: Cell<bool> = const { Cell::new(false) };
    // How many traces of the place being traced were refused and left it
    // open before.
    static TRIED: Cell<u32> = const { Cell::new(0) };
    // The containers that instructions of the trace on this thread filled
    // in place, by address (see filled).
    static FILLED: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
}

// A place is about to be traced; `tried`: how many of its traces were
// refused here before and left it open (what later() said, counted by the
// place's owner).
pub fn place(tried: u32) {
    TRIED.set(tried);
    AGAIN.set(false);
    FILLED.with_borrow_mut(|f| f.clear());
}

// The addresses of the containers that an instruction of the trace on this
// thread filled since place() (kinds::filled says which instructions): a
// display or a call's arguments under construction, which only the frame's
// stack refers to. One of them that an earlier block made is among the
// trace's args, and holds what the body added when the trace is given up.
pub fn filled() -> Vec<usize> {
    FILLED.take()
}

// Whether the trace that just failed on this thread was refused for code
// that is observed again, so that its place is left open. Asked once per
// failed trace.
pub fn later() -> bool {
    AGAIN.replace(false)
}

// Whether the refusal step() just made leaves its place open.
pub fn pending() -> bool {
    AGAIN.get()
}

// Whether the frame of `c` that is about to run is observed: it runs the
// plan that records, whatever else would run it.
pub unsafe fn observes(c: &Compiled) -> bool {
    WANTED.load(Ordering::Relaxed) != 0 && own(c).covered.wanted.load(Ordering::Relaxed) != 0
}

// The frame of `c` that observes() was true for is over (`returned`: by a
// return, with no exception set). What its code did until now is judged;
// when the frame ran the instruction a trace was refused for (or FRAMES
// frames have not), the code is observed no more.
pub unsafe fn observed(c: &Compiled, returned: bool) {
    let units = &own(c).covered;
    let wanted = units.wanted.load(Ordering::Relaxed);
    if wanted == 0 {
        return;
    }
    let py = Python::assume_attached();
    let judged = if returned { super::outside_executor(|| super::judge(py, c)) } else { None };
    let ran = judged.is_some() && super::entered(c, wanted - 1);
    if ran && judged == Some(false) {
        units.more.store(true, Ordering::Relaxed);
    }
    if (ran || units.frames.fetch_sub(1, Ordering::Relaxed) <= 1) && units.wanted.swap(0, Ordering::Relaxed) != 0 {
        WANTED.fetch_sub(1, Ordering::Relaxed);
    }
}

fn has(c: &Compiled, unit: u32) -> bool {
    let at = |c: &Compiled| c.covered.seen.lock().unwrap().get(unit as usize).copied().unwrap_or(false);
    at(c) || at(unsafe { own(c) })
}

// While it lives, a call the optimizer did not look into is running.
pub struct Trusted;

impl Trusted {
    pub fn enter() -> Trusted {
        TRUSTED.set(TRUSTED.get() + 1);
        Trusted
    }
}

impl Drop for Trusted {
    fn drop(&mut self) {
        TRUSTED.set(TRUSTED.get() - 1);
    }
}

// Whether a Trusted lives: what runs is under a call nobody looked into.
pub fn trusted() -> bool {
    TRUSTED.get() > 0
}

// Whether Dynamo puts calls of the Python function `f` of torch in its
// graphs. Asking runs Python code: not the body's, and not on the executor.
unsafe fn whole(py: Python<'_>, f: *mut PyObject) -> bool {
    if let Some(known) = WHOLE.with_borrow(|m| m.get(&(f as usize)).copied()) {
        return known;
    }
    let f = Bound::from_borrowed_ptr(py, f);
    let torch = f.getattr("__module__").and_then(|m| m.extract::<String>()).is_ok_and(|m| m.starts_with("torch"));
    let known = torch && {
        let _paused = super::reads::Recording::pause();
        super::outside_executor(|| super::is_in_graph(&f))
    };
    WHOLE.with_borrow_mut(|m| m.insert(f.as_ptr() as usize, known));
    known
}

// Asked before instruction `ins` of `c` runs in a traced body (`under`: in a
// frame under it); `operands`: the objects of its operands, if kinds.rs asks
// about it, `callee`: what it calls, if it is a call, and `seen`: the callee
// recorded for it.
pub unsafe fn step(c: Option<&Compiled>, under: bool, ins: &Instr, operands: &[*mut PyObject], callee: Option<*mut PyObject>, seen: Option<&effects::Callee>) -> Result<bool, String> {
    // Kept whatever is answered below (see filled).
    if let Some(o) = kinds::filled(&ins.op).and_then(|k| operands.get(k)) {
        FILLED.with_borrow_mut(|f| {
            if f.last() != Some(&(*o as usize)) {
                f.push(*o as usize);
            }
        });
    }
    // A trace that is refused already runs on to its end as the program
    // takes it (its handlers, its cleanup): nothing more is asked of it.
    if TRUSTED.get() > 0 || super::region::refused() {
        return Ok(false);
    }
    let py = Python::assume_attached();
    let unit = ins.origin;
    // As the callee of a call is recorded (see slow_step).
    let f = callee.filter(|f| !f.is_null());
    let partial = f.map_or(std::ptr::null_mut(), |f| tr_partial_func(f));
    let call = f.filter(|_| partial.is_null()).map_or(std::ptr::null_mut(), |f| tr_type_function(f, 0));
    let g = f.map_or(std::ptr::null_mut(), |f| tr_bound_function(if !partial.is_null() { partial } else if call.is_null() { f } else { call }));
    let taker = !g.is_null() && shown::takes(g);
    // The class of an operand the instruction takes from or changes that
    // no verdict saw there.
    let unseen = |c: &Compiled| {
        let at = |c: &Compiled| c.covered.shown.unseen(unit, &ins.op, operands, taker);
        if operands.is_empty() { None } else { at(c).and_then(|_| at(own(c))) }
    };
    // The body's own instructions are those the optimizer put in it.
    let covered = !under || c.is_some_and(|c| has(c, unit));
    let mut kind = c.filter(|_| covered).and_then(unseen);
    // A Python kernel of an aten op is part of the op (see pure_indirect).
    if let Some(c) = c
        && !c.filename.contains("/torch/_native/")
        && (!covered || kind.is_some())
    {
        // A tensor's hash (`id(self)`): the optimizer takes it by a rule
        // of the instruction that C code entered it for (the key of a read
        // the region is checked on: see pure_instr), not by a verdict on
        // its code, which says for `id` what a region may not depend on.
        if !covered && c.qualname.ends_with("Tensor.__hash__") && c.filename.contains("/torch/") {
            return Ok(false);
        }
        // Torch's hand-over to the trace's function mode is covered as it
        // is.
        if !covered && let Some(block) = super::hand_over(c, unit) {
            judged(c, block.into_iter().map(|u| (u, None, None)));
            return Ok(false);
        }
        let units = &own(c).covered;
        let more = under && units.more.load(Ordering::Relaxed);
        // Judged now, if its frames ran since: asking runs no code of the
        // body's. The body's own instruction is judged as the optimizer
        // judged it, on what was observed until now.
        let mut why = None;
        let verdict = if more {
            Some(false)
        } else {
            let _paused = super::reads::Recording::pause();
            if under {
                super::outside_executor(|| super::judge(py, c))
            } else {
                why = super::outside_executor(|| super::judge_site(py, c, unit, &ins.op)).err();
                Some(why.is_none())
            }
        };
        if verdict == Some(true) && (!under || has(c, unit)) {
            kind = unseen(c);
        }
        if verdict != Some(true) || (under && !has(c, unit)) || kind.is_some() {
            let none = kind.is_none() && units.seen.lock().unwrap().is_empty() && c.covered.seen.lock().unwrap().is_empty();
            let line = if none || c.line(unit) == 0 { String::new() } else { format!("line {} of ", c.line(unit)) };
            // The place's run as written comes to this code next: its
            // frames are observed, and the place is traced again.
            let nth = TRIED.get() + 1;
            let again = verdict != Some(false) && nth < LATER;
            if again && units.wanted.swap(unit + 1, Ordering::Relaxed) == 0 {
                WANTED.fetch_add(1, Ordering::Relaxed);
            }
            if again {
                units.frames.store(FRAMES, Ordering::Relaxed);
            }
            AGAIN.set(again);
            let what = match &kind {
                Some(k) => format!("it runs {line}{} on a {k}, which no call before it showed there", c.qualname),
                None => format!("it runs {line}{}, which no call before it ran", c.qualname),
            };
            return Err(if again {
                format!("{what} (trace {nth} of {LATER}: that code is observed as the place runs as written, then the place is traced again)")
            } else if more {
                format!("{what}, and which did more than compute when it was observed")
            } else if let Some(why) = why {
                format!("{what}, and as it was observed since: {why}")
            } else if verdict == Some(false) {
                what
            } else {
                format!("{what} (trace {nth} of {LATER})")
            });
        }
    }
    if g.is_null() {
        return Ok(false);
    }
    if super::user_pure(g) || c.is_some_and(|c| c.declared.lock().unwrap().contains(&unit)) {
        return Ok(true);
    }
    if ffi::PyFunction_Check(g) != 0 {
        return Ok(partial.is_null() && call.is_null() && whole(py, g));
    }
    if seen.and_then(|s| s.get(py)).is_some_and(|s| s.as_ptr() == g) || c.is_some_and(|c| called(c, unit, g)) {
        return Ok(false);
    }
    // (What a partial holds for a builtin that takes items is not among the
    // call's operands.)
    let g = Bound::from_borrowed_ptr(py, g);
    if (shown::computes(&g) && (partial.is_null() || !taker)) || super::writes::type_question(g.as_ptr()) || effects::lru_cache_wrapper(g.as_ptr()) || super::plain_class(py, &g) {
        return Ok(false);
    }
    Err(format!("it calls {}, which no call before it called there", g.repr().map_or("?".into(), |r| r.to_string())))
}
