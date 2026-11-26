// Interprets one activation of a Compiled CFG on a CPython frame. Locals live
// in the frame (localsplus); SSA values are owned references in `vals`.

use std::os::raw::c_int;
use std::sync::Arc;

use pyo3::ffi::{self, PyObject};

use crate::bindings::ffi as tffi;
use crate::cfg::ir::*;

use super::effects::{self, Recorder};
use super::reads;
use super::writes;
use super::scalars;
use super::ffi::*;
use super::{Compiled, Prepared};

// torch._rust_inductor._opaque_call (see Act::opaque_call), once looked up.
fn opaque_helper(py: pyo3::Python<'_>) -> pyo3::PyResult<pyo3::Bound<'_, pyo3::PyAny>> {
    use pyo3::types::PyAnyMethods;
    static HELPER: pyo3::sync::PyOnceLock<pyo3::Py<pyo3::PyAny>> = pyo3::sync::PyOnceLock::new();
    HELPER
        .get_or_try_init(py, || Ok::<_, pyo3::PyErr>(py.import("torch._rust_inductor")?.getattr("_opaque_call")?.unbind()))
        .map(|h| h.bind(py).clone())
}

// How an exception was raised: Error adds a traceback entry for this frame
// (the default loop's `error:`); Reraise doesn't (`exception_unwind:`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Exc {
    Error,
    Reraise,
}

type R = Result<(), Exc>;

pub enum Outcome {
    Return(*mut PyObject),
    // A generator suspended at the Yield with origin `unit`, its stack saved
    // in the frame.
    Yield { value: *mut PyObject, unit: u32 },
    // An exception is set.
    Error,
    // Only from an outlined Cfg: it left with these values as its owner's
    // outputs, or unwound to its owner's handlers[k] with these args (an
    // exception is set).
    Leave(Vec<*mut PyObject>),
    Outer(usize, Vec<*mut PyObject>),
    // Only while tracing a span (see span_trace): about to run this block,
    // its params set, which is the span's exit or outside it.
    Stopped(BlockId),
}

// Where an exception goes from a raise point.
enum Unwound {
    Block(BlockId),
    Outer(usize, Vec<*mut PyObject>),
    Propagate,
}

pub enum Start {
    Entry,
    // A generator frame resumed (or thrown into) at body[i] of version `ver`
    // of block b: a Yield, or RETURN_GENERATOR on first resume. The sent value
    // is on top of the frame's stack, above the stack saved at suspension.
    Resume { b: BlockId, ver: usize, i: usize, throw: bool },
}

fn status(rc: c_int) -> R {
    match rc {
        0 => Ok(()),
        -2 => Err(Exc::Reraise),
        _ => Err(Exc::Error),
    }
}

fn check(p: *mut PyObject) -> Result<*mut PyObject, Exc> {
    if p.is_null() {
        Err(Exc::Error)
    } else {
        Ok(p)
    }
}

unsafe fn xincref(p: *mut PyObject) -> *mut PyObject {
    ffi::Py_XINCREF(p);
    p
}

unsafe fn new_bool(b: bool) -> *mut PyObject {
    xincref(if b { ffi::Py_True() } else { ffi::Py_False() })
}

struct Act<'a> {
    // None when running a graph (outside any frame).
    c: Option<&'a Compiled>,
    p: &'a Prepared,
    cfg: &'a Cfg,
    f: *mut Frame,
    locals: *mut *mut PyObject,
    // By index: Prepared::consts and Prepared::names (borrowed).
    consts: *const *mut PyObject,
    names: *const *mut PyObject,
    vals: Vec<*mut PyObject>,
    tstate: *mut ffi::PyThreadState,
    // The activation's recorder (null when not recording, see recording()),
    // and `rec`: the same, or null while running an optimized version.
    scope_rec: *mut Recorder,
    rec: *mut Recorder,
    // The frame's prev_instr, the code units it points into, and the
    // interpreter's eval-breaker flag.
    prev_instr: *mut *mut u16,
    code_units: *mut u16,
    breaker_flag: *const std::sync::atomic::AtomicI32,
    // Scratch, reused to avoid allocating per instr or block: an instr's
    // args, a version's outputs, an edge's args.
    args: Vec<*mut PyObject>,
    outs: Vec<*mut PyObject>,
    edge: Vec<*mut PyObject>,
    // The span being traced (see span_trace), or null.
    stop: *const Span,
    // Whether this activation runs a traced region's body (or a frame under
    // it): what its instrs read is recorded (see reads.rs).
    reading: bool,
    // Whether this activation runs a frame under a traced body: it runs
    // only what the optimizer's verdict covered (see covered.rs).
    under_body: bool,
}

// An activation's buffers, pooled per thread so a call allocates nothing.
// Pooled `vals` are all null: an activation drops every value it holds before
// it ends (run_blocks drops what's left of the version it leaves by).
#[derive(Default)]
struct Bufs {
    vals: Vec<*mut PyObject>,
    args: Vec<*mut PyObject>,
    outs: Vec<*mut PyObject>,
    edge: Vec<*mut PyObject>,
}

thread_local! {
    static POOL: std::cell::RefCell<Vec<Bufs>> = const { std::cell::RefCell::new(Vec::new()) };
}

impl Bufs {
    // With at least `n` null values.
    fn take(n: usize) -> Bufs {
        let mut b = POOL.with_borrow_mut(|p| p.pop()).unwrap_or_default();
        if b.vals.len() < n {
            b.vals.resize(n, std::ptr::null_mut());
        }
        b
    }

    fn give(self) {
        debug_assert!(self.vals.iter().all(|v| v.is_null()), "an activation ended holding values");
        POOL.with_borrow_mut(|p| p.push(self));
    }
}

// A version's body, laid out for run_blocks.
pub struct Code {
    pub steps: Vec<Step>,
    // The values each step drops after it runs (Step::kill ranges).
    pub kills: Vec<ValueId>,
    // What's left when the version runs to its end: the values no step drops
    // (those its outputs or only its unwind segments use).
    pub exit_kill: Vec<ValueId>,
    // For each exit of the terminator (Jump: one, Branch: then and else) that
    // goes to a block, how the version's values pass to its params.
    pub edges: Vec<Option<Transfer>>,
}

pub struct Transfer {
    pub target: BlockId,
    // (source value, target param, whether the source moves rather than
    // being shared: its last use, when the version holds it).
    pub moves: Vec<(ValueId, ValueId, bool)>,
    // exit_kill minus the moved values.
    pub rest: Vec<ValueId>,
}

pub struct Step {
    pub v: ValueId,
    pub op: Fast,
    pub flags: u8,
    pub kill: (u32, u32),
}

// The ops run_blocks runs inline when nothing is recorded: they can't raise
// or enter a frame.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Fast {
    Const(u32),
    LoadLocal(u32),
    // (slot, value)
    StoreLocal(u32, ValueId),
    No,
}

impl Code {
    // By block, then version. `flags` by instr (FLAG_*).
    pub fn build(cfg: &Cfg, live: &super::live::Liveness, flags: &[u8]) -> Vec<Vec<Code>> {
        // Only params and instrs with FLAG_VALUE ever hold an object.
        let holds = |v: &ValueId| flags[v.idx()] & FLAG_VALUE != 0 || matches!(cfg[*v].op, Op::Param(_));
        let version = |blk: &Block, ver: &Version, lv: &super::live::VersionLive| {
            let mut kills = vec![];
            let steps = ver
                .body
                .iter()
                .zip(&lv.after_kill)
                .map(|(v, after)| {
                    let start = kills.len() as u32;
                    kills.extend(after.iter().filter(|k| holds(k)));
                    let op = match cfg[*v].op {
                        Op::Const(k) => Fast::Const(k),
                        Op::LoadLocal { slot } => Fast::LoadLocal(slot),
                        Op::StoreLocal { slot } => Fast::StoreLocal(slot, cfg[*v].args[0]),
                        _ => Fast::No,
                    };
                    Step { v: *v, op, flags: flags[v.idx()], kill: (start, kills.len() as u32) }
                })
                .collect();
            let dropped: std::collections::HashSet<&ValueId> = lv.entry_kill.iter().chain(&kills).collect();
            let mut exit_kill: Vec<ValueId> =
                blk.params.iter().chain(&ver.inputs).chain(&ver.body).filter(|v| holds(v) && !dropped.contains(v)).copied().collect();
            exit_kill.sort();
            let exits: Vec<&Exit> = match &blk.term {
                Terminator::Jump(e) => vec![e],
                Terminator::Branch { then, else_, .. } => vec![then, else_],
                _ => vec![],
            };
            let edges = exits
                .into_iter()
                .map(|e| {
                    let Exit::Block(edge) = e else { return None };
                    let srcs: Vec<ValueId> = edge.args.iter().map(|o| ver.outputs[*o as usize]).collect();
                    let owned = |v: &ValueId| exit_kill.contains(v);
                    let moves: Vec<(ValueId, ValueId, bool)> = srcs
                        .iter()
                        .zip(&cfg[edge.target].params)
                        .enumerate()
                        .map(|(i, (src, dst))| (*src, *dst, owned(src) && !srcs[i + 1..].contains(src)))
                        .collect();
                    let rest = exit_kill.iter().filter(|v| !moves.iter().any(|(s, _, m)| *m && s == *v)).copied().collect();
                    Some(Transfer { target: edge.target, moves, rest })
                })
                .collect();
            Code { steps, kills, exit_kill, edges }
        };
        cfg.blocks
            .iter()
            .zip(&live.versions)
            .map(|(blk, lvs)| blk.versions.iter().zip(lvs).map(|(ver, lv)| version(blk, ver, lv)).collect())
            .collect()
    }
}

// Whether `guard` pins `x` down (it's NULL, or a pinned object), so it can
// be passed through as it is.
fn pins_down(guard: &Guard, x: ValueId) -> bool {
    guard.checks.iter().any(|(t, c)| *t == Some(x) && matches!(c, Check::Null | Check::Is(_)))
}

// How run_blocks continues after a step.
enum Next {
    Step,
    // A raise went to this handler (entered).
    Block(BlockId),
    Done(Outcome),
}

pub const FLAG_RAISE: u8 = 1;
// Its own id holds a value (not a multi-output instr's), observed when it's
// defined.
pub const FLAG_VALUE: u8 = 2;
// A call whose callable is args[0] (or args[1] when args[0] is NULL).
pub const FLAG_CALL: u8 = 4;
// CALL_FUNCTION_EX, whose callable is args[0] (its NULL slot isn't an arg).
pub const FLAG_CALL_EX: u8 = 8;
pub const FLAG_YIELD: u8 = 16;

// Runs `f` with a recorder for p's table at `site`, or with none on this
// thread if p doesn't record or recording is off.
fn recording<R>(p: &Prepared, site: Site, f: impl FnOnce(*mut Recorder) -> R) -> R {
    if p.record && super::RECORD.load(std::sync::atomic::Ordering::Relaxed) && !super::region::tracing() {
        effects::scope(&p.effects, site, f)
    } else {
        effects::unrecorded(|| f(std::ptr::null_mut()))
    }
}

// Runs the frame, which the caller has linked.
pub unsafe fn run(c: &Compiled, prep: &Prepared, f: *mut Frame, start: Start) -> Outcome {
    let site = match start {
        Start::Entry => Site::Block(ENTRY),
        Start::Resume { b, .. } => Site::Block(b),
    };
    recording(prep, site, |rec| {
        let Bufs { vals, args, outs, edge } = Bufs::take(prep.cfg.instrs.len());
        let mut act = Act {
            c: Some(c),
            p: prep,
            cfg: &prep.cfg,
            f,
            locals: tr_frame_localsplus(f),
            consts: prep.consts.0.as_ptr(),
            names: prep.names.0.as_ptr(),
            vals,
            tstate: ffi::PyThreadState_Get(),
            scope_rec: rec,
            rec,
            prev_instr: tr_frame_prev_instr_slot(f),
            code_units: tr_frame_code_units(f),
            breaker_flag: std::sync::atomic::AtomicI32::from_ptr(tr_eval_breaker_flag(ffi::PyThreadState_Get())),
            args,
            outs,
            edge,
            stop: std::ptr::null(),
            reading: reads::entered_by_body(),
            under_body: reads::entered_by_body(),
        };
        // Between this frame's instrs no instr is running (the caller puts
        // back what was, once the frame is gone).
        if act.reading {
            reads::function(tr_frame_func(f));
            reads::set_in_instr(false);
        }
        let result = match start {
            Start::Entry => act.run_blocks(ENTRY, None),
            Start::Resume { b, ver, i, throw } => match act.resume(b, ver, i, throw) {
                Some(Ok(h)) => act.run_blocks(h, None),
                Some(Err(next)) => act.run_blocks(b, Some((ver, next))),
                None => Outcome::Error,
            },
        };
        act.into_bufs().give();
        result
    })
}

// Runs a Cfg of aten ops (see graph.rs) on `inputs` (borrowed), outside any
// frame: Ok(its outputs, owned), or Err with an exception set.
pub unsafe fn run_graph(prep: &Prepared, inputs: &[*mut PyObject]) -> Result<Vec<*mut PyObject>, ()> {
    let Bufs { vals, args, outs, edge } = Bufs::take(prep.cfg.instrs.len());
    let mut dummy: *mut u16 = std::ptr::null_mut();
    let mut act = Act {
        c: None,
        p: prep,
        cfg: &prep.cfg,
        f: std::ptr::null_mut(),
        locals: std::ptr::null_mut(),
        consts: std::ptr::null(),
        names: std::ptr::null(),
        vals,
        tstate: ffi::PyThreadState_Get(),
        scope_rec: std::ptr::null_mut(),
        rec: std::ptr::null_mut(),
        prev_instr: &mut dummy,
        code_units: std::ptr::null_mut(),
        breaker_flag: std::sync::atomic::AtomicI32::from_ptr(tr_eval_breaker_flag(ffi::PyThreadState_Get())),
        args,
        outs,
        edge,
        stop: std::ptr::null(),
        reading: false,
        under_body: false,
    };
    for (obj, p) in inputs.iter().zip(&prep.cfg[ENTRY].params) {
        act.set(*p, xincref(*obj));
    }
    let result = act.run_blocks(ENTRY, None);
    act.into_bufs().give();
    match result {
        Outcome::Leave(outs) => Ok(outs),
        Outcome::Error => Err(()),
        _ => unreachable!("verified: a graph leaves"),
    }
}

impl Act<'_> {
    // Points the frame at code unit `unit`, as if that instruction is executing.
    // (Without a frame, prev_instr is a dummy and code_units null.)
    #[inline]
    unsafe fn set_instr(&self, unit: c_int) {
        *self.prev_instr = self.code_units.wrapping_add(unit as usize);
    }

    unsafe fn traceback_here(&self) {
        if !self.f.is_null() {
            tr_traceback_here(self.f);
        }
    }

    fn into_bufs(self) -> Bufs {
        Bufs { vals: self.vals, args: self.args, outs: self.outs, edge: self.edge }
    }

    // Moves the frame's saved stack back into SSA values. Returns where to
    // continue: Ok(the handler a throw unwound to), or Err(the index after the
    // resume point); None if the frame is done (an exception is set).
    unsafe fn resume(&mut self, b: BlockId, ver: usize, i: usize, throw: bool) -> Option<Result<BlockId, usize>> {
        let cfg = self.cfg;
        let v = cfg[b].versions[ver].body[i];
        let sent = tr_frame_pop(self.f);
        let saved: &[ValueId] = match cfg[v].op {
            Op::Yield => &cfg[v].args[..cfg[v].args.len() - 1],
            _ => &[],
        };
        if tr_frame_stack_depth(self.f) as usize != saved.len() {
            ffi::Py_DECREF(sent);
            super::fail(
                "CFG executor: unexpected stack depth in a suspended generator",
                ffi::PyExc_RuntimeError,
            );
            return None;
        }
        for a in saved.iter().rev() {
            let obj = tr_frame_pop(self.f);
            self.set(*a, obj);
        }
        if throw {
            ffi::Py_DECREF(sent);
            return match self.unwind(b, ver, i, Exc::Error) {
                Unwound::Block(h) => Some(Ok(h)),
                Unwound::Propagate => None,
                Unwound::Outer(..) => unreachable!("resume points are in the top-level Cfg"),
            };
        }
        self.set(v, sent);
        self.observe(v);
        let code = &self.p.code[b.idx()][ver];
        let (k0, k1) = code.steps[i].kill;
        self.drop_all(&code.kills[k0 as usize..k1 as usize]);
        Some(Err(i + 1))
    }

    fn val(&self, v: ValueId) -> *mut PyObject {
        self.vals[v.idx()]
    }

    unsafe fn set(&mut self, v: ValueId, obj: *mut PyObject) {
        let old = std::mem::replace(&mut self.vals[v.idx()], obj);
        ffi::Py_XDECREF(old);
    }

    // Joins v's current object into its type, if recording.
    #[inline]
    unsafe fn observe(&self, v: ValueId) {
        if !self.rec.is_null() && !(&*self.p.types.get())[v.idx()].skips(self.val(v)) {
            self.observe_slow(v);
        }
    }

    #[inline(never)]
    unsafe fn observe_slow(&self, v: ValueId) {
        // A Const is a co_consts object: the same immutable one every time.
        let constant = matches!(self.cfg[v].op, Op::Const(_));
        (&mut *self.p.types.get())[v.idx()].observe(self.val(v), constant);
    }

    // The args' objects, in self.args (valid until the next call).
    fn args(&mut self, ins: &Instr) -> *const *mut PyObject {
        self.args.clear();
        self.args.extend(ins.args.iter().map(|a| self.vals[a.idx()]));
        self.args.as_ptr()
    }

    unsafe fn local(&self, slot: u32) -> *mut *mut PyObject {
        self.locals.add(slot as usize)
    }

    unsafe fn name(&self, i: u32) -> *mut PyObject {
        *self.names.add(i as usize)
    }

    unsafe fn konst(&self, i: u32) -> *mut PyObject {
        *self.consts.add(i as usize)
    }

    // Moves the outputs of a multi-output instr into its Project instrs.
    unsafe fn distribute(&mut self, v: ValueId, outs: &[*mut PyObject]) {
        let p = self.p;
        for (k, obj) in outs.iter().enumerate() {
            match p.projects[v.idx()].get(k).copied().flatten() {
                Some(proj) => self.set(proj, *obj),
                // An output nobody reads (simplify never drops Projects, but be safe).
                None => ffi::Py_XDECREF(*obj),
            }
        }
    }

    unsafe fn drop_all(&mut self, vs: &[ValueId]) {
        for v in vs {
            self.set(*v, std::ptr::null_mut());
        }
    }

    // Drops what is left of a version's values when control leaves the block.
    unsafe fn clear(&mut self, b: BlockId, ver: usize) {
        let blk = &self.cfg[b];
        for v in blk.params.iter().chain(&blk.versions[ver].inputs).chain(&blk.versions[ver].body) {
            self.set(*v, std::ptr::null_mut());
        }
    }

    // Leaves a version that ran to its end by a normal edge.
    unsafe fn transfer(&mut self, t: &Transfer) {
        let mut args = std::mem::take(&mut self.edge);
        for (src, _, moved) in &t.moves {
            let p = &mut self.vals[src.idx()];
            args.push(if *moved { std::mem::replace(p, std::ptr::null_mut()) } else { xincref(*p) });
        }
        self.drop_all(&t.rest);
        for ((_, dst, _), obj) in t.moves.iter().zip(args.drain(..)) {
            self.set(*dst, obj);
            self.observe(*dst);
        }
        self.edge = args;
    }

    // Drops what is left of a version's values when it ran to the end.
    unsafe fn clear_at_exit(&mut self, b: BlockId, ver: usize) {
        let p = self.p;
        self.drop_all(&p.code[b.idx()][ver].exit_kill);
    }

    // Enters `target` (after clear()), with owned values for its explicit
    // params in order and `implicit` supplying the others.
    unsafe fn enter(
        &mut self,
        target: BlockId,
        explicit: impl IntoIterator<Item = *mut PyObject>,
        mut implicit: impl FnMut(ParamKind) -> *mut PyObject,
    ) {
        let mut explicit = explicit.into_iter();
        for p in &self.cfg[target].params {
            let kind = self.cfg.param_kind(*p).expect("block params are Params");
            let obj = if kind.is_implicit() {
                implicit(kind)
            } else {
                explicit.next().expect("arg count checked by the verifier")
            };
            self.set(*p, obj);
            self.observe(*p);
        }
    }

    // The version of `b` to run: the first whose guard holds, with its
    // inputs read. None: run its origin.
    // The first version from `start` on whose guard holds, after a region
    // version before it declined to run: the last version's guard, which
    // the region's implied, holds if no other's does.
    unsafe fn select_from(&mut self, b: BlockId, start: usize) -> usize {
        let blk = &self.cfg[b];
        let last = blk.versions.len() - 1;
        self.drop_reads(b, blk.reads.len(), None);
        let mut done = 0;
        let chosen = (start..last).find(|vi| self.guard_holds(b, *vi, &mut done) == Some(true)).unwrap_or(last);
        let k = blk.versions[chosen].inputs.iter().map(|x| self.p.read_pos[x.idx()]).max();
        if let Some(k) = k
            && !self.read_upto(b, &mut done, k as usize)
        {
            unreachable!("the reads ran before");
        }
        self.drop_reads(b, done, Some(chosen));
        self.drop_all(&self.p.live.versions[b.idx()][chosen].entry_kill);
        chosen
    }

    unsafe fn select(&mut self, b: BlockId) -> Option<usize> {
        let blk = &self.cfg[b];
        let vers = &blk.versions;
        let ver = if vers.len() == 1 && vers[0].guard.checks.is_empty() && vers[0].inputs.is_empty() {
            0
        } else {
            let mut done = 0;
            let mut chosen = None;
            for vi in 0..vers.len() {
                match self.guard_holds(b, vi, &mut done) {
                    Some(true) => {
                        chosen = Some(vi);
                        break;
                    }
                    Some(false) => {}
                    None => break,
                }
            }
            self.drop_reads(b, done, chosen);
            chosen?
        };
        self.drop_all(&self.p.live.versions[b.idx()][ver].entry_kill);
        Some(ver)
    }

    // Whether version vi's guard holds, running b's reads (from `*done`) as
    // far as its checks need, and then as far as its inputs need. None: a
    // read raised (cleared).
    unsafe fn guard_holds(&mut self, b: BlockId, vi: usize, done: &mut usize) -> Option<bool> {
        let cfg = self.cfg;
        let ver = &cfg[b].versions[vi];
        for (x, c) in &ver.guard.checks {
            if let Some(x) = x {
                let k = self.p.read_pos[x.idx()];
                if k != u32::MAX && !self.read_upto(b, done, k as usize) {
                    return None;
                }
            }
            if !self.check(c, x.map_or(std::ptr::null_mut(), |x| self.val(x))) {
                return Some(false);
            }
        }
        match ver.inputs.iter().map(|x| self.p.read_pos[x.idx()]).max() {
            Some(k) if !self.read_upto(b, done, k as usize) => None,
            _ => Some(true),
        }
    }

    // Runs b's reads from `*done` through the one at `upto`. False if one
    // raised (cleared: a miss).
    unsafe fn read_upto(&mut self, b: BlockId, done: &mut usize, upto: usize) -> bool {
        let cfg = self.cfg;
        while *done <= upto {
            let v = cfg[b].reads[*done];
            *done += 1;
            if self.exec(v, &cfg[v]).is_err() {
                ffi::PyErr_Clear();
                return false;
            }
        }
        true
    }

    // Drops what b's first `done` reads left, but version `keep`'s inputs.
    unsafe fn drop_reads(&mut self, b: BlockId, done: usize, keep: Option<usize>) {
        let blk = &self.cfg[b];
        for r in &blk.reads[..done] {
            if keep.is_none_or(|vi| !blk.versions[vi].inputs.contains(r)) {
                self.set(*r, std::ptr::null_mut());
            }
        }
    }

    // Whether `c` holds of `o` (NULL for a global check, or a NULL value).
    unsafe fn check(&self, c: &Check, o: *mut PyObject) -> bool {
        if o.is_null() && !c.global() {
            return *c == Check::Null;
        }
        let tensor = |o: *mut PyObject| tffi::tensor_by_type(o as usize).as_ref();
        match c {
            Check::PlainTensor => super::is_plain_tensor(o),
            Check::Exact(t) => {
                let tp = ffi::Py_TYPE(o);
                tp == match t {
                    Builtin::Int => &raw mut ffi::PyLong_Type,
                    Builtin::Float => &raw mut ffi::PyFloat_Type,
                    Builtin::Bool => &raw mut ffi::PyBool_Type,
                    Builtin::Str => &raw mut ffi::PyUnicode_Type,
                }
            }
            Check::NoTorchFunctionMode => !tffi::torch_function_mode_active(),
            Check::GradEnabled(on) => tffi::grad_mode_enabled() == *on,
            Check::RequiresGrad(on) => tensor(o).is_some_and(|t| t.requires_grad() == *on),
            Check::TensorMeta(m) => tensor(o)
                .is_some_and(|t| tffi::tensor_matches(t, m.dtype, m.device_type, m.device_index, &m.sizes, &m.strides)),
            Check::Value(l) => {
                let tp = ffi::Py_TYPE(o);
                match l {
                    Lit::None => o == ffi::Py_None(),
                    Lit::Bool(b) => o == if *b { ffi::Py_True() } else { ffi::Py_False() },
                    Lit::Int(n) => {
                        let mut overflow = 0;
                        tp == &raw mut ffi::PyLong_Type
                            && ffi::PyLong_AsLongLongAndOverflow(o, &mut overflow) == *n
                            && overflow == 0
                    }
                    Lit::Float(bits) => tp == &raw mut ffi::PyFloat_Type && ffi::PyFloat_AsDouble(o).to_bits() == *bits,
                    _ => false,
                }
            }
            Check::NoForwardAD => !tffi::forward_ad_active(),
            Check::NoSavedTensorHooks => !tffi::saved_tensor_hooks_active(),
            Check::Is(k) => o == self.p.pins[*k as usize].as_ptr(),
            Check::Null => false,
            Check::Ints(tuple, ns) => {
                let exact = if *tuple { ffi::PyTuple_CheckExact(o) } else { ffi::PyList_CheckExact(o) } != 0;
                exact
                    && ffi::Py_SIZE(o) as usize == ns.len()
                    && ns.iter().enumerate().all(|(k, n)| {
                        let x = if *tuple { ffi::PyTuple_GET_ITEM(o, k as isize) } else { ffi::PyList_GET_ITEM(o, k as isize) };
                        let mut overflow = 0;
                        ffi::Py_TYPE(x) == &raw mut ffi::PyLong_Type && ffi::PyLong_AsLongLongAndOverflow(x, &mut overflow) == *n && overflow == 0
                    })
            }
            Check::TypeAttr { name, pin } => {
                if self.reading {
                    reads::class_of(o, self.name(*name));
                }
                tr_type_attr_is(o, self.name(*name), self.p.pins[*pin as usize].as_ptr()) != 0
            }
            Check::Getattr { name, pin } => {
                if self.reading {
                    reads::attr(o, self.name(*name), std::ptr::null_mut(), true);
                }
                tr_getattr_falls_back(o, self.name(*name), self.p.pins[*pin as usize].as_ptr()) != 0
            }
        }
    }

    // No version of `b` matched: run its origin on b's params, then continue
    // as b would with the outputs it leaves with.
    unsafe fn miss(&mut self, b: BlockId) -> Result<BlockId, Outcome> {
        self.p.misses[b.idx()].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let params = self.take_params(b);
        let result = self.run_origin(b, params);
        self.after_origin(b, result)
    }

    // Runs b's origin with its own autograd node (see autograd.rs).
    unsafe fn bootstrap(&mut self, b: BlockId) -> Result<BlockId, Outcome> {
        if !self.cfg[b].reads.is_empty() {
            return self.bootstrap_inputs(b);
        }
        let params = self.take_params(b);
        self.p.kept.store(true, std::sync::atomic::Ordering::Relaxed);
        let at = super::autograd::Where { compiled: super::shared(self.c.expect("a frame's Act")), prep: self.p as *const Prepared as usize, block: b };
        // Params passed through as they are: those the guard pins down (a
        // NULL, a pinned callee) and those no op reads (only carried along).
        let blk = &self.cfg[b];
        let read = |q: ValueId| blk.versions[0].body.iter().any(|v| self.cfg[*v].args.contains(&q));
        let pinned: Vec<bool> = (0..params.len() as u32)
            .map(|i| pins_down(&blk.versions[0].guard, blk.params[i as usize]) || !read(blk.params[i as usize]))
            .collect();
        let started = super::outside_executor(|| super::autograd::Bootstrap::start(&params, &pinned, at));
        let result = match started {
            Ok((boot, call)) => {
                let result = super::outside_executor(|| self.run_origin(b, call));
                let ran = matches!(result, Outcome::Leave(_));
                match super::outside_executor(|| boot.finish(result)) {
                    // The traced run raised: stop bootstrapping b, and let
                    // the exception go where it went (running the block
                    // again would repeat what it did before raising).
                    r @ (Outcome::Error | Outcome::Outer(..)) if !ran => {
                        self.stop_bootstrapping(b);
                        r
                    }
                    // The block ran but its node couldn't be made: stop
                    // bootstrapping b, and run it as usual.
                    r @ (Outcome::Error | Outcome::Outer(..)) => {
                        if let Outcome::Outer(_, args) = r {
                            args.into_iter().for_each(|a| ffi::Py_XDECREF(a));
                        }
                        let py = pyo3::Python::assume_attached();
                        let why = pyo3::PyErr::take(py).map_or("?".into(), |e| e.to_string());
                        self.p.autograd[b.idx()].store(false, std::sync::atomic::Ordering::Relaxed);
                        *self.p.autograd_why[b.idx()].get() = Some(format!("the traced run raised {why}"));
                        self.run_origin(b, params.iter().map(|p| xincref(*p)).collect())
                    }
                    r => r,
                }
            }
            Err(e) => {
                e.restore(pyo3::Python::assume_attached());
                Outcome::Error
            }
        };
        for p in params {
            ffi::Py_XDECREF(p);
        }
        self.after_origin(b, result)
    }

    // After b raised under the tracer: b won't get its own node. The
    // exception stays set.
    unsafe fn stop_bootstrapping(&self, b: BlockId) {
        let py = pyo3::Python::assume_attached();
        let why = pyo3::PyErr::take(py).map(|e| {
            let why = e.to_string();
            e.restore(py);
            why
        });
        self.p.autograd[b.idx()].store(false, std::sync::atomic::Ordering::Relaxed);
        *self.p.autograd_why[b.idx()].get() = Some(format!("the traced run raised {}", why.as_deref().unwrap_or("?")));
    }

    // Bootstraps block b, whose version has reads in its guard: checks them
    // and reads its inputs, then runs (outside the executor, under the
    // tracer) the version's body on the params and inputs, which the node's
    // edges then go to, instead of the origin (which would read them again).
    // A miss runs the origin, as it would without autograd; a raise in the
    // body unwinds from the version.
    unsafe fn bootstrap_inputs(&mut self, b: BlockId) -> Result<BlockId, Outcome> {
        let cfg = self.cfg;
        let blk = &cfg[b];
        let ver = &blk.versions[0];
        let mut done = 0;
        let held = self.guard_holds(b, 0, &mut done) == Some(true);
        self.drop_reads(b, done, held.then_some(0));
        if !held {
            return self.miss(b);
        }
        let saved: Vec<*mut PyObject> = blk.params.iter().map(|p| xincref(self.val(*p))).collect();
        let rerun = |this: &mut Self, saved: Vec<*mut PyObject>| {
            if !ffi::PyErr_Occurred().is_null() {
                ffi::PyErr_Clear();
            }
            this.clear(b, 0);
            let result = this.run_origin(b, saved);
            this.after_origin(b, result)
        };
        let ids: Vec<ValueId> = blk.params.iter().chain(&ver.inputs).copied().collect();
        let args: Vec<*mut PyObject> = ids.iter().map(|x| self.val(*x)).collect();
        let needs = args.iter().any(|o| !o.is_null() && tffi::tensor_by_type(*o as usize).as_ref().is_some_and(|t| t.requires_grad()));
        // Run the rest; Leave with the outputs, or Error (where it raised,
        // and how, in `failed`).
        let mut failed: Option<(usize, Exc)> = None;
        let mut run_rest = |this: &mut Self| {
            for (i, v) in ver.body.iter().enumerate() {
                this.set_instr(cfg[*v].origin as c_int);
                if let Err(how) = this.exec(*v, &cfg[*v]) {
                    failed = Some((i, how));
                    return Outcome::Error;
                }
            }
            Outcome::Leave(ver.outputs.iter().map(|o| xincref(this.val(*o))).collect())
        };
        let result = if needs {
            // Passed through as they are: what the guard pins down, what
            // the body doesn't read (only passes on as an output), and pinned
            // objects and constants.
            let compute_reads = |q: ValueId| ver.body.iter().any(|v| cfg[*v].args.contains(&q));
            let pinned: Vec<bool> = ids
                .iter()
                .map(|x| pins_down(&ver.guard, *x) || !compute_reads(*x) || matches!(cfg[*x].op, Op::Pinned(_) | Op::Const(_) | Op::Lit(_)))
                .collect();
            self.p.kept.store(true, std::sync::atomic::Ordering::Relaxed);
            let at = super::autograd::Where { compiled: super::shared(self.c.expect("a frame's Act")), prep: self.p as *const Prepared as usize, block: b };
            match super::outside_executor(|| super::autograd::Bootstrap::start(&args, &pinned, at)) {
                Ok((boot, call)) => {
                    for (x, obj) in ids.iter().zip(call) {
                        let old = std::mem::replace(&mut self.vals[x.idx()], obj);
                        ffi::Py_XDECREF(old);
                    }
                    let result = super::outside_executor(|| run_rest(self));
                    match super::outside_executor(|| boot.finish(result)) {
                        Outcome::Error if failed.is_some() => {
                            self.stop_bootstrapping(b);
                            Outcome::Error
                        }
                        // As in bootstrap: the block ran but its node
                        // couldn't be made; stop bootstrapping b, and run it
                        // as usual.
                        Outcome::Error => {
                            let py = pyo3::Python::assume_attached();
                            let why = pyo3::PyErr::take(py).map_or("?".into(), |e| e.to_string());
                            self.p.autograd[b.idx()].store(false, std::sync::atomic::Ordering::Relaxed);
                            *self.p.autograd_why[b.idx()].get() = Some(format!("the traced run raised {why}"));
                            return rerun(self, saved);
                        }
                        r => r,
                    }
                }
                Err(e) => {
                    e.restore(pyo3::Python::assume_attached());
                    Outcome::Error
                }
            }
        } else {
            run_rest(self)
        };
        for p in saved {
            ffi::Py_XDECREF(p);
        }
        if let Some((i, how)) = failed {
            return match self.unwind(b, 0, i, how) {
                Unwound::Block(h) => Ok(h),
                Unwound::Outer(k, args) => Err(Outcome::Outer(k, args)),
                Unwound::Propagate => Err(Outcome::Error),
            };
        }
        self.clear(b, 0);
        self.after_origin(b, result)
    }

    // Whether b's block may bootstrap now: its version's guard holds (but its
    // checks on reads, see bootstrap_inputs), grad mode is on and a tensor
    // param requires grad (or, with inputs, grad mode is on: they're only
    // known once read).
    unsafe fn wants_autograd(&self, b: BlockId) -> bool {
        let blk = &self.cfg[b];
        super::autograd::enabled()
            && tffi::grad_mode_enabled()
            && blk.versions[0].guard.checks.iter().all(|(x, c)| match x {
                None => self.check(c, std::ptr::null_mut()),
                Some(x) if blk.params.contains(x) => self.check(c, self.val(*x)),
                Some(_) => true,
            })
            && (!blk.versions[0].inputs.is_empty()
                || blk.params.iter().any(|p| tffi::tensor_by_type(self.val(*p) as usize).as_ref().is_some_and(|t| t.requires_grad())))
    }

    unsafe fn take_params(&mut self, b: BlockId) -> Vec<*mut PyObject> {
        self.cfg[b].params.iter().map(|p| std::mem::replace(&mut self.vals[p.idx()], std::ptr::null_mut())).collect()
    }

    // Runs b's origin on `params` (owned).
    unsafe fn run_origin(&self, b: BlockId, params: Vec<*mut PyObject>) -> Outcome {
        let origin = self.p.origins[b.idx()].as_ref().expect("verified: a block without a matching version has an origin");
        recording(origin, Site::Block(ENTRY), |rec| {
            let Bufs { vals, args, outs, edge } = Bufs::take(origin.cfg.instrs.len());
            let mut sub = Act {
                c: self.c,
                p: origin,
                cfg: &origin.cfg,
                vals,
                scope_rec: rec,
                rec,
                args,
                outs,
                edge,
                locals: self.locals,
                consts: self.consts,
                names: self.names,
                f: self.f,
                tstate: self.tstate,
                prev_instr: self.prev_instr,
                code_units: self.code_units,
                breaker_flag: self.breaker_flag,
                stop: std::ptr::null(),
                reading: self.reading,
                under_body: self.under_body,
            };
            for (obj, q) in params.into_iter().zip(&origin.cfg[ENTRY].params) {
                sub.set(*q, obj);
            }
            let result = sub.run_blocks(ENTRY, None);
            sub.into_bufs().give();
            result
        })
    }

    // Continues as b would after its origin ran with `result`.
    unsafe fn after_origin(&mut self, b: BlockId, result: Outcome) -> Result<BlockId, Outcome> {
        match result {
            Outcome::Leave(outs) => {
                let next = self.take_exit(b, &outs, None);
                for o in outs {
                    ffi::Py_XDECREF(o);
                }
                next
            }
            Outcome::Outer(k, args) => {
                let h = self.cfg[b].handlers[k];
                self.enter_handler(h, args);
                Ok(h)
            }
            Outcome::Error => Err(Outcome::Error),
            Outcome::Return(_) | Outcome::Yield { .. } => {
                unreachable!("verified: an outlined Cfg leaves rather than returns, and has no resume points")
            }
            Outcome::Stopped(_) => unreachable!("an origin's run has no span to stop"),
        }
    }

    // Takes b's exit for its outputs `outs` (borrowed): Ok(the next block, or
    // the handler a raise went to), or the frame's (or outlined Cfg's)
    // outcome. `clear` is the version that ran, whose values are dropped once
    // the args are taken.
    unsafe fn take_exit(&mut self, b: BlockId, outs: &[*mut PyObject], clear: Option<usize>) -> Result<BlockId, Outcome> {
        let blk = &self.cfg[b];
        let out = |o: &OutIdx| outs[*o as usize];
        let how = match &blk.term {
            Terminator::Raise { args, .. } => {
                self.set_instr(blk.term_origin as c_int);
                let exc = args.first().map_or(std::ptr::null_mut(), out);
                let cause = args.get(1).map_or(std::ptr::null_mut(), out);
                Some(status(tr_do_raise(exc, cause)).unwrap_err())
            }
            Terminator::Reraise { exc, lasti, .. } => {
                let instr = lasti.map_or(blk.term_origin as c_int, |l| ffi::PyLong_AsLong(out(&l)) as c_int);
                self.set_instr(instr);
                ffi::PyErr_SetRaisedException(xincref(out(exc)));
                Some(Exc::Reraise)
            }
            _ => None,
        };
        if let (Some(how), Some(unwind)) = (how, blk.term.unwind()) {
            if how == Exc::Error {
                self.traceback_here();
            }
            let take = |os: &[OutIdx]| -> Vec<*mut PyObject> { os.iter().map(|o| xincref(out(o))).collect() };
            let TermUnwind::Block(k, args) = unwind else {
                if let Some(vi) = clear {
                    self.clear_at_exit(b, vi);
                }
                return Err(Outcome::Error);
            };
            let (h, args) = (blk.handlers[*k], take(args));
            if let Some(vi) = clear {
                self.clear_at_exit(b, vi);
            }
            self.enter_handler(h, args);
            return Ok(h);
        }
        let exit = match &blk.term {
            Terminator::Jump(e) => e,
            Terminator::Branch { cond, then, else_ } => {
                if outs[*cond as usize] == ffi::Py_True() {
                    then
                } else {
                    else_
                }
            }
            Terminator::Raise { .. } | Terminator::Reraise { .. } => unreachable!("handled above"),
            Terminator::Unreachable => unreachable!("verified CFGs have no unreachable terminators"),
        };
        let take = |os: &[OutIdx]| -> Vec<*mut PyObject> { os.iter().map(|o| xincref(outs[*o as usize])).collect() };
        match exit {
            Exit::Block(edge) => {
                let mut args = std::mem::take(&mut self.edge);
                args.extend(edge.args.iter().map(|o| xincref(outs[*o as usize])));
                if let Some(vi) = clear {
                    self.clear_at_exit(b, vi);
                }
                self.enter(edge.target, args.drain(..), |_| unreachable!("edges have no implicit params"));
                self.edge = args;
                Ok(edge.target)
            }
            Exit::Return(o) => {
                // A finished frame's f_lineno is its return's line.
                self.set_instr(blk.term_origin as c_int);
                let r = xincref(outs[*o as usize]);
                if let Some(vi) = clear {
                    self.clear_at_exit(b, vi);
                }
                Err(Outcome::Return(r))
            }
            Exit::Leave(os) => {
                let r = take(os);
                if let Some(vi) = clear {
                    self.clear_at_exit(b, vi);
                }
                Err(Outcome::Leave(r))
            }
        }
    }

    // Where CPython checks its eval breaker (RESUME, after calls,
    // JUMP_BACKWARD), run it; it can raise (e.g. KeyboardInterrupt).
    #[inline]
    unsafe fn breaker(&self) -> R {
        if (*self.breaker_flag).load(std::sync::atomic::Ordering::Relaxed) != 0 && !super::trampoline::eval_breaker() {
            return Err(Exc::Error);
        }
        Ok(())
    }

    // An exception is set; find where the segment covering body[at] of the
    // version sends it, and enter the handler if it's in this Cfg.
    unsafe fn unwind(&mut self, b: BlockId, ver: usize, at: usize, how: Exc) -> Unwound {
        if how == Exc::Error {
            // A failure replaces the pending exception, which is what CPython
            // does too.
            self.traceback_here();
        }
        let cfg = self.cfg;
        let blk = &cfg[b];
        let take = |args: &[ValueId]| -> Vec<*mut PyObject> { args.iter().map(|a| xincref(self.val(*a))).collect() };
        let (h, args) = match super::live::segment_at(&blk.versions[ver], at).map(|s| &s.handler) {
            None | Some(Handler::Propagate) => {
                self.clear(b, ver);
                return Unwound::Propagate;
            }
            Some(Handler::Block(k, args)) => (blk.handlers[*k], take(args)),
            Some(Handler::Outer(k, args)) => {
                let args = take(args);
                self.clear(b, ver);
                return Unwound::Outer(*k, args);
            }
        };
        self.clear(b, ver);
        self.enter_handler(h, args);
        Unwound::Block(h)
    }

    // Enters handler block h with `args` (owned) for its explicit params and
    // the pending exception (and lasti) for the implicit ones.
    unsafe fn enter_handler(&mut self, h: BlockId, args: Vec<*mut PyObject>) {
        let lasti = ffi::PyLong_FromLong(tr_frame_get_instr(self.f) as _);
        let exc = ffi::PyErr_GetRaisedException();
        let (mut lasti, mut exc) = (Some(lasti), Some(exc));
        self.enter(h, args, |kind| match kind {
            ParamKind::Lasti => lasti.take().expect("one lasti param"),
            ParamKind::Exc => exc.take().expect("one exc param"),
            ParamKind::Stack(_) => unreachable!("stack params are explicit"),
        });
        // A handler without a lasti param doesn't take it.
        if let Some(l) = lasti {
            ffi::Py_XDECREF(l);
        }
    }

    // Runs from block b: from the start of the version select() picks, or at
    // `resume` = (version, index).
    unsafe fn run_blocks(&mut self, mut b: BlockId, mut resume: Option<(usize, usize)>) -> Outcome {
        let cfg = self.cfg;
        'blocks: loop {
            if !self.stop.is_null() {
                let sp = &*self.stop;
                if Some(b) == sp.exit || sp.blocks.binary_search(&b).is_err() || sp.never.contains(&b) {
                    return Outcome::Stopped(b);
                }
            } else if !cfg.spans.is_empty()
                && resume.is_none()
                && let Some(si) = cfg.spans.iter().position(|sp| sp.entry == b)
            {
                match self.span(si) {
                    Some(Ok(next)) => {
                        b = next;
                        continue 'blocks;
                    }
                    Some(Err(outcome)) => return outcome,
                    None => {}
                }
            }
            let blk = &cfg[b];
            // An optimized block (one with an origin) doesn't record.
            let rec = if self.p.origins[b.idx()].is_some() { std::ptr::null_mut() } else { self.scope_rec };
            if rec != self.rec {
                if let Some(old) = self.rec.as_mut() {
                    old.flush();
                }
                effects::set_current(rec);
                self.rec = rec;
            }
            if let Some(rec) = self.rec.as_mut() {
                rec.at(Site::Block(b));
            }
            let (mut vi, first) = match resume.take() {
                Some(r) => r,
                None if self.p.autograd[b.idx()].load(std::sync::atomic::Ordering::Relaxed) && self.wants_autograd(b) => match self.bootstrap(b) {
                    Ok(next) => {
                        b = next;
                        continue 'blocks;
                    }
                    Err(outcome) => return outcome,
                },
                None => match self.select(b) {
                    Some(vi) => (vi, 0),
                    None => match self.miss(b) {
                        Ok(next) => {
                            b = next;
                            continue 'blocks;
                        }
                        Err(outcome) => return outcome,
                    },
                },
            };
            let mut i = first;
            // The region version runs the variant of its place that is for
            // this call. With none (or in a trace, which wouldn't see a
            // region's ops) the next version whose guard holds runs: the
            // version the variants were traced from, whose guard is the
            // region version's, if it comes next.
            while first == 0 && let Some(&head) = blk.versions[vi].body.first() && let Op::Region(r) = self.cfg[head].op {
                let args: Vec<*mut PyObject> = self.cfg[head].args.iter().map(|a| self.val(*a)).collect();
                let called = if super::region::tracing() { Ok(None) } else { self.place_call(&self.cfg.regions[r as usize].variants, &args) };
                match called {
                    Ok(Some(outs)) => {
                        if std::env::var_os("TORCH_RUST_REGION_CHECK").is_some() {
                            self.check_region(b, r, &outs);
                        }
                        self.distribute(head, &outs);
                        let code = &self.p.code[b.idx()][vi];
                        let step = &code.steps[0];
                        self.drop_all(&code.kills[step.kill.0 as usize..step.kill.1 as usize]);
                        i = 1;
                        break;
                    }
                    Ok(None) if super::region::tracing() || vi + 2 == blk.versions.len() => {
                        vi = blk.versions.len() - 1;
                        self.drop_all(&self.p.live.versions[b.idx()][vi].entry_kill);
                        break;
                    }
                    Ok(None) => {
                        vi = self.select_from(b, vi + 1);
                    }
                    Err(()) => match self.unwind(b, vi, 0, Exc::Error) {
                        Unwound::Block(h) => {
                            b = h;
                            continue 'blocks;
                        }
                        Unwound::Outer(k, args) => return Outcome::Outer(k, args),
                        Unwound::Propagate => return Outcome::Error,
                    },
                }
            }
            let ver = &blk.versions[vi];
            let code = &self.p.code[b.idx()][vi];
            let fast = self.rec.is_null();
            if first == 0 && vi + 1 == blk.versions.len() && blk.origin.is_some() && self.c.is_some() && !super::region::tracing() {
                if super::region::observing() {
                    self.region_observe(b, vi);
                }
                if super::region::enabled()
                    && !self.p.region_off[b.idx()].load(std::sync::atomic::Ordering::Relaxed)
                    && self.region_room(b, vi)
                    && super::region::due(&self.p.region_runs[b.idx()], &self.p.region_off[b.idx()])
                    && self.region_worthy(b, vi)
                    && self.region_trace(b, vi)
                {
                    i = code.steps.len();
                }
            }
            while i < code.steps.len() {
                // Steps' kill ranges are within `kills`, and instrs index
                // values within `vals`.
                let step = code.steps.get_unchecked(i);
                let kills = code.kills.get_unchecked(step.kill.0 as usize..step.kill.1 as usize);
                i += 1;
                if fast && step.op != Fast::No {
                    // A value is null until its instr defines it.
                    let vals = self.vals.as_mut_ptr();
                    match step.op {
                        Fast::Const(k) => {
                            debug_assert!(self.vals[step.v.idx()].is_null());
                            *vals.add(step.v.idx()) = xincref(self.konst(k));
                        }
                        Fast::LoadLocal(slot) => {
                            debug_assert!(self.vals[step.v.idx()].is_null());
                            *vals.add(step.v.idx()) = xincref(*self.local(slot));
                        }
                        Fast::StoreLocal(slot, x) => {
                            let p = self.local(slot);
                            let old = std::mem::replace(&mut *p, xincref(*vals.add(x.idx())));
                            ffi::Py_XDECREF(old);
                        }
                        Fast::No => unreachable!(),
                    }
                    for k in kills {
                        let p = vals.add(k.idx());
                        let old = std::mem::replace(&mut *p, std::ptr::null_mut());
                        ffi::Py_XDECREF(old);
                    }
                    continue;
                }
                match self.slow_step(b, vi, i - 1, step, kills) {
                    Next::Step => {}
                    Next::Block(h) => {
                        b = h;
                        continue 'blocks;
                    }
                    Next::Done(outcome) => return outcome,
                }
            }
            let edge = match &blk.term {
                Terminator::Jump(_) => code.edges[0].as_ref(),
                Terminator::Branch { cond, .. } => {
                    let taken = self.vals[ver.outputs[*cond as usize].idx()] == ffi::Py_True();
                    code.edges[if taken { 0 } else { 1 }].as_ref()
                }
                _ => None,
            };
            if let Some(t) = edge {
                self.transfer(t);
                b = t.target;
                continue;
            }
            let mut outs = std::mem::take(&mut self.outs);
            outs.extend(ver.outputs.iter().map(|o| self.vals[o.idx()]));
            let next = self.take_exit(b, &outs, Some(vi));
            outs.clear();
            self.outs = outs;
            match next {
                Ok(next) => b = next,
                Err(outcome) => return outcome,
            }
        }
    }

    // Runs the variant of a place that is for a call with these args (the
    // place's args, borrowed): the first of those to ask (see
    // region::Variants) whose own checks hold and whose region takes the
    // call. As region::call: None if there is none.
    unsafe fn place_call(&self, variants: &super::region::Variants, args: &[*mut PyObject]) -> Result<Option<Vec<*mut PyObject>>, ()> {
        let mut key = None;
        let mut i = 0;
        while let Some(v) = variants.candidate(args, &mut key, i) {
            i += 1;
            let own = v.facts_hold(args)
                && v.checks.iter().all(|c| match c {
                    super::region::Own::Is(k, o) => args[*k as usize] == o.as_ptr(),
                    super::region::Own::Check(k, c) => self.check(c, k.map_or(std::ptr::null_mut(), |k| args[k as usize])),
                });
            if !own {
                super::region::passed_over();
                continue;
            }
            match super::region::call(&v.spec, args) {
                Ok(None) => {}
                r => {
                    if matches!(r, Ok(Some(_))) {
                        variants.ran(&v);
                    }
                    return r;
                }
            }
        }
        Ok(None)
    }

    // Whether block b, about to run its last version vi, may be traced into
    // another region: its place (as it is now: this frame's Cfg may be from
    // before the place had regions) has room for this run's key.
    unsafe fn region_room(&self, b: BlockId, vi: usize) -> bool {
        let Some(c) = self.c else { return false };
        let cfg = super::region_cfg(c, self.p as *const Prepared as usize).unwrap_or(self.cfg);
        let Some(place) = super::place_of(cfg, b) else { return true };
        let blk = &self.cfg[b];
        let args: Vec<*mut PyObject> = blk.params.iter().chain(&blk.versions[vi].inputs).map(|x| self.val(*x)).collect();
        place.variants.room(&args)
    }

    // Debugging (TORCH_RUST_REGION_CHECK): runs the version region r was
    // traced from (the last) on the inputs it just ran on, and has
    // torch._rust_inductor compare their outputs.
    unsafe fn check_region(&mut self, b: BlockId, r: u32, outs: &[*mut PyObject]) {
        let cfg = self.cfg;
        let src = &cfg[b].versions[cfg[b].versions.len() - 1];
        let code = &self.p.code[b.idx()][cfg[b].versions.len() - 1];
        let ok = code.steps.iter().all(|step| {
            self.set_instr(cfg[step.v].origin as c_int);
            self.exec(step.v, &cfg[step.v]).is_ok()
        });
        let py = pyo3::Python::assume_attached();
        if ok {
            let item = |p: *mut PyObject| pyo3::Bound::from_borrowed_ptr_or_opt(py, p).map_or_else(|| py.None(), |b| b.unbind());
            let mine: Vec<_> = outs.iter().map(|p| item(*p)).collect();
            let theirs: Vec<_> = src.outputs.iter().map(|o| item(self.val(*o))).collect();
            let name = &self.cfg.regions[r as usize].name;
            let check = || -> pyo3::PyResult<()> {
                use pyo3::types::PyAnyMethods;
                py.import("torch._rust_inductor")?.as_any().call_method1("check", (name, mine, theirs))?;
                Ok(())
            };
            if let Err(e) = check() {
                e.print(py);
            }
        } else {
            pyo3::PyErr::fetch(py);
        }
        for v in &src.body {
            self.set(*v, std::ptr::null_mut());
        }
    }

    // What a region of version vi of block b is called.
    fn block_name(&self, c: &Compiled, b: BlockId, vi: usize) -> String {
        let lines: Vec<i32> = self.cfg[b].versions[vi].body.iter().map(|v| c.line(self.cfg[*v].origin)).filter(|l| *l > 0).collect();
        format!("{}:bb{} lines {}-{}", c.qualname, b.0, lines.iter().min().unwrap_or(&0), lines.iter().max().unwrap_or(&0))
    }

    // Whether version vi of block b does enough to be worth tracing: its
    // instrs ran at least two torch ops, those of the frames a call entered
    // included (see site_ops). Remembered when it doesn't, and said.
    unsafe fn region_worthy(&self, b: BlockId, vi: usize) -> bool {
        let ops = |v: &ValueId| if matches!(self.cfg[*v].op, Op::Aten { .. }) { 1 } else { super::site_ops(self.p, *v, &self.p.clones, 0) };
        let work: usize = self.cfg[b].versions[vi].body.iter().map(ops).sum();
        // A write to something that outlives the block isn't replayed by a
        // region: the version keeps running as it is, so the write is made
        // on every call, with that call's value.
        let body = &self.cfg[b].versions[vi].body;
        let code = self.c.map_or(0, |c| c.code);
        let escapes = body.iter().any(|v| super::escaping_write(self.cfg, code, body, *v));
        if work < 2 || escapes {
            self.p.region_off[b.idx()].store(true, std::sync::atomic::Ordering::Relaxed);
            if let (Some(c), true) = (self.c, work > 0) {
                let (state, why) = if escapes {
                    ("refused", "it stores into something that outlives it, so it runs as it is".to_string())
                } else {
                    ("too small", format!("{work} op, fewer than 2"))
                };
                super::note_untraced(self.block_name(c, b, vi), state, why);
            }
        }
        work >= 2 && !escapes
    }

    // At span si's entry: runs a region of it whose checks hold, or traces
    // the span into one. Some(where the frame goes on) if either ran it;
    // None: run its blocks as usual.
    unsafe fn span(&mut self, si: usize) -> Option<Result<BlockId, Outcome>> {
        let cfg = self.cfg;
        let sp = &cfg.spans[si];
        if super::region::tracing() || self.c.is_none() {
            return None;
        }
        if !super::region::enabled() {
            if super::region::observing() && !sp.state.off.load(std::sync::atomic::Ordering::Relaxed) {
                let args = self.span_args(sp)?;
                super::region::observe(&mut sp.state.types.lock().unwrap(), &args);
                args.into_iter().for_each(|a| ffi::Py_XDECREF(a));
            }
            return None;
        }
        let args = self.span_args(sp)?;
        let called = match self.place_call(&sp.state.regions, &args) {
            Ok(None) => None,
            r => Some(r),
        };
        let traced = match called {
            None if !sp.state.off.load(std::sync::atomic::Ordering::Relaxed) && sp.state.regions.room(&args) => {
                if super::region::observing() {
                    super::region::observe(&mut sp.state.types.lock().unwrap(), &args);
                }
                if super::region::due(&sp.state.runs, &sp.state.off) {
                    self.span_trace(sp, &args)
                } else {
                    None
                }
            }
            _ => None,
        };
        args.into_iter().for_each(|a| ffi::Py_XDECREF(a));
        match called {
            Some(Ok(Some(outs))) => Some(self.span_leave(sp, outs)),
            Some(_) => Some(match self.unwind(sp.entry, 0, 0, Exc::Error) {
                Unwound::Block(h) => Ok(h),
                Unwound::Outer(k, args) => Err(Outcome::Outer(k, args)),
                Unwound::Propagate => Err(Outcome::Error),
            }),
            None => traced,
        }
    }

    // Span sp's args (owned), or None (no exception set) if an attribute
    // isn't there to read.
    unsafe fn span_args(&self, sp: &Span) -> Option<Vec<*mut PyObject>> {
        let mut args = Vec::with_capacity(sp.inputs.len());
        for (slot, attr) in &sp.inputs {
            let x = *self.local(*slot);
            let a = match attr {
                None => xincref(x),
                Some(CELL) if !x.is_null() => PyCell_Get(x),
                Some(n) if !x.is_null() && (*n as usize) < self.p.names.0.len() => ffi::PyObject_GetAttr(x, self.name(*n)),
                Some(_) => std::ptr::null_mut(),
            };
            if a.is_null() && attr.is_some() {
                ffi::PyErr_Clear();
                args.into_iter().for_each(|a| ffi::Py_XDECREF(a));
                return None;
            }
            args.push(a);
        }
        args.extend(self.cfg.explicit_params(sp.entry).into_iter().map(|p| xincref(self.val(p))));
        Some(args)
    }

    // Leaves span sp with a region's outputs (owned): the locals it leaves,
    // then the exit's explicit params, or what it returns if it does.
    unsafe fn span_leave(&mut self, sp: &Span, outs: Vec<*mut PyObject>) -> Result<BlockId, Outcome> {
        for (s, o) in sp.live_out.iter().zip(&outs) {
            let old = std::mem::replace(&mut *self.local(*s), *o);
            ffi::Py_XDECREF(old);
        }
        match sp.exit {
            Some(x) => {
                for (q, o) in self.cfg.explicit_params(x).into_iter().zip(&outs[sp.live_out.len()..]) {
                    self.set(q, *o);
                }
                Ok(x)
            }
            None => Err(Outcome::Return(outs[sp.live_out.len()])),
        }
    }

    // Runs span sp's blocks under a trace (see region.rs), stopping at its
    // exit, and adds the region traced. Some(where the frame goes on) if it
    // ran them to the exit; None if it didn't run them (or gave up partway,
    // having put back the locals it read: they only compute, so running them
    // again from the start is the same).
    unsafe fn span_trace(&mut self, sp: &Span, args: &[*mut PyObject]) -> Option<Result<BlockId, Outcome>> {
        let held: Vec<*mut PyObject> = sp.live_in.iter().map(|s| xincref(*self.local(*s))).collect();
        // The cells it reads hold what the trace passes while it runs.
        let cells: Vec<(*mut PyObject, *mut PyObject)> = sp.inputs.iter().filter(|(_, a)| *a == Some(CELL)).map(|(s, _)| (*self.local(*s), PyCell_Get(*self.local(*s)))).collect();
        let mut ran = false;
        let mut ended: Option<Outcome> = None;
        let this = self as *mut Self;
        let mut body = |objs: &[*mut PyObject]| -> Result<Vec<*mut PyObject>, ()> {
            let this = &mut *this;
            ran = true;
            this.reading = reads::on();
            for ((s, attr), o) in sp.inputs.iter().zip(objs) {
                if attr.is_none() {
                    let old = std::mem::replace(&mut *this.local(*s), xincref(*o));
                    ffi::Py_XDECREF(old);
                } else if *attr == Some(CELL) && PyCell_Set(*this.local(*s), *o) < 0 {
                    return Err(());
                }
            }
            for (p, o) in this.cfg.explicit_params(sp.entry).into_iter().zip(&objs[sp.inputs.len()..]) {
                this.set(p, xincref(*o));
            }
            this.stop = sp;
            let out = this.run_blocks(sp.entry, None);
            this.stop = std::ptr::null();
            this.reading = false;
            let mut outs: Vec<*mut PyObject> = sp.live_out.iter().map(|s| xincref(*this.local(*s))).collect();
            super::region::kept(&outs);
            // A run that went on past a refusal (its code caught what the
            // trace raised, or it hands a stand-in on: see region::kept) is
            // no run of the program's.
            let stands = !super::region::refused();
            let done = match &out {
                Outcome::Stopped(x) => {
                    outs.extend(this.cfg.explicit_params(*x).into_iter().map(|q| xincref(this.val(q))));
                    stands && Some(*x) == sp.exit && !sp.never.contains(x)
                }
                Outcome::Return(v) => {
                    outs.push(xincref(*v));
                    stands && sp.exit.is_none()
                }
                _ => false,
            };
            ended = Some(out);
            if done {
                return Ok(outs);
            }
            outs.into_iter().for_each(|o| ffi::Py_XDECREF(o));
            if pyo3::ffi::PyErr_Occurred().is_null() && !stands {
                super::region::raise_refusal(pyo3::Python::assume_attached());
            } else if pyo3::ffi::PyErr_Occurred().is_null() {
                ffi::PyErr_SetString(ffi::PyExc_RuntimeError, c"the span left by another way or ran a block never seen run".as_ptr());
            }
            Err(())
        };
        let types = sp.state.types.lock().unwrap().clone();
        let passed: Vec<u32> = sp.passed.iter().map(|k| sp.inputs.len() as u32 + k).collect();
        // A span is traced MAX_GAVE_UP times at most, whatever gave it up.
        super::covered::place(sp.state.gave_up.load(std::sync::atomic::Ordering::Relaxed));
        let traced = super::outside_executor(|| super::region::trace(&sp.name, args, &types, &passed, true, sp.asked, &mut body));
        for (cell, old) in cells {
            PyCell_Set(cell, old);
            ffi::Py_XDECREF(old);
        }
        let debug = std::env::var_os("TORCH_RUST_REGION_DEBUG").is_some();
        sp.state.runs.store(0, std::sync::atomic::Ordering::Relaxed);
        let stands = !super::region::refused();
        let leave = match ended {
            Some(Outcome::Stopped(x)) if stands && Some(x) == sp.exit && !sp.never.contains(&x) => Some(Ok(x)),
            Some(Outcome::Return(v)) if stands && sp.exit.is_none() => Some(Err(Outcome::Return(v))),
            Some(Outcome::Stopped(x)) => {
                for q in &self.cfg[x].params {
                    self.set(*q, std::ptr::null_mut());
                }
                None
            }
            Some(Outcome::Return(v)) => {
                ffi::Py_XDECREF(v);
                None
            }
            _ => None,
        };
        let Some(leave) = leave else {
            // What the run wrote before it gave up is written again by the
            // blocks, which run from the span's entry.
            writes::undo();
            if debug {
                let at = match &ended {
                    Some(Outcome::Stopped(x)) => format!("stopped at bb{}", x.0),
                    Some(Outcome::Error) => "raised".into(),
                    None => "didn't run".into(),
                    _ => "left otherwise".into(),
                };
                eprintln!("region {}: gave up, {at}: {}", sp.name, traced.as_ref().err().map_or("?".into(), |e| e.to_string()));
            }
            // Another call may stay on the ways seen, or find the code this
            // one was refused for judged (see covered.rs): it is traced
            // again, a few times.
            super::covered::later();
            if sp.state.gave_up.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1 >= super::region::MAX_GAVE_UP {
                sp.state.off.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            for (s, o) in sp.live_in.iter().zip(held) {
                let old = std::mem::replace(&mut *self.local(*s), o);
                ffi::Py_XDECREF(old);
            }
            // The entry's stack values too: the run that gave up dropped each
            // at its last use, and the blocks are about to run from them.
            for (p, o) in self.cfg.explicit_params(sp.entry).into_iter().zip(&args[sp.inputs.len()..]) {
                self.set(p, xincref(*o));
            }
            pyo3::ffi::PyErr_Clear();
            return None;
        };
        held.into_iter().for_each(|o| ffi::Py_XDECREF(o));
        // What the traced run leaves to the program holds floats, not the
        // float subclass the trace computes with (see region::plain).
        for s in &sp.live_out {
            let slot = self.local(*s);
            let new = super::outside_executor(|| super::region::plain(*slot));
            let old = std::mem::replace(&mut *slot, new);
            ffi::Py_XDECREF(old);
        }
        let leave = match leave {
            Ok(x) => {
                for q in self.cfg.explicit_params(x) {
                    let new = super::outside_executor(|| super::region::plain(self.val(q)));
                    self.set(q, new);
                }
                Ok(x)
            }
            Err(Outcome::Return(v)) => {
                let new = super::outside_executor(|| super::region::plain(v));
                ffi::Py_XDECREF(v);
                Err(Outcome::Return(new))
            }
            other => other,
        };
        match traced {
            Ok(Some(t)) => sp.state.regions.push(t.variant),
            Ok(None) => sp.state.off.store(true, std::sync::atomic::Ordering::Relaxed),
            Err(e) => {
                if debug {
                    eprintln!("region {}: trace failed: {e}", sp.name);
                }
                sp.state.off.store(true, std::sync::atomic::Ordering::Relaxed);
            }
        }
        Some(leave)
    }

    // A call of a callable merge_across made opaque, while a region is
    // traced: torch._rust_inductor._opaque_call(callee, *args) records it as
    // one graph node.
    unsafe fn opaque_call(&self, args: *const *mut PyObject, argc: c_int, kw: *mut PyObject) -> *mut PyObject {
        let py = pyo3::Python::assume_attached();
        let helper = match opaque_helper(py) {
            Ok(h) => h,
            Err(e) => {
                e.restore(py);
                return std::ptr::null_mut();
            }
        };
        let (callee, rest, total) = if (*args).is_null() { (*args.add(1), args.add(2), argc) } else { (*args, args.add(1), argc + 1) };
        let mut all: Vec<*mut PyObject> = vec![callee];
        all.extend((0..total as usize).map(|i| *rest.add(i)));
        let nkw = if kw.is_null() { 0 } else { ffi::PyTuple_GET_SIZE(kw) as usize };
        ffi::PyObject_Vectorcall(helper.as_ptr(), all.as_ptr(), all.len() - nkw, kw)
    }

    // Whether the call at code unit `origin` of this frame's code was made
    // opaque (see _cfg_opaque_site).
    fn opaque_site(&self, origin: u32) -> bool {
        self.c.is_some_and(|c| super::opaque_site(c.code, origin))
    }

    // opaque_call for CALL_FUNCTION_EX: `callee(*args, **kwargs)`.
    unsafe fn opaque_call_ex(&self, callee: *mut PyObject, args: *mut PyObject, kwargs: *mut PyObject) -> *mut PyObject {
        let py = pyo3::Python::assume_attached();
        let helper = match opaque_helper(py) {
            Ok(h) => h,
            Err(e) => {
                e.restore(py);
                return std::ptr::null_mut();
            }
        };
        let items = ffi::PySequence_Tuple(args);
        if items.is_null() {
            return std::ptr::null_mut();
        }
        let n = ffi::PyTuple_GET_SIZE(items);
        let all = ffi::PyTuple_New(n + 1);
        ffi::Py_INCREF(callee);
        ffi::PyTuple_SET_ITEM(all, 0, callee);
        for i in 0..n {
            let x = ffi::PyTuple_GET_ITEM(items, i);
            ffi::Py_INCREF(x);
            ffi::PyTuple_SET_ITEM(all, i + 1, x);
        }
        ffi::Py_DECREF(items);
        let out = ffi::PyObject_Call(helper.as_ptr(), all, kwargs);
        ffi::Py_DECREF(all);
        out
    }

    // Joins this run's params and inputs of block b's version vi (the last)
    // into the types its trace gets.
    unsafe fn region_observe(&mut self, b: BlockId, vi: usize) {
        if !self.p.region_off[b.idx()].load(std::sync::atomic::Ordering::Relaxed) {
            let blk = &self.cfg[b];
            let args: Vec<*mut PyObject> = blk.params.iter().chain(&blk.versions[vi].inputs).map(|x| self.val(*x)).collect();
            super::region::observe(&mut *self.p.region_types[b.idx()].get(), &args);
        }
    }

    // Traces this run of version vi of block b into a region (see
    // region.rs), its inputs already read. True: the body ran, its values set
    // as run_blocks' steps would leave them; false: run it as usual.
    unsafe fn region_trace(&mut self, b: BlockId, vi: usize) -> bool {
        let cfg = self.cfg;
        let blk = &cfg[b];
        let ver = &blk.versions[vi];
        let code = &self.p.code[b.idx()][vi];
        let ids: Vec<ValueId> = blk.params.iter().chain(&ver.inputs).copied().collect();
        let args: Vec<*mut PyObject> = ids.iter().map(|x| self.val(*x)).collect();
        // Params the version doesn't use, already dropped by select().
        let entry_kill = &self.p.live.versions[b.idx()][vi].entry_kill;
        let unused: Vec<u32> = ids.iter().enumerate().filter(|(_, x)| entry_kill.contains(x)).map(|(i, _)| i as u32).collect();
        // Kept to run the body again when the trace fails (it's pure).
        let held: Vec<*mut PyObject> = args.iter().map(|a| xincref(*a)).collect();
        let mut failed = false;
        let mut ran = false;
        let this = self as *mut Self;
        let mut body = |objs: &[*mut PyObject]| -> Result<Vec<*mut PyObject>, ()> {
            let this = &mut *this;
            ran = true;
            for (i, (x, o)) in ids.iter().zip(objs).enumerate() {
                if !unused.contains(&(i as u32)) {
                    this.set(*x, xincref(*o));
                }
            }
            this.reading = reads::on();
            for step in &code.steps {
                let v = step.v;
                this.set_instr(cfg[v].origin as c_int);
                let callee = this.callee(step.flags, &cfg[v]);
                if this.guarded(v, &cfg[v], callee).is_err() {
                    failed = true;
                    this.reading = false;
                    return Err(());
                }
                this.drop_all(&code.kills[step.kill.0 as usize..step.kill.1 as usize]);
            }
            this.reading = false;
            // A stand-in for a float that the run hands on where it cannot
            // be given back refuses the trace too (see region::kept).
            super::region::kept(&ver.outputs.iter().map(|o| this.val(*o)).collect::<Vec<_>>());
            if super::region::refused() {
                // Its code caught what the trace raised to refuse it, and
                // went on: no run of the program's.
                failed = true;
                super::region::raise_refusal(pyo3::Python::assume_attached());
                return Err(());
            }
            Ok(ver.outputs.iter().map(|o| xincref(this.val(*o))).collect())
        };
        let c = self.c.expect("checked by the caller");
        let name = self.block_name(c, b, vi);
        // What the region needn't check: what the version doesn't read, or only
        // passes on.
        let read: std::collections::HashSet<ValueId> = ver.body.iter().flat_map(|v| cfg[*v].args.iter().copied()).collect();
        let unread: Vec<u32> = (0..ids.len() as u32).filter(|i| unused.contains(i) || !read.contains(&ids[*i as usize])).collect();
        let types = (*self.p.region_types[b.idx()].get()).clone();
        let later = &self.p.region_later[b.idx()];
        super::covered::place(later.load(std::sync::atomic::Ordering::Relaxed));
        let traced = super::outside_executor(|| super::region::trace(&name, &args, &types, &unread, false, false, &mut body));
        // Refused for code no call ran before: the block runs it now, as
        // written, and is traced again (see covered.rs).
        let again = super::covered::later();
        if again {
            later.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        let debug = std::env::var_os("TORCH_RUST_REGION_DEBUG").is_some();
        if failed {
            // The body raised under tracing (which also raises on what only
            // tracing can't do, like reading a tensor's value): run it again
            // as usual, from its inputs.
            if debug && let Err(e) = &traced {
                eprintln!("region {name}: the body raised under tracing: {e}");
            }
            // What it wrote before it raised is written again by the run
            // that follows.
            writes::undo();
            if again {
                self.p.region_runs[b.idx()].store(0, std::sync::atomic::Ordering::Relaxed);
            } else {
                self.p.region_off[b.idx()].store(true, std::sync::atomic::Ordering::Relaxed);
            }
            for v in &ver.body {
                self.set(*v, std::ptr::null_mut());
            }
            for (i, (x, o)) in ids.iter().zip(held).enumerate() {
                if unused.contains(&(i as u32)) {
                    ffi::Py_XDECREF(o);
                } else {
                    self.set(*x, o);
                }
            }
            return false;
        }
        held.into_iter().for_each(|o| ffi::Py_XDECREF(o));
        if ran {
            // The program gets floats, not the float subclass the trace
            // computes with (see region::plain).
            for o in &ver.outputs {
                let new = super::outside_executor(|| super::region::plain(self.val(*o)));
                self.set(*o, new);
            }
        }
        match traced {
            Ok(Some(t)) if ran => match super::add_region(c, self.p as *const Prepared as usize, b, t) {
                Ok(true) => {}
                Ok(false) => {
                    if debug {
                        eprintln!("region {name}: the CFG changed since the block ran");
                    }
                }
                Err(e) => {
                    if debug {
                        eprintln!("region {name}: not added: {e}");
                    }
                    self.p.region_off[b.idx()].store(true, std::sync::atomic::Ordering::Relaxed);
                }
            },
            Ok(_) => self.p.region_off[b.idx()].store(true, std::sync::atomic::Ordering::Relaxed),
            Err(_) if again => {}
            Err(e) => {
                if debug {
                    eprintln!("region {name}: trace failed: {e}");
                    e.display(pyo3::Python::assume_attached());
                }
                self.p.region_off[b.idx()].store(true, std::sync::atomic::Ordering::Relaxed);
            }
        }
        self.p.region_runs[b.idx()].store(0, std::sync::atomic::Ordering::Relaxed);
        ran
    }

    // Runs body[i] of version `vi` of block b (step) and drops `kills`.
    #[inline(always)]
    unsafe fn slow_step(&mut self, b: BlockId, vi: usize, i: usize, step: &Step, kills: &[ValueId]) -> Next {
        let v = &step.v;
        let ins = &self.cfg[*v];
        let flags = step.flags;
        if flags & FLAG_YIELD != 0 {
            let (value, stack) = ins.args.split_last().expect("Yield has the yielded value");
            for a in stack {
                tr_frame_push(self.f, xincref(self.val(*a)));
            }
            let value = xincref(self.val(*value));
            self.clear(b, vi);
            return Next::Done(Outcome::Yield { value, unit: ins.origin });
        }
        if flags & FLAG_RAISE != 0 {
            self.set_instr(ins.origin as c_int);
        }
        // The callable a call's frame (if any) would be entered for directly.
        let callee = self.callee(flags, ins);
        let rec = self.rec;
        if !rec.is_null() {
            // The function a frame entered for the instr would run directly:
            // the callable's, or the one its (or the receiver's) class runs.
            let direct = match (callee, &ins.op) {
                (Some(c), _) if !c.is_null() => {
                    let func = tr_partial_func(c);
                    let via_type = if func.is_null() { tr_type_function(c, 0) } else { std::ptr::null_mut() };
                    Some(if !func.is_null() {
                        (func, effects::Via::Partial)
                    } else if via_type.is_null() {
                        (c, effects::Via::Callable)
                    } else {
                        (via_type, effects::Via::Call)
                    })
                }
                (None, Op::LoadAttr(_) | Op::LoadMethod(_)) => {
                    let f = tr_type_function(self.val(ins.args[0]), 1);
                    (!f.is_null()).then_some((f, effects::Via::Getattr))
                }
                _ => None,
            };
            if let Some((f, via)) = direct {
                effects::record_callee(&self.p.callees, *v, f, via);
            }
            (*rec).at(Site::Instr(*v));
            // A partial's function is entered by the partial, not by the
            // instr: its frames stay indirect ones.
            (*rec).set_callee(direct.map_or(std::ptr::null_mut(), |(f, via)| if via == effects::Via::Partial { callee.unwrap_or(f) } else { f }));
        }
        let clone = if self.p.clones.is_empty() { None } else { self.p.clones.get(v) };
        if let Some(cl) = clone {
            super::NEXT_CLONE.set((cl.code.as_ptr() as usize, self.f as usize, Arc::as_ptr(&cl.compiled) as usize));
        }
        let done = self.guarded(*v, ins, callee);
        if clone.is_some() {
            super::NEXT_CLONE.set((0, 0, 0));
        }
        if !rec.is_null() {
            (*rec).set_callee(std::ptr::null_mut());
            (*rec).at(Site::Block(b));
        }
        if let Err(how) = done.and_then(|()| if callee.is_some() { self.breaker() } else { Ok(()) }) {
            return match self.unwind(b, vi, i, how) {
                Unwound::Block(h) => Next::Block(h),
                Unwound::Outer(k, args) => Next::Done(Outcome::Outer(k, args)),
                Unwound::Propagate => Next::Done(Outcome::Error),
            };
        }
        if flags & FLAG_VALUE != 0 {
            self.observe(*v);
        }
        self.drop_all(kills);
        Next::Step
    }

    // The callable instr `ins` (with `flags`) calls, if it is a call.
    #[inline(always)]
    unsafe fn callee(&self, flags: u8, ins: &Instr) -> Option<*mut PyObject> {
        if flags & FLAG_CALL != 0 {
            let c = self.val(ins.args[0]);
            Some(if c.is_null() { self.val(ins.args[1]) } else { c })
        } else if flags & FLAG_CALL_EX != 0 {
            Some(self.val(ins.args[0]))
        } else {
            None
        }
    }

    // Runs instr v (a call of `callee`, if it is one), but for what is asked
    // first. A backward is nothing a region does again: a traced body that
    // comes to one is refused before the engine runs, and anywhere else the
    // engine runs outside the executor (see backward.rs). And a traced body
    // runs what the optimizer's verdict covered, and nothing else (see
    // covered.rs).
    #[inline(always)]
    unsafe fn guarded(&mut self, v: ValueId, ins: &Instr, callee: Option<*mut PyObject>) -> R {
        match callee {
            Some(f) if super::backward::is(f) => {
                if self.reading {
                    super::backward::refuse();
                    Err(Exc::Error)
                } else {
                    super::outside_executor(|| self.exec(v, ins))
                }
            }
            _ if self.reading => match super::covered::step(self.c, self.under_body, ins.origin, callee, (*self.p.callees.get()).get(&v)) {
                Ok(trusted) => {
                    let _trusted = trusted.then(super::covered::Trusted::enter);
                    self.exec(v, ins)
                }
                Err(why) => {
                    super::region::refuse(pyo3::Python::assume_attached(), &why, super::covered::pending());
                    Err(Exc::Error)
                }
            },
            _ => self.exec(v, ins),
        }
    }

    // Whether LoadAttr/LoadMethod `ins` of name `n` reads what the span being
    // traced takes as an input (an attribute of a local it only reads
    // attributes of): the span's region is handed that, on every call.
    unsafe fn span_input(&self, ins: &Instr, n: u32) -> bool {
        match (self.stop.as_ref(), &self.cfg[ins.args[0]].op) {
            (Some(sp), Op::LoadLocal { slot } | Op::LoadLocalChecked { slot }) => sp.inputs.contains(&(*slot, Some(n))),
            _ => false,
        }
    }

    // Reports the write the generic instruction `ins` is about to make (a
    // subscript store or deletion, an attribute deletion) to the trace (see
    // writes.rs).
    unsafe fn note_write(&self, opcode: u8, oparg: u32, ins: &Instr) -> writes::Token {
        use crate::cfg::opcodes::*;
        let a = |k: usize| self.val(ins.args[k]);
        match opcode {
            STORE_SUBSCR => writes::store_item(a(1), a(2), a(0), None),
            DELETE_SUBSCR => writes::store_item(a(0), a(1), std::ptr::null_mut(), None),
            DELETE_ATTR => writes::store_attr(a(0), self.name(oparg), std::ptr::null_mut(), false),
            _ => 0,
        }
    }

    // Reports what the generic instruction `ins` (just run; `ran`: it entered
    // a Python frame) read to the trace's reads (see reads.rs).
    unsafe fn note_read(&self, opcode: u8, oparg: u32, ins: &Instr, ran: bool) {
        use crate::cfg::opcodes::*;
        let a = |k: usize| self.val(ins.args[k]);
        match opcode {
            BINARY_SUBSCR => reads::subscript(a(0), a(1), ran),
            CONTAINS_OP => reads::contains(a(1), a(0), ran),
            LOAD_SUPER_ATTR => reads::super_attr(a(1), a(2), self.name(oparg >> 2)),
            // They build or change a container: no read of it.
            BUILD_LIST | BUILD_MAP | BUILD_SET | BUILD_STRING | BUILD_SLICE | BUILD_CONST_KEY_MAP | LIST_APPEND | SET_ADD | MAP_ADD
            | STORE_SUBSCR | DELETE_SUBSCR | STORE_SLICE | STORE_GLOBAL | DELETE_GLOBAL | DELETE_ATTR => {}
            // They read their last operand into the container before it.
            LIST_EXTEND | SET_UPDATE | DICT_UPDATE | DICT_MERGE => reads::used(&[a(ins.args.len() - 1)]),
            CALL_FUNCTION_EX => {
                if reads::reads_args(a(0)) {
                    let all: Vec<*mut PyObject> = ins.args.iter().map(|x| self.val(*x)).collect();
                    reads::used(&all);
                } else {
                    reads::called(a(0));
                }
            }
            _ => {
                let all: Vec<*mut PyObject> = ins.args.iter().map(|x| self.val(*x)).collect();
                reads::used(&all);
            }
        }
    }

    #[inline]
    unsafe fn exec(&mut self, v: ValueId, ins: &Instr) -> R {
        if !self.reading {
            return self.exec_instr(v, ins);
        }
        // A stand-in for a scalar input among the operands: is its value
        // about to be read without it seeing that (see scalars.rs)?
        if ins.args.iter().any(|x| scalars::stand_in(self.val(*x))) {
            let operands: Vec<*mut PyObject> = ins.args.iter().map(|x| self.val(*x)).collect();
            let name = if let Op::StoreAttr(n) = ins.op { self.name(n) } else { std::ptr::null_mut() };
            scalars::before(ins, &operands, name);
        }
        let outer = reads::set_in_instr(true);
        let r = self.exec_instr(v, ins);
        reads::set_in_instr(outer);
        r
    }

    unsafe fn exec_instr(&mut self, v: ValueId, ins: &Instr) -> R {
        let a = |k: usize| self.val(ins.args[k]);
        let obj = match &ins.op {
            Op::Const(i) => xincref(self.konst(*i)),
            // Written by the instr that produces it.
            Op::Project(_) => return Ok(()),
            Op::ToBool => {
                let r = ffi::PyObject_IsTrue(a(0));
                if r < 0 {
                    return Err(Exc::Error);
                }
                if self.reading {
                    reads::truth(a(0));
                }
                new_bool(r != 0)
            }
            Op::IsNone => new_bool(a(0) == ffi::Py_None()),
            Op::LoadLocal { slot } => xincref(*self.local(*slot)),
            Op::LoadLocalChecked { slot } => {
                let x = *self.local(*slot);
                if x.is_null() {
                    tr_unbound_local(self.f, *slot as c_int);
                    return Err(Exc::Error);
                }
                xincref(x)
            }
            Op::StoreLocal { slot } => {
                let p = self.local(*slot);
                let old = std::mem::replace(&mut *p, xincref(a(0)));
                ffi::Py_XDECREF(old);
                return Ok(());
            }
            Op::DeleteLocal { slot } => {
                let p = self.local(*slot);
                if (*p).is_null() {
                    tr_unbound_local(self.f, *slot as c_int);
                    return Err(Exc::Error);
                }
                let old = std::mem::replace(&mut *p, std::ptr::null_mut());
                ffi::Py_DECREF(old);
                return Ok(());
            }
            Op::LoadGlobal(n) => {
                let cache = self.p.globals[self.p.global_slot[v.idx()] as usize].get();
                let x = check(tr_load_global(self.f, *n as c_int, cache))?;
                if self.reading {
                    reads::global(tr_frame_globals(self.f), tr_frame_builtins(self.f), self.name(*n), x);
                }
                x
            }
            Op::LoadGlobalIn { name, globals, builtins } => {
                let name = self.name(*name);
                let mut x = ffi::PyDict_GetItemWithError(self.p.pins[*globals as usize].as_ptr(), name);
                if x.is_null() && ffi::PyErr_Occurred().is_null() {
                    x = ffi::PyDict_GetItemWithError(self.p.pins[*builtins as usize].as_ptr(), name);
                }
                if x.is_null() {
                    if ffi::PyErr_Occurred().is_null() {
                        ffi::PyErr_Format(ffi::PyExc_NameError, c"name '%U' is not defined".as_ptr(), name);
                    }
                    return Err(Exc::Error);
                }
                if self.reading {
                    reads::global(self.p.pins[*globals as usize].as_ptr(), self.p.pins[*builtins as usize].as_ptr(), name, x);
                }
                xincref(x)
            }
            Op::LoadAttr(n) => {
                let before = if self.reading { reads::frames() } else { 0 };
                let x = ffi::PyObject_GetAttr(a(0), self.name(*n));
                if self.reading && !self.span_input(ins, *n) {
                    reads::attr(a(0), self.name(*n), x, reads::frames() != before);
                }
                check(x)?
            }
            Op::LoadMethod(n) => {
                let mut out = [std::ptr::null_mut(); 2];
                let before = if self.reading { reads::frames() } else { 0 };
                let rc = tr_load_method(self.name(*n), a(0), out.as_mut_ptr());
                if self.reading && !self.span_input(ins, *n) {
                    let x = if out[0].is_null() { out[1] } else { out[0] };
                    reads::attr(a(0), self.name(*n), if rc < 0 { std::ptr::null_mut() } else { x }, reads::frames() != before);
                }
                status(rc)?;
                self.distribute(v, &out);
                return Ok(());
            }
            Op::StoreAttr(n) => {
                // A write the region makes again (see writes.rs).
                let token = if self.reading { writes::store_attr(a(1), self.name(*n), a(0), false) } else { 0 };
                refused_write(token)?;
                let rc = ffi::PyObject_SetAttr(a(1), self.name(*n), a(0));
                writes::done(token, rc >= 0);
                if rc < 0 {
                    return Err(Exc::Error);
                }
                return Ok(());
            }
            Op::Call { argc, kw_names } => {
                let kw = kw_names.map_or(std::ptr::null_mut(), |k| self.konst(k));
                let args = self.args(ins);
                let callee = if (*args).is_null() { *args.add(1) } else { *args };
                if super::region::tracing() && (super::opaque(callee) || self.opaque_site(ins.origin)) {
                    // Run as it is on every call: what it reads isn't the
                    // region's to check.
                    let _paused = reads::Recording::pause();
                    check(super::outside_executor(|| self.opaque_call(args, *argc as c_int, kw)))?
                } else if self.reading && kw.is_null() && writes::writes_call(callee) {
                    // C code that writes (object.__setattr__, list.append,
                    // ...): a write the region makes again.
                    let all: Vec<*mut PyObject> = ins.args.iter().map(|x| self.val(*x)).collect();
                    let token = writes::write_call(callee, &all[if all[0].is_null() { 2 } else { 1 }..]);
                    refused_write(token)?;
                    let r = tr_call(args, *argc as c_int, kw);
                    writes::done(token, !r.is_null());
                    check(r)?
                } else if self.reading && reads::reads_args(callee) {
                    let before = reads::frames();
                    let r = tr_call(args, *argc as c_int, kw);
                    let all: Vec<*mut PyObject> = ins.args.iter().map(|x| self.val(*x)).collect();
                    let given = &all[if all[0].is_null() { 2 } else { 1 }..];
                    // A call that raised read what it was handed too (a
                    // lookup that found nothing).
                    reads::call(callee, given, r, reads::frames() != before);
                    // C code that was handed a stand-in for a scalar input
                    // (see scalars.rs).
                    let r = if scalars::handed(callee, given) { scalars::called(callee, given, kw, r) } else { r };
                    check(r)?
                } else if self.reading && scalars::handed(callee, &ins.args.iter().map(|x| self.val(*x)).collect::<Vec<_>>()[1..]) {
                    // As above, for C code that is no builtin function (an
                    // instance of a C class that is called).
                    reads::called(callee);
                    let all: Vec<*mut PyObject> = ins.args.iter().map(|x| self.val(*x)).collect();
                    let r = tr_call(args, *argc as c_int, kw);
                    check(scalars::called(callee, &all[if all[0].is_null() { 2 } else { 1 }..], kw, r))?
                } else {
                    if self.reading {
                        reads::called(callee);
                    }
                    check(tr_call(args, *argc as c_int, kw))?
                }
            }
            Op::Binary(k) => {
                let r = check(tr_binary_op(*k as c_int, a(0), a(1)))?;
                if self.reading {
                    reads::used(&[a(0), a(1)]);
                }
                r
            }
            Op::Aten { .. } => {
                let (f, kw) = &self.p.aten[&v];
                let (kw, nkw) = kw.as_ref().map_or((std::ptr::null_mut(), 0), |k| (k.as_ptr(), ffi::PyTuple_GET_SIZE(k.as_ptr()) as usize));
                let args = self.args(ins);
                let r = check(ffi::PyObject_Vectorcall(f.as_ptr(), args, ins.args.len() - nkw, kw))?;
                if self.p.projects[v.idx()].is_empty() {
                    r
                } else {
                    // A tuple or (Tensor[]) list of the outputs.
                    let n = ffi::PySequence_Size(r);
                    let outs: Vec<*mut PyObject> = (0..n.max(0)).map(|k| ffi::PySequence_GetItem(r, k)).collect();
                    ffi::Py_DECREF(r);
                    if n < 0 || outs.iter().any(|o| o.is_null()) || outs.len() != self.p.projects[v.idx()].len() {
                        if n >= 0 && ffi::PyErr_Occurred().is_null() {
                            super::fail("CFG executor: an aten op returned an unexpected number of outputs", ffi::PyExc_RuntimeError);
                        }
                        outs.into_iter().for_each(|o| ffi::Py_XDECREF(o));
                        return Err(Exc::Error);
                    }
                    self.distribute(v, &outs);
                    return Ok(());
                }
            }
            Op::Lit(i) => xincref(self.p.literals[*i as usize].as_ptr()),
            Op::Pinned(i) => xincref(self.p.pins[*i as usize].as_ptr()),
            Op::Expect(_) => {
                unreachable!("checks are lifted into guards (see cfg/guard.rs)")
            }
            Op::Autograd(t) => {
                let params: Vec<*mut PyObject> = ins.args.iter().map(|a| self.val(*a)).collect();
                match super::autograd::forward(&self.p.trainings[*t as usize], &params) {
                    Ok(outs) => {
                        self.distribute(v, &outs);
                        return Ok(());
                    }
                    Err(e) => {
                        e.restore(pyo3::Python::assume_attached());
                        return Err(Exc::Error);
                    }
                }
            }
            Op::Region(_) => unreachable!("run_blocks runs a region version's head"),
            Op::Compare(k) => {
                let r = check(ffi::PyObject_RichCompare(a(0), a(1), *k as c_int))?;
                if self.reading {
                    reads::used(&[a(0), a(1)]);
                }
                r
            }
            Op::BuildTuple(n) => {
                let t = check(ffi::PyTuple_New(*n as ffi::Py_ssize_t))?;
                for (i, x) in ins.args.iter().enumerate() {
                    ffi::PyTuple_SetItem(t, i as ffi::Py_ssize_t, xincref(self.val(*x)));
                }
                t
            }
            Op::GetIter => {
                let r = check(ffi::PyObject_GetIter(a(0)))?;
                if self.reading {
                    reads::iterated(a(0));
                }
                r
            }
            Op::MakeFunction(flags) => check(tr_make_function(self.f, *flags as c_int, self.args(ins)))?,
            Op::MakeCell { slot } => {
                let p = self.local(*slot);
                let cell = check(PyCell_New(*p))?;
                let old = std::mem::replace(&mut *p, cell);
                ffi::Py_XDECREF(old);
                if self.reading {
                    super::made::cell(cell);
                }
                return Ok(());
            }
            Op::CopyFreeVars(n) => {
                let closure = PyFunction_GetClosure(tr_frame_func(self.f));
                let offset = tr_frame_nlocalsplus(self.f) as usize - *n as usize;
                for i in 0..*n as usize {
                    let cell = ffi::PyTuple_GetItem(closure, i as ffi::Py_ssize_t);
                    *self.locals.add(offset + i) = xincref(cell);
                }
                return Ok(());
            }
            Op::LoadDeref { slot } => {
                let x = PyCell_Get(*self.local(*slot));
                if x.is_null() {
                    tr_unbound_deref(self.f, *slot as c_int);
                    return Err(Exc::Error);
                }
                // A span's region is handed what its own frame's cells hold.
                if self.reading && self.stop.is_null() {
                    reads::cell(*self.local(*slot), x);
                }
                x
            }
            Op::StoreDeref { slot } => {
                if PyCell_Set(*self.local(*slot), a(0)) < 0 {
                    return Err(Exc::Error);
                }
                if self.reading {
                    super::made::cell(*self.local(*slot));
                }
                return Ok(());
            }
            Op::DeleteDeref { slot } => {
                let cell = *self.local(*slot);
                let old = PyCell_Get(cell);
                if old.is_null() {
                    tr_unbound_deref(self.f, *slot as c_int);
                    return Err(Exc::Error);
                }
                ffi::Py_DECREF(old);
                if PyCell_Set(cell, std::ptr::null_mut()) < 0 {
                    return Err(Exc::Error);
                }
                return Ok(());
            }
            Op::LoadFromDictOrDeref { slot } => {
                check(tr_load_from_dict_or_deref(self.f, a(0), *slot as c_int))?
            }
            Op::Bytecode { opcode: crate::cfg::opcodes::CALL_FUNCTION_EX, .. }
                if super::region::tracing() && (super::opaque(a(0)) || self.opaque_site(ins.origin)) =>
            {
                let _paused = reads::Recording::pause();
                check(super::outside_executor(|| self.opaque_call_ex(a(0), a(1), if ins.args.len() > 2 { a(2) } else { std::ptr::null_mut() })))?
            }
            Op::Bytecode { opcode, arg } if super::trampoline::needed(*opcode, *arg) => {
                let args = self.args(ins);
                check(super::trampoline::call(*opcode, *arg, std::slice::from_raw_parts(args, ins.args.len())))?
            }
            Op::Bytecode { opcode, arg } => {
                let n = self.p.nouts[v.idx()] as usize;
                let mut small = [std::ptr::null_mut(); 4];
                let mut big = vec![];
                let out: &mut [*mut PyObject] = if n <= small.len() {
                    &mut small[..n.max(1)]
                } else {
                    big.resize(n, std::ptr::null_mut());
                    &mut big
                };
                let args = self.args(ins);
                let before = if self.reading { reads::frames() } else { 0 };
                let token = if self.reading { self.note_write(*opcode, *arg, ins) } else { 0 };
                refused_write(token)?;
                let rc = tr_generic(self.f, *opcode as c_int, *arg as c_int, args, out.as_mut_ptr());
                writes::done(token, rc == 0);
                if self.reading {
                    // An instruction that raised read its operands too (a
                    // key that isn't there): the exception is kept.
                    let raised = if rc != 0 { pyo3::ffi::PyErr_GetRaisedException() } else { std::ptr::null_mut() };
                    self.note_read(*opcode, *arg, ins, reads::frames() != before);
                    if !raised.is_null() {
                        pyo3::ffi::PyErr_SetRaisedException(raised);
                    }
                }
                status(rc)?;
                match n {
                    0 => return Ok(()),
                    1 => out[0],
                    _ => {
                        self.distribute(v, out);
                        return Ok(());
                    }
                }
            }
            Op::Yield => unreachable!("handled by run_blocks"),
            Op::Resume | Op::EvalBreaker => return self.breaker(),
            Op::IterNext => {
                let item = ffi::PyIter_Next(a(0));
                if item.is_null() && !ffi::PyErr_Occurred().is_null() {
                    return Err(Exc::Error);
                }
                self.distribute(v, &[item, new_bool(!item.is_null())]);
                return Ok(());
            }
            Op::Send => {
                let mut out = std::ptr::null_mut();
                let yielded = match tr_send(a(0), a(1), &mut out) {
                    0 => true,
                    1 => false,
                    _ => return Err(Exc::Error),
                };
                self.distribute(v, &[out, new_bool(yielded)]);
                return Ok(());
            }
            Op::Null | Op::Unknown | Op::Param(_) | Op::Removed => {
                unreachable!("not an instruction")
            }
        };
        self.set(v, obj);
        if self.reading {
            let operands: Vec<*mut PyObject> = ins.args.iter().take(2).map(|x| self.val(*x)).collect();
            super::made::result(&ins.op, &operands, obj);
        }
        Ok(())
    }
}

// A write the trace could not make again (writes::REFUSED) is not made: the
// trace ends here with an error of its own, and the code runs again as it
// is.
unsafe fn refused_write(token: writes::Token) -> Result<(), Exc> {
    if token != writes::REFUSED {
        return Ok(());
    }
    super::region::raise_refusal(pyo3::Python::assume_attached());
    Err(Exc::Error)
}
