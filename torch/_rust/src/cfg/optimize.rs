// Optimization of a Cfg's blocks, following Dynamo's structure: the frame's
// locals are read and written only in glue code, around pure blocks that take
// the values they need as params and produce their results as outputs, the way
// Dynamo's bytecode loads locals, calls a graph and stores its outputs.
//
// Optimizing a block splits it at its barriers (ops the caller didn't judge
// movable, the places Dynamo would graph-break) into glue blocks and pure
// blocks. A stretch of movable ops becomes a pure block; its first load of
// each local moves to the glue before it, later loads reuse the value last
// loaded or stored, and its stores move to the glue after it (dropped when
// nothing follows but a return). Inside the stretch, as under Dynamo, the
// frame's locals are stale and old values are released late. If a pure block
// raises while stores are pending, it unwinds to a landing block that writes
// them and re-raises into the original handler, so a handler or a traceback
// sees the locals as CPython would have left them.
//
// Each pure block is then outlined into itself: it keeps the original code as
// its origin (what runs when no version's guard holds, and what a later
// optimization can start again from) and gets the optimized code as its
// version, where Python-level operators are replaced by the aten ops they ran
// (`a + b` by aten::add.Tensor). Replacing is valid when the operator would
// dispatch straight to that op: the guard checks that each tensor param is a
// plain tensor (no __torch_function__ or operator overrides), each scalar
// param exactly its builtin type, and that no __torch_function__ mode is
// active, as Dynamo guards its graph inputs and torch-function state. An
// operand computed in the block counts only if it's a constant or the result
// of a replaced op (an aten op on plain tensors returns a plain tensor), since
// nothing checks values in the middle of a version. Dtypes, devices and
// shapes don't matter here: the same op runs whatever they are.
//
// Calls to torch functions stay calls (as in Dynamo's graphs; autograd's
// tracer sees the aten ops below them), but a callee read from a global
// through module attributes (`F.linear`, `torch.relu`) is pinned: the version
// calls the object those loads produced when it was optimized, and its guard
// reads them and checks they still produce it (see cfg/guard.rs).

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use super::ir::{
    Block, BlockId, Builtin, Cfg, Check, Edge, Exit, Handler, NULL, Op, ParamKind, Pin, PinPath, TermUnwind, Terminator,
    UnwindSegment, ValueId, Version,
};
use super::inline::{self, Inlinee};
use super::outline::{self, Outline};

// What the caller observed about an operator it may replace: the one aten op
// every visit ran, with the operator's args in order, and what each arg was.
#[derive(Clone)]
pub struct AtenOp {
    pub name: String,
    pub args: Vec<Arg>,
}

// What the caller observed about a call whose callee may be pinned: the
// callee's name, and the call's args after the callable if it always returned
// a plain tensor (each None if it wasn't a plain tensor or builtin scalar).
#[derive(Clone)]
pub struct PinnedCall {
    pub name: String,
    pub args: Option<Vec<Option<Arg>>>,
    // Whether the callee is torch._C._has_torch_function{,_unary,_variadic}:
    // False for plain tensors when no torch function mode is active.
    pub has_torch_function: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Arg {
    PlainTensor,
    Scalar(Builtin),
}

// What the caller observed running the Cfg that the optimizer relies on.
#[derive(Clone, Default)]
pub struct Observed {
    // The instrs that may sit in a pure block.
    pub movable: HashSet<ValueId>,
    // The operators that may be replaced.
    pub aten: HashMap<ValueId, AtenOp>,
    // The calls whose callee may be pinned.
    pub calls: HashMap<ValueId, PinnedCall>,
    // The values only ever seen NULL.
    pub nulls: HashSet<ValueId>,
}

// Optimizes `blocks` (each split, and its pure blocks outlined into
// themselves), then renumbers the blocks. Returns each old block's new id.
// Each pin added to Cfg::pins gets the call it was made for pushed on
// `pinned`. With `strict`, a block that can't be optimized is an error;
// otherwise it's left as it is.
pub fn optimize_blocks(
    cfg: &mut Cfg,
    blocks: &[BlockId],
    obs: &Observed,
    inlines: &HashMap<ValueId, Inlinee>,
    pinned: &mut Vec<ValueId>,
    elided: &mut Vec<(usize, ValueId, ValueId)>,
    strict: bool,
) -> Result<Vec<Option<BlockId>>, String> {
    let mut obs = obs.clone();
    let mut pure = vec![];
    // Each pure block param, and the value it stands for (what was observed).
    let mut stands_for = HashMap::new();
    for &b in blocks {
        if b.idx() >= cfg.blocks.len() {
            return Err(format!("no block bb{}", b.0));
        }
        if !outline::outlinable(cfg, b) {
            if strict {
                return Err(format!(
                    "bb{} can't be optimized: it already has an origin or several versions, or a generator suspends in it",
                    b.0
                ));
            }
            continue;
        }
        pure.extend(split(cfg, b, &obs.movable, &mut stands_for));
    }
    let map = cfg.compact_blocks();
    for p in pure {
        let p = map[p.idx()].expect("a pure block is reachable");
        outline::outline(cfg, &[p], Outline::Merge);
        let calls: Vec<ValueId> = cfg[p].versions[0].body.iter().copied().filter(|v| inlines.contains_key(v)).collect();
        let nulls = obs.nulls.clone();
        let is_null = |v: ValueId| v == NULL || nulls.contains(stands_for.get(&v).unwrap_or(&v));
        // The calls whose elision keeps every check in the guard (see
        // cfg/guard.rs), found by trying each on a copy.
        let mut keep: Vec<ValueId> = vec![];
        for v in calls {
            let mut trial = keep.clone();
            trial.push(v);
            let (mut c, mut o) = (cfg.clone(), obs.clone());
            if finish_block(&mut c, p, &trial, inlines, &mut o, &is_null, &mut vec![], &mut vec![]) {
                keep.push(v);
            }
        }
        let ok = finish_block(cfg, p, &keep, inlines, &mut obs, &is_null, pinned, elided);
        debug_assert!(ok, "the elisions kept were tried");
    }
    Ok(map)
}

// Elides `calls` in pure block p's version, then pins callees and replaces
// operators; false if a check of an elided callee can't be in the guard (it
// depends on what the version computes).
#[allow(clippy::too_many_arguments)]
fn finish_block(
    cfg: &mut Cfg,
    p: BlockId,
    calls: &[ValueId],
    inlines: &HashMap<ValueId, Inlinee>,
    obs: &mut Observed,
    is_null: &dyn Fn(ValueId) -> bool,
    pinned: &mut Vec<ValueId>,
    elided: &mut Vec<(usize, ValueId, ValueId)>,
) -> bool {
    let mut any = false;
    for &v in calls {
        if let Some(m) = inline::elide(cfg, p, v, &inlines[&v], obs) {
            elided.extend(m);
            any = true;
        }
    }
    pin_calls(cfg, p, &obs.calls, is_null, pinned);
    use_aten(cfg, p, &obs.aten, &obs.calls, any);
    inline::tidy(cfg, p)
}

// A module's type has only dunder attributes, so any other name is read from
// the module's __dict__ (and LoadMethod gives no method for it).
fn plain_name(cfg: &Cfg, n: u32) -> bool {
    !cfg.names.get(n as usize).is_some_and(|s| s.starts_with("__"))
}

// The global and module attributes `v` was read through in `body`, if it was.
fn load_path(cfg: &Cfg, body: &HashSet<ValueId>, v: ValueId) -> Option<PinPath> {
    if !body.contains(&v) {
        return None;
    }
    match cfg[v].op {
        Op::LoadGlobal(n) => Some(PinPath { global: n, attrs: vec![], ns: None }),
        Op::LoadGlobalIn { name, globals, builtins } => Some(PinPath { global: name, attrs: vec![], ns: Some((globals, builtins)) }),
        Op::LoadAttr(n) if plain_name(cfg, n) => {
            let mut p = load_path(cfg, body, cfg[v].args[0])?;
            p.attrs.push(n);
            Some(p)
        }
        _ => None,
    }
}

// Pins the callees of `calls` in pure block p's version (see the top of the
// file), adding to its guard, and guards the params `is_null` says were
// always NULL on being NULL. A callee loaded in the block gets an Expect
// on its load (which lift_guard makes a check of the guard); one passed in
// as a param, a check of the param's identity.
fn pin_calls(cfg: &mut Cfg, p: BlockId, calls: &HashMap<ValueId, PinnedCall>, is_null: &dyn Fn(ValueId) -> bool, pinned: &mut Vec<ValueId>) {
    let body = cfg[p].versions[0].body.clone();
    let handlers: HashMap<ValueId, Handler> = body.iter().enumerate().map(|(i, x)| (*x, cfg[p].versions[0].handler_at(i))).collect();
    let params = cfg[p].params.clone();
    let in_body: HashSet<ValueId> = body.iter().copied().collect();
    let mut checks: Vec<(Option<ValueId>, Check)> = params.iter().filter(|q| is_null(**q)).map(|q| (Some(*q), Check::Null)).collect();
    // A LoadMethod (and its NULL) only a pinned callee reads, dropped if
    // nothing else reads it.
    let mut chain = vec![];
    // Pinned values for the calls, to put at the start of the body.
    let mut prologue = vec![];
    // The reads and checks of a loaded callee, after the instr they follow.
    let mut after: HashMap<ValueId, Vec<ValueId>> = HashMap::new();
    // A loaded callee's check can go in the guard only if its loads come
    // before anything that writes (see inline::tidy); a call pinned here
    // doesn't write.
    let mut first_write = body.len();
    let mut pinned_here: HashSet<ValueId> = HashSet::new();
    for (i, &v) in body.iter().enumerate() {
        if let Some(&prev) = i.checked_sub(1).map(|j| &body[j]) {
            if first_write == body.len() && !pinned_here.contains(&prev) && super::guard::writes(cfg, prev) {
                first_write = i - 1;
            }
        }
        let Some(call) = calls.get(&v) else {
            continue;
        };
        if !matches!(cfg[v].op, Op::Call { .. }) || cfg[v].args.len() < 2 {
            continue;
        }
        let (a0, a1) = (cfg[v].args[0], cfg[v].args[1]);
        let param = |x: ValueId| params.iter().position(|q| *q == x);
        enum Form {
            // NULL, then the callee loaded in the block.
            Loaded,
            // A LoadMethod's projects, of a module loaded in the block.
            Method(ValueId),
            // NULL (maybe a param), then the callee as param i.
            Param(usize),
        }
        let form = match (&cfg[a0].op, &cfg[a1].op) {
            _ if is_null(a0) && param(a1).is_some() => Form::Param(param(a1).unwrap()),
            _ if a0 == NULL => Form::Loaded,
            (Op::Project(0), Op::Project(1)) if cfg[a0].args[0] == cfg[a1].args[0] => Form::Method(cfg[a1].args[0]),
            _ => continue,
        };
        let path = match form {
            Form::Loaded => load_path(cfg, &in_body, a1),
            Form::Method(m) => match cfg[m].op {
                Op::LoadMethod(n) if in_body.contains(&m) && plain_name(cfg, n) => load_path(cfg, &in_body, cfg[m].args[0]).map(|mut p| {
                    p.attrs.push(n);
                    p
                }),
                _ => None,
            },
            Form::Param(_) => None,
        };
        if path.is_none() && !matches!(form, Form::Param(_)) {
            continue;
        }
        let checked_at = match form {
            Form::Loaded => Some(a1),
            Form::Method(m) => Some(cfg[m].args[0]),
            Form::Param(_) => None,
        };
        if checked_at.is_some_and(|x| body.iter().position(|y| *y == x).is_none_or(|j| j >= first_write)) {
            continue;
        }
        pinned_here.insert(v);
        let pin = Pin { name: call.name.clone(), path };
        let k = match cfg.pins.iter().position(|q| *q == pin && q.path.is_some()) {
            Some(k) => k,
            None => {
                cfg.pins.push(pin);
                pinned.push(v);
                cfg.pins.len() - 1
            }
        } as u32;
        cfg[v].args[0] = NULL;
        let origin = cfg[v].origin;
        let pv = cfg.add(Op::Pinned(k), vec![], origin);
        prologue.push(pv);
        cfg[v].args[1] = pv;
        match form {
            Form::Param(i) => {
                let c = (Some(params[i]), Check::Is(k));
                if !checks.contains(&c) {
                    checks.push(c);
                }
            }
            Form::Loaded => {
                let e = cfg.add(Op::expect(Check::Is(k)), vec![a1], origin);
                after.entry(a1).or_default().push(e);
            }
            Form::Method(m) => {
                // The attribute the LoadMethod found (a module's: no method).
                let Op::LoadMethod(n) = cfg[m].op else { unreachable!("Form::Method") };
                let module = cfg[m].args[0];
                let attr = cfg.add(Op::LoadAttr(n), vec![module], origin);
                let e = cfg.add(Op::expect(Check::Is(k)), vec![attr], origin);
                after.entry(module).or_default().extend([attr, e]);
                chain.extend([a0, a1, m]);
            }
        }
    }
    if checks.is_empty() && after.is_empty() && prologue.is_empty() {
        return;
    }
    // Drop the loads (and a LoadMethod's NULL) nothing reads any more, each
    // before the one it reads; the guard ensures they wouldn't have raised.
    let ver = &cfg[p].versions[0];
    let mut used: HashMap<ValueId, usize> = HashMap::new();
    let handler_uses = ver.unwind.iter().flat_map(|s| handler_args(&s.handler));
    for x in body.iter().flat_map(|v| cfg[*v].args.clone()).chain(ver.outputs.iter().copied()).chain(handler_uses) {
        *used.entry(x).or_default() += 1;
    }
    let mut dead = HashSet::new();
    for x in chain {
        let unused = |x: &ValueId| used.get(x).copied().unwrap_or(0) == 0;
        let removable = match cfg[x].op {
            Op::LoadGlobal(_) | Op::LoadGlobalIn { .. } | Op::LoadAttr(_) | Op::Project(_) => unused(&x),
            // Its projects (the NULL and the pinned callee) are its outputs.
            Op::LoadMethod(_) => body.iter().all(|y| dead.contains(y) || !(matches!(cfg[*y].op, Op::Project(_)) && cfg[*y].args[0] == x)),
            _ => false,
        };
        if removable && dead.insert(x) {
            for a in cfg[x].args.clone() {
                *used.entry(a).or_default() -= 1;
            }
        }
    }
    let mut out = prologue;
    let mut handler: HashMap<ValueId, Handler> = HashMap::new();
    for x in body.iter().filter(|v| !dead.contains(*v)) {
        out.push(*x);
        for e in after.get(x).into_iter().flatten() {
            handler.insert(*e, handlers[x].clone());
            out.push(*e);
        }
    }
    let first = body.first().map_or(Handler::Propagate, |x| handlers[x].clone());
    let ver = &mut cfg[p].versions[0];
    ver.body = out;
    ver.set_unwind(|x| handlers.get(&x).or(handler.get(&x)).cloned().unwrap_or_else(|| first.clone()));
    ver.guard.checks.extend(checks);
    cfg.normalize_unwind(p);
}

// Replaces the operators of pure block p's version with aten ops where the
// guard can make that valid, and adds to the guard. A pinned call returning a
// plain tensor counts as a replaced op when its args are known as an
// operator's must be.
fn use_aten(cfg: &mut Cfg, p: BlockId, aten: &HashMap<ValueId, AtenOp>, calls: &HashMap<ValueId, PinnedCall>, elided: bool) {
    let params = cfg[p].params.clone();
    let body = cfg[p].versions[0].body.clone();
    let mut checks: BTreeSet<(Option<ValueId>, Check)> = BTreeSet::new();
    // Values read in a version with elided calls (whose reads go in its
    // guard, see cfg/guard.rs), and the checks they need there.
    let mut read_checks: BTreeSet<(ValueId, Check)> = BTreeSet::new();
    let is_read = |cfg: &Cfg, x: ValueId| elided && !matches!(cfg[x].op, Op::Param(_) | Op::Const(_) | Op::Pinned(_) | Op::Lit(_)) && inline::is_read(cfg, x);
    // Results of replaced ops: plain tensors whenever the guard holds.
    let mut tensors: HashSet<ValueId> = HashSet::new();
    // has_torch_function calls of plain tensors, which are False.
    let mut falses = vec![];
    for v in body {
        if let Some(PinnedCall { has_torch_function: true, .. }) = calls.get(&v) {
            let a = cfg[v].args.clone();
            let plain = |x: &ValueId| tensors.contains(x) || params.contains(x);
            if a.len() > 2 && a[0] == NULL && matches!(cfg[a[1]].op, Op::Pinned(_)) && a[2..].iter().all(plain) {
                for x in a[2..].iter().filter(|x| params.contains(x)) {
                    checks.insert((Some(*x), Check::PlainTensor));
                }
                falses.push(v);
            }
            continue;
        }
        let pinned_call = matches!(cfg[v].op, Op::Call { .. }) && cfg[v].args.get(1).is_some_and(|c| matches!(cfg[*c].op, Op::Pinned(_)));
        let (args, kinds): (Vec<ValueId>, Vec<Option<Arg>>) = match (aten.get(&v), calls.get(&v)) {
            (Some(a), _) => (cfg[v].args.clone(), a.args.iter().copied().map(Some).collect()),
            (_, Some(PinnedCall { args: Some(kinds), .. })) if pinned_call => (cfg[v].args[2..].to_vec(), kinds.clone()),
            _ => continue,
        };
        if args.len() != kinds.len() {
            continue;
        }
        let mut need = vec![];
        let mut need_read = vec![];
        let ok = args.iter().zip(&kinds).all(|(x, kind)| match (params.iter().position(|q| q == x), kind) {
            (None, Some(kind)) if is_read(cfg, *x) && !tensors.contains(x) => {
                need_read.push((*x, match kind {
                    Arg::PlainTensor => Check::PlainTensor,
                    Arg::Scalar(t) => Check::Exact(*t),
                }));
                true
            }
            (Some(_), Some(kind)) => {
                need.push((Some(*x), match kind {
                    Arg::PlainTensor => Check::PlainTensor,
                    Arg::Scalar(t) => Check::Exact(*t),
                }));
                true
            }
            (Some(_), None) => false,
            (None, Some(Arg::PlainTensor)) => tensors.contains(x),
            // A pinned object is fixed too (an elided call's default).
            (None, Some(Arg::Scalar(_))) => matches!(cfg[*x].op, Op::Const(_) | Op::Pinned(_)),
            (None, _) => matches!(cfg[*x].op, Op::Const(_)),
        });
        if ok {
            checks.extend(need);
            read_checks.extend(need_read);
            if !pinned_call {
                cfg[v].op = Op::Aten { name: aten[&v].name.clone(), kwnames: vec![] };
            }
            tensors.insert(v);
        }
    }
    if !tensors.is_empty() || !falses.is_empty() {
        checks.insert((None, Check::NoTorchFunctionMode));
        cfg[p].versions[0].guard.checks.extend(checks);
    }
    if !falses.is_empty() {
        cfg.literals.push(super::ir::Lit::Bool(false));
        let k = cfg.literals.len() as u32 - 1;
        for v in falses {
            cfg[v].op = Op::Lit(k);
            cfg[v].args.clear();
        }
    }
    // Each read's checks right after it.
    if !read_checks.is_empty() {
        let ver = &cfg[p].versions[0];
        let handlers: HashMap<ValueId, Handler> = ver.body.iter().enumerate().map(|(i, x)| (*x, ver.handler_at(i))).collect();
        let body = std::mem::take(&mut cfg[p].versions[0].body);
        let mut out = vec![];
        for v in body {
            out.push(v);
            for (_, c) in read_checks.iter().filter(|(x, _)| *x == v) {
                let origin = cfg[v].origin;
                out.push(cfg.add(Op::expect(c.clone()), vec![v], origin));
            }
        }
        let ver = &mut cfg[p].versions[0];
        ver.body = out;
        ver.set_unwind(|x| handlers.get(&x).cloned().unwrap_or(Handler::Propagate));
    }
}

enum Class {
    Load(u32),
    Store(u32),
    Compute,
    Barrier,
}

fn classify(cfg: &Cfg, v: ValueId, movable: &HashSet<ValueId>) -> Class {
    match cfg[v].op {
        Op::LoadLocal { slot } => Class::Load(slot),
        Op::StoreLocal { slot } => Class::Store(slot),
        // They read or write the frame in ways that must stay in place.
        Op::LoadLocalChecked { .. }
        | Op::DeleteLocal { .. }
        | Op::MakeCell { .. }
        | Op::CopyFreeVars(_)
        | Op::LoadDeref { .. }
        | Op::StoreDeref { .. }
        | Op::DeleteDeref { .. }
        | Op::LoadFromDictOrDeref { .. } => Class::Barrier,
        // Written by its producer when that runs, so it stays with it.
        Op::Project(_) => classify(cfg, cfg[v].args[0], movable),
        _ if movable.contains(&v) => Class::Compute,
        _ => Class::Barrier,
    }
}

// (slot, value): stores a pure block hasn't written to the frame yet.
type Pending = Vec<(u32, ValueId)>;

// A raising op of a pure block: where it unwinds in the original block, and
// the stores pending when it runs.
struct Raise {
    at: ValueId,
    handler: Handler,
    pending: Pending,
}

enum Part {
    Glue(Vec<ValueId>),
    Pure(Vec<ValueId>, Vec<Raise>),
}

impl Part {
    fn instrs(&self) -> &[ValueId] {
        match self {
            Part::Glue(is) | Part::Pure(is, _) => is,
        }
    }
}

fn handler_args(h: &Handler) -> Vec<ValueId> {
    match h {
        Handler::Block(_, args) | Handler::Outer(_, args) => args.clone(),
        Handler::Propagate => vec![],
    }
}

fn map_handler(h: &Handler, f: impl Fn(ValueId) -> ValueId) -> Handler {
    match h {
        Handler::Block(k, args) => Handler::Block(*k, args.iter().map(|a| f(*a)).collect()),
        Handler::Outer(k, args) => Handler::Outer(*k, args.iter().map(|a| f(*a)).collect()),
        Handler::Propagate => Handler::Propagate,
    }
}

// Splits block b (one version, no origin) into a chain of glue and pure
// blocks; b stays the first glue block. Returns the pure blocks (new ids, not
// yet renumbered), none if b has no stretch worth splitting out.
fn split(cfg: &mut Cfg, b: BlockId, movable: &HashSet<ValueId>, stands_for: &mut HashMap<ValueId, ValueId>) -> Vec<BlockId> {
    let ver = cfg[b].versions[0].clone();
    let returns = matches!(cfg[b].term, Terminator::Jump(Exit::Return(_)));
    let handler_at = |i: usize| -> Handler {
        ver.unwind.iter().rev().find(|s| s.from <= i).map_or(Handler::Propagate, |s| s.handler.clone())
    };

    // The parts, in the original block's namespace.
    let mut parts: Vec<Part> = vec![];
    let mut glue: Vec<ValueId> = vec![];
    let mut subst: HashMap<ValueId, ValueId> = HashMap::new();
    let get = |subst: &HashMap<ValueId, ValueId>, v: ValueId| subst.get(&v).copied().unwrap_or(v);
    let mut i = 0;
    while i < ver.body.len() {
        let start = i;
        while i < ver.body.len() && !matches!(classify(cfg, ver.body[i], movable), Class::Barrier) {
            i += 1;
        }
        let worth = ver.body[start..i].iter().any(|v| {
            matches!(classify(cfg, *v, movable), Class::Compute)
                && cfg[*v].op.can_raise()
                && !matches!(cfg[*v].op, Op::Resume | Op::EvalBreaker)
        });
        if worth {
            let mut known: HashMap<u32, ValueId> = HashMap::new();
            let mut pending: BTreeMap<u32, (ValueId, u32)> = BTreeMap::new();
            let (mut body, mut raises) = (vec![], vec![]);
            for k in start..i {
                let v = ver.body[k];
                match classify(cfg, v, movable) {
                    Class::Load(slot) => match known.get(&slot) {
                        Some(&x) => {
                            subst.insert(v, x);
                            cfg[v].op = Op::Removed;
                        }
                        None => {
                            known.insert(slot, v);
                            glue.push(v);
                        }
                    },
                    Class::Store(slot) => {
                        let x = get(&subst, cfg[v].args[0]);
                        known.insert(slot, x);
                        pending.insert(slot, (x, cfg[v].origin));
                        cfg[v].op = Op::Removed;
                        cfg[v].args.clear();
                    }
                    Class::Compute => {
                        if cfg[v].op.can_raise() {
                            let pending = pending.iter().map(|(s, (x, _))| (*s, *x)).collect();
                            raises.push(Raise { at: v, handler: handler_at(k), pending });
                        }
                        body.push(v);
                    }
                    Class::Barrier => unreachable!("runs hold no barriers"),
                }
            }
            parts.push(Part::Glue(std::mem::take(&mut glue)));
            parts.push(Part::Pure(body, raises));
            if !(returns && i == ver.body.len()) {
                for (slot, (x, origin)) in pending {
                    glue.push(cfg.add(Op::StoreLocal { slot }, vec![x], origin));
                }
            }
        } else {
            glue.extend(&ver.body[start..i]);
        }
        if i < ver.body.len() {
            glue.push(ver.body[i]);
            i += 1;
        }
    }
    if parts.is_empty() {
        return vec![];
    }
    parts.push(Part::Glue(glue));

    // Forwarded loads, everywhere they're used.
    for part in &parts {
        for v in part.instrs() {
            let args = cfg[*v].args.iter().map(|a| get(&subst, *a)).collect();
            cfg[*v].args = args;
        }
    }
    for part in &mut parts {
        if let Part::Pure(_, raises) = part {
            for r in raises {
                r.handler = map_handler(&r.handler, |a| get(&subst, a));
                r.pending.iter_mut().for_each(|(_, x)| *x = get(&subst, *x));
            }
        }
    }
    let outputs: Vec<ValueId> = ver.outputs.iter().map(|o| get(&subst, *o)).collect();
    let orig_index: HashMap<ValueId, usize> = ver.body.iter().enumerate().map(|(k, v)| (*v, k)).collect();
    // A glue instr's handler, in the original namespace with loads forwarded.
    let glue_handler = |v: ValueId| -> Option<Handler> {
        let oi = orig_index.get(&v)?;
        Some(map_handler(&handler_at(*oi), |a| get(&subst, a)))
    };

    // The values each part needs from earlier ones, which become its params.
    let n = parts.len();
    let mut live_in: Vec<Vec<ValueId>> = vec![vec![]; n];
    let mut live: HashSet<ValueId> = outputs.iter().copied().filter(|v| !v.is_reserved()).collect();
    for k in (0..n).rev() {
        let part = &parts[k];
        let mut uses: Vec<ValueId> = part.instrs().iter().flat_map(|v| cfg[*v].args.clone()).collect();
        match part {
            Part::Glue(is) => {
                for v in is.iter().filter(|v| cfg[**v].op.can_raise()) {
                    uses.extend(glue_handler(*v).map(|h| handler_args(&h)).unwrap_or_default());
                }
            }
            Part::Pure(_, raises) => {
                for r in raises {
                    uses.extend(handler_args(&r.handler));
                    uses.extend(r.pending.iter().map(|(_, x)| *x));
                }
            }
        }
        live.extend(uses.into_iter().filter(|v| !v.is_reserved()));
        for v in part.instrs() {
            live.remove(v);
        }
        let mut l: Vec<ValueId> = live.iter().copied().collect();
        l.sort();
        live_in[k] = l;
    }
    debug_assert!(live_in[0].iter().all(|v| cfg[b].params.contains(v)));

    // b is part 0; the other parts follow it, then the landing blocks.
    let orig_handlers = cfg[b].handlers.clone();
    let term = cfg[b].term.clone();
    let term_origin = cfg[b].term_origin;
    let base = cfg.blocks.len();
    let id_of = |k: usize| if k == 0 { b } else { BlockId((base + k - 1) as u32) };
    let mut new_blocks: Vec<Block> = vec![];
    let mut landings: Vec<((Handler, Pending), Block)> = vec![];
    let mut pure_ids = vec![];
    for (k, part) in parts.iter().enumerate() {
        let mut blk = Block::new();
        let mut local: HashMap<ValueId, ValueId> = HashMap::new();
        if k == 0 {
            blk.params = cfg[b].params.clone();
        } else {
            for (j, v) in live_in[k].iter().enumerate() {
                let p = cfg.add(Op::Param(ParamKind::Stack(j as u32)), vec![], term_origin);
                local.insert(*v, p);
                stands_for.insert(p, *stands_for.get(v).unwrap_or(v));
                blk.params.push(p);
            }
        }
        let loc = |v: ValueId| local.get(&v).copied().unwrap_or(v);
        for v in part.instrs() {
            let args = cfg[*v].args.iter().map(|a| loc(*a)).collect();
            cfg[*v].args = args;
        }
        let mut ver = Version {
            body: part.instrs().to_vec(),
            ..Version::default()
        };
        let push_seg = |ver: &mut Version, from: usize, handler: Handler| {
            if ver.unwind.last().map_or(handler != Handler::Propagate, |s| s.handler != handler) {
                ver.unwind.push(UnwindSegment { from, handler });
            }
        };
        let index_of = |handlers: &mut Vec<BlockId>, h: BlockId| match handlers.iter().position(|x| *x == h) {
            Some(i) => i,
            None => {
                handlers.push(h);
                handlers.len() - 1
            }
        };
        let mut handlers: Vec<BlockId> = vec![];
        match part {
            Part::Glue(is) => {
                handlers.clone_from(&orig_handlers);
                for (pos, v) in is.iter().enumerate() {
                    if let (true, Some(h)) = (cfg[*v].op.can_raise(), glue_handler(*v)) {
                        push_seg(&mut ver, pos, map_handler(&h, loc));
                    }
                }
            }
            Part::Pure(body, raises) => {
                pure_ids.push(id_of(k));
                for r in raises {
                    let pos = body.iter().position(|v| *v == r.at).expect("the raise is in the body");
                    let handler = if r.pending.is_empty() {
                        match &r.handler {
                            Handler::Block(h, args) => Handler::Block(
                                index_of(&mut handlers, orig_handlers[*h]),
                                args.iter().map(|a| loc(*a)).collect(),
                            ),
                            h => h.clone(),
                        }
                    } else {
                        let key = (r.handler.clone(), r.pending.clone());
                        let li = match landings.iter().position(|(k, _)| *k == key) {
                            Some(li) => li,
                            None => {
                                let origin = cfg[r.at].origin;
                                landings.push((key, landing(cfg, &r.pending, &r.handler, &orig_handlers, origin)));
                                landings.len() - 1
                            }
                        };
                        let target = BlockId((base + n - 1 + li) as u32);
                        let args = r.pending.iter().map(|(_, x)| *x).chain(handler_args(&r.handler));
                        Handler::Block(index_of(&mut handlers, target), args.map(loc).collect())
                    };
                    push_seg(&mut ver, pos, handler);
                }
            }
        }
        blk.handlers = handlers;
        if k + 1 < n {
            ver.outputs = live_in[k + 1].iter().map(|v| loc(*v)).collect();
            blk.term = Terminator::Jump(Exit::Block(Edge {
                target: id_of(k + 1),
                args: (0..ver.outputs.len() as u32).collect(),
            }));
        } else {
            ver.outputs = outputs.iter().map(|v| loc(*v)).collect();
            blk.term = term.clone();
        }
        blk.term_origin = term_origin;
        blk.versions.push(ver);
        if k == 0 {
            cfg[b] = blk;
        } else {
            new_blocks.push(blk);
        }
    }
    cfg.blocks.extend(new_blocks);
    cfg.blocks.extend(landings.into_iter().map(|(_, l)| l));
    for k in std::iter::once(b.idx()).chain(base..cfg.blocks.len()) {
        cfg.normalize_unwind(BlockId(k as u32));
    }
    pure_ids
}

// A block that writes `pending` to the frame and re-raises into `handler` (an
// index into `handlers`, the original block's) with the same lasti, so the
// handler or a traceback sees what CPython would have. Its params: the pending
// values, the handler's args, then lasti and the exception.
fn landing(cfg: &mut Cfg, pending: &[(u32, ValueId)], handler: &Handler, handlers: &[BlockId], origin: u32) -> Block {
    let mut blk = Block::new();
    let mut param = |cfg: &mut Cfg, kind: ParamKind| {
        let p = cfg.add(Op::Param(kind), vec![], origin);
        blk.params.push(p);
        p
    };
    let n_args = handler_args(handler).len();
    let stored: Vec<(u32, ValueId)> =
        pending.iter().enumerate().map(|(j, (slot, _))| (*slot, param(cfg, ParamKind::Stack(j as u32)))).collect();
    let args: Vec<ValueId> = (0..n_args).map(|j| param(cfg, ParamKind::Stack((pending.len() + j) as u32))).collect();
    let lasti = param(cfg, ParamKind::Lasti);
    let exc = param(cfg, ParamKind::Exc);
    let mut ver = Version::default();
    for (slot, p) in stored {
        ver.body.push(cfg.add(Op::StoreLocal { slot }, vec![p], origin));
    }
    ver.outputs = [exc, lasti].into_iter().chain(args).collect();
    let unwind = match handler {
        Handler::Block(h, _) => {
            blk.handlers = vec![handlers[*h]];
            TermUnwind::Block(0, (2..2 + n_args as u32).collect())
        }
        Handler::Propagate => TermUnwind::Propagate,
        Handler::Outer(..) => unreachable!("a top-level block has no outer handlers"),
    };
    blk.term = Terminator::Reraise {
        exc: 0,
        lasti: Some(1),
        unwind,
    };
    blk.term_origin = origin;
    blk.versions.push(ver);
    blk
}
