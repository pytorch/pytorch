// Elision of a Python call (see cfg/optimize.rs): the callee's observed path,
// from its entry to a return, each branch going the one way it was always
// seen to go, copied into the version of the pure block holding the call.
// The callee's locals and cells become SSA values, its globals are read from
// its own namespace, and each branch condition becomes an Expect, as does the
// callee's identity. Those checks have to end up in the version's guard
// (see guard.rs), so a failing one runs the block's origin, which makes
// the real call; an elision whose checks depend on what the version computes
// is refused. A raise in the elided code unwinds from the version as the
// call would have, without the callee's frame in the traceback.
//
// What can be elided: a call (Op::Call, or CALL_FUNCTION_EX of a tuple this
// elision built and an empty dict) of a function, a bound method or an
// instance whose class's __call__ is a function; and an attribute load that
// falls back to its class's __getattr__.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::ir::{BlockId, Cfg, Check, Exit, Handler, Lit, NULL, Op, OutIdx, Terminator, ValueId, ENTRY};
use super::opcodes;
use super::optimize::Observed;

// What the caller worked out about a call it elides (in the caller's
// namespace for pins and names, the callee's for values and blocks).
pub struct Inlinee {
    // Distinguishes inlinees (nested ones too) in what elide returns.
    pub id: usize,
    // What was observed of the callee's values (`movable` unused).
    pub obs: Observed,
    // The callee's unoptimized Cfg.
    pub cfg: Arc<Cfg>,
    // The way each Branch on the path goes, by block.
    pub dirs: HashMap<BlockId, bool>,
    pub site: Site,
    // What each of the callee's parameter slots starts as.
    pub bind: Vec<Bind>,
    // Cfg::pins[function]: the function that runs.
    pub function: u32,
    // A check that the function's __defaults__ is still this tuple, when a
    // default was bound: (the pin of the tuple, the name of "__defaults__").
    pub defaults: Option<(u32, u32)>,
    // The function's globals and builtins.
    pub globals: u32,
    pub builtins: u32,
    // By the callee's const index: the caller's pin of that object.
    pub consts: Vec<u32>,
    // Elided calls on the path, by the callee's instr.
    pub nested: HashMap<ValueId, Inlinee>,
    // Names (in the caller) of "__func__", "__self__", "__call__".
    pub names: [u32; 3],
}

// How the elided instr calls the function.
pub enum Site {
    // A Call (NULL or, with `method`, a method from LoadMethod, then the
    // callable or self, then args) or a CALL_FUNCTION_EX (callable, tuple,
    // dict) of the function itself, a bound method of it, or an instance
    // whose class's __call__ it is.
    Call { ex: bool, method: bool, callable: Callable },
    // A LoadAttr or LoadMethod of Cfg::names[name] (the string Cfg::pins[str])
    // falling back to the receiver's class's __getattr__.
    Getattr { name: u32, str: u32 },
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Callable {
    Function,
    BoundMethod,
    Instance,
}

pub enum Bind {
    // The call's k-th positional arg (counting self).
    Arg(usize),
    // A default: Cfg::pins[i].
    Pin(u32),
    // *args: a tuple of the positional args from k on.
    VarArgs(usize),
    // **kwargs: an empty dict.
    VarKw,
}

// What the values made so far are known to be.
#[derive(Default)]
struct Known {
    tuples: HashMap<ValueId, Vec<ValueId>>,
    empty_dicts: HashSet<ValueId>,
}

// Elides instr `v` of block p's only version, putting `inl`'s path where it
// was (raising into the call's handler). What was observed of the callee's
// values is added to `obs` for the instrs they became. Returns, for each
// callee instr copied (nested ones too), (inlinee id, callee instr, instr),
// or None (nothing changed) if the path can't be copied after all.
pub fn elide(cfg: &mut Cfg, p: BlockId, v: ValueId, inl: &Inlinee, obs: &mut Observed) -> Option<Vec<(usize, ValueId, ValueId)>> {
    let body = cfg[p].versions[0].body.clone();
    let at = body.iter().position(|x| *x == v).expect("the instr is in the block");
    let handlers: HashMap<ValueId, Handler> = body.iter().enumerate().map(|(i, x)| (*x, cfg[p].versions[0].handler_at(i))).collect();
    let origin = cfg[v].origin;
    let a = cfg[v].args.clone();
    let mut out = body[..at].to_vec();
    let mut map = vec![];
    let mut added = Observed::default();
    let result = splice(cfg, inl, &a, origin, &mut out, &mut map, &mut added, &mut Known::default())?;
    obs.aten.extend(added.aten);
    obs.calls.extend(added.calls);
    obs.nulls.extend(added.nulls);
    let method = matches!(cfg[v].op, Op::LoadMethod(_));
    let mut subst: HashMap<ValueId, ValueId> = HashMap::from([(v, result)]);
    let mut rest = vec![];
    for x in &body[at + 1..] {
        // A LoadMethod's projects: NULL (no method) and the attribute.
        if method && matches!(cfg[*x].op, Op::Project(_)) && cfg[*x].args[0] == v {
            let Op::Project(k) = cfg[*x].op else { unreachable!() };
            subst.insert(*x, if k == 0 { NULL } else { result });
            cfg[*x].op = Op::Removed;
            cfg[*x].args.clear();
            continue;
        }
        rest.push(*x);
    }
    let get = |x: ValueId| *subst.get(&x).unwrap_or(&x);
    for x in &rest {
        let args: Vec<ValueId> = cfg[*x].args.iter().map(|a| get(*a)).collect();
        cfg[*x].args = args;
    }
    out.extend(rest);
    cfg[v].op = Op::Removed;
    cfg[v].args.clear();
    let ver = &mut cfg[p].versions[0];
    ver.body = out;
    for o in ver.outputs.iter_mut() {
        *o = get(*o);
    }
    ver.set_unwind(|x| match handlers.get(&x).unwrap_or(&handlers[&v]) {
        Handler::Block(k, args) => Handler::Block(*k, args.iter().map(|a| get(*a)).collect()),
        Handler::Outer(k, args) => Handler::Outer(*k, args.iter().map(|a| get(*a)).collect()),
        Handler::Propagate => Handler::Propagate,
    });
    Some(map)
}

// Tidies block p's version after elisions (and the pinning and aten
// replacement that follow them): drops values nothing uses that can't
// raise, and lifts the reads and checks into its guard. False if a check
// stays in the body.
pub fn tidy(cfg: &mut Cfg, p: BlockId) -> bool {
    // Known bools: ToBool of one is itself, and an Expect of it holds (or
    // the version can't run: left for the origin).
    let handlers: HashMap<ValueId, Handler> = {
        let ver = &cfg[p].versions[0];
        ver.body.iter().enumerate().map(|(i, x)| (*x, ver.handler_at(i))).collect()
    };
    let mut known: HashMap<ValueId, bool> = HashMap::new();
    let mut subst: HashMap<ValueId, ValueId> = HashMap::new();
    let mut kept = vec![];
    for v in cfg[p].versions[0].body.clone() {
        let a: Vec<ValueId> = cfg[v].args.iter().map(|x| *subst.get(x).unwrap_or(x)).collect();
        cfg[v].args = a.clone();
        match &cfg[v].op {
            Op::Lit(k) => {
                if let Lit::Bool(b) = cfg.literals[*k as usize] {
                    known.insert(v, b);
                }
            }
            Op::ToBool if known.contains_key(&a[0]) => {
                subst.insert(v, a[0]);
                continue;
            }
            Op::Expect(c) if matches!(**c, Check::Value(Lit::Bool(want)) if known.get(&a[0]) == Some(&want)) => continue,
            _ => {}
        }
        kept.push(v);
    }
    let ver = &mut cfg[p].versions[0];
    for o in ver.outputs.iter_mut() {
        *o = *subst.get(o).unwrap_or(o);
    }
    ver.body = kept;
    let ver = &cfg[p].versions[0];
    let mut used: HashMap<ValueId, usize> = HashMap::new();
    for x in ver.body.iter().flat_map(|v| cfg[*v].args.iter()).chain(&ver.outputs) {
        *used.entry(*x).or_default() += 1;
    }
    let mut body = ver.body.clone();
    loop {
        let dead: Vec<ValueId> = body
            .iter()
            .copied()
            .filter(|v| {
                let op = &cfg[*v].op;
                // An empty dict can only fail to be allocated.
                let quiet = !op.can_raise() || matches!(op, Op::Bytecode { opcode: opcodes::BUILD_MAP, arg: 0 });
                quiet && op.produces_value() && used.get(v).copied().unwrap_or(0) == 0
            })
            .filter(|v| !matches!(cfg[*v].op, Op::Project(_)))
            .collect();
        if dead.is_empty() {
            break;
        }
        for v in &dead {
            for a in cfg[*v].args.clone() {
                *used.entry(a).or_default() -= 1;
            }
        }
        body.retain(|v| !dead.contains(v));
    }
    cfg[p].versions[0].body = body;
    // A read after something that writes Python state stays where it is.
    let mut wrote = false;
    super::guard::lift_guard(
        cfg,
        p,
        |cfg, x| {
            let stale = wrote;
            wrote |= super::guard::writes(cfg, x);
            stale
        },
        |x| handlers[&x].clone(),
    )
}

pub fn is_check(op: &Op) -> bool {
    matches!(op, Op::Expect(_))
}

// Copies inl's path into `out`; returns the value it returns. `call`: the
// elided instr's args.
#[allow(clippy::too_many_arguments)]
fn splice(
    cfg: &mut Cfg,
    inl: &Inlinee,
    call: &[ValueId],
    origin: u32,
    out: &mut Vec<ValueId>,
    map: &mut Vec<(usize, ValueId, ValueId)>,
    obs: &mut Observed,
    known: &mut Known,
) -> Option<ValueId> {
    let callee = &*inl.cfg;
    let add = |cfg: &mut Cfg, out: &mut Vec<ValueId>, op: Op, a: Vec<ValueId>| {
        let x = cfg.add(op, a, origin);
        out.push(x);
        x
    };
    let [func_name, self_name, call_name] = inl.names;
    // The positional args, counting self, and the checks that the function
    // is the one that runs.
    let mut args: Vec<ValueId> = vec![];
    match &inl.site {
        Site::Call { ex, method, callable } => {
            let (target, rest): (ValueId, Vec<ValueId>) = if *ex {
                if call.len() == 3 && !known.empty_dicts.contains(&call[2]) {
                    return None;
                }
                (call[0], known.tuples.get(&call[1])?.clone())
            } else if *method {
                // A method call through LoadMethod: the function, then self.
                (call[0], call[1..].to_vec())
            } else {
                (call[1], call[2..].to_vec())
            };
            match callable {
                Callable::Function => {
                    add(cfg, out, Op::expect(Check::Is(inl.function)), vec![target]);
                }
                Callable::BoundMethod => {
                    let f = add(cfg, out, Op::LoadAttr(func_name), vec![target]);
                    add(cfg, out, Op::expect(Check::Is(inl.function)), vec![f]);
                    args.push(add(cfg, out, Op::LoadAttr(self_name), vec![target]));
                }
                Callable::Instance => {
                    add(cfg, out, Op::expect(Check::TypeAttr { name: call_name, pin: inl.function }), vec![target]);
                    args.push(target);
                }
            }
            args.extend(rest);
        }
        Site::Getattr { name, str } => {
            add(cfg, out, Op::expect(Check::Getattr { name: *name, pin: inl.function }), vec![call[0]]);
            args.push(call[0]);
            args.push(add(cfg, out, Op::Pinned(*str), vec![]));
        }
    }
    if let Some((tuple, name)) = inl.defaults {
        let f = add(cfg, out, Op::Pinned(inl.function), vec![]);
        let d = add(cfg, out, Op::LoadAttr(name), vec![f]);
        add(cfg, out, Op::expect(Check::Is(tuple)), vec![d]);
    }
    let mut slots: HashMap<u32, ValueId> = HashMap::new();
    for (i, b) in inl.bind.iter().enumerate() {
        let x = match b {
            Bind::Arg(k) => *args.get(*k)?,
            Bind::Pin(pin) => add(cfg, out, Op::Pinned(*pin), vec![]),
            Bind::VarArgs(k) => {
                let items = args.get(*k..)?.to_vec();
                let t = add(cfg, out, Op::BuildTuple(items.len() as u32), items.clone());
                known.tuples.insert(t, items);
                t
            }
            Bind::VarKw => {
                let d = add(cfg, out, Op::Bytecode { opcode: opcodes::BUILD_MAP, arg: 0 }, vec![]);
                known.empty_dicts.insert(d);
                d
            }
        };
        slots.insert(i as u32, x);
    }
    // The callee's names, appended to the caller's (each once).
    let mut names: HashMap<u32, u32> = HashMap::new();
    let mut name = |cfg: &mut Cfg, n: u32| -> u32 {
        *names.entry(n).or_insert_with(|| {
            let s = &callee.names[n as usize];
            match cfg.names.iter().position(|x| x == s) {
                Some(i) => i as u32,
                None => {
                    cfg.names.push(s.clone());
                    cfg.names.len() as u32 - 1
                }
            }
        })
    };
    let konst = |cfg: &mut Cfg, k: u32| -> u32 {
        let pin = inl.consts[k as usize];
        let i = cfg.const_pins.iter().position(|p| *p == pin).unwrap_or_else(|| {
            cfg.const_pins.push(pin);
            cfg.const_pins.len() - 1
        });
        (cfg.const_reprs.len() + i) as u32
    };
    let mut vals: HashMap<ValueId, ValueId> = HashMap::new();
    let mut b = ENTRY;
    let mut params: Vec<ValueId> = vec![];
    loop {
        let blk = &callee[b];
        for (p, x) in blk.params.iter().zip(&params) {
            vals.insert(*p, *x);
        }
        let ver = &blk.versions[0];
        for x in &ver.body {
            let ins = &callee[*x];
            let get = |vals: &HashMap<ValueId, ValueId>, a: &ValueId| if a.is_reserved() { Some(*a) } else { vals.get(a).copied() };
            let a: Vec<ValueId> = ins.args.iter().map(|a| get(&vals, a)).collect::<Option<_>>()?;
            let new = match &ins.op {
                Op::Resume | Op::MakeCell { .. } => continue,
                Op::LoadLocal { slot } | Op::LoadLocalChecked { slot } | Op::LoadDeref { slot } => *slots.get(slot)?,
                Op::StoreLocal { slot } | Op::StoreDeref { slot } => {
                    slots.insert(*slot, a[0]);
                    continue;
                }
                Op::Bytecode { opcode: opcodes::DICT_MERGE, arg: 1 } => {
                    // (function, dict, update): merging an empty dict into
                    // one this elision built empty changes nothing.
                    if !(known.empty_dicts.contains(&a[1]) && known.empty_dicts.contains(&a[2])) {
                        return None;
                    }
                    continue;
                }
                Op::LoadGlobal(n) => {
                    let n = name(cfg, *n);
                    add(cfg, out, Op::LoadGlobalIn { name: n, globals: inl.globals, builtins: inl.builtins }, a)
                }
                Op::Const(k) => {
                    let k = konst(cfg, *k);
                    add(cfg, out, Op::Const(k), a)
                }
                Op::LoadAttr(_) | Op::LoadMethod(_) if inl.nested.contains_key(x) => splice(cfg, &inl.nested[x], &a, origin, out, map, obs, known)?,
                Op::LoadAttr(n) => {
                    let n = name(cfg, *n);
                    add(cfg, out, Op::LoadAttr(n), a)
                }
                Op::LoadMethod(n) => {
                    let n = name(cfg, *n);
                    add(cfg, out, Op::LoadMethod(n), a)
                }
                Op::StoreAttr(n) => {
                    let n = name(cfg, *n);
                    add(cfg, out, Op::StoreAttr(n), a)
                }
                // A projection of an elided LoadMethod: NULL, then the value.
                Op::Project(k) if inl.nested.contains_key(&ins.args[0]) => {
                    if *k == 0 {
                        NULL
                    } else {
                        a[0]
                    }
                }
                Op::Call { .. } | Op::Bytecode { opcode: opcodes::CALL_FUNCTION_EX, .. } if inl.nested.contains_key(x) => {
                    splice(cfg, &inl.nested[x], &a, origin, out, map, obs, known)?
                }
                Op::Call { argc, kw_names } => {
                    let kw_names = kw_names.map(|k| konst(cfg, k));
                    add(cfg, out, Op::Call { argc: *argc, kw_names }, a)
                }
                Op::BuildTuple(_) => {
                    let t = add(cfg, out, ins.op.clone(), a.clone());
                    known.tuples.insert(t, a);
                    t
                }
                Op::Bytecode { opcode: opcodes::BUILD_MAP, arg: 0 } => {
                    let d = add(cfg, out, ins.op.clone(), a);
                    known.empty_dicts.insert(d);
                    d
                }
                op => add(cfg, out, op.clone(), a),
            };
            vals.insert(*x, new);
            map.push((inl.id, *x, new));
            if let Some(o) = inl.obs.aten.get(x) {
                obs.aten.insert(new, o.clone());
            }
            if let Some(o) = inl.obs.calls.get(x) {
                obs.calls.insert(new, o.clone());
            }
            if inl.obs.nulls.contains(x) {
                obs.nulls.insert(new);
            }
        }
        let output = |vals: &HashMap<ValueId, ValueId>, o: OutIdx| {
            let x = ver.outputs[o as usize];
            if x.is_reserved() { Some(x) } else { vals.get(&x).copied() }
        };
        let exit = match &blk.term {
            Terminator::Jump(e) => e,
            Terminator::Branch { cond, then, else_ } => {
                let dir = inl.dirs[&b];
                let c = output(&vals, *cond)?;
                add(cfg, out, Op::expect(Check::Value(Lit::Bool(dir))), vec![c]);
                if dir { then } else { else_ }
            }
            _ => return None,
        };
        match exit {
            Exit::Return(o) => return output(&vals, *o),
            Exit::Block(e) => {
                params = e.args.iter().map(|o| output(&vals, *o)).collect::<Option<_>>()?;
                b = e.target;
            }
            Exit::Leave(_) => return None,
        }
    }
}

// Ops that only read (or check) what they're given, the namespaces they name
// and the frame's locals: what a guard may read (see guard.rs). That
// includes a call of a pinned callee with no args (a predicate like torch._C._is_tracing()).
pub fn is_read(cfg: &Cfg, v: ValueId) -> bool {
    let ins = &cfg[v];
    if let Op::Call { .. } = ins.op {
        return ins.args.len() == 2 && ins.args[0] == NULL && matches!(cfg[ins.args[1]].op, Op::Pinned(_));
    }
    matches!(
        ins.op,
        Op::Resume
            | Op::LoadLocal { .. }
            | Op::LoadGlobal(_)
            | Op::LoadGlobalIn { .. }
            | Op::LoadAttr(_)
            | Op::LoadMethod(_)
            | Op::Project(_)
            | Op::Pinned(_)
            | Op::Const(_)
            | Op::Lit(_)
            | Op::Expect(_)
            | Op::IsNone
            | Op::ToBool
            | Op::Compare(_)
            | Op::BuildTuple(_)
            | Op::Bytecode {
                opcode: opcodes::CONTAINS_OP | opcodes::BINARY_SUBSCR | opcodes::IS_OP | opcodes::BUILD_MAP,
                ..
            }
    )
}
