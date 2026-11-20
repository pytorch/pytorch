// Splits a joint graph (see exec/joint.rs) into a forward and a backward, as
// AOTAutograd's min-cut rematerialization partitioner does.
//
// The backward needs some forward values. Each is either saved by the forward
// or recomputed by the backward from values that are. Which is a minimum cut
// in a flow network: every forward value v is an edge v_in -> v_out whose
// capacity is what saving v costs (its size), with an edge of infinite
// capacity u_out -> v_in for each arg u of v. The source feeds the inputs
// (which a node saves anyway, so saving them is free) and every value that
// may not be recomputed (anything but pointwise ops and views); the values the
// backward uses feed the sink. The cut edges are the values to save, and what
// the backward needs on the sink side is recomputed there.
//
// With nothing recomputable this saves exactly the values the backward uses.
// Literals are never saved: the backward makes its own.

use std::collections::{HashMap, HashSet, VecDeque};

use super::ir::*;

pub struct Split {
    pub forward: Cfg,
    pub backward: Cfg,
}

// `joint` is one block whose params are n_inputs forward inputs then the
// tangents, whose first fwd_len body instrs are the forward, and whose
// outputs are n_fwd_outputs forward outputs then the gradients. `bytes`
// gives a value's size (None: unknown, never worth saving if avoidable).
pub fn partition(
    joint: &Cfg,
    n_inputs: usize,
    n_fwd_outputs: usize,
    fwd_len: usize,
    bytes: &dyn Fn(ValueId) -> Option<u64>,
    recomputable: &dyn Fn(ValueId) -> bool,
) -> Split {
    let blk = &joint[ENTRY];
    let ver = &blk.versions[0];
    let (fwd_body, bwd_body) = ver.body.split_at(fwd_len);
    let (inputs, tangents) = blk.params.split_at(n_inputs);
    let (fwd_outs, grads) = ver.outputs.split_at(n_fwd_outputs);
    let is_lit = |v: &ValueId| matches!(joint[*v].op, Op::Lit(_));
    let values: Vec<ValueId> = inputs.iter().chain(fwd_body).filter(|v| !is_lit(v)).copied().collect();
    let index: HashMap<ValueId, usize> = values.iter().enumerate().map(|(i, v)| (*v, i)).collect();
    let used_by_backward: Vec<ValueId> = {
        let mut seen = HashSet::new();
        bwd_body.iter().flat_map(|v| joint[*v].args.iter()).chain(grads).filter(|u| index.contains_key(u) && seen.insert(**u)).copied().collect()
    };

    // Nodes: 2i is values[i]_in, 2i+1 its _out; then the source and sink.
    let (source, sink) = (2 * values.len(), 2 * values.len() + 1);
    let mut g = Flow::new(sink + 1);
    for (i, v) in values.iter().enumerate() {
        let input = i < inputs.len();
        g.edge(2 * i, 2 * i + 1, if input { 0 } else { bytes(*v).unwrap_or(INF / 4) });
        if input || !recomputable(*v) {
            g.edge(source, 2 * i, INF);
        }
        for a in &joint[*v].args {
            if let Some(j) = index.get(a) {
                g.edge(2 * j + 1, 2 * i, INF);
            }
        }
    }
    for u in &used_by_backward {
        g.edge(2 * index[u] + 1, sink, INF);
    }
    g.max_flow(source, sink);
    let reach = g.reachable(source);
    let saved: Vec<ValueId> = values.iter().enumerate().filter(|(i, _)| reach[2 * i] && !reach[2 * i + 1]).map(|(_, v)| *v).collect();

    // What the backward recomputes: the forward instrs it needs that aren't
    // saved, in forward order, with the literals those and its own instrs use.
    let saved_set: HashSet<ValueId> = saved.iter().copied().collect();
    let mut need: HashSet<ValueId> = HashSet::new();
    let mut work: Vec<ValueId> = bwd_body.iter().flat_map(|v| joint[*v].args.iter()).chain(grads).copied().collect();
    while let Some(u) = work.pop() {
        let in_forward = index.contains_key(&u) || (is_lit(&u) && fwd_body.contains(&u));
        if !in_forward || saved_set.contains(&u) || !need.insert(u) {
            continue;
        }
        work.extend(joint[u].args.iter().copied());
        // A Project's producer brings its other Projects along.
        if let Op::Project(_) = joint[u].op {
            work.push(joint[u].args[0]);
        }
    }
    let recomputed = fwd_body.iter().filter(|v| need.contains(v) || is_projected_from(joint, v, &need));
    let bwd: Vec<ValueId> = recomputed.chain(bwd_body).copied().collect();
    let fwd_outputs: Vec<ValueId> = fwd_outs.iter().chain(&saved).copied().collect();
    let bwd_params: Vec<ValueId> = saved.iter().chain(tangents).copied().collect();
    Split {
        forward: extract(joint, inputs, fwd_body, &fwd_outputs),
        backward: extract(joint, &bwd_params, &bwd, grads),
    }
}

// Whether v is a Project of an instr in `need`.
fn is_projected_from(joint: &Cfg, v: &ValueId, need: &HashSet<ValueId>) -> bool {
    matches!(joint[*v].op, Op::Project(_)) && need.contains(&joint[*v].args[0])
}

const INF: u64 = u64::MAX / 4;

// A flow network, for Dinic's max-flow algorithm.
struct Flow {
    // By node: indices into `edges`. Edge e's reverse is e ^ 1.
    adj: Vec<Vec<usize>>,
    edges: Vec<(usize, u64)>,
}

impl Flow {
    fn new(n: usize) -> Self {
        Flow { adj: vec![vec![]; n], edges: vec![] }
    }

    fn edge(&mut self, from: usize, to: usize, cap: u64) {
        self.adj[from].push(self.edges.len());
        self.edges.push((to, cap));
        self.adj[to].push(self.edges.len());
        self.edges.push((from, 0));
    }

    fn levels(&self, s: usize) -> Vec<Option<usize>> {
        let mut level = vec![None; self.adj.len()];
        level[s] = Some(0);
        let mut q = VecDeque::from([s]);
        while let Some(u) = q.pop_front() {
            for &e in &self.adj[u] {
                let (v, cap) = self.edges[e];
                if cap > 0 && level[v].is_none() {
                    level[v] = Some(level[u].unwrap() + 1);
                    q.push_back(v);
                }
            }
        }
        level
    }

    fn push(&mut self, u: usize, t: usize, f: u64, level: &[Option<usize>], next: &mut [usize]) -> u64 {
        if u == t {
            return f;
        }
        while next[u] < self.adj[u].len() {
            let e = self.adj[u][next[u]];
            let (v, cap) = self.edges[e];
            if cap > 0 && level[v] == level[u].map(|l| l + 1) {
                let pushed = self.push(v, t, f.min(cap), level, next);
                if pushed > 0 {
                    self.edges[e].1 -= pushed;
                    self.edges[e ^ 1].1 += pushed;
                    return pushed;
                }
            }
            next[u] += 1;
        }
        0
    }

    fn max_flow(&mut self, s: usize, t: usize) {
        loop {
            let level = self.levels(s);
            if level[t].is_none() {
                return;
            }
            let mut next = vec![0; self.adj.len()];
            while self.push(s, t, INF, &level, &mut next) > 0 {}
        }
    }

    // Nodes reachable from s in the residual network.
    fn reachable(&self, s: usize) -> Vec<bool> {
        self.levels(s).iter().map(Option::is_some).collect()
    }
}

// A graph of `body` (instrs of `src`) taking `params` (values of `src`) and
// leaving with `outputs`, without the instrs those don't need.
fn extract(src: &Cfg, params: &[ValueId], body: &[ValueId], outputs: &[ValueId]) -> Cfg {
    let mut live: HashSet<ValueId> = outputs.iter().copied().collect();
    for v in body.iter().rev() {
        if live.contains(v) {
            live.extend(src[*v].args.iter().copied());
        }
    }
    let mut cfg = Cfg::new(vec![], vec![], vec![]);
    cfg.literals = src.literals.clone();
    let mut map: HashMap<ValueId, ValueId> = HashMap::new();
    let mut blk = Block::new();
    for (i, p) in params.iter().enumerate() {
        let np = cfg.add(Op::Param(ParamKind::Stack(i as u32)), vec![], 0);
        map.insert(*p, np);
        blk.params.push(np);
    }
    let get = |map: &HashMap<ValueId, ValueId>, v: &ValueId| if v.is_reserved() { *v } else { map[v] };
    let mut nbody = vec![];
    for v in body.iter().filter(|v| live.contains(v)) {
        let ins = &src[*v];
        let args = ins.args.iter().map(|a| get(&map, a)).collect();
        let nv = cfg.add(ins.op.clone(), args, ins.origin);
        map.insert(*v, nv);
        nbody.push(nv);
    }
    let outs: Vec<ValueId> = outputs.iter().map(|o| get(&map, o)).collect();
    let n = outs.len();
    blk.versions.push(Version { body: nbody, outputs: outs, ..Version::default() });
    blk.term = Terminator::Jump(Exit::Leave((0..n as u32).collect()));
    cfg.boundary = Some(Boundary { n_outputs: n, handler_arity: vec![] });
    cfg.blocks.push(blk);
    cfg
}

// Puts a version running `t` with its own autograd node before b's others,
// guarded by b's first version's checks and `checks` (each on the node's arg
// at that position, or global). Its exceptions go
// where b's first version's do; that has to be one handler taking only b's
// params (the training version has no other values).
pub fn add_training_version(cfg: &mut Cfg, b: BlockId, checks: Vec<(Option<u32>, Check)>, t: Training) -> Result<(), String> {
    let blk = &cfg[b];
    let first = &blk.versions[0];
    let mut handlers = first.unwind.iter().map(|s| &s.handler);
    let handler = handlers.next().cloned();
    if handlers.any(|h| Some(h) != handler.as_ref()) {
        return Err("its ops raise into several handlers".into());
    }
    if let Some(Handler::Block(_, args) | Handler::Outer(_, args)) = &handler {
        if args.iter().any(|a| !blk.params.contains(a) && !a.is_reserved()) {
            return Err("it raises into a handler taking values it made".into());
        }
    }
    let mut guard = first.guard.clone();
    let params = blk.params.clone();
    let n = t.outputs.len();
    let origin = first.body.first().map_or(0, |v| cfg[*v].origin);
    // The node takes the params, then the reads the first version takes;
    // `checks` name them by that position.
    let inputs = first.inputs.clone();
    let args: Vec<ValueId> = params.iter().chain(&inputs).copied().collect();
    guard.checks.extend(checks.into_iter().map(|(i, c)| (i.map(|i| args[i as usize]), c)));
    cfg.trainings.push(std::sync::Arc::new(t));
    let v = cfg.add(Op::Autograd(cfg.trainings.len() as u32 - 1), args, origin);
    let outputs: Vec<ValueId> = (0..n).map(|k| cfg.add(Op::Project(k as u32), vec![v], origin)).collect();
    let mut body = vec![v];
    body.extend(outputs.iter().copied());
    let unwind = handler.map(|h| vec![UnwindSegment { from: 0, handler: h }]).unwrap_or_default();
    cfg[b].versions.insert(0, Version { guard, body, outputs, unwind, inputs });
    Ok(())
}
