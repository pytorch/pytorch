//! Liveness of SSA values, and what follows from it: when registers are
//! released, and what the generic tier needs when specialized code bails out
//! in the middle of a block.

use crate::ir::*;

#[derive(Clone)]
pub struct BitSet {
    words: Vec<u64>,
}

impl BitSet {
    pub fn new(n: usize) -> BitSet {
        BitSet { words: vec![0; n.div_ceil(64)] }
    }

    #[inline]
    pub fn contains(&self, v: V) -> bool {
        let v = v as usize;
        self.words.get(v / 64).is_some_and(|w| w & (1 << (v % 64)) != 0)
    }

    /// Returns whether the value was newly inserted.
    #[inline]
    pub fn insert(&mut self, v: V) -> bool {
        let v = v as usize;
        let bit = 1u64 << (v % 64);
        let word = &mut self.words[v / 64];
        let fresh = *word & bit == 0;
        *word |= bit;
        fresh
    }

    #[inline]
    pub fn remove(&mut self, v: V) {
        let v = v as usize;
        self.words[v / 64] &= !(1u64 << (v % 64));
    }

    pub fn union_with(&mut self, other: &BitSet) -> bool {
        let mut changed = false;
        for (a, b) in self.words.iter_mut().zip(&other.words) {
            let merged = *a | *b;
            changed |= merged != *a;
            *a = merged;
        }
        changed
    }

    pub fn iter(&self) -> impl Iterator<Item = V> + '_ {
        self.words.iter().enumerate().flat_map(|(i, w)| {
            let mut w = *w;
            std::iter::from_fn(move || {
                if w == 0 {
                    return None;
                }
                let bit = w.trailing_zeros();
                w &= w - 1;
                Some((i * 64) as V + bit)
            })
        })
    }
}

pub struct Analysis {
    /// Live at the start of the block, after the phi moves of the incoming
    /// edge.
    pub live_in: Vec<BitSet>,
    pub live_out: Vec<BitSet>,
    /// Per block and predecessor (in `preds` order): the (dst, src) register
    /// moves that implement the phis.
    pub edge_moves: Vec<Vec<Vec<(V, V)>>>,
    /// Per block and predecessor: registers that die on that edge.
    pub edge_kills: Vec<Vec<Vec<V>>>,
    /// Defining block of each value; NONE for the undefined value.
    pub def_block: Vec<u32>,
    /// Index of the defining instruction in its block; NONE for parameters
    /// and phis.
    pub def_index: Vec<u32>,
    pub use_count: Vec<u32>,
}

impl Analysis {
    pub fn def_op<'a>(&self, func: &'a Function, v: V) -> Option<&'a Op> {
        let (b, i) = (self.def_block[v as usize], self.def_index[v as usize]);
        if b == NONE || i == NONE {
            return None;
        }
        Some(&func.blocks[b as usize].insts[i as usize].op)
    }

    /// Values available when block `b` is entered: defined on every path to
    /// it and not yet released.
    pub fn available_at_entry(&self, b: usize, v: V) -> bool {
        self.live_in[b].contains(v)
    }

    /// The values the generic code of block `b` needs when it is resumed at
    /// instruction `index`.
    pub fn live_before(&self, func: &Function, b: usize, index: usize) -> BitSet {
        let block = &func.blocks[b];
        let mut live = self.live_out[b].clone();
        block.term.for_each_use(|v| {
            live.insert(v);
        });
        for inst in block.insts[index..].iter().rev() {
            if inst.dst != NONE {
                live.remove(inst.dst);
            }
            inst.op.for_each_use(|v| {
                live.insert(v);
            });
        }
        live.remove(UNDEF);
        live
    }
}

pub fn analyze(func: &Function) -> Analysis {
    let n = func.nvalues as usize;
    let nblocks = func.blocks.len();

    let mut def_block = vec![NONE; n];
    let mut def_index = vec![NONE; n];
    let mut use_count = vec![0u32; n];
    for v in &func.params {
        def_block[*v as usize] = 0;
    }
    for (b, block) in func.blocks.iter().enumerate() {
        for phi in &block.phis {
            def_block[phi.dst as usize] = b as u32;
            for s in &phi.srcs {
                use_count[*s as usize] += 1;
            }
        }
        for (i, inst) in block.insts.iter().enumerate() {
            if inst.dst != NONE {
                def_block[inst.dst as usize] = b as u32;
                def_index[inst.dst as usize] = i as u32;
            }
            inst.op.for_each_use(|v| use_count[v as usize] += 1);
        }
        block.term.for_each_use(|v| use_count[v as usize] += 1);
    }

    // Upward-exposed uses and definitions of the instructions of each block.
    let mut gen: Vec<BitSet> = Vec::with_capacity(nblocks);
    let mut kill: Vec<BitSet> = Vec::with_capacity(nblocks);
    for block in &func.blocks {
        let mut g = BitSet::new(n);
        let mut k = BitSet::new(n);
        for inst in &block.insts {
            inst.op.for_each_use(|v| {
                if !k.contains(v) {
                    g.insert(v);
                }
            });
            if inst.dst != NONE {
                k.insert(inst.dst);
            }
        }
        block.term.for_each_use(|v| {
            if !k.contains(v) {
                g.insert(v);
            }
        });
        g.remove(UNDEF);
        gen.push(g);
        kill.push(k);
    }

    let succs: Vec<Vec<B>> = func.blocks.iter().map(|b| b.term.successors()).collect();
    let mut live_in: Vec<BitSet> = gen.clone();
    let mut live_out: Vec<BitSet> = (0..nblocks).map(|_| BitSet::new(n)).collect();
    let mut changed = true;
    while changed {
        changed = false;
        for b in (0..nblocks).rev() {
            let mut out = BitSet::new(n);
            for s in &succs[b] {
                let s = *s as usize;
                let succ = &func.blocks[s];
                // What the successor needs from its predecessors: its
                // live-ins that are not phis, and the phi sources of this
                // edge.
                let mut need = live_in[s].clone();
                for phi in &succ.phis {
                    need.remove(phi.dst);
                }
                out.union_with(&need);
                if let Some(pi) = succ.preds.iter().position(|p| *p as usize == b) {
                    for phi in &succ.phis {
                        if phi.srcs[pi] != UNDEF {
                            out.insert(phi.srcs[pi]);
                        }
                    }
                }
            }
            if live_out[b].union_with(&out) {
                changed = true;
            }
            let mut inn = live_out[b].clone();
            for v in kill[b].iter() {
                inn.remove(v);
            }
            inn.union_with(&gen[b]);
            if live_in[b].union_with(&inn) {
                changed = true;
            }
        }
    }

    let mut edge_moves = Vec::with_capacity(nblocks);
    let mut edge_kills = Vec::with_capacity(nblocks);
    for (b, block) in func.blocks.iter().enumerate() {
        let mut moves = Vec::with_capacity(block.preds.len());
        let mut kills = Vec::with_capacity(block.preds.len());
        for (pi, pred) in block.preds.iter().enumerate() {
            let m: Vec<(V, V)> = block.phis.iter().map(|phi| (phi.dst, phi.srcs[pi])).collect();
            let mut k: Vec<V> = live_out[*pred as usize]
                .iter()
                .filter(|v| !live_in[b].contains(*v))
                .collect();
            for phi in &block.phis {
                if !live_in[b].contains(phi.dst) {
                    k.push(phi.dst);
                }
            }
            moves.push(m);
            kills.push(k);
        }
        edge_moves.push(moves);
        edge_kills.push(kills);
    }

    Analysis { live_in, live_out, edge_moves, edge_kills, def_block, def_index, use_count }
}

/// Fills in the kill lists of a block's code: every value is released right
/// after its last use, unless it is live out of the block. Returns the values
/// that die at the terminator.
pub fn compute_kills(insts: &mut [Inst], term: &Term, live_out: &BitSet) -> Vec<V> {
    let mut live = live_out.clone();
    let mut term_kills = Vec::new();
    term.for_each_use(|v| {
        if v != UNDEF && live.insert(v) {
            term_kills.push(v);
        }
    });
    for inst in insts.iter_mut().rev() {
        let mut kills = Vec::new();
        if inst.dst != NONE {
            if !live.contains(inst.dst) {
                // Never read: release it as soon as it is produced.
                kills.push(inst.dst);
            }
            live.remove(inst.dst);
        }
        inst.op.for_each_use(|v| {
            if v != UNDEF && live.insert(v) {
                kills.push(v);
            }
        });
        inst.kills = kills;
    }
    term_kills
}
