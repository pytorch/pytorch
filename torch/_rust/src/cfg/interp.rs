// Abstract interpreter: one pass over the partitioned blocks in RPO, simulating
// the value stack with SSA values. Fast locals are frame memory and lower to
// explicit Load/Store ops.

use super::blocks::Partition;
use super::bytecode::Decoded;
use super::ir::*;
use super::opcodes::{self as oc, In, Out};
use super::CfgError;

// Terminators as the interpreter builds them, from values; finish() turns the
// values into outputs of the block's version.
enum VExit {
    Block(BlockId, Vec<ValueId>),
    Return(ValueId),
}

enum VTerm {
    Jump(VExit),
    Branch(ValueId, VExit, VExit),
    Raise(Vec<ValueId>, Handler),
    Reraise(ValueId, Option<ValueId>, Handler),
}

#[derive(Clone)]
struct State {
    stack: Vec<ValueId>,
    // Pending KW_NAMES const index, consumed by the next CALL.
    kw_names: Option<u32>,
}

struct Interp<'a> {
    dec: &'a Decoded,
    part: &'a Partition,
    cfg: Cfg,
    nslots: usize,
    reached: Vec<bool>,
    expected_depth: Vec<Option<usize>>,
    cur: BlockId,
    cur_insn: usize,
}

// `cfg` is fresh from Cfg::new; its slot_names give the slot layout.
pub fn interpret(mut cfg: Cfg, dec: &Decoded, part: &Partition) -> Result<Cfg, CfgError> {
    let nblocks = part.blocks.len();
    cfg.blocks = (0..nblocks)
        .map(|_| Block {
            versions: vec![Version::default()],
            ..Block::new()
        })
        .collect();
    let mut it = Interp {
        dec,
        part,
        nslots: cfg.slot_names.len(),
        cfg,
        reached: vec![false; nblocks],
        expected_depth: vec![None; nblocks],
        cur: ENTRY,
        cur_insn: 0,
    };
    it.reached[0] = true;
    for b in 0..nblocks {
        if !it.reached[b] {
            continue;
        }
        it.cur = BlockId(b as u32);
        let st = it.entry_state()?;
        it.run(part.blocks[b].start, part.blocks[b].end, st)?;
    }
    let mut cfg = it.cfg;
    for b in 0..nblocks {
        if !it.reached[b] {
            cfg.remove_block(BlockId(b as u32));
        }
    }
    cfg.compact_blocks();
    Ok(cfg)
}

impl Interp<'_> {
    fn err(&self, reason: impl Into<String>) -> CfgError {
        let insn = &self.dec.insns[self.cur_insn];
        CfgError::unsupported(insn.unit * 2, oc::name(insn.op), reason)
    }

    fn param(&mut self, kind: ParamKind) -> ValueId {
        let p = self.cfg.add(Op::Param(kind), vec![], 0);
        self.cfg[self.cur].params.push(p);
        p
    }

    fn ver(&mut self) -> &mut Version {
        &mut self.cfg[self.cur].versions[0]
    }

    // Values are local to a block, so everything on the stack enters as a
    // param.
    fn entry_state(&mut self) -> Result<State, CfgError> {
        let b = self.cur;
        let handler = self.part.blocks[b.idx()].handler;
        let depth = match handler {
            _ if b == ENTRY => 0,
            Some((d, _)) => d,
            None => self.expected_depth[b.idx()].ok_or_else(|| self.err("block without a reached predecessor"))?,
        };
        let mut stack: Vec<ValueId> = (0..depth).map(|i| self.param(ParamKind::Stack(i as u32))).collect();
        if let Some((_, lasti)) = handler {
            if lasti {
                stack.push(self.param(ParamKind::Lasti));
            }
            stack.push(self.param(ParamKind::Exc));
        }
        Ok(State {
            stack,
            kw_names: None,
        })
    }

    fn jump_edge(&mut self, t: BlockId, st: &State) -> Result<VExit, CfgError> {
        if st.kw_names.is_some() {
            return Err(self.err("KW_NAMES pending across a block boundary"));
        }
        self.reached[t.idx()] = true;
        let d = *self.expected_depth[t.idx()].get_or_insert(st.stack.len());
        if d != st.stack.len() {
            return Err(self.err(format!("stack depth mismatch at bb{}: {} vs {}", t.0, d, st.stack.len())));
        }
        Ok(VExit::Block(t, st.stack.clone()))
    }

    // Where a raise at the current instruction goes, with `st` the state
    // before it.
    fn handler(&mut self, st: &State, is_yield: bool) -> Result<Handler, CfgError> {
        Ok(match self.dec.insn_entry[self.cur_insn] {
            None => Handler::Propagate,
            Some(e) => {
                let ent = &self.dec.entries[e];
                let hb = self.part.entry_block[e].ok_or_else(|| self.err("unreachable handler"))?;
                let d = ent.depth;
                if st.stack.len() < d {
                    return Err(self.err("stack shallower than the handler depth"));
                }
                // The yielded value has already gone to the caller when a
                // throw() unwinds at the yield.
                let yielded = is_yield && st.stack.len() == d && d > 0;
                let stack = st.stack[..d].iter().enumerate();
                let args = stack.map(|(k, v)| if yielded && k == d - 1 { UNKNOWN } else { *v }).collect();
                self.reached[hb.idx()] = true;
                let handlers = &mut self.cfg[self.cur].handlers;
                let k = handlers.iter().position(|h| *h == hb).unwrap_or_else(|| {
                    handlers.push(hb);
                    handlers.len() - 1
                });
                Handler::Block(k, args)
            }
        })
    }

    // Record that the op about to be placed at body[at] can raise, with `st`
    // the state before the op.
    fn raise_point(&mut self, st: &State, at: usize, is_yield: bool) -> Result<(), CfgError> {
        let handler = self.handler(st, is_yield)?;
        let ver = self.ver();
        let new_segment = match ver.unwind.last() {
            None => handler != Handler::Propagate,
            Some(last) => last.handler != handler,
        };
        if new_segment {
            ver.unwind.push(UnwindSegment { from: at, handler });
        }
        Ok(())
    }

    fn emit(&mut self, st: &State, op: Op, args: Vec<ValueId>) -> Result<ValueId, CfgError> {
        if op.can_raise() {
            let at = self.ver().body.len();
            let is_yield = op == Op::Yield;
            self.raise_point(st, at, is_yield)?;
        }
        let v = self.cfg.add(op, args, self.origin());
        self.ver().body.push(v);
        Ok(v)
    }

    fn finish(&mut self, term: VTerm) -> Result<(), CfgError> {
        let origin = self.origin();
        let ver = self.ver();
        let mut out = |v: ValueId| {
            let k = ver.outputs.iter().position(|o| *o == v).unwrap_or_else(|| {
                ver.outputs.push(v);
                ver.outputs.len() - 1
            });
            k as OutIdx
        };
        let mut exit = |e: VExit| match e {
            VExit::Block(target, args) => Exit::Block(Edge {
                target,
                args: args.into_iter().map(&mut out).collect(),
            }),
            VExit::Return(v) => Exit::Return(out(v)),
        };
        let unwind = |h: Handler, out: &mut dyn FnMut(ValueId) -> OutIdx| match h {
            Handler::Block(k, args) => TermUnwind::Block(k, args.into_iter().map(out).collect()),
            Handler::Outer(..) => unreachable!("the interpreter builds top-level Cfgs"),
            Handler::Propagate => TermUnwind::Propagate,
        };
        let term = match term {
            VTerm::Jump(e) => Terminator::Jump(exit(e)),
            VTerm::Branch(cond, then, else_) => {
                let (then, else_) = (exit(then), exit(else_));
                Terminator::Branch {
                    cond: out(cond),
                    then,
                    else_,
                }
            }
            VTerm::Raise(args, h) => Terminator::Raise {
                args: args.into_iter().map(&mut out).collect(),
                unwind: unwind(h, &mut out),
            },
            VTerm::Reraise(exc, lasti, h) => Terminator::Reraise {
                exc: out(exc),
                lasti: lasti.map(&mut out),
                unwind: unwind(h, &mut out),
            },
        };
        let blk = &mut self.cfg[self.cur];
        blk.term = term;
        blk.term_origin = origin;
        Ok(())
    }

    fn origin(&self) -> u32 {
        self.dec.insns[self.cur_insn].unit
    }

    fn peek(&self, st: &State, n: usize) -> Result<Vec<ValueId>, CfgError> {
        if st.stack.len() < n {
            return Err(self.err("stack underflow"));
        }
        Ok(st.stack[st.stack.len() - n..].to_vec())
    }

    // The i-th value from the top of the stack (0 = TOS).
    fn top(&self, st: &State, i: usize) -> Result<ValueId, CfgError> {
        match st.stack.len().checked_sub(i + 1) {
            Some(k) => Ok(st.stack[k]),
            None => Err(self.err("stack underflow")),
        }
    }

    fn pop(&self, st: &mut State, n: usize) -> Result<(), CfgError> {
        if st.stack.len() < n {
            return Err(self.err("stack underflow"));
        }
        st.stack.truncate(st.stack.len() - n);
        Ok(())
    }

    fn slot(&self, i: u32) -> Result<u32, CfgError> {
        if (i as usize) < self.nslots {
            Ok(i)
        } else {
            Err(self.err("slot index out of range"))
        }
    }

    fn push(&mut self, st: &mut State, op: Op) -> Result<(), CfgError> {
        let v = self.emit(st, op, vec![])?;
        st.stack.push(v);
        Ok(())
    }

    // Emit `op` consuming the top value, producing nothing.
    fn store(&mut self, st: &mut State, op: Op) -> Result<(), CfgError> {
        let v = self.top(st, 0)?;
        self.emit(st, op, vec![v])?;
        self.pop(st, 1)
    }

    // Emit `op` consuming the top `n` values and push its result.
    fn unary_like(&mut self, st: &mut State, op: Op, n: usize) -> Result<(), CfgError> {
        let ins = self.peek(st, n)?;
        let v = self.emit(st, op, ins)?;
        self.pop(st, n)?;
        st.stack.push(v);
        Ok(())
    }

    fn run(&mut self, start: usize, end: usize, mut st: State) -> Result<(), CfgError> {
        let part = self.part;
        let succs = &part.blocks[self.cur.idx()].succs;
        for i in start..end {
            self.cur_insn = i;
            let insn = self.dec.insns[i];
            let arg = insn.arg;
            match insn.op {
                oc::NOP => {}
                oc::RESUME => {
                    self.emit(&st, Op::Resume, vec![])?;
                }
                oc::POP_TOP => self.pop(&mut st, 1)?,
                oc::PUSH_NULL => st.stack.push(NULL),
                oc::END_FOR => self.pop(&mut st, 2)?,
                oc::END_SEND => {
                    let v = self.top(&st, 0)?;
                    self.pop(&mut st, 2)?;
                    st.stack.push(v);
                }
                oc::COPY => {
                    let v = self.top(&st, arg as usize - 1)?;
                    st.stack.push(v);
                }
                oc::SWAP => {
                    let n = arg as usize;
                    self.peek(&st, n)?;
                    let len = st.stack.len();
                    st.stack.swap(len - 1, len - n);
                }
                oc::KW_NAMES => st.kw_names = Some(arg),
                oc::LOAD_CONST => self.push(&mut st, Op::Const(arg))?,
                oc::LOAD_FAST | oc::LOAD_CLOSURE => {
                    let slot = self.slot(arg)?;
                    self.push(&mut st, Op::LoadLocal { slot })?;
                }
                oc::LOAD_FAST_CHECK => {
                    let slot = self.slot(arg)?;
                    self.push(&mut st, Op::LoadLocalChecked { slot })?;
                }
                oc::LOAD_FAST_AND_CLEAR => {
                    let slot = self.slot(arg)?;
                    self.push(&mut st, Op::LoadLocal { slot })?;
                    self.emit(&st, Op::StoreLocal { slot }, vec![NULL])?;
                }
                oc::STORE_FAST => {
                    let slot = self.slot(arg)?;
                    self.store(&mut st, Op::StoreLocal { slot })?;
                }
                oc::DELETE_FAST => {
                    let slot = self.slot(arg)?;
                    self.emit(&st, Op::DeleteLocal { slot }, vec![])?;
                }
                oc::MAKE_CELL => {
                    let slot = self.slot(arg)?;
                    self.emit(&st, Op::MakeCell { slot }, vec![])?;
                }
                oc::COPY_FREE_VARS => {
                    if arg as usize > self.nslots {
                        return Err(self.err("more free vars than slots"));
                    }
                    self.emit(&st, Op::CopyFreeVars(arg), vec![])?;
                }
                oc::LOAD_DEREF => {
                    let slot = self.slot(arg)?;
                    self.push(&mut st, Op::LoadDeref { slot })?;
                }
                oc::STORE_DEREF => {
                    let slot = self.slot(arg)?;
                    self.store(&mut st, Op::StoreDeref { slot })?;
                }
                oc::DELETE_DEREF => {
                    let slot = self.slot(arg)?;
                    self.emit(&st, Op::DeleteDeref { slot }, vec![])?;
                }
                oc::LOAD_FROM_DICT_OR_DEREF => {
                    let slot = self.slot(arg)?;
                    self.unary_like(&mut st, Op::LoadFromDictOrDeref { slot }, 1)?;
                }
                oc::LOAD_GLOBAL => {
                    let v = self.emit(&st, Op::LoadGlobal(arg >> 1), vec![])?;
                    if arg & 1 != 0 {
                        st.stack.push(NULL);
                    }
                    st.stack.push(v);
                }
                oc::LOAD_ATTR => {
                    let owner = self.peek(&st, 1)?;
                    if arg & 1 != 0 {
                        let m = self.emit(&st, Op::LoadMethod(arg >> 1), owner)?;
                        self.pop(&mut st, 1)?;
                        let meth = self.emit(&st, Op::Project(0), vec![m])?;
                        let this = self.emit(&st, Op::Project(1), vec![m])?;
                        st.stack.push(meth);
                        st.stack.push(this);
                    } else {
                        self.unary_like(&mut st, Op::LoadAttr(arg >> 1), 1)?;
                    }
                }
                oc::STORE_ATTR => {
                    let ins = self.peek(&st, 2)?;
                    self.emit(&st, Op::StoreAttr(arg), ins)?;
                    self.pop(&mut st, 2)?;
                }
                oc::COMPARE_OP => self.unary_like(&mut st, Op::Compare(arg >> 4), 2)?,
                oc::BINARY_OP => self.unary_like(&mut st, Op::Binary(arg), 2)?,
                oc::BUILD_TUPLE => self.unary_like(&mut st, Op::BuildTuple(arg), arg as usize)?,
                oc::GET_ITER => self.unary_like(&mut st, Op::GetIter, 1)?,
                oc::MAKE_FUNCTION => {
                    let n = (arg & 0xf).count_ones() as usize + 1;
                    self.unary_like(&mut st, Op::MakeFunction(arg), n)?;
                }
                oc::CALL => {
                    let kw_names = st.kw_names.take();
                    let op = Op::Call { argc: arg, kw_names };
                    self.unary_like(&mut st, op, arg as usize + 2)?;
                }
                // Yield reads the whole stack: a suspended frame holds it, so
                // it stays alive (and is saved to the frame) across the yield.
                oc::YIELD_VALUE => {
                    let v = self.emit(&st, Op::Yield, st.stack.clone())?;
                    self.pop(&mut st, 1)?;
                    st.stack.push(v);
                }
                oc::RETURN_CONST => {
                    let v = self.emit(&st, Op::Const(arg), vec![])?;
                    return self.finish(VTerm::Jump(VExit::Return(v)));
                }
                oc::RETURN_VALUE => {
                    let v = self.top(&st, 0)?;
                    return self.finish(VTerm::Jump(VExit::Return(v)));
                }
                oc::POP_JUMP_IF_TRUE
                | oc::POP_JUMP_IF_FALSE
                | oc::POP_JUMP_IF_NONE
                | oc::POP_JUMP_IF_NOT_NONE => {
                    let x = self.peek(&st, 1)?;
                    let op = if matches!(insn.op, oc::POP_JUMP_IF_TRUE | oc::POP_JUMP_IF_FALSE) {
                        Op::ToBool
                    } else {
                        Op::IsNone
                    };
                    let cond = self.emit(&st, op, x)?;
                    self.pop(&mut st, 1)?;
                    let then = self.jump_edge(succs[0], &st)?;
                    let else_ = self.jump_edge(succs[1], &st)?;
                    return self.finish(VTerm::Branch(cond, then, else_));
                }
                oc::JUMP_FORWARD | oc::JUMP_BACKWARD_NO_INTERRUPT => {
                    let call = self.jump_edge(succs[0], &st)?;
                    return self.finish(VTerm::Jump(call));
                }
                oc::JUMP_BACKWARD => {
                    self.emit(&st, Op::EvalBreaker, vec![])?;
                    let call = self.jump_edge(succs[0], &st)?;
                    return self.finish(VTerm::Jump(call));
                }
                oc::FOR_ITER => {
                    let iter = self.peek(&st, 1)?;
                    let r = self.emit(&st, Op::IterNext, iter)?;
                    let item = self.emit(&st, Op::Project(0), vec![r])?;
                    let more = self.emit(&st, Op::Project(1), vec![r])?;
                    let mut next = st.clone();
                    next.stack.push(item);
                    let then = self.jump_edge(succs[0], &next)?;
                    self.pop(&mut st, 1)?;
                    let else_ = self.jump_edge(succs[1], &st)?;
                    return self.finish(VTerm::Branch(more, then, else_));
                }
                oc::SEND => {
                    let rv = self.peek(&st, 2)?;
                    let r = self.emit(&st, Op::Send, rv)?;
                    let out = self.emit(&st, Op::Project(0), vec![r])?;
                    let yielded = self.emit(&st, Op::Project(1), vec![r])?;
                    self.pop(&mut st, 1)?;
                    st.stack.push(out);
                    let then = self.jump_edge(succs[0], &st)?;
                    let else_ = self.jump_edge(succs[1], &st)?;
                    return self.finish(VTerm::Branch(yielded, then, else_));
                }
                oc::RAISE_VARARGS => {
                    let args = self.peek(&st, arg as usize)?;
                    let h = self.handler(&st, false)?;
                    return self.finish(VTerm::Raise(args, h));
                }
                oc::RERAISE => {
                    let vals = self.peek(&st, arg as usize + 1)?;
                    let lasti = (arg > 0).then(|| vals[0]);
                    let h = self.handler(&st, false)?;
                    return self.finish(VTerm::Reraise(vals[arg as usize], lasti, h));
                }
                op => self.generic(&mut st, op, arg)?,
            }
        }
        // Fell through into the next block.
        let call = self.jump_edge(succs[0], &st)?;
        self.finish(VTerm::Jump(call))
    }

    fn generic(&mut self, st: &mut State, op: u8, arg: u32) -> Result<(), CfgError> {
        if !(oc::caches(op).is_some()
            && !matches!(op, oc::CACHE | oc::RESERVED | oc::INTERPRETER_EXIT | oc::EXTENDED_ARG))
        {
            return Err(self.err("unsupported opcode"));
        }
        let sig = oc::signature(op, arg, false).ok_or_else(|| self.err("no stack signature"))?;
        let ins = self.peek(st, sig.inputs.len())?;
        let args = ins
            .iter()
            .zip(&sig.inputs)
            .filter(|(_, k)| **k == In::Used)
            .map(|(v, _)| *v)
            .collect();
        let v = self.emit(st, Op::Bytecode { opcode: op, arg }, args)?;
        let nnew = sig.outputs.iter().filter(|o| **o == Out::New).count();
        let mut outs = Vec::with_capacity(sig.outputs.len());
        for (k, o) in sig.outputs.iter().enumerate() {
            outs.push(match o {
                Out::Keep(j) => ins[*j],
                Out::New if nnew == 1 => v,
                Out::New => self.emit(st, Op::Project(k as u32), vec![v])?,
            });
        }
        self.pop(st, sig.inputs.len())?;
        st.stack.extend(outs);
        Ok(())
    }
}
