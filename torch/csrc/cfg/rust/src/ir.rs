//! The SSA control-flow-graph IR.
//!
//! A `Function` is built once from the bytecode of a Python function and never
//! changes. Its instructions have exactly the semantics of the bytecode they
//! came from.

use crate::pyffi::{self, PyObj};
use std::fmt::Write;

pub type V = u32;
pub type B = u32;

pub const NONE: V = u32::MAX;
/// The undefined value (an unbound local). Its register is always null.
pub const UNDEF: V = 0;

pub const CMP_SYMBOLS: [&str; 6] = ["<", "<=", "==", "!=", ">", ">="];

pub const BINARY_SYMBOLS: [&str; 26] = [
    "+", "&", "//", "<<", "@", "*", "%", "|", "**", ">>", "-", "/", "^", "+=", "&=", "//=", "<<=",
    "@=", "*=", "%=", "|=", "**=", ">>=", "-=", "/=", "^=",
];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum UnaryKind {
    Neg,
    Pos,
    Invert,
    Not,
}

#[derive(Clone)]
pub enum Op {
    Const(u32),
    LoadGlobal(u32),
    LoadDeref(u32),
    LoadAttr(V, u32),
    StoreAttr { obj: V, name: u32, val: V },
    StoreGlobal(u32, V),
    Binary(u8, V, V),
    Unary(UnaryKind, V),
    Compare(u8, V, V),
    Is(V, V, bool),
    /// item, container, invert
    Contains(V, V, bool),
    /// Truthiness as an exact bool; branches only take exact bools.
    ToBool(V),
    IsNone(V, bool),
    GetItem(V, V),
    SetItem { obj: V, key: V, val: V },
    DelItem(V, V),
    BuildTuple(Vec<V>),
    BuildList(Vec<V>),
    BuildSet(Vec<V>),
    /// key, value, key, value, ...
    BuildMap(Vec<V>),
    /// The third operand is NONE for a two-argument slice.
    BuildSlice(V, V, V),
    BuildString(Vec<V>),
    Format { val: V, conv: u8, spec: V },
    ListAppend(V, V),
    SetAdd(V, V),
    MapAdd(V, V, V),
    ListExtend(V, V),
    SetUpdate(V, V),
    DictUpdate(V, V),
    DictMerge(V, V),
    ListToTuple(V),
    /// Unpacks an iterable of exactly n items into a tuple.
    Unpack(V, u32),
    TupleGet(V, u32),
    Call { callee: V, args: Vec<V>, kwnames: u32 },
    CallEx { callee: V, args: V, kwargs: V },
    GetIter(V),
    /// The next item, or null when the iterator is exhausted.
    IterNext(V),
    /// Raises UnboundLocalError if the value is undefined.
    CheckBound(V, u32),
}

#[derive(Clone)]
pub enum Term {
    Jump(B),
    Br { cond: V, then_: B, else_: B },
    BrNull { v: V, null: B, nonnull: B },
    Return(V),
    Raise { exc: V, cause: V },
}

impl Term {
    pub fn for_each_use(&self, mut f: impl FnMut(V)) {
        match self {
            Term::Jump(_) => {}
            Term::Br { cond, .. } => f(*cond),
            Term::BrNull { v, .. } => f(*v),
            Term::Return(v) => f(*v),
            Term::Raise { exc, cause } => {
                f(*exc);
                if *cause != NONE {
                    f(*cause);
                }
            }
        }
    }

    pub fn successors(&self) -> Vec<B> {
        match self {
            Term::Jump(b) => vec![*b],
            Term::Br { then_, else_, .. } => vec![*then_, *else_],
            Term::BrNull { null, nonnull, .. } => vec![*nonnull, *null],
            Term::Return(_) | Term::Raise { .. } => vec![],
        }
    }
}

impl Op {
    pub fn for_each_use(&self, mut f: impl FnMut(V)) {
        let opt = |v: V, f: &mut dyn FnMut(V)| {
            if v != NONE {
                f(v)
            }
        };
        match self {
            Op::Const(_) | Op::LoadGlobal(_) | Op::LoadDeref(_) => {}
            Op::LoadAttr(a, _)
            | Op::StoreGlobal(_, a)
            | Op::Unary(_, a)
            | Op::ToBool(a)
            | Op::IsNone(a, _)
            | Op::ListToTuple(a)
            | Op::Unpack(a, _)
            | Op::TupleGet(a, _)
            | Op::GetIter(a)
            | Op::IterNext(a)
            | Op::CheckBound(a, _) => f(*a),
            Op::Binary(_, a, b)
            | Op::Compare(_, a, b)
            | Op::Is(a, b, _)
            | Op::Contains(a, b, _)
            | Op::GetItem(a, b)
            | Op::DelItem(a, b)
            | Op::ListAppend(a, b)
            | Op::SetAdd(a, b)
            | Op::ListExtend(a, b)
            | Op::SetUpdate(a, b)
            | Op::DictUpdate(a, b)
            | Op::DictMerge(a, b) => {
                f(*a);
                f(*b)
            }
            Op::StoreAttr { obj, val, .. } => {
                f(*obj);
                f(*val)
            }
            Op::SetItem { obj, key, val } => {
                f(*obj);
                f(*key);
                f(*val)
            }
            Op::MapAdd(a, b, c) => {
                f(*a);
                f(*b);
                f(*c)
            }
            Op::BuildSlice(a, b, c) => {
                f(*a);
                f(*b);
                opt(*c, &mut f)
            }
            Op::Format { val, spec, .. } => {
                f(*val);
                opt(*spec, &mut f)
            }
            Op::BuildTuple(vs)
            | Op::BuildList(vs)
            | Op::BuildSet(vs)
            | Op::BuildMap(vs)
            | Op::BuildString(vs) => vs.iter().for_each(|v| f(*v)),
            Op::Call { callee, args, .. } => {
                f(*callee);
                args.iter().for_each(|v| f(*v))
            }
            Op::CallEx { callee, args, kwargs } => {
                f(*callee);
                f(*args);
                opt(*kwargs, &mut f)
            }
        }
    }
}

#[derive(Clone)]
pub struct Inst {
    pub dst: V,
    pub op: Op,
    /// Offset of the bytecode instruction this came from.
    pub bc: u32,
    /// Values whose last use is this instruction; their registers are
    /// released right after it.
    pub kills: Vec<V>,
}

#[derive(Clone)]
pub struct Phi {
    pub dst: V,
    /// One source per predecessor, in the order of `Block::preds`.
    pub srcs: Vec<V>,
}

pub struct Block {
    pub phis: Vec<Phi>,
    pub insts: Vec<Inst>,
    pub term: Term,
    pub preds: Vec<B>,
    pub bc_start: u32,
}

pub struct Function {
    pub name: String,
    pub qualname: String,
    pub filename: String,
    pub firstlineno: i32,
    /// co_consts, followed by co_names, followed by constants the frontend
    /// introduced.
    pub consts: Vec<PyObj>,
    pub blocks: Vec<Block>,
    pub nvalues: u32,
    /// The value of each parameter, in co_varnames order.
    pub params: Vec<V>,
    pub value_names: Vec<Option<String>>,
    /// (start offset, end offset, line)
    pub lines: Vec<(u32, u32, i32)>,

    pub globals: PyObj,
    pub builtins: PyObj,
    pub closure: Option<PyObj>,
    pub defaults: Option<PyObj>,
    pub kwdefaults: Option<PyObj>,
    pub argcount: usize,
    pub posonlyargcount: usize,
    pub kwonlyargcount: usize,
    pub has_varargs: bool,
    pub has_varkw: bool,
    /// Interned names of the parameters, in co_varnames order.
    pub param_names: Vec<PyObj>,
}

impl Function {
    pub fn line_of(&self, bc: u32) -> i32 {
        for (start, end, line) in &self.lines {
            if bc >= *start && bc < *end {
                return *line;
            }
        }
        self.firstlineno
    }

    pub fn value_name(&self, v: V) -> String {
        if v == NONE {
            return "<none>".to_string();
        }
        if v == UNDEF {
            return "undef".to_string();
        }
        match self.value_names.get(v as usize).and_then(|n| n.as_ref()) {
            Some(name) => format!("%{}.{}", name, v),
            None => format!("%{}", v),
        }
    }

    fn const_repr(&self, idx: u32) -> String {
        let mut s = unsafe { pyffi::repr_string(self.consts[idx as usize].as_ptr()) };
        if s.len() > 60 {
            s.truncate(57);
            s.push_str("...");
        }
        s
    }

    fn name_str(&self, idx: u32) -> String {
        unsafe { pyffi::str_to_string(self.consts[idx as usize].as_ptr()) }
    }

    fn values(&self, vs: &[V]) -> String {
        vs.iter().map(|v| self.value_name(*v)).collect::<Vec<_>>().join(", ")
    }

    fn call_args(&self, args: &[V], kwnames: u32) -> String {
        if kwnames == NONE {
            return self.values(args);
        }
        let names = self.consts[kwnames as usize].as_ptr();
        let nkw = unsafe { pyo3::ffi::PyTuple_GET_SIZE(names) } as usize;
        let npos = args.len() - nkw;
        let mut parts: Vec<String> = args[..npos].iter().map(|v| self.value_name(*v)).collect();
        for i in 0..nkw {
            let name = unsafe {
                pyffi::str_to_string(pyo3::ffi::PyTuple_GET_ITEM(names, i as isize))
            };
            parts.push(format!("{}={}", name, self.value_name(args[npos + i])));
        }
        parts.join(", ")
    }

    pub fn op_repr(&self, op: &Op) -> String {
        let n = |v: &V| self.value_name(*v);
        match op {
            Op::Const(c) => format!("const {}", self.const_repr(*c)),
            Op::LoadGlobal(name) => format!("load_global '{}'", self.name_str(*name)),
            Op::LoadDeref(i) => format!("load_deref {}", i),
            Op::LoadAttr(a, name) => format!("load_attr {} '{}'", n(a), self.name_str(*name)),
            Op::StoreAttr { obj, name, val } => {
                format!("store_attr {} '{}' {}", n(obj), self.name_str(*name), n(val))
            }
            Op::StoreGlobal(name, v) => format!("store_global '{}' {}", self.name_str(*name), n(v)),
            Op::Binary(k, a, b) => format!("binary {} {} {}", BINARY_SYMBOLS[*k as usize], n(a), n(b)),
            Op::Unary(k, a) => format!("unary {:?} {}", k, n(a)).to_lowercase(),
            Op::Compare(k, a, b) => format!("compare {} {} {}", CMP_SYMBOLS[*k as usize], n(a), n(b)),
            Op::Is(a, b, inv) => format!("{} {} {}", if *inv { "is_not" } else { "is" }, n(a), n(b)),
            Op::Contains(a, b, inv) => {
                format!("{} {} {}", if *inv { "not_in" } else { "in" }, n(a), n(b))
            }
            Op::ToBool(a) => format!("to_bool {}", n(a)),
            Op::IsNone(a, inv) => format!("{} {}", if *inv { "is_not_none" } else { "is_none" }, n(a)),
            Op::GetItem(a, b) => format!("get_item {} {}", n(a), n(b)),
            Op::SetItem { obj, key, val } => format!("set_item {} {} {}", n(obj), n(key), n(val)),
            Op::DelItem(a, b) => format!("del_item {} {}", n(a), n(b)),
            Op::BuildTuple(vs) => format!("build_tuple ({})", self.values(vs)),
            Op::BuildList(vs) => format!("build_list [{}]", self.values(vs)),
            Op::BuildSet(vs) => format!("build_set {{{}}}", self.values(vs)),
            Op::BuildMap(vs) => format!("build_map {{{}}}", self.values(vs)),
            Op::BuildSlice(a, b, c) => {
                if *c == NONE {
                    format!("build_slice {} {}", n(a), n(b))
                } else {
                    format!("build_slice {} {} {}", n(a), n(b), n(c))
                }
            }
            Op::BuildString(vs) => format!("build_string ({})", self.values(vs)),
            Op::Format { val, conv, spec } => {
                if *spec == NONE {
                    format!("format {} conv={}", n(val), conv)
                } else {
                    format!("format {} conv={} spec={}", n(val), conv, n(spec))
                }
            }
            Op::ListAppend(a, b) => format!("list_append {} {}", n(a), n(b)),
            Op::SetAdd(a, b) => format!("set_add {} {}", n(a), n(b)),
            Op::MapAdd(a, b, c) => format!("map_add {} {} {}", n(a), n(b), n(c)),
            Op::ListExtend(a, b) => format!("list_extend {} {}", n(a), n(b)),
            Op::SetUpdate(a, b) => format!("set_update {} {}", n(a), n(b)),
            Op::DictUpdate(a, b) => format!("dict_update {} {}", n(a), n(b)),
            Op::DictMerge(a, b) => format!("dict_merge {} {}", n(a), n(b)),
            Op::ListToTuple(a) => format!("list_to_tuple {}", n(a)),
            Op::Unpack(a, k) => format!("unpack {} {}", n(a), k),
            Op::TupleGet(a, i) => format!("tuple_get {} {}", n(a), i),
            Op::Call { callee, args, kwnames } => {
                format!("call {}({})", n(callee), self.call_args(args, *kwnames))
            }
            Op::CallEx { callee, args, kwargs } => {
                if *kwargs == NONE {
                    format!("call_ex {}(*{})", n(callee), n(args))
                } else {
                    format!("call_ex {}(*{}, **{})", n(callee), n(args), n(kwargs))
                }
            }
            Op::GetIter(a) => format!("get_iter {}", n(a)),
            Op::IterNext(a) => format!("iter_next {}", n(a)),
            Op::CheckBound(a, _) => format!("check_bound {}", n(a)),
        }
    }

    pub fn term_repr(&self, term: &Term) -> String {
        match term {
            Term::Jump(b) => format!("jump b{}", b),
            Term::Br { cond, then_, else_ } => {
                format!("br {} b{} b{}", self.value_name(*cond), then_, else_)
            }
            Term::BrNull { v, null, nonnull } => {
                format!("br_exhausted {} b{} b{}", self.value_name(*v), null, nonnull)
            }
            Term::Return(v) => format!("return {}", self.value_name(*v)),
            Term::Raise { exc, cause } => {
                if *cause == NONE {
                    format!("raise {}", self.value_name(*exc))
                } else {
                    format!("raise {} from {}", self.value_name(*exc), self.value_name(*cause))
                }
            }
        }
    }

    pub fn dump_code(&self, out: &mut String, insts: &[Inst], term: &Term) {
        for inst in insts {
            if inst.dst != NONE {
                let _ = writeln!(out, "    {} = {}", self.value_name(inst.dst), self.op_repr(&inst.op));
            } else {
                let _ = writeln!(out, "    {}", self.op_repr(&inst.op));
            }
        }
        let _ = writeln!(out, "    {}", self.term_repr(term));
    }

    pub fn dump_block_header(&self, out: &mut String, b: usize) {
        let block = &self.blocks[b];
        let preds: Vec<String> = block.preds.iter().map(|p| format!("b{}", p)).collect();
        let _ = writeln!(
            out,
            "b{}:    ; line {}, preds: [{}]",
            b,
            self.line_of(block.bc_start),
            preds.join(", ")
        );
        for phi in &block.phis {
            let srcs: Vec<String> = phi
                .srcs
                .iter()
                .zip(&block.preds)
                .map(|(v, p)| format!("b{}: {}", p, self.value_name(*v)))
                .collect();
            let _ = writeln!(out, "    {} = phi [{}]", self.value_name(phi.dst), srcs.join(", "));
        }
    }

    pub fn dump(&self) -> String {
        let mut out = String::new();
        let params: Vec<String> = self.params.iter().map(|v| self.value_name(*v)).collect();
        let _ = writeln!(
            out,
            "function {}({})    ; {}:{}",
            self.qualname,
            params.join(", "),
            self.filename,
            self.firstlineno
        );
        for b in 0..self.blocks.len() {
            self.dump_block_header(&mut out, b);
            self.dump_code(&mut out, &self.blocks[b].insts, &self.blocks[b].term);
        }
        out
    }
}
