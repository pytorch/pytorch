//! What C code does to an operand, and when that is taken to have no effect.
//!
//! The rule. (1) What an instruction, or a builtin of the engine's table,
//! does to an operand in C code has no effect only when the operand was seen
//! to be of a kind listed here for that use: reading it (an attribute, an
//! item, its length, its truth, a comparison, arithmetic) or taking its
//! items (iteration, unpacking). (2) Any other operand, and any callee that
//! is in none of the engine's tables, ends the region there, until an entry
//! is added here with its test or a pass list declares the call pure
//! (vm.passes.PureCalls). (3) A callable handed to a builtin is judged as a
//! call of that callable, and nothing is pure because another tracer's table
//! lists it.
//!
//! A callee that ran operators is no exception to (2): the operators say
//! what a call did only where the callee is the dispatcher's own way to them,
//! one of torch's C functions or tensor methods, a tensor class called with
//! data (torch.Tensor, torch.FloatTensor) or the `_op` of an operator
//! object of torch.ops (`runs_operators`). Any other C callee may have done
//! more than run them: a node of the autograd graph that is called binds a
//! leaf's .grad. The autograd engine's own call is the named case
//! (exec/backward.rs, asked first in `pure_instr` for its reason); these
//! tables decide for every callee, and for that one the same.
//!
//! Given: an instruction of a profiled Cfg that entered no frame of Python
//! code (those are judged from their own code), and the kinds its operands
//! were seen to have (exec/types.rs).
//!
//! Guarantees: `instr` and `builtin` are Ok only if every operand is
//!   - a number, a string, None, a tuple, a frozenset, a tensor, or an exact
//!     list, dict or set (read, or iterated with a new iterator);
//!   - an object of a class defined in Python whose C bases are all in the
//!     class table (what its own methods do is in their frames);
//!   - a defaultdict, read as a dict is; subscripted, only by a key that is
//!     the traced call's on every call the region is entered for (a number
//!     or a string the code was given, or an object the reads lead to): the
//!     region's reads then say that the key is there;
//!   - an iterator, where it is only read; where its items are taken, only
//!     if the same frame made it from operands whose items may be taken.
//! Err says which operand is not, for torch.vm.regions.
//!
//! May not assume: that a kind is right for a value that was never computed
//! (it is passed: its instruction did not run); that the items of a list are
//! of shown kinds (taking the items of an item is not judged here).

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use pyo3::ffi;
use pyo3::prelude::*;

use super::types::Kind;
use super::{cell_stores, free_index, opcodes, Free, Op, Own, Prepared, ValueId};

// What is asked of an operand: to be read, or to have its items taken.
#[derive(Clone, Copy, PartialEq)]
enum Use {
    Read,
    Take,
    // An attribute, or the truth: asked of a value of any number of kinds
    // (a loop over the modules of a model has one of every class).
    Ask,
}

// Of a class: reading an object of it, or taking its items (iteration makes
// a new iterator), runs C code without an effect.
pub const READ: u8 = 1;
// An iterator: taking an item advances it; reading it does nothing.
pub const ITER: u8 = 4;
// A callable implemented in C; which one, the class does not say.
pub const CALLS: u8 = 8;
// A dict that puts in a key it is read for and lacks (a defaultdict): its
// other reads are a dict's.
pub const DEFAULTS: u8 = 16;
// An operator object of torch.ops: its `_op` is the dispatcher's call of the
// operator.
pub const OPS: u8 = 32;

struct Tables {
    // C-implemented classes, by address: what is shown of their objects.
    classes: HashMap<usize, u8>,
    // The classes of C methods: a class that has one of its own in its
    // namespace is implemented in C.
    c_members: HashSet<usize>,
    // Builtins that take the items of their arguments, that call an
    // argument, that make an iterator over their arguments, and that ask
    // nothing of their arguments.
    takers: HashSet<usize>,
    callers: HashSet<usize>,
    makers: HashSet<usize>,
    any: HashSet<usize>,
    // torch's own C functions and tensor methods that compute from their
    // arguments: no state setter, nothing in place, no hook.
    torch: HashSet<usize>,
    // What `object` does for every class: reads of their operands.
    readers: HashSet<usize>,
    // Builtins that read an item by its key.
    subscripts: HashSet<usize>,
    // torch's C functions and tensor methods, whatever their names, and its
    // tensor classes.
    operators: HashSet<usize>,
    // `id`, and the hashes: their result is which object the operand is.
    ids: HashSet<usize>,
    hashes: HashSet<usize>,
}

const TABLES: &std::ffi::CStr = c"
import builtins, collections, contextvars, functools, itertools, math, operator, re, types, typing
import torch
R, I, C, D, O = 1, 4, 8, 16, 32
classes = {}
def put(flags, *cs):
    for c in cs:
        classes[c] = classes.get(c, 0) | flags
od = collections.OrderedDict()
put(R, tuple, frozenset, str, bytes, range, list, dict, set, type({}.keys()), type({}.values()), type({}.items()),
    collections.OrderedDict, type(od.keys()), type(od.values()), type(od.items()), types.MappingProxyType, torch.Size,
    int, float, bool, complex, type(None), type(Ellipsis), type(NotImplemented), slice, type, types.FunctionType,
    types.MethodType, types.ModuleType, types.CodeType, types.CellType, types.SimpleNamespace, property, staticmethod,
    classmethod, super, functools.partial, type(len), type(str.join), type(int.__add__), type((1).__add__),
    type(dict.__dict__['fromkeys']), types.GetSetDescriptorType, types.MemberDescriptorType, re.Pattern, re.Match,
    torch.dtype, torch.device, torch.layout, torch.memory_format, torch.qscheme, torch.finfo, torch.iinfo,
    torch._C.TensorBase, torch._C.Generator, torch._C._FunctionBase, type(torch.Tensor),
    type(functools.lru_cache(lambda: 0)), types.GenericAlias, types.UnionType, typing.Generic,
    type(torch._C._VariableFunctions), types.FrameType, contextvars.ContextVar,
    *[c for c in vars(builtins).values() if isinstance(c, type) and issubclass(c, BaseException)])
put(D, collections.defaultdict)
put(R | O, torch._ops.OpOverload, torch._ops.OpOverloadPacket)
put(C, type, type(len), type(str.join), type(int.__add__), type((1).__add__), type(dict.__dict__['fromkeys']), functools.partial,
    type(torch.Tensor), type(functools.lru_cache(lambda: 0)))
iterators = [iter([]), iter(()), iter({}), iter({}.values()), iter({}.items()), iter(set()), iter(''), iter(b''),
             iter(range(0)), reversed([]), reversed({}), iter(od), iter(od.values()), iter(od.items()),
             iter(lambda: 0, 0), (x for x in ()), iter(torch.Size([]))]
put(I, zip, map, filter, enumerate, reversed, *[type(i) for i in iterators],
    *[c for c in vars(itertools).values() if isinstance(c, type)])
c_members = [type(str.join), type(int.__add__), type(dict.__dict__['fromkeys'])]
lazy = [zip, map, filter, enumerate, reversed, iter, *[c for c in vars(itertools).values() if isinstance(c, type)],
        vars(itertools.chain)['from_iterable']]
takers = lazy + [tuple, list, dict, set, frozenset, min, max, sum, sorted, any, all, str.join, bytes.join,
                 math.prod, math.fsum, vars(dict)['fromkeys'], set.union, set.intersection, set.difference,
                 set.symmetric_difference, set.issubset, set.issuperset, set.isdisjoint, torch.Size,
                 operator.contains, list.__contains__, tuple.__contains__]
callers = [map, filter, sorted, min, max, iter, itertools.starmap, itertools.accumulate, itertools.takewhile,
           itertools.dropwhile, itertools.filterfalse, itertools.groupby, functools.reduce]
any_ = [type, isinstance, issubclass, callable, functools.partial, torch.is_tensor]
# Tells a telemetry hook, once per process and key, that an API was used (torchvision's operators begin with
# it). A region exists only after its code ran as written, so the report is made; no program reads it.
any_ += [torch._C._log_api_usage_once]
# Builtins that read an item by its key: of a defaultdict, that may put the key in.
subscripts = [dict.__getitem__, operator.getitem]
# What `object` itself does for every class that does not say otherwise: reads, and making an object.
readers = [object.__getattribute__, object.__eq__, object.__ne__, object.__repr__, object.__str__,
           object.__format__, object.__init__, object.__new__, object.__init_subclass__, object.__dir__]
# No effect, but the result is which object the operand is (or, for a hash of a number or a string, its value).
ids = [id]
hashes = [hash, object.__hash__]
# torch's C functions and tensor methods, but for what sets state or hands something out: an in-place operation
# (a trailing underscore), flags and hooks of autograd, values given to Python, and what a tensor says of itself
# that a region does not check (`unchecked`: the one list of those, UNCHECKED_OF_A_TENSOR of exec/reads.rs).
state = {'requires_grad_', 'retain_grad', 'register_hook', 'register_post_accumulate_grad_hook', 'backward', 'set_',
         'resize_', 'resize_as_', 'share_memory_', 'tolist', 'numpy', 'item', 'seed', 'manual_seed', 'record_stream'}
def computes(name):
    dunder = name.startswith('__') and name.endswith('__')
    inplace = name.endswith('_') and not dunder
    listed = name in state or name in unchecked
    return not inplace and not listed and not name.startswith(('_set', 'set_', '__set', '__del', '__i'))
spaces = (torch._C._VariableFunctions, torch._C._nn, torch._C._linalg, torch._C._fft, torch._C._special, torch._C._sparse)
def named(ok):
    out = [getattr(s, n) for s in spaces for n in dir(s) if ok(n) and callable(getattr(s, n, None))]
    return out + [f for n, f in vars(torch._C.TensorBase).items() if callable(f) and ok(n) and n not in ('__new__', '__init__')]
torch_ = named(computes)
# All of them, the in-place ones too: what a call of one that ran operators did is those operators. And the
# classes whose call makes a tensor of what it is handed, as torch.tensor does (torch.Tensor, torch.FloatTensor).
operators = named(lambda name: True) + [torch.Tensor, *torch._tensor_classes]
";

static T: OnceLock<Tables> = OnceLock::new();

// Makes the tables. Runs Python code: called where a function goes onto the
// executor, not from inside it.
pub fn ready() {
    tables();
}

fn tables() -> &'static Tables {
    T.get_or_init(|| {
        Python::attach(|py| {
            let g = pyo3::types::PyDict::new(py);
            let ran = g.set_item("unchecked", super::reads::UNCHECKED_OF_A_TENSOR.to_vec()).and_then(|_| py.run(TABLES, Some(&g), None));
            if let Err(e) = &ran {
                e.print(py);
            }
            let set = |name: &str| -> HashSet<usize> {
                let items = g.get_item(name).ok().flatten().and_then(|x| x.try_iter().ok());
                items.map(|it| it.flatten().map(|x| x.as_ptr() as usize).collect()).unwrap_or_default()
            };
            let mut classes = HashMap::new();
            let items = g.get_item("classes").ok().flatten().and_then(|d| d.call_method0("items").ok()).and_then(|i| i.try_iter().ok());
            for item in items.into_iter().flatten().flatten() {
                if let (Ok(k), Ok(v)) = (item.get_item(0), item.get_item(1)) {
                    classes.insert(k.as_ptr() as usize, v.extract::<u8>().unwrap_or(0));
                }
            }
            // The tables refer to their objects by address: kept for good.
            std::mem::forget(g.clone().unbind());
            Tables { classes, c_members: set("c_members"), takers: set("takers"), callers: set("callers"), makers: set("lazy"), any: set("any_"), torch: set("torch_"), readers: set("readers"), ids: set("ids"), hashes: set("hashes"), subscripts: set("subscripts"), operators: set("operators") }
        })
    })
}

// Whether `f` is one of torch's own C functions that only compute.
pub fn torch_computes(f: &Bound<'_, PyAny>) -> bool {
    tables().torch.contains(&(f.as_ptr() as usize))
}

// Whether `f` is a builtin of the engine's tables: what it is handed is
// judged where the call is (see `builtin`).
pub fn is_builtin(f: &Bound<'_, PyAny>) -> bool {
    let (t, at) = (tables(), f.as_ptr() as usize);
    super::pure_builtins().contains(&at) || t.readers.contains(&at) || t.any.contains(&at) || t.ids.contains(&at) || t.hashes.contains(&at)
}

// Whether call `v` of the C callable `f`, which ran operators, did what
// those operators say and no more: `f` is one of torch's own C functions,
// tensor methods or tensor classes, or the `_op` of an operator object of
// torch.ops.
pub fn runs_operators(prep: &Prepared, v: ValueId, f: &Bound<'_, PyAny>) -> bool {
    if tables().operators.contains(&(f.as_ptr() as usize)) {
        return true;
    }
    let cfg = &prep.cfg;
    let operator = |x: ValueId| kinds(prep, x).is_some_and(|ks| ks.iter().all(|k| matches!(k, Kind::Object(c) if c.shown() & OPS != 0)));
    cfg[v].args.iter().take(2).any(|a| match cfg[*a].op {
        Op::LoadAttr(n) | Op::LoadMethod(n) => cfg.names[n as usize] == "_op" && operator(cfg[*a].args[0]),
        _ => false,
    })
}

// Whether `f` is in one of the engine's tables of callables that only
// compute: of torch's, or of the builtins.
pub fn computes(f: &Bound<'_, PyAny>) -> bool {
    torch_computes(f) || is_builtin(f)
}

// Whether `f` is a builtin that takes the items of what it is handed, makes
// an iterator over it, or calls it.
pub fn takes(f: *mut ffi::PyObject) -> bool {
    let (t, at) = (tables(), f as usize);
    t.takers.contains(&at) || t.callers.contains(&at)
}

// What is shown of the objects of class `tp`: for types.rs, when it first
// sees an object of the class. No Python code runs.
pub unsafe fn class_flags(tp: *mut ffi::PyTypeObject) -> u8 {
    let Some(t) = T.get() else { return 0 };
    let mro = (*tp).tp_mro;
    if mro.is_null() {
        return 0;
    }
    // Defined in Python over `object`: all it does is in its frames.
    let (mut flags, mut more) = (READ, 0);
    for i in 0..ffi::PyTuple_GET_SIZE(mro) {
        let base = ffi::PyTuple_GET_ITEM(mro, i);
        if base == std::ptr::addr_of_mut!(ffi::PyBaseObject_Type) as *mut ffi::PyObject {
            continue;
        }
        match t.classes.get(&(base as usize)) {
            Some(f) => {
                flags &= f;
                more |= f & (ITER | CALLS | DEFAULTS | OPS);
            }
            None if python_defined(base as *mut ffi::PyTypeObject, t) => {}
            None => return 0,
        }
    }
    flags | more
}

// A class made by a class statement: the C methods in its own namespace
// are those of `object` or of a class of the table (`__hash__ =
// object.__hash__`, an enum's `__format__ = str.__format__`).
unsafe fn python_defined(tp: *mut ffi::PyTypeObject, t: &Tables) -> bool {
    if (*tp).tp_flags & ffi::Py_TPFLAGS_HEAPTYPE == 0 || (*tp).tp_dict.is_null() {
        return false;
    }
    let (mut pos, mut key, mut value) = (0, std::ptr::null_mut(), std::ptr::null_mut());
    while ffi::PyDict_Next((*tp).tp_dict, &mut pos, &mut key, &mut value) != 0 {
        if !t.c_members.contains(&(ffi::Py_TYPE(value) as usize)) {
            continue;
        }
        let of = ffi::PyObject_GetAttrString(value, c"__objclass__".as_ptr());
        if of.is_null() {
            ffi::PyErr_Clear();
            return false;
        }
        ffi::Py_DECREF(of);
        let object = std::ptr::addr_of_mut!(ffi::PyBaseObject_Type) as *mut ffi::PyObject;
        if of != object && t.classes.get(&(of as usize)).is_none_or(|f| f & READ == 0) {
            return false;
        }
    }
    true
}

// Whether what is done to operand `x` is shown to have no effect; Err names
// its kind.
fn operand(py: Python<'_>, prep: &Prepared, x: ValueId, need: Use, own: &Own) -> Result<(), String> {
    let types = unsafe { &*prep.types.get() };
    let Some(ty) = types.get(x.idx()).and_then(|t| t.ty.as_ref()) else { return Ok(()) };
    let Some(kinds) = ty.kinds() else {
        // Too many kinds were seen to keep them: only what every object
        // answers without an effect may be asked.
        return if need == Use::Ask { Ok(()) } else { Err("a value of too many kinds".into()) };
    };
    for k in kinds {
        let Kind::Object(c) = k else { continue };
        let shown = c.shown();
        let iterator = shown & ITER != 0;
        if shown & (READ | DEFAULTS) != 0 || (iterator && (need != Use::Take || made_here(py, prep, x, own, &mut HashSet::new()))) {
            continue;
        }
        let what = if iterator { "an iterator this code did not make: taking its items advances it" } else { "not shown to be only read" };
        return Err(format!("a {} ({what})", c.name()));
    }
    Ok(())
}

// Whether a call of a builtin that makes an iterator over its arguments
// made one whose items may be taken: every argument's may, and none is a
// callable implemented in C.
fn made_of_shown(py: Python<'_>, prep: &Prepared, call: ValueId, own: &Own) -> bool {
    let types = unsafe { &*prep.types.get() };
    handed(prep, call).iter().all(|a| {
        let kinds = types.get(a.idx()).and_then(|t| t.ty.as_ref()).and_then(|ty| ty.kinds());
        operand(py, prep, *a, Use::Take, own).is_ok() && !kinds.is_some_and(|ks| ks.iter().any(|k| matches!(k, Kind::Object(c) if c.shown() & CALLS != 0)))
    })
}

// What is asked of how a value came about.
#[derive(Clone, Copy, PartialEq)]
enum Way {
    // An iterator its frame made from operands whose items may be taken:
    // GetIter, a call of a builtin that makes one, a generator.
    Made,
    // An object the region's reads lead to: read from a namespace, an
    // attribute or a cell, or an item of a container that was (the reads
    // have a container's items with it).
    Read,
}

// Whether value x came about that way on every path: directly, from a local
// slot (not an argument's) every store to which did, from a cell likewise,
// or as a block's stack param every edge into which passes one that did (as
// fresh_container, for containers).
fn every_way(py: Python<'_>, prep: &Prepared, x: ValueId, way: Way, own: &Own, seen: &mut HashSet<ValueId>) -> bool {
    let cfg = &prep.cfg;
    if !seen.insert(x) {
        return true;
    }
    match (way, &cfg[x].op) {
        (Way::Made, Op::GetIter) => operand(py, prep, cfg[x].args[0], Use::Take, own).is_ok(),
        // (A call with a star hands over a tuple, whose items are not judged.)
        (Way::Made, Op::Call { .. } | Op::Bytecode { opcode: opcodes::CALL_FUNCTION_EX, .. }) => {
            // A call that entered a Python function's frame and gave an
            // iterator made a generator: its frames are judged where they
            // are resumed.
            let effects = unsafe { &*prep.effects.get() };
            let entered = effects.get(super::Site::Instr(x)).iter().any(|(seq, _)| {
                seq.iter().any(|e| matches!(e, super::effects::Effect::Frame { how: super::effects::How::Direct, .. }))
            });
            let callees = unsafe { &*prep.callees.get() };
            let maker = callees.get(&x).and_then(|c| c.get(py)).is_some_and(|f| tables().makers.contains(&(f.as_ptr() as usize)));
            entered || (maker && made_of_shown(py, prep, x, own))
        }
        // A generator expression's first argument is the iterator its
        // caller made for it (see genexpr_over_made).
        (Way::Made, Op::LoadLocal { slot: 0 } | Op::LoadLocalChecked { slot: 0 }) if own.fresh_self => true,
        (Way::Read, Op::LoadGlobal(_) | Op::LoadGlobalIn { .. } | Op::LoadAttr(_) | Op::Const(_) | Op::Lit(_) | Op::Pinned(_)) => true,
        (Way::Read, Op::GetIter | Op::Bytecode { opcode: opcodes::BINARY_SUBSCR, .. }) => every_way(py, prep, cfg[x].args[0], way, own, seen),
        // The item FOR_ITER took: of what its iterator is over.
        (Way::Read, Op::Project(0)) if matches!(cfg[cfg[x].args[0]].op, Op::IterNext) => every_way(py, prep, cfg[cfg[x].args[0]].args[0], way, own, seen),
        (_, Op::LoadLocal { slot } | Op::LoadLocalChecked { slot }) => {
            // Stores of values never computed are on paths never taken.
            let types = unsafe { &*prep.types.get() };
            let stored = |i: &&crate::cfg::ir::Instr| matches!(i.op, Op::StoreLocal { slot: s } if s == *slot && types.get(i.args[0].idx()).is_some_and(|t| t.ty.is_some()));
            let mut stores = cfg.instrs.iter().filter(stored).peekable();
            (*slot as usize) >= own.nargs && stores.peek().is_some() && stores.all(|i| every_way(py, prep, i.args[0], way, own, seen))
        }
        (_, Op::LoadDeref { slot }) => match free_index(cfg, *slot) {
            // A cell of the function's closure: its content is one of the reads.
            Some(i) => way == Way::Read || own.free.get(i) == Some(&Free::Fresh),
            None => cell_stores(cfg, *slot, own).is_some_and(|xs| !xs.is_empty() && xs.iter().all(|x| every_way(py, prep, *x, way, own, seen))),
        },
        (_, Op::Param(crate::cfg::ir::ParamKind::Stack(_))) => {
            let Some((b, pos)) = cfg.block_ids().find_map(|b| {
                let explicit = cfg[b].params.iter().filter(|p| !matches!(cfg[**p].op, Op::Param(k) if k.is_implicit()));
                explicit.clone().position(|p| *p == x).map(|i| (b, i))
            }) else {
                return false;
            };
            if cfg.block_ids().any(|p| cfg[p].handlers.contains(&b)) {
                return false;
            }
            let mut edges = 0;
            for p in cfg.block_ids() {
                for exit in cfg[p].term.exits() {
                    let crate::cfg::ir::Exit::Block(e) = exit else { continue };
                    if e.target != b {
                        continue;
                    }
                    for ver in &cfg[p].versions {
                        edges += 1;
                        if !every_way(py, prep, ver.outputs[e.args[pos] as usize], way, own, seen) {
                            return false;
                        }
                    }
                }
            }
            edges > 0
        }
        _ => false,
    }
}

fn made_here(py: Python<'_>, prep: &Prepared, x: ValueId, own: &Own, seen: &mut HashSet<ValueId>) -> bool {
    every_way(py, prep, x, Way::Made, own, seen)
}

// Whether call `x` of `func` makes a generator expression over an iterator
// its frame made: the generator's frames then take the items of what is
// their own (its first argument, as an __init__'s self is its own).
pub fn genexpr_over_made(py: Python<'_>, prep: &Prepared, x: ValueId, func: &str, own: &Own) -> bool {
    func.ends_with("<genexpr>") && handed(prep, x).first().is_some_and(|a| made_here(py, prep, *a, own, &mut HashSet::new()))
}

// Whether the key of the subscript or membership test `v` is an object the
// region's reads lead to: the hash it is looked up by (a tensor's is which
// tensor it is) is then that object's on every call.
pub fn key_read(py: Python<'_>, prep: &Prepared, v: ValueId, own: &Own) -> bool {
    let ins = &prep.cfg[v];
    let key = match ins.op {
        Op::Bytecode { opcode: opcodes::BINARY_SUBSCR, .. } => ins.args.get(1),
        Op::Bytecode { opcode: opcodes::CONTAINS_OP, .. } => ins.args.first(),
        _ => None,
    };
    key.is_some_and(|k| every_way(py, prep, *k, Way::Read, own, &mut HashSet::new()))
}

// The kinds value x was seen to have (None: never computed, or too many).
fn kinds(prep: &Prepared, x: ValueId) -> Option<&[Kind]> {
    let types = unsafe { &*prep.types.get() };
    types.get(x.idx()).and_then(|t| t.ty.as_ref()).and_then(|ty| ty.kinds())
}

// Whether x was seen to be an object of a class with this flag.
fn has(prep: &Prepared, x: ValueId, flag: u8) -> bool {
    kinds(prep, x).is_some_and(|ks| ks.iter().any(|k| matches!(k, Kind::Object(c) if c.shown() & flag != 0)))
}

// What a call hands its callee: CALL's operands are (NULL, callable, args)
// or (method, self, args), and which it was is seen in the first one's kind.
fn handed(prep: &Prepared, call: ValueId) -> &[ValueId] {
    let args = &prep.cfg[call].args;
    let types = unsafe { &*prep.types.get() };
    let null = |a: &ValueId| matches!(types.get(a.idx()).and_then(|t| t.ty.as_ref()).and_then(|ty| ty.kinds()), Some([Kind::Null]));
    let from = if args.first().is_some_and(|a| *a == crate::cfg::ir::NULL || null(a)) { 2 } else { 1 };
    args.get(from..).unwrap_or(&[])
}

// The name an attribute or method load asks a tensor for, if it is one that
// a region neither checks nor computes (exec/reads.rs): a trace that makes
// the load is refused, so the load is made where the code makes it, behind
// the region of what is in front of it.
pub fn asks_unchecked(prep: &Prepared, v: ValueId) -> Option<&str> {
    let cfg = &prep.cfg;
    let (Op::LoadAttr(n) | Op::LoadMethod(n)) = cfg[v].op else { return None };
    let name = cfg.names[n as usize].as_str();
    let tensor = kinds(prep, cfg[v].args[0]).is_some_and(|ks| ks.iter().any(|k| matches!(k, Kind::Tensor(_))));
    (tensor && super::reads::UNCHECKED_OF_A_TENSOR.contains(&name)).then_some(name)
}

// An instruction that is no call and entered no frame: what C code of its
// operands' types did.
pub fn instr(py: Python<'_>, prep: &Prepared, v: ValueId, own: &Own) -> Result<(), String> {
    let ins = &prep.cfg[v];
    super::rules::advice(&ins.op)?;
    let reads = |args: &[ValueId]| args.iter().try_for_each(|a| operand(py, prep, *a, Use::Read, own));
    let takes = |args: &[ValueId]| args.iter().try_for_each(|a| operand(py, prep, *a, Use::Take, own));
    match ins.op {
        // d[key] puts the key in if d lacks it. It does not on any call a
        // region is entered for when the key is the traced call's on every
        // such call: a number or a string the code was given (a constant, an
        // argument, which the region is found by) or an object the reads
        // lead to. The subscript is among the region's reads (the item, or
        // the dict's whole content), taken after it ran: a call on which the
        // key is not there is no call of the region, and runs as written. A
        // key the code computes may be one the traced run put in itself,
        // from a read that is dropped because of it (d[len(d)]).
        Op::Bytecode { opcode: opcodes::BINARY_SUBSCR, .. } if has(prep, ins.args[0], DEFAULTS) => {
            let key = ins.args[1];
            let value = kinds(prep, key).is_some_and(|ks| ks.iter().all(|k| matches!(k, Kind::None | Kind::Bool(_) | Kind::Int(_) | Kind::Float | Kind::Str(_))));
            let given = match prep.cfg[key].op {
                Op::Const(_) | Op::Lit(_) | Op::Pinned(_) => true,
                Op::LoadLocal { slot } | Op::LoadLocalChecked { slot } => (slot as usize) < own.nargs,
                _ => false,
            };
            if (value && given) || every_way(py, prep, key, Way::Read, own, &mut HashSet::new()) {
                Ok(())
            } else {
                Err("a defaultdict and a key the region is not found by: a key it lacks is put in".into())
            }
        }
        Op::LoadAttr(_) | Op::LoadMethod(_) | Op::ToBool => match asks_unchecked(prep, v) {
            Some(name) => Err(format!("a tensor asked for its .{name}, which no region checks")),
            None => ins.args.iter().try_for_each(|a| operand(py, prep, *a, Use::Ask, own)),
        },
        Op::Binary(_) | Op::Compare(_) => reads(&ins.args),
        Op::GetIter | Op::IterNext => takes(&ins.args),
        Op::Bytecode { opcode, .. } => match opcode {
            // `x in it` takes items of an iterator; the others take those of
            // their source (the container they add to is the frame's own).
            opcodes::UNPACK_SEQUENCE | opcodes::UNPACK_EX | opcodes::CONTAINS_OP | opcodes::LIST_EXTEND | opcodes::SET_UPDATE | opcodes::DICT_UPDATE | opcodes::DICT_MERGE => {
                takes(&ins.args)
            }
            _ => reads(&ins.args),
        },
        _ => Ok(()),
    }
}

// `id(o)` and a hash have no effect, but what they give is which object `o`
// is: the traced call's, unless the region is found by that object. It is
// when the object was read from a namespace, an attribute or a cell (the
// region's reads lead to it); a tensor, a list, a dict or a set is known to
// a region by what it is like, an argument may be, and what the code made
// itself is another object on every call. A hash of None, a bool, a number
// or a string is of its value (`by_value`).
fn by_identity(prep: &Prepared, v: ValueId, by_value: bool) -> Result<(), String> {
    let types = unsafe { &*prep.types.get() };
    for a in handed(prep, v) {
        let Some(kinds) = types.get(a.idx()).and_then(|t| t.ty.as_ref()).and_then(|ty| ty.kinds()) else { continue };
        let read = matches!(prep.cfg[*a].op, Op::LoadGlobal(_) | Op::LoadGlobalIn { .. } | Op::LoadAttr(_) | Op::LoadDeref { .. } | Op::Const(_) | Op::Lit(_) | Op::Pinned(_));
        let found = kinds.iter().all(|k| match k {
            Kind::None | Kind::Bool(_) => true,
            Kind::Int(_) | Kind::Float | Kind::Str(_) => by_value,
            Kind::Object(_) => read,
            _ => false,
        });
        if !found {
            return Err("what it gives is which object its operand is, and the region is not found by that object".into());
        }
    }
    Ok(())
}

// A call of `f`, a builtin of the engine's table, that entered no frame.
pub fn builtin(py: Python<'_>, prep: &Prepared, v: ValueId, f: &Bound<'_, PyAny>, partial: bool, own: &Own) -> Result<(), String> {
    let (t, at) = (tables(), f.as_ptr() as usize);
    if t.any.contains(&at) {
        return Ok(());
    }
    if t.ids.contains(&at) || t.hashes.contains(&at) {
        return by_identity(prep, v, t.hashes.contains(&at));
    }
    let (takes, calls) = (t.takers.contains(&at), t.callers.contains(&at));
    if partial && (takes || calls) {
        return Err("it is handed arguments a partial holds".into());
    }
    let types = unsafe { &*prep.types.get() };
    for a in handed(prep, v) {
        // Its other reads (its values, its keys, a length, get) are a dict's.
        if t.subscripts.contains(&at) && has(prep, *a, DEFAULTS) {
            return Err("it reads a key of a defaultdict, which puts in a key it lacks".into());
        }
        operand(py, prep, *a, if takes { Use::Take } else { Use::Read }, own)?;
        let kinds = types.get(a.idx()).and_then(|t| t.ty.as_ref()).and_then(|ty| ty.kinds());
        if calls && kinds.is_some_and(|ks| ks.iter().any(|k| matches!(k, Kind::Object(c) if c.shown() & CALLS != 0))) {
            return Err("it is handed a callable implemented in C, which is judged as a call of it".into());
        }
    }
    Ok(())
}
