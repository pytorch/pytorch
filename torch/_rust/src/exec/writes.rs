// The writes of a region's traced run to what outlives it, and how a region
// makes them again.
//
// A region is the tensor ops of one traced run. A write that run makes to an
// object that is still there when it ends (`obj.attr = v`, `d[k] = v`,
// `lst[i] = v`, `lst.append(v)`, `s.add(v)`, and the deletions) is part of
// what the code does. This module records each as a call of C code,
// `how(target, *values)`, and a run of the region makes the calls again, in
// the trace's order, after its graph ran.
//
// What it is given
//   - by the optimizer (mod.rs, spans.rs): the question whether an
//     instruction is a write a region takes with it (taken, taken_call,
//     taken_change), which the pass list answers (the Effects pass for all,
//     a cfg pass for one instruction: _cfg_write_site); and whether a call
//     is one of the two questions about a class that a module's store asks
//     (type_question). With no write taken by the pass list every answer is
//     no, but for a subscript or slice store into a tensor: that is torch
//     ops on the tensor, as an in-place method is, and no write to record;
//   - by the executor (run.rs), while a region's body is traced: each store
//     or deletion of an attribute or an item and each call of a builtin that
//     writes, right before it is made (store_attr, store_item, store_slice,
//     write_call), and whether it then succeeded (done);
//   - by reads.rs: what the trace has read so far (to tell the objects that
//     died with the run), and the place where it notes that an item was
//     written (what the run reads of it afterwards is its own) and what
//     each write that is recorded left there (wrote.rs: a read that is no
//     longer as it was read is explained by that alone);
//   - by the region tracer (torch/vm/_writes.py): for each write, how its
//     target and values are made on a later run (a RegionOut::Call each).
//
// What it guarantees
//   - Every write the traced run made in one of the ways above, to an object
//     that outlives the run, is in _region_writes()'s list, in the order
//     made. A write made by Python code of the object's class (a
//     __setattr__, a property, a __setitem__) is none itself: that code runs
//     under the executor, and the stores it makes are the writes.
//   - A write that cannot be made again (a C setter, a C container other
//     than a dict or a list) is not made at all: its token is REFUSED and
//     the trace is refused as any other is (region.rs's _region_refuse,
//     through torch._rust_inductor._refuse): the executor raises the refusal
//     in the write's place, and the trace is given up even if the body's
//     code caught it. No write this module could not record is ever made
//     with the tracer's stand-ins in it.
//   - ONE answer to "is this call a write a region takes": writer(). What
//     it names, write_call records or refuses, case by case; there is no
//     third outcome. An answer of "none" (token 0) is given only where it
//     is said why the call writes nothing a region has to make again: the
//     target is the trace's own, code of the target's class runs in its
//     place, or C code is certain to raise.
//   - Before it records a store, a deletion or a membership change it asks
//     the dict or the set about the key (an undo puts back what was there)
//     only where that runs no code of the program (keys.rs); under any
//     other key the trace is refused, but for a dict or a set the run
//     made itself, whose write is then not recorded: it needs no undo and
//     no region makes it again. (The read of a defaultdict, which records
//     no write, asks as the reads do: load_item.)
//   - take_back() takes back what a trace that is given up wrote, the last
//     write first, before the trace's gate, for the executor to run the code
//     again as it is; a write that cannot be taken back is an error raised
//     in the program (undo), never a run over what is half taken back.
//   - No record of a write is dropped between the body's last instruction
//     and the trace's gate: what a store replaced, a write nothing can see
//     and a write that was taken back are parked (parked.rs) and let go of
//     at the gate, behind the tracer's modes. An object the program gave a
//     finalizer therefore never dies of this module while the tracer's
//     dispatch modes are on.
//   - replay() builds what every write takes before it makes the first: a
//     region that raises before then has written nothing.
//
// What it may not assume
//   - that a write it was told of happens (it may raise: done(token, false));
//   - that the target is the same object on a later call: the tracer says
//     how the target is found, and the reads decide whether it is the one;
//   - that a value can be made again: the tracer refuses the trace if not;
//   - that everything that runs under the trace is the body: the trace's
//     own records change too (reads::owned), and what a call the optimizer
//     did not look into writes (covered::Trusted: one the pass list says
//     only computes, a torch function another tracer keeps whole) is taken
//     to be its own, as before there were writes to record: neither
//     recorded nor refused.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use pyo3::ffi;
use pyo3::ffi::PyObject;
use pyo3::prelude::*;

use super::ffi::*;
use super::reads;
use super::covered;
use super::keys;
use super::made;
use super::parked;
use super::wrote::How;
use super::{Prepared, types};
use crate::cfg::ir::{Instr, Op, RegionOut, ValueId};
use crate::cfg::opcodes;

// ---- Which writes a region takes --------------------------------------

// Whether regions take the writes of their traced run with them (the Effects
// pass), instead of ending in front of each.
static ALL: AtomicBool = AtomicBool::new(false);

// What a cfg pass said of one instruction, by (code object address, code
// unit): true takes the write into the region whatever ALL says, false
// leaves it as a boundary.
static SITES: Mutex<Vec<((usize, u32), bool)>> = Mutex::new(Vec::new());
static SITES_ANY: AtomicBool = AtomicBool::new(false);

/// Let regions hold the writes of their code to objects that outlive them.
#[pyfunction]
pub fn _set_cfg_effects(on: bool) {
    ALL.store(on, Ordering::Relaxed);
}

/// What a cfg pass says of the write at code unit `origin` of code object
/// `code` (the `code` and an instr's `origin` in a cfg view): `take` it into
/// the region (True), leave it as a boundary (False), or nothing (None: as
/// the pass list says for all writes).
#[pyfunction]
#[pyo3(signature = (code, origin, take=None))]
pub fn _cfg_write_site(code: usize, origin: u32, take: Option<bool>) {
    let mut s = SITES.lock().unwrap();
    s.retain(|(site, _)| *site != (code, origin));
    if let Some(take) = take {
        s.push(((code, origin), take));
    }
    SITES_ANY.store(!s.is_empty(), Ordering::Relaxed);
}

// Whether a region takes the write at code unit `origin` of `code`, if it is
// one that can be recorded.
pub(super) fn wanted(code: usize, origin: u32) -> bool {
    if SITES_ANY.load(Ordering::Relaxed)
        && let Some((_, take)) = SITES.lock().unwrap().iter().find(|(site, _)| *site == (code, origin))
    {
        return *take;
    }
    ALL.load(Ordering::Relaxed)
}

// Whether the pass list takes some write into a region.
fn taking() -> bool {
    ALL.load(Ordering::Relaxed) || (SITES_ANY.load(Ordering::Relaxed) && SITES.lock().unwrap().iter().any(|(_, take)| *take))
}

// The traced run is about to change `container`. Where no write is taken, a
// region's run changes only containers it made itself (the optimizer lets
// no other change into a region): what it then reads of one is its own, as
// it was before there were writes to record. Where writes are taken the
// container may be one that was there before: reads.rs is told (see its
// Group::late).
fn changes(container: *mut PyObject) {
    if taking() {
        reads::changed(container);
    }
}

// Whether `f` is type.__instancecheck__ or type.__subclasscheck__, asked
// where the pass list takes some write. A store through nn.Module's
// __setattr__ ends in them (`isinstance(value, Parameter)`, whose metaclass
// defines the question in Python and passes it on): C code that reads the
// class of its argument and nothing else. Without a taken write they stay
// what they are to the optimizer: C code it has no rule for.
pub(super) fn type_question(f: *mut PyObject) -> bool {
    taking() && unsafe {
        ffi::Py_TYPE(f) == &raw mut ffi::PyMethodDescr_Type
            && (*(f as *mut ffi::PyDescrObject)).d_type == &raw mut ffi::PyType_Type
            && matches!(reads::text(tr_descr_name(f)).as_str(), "__instancecheck__" | "__subclasscheck__")
    }
}

// Whether a cfg pass left the write at code unit `origin` of `code` as a
// boundary: the instruction there sits in no region.
pub(super) fn left(code: usize, origin: u32) -> bool {
    SITES_ANY.load(Ordering::Relaxed) && SITES.lock().unwrap().iter().any(|(site, take)| *site == (code, origin) && !*take)
}

// The stores and deletions that are recorded, by instruction.
fn recordable(op: &Op) -> bool {
    matches!(op, Op::StoreAttr(_)) || matches!(op, Op::Bytecode { opcode: opcodes::STORE_SUBSCR | opcodes::DELETE_SUBSCR | opcodes::DELETE_ATTR, .. })
}

// Whether instr `ins` of `code` (with what `prep` observed), a store or a
// deletion, is one a region takes with it: where wanted, a store or deletion
// of an attribute of an object, or of an item of a dict, a list or an
// object; and whatever the pass list says, a subscript or slice store into
// what was a tensor on every observed call, which is torch ops on that
// tensor (nothing is recorded for it: see store_item and store_slice, which
// also say what the trace does if it finds something else there).
pub(super) fn taken(prep: &Prepared, code: usize, ins: &Instr) -> bool {
    use types::{Kind, Ty};
    let types = unsafe { &*prep.types.get() };
    let all = |a: Option<&ValueId>, ok: fn(&Kind) -> bool| {
        a.and_then(|a| types.get(a.idx())).and_then(|t| t.ty.as_ref()).and_then(Ty::kinds).is_some_and(|ks| !ks.is_empty() && ks.iter().all(ok))
    };
    let wanted = || wanted(code, ins.origin);
    match ins.op {
        Op::StoreAttr(_) => wanted() && all(ins.args.get(1), |k| matches!(k, Kind::Object(_))),
        Op::Bytecode { opcode: opcodes::DELETE_ATTR, .. } => wanted() && all(ins.args.first(), |k| matches!(k, Kind::Object(_))),
        Op::Bytecode { opcode: opcodes::STORE_SUBSCR, .. } => {
            all(ins.args.get(1), |k| matches!(k, Kind::Tensor(_))) || (wanted() && all(ins.args.get(1), |k| matches!(k, Kind::Tensor(_) | Kind::Dict(_) | Kind::List(_) | Kind::Object(_))))
        }
        Op::Bytecode { opcode: opcodes::STORE_SLICE, .. } => all(ins.args.get(1), |k| matches!(k, Kind::Tensor(_))),
        Op::Bytecode { opcode: opcodes::DELETE_SUBSCR, .. } => wanted() && all(ins.args.first(), |k| matches!(k, Kind::Dict(_) | Kind::List(_) | Kind::Object(_))),
        _ => false,
    }
}

// Whether the store or deletion `ins` of `code` is recorded when a region is
// traced, so that it needn't keep its block from being one.
pub(super) fn taken_op(code: usize, ins: &Instr) -> bool {
    recordable(&ins.op) && wanted(code, ins.origin)
}

// Whether the call at code unit `origin` of `code`, of the builtin `f`, is a
// write a region takes with it (see write_call): object.__setattr__ and
// __delattr__, list.append, set.add and discard, and the item stores and
// deletions of dicts and lists.
pub(super) fn taken_call(code: usize, origin: u32, f: &Bound<'_, PyAny>) -> bool {
    wanted(code, origin) && unsafe { writer(f.as_ptr()) }.is_some()
}

// As taken and taken_call, for the change instr v makes to a container (see
// spans.rs's changed_container): the container needn't be the span's own.
pub(super) fn taken_change(py: Python<'_>, prep: &Prepared, code: usize, v: ValueId) -> bool {
    let ins = &prep.cfg[v];
    match ins.op {
        Op::Bytecode { opcode: opcodes::STORE_SUBSCR | opcodes::DELETE_SUBSCR, .. } => taken(prep, code, ins),
        Op::Call { .. } => {
            let callees = unsafe { &*prep.callees.get() };
            callees.get(&v).and_then(|c| c.get(py)).is_some_and(|f| taken_call(code, ins.origin, &f))
        }
        _ => false,
    }
}

// For a cfg pass's view of instr v of `code`: how it writes, and whether a
// region takes the write with it as things stand. None: it is no write this
// module knows.
pub(super) fn view(py: Python<'_>, prep: &Prepared, code: usize, v: ValueId) -> Option<(&'static str, bool)> {
    let ins = &prep.cfg[v];
    let how = match ins.op {
        Op::StoreAttr(_) => "store attr",
        Op::Bytecode { opcode: opcodes::DELETE_ATTR, .. } => "delete attr",
        Op::Bytecode { opcode: opcodes::STORE_SUBSCR, .. } => "store item",
        Op::Bytecode { opcode: opcodes::DELETE_SUBSCR, .. } => "delete item",
        Op::Call { .. } => {
            let callees = unsafe { &*prep.callees.get() };
            let f = callees.get(&v).and_then(|c| c.get(py))?;
            return unsafe { writer(f.as_ptr()) }.map(|_| ("call", taken_call(code, ins.origin, &f)));
        }
        _ => return None,
    };
    Some((how, taken(prep, code, ins)))
}

// ---- The C code a write is made by ---------------------------------------

// The C code whose calls are the writes a region can make again, and the
// calls that take a write back.
struct Writers {
    setattr: Py<PyAny>,
    delattr: Py<PyAny>,
    dict_setitem: Py<PyAny>,
    dict_delitem: Py<PyAny>,
    list_setitem: Py<PyAny>,
    list_delitem: Py<PyAny>,
    list_append: Py<PyAny>,
    list_pop: Py<PyAny>,
    list_insert: Py<PyAny>,
    set_add: Py<PyAny>,
    set_discard: Py<PyAny>,
    // collections.defaultdict, and the descriptor of its default_factory.
    defaultdict: Py<PyAny>,
    default_factory: Py<PyAny>,
    // The names a class is asked for to find how a store is carried out:
    // of an attribute, of its deletion, of an item, of its deletion.
    asked: [Py<PyAny>; 4],
}

fn writers() -> &'static Writers {
    static W: OnceLock<Writers> = OnceLock::new();
    W.get_or_init(|| {
        let py = unsafe { Python::assume_attached() };
        let at = |code: &std::ffi::CStr| py.eval(code, None, None).expect("a builtin").unbind();
        Writers {
            setattr: at(c"object.__setattr__"),
            delattr: at(c"object.__delattr__"),
            dict_setitem: at(c"dict.__dict__['__setitem__']"),
            dict_delitem: at(c"dict.__dict__['__delitem__']"),
            list_setitem: at(c"list.__dict__['__setitem__']"),
            list_delitem: at(c"list.__dict__['__delitem__']"),
            list_append: at(c"list.__dict__['append']"),
            list_pop: at(c"list.__dict__['pop']"),
            list_insert: at(c"list.__dict__['insert']"),
            set_add: at(c"set.__dict__['add']"),
            set_discard: at(c"set.__dict__['discard']"),
            defaultdict: at(c"__import__('collections').defaultdict"),
            default_factory: at(c"__import__('collections').defaultdict.__dict__['default_factory']"),
            asked: [at(c"'__setattr__'"), at(c"'__delattr__'"), at(c"'__setitem__'"), at(c"'__delitem__'")],
        }
    })
}

// What a call of C code writes, for the calls this module records.
#[derive(Clone, Copy, PartialEq)]
enum Writer {
    SetAttr,
    DelAttr,
    DictSet,
    DictDel,
    ListSet,
    ListDel,
    Append,
    Add,
    Discard,
}

impl Writer {
    // As a refusal names the call, and how many arguments it takes with
    // the target.
    fn shape(self) -> (&'static str, usize) {
        match self {
            Writer::SetAttr => ("object.__setattr__", 3),
            Writer::DelAttr => ("object.__delattr__", 2),
            Writer::DictSet => ("a dict's __setitem__", 3),
            Writer::DictDel => ("a dict's __delitem__", 2),
            Writer::ListSet => ("list.__setitem__", 3),
            Writer::ListDel => ("list.__delitem__", 2),
            Writer::Append => ("list.append", 2),
            Writer::Add => ("set.add", 2),
            Writer::Discard => ("set.discard", 2),
        }
    }
}

// THE answer to "is a call of `f` (a method of a builtin type, as its class
// has it) a write a region takes with it", for the optimizer (taken_call,
// view), for the executor (writes_call) and for the recorder: what it names
// here, write_call records or refuses. None: it is no write of this module.
unsafe fn writer(f: *mut PyObject) -> Option<Writer> {
    let w = writers();
    let named = [
        (&w.setattr, Writer::SetAttr),
        (&w.delattr, Writer::DelAttr),
        (&w.list_setitem, Writer::ListSet),
        (&w.list_delitem, Writer::ListDel),
        (&w.list_append, Writer::Append),
        (&w.set_add, Writer::Add),
        (&w.set_discard, Writer::Discard),
    ];
    named.iter().find(|(x, _)| x.as_ptr() == f).map(|(_, what)| *what).or_else(|| dict_slot(f))
}

// The __setitem__ or __delitem__ of dict or of a C subclass of it (an
// OrderedDict's), if descriptor `f` is one.
unsafe fn dict_slot(f: *mut PyObject) -> Option<Writer> {
    let tp = ffi::Py_TYPE(f);
    if (tp != &raw mut ffi::PyWrapperDescr_Type && tp != &raw mut ffi::PyMethodDescr_Type) || ffi::PyType_IsSubtype((*(f as *mut ffi::PyDescrObject)).d_type, &raw mut ffi::PyDict_Type) == 0 {
        return None;
    }
    match reads::text(tr_descr_name(f)).as_str() {
        "__setitem__" => Some(Writer::DictSet),
        "__delitem__" => Some(Writer::DictDel),
        _ => None,
    }
}

// ---- Recording, while the body is traced ---------------------------------

// A write of the traced run: `how` called with the target and `args`, and
// the call that takes it back if the trace gives up and its code is run
// again (None: nothing to take back).
struct Write {
    how: Py<PyAny>,
    target: Py<PyAny>,
    args: Vec<Py<PyAny>>,
    undo: Option<(Py<PyAny>, Vec<Py<PyAny>>)>,
}

impl parked::Holds for Write {
    fn each(&self, f: &mut dyn FnMut(*mut PyObject)) {
        self.how.each(f);
        self.target.each(f);
        self.args.each(f);
        self.undo.each(f);
    }
}

thread_local! {
    // The writes of the trace on this thread, in order: from begin() until
    // _region_writes takes them or undo() takes them back.
    static WRITES: RefCell<Vec<Write>> = const { RefCell::new(Vec::new()) };
    // The first error of a write that could not be taken back, until the
    // executor raises it (undo).
    static NOT_UNDONE: RefCell<Option<PyErr>> = const { RefCell::new(None) };
    // The first error of a place the run wrote a stand-in of the tracer's
    // to that could not be given the plain value, until the executor raises
    // it (stood).
    static NOT_PLAIN: RefCell<Option<PyErr>> = const { RefCell::new(None) };
}

// A token of a write recorded before it is made: 0 for none; REFUSED for a
// write that was not to be made (see refused).
pub type Token = usize;
pub const REFUSED: Token = usize::MAX;

// Starts the writes of a trace on this thread.
pub fn begin() {
    // Looked up here, outside the executor: not a read of the body's.
    writers();
    WRITES.with_borrow_mut(|w| w.clear());
    NOT_UNDONE.set(None);
    NOT_PLAIN.set(None);
}

fn recording() -> bool {
    reads::on() && !covered::trusted()
}

// The trace cannot make this write again: it is not made, and the trace is
// refused (the executor raises the refusal in the write's place).
fn refused(why: String) -> Token {
    let py = unsafe { Python::assume_attached() };
    if let Err(e) = py.import("torch._rust_inductor").and_then(|m| m.call_method1("_refuse", (format!("it {why}"),))) {
        super::region::_region_refuse(e.into_value(py).into_any());
    }
    REFUSED
}

unsafe fn push(how: &Py<PyAny>, target: *mut PyObject, args: &[*mut PyObject], undo: Option<(&Py<PyAny>, Vec<*mut PyObject>)>) -> Token {
    let py = Python::assume_attached();
    let w = Write {
        how: how.clone_ref(py),
        target: reads::own(target),
        args: args.iter().map(|a| reads::own(*a)).collect(),
        undo: undo.map(|(f, a)| (f.clone_ref(py), a.into_iter().map(|x| reads::own(x)).collect())),
    };
    WRITES.with_borrow_mut(|ws| {
        ws.push(w);
        ws.len()
    })
}

// The write recorded as `token` was made (`ok`), or raised: then it is none.
pub fn done(token: Token, ok: bool) {
    if token != 0 && token != REFUSED && !ok {
        let last = WRITES.with_borrow_mut(|ws| ws.len() == token && ws.pop().is_some());
        if last {
            reads::unwrote(token);
        }
    }
}

// A store (`old`: what was there, NULL for nothing) or a deletion (`value`
// NULL) by `set` / `delete`, with the call that takes it back. `holder`:
// what the reads know the item by (an instance's __dict__ for an attribute
// stored there, else the target), `how`: as an item of it or as a slot.
unsafe fn store(holder: *mut PyObject, how: How, target: *mut PyObject, key: *mut PyObject, value: *mut PyObject, old: *mut PyObject, set: (&Py<PyAny>, &Py<PyAny>), delete: (&Py<PyAny>, &Py<PyAny>)) -> Token {
    if value.is_null() && old.is_null() {
        // A deletion of what is not there raises, in C: the holder was
        // asked under a key whose lookup is C code (see askable), and the
        // deletion makes the same lookup.
        return 0;
    }
    let token = if value.is_null() {
        push(delete.0, target, &[key], Some((set.1, vec![key, old])))
    } else if old.is_null() {
        push(set.0, target, &[key, value], Some((delete.1, vec![key])))
    } else {
        push(set.0, target, &[key, value], Some((set.1, vec![key, old])))
    };
    reads::wrote(holder, token, how, key, value, old.is_null() as isize - value.is_null() as isize);
    token
}

// Why this module does not ask `holder` (a dict or a set) what it has under
// `key`: the lookup would run code of the program (keys.rs). None: it is C
// code that reads the key and the holder's keys alone.
unsafe fn unaskable(holder: *mut PyObject, key: *mut PyObject) -> Option<String> {
    if !keys::plain(key) {
        Some(format!("under a {}, whose hash or comparison is code of its class", reads::type_name(key)))
    } else if !keys::holds_plain(holder) {
        Some("that holds a key whose hash or comparison is code of its class".into())
    } else {
        None
    }
}

// `obj.name = value` (value null: `del obj.name`) is about to run; `generic`:
// as object.__setattr__ does it, whatever obj's class defines. `name` is an
// exact str (a code object's name, or see write_call).
pub unsafe fn store_attr(obj: *mut PyObject, name: *mut PyObject, value: *mut PyObject, generic: bool) -> Token {
    if !recording() {
        return 0;
    }
    let del = value.is_null();
    let w = writers();
    let (set, delete) = ((&w.setattr, &w.setattr), (&w.delattr, &w.delattr));
    // How the store is carried out is decided by what obj's class has under
    // the name, and (but for object.__setattr__ itself) under __setattr__:
    // both are reads, so that a region makes the write again only for a
    // class that still carries it out this way. (Reads of the class, which
    // hold nothing of obj.)
    if !generic {
        reads::class_has(obj, w.asked[del as usize].as_ptr());
    }
    reads::class_has(obj, name);
    match tr_store_attr_kind(ffi::Py_TYPE(obj), name, generic as std::ffi::c_int, del as std::ffi::c_int) {
        // Python code of the class runs: its stores are what happens.
        0 => 0,
        // The plain store into the instance's dict.
        1 => {
            let d = reads::instance_dict(obj);
            if d.is_null() {
                return refused(format!("stores .{} of a {} that has no __dict__", reads::text(name), reads::type_name(obj)));
            }
            if let Some(why) = unaskable(d, name) {
                return refused(format!("{} .{} of a {} whose __dict__ is one {why}", if del { "deletes" } else { "stores" }, reads::text(name), reads::type_name(obj)));
            }
            if del {
                // A delete raises unless the name is there.
                reads::needs_key(d, name);
            }
            reads::wrote_item(d, name);
            store(d, How::Item, obj, name, value, reads::dict_get(d, name), set, delete)
        }
        // A slot: the generic store, through the slot's descriptor.
        3 => {
            let old = tr_slot_get(obj, name);
            reads::wrote_item(obj, name);
            let token = store(obj, How::Slot, obj, name, value, old, set, delete);
            ffi::Py_XDECREF(old);
            token
        }
        _ => refused(format!("{} .{} of a {}, which a region cannot do again", if del { "deletes" } else { "stores" }, reads::text(name), reads::type_name(obj))),
    }
}

// `container[key] = value` (value null: `del container[key]`) is about to
// run, by `how` if a C method was called for it (then no code of the
// container's class runs, whatever the class defines).
pub unsafe fn store_item(container: *mut PyObject, key: *mut PyObject, value: *mut PyObject, how: Option<*mut PyObject>) -> Token {
    if !recording() || reads::tensor(container) || reads::owned(container) {
        return 0;
    }
    let w = writers();
    let del = value.is_null();
    let py = Python::assume_attached();
    // Whether code of the container's class runs for the instruction is
    // decided by what the class has under __setitem__ (__delitem__): a read.
    if how.is_none() {
        reads::class_has(container, w.asked[2 + del as usize].as_ptr());
    }
    // A C subclass of dict (an OrderedDict) stores by its own slot.
    let slot = if how.is_none() && ffi::PyDict_Check(container) != 0 && ffi::PyDict_CheckExact(container) == 0 && tr_python_setitem(ffi::Py_TYPE(container), del as std::ffi::c_int) == 0 {
        let name = pyo3::types::PyString::intern(py, if del { "__delitem__" } else { "__setitem__" });
        let d = tr_type_lookup(ffi::Py_TYPE(container), name.as_ptr());
        (!d.is_null()).then_some(d)
    } else {
        None
    };
    let how = how.or(slot);
    let called = how.map(|f| Py::<PyAny>::from_borrowed_ptr(py, f));
    if ffi::PyDict_Check(container) != 0 && (how.is_some() || ffi::PyDict_CheckExact(container) != 0) {
        if let Some(why) = unaskable(container, key) {
            // A dict the run made is the run's own, as it was before there
            // were writes to record: nothing of it is taken back, and what
            // a region hands on of it, it builds as the run left it.
            if made::is(container) {
                return 0;
            }
            return refused(format!("{} an item of a {} {why}", if del { "deletes" } else { "stores" }, reads::type_name(container)));
        }
        if del {
            reads::needs_key(container, key);
        }
        reads::wrote_item(container, key);
        changes(container);
        let set = called.as_ref().filter(|_| !del).unwrap_or(&w.dict_setitem);
        let delete = called.as_ref().filter(|_| del).unwrap_or(&w.dict_delitem);
        return store(container, How::Item, container, key, value, reads::dict_get(container, key), (set, &w.dict_setitem), (delete, &w.dict_delitem));
    }
    if ffi::PyList_CheckExact(container) != 0 && ffi::PyLong_CheckExact(key) != 0 {
        let n = ffi::PyList_GET_SIZE(container);
        let mut i = ffi::PyLong_AsSsize_t(key);
        if i < 0 {
            i += n;
        }
        if i < 0 || i >= n {
            // It raises, in C: the index (an exact int; one too large for
            // a Py_ssize_t left -1 and an error) is not one of this list.
            ffi::PyErr_Clear();
            return 0;
        }
        let old = ffi::PyList_GET_ITEM(container, i);
        // The index is in range for a list as long as this one.
        reads::needs_length(container);
        // (After a deletion the index holds what was behind it, which the
        // run did not write: what it reads there is a read.)
        if !del {
            reads::wrote_item(container, key);
        }
        changes(container);
        let token = if del { push(&w.list_delitem, container, &[key], Some((&w.list_insert, vec![key, old]))) } else { push(&w.list_setitem, container, &[key, value], Some((&w.list_setitem, vec![key, old]))) };
        reads::wrote(container, token, How::Index(i), std::ptr::null_mut(), value, -(del as isize));
        return token;
    }
    if how.is_none() && tr_python_setitem(ffi::Py_TYPE(container), del as std::ffi::c_int) != 0 {
        // The class's own code runs: its stores are what happens.
        return 0;
    }
    refused(format!("{} an item of a {}, which a region cannot do again", if del { "deletes" } else { "stores" }, reads::type_name(container)))
}

// `container[start:stop] = value` is about to run. Into a tensor it is torch
// ops on that tensor, and into a container the run made it is the run's own;
// the code of a Python class runs under the executor. Into anything else it
// is a write no region makes again: the trace is refused.
pub unsafe fn store_slice(container: *mut PyObject) -> Token {
    if !recording() || reads::tensor(container) || reads::owned(container) || made::is(container) || tr_python_setitem(ffi::Py_TYPE(container), 0) != 0 {
        return 0;
    }
    refused(format!("stores a slice of a {}, which a region cannot do again", reads::type_name(container)))
}

// `container[key]` is about to run. A defaultdict that is asked for a key it
// does not have stores what its factory makes under that key, in C: a write
// no region makes again, and none to record. The reads are told (see
// reads::fills): a region is entered only where the key is there.
pub unsafe fn load_item(container: *mut PyObject, key: *mut PyObject) -> Token {
    if !recording() || ffi::PyDict_Check(container) == 0 || ffi::PyDict_CheckExact(container) != 0 || reads::owned(container) || made::is(container) {
        return 0;
    }
    let w = writers();
    let class = w.defaultdict.as_ptr() as *mut ffi::PyTypeObject;
    if ffi::PyObject_TypeCheck(container, class) == 0 {
        return 0;
    }
    // (Asked as the reads ask a dict about a key, under any key: a
    // defaultdict keyed by tensors is read inside regions. This records no
    // write and holds no record while it asks.)
    if !reads::dict_get(container, key).is_null() {
        return 0;
    }
    // The factory as defaultdict's own descriptor gives it (C code that
    // reads the field), whatever a subclass has under the name.
    let Some(get) = (*ffi::Py_TYPE(w.default_factory.as_ptr())).tp_descr_get else {
        return refused(format!("reads a key a {} does not have, and its factory cannot be asked for", reads::type_name(container)));
    };
    let factory = get(w.default_factory.as_ptr(), container, class as *mut PyObject);
    if factory.is_null() {
        ffi::PyErr_Clear();
        return refused(format!("reads a key a {} does not have, and its factory cannot be asked for", reads::type_name(container)));
    }
    ffi::Py_DECREF(factory);
    if factory != ffi::Py_None() {
        reads::fills(container);
    }
    0
}

// The builtin `callee`, which writer() says is a write, is about to be
// called with these positional arguments (`keywords`: and with keyword
// arguments): the write is recorded, or the trace is refused in its place.
// Every case that is neither says why the call writes nothing a region has
// to make again.
pub unsafe fn write_call(callee: *mut PyObject, args: &[*mut PyObject], keywords: bool) -> Token {
    if !recording() {
        return 0;
    }
    let w = writers();
    let f = tr_bound_function(callee);
    let Some(what) = writer(f) else {
        return refused("calls C code its trace took for a write and cannot record".into());
    };
    let me = tr_builtin_self(callee);
    let mut all = vec![];
    if !me.is_null() {
        all.push(me);
    }
    all.extend_from_slice(args);
    let (name, takes) = what.shape();
    // None of them takes a keyword, or another number of arguments: the
    // call raises. It is not made under the trace.
    if keywords || all.len() != takes {
        return refused(format!("calls {name} with {} arguments{}", all.len(), if keywords { " and keywords" } else { "" }));
    }
    let of = |x: *mut PyObject| reads::type_name(x);
    match what {
        Writer::SetAttr | Writer::DelAttr => {
            // A name of a subclass of str is a key of the instance's dict
            // with a hash and a comparison of its own; any other name raises.
            if ffi::PyUnicode_CheckExact(all[1]) == 0 {
                return refused(format!("calls {name} with a name that is a {}, not a str", of(all[1])));
            }
            store_attr(all[0], all[1], if what == Writer::SetAttr { all[2] } else { std::ptr::null_mut() }, true)
        }
        Writer::DictSet | Writer::DictDel => {
            // dict's, or that of a C subclass of it (an OrderedDict's).
            if ffi::PyDict_Check(all[0]) == 0 {
                return refused(format!("calls {name} on a {}", of(all[0])));
            }
            store_item(all[0], all[1], if what == Writer::DictSet { all[2] } else { std::ptr::null_mut() }, Some(f))
        }
        Writer::ListSet | Writer::ListDel => {
            if ffi::PyList_Check(all[0]) == 0 {
                return refused(format!("calls {name} on a {}", of(all[0])));
            }
            // (By `f`: the list's own code does not run, whatever its class
            // defines; store_item records it for an exact list alone.)
            store_item(all[0], all[1], if what == Writer::ListSet { all[2] } else { std::ptr::null_mut() }, Some(f))
        }
        Writer::Append => {
            if ffi::PyList_Check(all[0]) == 0 {
                return refused(format!("calls {name} on a {}", of(all[0])));
            }
            // The trace's own record (make_fx's): no write of the body's.
            if reads::owned(all[0]) {
                return 0;
            }
            changes(all[0]);
            let token = push(&w.list_append, all[0], &[all[1]], Some((&w.list_pop, vec![])));
            reads::wrote(all[0], token, How::Append, std::ptr::null_mut(), all[1], 1);
            token
        }
        Writer::Add | Writer::Discard => {
            if ffi::PySet_Check(all[0]) == 0 {
                return refused(format!("calls {name} on a {}", of(all[0])));
            }
            // The trace's own record (make_fx's): no write of the body's.
            if reads::owned(all[0]) {
                return 0;
            }
            if let Some(why) = unaskable(all[0], all[1]) {
                // A set the run made is the run's own (see store_item).
                if made::is(all[0]) {
                    return 0;
                }
                return refused(format!("changes a {} {why}", of(all[0])));
            }
            let had = ffi::PySet_Contains(all[0], all[1]);
            if had < 0 {
                ffi::PyErr_Clear();
                return refused(format!("changes a {} that cannot say whether it has the member", of(all[0])));
            }
            changes(all[0]);
            let add = what == Writer::Add;
            let undo = if add == (had == 1) { None } else if add { Some((&w.set_discard, vec![all[1]])) } else { Some((&w.set_add, vec![all[1]])) };
            let token = push(if add { &w.set_add } else { &w.set_discard }, all[0], &[all[1]], undo);
            let grew = if add == (had == 1) { 0 } else if add { 1 } else { -1 };
            reads::wrote(all[0], token, if add { How::Add } else { How::Discard }, std::ptr::null_mut(), all[1], grew);
            token
        }
    }
}

// Whether a call of the builtin `callee` is a write (see writer): the
// executor then asks write_call before it makes the call.
pub unsafe fn writes_call(callee: *mut PyObject) -> bool {
    writer(tr_bound_function(callee)).is_some()
}

// ---- When the body is over -------------------------------------------------

// The trace gave up: what it wrote is taken back, the last write first.
// Each undo is C code on what the write left; it can fail only where
// something that is no recorded write changed the target since (a callee
// the pass list declares pure that writes, a finalizer). Every write is
// tried, whatever one before it did; the first failure is kept for undo().
// Called before the trace's gate (region::trace; and by the tracer in front
// of what it gives its arguments' containers back: _region_given_up). The
// records are parked: a value the trace stored, which its record alone
// holds once the store is taken back, dies at the gate.
pub(super) fn take_back() {
    let ws = WRITES.take();
    let py = unsafe { Python::assume_attached() };
    for w in ws.into_iter().rev() {
        if let Some((f, args)) = &w.undo {
            let all: Vec<Py<PyAny>> = std::iter::once(&w.target).chain(args).map(|x| x.clone_ref(py)).collect();
            if let Err(err) = pyo3::types::PyTuple::new(py, all).and_then(|t| f.bind(py).call1(t)) {
                NOT_UNDONE.with_borrow_mut(|first| {
                    first.get_or_insert(err);
                });
            }
        }
        parked::park(w);
    }
}

// The trace gave up and the executor is about to run its code again. What
// the trace wrote was taken back before its gate (take_back). Err: a write
// could not be taken back. What the trace wrote to is then neither as the
// call found it nor as the code leaves it, and the code is not run again
// over it: the error is raised in the program, in the place of the code.
pub fn undo() -> PyResult<()> {
    let Some(cause) = NOT_UNDONE.take() else { return Ok(()) };
    let py = unsafe { Python::assume_attached() };
    let err = pyo3::exceptions::PyRuntimeError::new_err(
        "torch.vm: a trace was given up and what it had written could not be taken back, so the objects it wrote to are not as the call found them. \
         Something that is no store of the traced code changed one of them meanwhile (a function the pass list declares pure that writes, a finalizer).",
    );
    err.set_cause(py, Some(cause));
    Err(err)
}

/// The run of the trace on this thread stands, and a place it wrote one of
/// the tracer's stand-ins for a float to could not be given the float
/// (torch/vm/_writes.py): ``error`` says why. The executor raises it in the
/// program when the trace is over, in the place of going on with an object
/// of the tracer's in the program's state.
#[pyfunction]
pub fn _region_not_plain(error: Bound<'_, PyAny>) {
    NOT_PLAIN.with_borrow_mut(|first| {
        first.get_or_insert_with(|| PyErr::from_value(error));
    });
}

// The trace is over and the run of its body stands. Err: a place the run
// wrote to still holds a stand-in of the tracer's (_region_not_plain). The
// error is raised in the program behind the traced code.
pub fn stood() -> PyResult<()> {
    let Some(cause) = NOT_PLAIN.take() else { return Ok(()) };
    let py = unsafe { Python::assume_attached() };
    let err = pyo3::exceptions::PyRuntimeError::new_err(
        "torch.vm: a traced call stored a float it computed from a float argument, and the object of the tracer's that stood for the float while the call was traced could not be replaced by the float: the object it was stored into still holds it.",
    );
    err.set_cause(py, Some(cause));
    Err(err)
}

// The body is over and its run stands: nothing is taken back from here on,
// and the records no longer count what the stores replaced. The tracer's
// modes are still on: a replaced value is parked, in the order of the
// stores, and dies at the trace's gate (parked.rs), where the finalizer the
// program may have given it runs as the program's code.
pub(super) fn let_go() {
    let replaced: Vec<_> = WRITES.with_borrow_mut(|ws| ws.iter_mut().filter_map(|w| w.undo.take()).collect());
    if !replaced.is_empty() {
        parked::park(replaced);
    }
}

/// The writes of the traced run that a region is to make again, once the
/// body is over, in order: ``(how, target, args)``, each made as
/// ``how(target, *args)``. A write to an object that nothing but the trace's
/// own records refers to any more is none: nothing can see it (a list the
/// run made, filled and let go of).
#[pyfunction]
pub fn _region_writes(py: Python<'_>) -> PyResult<Vec<(Py<PyAny>, Py<PyAny>, Py<pyo3::types::PyTuple>)>> {
    // The records of writes nothing can see, and the entries swept with
    // them, are parked here and go at the trace's gate (parked.rs): this
    // runs under the tracer's modes. What an object's count would be had
    // they gone is asked of parked::refs.
    let _releasing = super::foreign::Releasing::start();
    // (Already done where the reads were compared first; the counts below
    // are of the records without what the stores replaced.)
    let_go();
    let mut ws = WRITES.take();
    loop {
        // The records of what the run read from its temporaries hold what
        // those held (a call's arguments, in the cells of a closure): they
        // go first, and again whenever a write that held a temporary went.
        reads::sweep_trace();
        // The references the records themselves hold, by object.
        let mut mine: HashMap<usize, isize> = HashMap::new();
        {
            let mut hold = |o: &Py<PyAny>| *mine.entry(o.as_ptr() as usize).or_insert(0) += 1;
            for w in &ws {
                hold(&w.target);
                w.args.iter().for_each(&mut hold);
            }
            reads::held_by_trace(&mut hold);
        }
        parked::settle();
        let dead: HashSet<usize> = ws.iter().map(|w| w.target.as_ptr() as usize).filter(|p| unsafe { parked::refs(*p as *mut PyObject) } == mine[p]).collect();
        if dead.is_empty() {
            break;
        }
        let (gone, stay): (Vec<Write>, Vec<Write>) = ws.into_iter().partition(|w| dead.contains(&(w.target.as_ptr() as usize)));
        ws = stay;
        parked::park(gone);
    }
    // What the tracer's Python code is handed stays counted until the
    // trace's gate (a value the run stored and then replaced is held by its
    // record alone).
    for w in &ws {
        unsafe {
            super::foreign::keep(w.target.as_ptr());
            w.args.iter().for_each(|a| super::foreign::keep(a.as_ptr()));
        }
    }
    ws.into_iter().map(|w| Ok((w.how, w.target, pyo3::types::PyTuple::new(py, w.args)?.unbind()))).collect()
}

// ---- On every run of the region --------------------------------------------

// Makes a region's writes, once its graph ran and its outputs are made:
// `writes` are RegionOut::Call(how, [target, values...]) with `how` an index
// into `objects`, and `make` builds what a template stands for (NULL with an
// exception set if it cannot). What each write takes is built before the
// first is made, so that a region that raises until then has written
// nothing; the writes themselves are C code that does not raise (what each
// needs was read by the trace, and is as it was: the class of the target and
// how it carries the store out, that a key to delete is there, that an index
// is one of the list, and a key is hashed and compared by C code). Should one
// raise all the same, the writes before it stay made and those behind it are
// not made, as the store raising in the code leaves it. False with an
// exception set if something failed.
pub unsafe fn replay(writes: &[RegionOut], objects: &[Py<PyAny>], make: &dyn Fn(&RegionOut) -> *mut PyObject) -> bool {
    let mut calls: Vec<(*mut PyObject, *mut PyObject)> = Vec::with_capacity(writes.len());
    let mut ok = true;
    for w in writes {
        let RegionOut::Call(how, items) = w else { unreachable!("a write is a call") };
        let all = ffi::PyTuple_New(items.len() as isize);
        for (i, it) in items.iter().enumerate() {
            let x = if all.is_null() { all } else { make(it) };
            if x.is_null() {
                ok = false;
                break;
            }
            ffi::PyTuple_SET_ITEM(all, i as isize, x);
        }
        calls.push((objects[*how as usize].as_ptr(), all));
        if !ok {
            break;
        }
    }
    for (how, all) in calls {
        if ok {
            let x = ffi::PyObject_Call(how, all, std::ptr::null_mut());
            ok = !x.is_null();
            ffi::Py_XDECREF(x);
        }
        ffi::Py_XDECREF(all);
    }
    ok
}
