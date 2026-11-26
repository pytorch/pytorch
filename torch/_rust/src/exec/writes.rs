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
//     written (what the run reads of it afterwards is its own);
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
//   - undo() takes back what the trace wrote, the last write first, for the
//     executor to run the code again as it is.
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
use super::made;
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
    wanted(code, origin) && unsafe { writer(f.as_ptr()) }
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
            return unsafe { writer(f.as_ptr()) }.then(|| ("call", taken_call(code, ins.origin, &f)));
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
        }
    })
}

// Whether `f` (a method of a builtin type, as its class has it) is C code
// whose call is a write that write_call records.
unsafe fn writer(f: *mut PyObject) -> bool {
    let w = writers();
    dict_slot(f) || [&w.setattr, &w.delattr, &w.list_setitem, &w.list_delitem, &w.list_append, &w.set_add, &w.set_discard].iter().any(|x| x.as_ptr() == f)
}

// Whether descriptor `f` is the __setitem__ or __delitem__ of dict or of a C
// subclass of it (an OrderedDict's).
unsafe fn dict_slot(f: *mut PyObject) -> bool {
    let tp = ffi::Py_TYPE(f);
    (tp == &raw mut ffi::PyWrapperDescr_Type || tp == &raw mut ffi::PyMethodDescr_Type)
        && ffi::PyType_IsSubtype((*(f as *mut ffi::PyDescrObject)).d_type, &raw mut ffi::PyDict_Type) != 0
        && matches!(reads::text(tr_descr_name(f)).as_str(), "__setitem__" | "__delitem__")
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

thread_local! {
    // The writes of the trace on this thread, in order: from begin() until
    // _region_writes takes them or undo() takes them back.
    static WRITES: RefCell<Vec<Write>> = const { RefCell::new(Vec::new()) };
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
        WRITES.with_borrow_mut(|ws| {
            if ws.len() == token {
                ws.pop();
            }
        });
    }
}

// A store (`old`: what was there, NULL for nothing) or a deletion (`value`
// NULL) by `set` / `delete`, with the call that takes it back.
unsafe fn store(target: *mut PyObject, key: *mut PyObject, value: *mut PyObject, old: *mut PyObject, set: (&Py<PyAny>, &Py<PyAny>), delete: (&Py<PyAny>, &Py<PyAny>)) -> Token {
    if value.is_null() {
        if old.is_null() {
            // It raises.
            return 0;
        }
        push(delete.0, target, &[key], Some((set.1, vec![key, old])))
    } else if old.is_null() {
        push(set.0, target, &[key, value], Some((delete.1, vec![key])))
    } else {
        push(set.0, target, &[key, value], Some((set.1, vec![key, old])))
    }
}

// `obj.name = value` (value null: `del obj.name`) is about to run; `generic`:
// as object.__setattr__ does it, whatever obj's class defines.
pub unsafe fn store_attr(obj: *mut PyObject, name: *mut PyObject, value: *mut PyObject, generic: bool) -> Token {
    if !recording() {
        return 0;
    }
    let del = value.is_null();
    let w = writers();
    let (set, delete) = ((&w.setattr, &w.setattr), (&w.delattr, &w.delattr));
    match tr_store_attr_kind(ffi::Py_TYPE(obj), name, generic as std::ffi::c_int, del as std::ffi::c_int) {
        // Python code of the class runs: its stores are what happens.
        0 => 0,
        // The plain store into the instance's dict.
        1 => {
            let d = reads::instance_dict(obj);
            if d.is_null() {
                return refused(format!("stores .{} of a {} that has no __dict__", reads::text(name), reads::type_name(obj)));
            }
            if del {
                // A delete raises unless the name is there.
                reads::needs_key(d, name);
            }
            reads::wrote_item(d, name);
            store(obj, name, value, reads::dict_get(d, name), set, delete)
        }
        // A slot: the generic store, through the slot's descriptor.
        3 => {
            let old = tr_slot_get(obj, name);
            reads::wrote_item(obj, name);
            let token = store(obj, name, value, old, set, delete);
            ffi::Py_XDECREF(old);
            token
        }
        _ => refused(format!("{} .{} of a {}, which a region cannot do again", if del { "deletes" } else { "stores" }, reads::text(name), reads::type_name(obj))),
    }
}

// `container[key] = value` (value null: `del container[key]`) is about to
// run, by `how` if a C method was called for it.
pub unsafe fn store_item(container: *mut PyObject, key: *mut PyObject, value: *mut PyObject, how: Option<*mut PyObject>) -> Token {
    if !recording() || reads::tensor(container) || reads::owned(container) {
        return 0;
    }
    let w = writers();
    let del = value.is_null();
    let py = Python::assume_attached();
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
        if del {
            reads::needs_key(container, key);
        }
        reads::wrote_item(container, key);
        changes(container);
        let set = called.as_ref().filter(|_| !del).unwrap_or(&w.dict_setitem);
        let delete = called.as_ref().filter(|_| del).unwrap_or(&w.dict_delitem);
        return store(container, key, value, reads::dict_get(container, key), (set, &w.dict_setitem), (delete, &w.dict_delitem));
    }
    if ffi::PyList_CheckExact(container) != 0 && ffi::PyLong_CheckExact(key) != 0 {
        let n = ffi::PyList_GET_SIZE(container);
        let mut i = ffi::PyLong_AsSsize_t(key);
        if i < 0 {
            i += n;
        }
        if i < 0 || i >= n {
            // It raises.
            ffi::PyErr_Clear();
            return 0;
        }
        let old = ffi::PyList_GET_ITEM(container, i);
        // The index is in range for a list as long as this one.
        reads::needs_length(container);
        reads::wrote_item(container, key);
        changes(container);
        return if del { push(&w.list_delitem, container, &[key], Some((&w.list_insert, vec![key, old]))) } else { push(&w.list_setitem, container, &[key, value], Some((&w.list_setitem, vec![key, old]))) };
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

// The builtin `callee` is about to be called with these positional
// arguments: a write a region makes again, if it is one of writers().
pub unsafe fn write_call(callee: *mut PyObject, args: &[*mut PyObject]) -> Token {
    if !recording() {
        return 0;
    }
    let w = writers();
    let f = tr_bound_function(callee);
    let me = tr_builtin_self(callee);
    let mut all = vec![];
    if !me.is_null() {
        all.push(me);
    }
    all.extend_from_slice(args);
    let is = |x: &Py<PyAny>| x.as_ptr() == f;
    let named = |name: &str| dict_slot(f) && reads::text(tr_descr_name(f)) == name;
    if is(&w.setattr) && all.len() == 3 && ffi::PyUnicode_CheckExact(all[1]) != 0 {
        store_attr(all[0], all[1], all[2], true)
    } else if is(&w.delattr) && all.len() == 2 && ffi::PyUnicode_CheckExact(all[1]) != 0 {
        store_attr(all[0], all[1], std::ptr::null_mut(), true)
    } else if all.len() == 3 && ffi::PyDict_Check(all[0]) != 0 && named("__setitem__") {
        // dict's, or that of a C subclass of it (an OrderedDict's).
        store_item(all[0], all[1], all[2], Some(f))
    } else if all.len() == 2 && ffi::PyDict_Check(all[0]) != 0 && named("__delitem__") {
        store_item(all[0], all[1], std::ptr::null_mut(), Some(f))
    } else if is(&w.list_setitem) && all.len() == 3 {
        store_item(all[0], all[1], all[2], None)
    } else if is(&w.list_delitem) && all.len() == 2 {
        store_item(all[0], all[1], std::ptr::null_mut(), None)
    } else if all.first().is_some_and(|x| reads::owned(*x)) {
        0
    } else if is(&w.list_append) && all.len() == 2 && ffi::PyList_Check(all[0]) != 0 {
        changes(all[0]);
        push(&w.list_append, all[0], &[all[1]], Some((&w.list_pop, vec![])))
    } else if (is(&w.set_add) || is(&w.set_discard)) && all.len() == 2 && ffi::PySet_Check(all[0]) != 0 {
        let had = ffi::PySet_Contains(all[0], all[1]);
        if had < 0 {
            // It raises.
            ffi::PyErr_Clear();
            return 0;
        }
        changes(all[0]);
        let add = is(&w.set_add);
        let undo = if add == (had == 1) { None } else if add { Some((&w.set_discard, vec![all[1]])) } else { Some((&w.set_add, vec![all[1]])) };
        push(if add { &w.set_add } else { &w.set_discard }, all[0], &[all[1]], undo)
    } else {
        0
    }
}

// Whether a call of the builtin `callee` is one write_call looks at.
pub unsafe fn writes_call(callee: *mut PyObject) -> bool {
    writer(tr_bound_function(callee))
}

// ---- When the body is over -------------------------------------------------

// The trace gave up and its code is run again: what it wrote is taken back,
// the last write first.
pub fn undo() {
    let ws = WRITES.take();
    let py = unsafe { Python::assume_attached() };
    for w in ws.into_iter().rev() {
        if let Some((f, args)) = w.undo {
            let mut all = vec![w.target.clone_ref(py)];
            all.extend(args);
            if let Err(err) = pyo3::types::PyTuple::new(py, all).and_then(|t| f.bind(py).call1(t)) {
                err.write_unraisable(py, None);
            }
        }
    }
}

/// The writes of the traced run that a region is to make again, once the
/// body is over, in order: ``(how, target, args)``, each made as
/// ``how(target, *args)``. A write to an object that nothing but the trace's
/// own records refers to any more is none: nothing can see it (a list the
/// run made, filled and let go of).
#[pyfunction]
pub fn _region_writes(py: Python<'_>) -> PyResult<Vec<(Py<PyAny>, Py<PyAny>, Py<pyo3::types::PyTuple>)>> {
    let mut ws = WRITES.take();
    // The run is over and stands: nothing is taken back from here on.
    ws.iter_mut().for_each(|w| w.undo = None);
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
        let dead: HashSet<usize> = ws.iter().map(|w| w.target.as_ptr() as usize).filter(|p| unsafe { ffi::Py_REFCNT(*p as *mut PyObject) } == mine[p]).collect();
        if dead.is_empty() {
            break;
        }
        ws.retain(|w| !dead.contains(&(w.target.as_ptr() as usize)));
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
// needs was read by the trace, and is as it was). False with an exception
// set if something failed.
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
