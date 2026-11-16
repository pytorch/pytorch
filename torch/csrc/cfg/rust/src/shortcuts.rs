//! Shortcuts through Python code whose outcome a few cheap checks decide.
//!
//! Some library functions sit in front of nearly every interesting call and
//! nearly always do the same thing: `nn.Module.__call__` calls `forward`
//! unless the module has hooks, and `nn.Module.__getattr__` looks a name up
//! in three dicts. A shortcut says, for one such function, what it comes down
//! to and under which conditions; the VM checks the conditions on every use
//! and otherwise runs the function itself. What the conditions and the
//! outcome are is decided where the function is defined, in Python (see
//! `torch/cfg_compile/_shortcuts.py`), and registered here.

use crate::capi;
use crate::pyffi::{self, Obj, PyObj, Shared};
use pyo3::ffi;
use std::cell::Cell;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

pub enum Condition {
    /// The entry of the instance's dict is None or an empty container. An
    /// instance without the entry does not take the shortcut.
    InstanceFalsy(PyObj),
    /// The attribute is None on the class and absent from the instance's
    /// dict, or None there.
    AttrNone(PyObj),
    /// The entry of this dict (the globals of a module) is None or an empty
    /// container. The last field remembers the entry for one version of
    /// the dict.
    GlobalFalsy(PyObj, PyObj, Cell<(u64, Obj)>),
    /// Calling this without arguments returns something false.
    CallFalsy(PyObj),
    /// torch.jit.trace is not recording on this thread.
    NotTracing,
    /// The instance's dict has no entry of this name, so the attribute is
    /// what the class says it is.
    NotShadowed(PyObj),
}

/// A call of an instance whose class has `function` as its `__call__` is a
/// call of the method `target` of the instance, if the conditions hold.
pub struct CallShortcut {
    pub function: PyObj,
    pub target: PyObj,
    pub conditions: Vec<Condition>,
}

/// For an instance whose class has `function` as its `__getattr__` (and the
/// default `__getattribute__`), an attribute that the normal lookup does not
/// find is an entry of one of these dicts of the instance's dict, looked at in
/// this order; if it is in none, `function` is called after all.
pub struct GetattrShortcut {
    pub function: PyObj,
    pub dicts: Vec<PyObj>,
}

/// The shortcuts are never freed: there are a handful per process, and the
/// VM uses them while Python code runs that could register others.
#[derive(Default)]
pub struct Shortcuts {
    pub calls: Vec<&'static CallShortcut>,
    pub getattrs: Vec<&'static GetattrShortcut>,
    /// Per type and version of the type: which getattr shortcut applies
    /// (index + 1), or 0.
    types: Vec<(usize, u32, usize)>,
}

const TYPE_SLOTS: usize = 64;

fn registry() -> &'static Shared<Shortcuts> {
    static REGISTRY: OnceLock<Shared<Shortcuts>> = OnceLock::new();
    REGISTRY.get_or_init(|| Shared::new(Shortcuts::default()))
}

pub fn register_call(shortcut: CallShortcut) {
    let mut all = registry().borrow_mut();
    all.calls.retain(|s| s.function.as_ptr() != shortcut.function.as_ptr());
    all.calls.push(Box::leak(Box::new(shortcut)));
    EPOCH.fetch_add(1, Ordering::Relaxed);
}

pub fn register_getattr(shortcut: GetattrShortcut) {
    let mut all = registry().borrow_mut();
    all.getattrs.retain(|s| s.function.as_ptr() != shortcut.function.as_ptr());
    all.getattrs.push(Box::leak(Box::new(shortcut)));
    all.types.clear();
    EPOCH.fetch_add(1, Ordering::Relaxed);
}

pub fn clear() {
    let mut all = registry().borrow_mut();
    all.calls.clear();
    all.getattrs.clear();
    all.types.clear();
    EPOCH.fetch_add(1, Ordering::Relaxed);
}

unsafe fn falsy(value: Obj) -> Option<bool> {
    if value == ffi::Py_None() {
        return Some(true);
    }
    if ffi::PyDict_Check(value) != 0 {
        return Some(ffi::PyDict_Size(value) == 0);
    }
    match ffi::PyObject_IsTrue(value) {
        r if r < 0 => None,
        r => Some(r == 0),
    }
}

/// The dict of an instance, borrowed; null if it has none.
unsafe fn instance_dict(obj: Obj) -> Obj {
    let slot = pyffi::_PyObject_GetDictPtr(obj);
    if slot.is_null() {
        null_mut()
    } else {
        *slot
    }
}

/// Bumped whenever the set of shortcuts changes.
static EPOCH: AtomicU64 = AtomicU64::new(1);

/// What looking at one object for a call shortcut found, for as long as
/// neither its dict nor its type changes: the function to call, and the
/// entries of its dict that still have to be empty at every call (they are
/// containers that are changed in place). Pointers are borrowed from the
/// dict and the type, which the versions vouch for.
#[derive(Clone, Copy)]
struct Resolved {
    epoch: u64,
    obj: usize,
    ty: usize,
    type_version: u32,
    dict: usize,
    dict_version: u64,
    shortcut: Option<&'static CallShortcut>,
    target: Obj,
    watched: [Obj; MAX_WATCHED],
}

const MAX_WATCHED: usize = 6;
const RESOLVED_SLOTS: usize = 128;

struct ResolvedCache(Vec<Cell<Resolved>>);

// Only touched with the GIL held.
unsafe impl Send for ResolvedCache {}
unsafe impl Sync for ResolvedCache {}

fn resolved() -> &'static ResolvedCache {
    static CACHE: OnceLock<ResolvedCache> = OnceLock::new();
    CACHE.get_or_init(|| {
        let empty = Resolved {
            epoch: 0,
            obj: 0,
            ty: 0,
            type_version: 0,
            dict: 0,
            dict_version: 0,
            shortcut: None,
            target: null_mut(),
            watched: [null_mut(); MAX_WATCHED],
        };
        ResolvedCache((0..RESOLVED_SLOTS).map(|_| Cell::new(empty)).collect())
    })
}

#[inline(always)]
unsafe fn dict_version(dict: Obj) -> u64 {
    (*(dict as *mut ffi::PyDictObject)).ma_version_tag
}

#[inline(always)]
unsafe fn type_version(ty: *mut ffi::PyTypeObject) -> u32 {
    if ffi::PyType_HasFeature(ty, ffi::Py_TPFLAGS_VALID_VERSION_TAG) != 0 {
        (*ty).tp_version_tag
    } else {
        0
    }
}

/// The conditions that are about neither the instance's dict nor its type:
/// they are checked at every call.
unsafe fn holds_now(condition: &Condition) -> bool {
    match condition {
        Condition::GlobalFalsy(globals, name, seen) => {
            let version = dict_version(globals.as_ptr());
            let (seen_version, mut value) = seen.get();
            if seen_version != version {
                value = ffi::PyDict_GetItemWithError(globals.as_ptr(), name.as_ptr());
                seen.set((version, value));
            }
            !value.is_null() && falsy(value) == Some(true)
        }
        Condition::CallFalsy(callable) => {
            let result = ffi::PyObject_CallNoArgs(callable.as_ptr());
            if result.is_null() {
                return false;
            }
            let outcome = falsy(result);
            ffi::Py_DECREF(result);
            outcome == Some(true)
        }
        Condition::NotTracing => capi::api().is_some_and(|api| (api.is_tracing)() == 0),
        _ => true,
    }
}

/// Looks at the dict and the type of `obj`: the target of the shortcut and
/// the dict entries to watch, or None if the shortcut does not apply to the
/// object as it is.
unsafe fn resolve(
    shortcut: &'static CallShortcut,
    ty: *mut ffi::PyTypeObject,
    dict: Obj,
) -> Option<(Obj, [Obj; MAX_WATCHED])> {
    let target = pyffi::_PyType_Lookup(ty, shortcut.target.as_ptr());
    if target.is_null()
        || ffi::PyFunction_Check(target) == 0
        || !ffi::PyDict_GetItemWithError(dict, shortcut.target.as_ptr()).is_null()
    {
        return None;
    }
    let mut watched = [null_mut(); MAX_WATCHED];
    let mut n = 0;
    for condition in &shortcut.conditions {
        match condition {
            Condition::InstanceFalsy(name) => {
                let value = ffi::PyDict_GetItemWithError(dict, name.as_ptr());
                if value.is_null() || n == MAX_WATCHED {
                    return None;
                }
                watched[n] = value;
                n += 1;
            }
            Condition::AttrNone(name) => {
                let own = ffi::PyDict_GetItemWithError(dict, name.as_ptr());
                let none = if own.is_null() {
                    pyffi::_PyType_Lookup(ty, name.as_ptr()) == ffi::Py_None()
                } else {
                    own == ffi::Py_None()
                };
                if !none {
                    return None;
                }
            }
            Condition::NotShadowed(name) => {
                if !ffi::PyDict_GetItemWithError(dict, name.as_ptr()).is_null() {
                    return None;
                }
            }
            _ => {}
        }
    }
    Some((target, watched))
}

/// What a call of `obj`, whose class has `call` as `__call__`, comes down
/// to: the function to call with `obj` as its first argument, or null if no
/// shortcut applies. Never leaves an error set.
pub unsafe fn call_target(obj: Obj, ty: *mut ffi::PyTypeObject, call: Obj) -> Obj {
    let dict = instance_dict(obj);
    if dict.is_null() {
        return null_mut();
    }
    let epoch = EPOCH.load(Ordering::Relaxed);
    let slot = resolved().0.get_unchecked((obj as usize >> 4) % RESOLVED_SLOTS);
    let mut entry = slot.get();
    let fresh = entry.epoch == epoch
        && entry.obj == obj as usize
        && entry.ty == ty as usize
        && entry.type_version != 0
        && entry.type_version == type_version(ty)
        && entry.dict == dict as usize
        && entry.dict_version == dict_version(dict);
    if !fresh {
        let found = registry().borrow().calls.iter().copied().find(|s| s.function.as_ptr() == call);
        let looked = found.and_then(|shortcut| resolve(shortcut, ty, dict).map(|r| (shortcut, r)));
        if !ffi::PyErr_Occurred().is_null() {
            ffi::PyErr_Clear();
            return null_mut();
        }
        entry = Resolved {
            epoch,
            obj: obj as usize,
            ty: ty as usize,
            // Read after the lookups, which may have given the type one.
            type_version: type_version(ty),
            dict: dict as usize,
            dict_version: dict_version(dict),
            shortcut: looked.map(|(shortcut, _)| shortcut),
            target: looked.map_or(null_mut(), |(_, (target, _))| target),
            watched: looked.map_or([null_mut(); MAX_WATCHED], |(_, (_, watched))| watched),
        };
        slot.set(entry);
    }
    let Some(shortcut) = entry.shortcut else {
        return null_mut();
    };
    let quiet = entry.watched.iter().all(|value| value.is_null() || falsy(*value) == Some(true))
        && shortcut.conditions.iter().all(|c| holds_now(c));
    if !ffi::PyErr_Occurred().is_null() {
        ffi::PyErr_Clear();
        return null_mut();
    }
    if quiet {
        entry.target
    } else {
        null_mut()
    }
}

/// Which getattr shortcut applies to instances of `ty`: its index plus one,
/// or 0.
unsafe fn getattr_shortcut(ty: *mut ffi::PyTypeObject) -> usize {
    let version = (*ty).tp_version_tag;
    let valid = version != 0 && ffi::PyType_HasFeature(ty, ffi::Py_TPFLAGS_VALID_VERSION_TAG) != 0;
    let slot = (ty as usize >> 4) % TYPE_SLOTS;
    {
        let all = registry().borrow();
        if all.getattrs.is_empty() {
            return 0;
        }
        if valid {
            if let Some((known, known_version, which)) = all.types.get(slot) {
                if *known == ty as usize && *known_version == version {
                    return *which;
                }
            }
        }
    }
    static NAMES: OnceLock<(PyObj, PyObj)> = OnceLock::new();
    let (getattr, getattribute) = NAMES.get_or_init(|| {
        (
            PyObj::from_owned(ffi::PyUnicode_InternFromString(c"__getattr__".as_ptr())),
            PyObj::from_owned(ffi::PyUnicode_InternFromString(c"__getattribute__".as_ptr())),
        )
    });
    let object = std::ptr::addr_of_mut!(ffi::PyBaseObject_Type);
    let hook = pyffi::_PyType_Lookup(ty, getattr.as_ptr());
    let plain = pyffi::_PyType_Lookup(ty, getattribute.as_ptr())
        == pyffi::_PyType_Lookup(object, getattribute.as_ptr());
    let mut all = registry().borrow_mut();
    let which = if hook.is_null() || !plain {
        0
    } else {
        all.getattrs.iter().position(|s| s.function.as_ptr() == hook).map_or(0, |i| i + 1)
    };
    // The lookups above may have given the type a version.
    let version = (*ty).tp_version_tag;
    if version != 0 && ffi::PyType_HasFeature(ty, ffi::Py_TPFLAGS_VALID_VERSION_TAG) != 0 {
        if all.types.is_empty() {
            all.types = vec![(0, 0, 0); TYPE_SLOTS];
        }
        all.types[slot] = (ty as usize, version, which);
    }
    which
}

/// `getattr(obj, name)` through a getattr shortcut. Returns a new reference;
/// null without an error if no shortcut applies to the type of `obj`, null
/// with an error if the lookup raised.
pub unsafe fn load_attr(obj: Obj, name: Obj) -> Obj {
    let ty = ffi::Py_TYPE(obj);
    let which = getattr_shortcut(ty);
    if which == 0 {
        return null_mut();
    }
    // What the type's own lookup does first: the normal one, where not
    // finding the attribute is not an error yet.
    let found = pyffi::_PyObject_GenericGetAttrWithDict(obj, name, null_mut(), 1);
    if !found.is_null() || !ffi::PyErr_Occurred().is_null() {
        return found;
    }
    let shortcut = registry().borrow().getattrs.get(which - 1).copied();
    let dict = instance_dict(obj);
    if let (Some(shortcut), false) = (shortcut, dict.is_null()) {
        for key in &shortcut.dicts {
            let entries = ffi::PyDict_GetItemWithError(dict, key.as_ptr());
            if entries.is_null() {
                continue;
            }
            if ffi::PyDict_Check(entries) == 0 {
                break;
            }
            let value = ffi::PyDict_GetItemWithError(entries, name);
            if !value.is_null() {
                ffi::Py_INCREF(value);
                return value;
            }
            if !ffi::PyErr_Occurred().is_null() {
                return null_mut();
            }
        }
    }
    // Not there either: the function itself says so, or does what the
    // shortcut does not know about. It is called as the type's lookup calls
    // it, after the normal lookup, which is not run a second time.
    match shortcut {
        Some(shortcut) => {
            let args = [obj, name];
            ffi::PyObject_Vectorcall(shortcut.function.as_ptr(), args.as_ptr(), 2, null_mut())
        }
        None => ffi::PyObject_GetAttr(obj, name),
    }
}
