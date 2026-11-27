//! How a region holds an object of the program that it compares by identity
//! or reads from when it is entered.
//!
//! Given: a Python object.
//!
//! Guarantees: a pin that was weakened (`Pin::weak`, `weaken`, `through`)
//! counts no reference to an object whose class takes a weak reference, so
//! the object lives exactly as long as the program keeps it. `live()` is
//! None from the moment the object died, and `as_ptr()` NULL: an address is
//! handed out only while the object at it is the pinned one (an address is
//! used again by another object), and `is(x)` is false for a dead pin and
//! for NULL. A `__dict__` pinned `through` its owner is the dict the owner
//! has now, found again on every use; one pinned `at` its owner is the dict
//! the owner has now only while that is the dict it had (see `at` for why
//! an address says so there). No Python code runs in `live`, `as_ptr`, `is`
//! and `dead`.
//!
//! May not assume: that a pin lets go of an object whose class takes no
//! weak reference (an exact dict, list or tuple, a cell, an int, a str, an
//! instance of a class with __slots__ and no __weakref__). Such a pin counts
//! its reference for as long as whatever holds the pin lives. Weakening
//! drops a counted reference, which runs the object's finalizer if it was
//! the last one: not to be done while a region is being entered.

use pyo3::ffi;
use pyo3::ffi::PyObject;
use pyo3::prelude::*;

use super::ffi::tr_instance_dict;

enum Held {
    Counted(Py<PyAny>),
    // A weak reference to the object.
    Weak(Py<PyAny>),
    // A weak reference to the object whose __dict__ is pinned.
    DictOf(Py<PyAny>),
    // The same, and the address of the __dict__ the object had.
    DictAt(Py<PyAny>, usize),
}

pub struct Pin(Held);

// What a weak reference is of (borrowed), NULL once that died. As CPython's
// PyWeakref_GET_OBJECT, without the call: every entry of a region asks.
#[inline]
unsafe fn referent(weak: &Py<PyAny>) -> *mut PyObject {
    let o = (*(weak.as_ptr() as *mut ffi::PyWeakReference)).wr_object;
    if o == ffi::Py_None() || ffi::Py_REFCNT(o) <= 0 { std::ptr::null_mut() } else { o }
}

impl Pin {
    pub unsafe fn counted(o: *mut PyObject) -> Pin {
        Pin(Held::Counted(Py::from_borrowed_ptr(Python::assume_attached(), o)))
    }

    pub unsafe fn weak(o: *mut PyObject) -> Pin {
        let mut p = Pin::counted(o);
        p.weaken();
        p
    }

    // From now on by a weak reference, where the object's class takes one.
    pub fn weaken(&mut self) {
        let Held::Counted(o) = &self.0 else { return };
        unsafe {
            if (*ffi::Py_TYPE(o.as_ptr())).tp_weaklistoffset == 0 {
                return;
            }
            let weak = ffi::PyWeakref_NewRef(o.as_ptr(), std::ptr::null_mut());
            if weak.is_null() {
                ffi::PyErr_Clear();
                return;
            }
            self.0 = Held::Weak(Py::from_owned_ptr(Python::assume_attached(), weak));
        }
    }

    // For a counted pin of the __dict__ `owner` has, with `owner` held
    // weakly: from now on the dict is found through the owner. Whether it
    // is.
    pub fn through(&mut self, owner: &Pin) -> bool {
        let (Held::Counted(dict), Held::Weak(weak)) = (&self.0, &owner.0) else { return false };
        unsafe {
            let o = referent(weak);
            if o.is_null() || tr_instance_dict(o) != dict.as_ptr() {
                return false;
            }
            self.0 = Held::DictOf(weak.clone_ref(Python::assume_attached()));
        }
        true
    }

    // For a counted pin of the __dict__ `owner` has, with `owner` held
    // weakly: from now on the pin counts nothing and says which dict the
    // owner has, by the dict's address. Whether it is.
    //
    // The one place where an address stands for an object, compared only
    // while the owner lives. It is exact: as long as the dict that was
    // pinned is alive no other object is at its address, so the owner's
    // dict at that address is that dict; and once it died nothing reads it
    // any more (whatever shows a dict to a reader, a view, a mappingproxy,
    // a bound method of it, keeps it alive), so a dict the owner then has
    // at that address is the only dict a reader can come by, and what was
    // read is compared in it. Everything else is `is` / `was`.
    pub fn at(&mut self, owner: &Pin) -> bool {
        let (Held::Counted(dict), Held::Weak(weak)) = (&self.0, &owner.0) else { return false };
        unsafe {
            let o = referent(weak);
            if o.is_null() || tr_instance_dict(o) != dict.as_ptr() {
                return false;
            }
            self.0 = Held::DictAt(weak.clone_ref(Python::assume_attached()), dict.as_ptr() as usize);
        }
        true
    }

    // None: the object died. Otherwise the object (borrowed), which for a
    // dict found through its owner is NULL while the owner has none, and for
    // one pinned `at` its owner also while the owner has another.
    #[inline]
    pub fn live(&self) -> Option<*mut PyObject> {
        unsafe {
            match &self.0 {
                Held::Counted(o) => Some(o.as_ptr()),
                Held::Weak(weak) => Some(referent(weak)).filter(|o| !o.is_null()),
                Held::DictOf(weak) => Some(referent(weak)).filter(|o| !o.is_null()).map(|o| tr_instance_dict(o)),
                Held::DictAt(weak, address) => Some(referent(weak)).filter(|o| !o.is_null()).map(|o| Some(tr_instance_dict(o)).filter(|d| *d as usize == *address).unwrap_or(std::ptr::null_mut())),
            }
        }
    }

    #[inline]
    pub fn as_ptr(&self) -> *mut PyObject {
        self.live().unwrap_or(std::ptr::null_mut())
    }

    #[inline]
    pub fn dead(&self) -> bool {
        self.live().is_none()
    }

    // Whether `now` (NULL: nothing) is the pinned object.
    #[inline]
    pub fn is(&self, now: *mut PyObject) -> bool {
        !now.is_null() && self.as_ptr() == now
    }

    // The reference it counts, if it counts one.
    pub fn counts(&self) -> Option<&Py<PyAny>> {
        match &self.0 {
            Held::Counted(o) => Some(o),
            _ => None,
        }
    }
}

// Whether `now` (NULL: nothing) is what `was` pinned (None: nothing).
#[inline]
pub fn was(was: &Option<Pin>, now: *mut PyObject) -> bool {
    match was {
        None => now.is_null(),
        Some(pin) => pin.is(now),
    }
}
