//! What the recorded writes of a traced run left where the run had read:
//! the one answer to "does a write of the run explain that this read is no
//! longer as it was read".
//!
//! Given: by writes.rs, as it records a write (and takes one back that
//! raised), and by the executor for a store into a cell: what was written
//! to (its address), how, the key, the value, and by how many items the
//! target grew. By reads.rs: for a read, how many writes the log had when
//! the read was made (`after`), and what was read.
//!
//! Guarantees: an answer is made of the writes recorded after the read
//! alone, in order: what the last of them to that very item (key, index,
//! slot, cell) put there; what a dict, a list or a set read whole holds once
//! each of them is made on what was read; by how many the target grew.
//! A write is to the item a read found only if its key is the read's: the
//! same object, or both of exactly one class (str or int) and the same
//! value. A write the log cannot place says that nothing explains the read
//! (`Since::Unplaced`), whatever is there: one under a key that `==` may
//! call the read's (see `key`), one to a list the run took an item out of
//! (what was behind it moved, for a read before the deletion and for one
//! after it), a way of writing that is none of the holder's kind. Keys and
//! values are counted until the log is dropped, so a pointer handed out is
//! of the object that was written; of the stores into one cell only the
//! last is counted (a cell holds one value, and the journal of writes.rs
//! has no record of a cell: the log alone would keep what an earlier
//! store put there alive behind the store that replaced it). No Python
//! code runs.
//!
//! May not assume: that the log has every change (a finalizer, C code, a
//! callee the optimizer did not look into write without a record): the
//! caller compares the answer with what is there now, and what differs is
//! not explained. That a write to a cell before the read was to the same
//! object: an address is used again, and only from the read on does the
//! read's entry hold the cell. (The record writes.rs keeps of a write
//! holds its target until the body is over: the list, the set or the dict
//! of an item store; for an attribute store the object, whose `__dict__`
//! is the target here.) That the keys of two classes are two items:
//! `d[True]`, `d[1.0]` and `d[1]` are one.

use std::collections::{HashMap, HashSet};

use pyo3::ffi::{self, PyObject};
use pyo3::prelude::*;

use super::exact;

#[derive(Clone, Copy, PartialEq)]
pub enum How {
    // Item `key` of a dict: `value`, or nothing.
    Item,
    // Slot `key` of an object: `value`, or nothing. No question about an
    // item takes one for an item, nor `slot` an item for one: an instance
    // of a dict subclass with __slots__ is the target of both under one
    // name.
    Slot,
    // The item at this index of a list, from the front: `value`, or taken out.
    Index(isize),
    // `value` at the end of a list.
    Append,
    // `value` into a set, or out of it.
    Add,
    Discard,
    // What a cell holds: `value`.
    Cell,
}

struct Wrote {
    target: usize,
    // writes.rs's token of the write (0: none, a cell's).
    token: usize,
    how: How,
    key: Option<Py<PyAny>>,
    value: Option<Py<PyAny>>,
    grew: isize,
}

// What the run's recorded writes since a read say of the item read.
pub enum Since {
    // None was to it: it is to be as it was read.
    Nothing,
    // The last to it put this there (NULL: took it out).
    Put(*mut PyObject),
    // One may have been to it and the log cannot place it: nothing
    // explains the read.
    Unplaced,
}

// Whether a write under the key `wrote` is to the dict item a read found
// under `read`.
enum Key {
    Same,
    Other,
    // `==` may say so, and is not asked: it is code of the key's class.
    Maybe,
}

// Decided by the exact classes alone. An item is read under an exact str or
// int (reads.rs records no other): a str is the key of a str of its
// content only, an int that of an int of its number, and a str is no
// number. A bool or a float that is the int's number is the int's item to
// the dict and another value to everything that compares exactly: Maybe.
// So is a key of any other class (a subclass of str or int, an enum, a
// tuple, an object with an `__eq__`), which this has no rule for.
unsafe fn key(wrote: *mut PyObject, read: *mut PyObject) -> Key {
    if wrote == read {
        return Key::Same;
    }
    let (w, r) = (ffi::Py_TYPE(wrote), ffi::Py_TYPE(read));
    let (int, float, text) = (&raw mut ffi::PyLong_Type, &raw mut ffi::PyFloat_Type, &raw mut ffi::PyUnicode_Type);
    if r != int && r != text {
        return Key::Maybe;
    }
    if w == r {
        return if exact::same(wrote, read) { Key::Same } else { Key::Other };
    }
    let number = w == int || w == float || w == &raw mut ffi::PyBool_Type;
    if r == text || w == text {
        return if number || w == text { Key::Other } else { Key::Maybe };
    }
    if !number {
        return Key::Maybe;
    }
    // A bool or a float against an int. (An int too large for a double is
    // no float's number; rounding can only make two numbers one: Maybe.)
    let a = if w == float { ffi::PyFloat_AS_DOUBLE(wrote) } else { ffi::PyLong_AsDouble(wrote) };
    let b = ffi::PyLong_AsDouble(read);
    if !ffi::PyErr_Occurred().is_null() {
        ffi::PyErr_Clear();
        return Key::Maybe;
    }
    if a == b { Key::Maybe } else { Key::Other }
}

#[derive(Default)]
pub struct Log {
    all: Vec<Wrote>,
    // What was written to: most reads are of something that was not.
    targets: HashSet<usize>,
    // The last store into each cell, by the cell's address.
    cells: HashMap<usize, usize>,
}

fn ptr(o: &Option<Py<PyAny>>) -> *mut PyObject {
    o.as_ref().map_or(std::ptr::null_mut(), |o| o.as_ptr())
}

impl Log {
    pub fn len(&self) -> usize {
        self.all.len()
    }

    // `key` and `value` borrowed, NULL for none. Returns what an earlier
    // store into the same cell put there: the cell has let go of it, no
    // question is about it any more, and the log is not to keep it alive
    // (no other record holds what was stored into a cell). The caller
    // drops it where no record is borrowed: a finalizer may run.
    #[must_use]
    pub unsafe fn add(&mut self, target: *mut PyObject, token: usize, how: How, key: *mut PyObject, value: *mut PyObject, grew: isize) -> Option<Py<PyAny>> {
        let own = |o: *mut PyObject| (!o.is_null()).then(|| Py::from_borrowed_ptr(Python::assume_attached(), o));
        self.targets.insert(target as usize);
        let over = if how == How::Cell { self.cells.insert(target as usize, self.all.len()).and_then(|n| self.all[n].value.take()) } else { None };
        self.all.push(Wrote { target: target as usize, token, how, key: own(key), value: own(value), grew });
        over
    }

    // The write recorded last, as `token`, raised: it is none.
    pub fn take_back(&mut self, token: usize) {
        if token != 0 && self.all.last().is_some_and(|w| w.token == token) {
            self.all.pop();
        }
    }

    // The writes since `after` to what `target` holds as a container or a
    // cell.
    fn of(&self, target: *mut PyObject, after: usize) -> impl Iterator<Item = &Wrote> {
        let from = if self.targets.contains(&(target as usize)) { after } else { self.all.len() };
        self.all.iter().skip(from).filter(move |w| w.target == target as usize && w.how != How::Slot)
    }

    // What the writes since `after` say of item `key` of the dict `target`.
    pub unsafe fn item(&self, target: *mut PyObject, after: usize, key: *mut PyObject) -> Since {
        let mut last = Since::Nothing;
        for w in self.of(target, after) {
            if w.how != How::Item {
                return Since::Unplaced;
            }
            match self::key(ptr(&w.key), key) {
                Key::Same => last = Since::Put(ptr(&w.value)),
                Key::Other => {}
                Key::Maybe => return Since::Unplaced,
            }
        }
        last
    }

    // What the writes since `after` say of the item at index `at` (from
    // the front) of the list `target`. A deletion from the list, whenever
    // the run made it, moved what was behind it: an index read before it
    // is of another item now, and one read after it is not of the item a
    // later call finds there.
    pub fn index(&self, target: *mut PyObject, after: usize, at: isize) -> Since {
        if !self.targets.contains(&(target as usize)) {
            return Since::Nothing;
        }
        let mut last = Since::Nothing;
        for (n, w) in self.all.iter().enumerate().filter(|(_, w)| w.target == target as usize && w.how != How::Slot) {
            match (w.how, &w.value) {
                (How::Index(_), None) => return Since::Unplaced,
                (How::Index(i), Some(v)) if n >= after && i == at => last = Since::Put(v.as_ptr()),
                (How::Index(_), Some(_)) | (How::Append, _) => {}
                _ => return Since::Unplaced,
            }
        }
        last
    }

    // What the writes since `after` say of slot `name` of the object
    // `target`. A slot is named by an exact str; a write under a name of
    // any other class cannot be placed.
    pub unsafe fn slot(&self, target: *mut PyObject, after: usize, name: *mut PyObject) -> Since {
        if !self.targets.contains(&(target as usize)) {
            return Since::Nothing;
        }
        let text = &raw mut ffi::PyUnicode_Type;
        let mut last = Since::Nothing;
        for w in self.all.iter().skip(after).filter(|w| w.target == target as usize && w.how == How::Slot) {
            let wrote = ptr(&w.key);
            if wrote.is_null() || ffi::Py_TYPE(wrote) != text || ffi::Py_TYPE(name) != text {
                return Since::Unplaced;
            }
            if exact::same(wrote, name) {
                last = Since::Put(ptr(&w.value));
            }
        }
        last
    }

    // What the last store into the cell `target` put there.
    pub fn cell(&self, target: *mut PyObject, after: usize) -> Option<*mut PyObject> {
        self.of(target, after).filter(|w| w.how == How::Cell).last().map(|w| ptr(&w.value))
    }

    // By how many items `target` grew.
    pub fn grew(&self, target: *mut PyObject, after: usize) -> isize {
        self.of(target, after).map(|w| w.grew).sum()
    }

    // The keys and values, pairwise and in order, of the dict `target` that
    // held `all`: a key that is there keeps its place, a new one goes last.
    pub unsafe fn dict(&self, target: *mut PyObject, after: usize, mut all: Vec<*mut PyObject>) -> Option<Vec<*mut PyObject>> {
        for w in self.of(target, after) {
            if w.how != How::Item {
                return None;
            }
            let key = ptr(&w.key);
            // (A key that `==` may call one that is there, and exact::same
            // does not, is added as a new one: the dict then holds fewer
            // items than this makes of it, and the caller's comparison
            // fails.)
            let at = (0..all.len()).step_by(2).find(|i| exact::same(all[*i], key));
            match (at, &w.value) {
                (Some(i), Some(v)) => all[i + 1] = v.as_ptr(),
                (Some(i), None) => drop(all.drain(i..i + 2)),
                (None, Some(v)) => all.extend([key, v.as_ptr()]),
                // A key that was not there when the dict was read.
                (None, None) => return None,
            }
        }
        Some(all)
    }

    // The items, in order, of the list `target` that held `all`.
    pub fn list(&self, target: *mut PyObject, after: usize, mut all: Vec<*mut PyObject>) -> Option<Vec<*mut PyObject>> {
        for w in self.of(target, after) {
            match (w.how, &w.value) {
                (How::Index(i), v) if i >= 0 && (i as usize) < all.len() => match v {
                    Some(v) => all[i as usize] = v.as_ptr(),
                    None => drop(all.remove(i as usize)),
                },
                (How::Append, Some(v)) => all.push(v.as_ptr()),
                _ => return None,
            }
        }
        Some(all)
    }

    // The members, in no order, of the set `target` that held `all`.
    pub unsafe fn set(&self, target: *mut PyObject, after: usize, mut all: Vec<*mut PyObject>) -> Option<Vec<*mut PyObject>> {
        for w in self.of(target, after) {
            let v = ptr(&w.value);
            let at = all.iter().position(|x| exact::same(*x, v));
            match (w.how, at) {
                (How::Add, None) => all.push(v),
                (How::Discard, Some(i)) => drop(all.swap_remove(i)),
                (How::Add, Some(_)) | (How::Discard, None) => {}
                _ => return None,
            }
        }
        Some(all)
    }
}
