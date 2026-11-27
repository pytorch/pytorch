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
//! cell) put there; what a dict, a list or a set read whole holds once
//! each of them is made on what was read; by how many the target grew.
//! A key is the read's key only if it is the same object or the same exact
//! value (exec/exact.rs). Keys and values are counted until the log is
//! dropped, so a pointer handed out is of the object that was written. No
//! Python code runs.
//!
//! May not assume: that the log has every change (a finalizer, C code, a
//! callee the optimizer did not look into write without a record): the
//! caller compares the answer with what is there now, and what differs is
//! not explained. That a write before the read was to the same object: an
//! address is used again, and only from the read on does the read's entry
//! hold the object. That a write it cannot place (a deletion from a list,
//! which moves what is behind it; a way of writing that is none of the
//! holder's kind) leaves a read item alone: the answer is then None.

use std::collections::HashSet;

use pyo3::ffi::PyObject;
use pyo3::prelude::*;

use super::exact;

#[derive(Clone, Copy, PartialEq)]
pub enum How {
    // Item `key` of a dict: `value`, or nothing.
    Item,
    // Slot `key` of an object: `value`, or nothing. No question here is
    // about a slot yet (a slot's read is compared when the trace is over,
    // see reads.rs), and none takes one for an item: an instance of a dict
    // subclass with __slots__ is the target of both under one name.
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

#[derive(Default)]
pub struct Log {
    all: Vec<Wrote>,
    // What was written to: most reads are of something that was not.
    targets: HashSet<usize>,
}

fn ptr(o: &Option<Py<PyAny>>) -> *mut PyObject {
    o.as_ref().map_or(std::ptr::null_mut(), |o| o.as_ptr())
}

impl Log {
    pub fn len(&self) -> usize {
        self.all.len()
    }

    // `key` and `value` borrowed, NULL for none.
    pub unsafe fn add(&mut self, target: *mut PyObject, token: usize, how: How, key: *mut PyObject, value: *mut PyObject, grew: isize) {
        let own = |o: *mut PyObject| (!o.is_null()).then(|| Py::from_borrowed_ptr(Python::assume_attached(), o));
        self.targets.insert(target as usize);
        self.all.push(Wrote { target: target as usize, token, how, key: own(key), value: own(value), grew });
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

    // What the last write to item `key` of the dict `target` put there,
    // NULL for nothing; None if there is none.
    pub unsafe fn item(&self, target: *mut PyObject, after: usize, key: *mut PyObject) -> Option<*mut PyObject> {
        let mut last = None;
        for w in self.of(target, after) {
            if w.how != How::Item {
                return None;
            }
            if exact::same(ptr(&w.key), key) {
                last = Some(ptr(&w.value));
            }
        }
        last
    }

    // What the last write to index `at` of the list `target` put there;
    // None if there is none, or an item was taken out of the list.
    pub fn index(&self, target: *mut PyObject, after: usize, at: isize) -> Option<*mut PyObject> {
        let mut last = None;
        for w in self.of(target, after) {
            match (w.how, &w.value) {
                (How::Index(i), Some(v)) if i == at => last = Some(v.as_ptr()),
                (How::Index(_), Some(_)) | (How::Append, _) => {}
                _ => return None,
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
