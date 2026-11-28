// What a trace lets go of before its gate, kept until the gate.
//
// While a region's body is traced the tracer's dispatch modes are on and its
// stand-ins sit in the program's containers, and they stay so while the
// tracer turns the run into a region (torch._rust_inductor: the code behind
// the body). An object of the program's that died there would run its
// finalizer as code of the program in a state that is not the program's. So
// the trace drops none of its records there: it PARKS them, and the gate of
// the trace (foreign.rs), behind the modes, drops them in the order parked.
//
// The trace also judges which objects nothing but its own records refers to
// (a write to one is none a region makes again, writes.rs; what was read
// from one is no check, reads.rs). It used to drop a record and read the
// reference counts that left. Here a parked record's references are counted
// as let go of without being let go of: `refs` is the count an object would
// have had the parked records been dropped, and `dead` says an object would
// have died of it.
//
// What it is given
//   - by writes.rs and reads.rs: every record they would have dropped
//     between the body's last instruction and the gate (`park`), each of a
//     type that says which references it counts (`Holds`);
//   - by foreign.rs: the trace's begin and its gate (`begin`, `drain`).
// What it guarantees
//   - a parked record is dropped by `drain` and nowhere else, the first
//     parked first, one at a time with nothing of this module borrowed (a
//     finalizer may run and may start a trace of its own);
//   - `refs(o)` is Py_REFCNT(o) less the references to `o` that parked
//     records count, and less those of objects that would have died with
//     them (an instance and its __dict__, a cell and what it holds), found
//     by the type's own tp_traverse: C code that runs no code of the
//     program's. It is never less than what dropping the records would have
//     left: a reference this module cannot see stays counted.
//   - `dead(o)`: every reference to `o` is one of those.
// What it may not assume, and where it then errs
//   - that a type's tp_traverse names every reference its instances own: a
//     type without one, and a tensor (whose traverse depends on who else
//     owns the tensor underneath), is taken to own nothing. The object it
//     alone holds then passes for alive: a write to it stays a write, what
//     was read from it stays a check.
//   - that the counts stay what they were: `settle` is asked before every
//     judgment, and a count that says more references are parked than an
//     object has makes every later answer "alive".

use std::any::Any;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};

use pyo3::ffi;
use pyo3::ffi::PyObject;
use pyo3::prelude::*;

// The references a record counts, each reported once per reference.
pub trait Holds {
    fn each(&self, f: &mut dyn FnMut(*mut PyObject));
}

impl Holds for Py<PyAny> {
    fn each(&self, f: &mut dyn FnMut(*mut PyObject)) {
        f(self.as_ptr());
    }
}

impl<T: Holds> Holds for Option<T> {
    fn each(&self, f: &mut dyn FnMut(*mut PyObject)) {
        if let Some(x) = self {
            x.each(f);
        }
    }
}

impl<T: Holds> Holds for Vec<T> {
    fn each(&self, f: &mut dyn FnMut(*mut PyObject)) {
        self.iter().for_each(|x| x.each(f));
    }
}

impl<A: Holds, B: Holds> Holds for (A, B) {
    fn each(&self, f: &mut dyn FnMut(*mut PyObject)) {
        self.0.each(f);
        self.1.each(f);
    }
}

// Parks what it holds when it goes out of scope, however that is.
pub struct Parks<T: Holds + Default + Any>(pub T);

impl<T: Holds + Default + Any> Parks<T> {
    // What it holds is handed on, not parked.
    pub fn keep(mut self) -> T {
        std::mem::take(&mut self.0)
    }
}

impl<T: Holds + Default + Any> std::ops::Deref for Parks<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T: Holds + Default + Any> std::ops::DerefMut for Parks<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

impl<T: Holds + Default + Any> Drop for Parks<T> {
    fn drop(&mut self) {
        park(std::mem::take(&mut self.0));
    }
}

// Takes the items of `v` that `gone` names out of it, in order.
pub fn part<T>(v: &mut Vec<T>, gone: impl Fn(&T) -> bool) -> Vec<T> {
    let (out, stay): (Vec<T>, Vec<T>) = std::mem::take(v).into_iter().partition(|x| gone(x));
    *v = stay;
    out
}

#[derive(Default)]
struct Ledger {
    // By object: how many references to it are parked (counted by a parked
    // record, or by an object that is `freed`).
    parked: HashMap<usize, isize>,
    // The objects every reference to which is parked.
    freed: HashSet<usize>,
    // A count said more than an object has: nothing is judged from here on.
    broken: bool,
}

thread_local! {
    static RECORDS: RefCell<VecDeque<Box<dyn Any>>> = const { RefCell::new(VecDeque::new()) };
    static LEDGER: RefCell<Ledger> = RefCell::new(Ledger::default());
}

unsafe extern "C" fn visit(child: *mut PyObject, arg: *mut std::ffi::c_void) -> std::ffi::c_int {
    if !child.is_null() {
        (*(arg as *mut Vec<*mut PyObject>)).push(child);
    }
    0
}

// The references `o` owns, as its type's tp_traverse names them. None for a
// type without one, and for a tensor.
unsafe fn owned(o: *mut PyObject, out: &mut Vec<*mut PyObject>) {
    let tp = ffi::Py_TYPE(o);
    if ffi::PyType_HasFeature(tp, ffi::Py_TPFLAGS_HAVE_GC) == 0 || super::reads::tensor(o) {
        return;
    }
    if let Some(traverse) = (*tp).tp_traverse {
        traverse(o, visit, out as *mut Vec<*mut PyObject> as *mut std::ffi::c_void);
    }
}

impl Ledger {
    // One more reference to `o` is parked. An object all of whose references
    // are, would have died: what it owns is parked with it.
    unsafe fn release(&mut self, o: *mut PyObject) {
        let mut todo = vec![o];
        while let Some(o) = todo.pop() {
            let n = self.parked.entry(o as usize).or_insert(0);
            *n += 1;
            let (n, has) = (*n, ffi::Py_REFCNT(o));
            if n > has {
                self.broken = true;
            } else if n == has && self.freed.insert(o as usize) {
                owned(o, &mut todo);
            }
        }
    }

    // An object whose other references went since (the executor or the
    // tracer let go of them) has only parked ones now.
    unsafe fn settle(&mut self) {
        loop {
            let late: Vec<usize> = self.parked.iter().filter(|(o, n)| **n == ffi::Py_REFCNT(**o as *mut PyObject) && !self.freed.contains(*o)).map(|(o, _)| *o).collect();
            if late.is_empty() {
                return;
            }
            for o in late {
                if self.freed.insert(o) {
                    let mut todo = vec![];
                    owned(o as *mut PyObject, &mut todo);
                    todo.into_iter().for_each(|c| self.release(c));
                }
            }
        }
    }
}

// A trace begins on this thread (under foreign.rs's `Releasing`: what a
// trace that never reached its gate left is dropped here).
pub fn begin() {
    let _ = drain();
}

// Keeps `record` until the gate; the references it counts are parked.
pub fn park<T: Holds + Any>(record: T) {
    LEDGER.with_borrow_mut(|l| record.each(&mut |o| unsafe { l.release(o) }));
    RECORDS.with_borrow_mut(|r| r.push_back(Box::new(record)));
}

// Brings the ledger up to the reference counts as they are: asked before a
// round of judgments.
pub fn settle() {
    LEDGER.with_borrow_mut(|l| unsafe { l.settle() });
}

// The references to `o` (alive) that are not parked.
pub unsafe fn refs(o: *mut PyObject) -> isize {
    LEDGER.with_borrow(|l| if l.broken { isize::MAX } else { ffi::Py_REFCNT(o) - l.parked.get(&(o as usize)).copied().unwrap_or(0) })
}

// Whether every reference to `o` (alive) is parked: it dies at the gate.
pub fn dead(o: *mut PyObject) -> bool {
    LEDGER.with_borrow(|l| !l.broken && l.freed.contains(&(o as usize)))
}

// The gate: drops what was parked, in the order parked. Under the caller's
// `Releasing` (foreign.rs). False: the judgments made of the ledger do not
// hold. An object is in `freed` because every reference to it was parked
// when it was judged, and it stays there: nothing withdraws it. Between a
// judgment and the gate no code of the program's runs (that is what the
// parking is for), and the tracer's own code takes no reference to an
// object it was not handed; should something refer to such an object
// again all the same (its count is no longer the parked one), or a count
// have said more than an object has, the trace is no region.
pub fn drain() -> bool {
    // The ledger first: its addresses are of objects the records keep alive.
    let ledger = LEDGER.take();
    let sound = !ledger.broken && ledger.freed.iter().all(|o| unsafe { ffi::Py_REFCNT(*o as *mut PyObject) } == ledger.parked[o]);
    while let Some(record) = RECORDS.with_borrow_mut(VecDeque::pop_front) {
        drop(record);
    }
    sound
}
