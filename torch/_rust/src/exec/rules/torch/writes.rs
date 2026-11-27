// What a call with a row of the torch family writes (see the head of
// rules/torch.rs): the tensors (writes_of), the row's `before`
// (writing_call), the hand-over to today's trace (hand_over), and whether
// a member only computes (only_computes).

use super::*;

// The tensors a call with the row `row` writes: for each operand the row
// names (Write::Operands), the operand, or the tensors an exact list or
// tuple operand holds; and whether a schema declares every one of those
// writes. None: the row writes no operand. Err: an operand the row names is
// not among the call's.
pub(super) unsafe fn writes_of(row: &Row, what: &What, site: &Site) -> Result<Option<(Vec<*mut PyObject>, bool)>, String> {
    let (What::Callee(f), Write::Operands(written)) = (*what, row.writes) else { return Ok(None) };
    let placed = placed(&operands(f, site)?, site.kwnames)?;
    let mut tensors = vec![];
    for w in written {
        let Some((_, o)) = placed.iter().find(|(place, _)| place == w.at) else { return Err(format!("it writes operand {} of the calls measured, which this call does not hand it in that place", w.at)) };
        let tp = ffi::Py_TYPE(*o);
        let items: Vec<*mut PyObject> = if tp == &raw mut ffi::PyTuple_Type {
            (0..ffi::PyTuple_GET_SIZE(*o)).map(|i| ffi::PyTuple_GET_ITEM(*o, i)).collect()
        } else if tp == &raw mut ffi::PyList_Type {
            (0..ffi::PyList_GET_SIZE(*o)).map(|i| ffi::PyList_GET_ITEM(*o, i)).collect()
        } else {
            vec![*o]
        };
        tensors.extend(items.into_iter().filter(|x| !x.is_null() && Group::PlainTensor.holds(ffi::Py_TYPE(*x))));
    }
    Ok(Some((tensors, written.iter().all(|w| w.declared))))
}

// The values a row that writes operands is for (plain_call, and formed_call
// if the row is for one form), with every operand the row names among the
// call's. Records nothing and tells nobody: whoever runs the call asks what
// it writes (writes_of; today's trace: hand_over).
pub(super) unsafe fn writing_call(row: &Row, what: &What, site: &Site) -> Result<(), String> {
    if row.form.is_empty() { plain_call(row, what, site)? } else { formed_call(row, what, site)? }
    writes_of(row, what, site).map(|_| ())
}

// Hands the tensors that a call with the row `row` writes to the trace that
// is running, before the call runs (covered::step, for the trace of today's
// engine; not a part of the license: the recorder asks the same rows). The
// trace's record of its writes is told of each (torch._rust_inductor
// ._row_writes, undo.py: Python, outside the executor): given up, the trace
// undoes its writes, the last first; and it no longer takes its body for
// one that leaves nothing behind. Err if the trace did not take them: the
// call then runs under no trace.
pub unsafe fn hand_over(row: &Row, what: &What, site: &Site) -> Result<(), String> {
    let Some((tensors, declared)) = writes_of(row, what, site)? else { return Ok(()) };
    let py = Python::assume_attached();
    let handed = pyo3::types::PyList::new(py, tensors.into_iter().map(|t| Bound::from_borrowed_ptr(py, t))).map_err(|_| "the tensors it writes could not be listed".to_string())?;
    match scalars::tell(py, "_row_writes", vec![handed.into_any(), pyo3::types::PyBool::new(py, declared).to_owned().into_any()]) {
        Some(_) => Ok(()),
        None => Err("the trace did not take the tensors it writes".into()),
    }
}

// Whether every row of the member `f` calls says that the call only
// computes: it writes no operand in any form of it that has a row. False
// for a callee that is no member with a row.
pub unsafe fn only_computes(f: *mut PyObject) -> bool {
    matches!(member(f).map(|m| m.taken), Some(Taken::Rows(rows)) if rows.iter().all(|r| matches!(r.writes, Write::None)))
}
