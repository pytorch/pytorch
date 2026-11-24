//! What the VM asks of a run before a callable may stand for its path.
//!
//! Two things that eager knows only by running: whether the sizes of what an
//! operator that is made of others gives follow from the sizes of what it is
//! given (`sized`), and whether a tensor that autograd saved for a
//! derivative was written to in place behind it (`saved_changed`). Both ask
//! eager through its Python API: the operator is run once more on tensors
//! that hold nothing, and the nodes of autograd are asked for what they
//! saved.

use crate::backward::{is_tensor, site_call, torch, GradMode};
use crate::pyffi::Obj;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

/// The sizes of a tensor and, with `dtypes`, its dtype; those of the items
/// of a list or tuple; or nothing for what is neither.
#[derive(PartialEq)]
enum Sizes {
    Tensor(Vec<i64>, usize),
    Several(Vec<Option<Sizes>>),
}

fn sizes(value: &Bound<'_, PyAny>, dtypes: bool) -> PyResult<Option<Sizes>> {
    if is_tensor(value) {
        let dtype = if dtypes { value.getattr("dtype")?.as_ptr() as usize } else { 0 };
        return Ok(Some(Sizes::Tensor(value.getattr("shape")?.extract()?, dtype)));
    }
    if value.is_instance_of::<PyList>() || value.is_instance_of::<PyTuple>() {
        return Ok(Some(Sizes::Several(value.try_iter()?.map(|item| sizes(&item?, dtypes)).collect::<PyResult<_>>()?)));
    }
    Ok(None)
}

/// A tensor of the same sizes and dtype that holds nothing (on the `meta`
/// device); a list or tuple with such tensors; anything else as it is.
fn hollow<'py>(py: Python<'py>, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if is_tensor(value) {
        let kwargs = PyDict::new(py);
        kwargs.set_item("dtype", value.getattr("dtype")?)?;
        kwargs.set_item("device", "meta")?;
        return torch(py)?.call_method("empty", (value.getattr("shape")?,), Some(&kwargs));
    }
    let kind = value.get_type();
    if kind.is(&py.get_type::<PyList>()) || kind.is(&py.get_type::<PyTuple>()) {
        let items: Vec<Bound<'py, PyAny>> = value.try_iter()?.map(|item| hollow(py, &item?)).collect::<PyResult<_>>()?;
        return Ok(if kind.is(&py.get_type::<PyList>()) { PyList::new(py, items)?.into_any() } else { PyTuple::new(py, items)?.into_any() });
    }
    Ok(value.clone())
}

fn asked<'py>(py: Python<'py>, kind: &str, head: &Bound<'py, PyAny>, operands: &Bound<'py, PyAny>, kwnames: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let given: Vec<Bound<'py, PyAny>> = operands.try_iter()?.map(|item| hollow(py, &item?)).collect::<PyResult<_>>()?;
    let _mode = GradMode::set(py, false)?;
    match kind {
        "site" => site_call(py, head, &|at| given.get(at).cloned().ok_or_else(|| PyRuntimeError::new_err("no such operand"))),
        "item" => given[0].get_item(&given[1]),
        _ => {
            let names: Vec<String> = if kwnames.is_none() { Vec::new() } else { kwnames.try_iter()?.map(|name| name?.extract()).collect::<PyResult<_>>()? };
            let mut positional = given;
            let named = positional.split_off(positional.len() - names.len());
            let kwargs = PyDict::new(py);
            for (name, value) in names.iter().zip(named) {
                kwargs.set_item(name, value)?;
            }
            let function = if kind == "method" { positional.remove(0).getattr(head.extract::<String>()?.as_str())? } else { head.clone() };
            function.call(PyTuple::new(py, positional)?, Some(&kwargs))
        }
    }
}

/// Whether the sizes of what an operation gave follow from the sizes of the
/// tensors it was given and from its other arguments. The operation is run
/// once more on tensors of those sizes and dtypes that hold nothing: one
/// that has to look at what a tensor holds to size its result (`nonzero`
/// inside `argwhere`, a count that is a tensor) cannot run there, and
/// neither can anything else this does not know how to ask, for which the
/// answer is no. `kind` is "site" for a kernel call (`head` describes it),
/// "call" for a call of the function `head`, "method" for a call of the
/// method of that name of the first operand, "item" for a subscript.
///
/// What it gives must have the dtypes of what the operation gave as well,
/// unless the thread is under autocast: autocast casts on the device of the
/// tensors of a call, and not where tensors hold nothing.
pub unsafe fn sized(kind: &str, head: Obj, operands: Obj, kwnames: Obj, result: Obj) -> bool {
    let py = Python::assume_attached();
    let of = |held: Obj| Bound::<PyAny>::from_borrowed_ptr(py, held);
    let cast = torch(py).and_then(|torch| torch.getattr("_C")?.call_method0("_is_any_autocast_enabled")?.extract::<bool>());
    let Ok(cast) = cast else {
        return false;
    };
    let Ok(Some(wanted)) = sizes(&of(result), !cast) else {
        return false;
    };
    match asked(py, kind, &of(head), &of(operands), &of(kwnames)).and_then(|found| sizes(&found, !cast)) {
        Ok(found) => found == Some(wanted),
        Err(_) => false,
    }
}

/// The names under which the nodes of a class hold what they saved.
static SAVED: Mutex<Option<HashMap<usize, Vec<String>>>> = Mutex::new(None);
/// How many nodes of autograd a run is followed back through.
const HISTORY: usize = 4096;

/// The names under which a node of autograd holds what it saved, in the
/// order of the names.
pub(crate) fn saved_names(node: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    let kind = node.get_type();
    let known = SAVED.lock().unwrap().get_or_insert_with(HashMap::new).get(&(kind.as_ptr() as usize)).cloned();
    if let Some(names) = known {
        return Ok(names);
    }
    let listed: Vec<String> = kind.dir()?.extract()?;
    let names: Vec<String> = listed.into_iter().filter(|name| name.starts_with("_saved_")).collect();
    SAVED.lock().unwrap().get_or_insert_with(HashMap::new).insert(kind.as_ptr() as usize, names.clone());
    Ok(names)
}

fn changed(py: Python<'_>, results: &Bound<'_, PyAny>) -> PyResult<String> {
    if !torch(py)?.call_method0("is_grad_enabled")?.extract::<bool>()? {
        return Ok(String::new());
    }
    let mut left: Vec<Bound<'_, PyAny>> = Vec::new();
    for result in results.try_iter()? {
        // Asked of the getter of the class under every tensor; what is
        // no tensor has no node.
        let node = unsafe { crate::graph::node_of(result?.as_ptr()) };
        let node = if node.is_null() { None } else { Some(unsafe { Bound::from_owned_ptr(py, node) }) }.filter(|node| !node.is_none());
        left.extend(node);
    }
    let mut seen: HashSet<usize> = HashSet::new();
    // The nodes that were looked at are held: one that died could give its
    // place to another.
    let mut held: Vec<Bound<'_, PyAny>> = Vec::new();
    while let Some(node) = left.pop() {
        if !seen.insert(node.as_ptr() as usize) {
            continue;
        }
        if seen.len() > HISTORY {
            return Ok(format!("what autograd saved is not looked at behind {HISTORY} functions"));
        }
        for name in &saved_names(&node)? {
            if let Err(error) = node.getattr(name.as_str()) {
                if !error.is_instance_of::<PyRuntimeError>(py) {
                    return Err(error);
                }
                let text = unsafe { crate::graph::message(error.value(py).as_ptr()) };
                if text.contains("modified by an inplace operation") {
                    return Ok("a tensor that autograd saved is written in place".to_string());
                }
                let first: String = text.lines().next().unwrap_or("").chars().take(80).collect();
                return Ok(format!("what autograd saved cannot be looked at: {first}"));
            }
        }
        for edge in node.getattr("next_functions")?.try_iter()? {
            let before = edge?.get_item(0)?;
            if !before.is_none() {
                left.push(before);
            }
        }
        held.push(node);
    }
    Ok(String::new())
}

/// Why the run that gave `results` is not exported, as far as what autograd
/// saved goes; None if nothing stands against it. An operation that wrote
/// in place to a tensor that an earlier one saved for its derivative makes
/// eager raise when it goes backward; a graph that ran without a tape, or
/// whose writes a compiler made functional, would not. The nodes that the
/// results came from are gone through, and those before them (a tensor
/// that was written to has the node of the write, and the one that saved
/// it comes before that): where there are more of them than `HISTORY`, or
/// one does not let what it saved be looked at, the answer is that it
/// cannot be shown.
pub unsafe fn saved_changed(results: Obj) -> Option<String> {
    let py = Python::assume_attached();
    match changed(py, &Bound::<PyAny>::from_borrowed_ptr(py, results)) {
        Ok(why) => Some(why).filter(|why| !why.is_empty()),
        Err(_) => Some("what autograd saved could not be asked for".to_string()),
    }
}
