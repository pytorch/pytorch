// The torch family: torch's own C callees, by what calls of them were
// measured to do.
//
// torch_members.tsv (generated: torch/_rust/tools/gen_torch_members.py) has
// one line per C callee with the verdicts of a sweep that called it on many
// operands, on the CPU and on CUDA, and looked at memory: whether a Python
// frame was entered under the call (P), an operand's bytes or key changed
// (W, K), a generator moved (D), the results' count and keys were the same
// for other data in operands of the same keys (S), two calls gave the same
// bits (R). No name and no schema is asked. What a line means is verdict()
// below, and nothing else:
//   - a ROW for a member that only computes: P no (or only to show a
//     warning: those frames are no part of the row and come to the gate as
//     any frame), W, K, D no, S key, R yes, at least 5 returning samples on
//     each device. The row is for a call whose operands are plain (before):
//     exact Tensor / Parameter, the immutable classes, torch's constants,
//     exact tuples, lists and dicts of those, and no `out=`;
//   - a STOP, with the measured verdict, for a member whose results follow
//     data, whose two calls differ, or that was not measured enough: no
//     trace runs it, wherever it is called;
//   - a ROW PER FORM for a member that only its results following data
//     stands against, where the samples tell its forms apart: behind such a
//     member's line the file has a line per form of the operands (their
//     classes, in the call's own shape: positional and keyword), computed
//     from the samples of that form alone and read by the same verdict(). A
//     call of a form whose line is a row has the row; a call of any other
//     form, measured or not, stops the trace (formed_call);
//   - LATER, with the family that will take it, for a member that writes or
//     re-keys an operand (writes), draws, makes a tensor of no tensor or
//     follows ambient state (ambient), returns a fact or a value of a tensor
//     (facts), or enters Python code under the call (native: on CUDA seven
//     members go through the Python router of torch._native, whose frames
//     covered.rs still takes for part of the operator by their path): the
//     code that decided before decides.
// A torch callee that is not in the file is not this family's (Taken::No):
// the name filter of shown.rs still decides for it.
//
// What it is given
//   - by rules::rows, a callee as the instruction has it: the callable, a
//     builtin bound to its receiver, a functools.partial of one;
//   - by ready(), called where a function goes onto the executor and never
//     under a trace: the moment to resolve each member's name to its object
//     (attribute loads on torch's own namespaces, in Python, once).
// What it guarantees
//   - a callee is matched by identity: a builtin and a method descriptor by
//     the C definition they are made from (so `x.add` bound to a tensor is
//     `Tensor.add`), anything else by its address;
//   - under a trace rule() runs no code: it reads the table made before;
//   - with no table (ready() did not run, or failed) every callee is a stop.
// What it may not assume
//   - that a row is all the license a call has: the checks that were there
//     before the table still run behind it.

use std::collections::HashMap;
use std::ffi::CStr;
use std::sync::OnceLock;

use pyo3::ffi;
use pyo3::ffi::PyObject;
use pyo3::prelude::*;

use super::super::ffi::{tr_bound_function, tr_partial_func};
use super::super::{reads, scalars};
use super::{Enters, Family, Group, Read, Res, Row, Scalars, Site, Taken, What, Write};

pub const FAMILY: Family = Family { name: "torch", rule, table };

const WHY: &str = "Measured on the CPU and on CUDA: the call enters no Python frame, changes no operand's bytes or key, moves no generator, gives the same bits twice, and the count and keys of its results are the same for other data in operands of the same keys. It is run again on each call's operands.";
const WHY_WARN: &str = "As a member that only computes, but it can show a warning: the frames of that are not the row's, and are asked about as any frame under the body.";
const ROW: Row = Row { operands: &[], reads: &[], writes: Write::None, result: Res::Entry, enters: Enters::Never, scalars: Scalars::Operand, why: WHY, before: Some(plain_call) };
const WHY_FORM: &str = "As a member that only computes, for the forms of operands measured so, each on its own samples: in another form of operands the count or the sizes of its results follow data, or it was not measured, and a call of that form stops the trace.";
const ROW_FORM: Row = Row { operands: &[], reads: &[Read::Class], writes: Write::None, result: Res::Entry, enters: Enters::Never, scalars: Scalars::Operand, why: WHY_FORM, before: Some(formed_call) };
const ROW_WARN: Row = Row { operands: &[], reads: &[], writes: Write::None, result: Res::Entry, enters: Enters::Foreign, scalars: Scalars::Operand, why: WHY_WARN, before: Some(plain_call) };

// Resolves `names` as attributes from a loaded module on: `found[i]` is the
// object, or None. Of a class it is the C member that the class or a base
// has itself (a Python method of the same name may hide it: Tensor.unflatten).
const RESOLVE: &CStr = c"
import sys
C = ('builtin_function_or_method', 'method_descriptor', 'wrapper_descriptor', 'classmethod_descriptor', 'getset_descriptor')
def at(name):
    parts = name.removesuffix('.__get__').split('.')
    found = sys.modules[parts[0]]
    for part in parts[1:]:
        own = [vars(c)[part] for c in found.__mro__ if part in vars(c)] if isinstance(found, type) else []
        own = [member for member in own if type(member).__name__ in C]
        found = own[0] if own else getattr(found, part)
    return found
found = []
for name in names:
    try:
        found.append(at(name))
    except Exception:
        found.append(None)
";

struct Member {
    name: &'static str,
    taken: Taken,
    // The object the name resolved to, by address (0: it did not).
    object: usize,
    // The lines of its forms, if the file has them: a row or a stop each.
    forms: Vec<(&'static str, Taken)>,
}

struct Members {
    all: Vec<Member>,
    // What identifies a member's C function (see key) -> its index.
    by_key: HashMap<usize, usize>,
    failed: bool,
}

static MEMBERS: OnceLock<Members> = OnceLock::new();

// What a measured line means, a member's or a form's: the columns are
// member, form, kind, cpu, cuda, out, P, W, K, D, S, R, A, L (the head of the
// file says what each is). A line that is none of the cases written here is
// a stop.
fn verdict(line: &[&str]) -> Taken {
    let &[_, _, _, cpu, cuda, _, p, w, k, d, s, r, a, l] = line else { return Taken::Stop("its line of the table of torch's members has other columns than the rules read") };
    let is = |verdict: &str, word: &str| verdict.split('@').next() == Some(word);
    let stop = |why: String| Taken::Stop(Box::leak(why.into_boxed_str()));
    let enough = |n: &str| n.parse::<u32>().is_ok_and(|n| n >= 5);
    if s.starts_with("data") {
        stop(format!("the count or the sizes of its results follow the data of its operands, not only their keys (measured: {s})"))
    } else if p.starts_with("yes") {
        Taken::Later("native")
    } else if !is(w, "no") || !is(k, "no") {
        Taken::Later("writes")
    } else if !is(d, "no") {
        Taken::Later("ambient")
    } else if r.starts_with("NO") {
        stop(format!("two calls on the same operands give different bits (measured: {r})"))
    } else if (l.contains("follows=") && !l.contains("follows=none")) || is(s, "no-sample") {
        Taken::Later("ambient")
    } else if a.starts_with("no-tensor-result") || l.contains("pyval=data") {
        Taken::Later("facts")
    } else if !is(s, "key") || !is(r, "yes") || !(is(p, "no") || p.starts_with("warn")) {
        stop(format!("its measured line is none the rules know (P {p}, S {s}, R {r})"))
    } else if !enough(cpu) || !enough(cuda) {
        stop(format!("it has under 5 returning samples on a device (cpu {cpu}, cuda {cuda})"))
    } else if p.starts_with("warn") {
        Taken::Rows(&[ROW_WARN])
    } else {
        Taken::Rows(&[ROW])
    }
}

// What identifies the C function behind a callable: for a builtin and a
// method descriptor the definition both are made from, else the object.
unsafe fn key(f: *mut PyObject) -> usize {
    let tp = ffi::Py_TYPE(f);
    if tp == &raw mut ffi::PyCFunction_Type {
        (*(f as *mut ffi::PyCFunctionObject)).m_ml as usize
    } else if tp == &raw mut ffi::PyMethodDescr_Type || tp == &raw mut ffi::PyClassMethodDescr_Type {
        (*(f as *mut ffi::PyMethodDescrObject)).d_method as usize
    } else {
        f as usize
    }
}

// The table, made on first use: runs Python code (RESOLVE).
fn members() -> &'static Members {
    MEMBERS.get_or_init(|| {
        let lines: Vec<Vec<&str>> = include_str!("torch_members.tsv").lines().filter(|l| !l.starts_with('#') && !l.is_empty()).skip(1).map(|l| l.split('\t').collect()).collect();
        // A form's line comes behind its member's: a row, or a stop (what a
        // member's line leaves to another family is no row of a form).
        let mut all: Vec<Member> = vec![];
        for line in &lines {
            match (line.get(1), all.last_mut()) {
                (Some(&"-"), _) => all.push(Member { name: line[0], taken: verdict(line), object: 0, forms: vec![] }),
                (Some(form), Some(member)) if member.name == line[0] => member.forms.push((form, match verdict(line) {
                    Taken::Rows(_) => Taken::Rows(&[ROW_FORM]),
                    Taken::Stop(why) => Taken::Stop(why),
                    _ => Taken::Stop("in this form it was not measured to only compute"),
                })),
                _ => all.push(Member { name: line[0], taken: Taken::Stop("its line of the table of torch's members is of no member"), object: 0, forms: vec![] }),
            }
        }
        for member in all.iter_mut().filter(|m| m.forms.iter().any(|(_, taken)| matches!(taken, Taken::Rows(_)))) {
            member.taken = Taken::Rows(&[ROW_FORM]);
        }
        let mut by_key = HashMap::new();
        let failed = Python::attach(|py| unsafe {
            let g = pyo3::types::PyDict::new(py);
            let ran = g.set_item("names", all.iter().map(|member| member.name).collect::<Vec<_>>()).and_then(|_| py.run(RESOLVE, Some(&g), None));
            let found = reads::dict_get(g.as_ptr(), ffi::PyUnicode_InternFromString(c"found".as_ptr()));
            if ran.is_err() || found.is_null() || ffi::PyList_GET_SIZE(found) != all.len() as ffi::Py_ssize_t {
                return true;
            }
            for (i, member) in all.iter_mut().enumerate() {
                let object = ffi::PyList_GET_ITEM(found, i as ffi::Py_ssize_t);
                if object != ffi::Py_None() {
                    member.object = object as usize;
                    by_key.insert(key(object), i);
                }
            }
            // The table refers to its objects by address: kept for good.
            std::mem::forget(g.unbind());
            false
        });
        Members { all, by_key, failed }
    })
}

// Makes the table. Runs Python code: called where a function goes onto the
// executor, not from inside it.
pub fn ready() {
    members();
    super::super::torch_constant_types();
}

// The member `f` calls: itself, the method a bound builtin is of, or what a
// functools.partial calls.
unsafe fn member(f: *mut PyObject) -> Option<&'static Member> {
    let members = MEMBERS.get()?;
    let partial = tr_partial_func(f);
    let called = tr_bound_function(if partial.is_null() { f } else { partial });
    members.by_key.get(&key(called)).map(|i| &members.all[*i])
}

// The name of the member `f` calls, for a refusal.
pub unsafe fn name(f: *mut PyObject) -> Option<&'static str> {
    member(f).map(|m| m.name)
}

fn rule(what: &What) -> Taken {
    let What::Callee(f) = *what else { return Taken::No };
    match MEMBERS.get() {
        None => Taken::Stop("the table of torch's members is not made"),
        Some(members) if members.failed => Taken::Stop("the table of torch's members could not be made"),
        Some(_) => unsafe { member(f) }.map_or(Taken::No, |m| m.taken),
    }
}

fn table() -> Vec<(String, Taken)> {
    members().all.iter().map(|m| (m.name.to_string(), m.taken)).collect()
}

/// Every member of the table of torch's C callees: its name, the address of
/// the object the name resolved to (0: it did not), and what the rules say
/// of it: ``"row"``, ``"stop"`` or ``"later:FAMILY"``.
#[pyfunction]
pub fn _rules_torch() -> Vec<(String, usize, String)> {
    members().all.iter().map(|m| (m.name.to_string(), m.object, status(m.taken))).collect()
}

fn status(taken: Taken) -> String {
    match taken {
        Taken::Rows(_) => "row".to_string(),
        Taken::Later(family) => format!("later:{family}"),
        _ => "stop".to_string(),
    }
}

/// Every form of operands the table has a line of its own for: the member,
/// the form, and what the rules say of a call of that form: ``"row"`` or
/// ``"stop"``.
#[pyfunction]
pub fn _rules_torch_forms() -> Vec<(String, String, String)> {
    members().all.iter().flat_map(|m| m.forms.iter().map(|(form, taken)| (m.name.to_string(), form.to_string(), status(*taken)))).collect()
}

unsafe fn is_out(name: *mut PyObject) -> bool {
    ffi::Py_TYPE(name) == &raw mut ffi::PyUnicode_Type && ffi::PyUnicode_CompareWithASCIIString(name, c"out".as_ptr()) == 0
}

// Whether a row takes the operand `o`: a plain tensor, an immutable, a
// constant of torch, the engine's stand-in for a number, or an exact tuple,
// list or dict (with str keys) of those. Err: why not.
unsafe fn plain(o: *mut PyObject, depth: u8) -> Result<(), String> {
    let tp = ffi::Py_TYPE(o);
    if [Group::PlainTensor, Group::Immutable, Group::TorchConstant].iter().any(|g| g.holds(tp)) || scalars::stand_in(o) {
        return Ok(());
    }
    let mut items = vec![];
    if tp == &raw mut ffi::PyTuple_Type {
        items.extend((0..ffi::PyTuple_GET_SIZE(o)).map(|i| ffi::PyTuple_GET_ITEM(o, i)));
    } else if tp == &raw mut ffi::PyList_Type {
        items.extend((0..ffi::PyList_GET_SIZE(o)).map(|i| ffi::PyList_GET_ITEM(o, i)));
    } else if tp == &raw mut ffi::PyDict_Type {
        let (mut at, mut k, mut v) = (0, std::ptr::null_mut(), std::ptr::null_mut());
        while ffi::PyDict_Next(o, &mut at, &mut k, &mut v) != 0 {
            if ffi::Py_TYPE(k) != &raw mut ffi::PyUnicode_Type || is_out(k) {
                return Err("it is handed a dict with `out` or with a key that is no str".into());
            }
            items.push(v);
        }
    } else {
        return Err(format!("it is handed a {}, and the member was measured on plain tensors, numbers and exact containers of them", reads::type_name(o)));
    }
    if depth == 8 || items.iter().any(|x| x.is_null()) {
        return Err("it is handed containers nested more than 8 deep, or one that is being filled".into());
    }
    items.into_iter().try_for_each(|x| plain(x, depth + 1))
}

// The operands of a call of `f` in the call's order: the receiver of a bound
// builtin, then what the instruction hands over (the values of the keywords
// last). Err: a callee whose own operands are not all among them.
unsafe fn operands(f: *mut PyObject, site: &Site) -> Result<Vec<*mut PyObject>, String> {
    if !tr_partial_func(f).is_null() {
        return Err("it is called through a functools.partial, whose own arguments are not among the operands".into());
    }
    let receiver = if ffi::Py_TYPE(f) == &raw mut ffi::PyCFunction_Type {
        // What a builtin is bound to: nothing, its module or its class
        // (nothing to look at), or the receiver of a method.
        let bound = (*(f as *mut ffi::PyCFunctionObject)).m_self;
        if bound.is_null() || ffi::PyType_Check(bound) != 0 || ffi::PyModule_Check(bound) != 0 { std::ptr::null_mut() } else { bound }
    } else if tr_bound_function(f) == f {
        std::ptr::null_mut()
    } else {
        return Err("it is called as a slot bound to its receiver, which is not among the operands".into());
    };
    Ok(std::iter::once(receiver).chain(site.operands.iter().copied()).filter(|o| !o.is_null() && *o != f).collect())
}

// The values a row is for: every operand plain, the receiver of a bound
// builtin among them, and no `out=`. Records nothing: the operands are what
// the call is run again on.
unsafe fn plain_call(what: &What, site: &Site) -> Result<(), String> {
    let What::Callee(f) = *what else { return Err("it is no call".into()) };
    let operands = operands(f, site)?;
    if !site.kwnames.is_null() && (0..ffi::PyTuple_GET_SIZE(site.kwnames)).any(|i| is_out(ffi::PyTuple_GET_ITEM(site.kwnames, i))) {
        return Err("it passes out=, which the member writes: that form was measured apart and has no row yet".into());
    }
    operands.into_iter().try_for_each(|o| plain(o, 0))
}

// The form of an operand as the column `form` of the table names it; None:
// an operand of a class, or nested deeper, than a form names.
unsafe fn form(o: *mut PyObject, depth: u8) -> Option<String> {
    let tp = ffi::Py_TYPE(o);
    let named = [(&raw mut ffi::PyLong_Type, "int"), (&raw mut ffi::PyFloat_Type, "float"), (&raw mut ffi::PyBool_Type, "bool"), (ffi::Py_TYPE(ffi::Py_None()), "None")];
    if Group::PlainTensor.holds(tp) {
        return Some("T".into());
    }
    if let Some((_, name)) = named.iter().find(|(named, _)| *named == tp) {
        return Some(name.to_string());
    }
    if tp as usize == super::super::torch_constant_types()[4] {
        return Some("Size".into());
    }
    let (name, items): (&str, Vec<*mut PyObject>) = if tp == &raw mut ffi::PyTuple_Type {
        ("tuple", (0..ffi::PyTuple_GET_SIZE(o)).map(|i| ffi::PyTuple_GET_ITEM(o, i)).collect())
    } else if tp == &raw mut ffi::PyList_Type {
        ("list", (0..ffi::PyList_GET_SIZE(o)).map(|i| ffi::PyList_GET_ITEM(o, i)).collect())
    } else {
        return None;
    };
    if depth == 8 {
        return None;
    }
    let mut forms = items.into_iter().map(|x| if x.is_null() { None } else { form(x, depth + 1) }).collect::<Option<Vec<_>>>()?;
    forms.sort();
    forms.dedup();
    Some(format!("{name}[{}]", forms.join(",")))
}

// The form of a call: of each positional operand in order, then name=form
// of each keyword in the order of the names.
unsafe fn call_form(operands: &[*mut PyObject], kwnames: *mut PyObject) -> Option<String> {
    let named = if kwnames.is_null() { 0 } else { ffi::PyTuple_GET_SIZE(kwnames) as usize };
    let positional = operands.len().checked_sub(named)?;
    let mut forms = operands[..positional].iter().map(|o| form(*o, 0)).collect::<Option<Vec<_>>>()?;
    let mut keywords = vec![];
    for (i, o) in operands[positional..].iter().enumerate() {
        let name = ffi::PyUnicode_AsUTF8(ffi::PyTuple_GET_ITEM(kwnames, i as ffi::Py_ssize_t));
        if name.is_null() {
            ffi::PyErr_Clear();
            return None;
        }
        keywords.push(format!("{}={}", CStr::from_ptr(name).to_str().ok()?, form(*o, 0)?));
    }
    keywords.sort();
    forms.extend(keywords);
    Some(forms.join(" "))
}

// The values a row of a member with forms is for: a plain call (plain_call)
// whose form has a line that is a row. Records nothing, as plain_call; it is
// asked wherever the call runs under a trace (the row reads classes), so a
// form that stops does so under a call taken at another's word too.
unsafe fn formed_call(what: &What, site: &Site) -> Result<(), String> {
    plain_call(what, site)?;
    let What::Callee(f) = *what else { return Err("it is no call".into()) };
    let Some(member) = member(f) else { return Err("it is no member of the table".into()) };
    let Some(form) = call_form(&operands(f, site)?, site.kwnames) else { return Err("it is handed an operand of a class no measured form of it has".into()) };
    match member.forms.iter().find(|(named, _)| *named == form) {
        Some((_, Taken::Rows(_))) => Ok(()),
        Some((_, Taken::Stop(why))) => Err(format!("called on ({form}): {why}")),
        _ => Err(format!("no call of it on ({form}) was measured apart, and on other operands the count or the sizes of its results follow data")),
    }
}
