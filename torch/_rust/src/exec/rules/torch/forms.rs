// The operands of a call of a member, as the rows of the torch family are
// for them (see the head of rules/torch.rs): plain operands (plain_call),
// the places they are handed in (placed), and the form of a call
// (formed_call).

use super::*;

pub(super) unsafe fn is_out(name: *mut PyObject) -> bool {
    ffi::Py_TYPE(name) == &raw mut ffi::PyUnicode_Type && ffi::PyUnicode_CompareWithASCIIString(name, c"out".as_ptr()) == 0
}

// Whether a row takes the operand `o`: a plain tensor, an immutable, a
// constant of torch, the engine's stand-in for a number, or an exact tuple,
// list or dict (with str keys) of those. Err: why not.
pub(super) unsafe fn plain(o: *mut PyObject, depth: u8) -> Result<(), String> {
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
pub(super) unsafe fn operands(f: *mut PyObject, site: &Site) -> Result<Vec<*mut PyObject>, String> {
    if !tr_partial_func(f).is_null() {
        return Err("it is called through a functools.partial, whose own arguments are not among the operands".into());
    }
    let receiver = if ffi::Py_TYPE(f) == &raw mut ffi::PyCFunction_Type {
        // What a builtin is bound to: nothing, its module or its class
        // (nothing to look at: exactly a module, exactly a type), or the
        // receiver of a method. A class with a metaclass, or a module of a
        // class of its own, is an object like any other, and no operand
        // the rows are for.
        let bound = (*(f as *mut ffi::PyCFunctionObject)).m_self;
        let exact = !bound.is_null() && (ffi::Py_TYPE(bound) == &raw mut ffi::PyType_Type || ffi::Py_TYPE(bound) == &raw mut ffi::PyModule_Type);
        if bound.is_null() || exact {
            std::ptr::null_mut()
        } else if ffi::PyType_Check(bound) != 0 || ffi::PyModule_Check(bound) != 0 {
            return Err(format!("it is bound to a {}, a class or a module that is not exactly a type or a module", reads::type_name(bound)));
        } else {
            bound
        }
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
pub(super) unsafe fn plain_call(_: &Row, what: &What, site: &Site) -> Result<(), String> {
    let What::Callee(f) = *what else { return Err("it is no call".into()) };
    let operands = operands(f, site)?;
    if !site.kwnames.is_null() && (0..ffi::PyTuple_GET_SIZE(site.kwnames)).any(|i| is_out(ffi::PyTuple_GET_ITEM(site.kwnames, i))) {
        return Err("it passes out=: where that was measured, the callee writes `out`, changes its key in place if it is not of the result's size, and writes an operand that is `out`; nothing tells those calls apart before they run".into());
    }
    operands.iter().try_for_each(|o| plain(*o, 0))?;
    match member(f) {
        Some(member) => sampled(member, &operands, site.kwnames),
        None => Err("it is no member of the table".into()),
    }
}

// The operands of a call with the place of each among them: its position
// (the receiver first), or the name of its keyword.
pub(super) unsafe fn placed(operands: &[*mut PyObject], kwnames: *mut PyObject) -> Result<Vec<(String, *mut PyObject)>, String> {
    let named = if kwnames.is_null() { 0 } else { ffi::PyTuple_GET_SIZE(kwnames) as usize };
    let positional = operands.len().checked_sub(named).ok_or("it is handed fewer operands than keywords")?;
    let mut found = Vec::with_capacity(operands.len());
    for (i, o) in operands.iter().enumerate() {
        if i < positional {
            found.push((i.to_string(), *o));
            continue;
        }
        let name = ffi::PyUnicode_AsUTF8(ffi::PyTuple_GET_ITEM(kwnames, (i - positional) as ffi::Py_ssize_t));
        if name.is_null() {
            ffi::PyErr_Clear();
            return Err("it is handed a keyword whose name is no UTF-8".into());
        }
        found.push((CStr::from_ptr(name).to_string_lossy().into_owned(), *o));
    }
    Ok(found)
}

// The class of an operand as the column `domain` of the table names it;
// None: an operand of a class, or nested deeper, than the column names.
pub(super) unsafe fn form(o: *mut PyObject, depth: u8) -> Option<String> {
    let tp = ffi::Py_TYPE(o);
    let named = [(&raw mut ffi::PyLong_Type, "int"), (&raw mut ffi::PyFloat_Type, "float"), (&raw mut ffi::PyBool_Type, "bool"), (ffi::Py_TYPE(ffi::Py_None()), "None"), (&raw mut ffi::PyUnicode_Type, "str"), (&raw mut ffi::PyComplex_Type, "complex")];
    if Group::PlainTensor.holds(tp) {
        return Some("T".into());
    }
    // The engine's stand-in for a float is the program's float.
    if scalars::stand_in(o) {
        return Some("float".into());
    }
    if let Some((_, name)) = named.iter().find(|(named, _)| *named == tp) {
        return Some(name.to_string());
    }
    let constants = super::super::super::torch_constant_types();
    if let Some(at) = constants.iter().position(|c| *c != 0 && *c == tp as usize) {
        return Some(["dtype", "device", "layout", "memory_format", "Size"][at].into());
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

// Whether an exact list or tuple has a tensor in it, at any depth; None:
// nested deeper than 8, or being filled.
pub(super) unsafe fn holds_tensor(o: *mut PyObject, depth: u8) -> Option<bool> {
    let tp = ffi::Py_TYPE(o);
    let items: Vec<*mut PyObject> = if tp == &raw mut ffi::PyTuple_Type {
        (0..ffi::PyTuple_GET_SIZE(o)).map(|i| ffi::PyTuple_GET_ITEM(o, i)).collect()
    } else if tp == &raw mut ffi::PyList_Type {
        (0..ffi::PyList_GET_SIZE(o)).map(|i| ffi::PyList_GET_ITEM(o, i)).collect()
    } else {
        return Some(Group::PlainTensor.holds(tp));
    };
    if depth == 8 || items.iter().any(|x| x.is_null()) {
        return None;
    }
    let mut any = false;
    for x in items {
        any |= holds_tensor(x, depth + 1)?;
    }
    Some(any)
}

// What a form of the table tells of a call's operands (the column `form`):
// per place, `T` for a tensor, `[T]` for an exact list or tuple with a tensor
// in it, `True` / `False` for a bool, and `K` for a keyword that holds no
// tensor (a bool by keyword is both). None: a dict among the operands (its
// items are no places), or a container nested deeper than a form tells.
pub(super) unsafe fn call_form(operands: &[*mut PyObject], kwnames: *mut PyObject) -> Option<Vec<(String, &'static str)>> {
    let placed = placed(operands, kwnames).ok()?;
    let positional = placed.len() - if kwnames.is_null() { 0 } else { ffi::PyTuple_GET_SIZE(kwnames) as usize };
    let mut found = vec![];
    for (i, (place, o)) in placed.into_iter().enumerate() {
        let tp = ffi::Py_TYPE(o);
        if tp == &raw mut ffi::PyDict_Type {
            return None;
        }
        if Group::PlainTensor.holds(tp) {
            found.push((place, "T"));
            continue;
        }
        if tp == &raw mut ffi::PyBool_Type {
            found.push((place.clone(), if o == ffi::Py_True() { "True" } else { "False" }));
        } else if holds_tensor(o, 0)? {
            found.push((place, "[T]"));
            continue;
        }
        if i >= positional {
            found.push((place, "K"));
        }
    }
    Some(found)
}

// The word of a form's line for the calls that pass no keyword but those
// that hold a tensor, among the lines of a form that tell keywords (no
// keyword has this name).
const NO_KEYWORD: &str = "-:K";

// Whether a call (call_form) is of the form `named` of the table: the same
// places hold a tensor, or a container with one, and each bool the form
// names has that value in the call. A bool of the call that the form does
// not name is of either value. A form that tells keywords (`alpha:K`, or
// NO_KEYWORD) is of the calls that pass exactly those keywords: a call
// that passes another set of them was not measured, and fits no such line.
// A form that tells none is of the calls with any.
pub(super) fn fits(named: &str, call: &[(String, &'static str)]) -> bool {
    let words: Vec<&str> = if named == "plain" { vec![] } else { named.split(' ').collect() };
    let none = words.contains(&NO_KEYWORD);
    let told: Vec<(&str, &str)> = words.iter().filter(|one| **one != NO_KEYWORD).filter_map(|one| one.split_once(':')).collect();
    // (A word that tells no place: the line of no form, `?`.)
    if told.len() + usize::from(none) != words.len() {
        return false;
    }
    let keyed = none || told.iter().any(|(_, one)| *one == "K");
    let exact = |one: &str| one == "T" || one == "[T]" || (keyed && one == "K");
    told.iter().all(|(place, one)| call.iter().any(|(at, had)| at == place && had == one)) && call.iter().filter(|(_, had)| exact(had)).all(|(at, had)| told.iter().any(|(place, one)| place == at && one == had))
}

// (A bool by keyword is shown once, by its value.)
pub(super) fn shown(call: &[(String, &'static str)]) -> String {
    let again = |i: usize| i > 0 && call[i].1 == "K" && call[i - 1].0 == call[i].0;
    call.iter().enumerate().filter(|(i, _)| !again(*i)).map(|(_, (place, one))| format!("{place}:{one}")).collect::<Vec<_>>().join(" ")
}

// The values a row of a member with forms is for: a plain call (plain_call)
// whose form has a line that is a row, and that is `row`'s form if `row` is
// for one. Of the lines that fit a call, the one that names most is the
// call's (of the lines of one form that tell keywords at most one fits).
// Records nothing, as plain_call; it is asked wherever the call
// runs under a trace (the row reads classes), so a form that stops does so
// under a call taken at another's word too.
pub(super) unsafe fn formed_call(row: &Row, what: &What, site: &Site) -> Result<(), String> {
    plain_call(row, what, site)?;
    let What::Callee(f) = *what else { return Err("it is no call".into()) };
    let Some(member) = member(f) else { return Err("it is no member of the table".into()) };
    let Some(call) = call_form(&operands(f, site)?, site.kwnames) else { return Err("it is handed a dict, or containers nested deeper than a measured form of it tells".into()) };
    let form = shown(&call);
    match member.forms.iter().filter(|(named, _)| fits(named, &call)).max_by_key(|(named, _)| named.len()) {
        Some((named, Taken::Rows(_))) if row.form.is_empty() || row.form == *named => Ok(()),
        Some((named, Taken::Rows(_))) => Err(format!("called on ({form}): that is the form ({named}), which has a row of its own")),
        Some((_, Taken::Stop(why))) => Err(format!("called on ({form}): {why}")),
        Some((named, _)) => Err(format!("called on ({form}): in the form ({named}) it was not measured to only compute")),
        None => Err(format!("no call of it with tensors in these places ({form}) was measured apart, and what calls of it do follows which operands are tensors, or their data")),
    }
}
