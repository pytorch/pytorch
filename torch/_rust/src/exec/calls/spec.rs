// Decodes the spec of `_calls_make` (its form is at the head of mod.rs) into
// the types of a list. The spec is read by exact type and C-level container
// reads only; a kind that is not listed is refused.

use pyo3::exceptions::PyValueError;
use pyo3::ffi::{self, PyObject};
use pyo3::prelude::*;

use super::{Assert, Callee, Entry, Flags, List, Tmpl, FLAG_NAMES, KEY_FIELDS, UNARY_OPS};
use crate::bindings::ffi::{tensor_by_type, tensor_meta};
use crate::cfg::ir::{BINARY_OPS, COMPARE_OPS};

type P = *mut PyObject;
type Decoded<T> = Result<T, String>;

unsafe fn items(o: P, what: &str) -> Decoded<Vec<P>> {
    unsafe {
        if ffi::PyTuple_CheckExact(o) != 0 {
            Ok((0..ffi::PyTuple_GET_SIZE(o)).map(|i| ffi::PyTuple_GET_ITEM(o, i)).collect())
        } else if ffi::PyList_CheckExact(o) != 0 {
            Ok((0..ffi::PyList_GET_SIZE(o)).map(|i| ffi::PyList_GET_ITEM(o, i)).collect())
        } else {
            Err(format!("{what} must be a tuple or a list"))
        }
    }
}

unsafe fn text<'a>(o: P, what: &str) -> Decoded<&'a str> {
    unsafe {
        let mut n = 0;
        let p = if ffi::PyUnicode_CheckExact(o) != 0 { ffi::PyUnicode_AsUTF8AndSize(o, &mut n) } else { std::ptr::null() };
        if p.is_null() {
            ffi::PyErr_Clear();
            return Err(format!("{what} must be a str"));
        }
        Ok(std::str::from_utf8_unchecked(std::slice::from_raw_parts(p as *const u8, n as usize)))
    }
}

unsafe fn index(o: P, what: &str) -> Decoded<u32> {
    unsafe {
        let v = if ffi::PyLong_CheckExact(o) != 0 { ffi::PyLong_AsLongLong(o) } else { -1 };
        ffi::PyErr_Clear();
        u32::try_from(v).map_err(|_| format!("{what} must be an int from 0 that fits 32 bits"))
    }
}

unsafe fn indices(o: P, what: &str) -> Decoded<Box<[u32]>> {
    unsafe { items(o, what)?.into_iter().map(|i| index(i, what)).collect() }
}

// ("kind", field, ...): the kind and the fields.
unsafe fn tagged<'a>(o: P, what: &str) -> Decoded<(&'a str, Vec<P>)> {
    unsafe {
        let all = items(o, what)?;
        let Some(first) = all.first() else { return Err(format!("{what} must start with its kind")) };
        Ok((text(*first, what)?, all[1..].to_vec()))
    }
}

fn fields(kind: &str, rest: &[P], n: usize) -> Decoded<()> {
    if rest.len() == n { Ok(()) } else { Err(format!("{kind} takes {n} fields, not {}", rest.len())) }
}

unsafe fn object(o: P) -> Py<PyAny> {
    unsafe { Py::from_borrowed_ptr(Python::assume_attached(), o) }
}

unsafe fn tmpls(o: P) -> Decoded<Box<[Tmpl]>> {
    unsafe { items(o, "templates")?.into_iter().map(|t| tmpl(t)).collect() }
}

unsafe fn tmpl(o: P) -> Decoded<Tmpl> {
    unsafe {
        let (kind, rest) = tagged(o, "a template")?;
        let slot = |what: &str| fields(kind, &rest, 1).and_then(|()| index(rest[0], what));
        Ok(match kind {
            "null" => fields(kind, &rest, 0).map(|()| Tmpl::Null)?,
            "in" => Tmpl::In(slot("an input")?),
            "pin" => Tmpl::Pin(slot("a pin")?),
            "res" => Tmpl::Res(slot("a result")?),
            "const" => Tmpl::Const(slot("a constant")?),
            "part" => {
                fields(kind, &rest, 2)?;
                Tmpl::Part(index(rest[0], "a result")?, indices(rest[1], "a path")?)
            }
            "seq" => {
                fields(kind, &rest, 2)?;
                match text(rest[0], "the kind of a seq")? {
                    "list" => Tmpl::Seq(true, tmpls(rest[1])?),
                    "tuple" => Tmpl::Seq(false, tmpls(rest[1])?),
                    other => return Err(format!("a seq is a list or a tuple, not {other:?}")),
                }
            }
            "map" => {
                fields(kind, &rest, 2)?;
                Tmpl::Map(items(rest[0], "the keys of a map")?.into_iter().map(|k| object(k)).collect(), tmpls(rest[1])?)
            }
            "slice" => {
                fields(kind, &rest, 3)?;
                Tmpl::Slice(Box::new([tmpl(rest[0])?, tmpl(rest[1])?, tmpl(rest[2])?]))
            }
            other => return Err(format!("no template kind {other:?}")),
        })
    }
}

unsafe fn callee(o: P) -> Decoded<Callee> {
    unsafe {
        let (kind, rest) = tagged(o, "a callee")?;
        let operator = |table: &[&str]| -> Decoded<u32> {
            fields(kind, &rest, 1)?;
            let op = text(rest[0], "an operator")?;
            table.iter().position(|t| *t == op).map(|k| k as u32).ok_or_else(|| format!("no {kind} operator {op:?}"))
        };
        let name = || -> Decoded<Py<pyo3::types::PyString>> {
            fields(kind, &rest, 1)?;
            text(rest[0], "an attribute name")?;
            Ok(Py::from_borrowed_ptr(Python::assume_attached(), rest[0]))
        };
        Ok(match kind {
            "vector" => {
                fields(kind, &rest, 2)?;
                Callee::Vector(object(rest[0]), text(rest[1], "the name of a callee")?.into())
            }
            "binary" => Callee::Binary(operator(&BINARY_OPS)?),
            "compare" => Callee::Compare(operator(&COMPARE_OPS)?),
            "unary" => {
                let u = UNARY_OPS[operator(&UNARY_OPS.map(|u| u.0))? as usize];
                Callee::Unary(u.1, u.2)
            }
            "getitem" => fields(kind, &rest, 0).map(|()| Callee::GetItem)?,
            "setitem" => fields(kind, &rest, 0).map(|()| Callee::SetItem)?,
            "delitem" => fields(kind, &rest, 0).map(|()| Callee::DelItem)?,
            "getattr" => Callee::GetAttr(name()?),
            "setattr" => Callee::SetAttr(name()?),
            "state" => fields(kind, &rest, 1).map(|()| Callee::State(object(rest[0])))?,
            other => return Err(format!("no callee kind {other:?}")),
        })
    }
}

unsafe fn flags(o: P) -> Decoded<Flags> {
    unsafe {
        let mut f = Flags::default();
        for item in items(o, "flags")? {
            if ffi::PyUnicode_CheckExact(item) != 0 {
                let name = text(item, "a flag")?;
                let bit = FLAG_NAMES.iter().position(|n| *n == name).ok_or_else(|| format!("no flag {name:?}"))?;
                f.bits |= 1 << bit;
                continue;
            }
            let (kind, rest) = tagged(item, "a flag")?;
            fields(kind, &rest, 1)?;
            match kind {
                "writes" => f.writes = indices(rest[0], "a written operand")?,
                "rekeys" => f.rekeys = indices(rest[0], "a rekeyed operand")?,
                other => return Err(format!("no flag {other:?}")),
            }
        }
        Ok(f)
    }
}

unsafe fn asserts(o: P) -> Decoded<Option<Assert>> {
    unsafe {
        let all = items(o, "asserts")?;
        let mut a = Assert::default();
        for item in &all {
            let (kind, rest) = tagged(*item, "an assert")?;
            match kind {
                "key" => {
                    fields(kind, &rest, 2)?;
                    for field in items(rest[0], "the fields of a key")? {
                        let field = text(field, "a field of a key")?;
                        a.mask |= 1 << KEY_FIELDS.iter().position(|f| *f == field).ok_or_else(|| format!("no key field {field:?}"))?;
                    }
                    let Some(t) = tensor_by_type(rest[1] as usize).as_ref() else { return Err("the key of an assert is read off a tensor".into()) };
                    tensor_meta(t, &mut a.key);
                    if !a.key.defined || !a.key.has_shape || !a.key.has_device {
                        return Err("the key of an assert is read off a tensor that has sizes and a device".into());
                    }
                }
                "count" => {
                    fields(kind, &rest, 1)?;
                    a.count = Some(index(rest[0], "a count")? as usize);
                }
                "is" => {
                    fields(kind, &rest, 1)?;
                    a.same = Some(index(rest[0], "an operand")?);
                }
                other => return Err(format!("no assert kind {other:?}")),
            }
        }
        Ok((!all.is_empty()).then_some(a))
    }
}

unsafe fn entry(o: P) -> Decoded<Entry> {
    unsafe {
        let e = items(o, "an entry")?;
        if e.len() != 8 {
            return Err(format!("an entry has 8 fields, not {}", e.len()));
        }
        let none = ffi::Py_None();
        Ok(Entry {
            callee: callee(e[0])?,
            args: tmpls(e[1])?,
            // Its exact type is the list's to check.
            kwnames: if e[2] == none { None } else { Some(Py::from_borrowed_ptr(Python::assume_attached(), e[2])) },
            out: if e[3] == none { None } else { Some(index(e[3], "out")?) },
            at: index(e[4], "at")?,
            flags: flags(e[5])?,
            assert: asserts(e[6])?,
            kills: indices(e[7], "kills")?,
        })
    }
}

/// Makes a call list from its spec; raises ValueError, with the reason, for a
/// spec that is not understood in full.
#[pyfunction]
pub fn _calls_make(spec: &Bound<'_, PyAny>) -> PyResult<List> {
    let decoded = unsafe {
        (|| {
            let top = items(spec.as_ptr(), "a call list")?;
            if top.len() != 5 {
                return Err(format!("a call list has 5 fields, not {}", top.len()));
            }
            let mut entries = Vec::new();
            for (k, e) in items(top[3], "entries")?.into_iter().enumerate() {
                entries.push(entry(e).map_err(|why| format!("entry {k}: {why}"))?);
            }
            let consts = items(top[2], "constants")?.into_iter().map(|c| object(c)).collect();
            List::new(index(top[0], "the count of inputs")?, index(top[1], "the count of pins")?, consts, entries, tmpls(top[4])?.into_vec())
        })()
    };
    decoded.map_err(|why| PyValueError::new_err(format!("torch.vm call list: {why}")))
}
