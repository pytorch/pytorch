//! The virtual machine: a register interpreter over the SSA graph.
//!
//! The instructions have the semantics of the bytecode they came from and go
//! through the CPython API, so anything Python can do runs here.

use crate::analysis::{self, Analysis};
use crate::ir::*;
use crate::pyffi::{self, new_ref, py_bool, xdecref, Obj, Shared};
use pyo3::ffi;
use std::ffi::CString;
use std::ptr::null_mut;

/// Inline cache of a global or module attribute load, keyed on the version
/// tags of the dictionaries involved: as long as a dict's tag is unchanged
/// the dict has not been modified, so the value it holds for the name is
/// still there (and still alive, the dict owns it).
#[derive(Clone, Copy)]
struct LoadCache {
    owner: Obj,
    version: u64,
    builtins_version: u64,
    value: Obj,
}

impl LoadCache {
    const EMPTY: LoadCache =
        LoadCache { owner: null_mut(), version: u64::MAX, builtins_version: u64::MAX, value: null_mut() };
}

/// Per-name caches; only touched with the GIL held.
struct LoadCaches(Vec<std::cell::Cell<LoadCache>>);

unsafe impl Send for LoadCaches {}
unsafe impl Sync for LoadCaches {}

#[inline(always)]
unsafe fn dict_version(dict: Obj) -> u64 {
    (*(dict as *mut ffi::PyDictObject)).ma_version_tag
}

pub struct Compiled {
    global_cache: LoadCaches,
    module_attr_cache: LoadCaches,
    /// Per constant: usable as a cached module attribute name.
    plain_name: Vec<bool>,
    pub func: Function,
    pub analysis: Analysis,
    pub generic: Vec<BlockCode>,
    frames: Shared<Vec<Vec<Obj>>>,
    name_c: CString,
    filename_c: CString,
}

type OpResult = Result<Obj, ()>;

unsafe fn set_error(exc: Obj, msg: String) {
    let c = CString::new(msg).unwrap_or_default();
    ffi::PyErr_SetString(exc, c.as_ptr());
}

#[inline(always)]
unsafe fn take(regs: &mut [Obj], v: V) -> Obj {
    std::mem::replace(regs.get_unchecked_mut(v as usize), null_mut())
}

#[inline(always)]
unsafe fn reg(regs: &[Obj], v: V) -> Obj {
    *regs.get_unchecked(v as usize)
}

#[inline(always)]
unsafe fn checked(obj: Obj) -> OpResult {
    if obj.is_null() {
        Err(())
    } else {
        Ok(obj)
    }
}

#[inline(always)]
unsafe fn status(rc: i32) -> OpResult {
    if rc < 0 {
        Err(())
    } else {
        Ok(null_mut())
    }
}

unsafe fn binary(kind: u8, a: Obj, b: Obj) -> Obj {
    match kind {
        0 => ffi::PyNumber_Add(a, b),
        1 => ffi::PyNumber_And(a, b),
        2 => ffi::PyNumber_FloorDivide(a, b),
        3 => ffi::PyNumber_Lshift(a, b),
        4 => ffi::PyNumber_MatrixMultiply(a, b),
        5 => ffi::PyNumber_Multiply(a, b),
        6 => ffi::PyNumber_Remainder(a, b),
        7 => ffi::PyNumber_Or(a, b),
        8 => ffi::PyNumber_Power(a, b, ffi::Py_None()),
        9 => ffi::PyNumber_Rshift(a, b),
        10 => ffi::PyNumber_Subtract(a, b),
        11 => ffi::PyNumber_TrueDivide(a, b),
        12 => ffi::PyNumber_Xor(a, b),
        13 => ffi::PyNumber_InPlaceAdd(a, b),
        14 => ffi::PyNumber_InPlaceAnd(a, b),
        15 => ffi::PyNumber_InPlaceFloorDivide(a, b),
        16 => ffi::PyNumber_InPlaceLshift(a, b),
        17 => ffi::PyNumber_InPlaceMatrixMultiply(a, b),
        18 => ffi::PyNumber_InPlaceMultiply(a, b),
        19 => ffi::PyNumber_InPlaceRemainder(a, b),
        20 => ffi::PyNumber_InPlaceOr(a, b),
        21 => ffi::PyNumber_InPlacePower(a, b, ffi::Py_None()),
        22 => ffi::PyNumber_InPlaceRshift(a, b),
        23 => ffi::PyNumber_InPlaceSubtract(a, b),
        24 => ffi::PyNumber_InPlaceTrueDivide(a, b),
        _ => ffi::PyNumber_InPlaceXor(a, b),
    }
}

/// `raise exc from cause`, as the interpreter does it.
unsafe fn do_raise(exc: Obj, cause: Obj) {
    let (ty, value) = if ffi::PyExceptionClass_Check(exc) != 0 {
        let value = ffi::PyObject_CallNoArgs(exc);
        if value.is_null() {
            return;
        }
        if ffi::PyExceptionInstance_Check(value) == 0 {
            set_error(
                ffi::PyExc_TypeError,
                "calling an exception class did not return an exception".to_string(),
            );
            ffi::Py_DECREF(value);
            return;
        }
        (exc, value)
    } else if ffi::PyExceptionInstance_Check(exc) != 0 {
        (ffi::Py_TYPE(exc) as Obj, new_ref(exc))
    } else {
        set_error(ffi::PyExc_TypeError, "exceptions must derive from BaseException".to_string());
        return;
    };
    if !cause.is_null() {
        let fixed = if ffi::PyExceptionClass_Check(cause) != 0 {
            let c = ffi::PyObject_CallNoArgs(cause);
            if c.is_null() {
                ffi::Py_DECREF(value);
                return;
            }
            c
        } else if ffi::PyExceptionInstance_Check(cause) != 0 {
            new_ref(cause)
        } else if cause == ffi::Py_None() {
            null_mut()
        } else {
            set_error(
                ffi::PyExc_TypeError,
                "exception causes must derive from BaseException".to_string(),
            );
            ffi::Py_DECREF(value);
            return;
        };
        ffi::PyException_SetCause(value, fixed);
    }
    ffi::PyErr_SetObject(ty, value);
    ffi::Py_DECREF(value);
}

impl Compiled {
    pub fn new(func: Function) -> Compiled {
        let analysis = analysis::analyze(&func);
        let mut generic = Vec::with_capacity(func.blocks.len());
        for (b, block) in func.blocks.iter().enumerate() {
            let mut insts = block.insts.clone();
            let term_kills =
                analysis::compute_kills(&mut insts, &block.term, &analysis.live_out[b]);
            generic.push(BlockCode { insts, term: block.term.clone(), term_kills });
        }
        let empty = || LoadCaches(vec![std::cell::Cell::new(LoadCache::EMPTY); func.consts.len()]);
        let plain_name = func
            .consts
            .iter()
            .map(|c| unsafe {
                ffi::PyUnicode_CheckExact(c.as_ptr()) != 0
                    && !pyffi::str_to_string(c.as_ptr()).starts_with("__")
            })
            .collect();
        Compiled {
            global_cache: empty(),
            module_attr_cache: empty(),
            plain_name,
            name_c: CString::new(func.name.clone()).unwrap_or_default(),
            filename_c: CString::new(func.filename.clone()).unwrap_or_default(),
            func,
            analysis,
            generic,
            frames: Shared::new(Vec::new()),
        }
    }

    // ------------------------------------------------------------------
    // Calling convention
    // ------------------------------------------------------------------

    unsafe fn bind_args(&self, regs: &mut [Obj], args: Obj, kwargs: Obj) -> Result<(), ()> {
        let f = &self.func;
        let nargs = ffi::PyTuple_GET_SIZE(args) as usize;
        let argcount = f.argcount;
        let nkw = if kwargs.is_null() { 0 } else { ffi::PyDict_Size(kwargs) as usize };
        for i in 0..nargs.min(argcount) {
            regs[f.params[i] as usize] = new_ref(ffi::PyTuple_GET_ITEM(args, i as isize));
        }
        let mut next = argcount + f.kwonlyargcount;
        if f.has_varargs {
            let rest = ffi::PyTuple_GetSlice(args, argcount.min(nargs) as isize, nargs as isize);
            if rest.is_null() {
                return Err(());
            }
            regs[f.params[next] as usize] = rest;
            next += 1;
        } else if nargs > argcount {
            set_error(
                ffi::PyExc_TypeError,
                format!(
                    "{}() takes {} positional argument{} but {} were given",
                    f.name,
                    argcount,
                    if argcount == 1 { "" } else { "s" },
                    nargs
                ),
            );
            return Err(());
        }
        let mut varkw: Obj = null_mut();
        if f.has_varkw {
            varkw = ffi::PyDict_New();
            if varkw.is_null() {
                return Err(());
            }
            regs[f.params[next] as usize] = varkw;
        }
        if nkw > 0 {
            let mut pos: ffi::Py_ssize_t = 0;
            let mut key: Obj = null_mut();
            let mut value: Obj = null_mut();
            while ffi::PyDict_Next(kwargs, &mut pos, &mut key, &mut value) != 0 {
                let mut found = None;
                for j in f.posonlyargcount..argcount + f.kwonlyargcount {
                    if f.param_names[j].as_ptr() == key {
                        found = Some(j);
                        break;
                    }
                }
                if found.is_none() {
                    for j in f.posonlyargcount..argcount + f.kwonlyargcount {
                        let eq = ffi::PyObject_RichCompareBool(
                            f.param_names[j].as_ptr(),
                            key,
                            ffi::Py_EQ,
                        );
                        if eq < 0 {
                            return Err(());
                        }
                        if eq > 0 {
                            found = Some(j);
                            break;
                        }
                    }
                }
                match found {
                    Some(j) => {
                        let slot = &mut regs[f.params[j] as usize];
                        if !slot.is_null() {
                            set_error(
                                ffi::PyExc_TypeError,
                                format!(
                                    "{}() got multiple values for argument '{}'",
                                    f.name,
                                    pyffi::str_to_string(key)
                                ),
                            );
                            return Err(());
                        }
                        *slot = new_ref(value);
                    }
                    None if !varkw.is_null() => {
                        if ffi::PyDict_SetItem(varkw, key, value) < 0 {
                            return Err(());
                        }
                    }
                    None => {
                        set_error(
                            ffi::PyExc_TypeError,
                            format!(
                                "{}() got an unexpected keyword argument '{}'",
                                f.name,
                                pyffi::str_to_string(key)
                            ),
                        );
                        return Err(());
                    }
                }
            }
        }
        if nargs < argcount {
            let ndefaults = match &f.defaults {
                Some(d) => ffi::PyTuple_GET_SIZE(d.as_ptr()) as usize,
                None => 0,
            };
            for i in nargs..argcount {
                let slot = &mut regs[f.params[i] as usize];
                if !slot.is_null() {
                    continue;
                }
                if i + ndefaults >= argcount {
                    let d = f.defaults.as_ref().unwrap().as_ptr();
                    *slot = new_ref(ffi::PyTuple_GET_ITEM(d, (i + ndefaults - argcount) as isize));
                } else {
                    set_error(
                        ffi::PyExc_TypeError,
                        format!(
                            "{}() missing 1 required positional argument: '{}'",
                            f.name,
                            pyffi::str_to_string(f.param_names[i].as_ptr())
                        ),
                    );
                    return Err(());
                }
            }
        }
        for j in argcount..argcount + f.kwonlyargcount {
            let slot = &mut regs[f.params[j] as usize];
            if !slot.is_null() {
                continue;
            }
            let mut default: Obj = null_mut();
            if let Some(kwdefaults) = &f.kwdefaults {
                default = ffi::PyDict_GetItemWithError(kwdefaults.as_ptr(), f.param_names[j].as_ptr());
                if default.is_null() && !ffi::PyErr_Occurred().is_null() {
                    return Err(());
                }
            }
            if default.is_null() {
                set_error(
                    ffi::PyExc_TypeError,
                    format!(
                        "{}() missing 1 required keyword-only argument: '{}'",
                        f.name,
                        pyffi::str_to_string(f.param_names[j].as_ptr())
                    ),
                );
                return Err(());
            }
            *slot = new_ref(default);
        }
        Ok(())
    }

    /// Calls the function. `args` is a tuple, `kwargs` a dict or null.
    /// Returns a new reference, or null with a Python error set.
    pub unsafe fn call(&self, args: Obj, kwargs: Obj) -> Obj {
        if ffi::Py_EnterRecursiveCall(c" while calling a cfg-compiled function".as_ptr()) != 0 {
            return null_mut();
        }
        let nregs = self.func.nvalues as usize;
        let mut regs = self.frames.borrow_mut().pop().unwrap_or_else(|| vec![null_mut(); nregs]);
        let result = match self.bind_args(&mut regs, args, kwargs) {
            Ok(()) => self.exec(&mut regs),
            Err(()) => null_mut(),
        };
        for r in regs.iter_mut() {
            xdecref(std::mem::replace(r, null_mut()));
        }
        self.frames.borrow_mut().push(regs);
        ffi::Py_LeaveRecursiveCall();
        result
    }

    unsafe fn fail(&self, bc: u32) -> Obj {
        let line = self.func.line_of(bc);
        pyffi::_PyTraceback_Add(self.name_c.as_ptr(), self.filename_c.as_ptr(), line);
        null_mut()
    }

    // ------------------------------------------------------------------
    // Generic instructions
    // ------------------------------------------------------------------

    unsafe fn vectorcall(&self, regs: &[Obj], callee: Obj, args: &[V], kwnames: u32) -> Obj {
        // One spare slot in front lets bound methods prepend self in place.
        let mut small: [Obj; 9] = [null_mut(); 9];
        let mut big: Vec<Obj>;
        let buf: &mut [Obj] = if args.len() < small.len() {
            &mut small[..]
        } else {
            big = vec![null_mut(); args.len() + 1];
            &mut big[..]
        };
        for (i, v) in args.iter().enumerate() {
            buf[i + 1] = reg(regs, *v);
        }
        let (kw, nkw) = if kwnames == NONE {
            (null_mut(), 0)
        } else {
            let kw = self.func.consts[kwnames as usize].as_ptr();
            (kw, ffi::PyTuple_GET_SIZE(kw) as usize)
        };
        let npos = args.len() - nkw;
        ffi::PyObject_Vectorcall(
            callee,
            buf.as_ptr().add(1),
            npos | ffi::PY_VECTORCALL_ARGUMENTS_OFFSET,
            kw,
        )
    }

    unsafe fn exec_op(&self, regs: &mut [Obj], op: &Op) -> OpResult {
        let f = &self.func;
        match op {
            Op::Const(c) => Ok(new_ref(f.consts.get_unchecked(*c as usize).as_ptr())),
            Op::LoadGlobal(name) => {
                let cache = self.global_cache.0.get_unchecked(*name as usize);
                let cached = cache.get();
                let versions =
                    (dict_version(f.globals.as_ptr()), dict_version(f.builtins.as_ptr()));
                if (cached.version, cached.builtins_version) == versions {
                    return Ok(new_ref(cached.value));
                }
                let name = f.consts[*name as usize].as_ptr();
                let mut v = ffi::PyDict_GetItemWithError(f.globals.as_ptr(), name);
                if v.is_null() {
                    if !ffi::PyErr_Occurred().is_null() {
                        return Err(());
                    }
                    v = ffi::PyDict_GetItemWithError(f.builtins.as_ptr(), name);
                    if v.is_null() {
                        if ffi::PyErr_Occurred().is_null() {
                            set_error(
                                ffi::PyExc_NameError,
                                format!("name '{}' is not defined", pyffi::str_to_string(name)),
                            );
                        }
                        return Err(());
                    }
                }
                cache.set(LoadCache {
                    owner: null_mut(),
                    version: versions.0,
                    builtins_version: versions.1,
                    value: v,
                });
                Ok(new_ref(v))
            }
            Op::LoadDeref(i) => {
                let closure = match &f.closure {
                    Some(c) => c.as_ptr(),
                    None => {
                        set_error(ffi::PyExc_SystemError, "function has no closure".to_string());
                        return Err(());
                    }
                };
                let cell = ffi::PyTuple_GET_ITEM(closure, *i as isize);
                let v = pyffi::PyCell_Get(cell);
                if v.is_null() {
                    if ffi::PyErr_Occurred().is_null() {
                        set_error(
                            ffi::PyExc_NameError,
                            "cannot access free variable where it is not associated with a value \
                             in enclosing scope"
                                .to_string(),
                        );
                    }
                    return Err(());
                }
                Ok(v)
            }
            Op::LoadAttr(a, name) => {
                let obj = reg(regs, *a);
                // torch.add and friends: an attribute of a plain module is an
                // entry of its dict.
                if ffi::PyModule_CheckExact(obj) != 0 && *self.plain_name.get_unchecked(*name as usize)
                {
                    let dict = ffi::PyModule_GetDict(obj);
                    if !dict.is_null() {
                        let cache = self.module_attr_cache.0.get_unchecked(*name as usize);
                        let cached = cache.get();
                        let version = dict_version(dict);
                        if cached.owner == obj && cached.version == version {
                            return Ok(new_ref(cached.value));
                        }
                        let v = ffi::PyDict_GetItemWithError(dict, f.consts[*name as usize].as_ptr());
                        if !v.is_null() {
                            cache.set(LoadCache { owner: obj, version, builtins_version: 0, value: v });
                            return Ok(new_ref(v));
                        }
                        if !ffi::PyErr_Occurred().is_null() {
                            return Err(());
                        }
                    }
                }
                checked(ffi::PyObject_GetAttr(obj, f.consts[*name as usize].as_ptr()))
            }
            Op::StoreAttr { obj, name, val } => status(ffi::PyObject_SetAttr(
                reg(regs, *obj),
                f.consts[*name as usize].as_ptr(),
                reg(regs, *val),
            )),
            Op::StoreGlobal(name, v) => status(ffi::PyDict_SetItem(
                f.globals.as_ptr(),
                f.consts[*name as usize].as_ptr(),
                reg(regs, *v),
            )),
            Op::Binary(kind, a, b) => checked(binary(*kind, reg(regs, *a), reg(regs, *b))),
            Op::Unary(kind, a) => {
                let a = reg(regs, *a);
                match kind {
                    UnaryKind::Neg => checked(ffi::PyNumber_Negative(a)),
                    UnaryKind::Pos => checked(ffi::PyNumber_Positive(a)),
                    UnaryKind::Invert => checked(ffi::PyNumber_Invert(a)),
                    UnaryKind::Not => {
                        let r = ffi::PyObject_IsTrue(a);
                        if r < 0 {
                            return Err(());
                        }
                        Ok(py_bool(r == 0))
                    }
                }
            }
            Op::Compare(kind, a, b) => {
                checked(ffi::PyObject_RichCompare(reg(regs, *a), reg(regs, *b), *kind as i32))
            }
            Op::Is(a, b, invert) => Ok(py_bool((reg(regs, *a) == reg(regs, *b)) != *invert)),
            Op::Contains(item, container, invert) => {
                let r = ffi::PySequence_Contains(reg(regs, *container), reg(regs, *item));
                if r < 0 {
                    return Err(());
                }
                Ok(py_bool((r == 1) != *invert))
            }
            Op::ToBool(a) => {
                let a = reg(regs, *a);
                if a == ffi::Py_True() || a == ffi::Py_False() {
                    return Ok(new_ref(a));
                }
                let r = ffi::PyObject_IsTrue(a);
                if r < 0 {
                    return Err(());
                }
                Ok(py_bool(r != 0))
            }
            Op::IsNone(a, invert) => Ok(py_bool((reg(regs, *a) == ffi::Py_None()) != *invert)),
            Op::GetItem(a, k) => checked(ffi::PyObject_GetItem(reg(regs, *a), reg(regs, *k))),
            Op::SetItem { obj, key, val } => {
                status(ffi::PyObject_SetItem(reg(regs, *obj), reg(regs, *key), reg(regs, *val)))
            }
            Op::DelItem(a, k) => status(ffi::PyObject_DelItem(reg(regs, *a), reg(regs, *k))),
            Op::BuildTuple(vs) => {
                let t = ffi::PyTuple_New(vs.len() as isize);
                if t.is_null() {
                    return Err(());
                }
                for (i, v) in vs.iter().enumerate() {
                    ffi::PyTuple_SET_ITEM(t, i as isize, new_ref(reg(regs, *v)));
                }
                Ok(t)
            }
            Op::BuildList(vs) => {
                let l = ffi::PyList_New(vs.len() as isize);
                if l.is_null() {
                    return Err(());
                }
                for (i, v) in vs.iter().enumerate() {
                    ffi::PyList_SET_ITEM(l, i as isize, new_ref(reg(regs, *v)));
                }
                Ok(l)
            }
            Op::BuildSet(vs) => {
                let s = ffi::PySet_New(null_mut());
                if s.is_null() {
                    return Err(());
                }
                for v in vs {
                    if ffi::PySet_Add(s, reg(regs, *v)) < 0 {
                        ffi::Py_DECREF(s);
                        return Err(());
                    }
                }
                Ok(s)
            }
            Op::BuildMap(vs) => {
                let d = ffi::PyDict_New();
                if d.is_null() {
                    return Err(());
                }
                for kv in vs.chunks(2) {
                    if ffi::PyDict_SetItem(d, reg(regs, kv[0]), reg(regs, kv[1])) < 0 {
                        ffi::Py_DECREF(d);
                        return Err(());
                    }
                }
                Ok(d)
            }
            Op::BuildSlice(a, b, c) => {
                let step = if *c == NONE { null_mut() } else { reg(regs, *c) };
                checked(ffi::PySlice_New(reg(regs, *a), reg(regs, *b), step))
            }
            Op::BuildString(vs) => {
                let t = ffi::PyTuple_New(vs.len() as isize);
                if t.is_null() {
                    return Err(());
                }
                for (i, v) in vs.iter().enumerate() {
                    ffi::PyTuple_SET_ITEM(t, i as isize, new_ref(reg(regs, *v)));
                }
                let empty = ffi::PyUnicode_New(0, 0);
                let s = if empty.is_null() { null_mut() } else { ffi::PyUnicode_Join(empty, t) };
                xdecref(empty);
                ffi::Py_DECREF(t);
                checked(s)
            }
            Op::Format { val, conv, spec } => {
                let val = reg(regs, *val);
                let converted = match conv {
                    1 => ffi::PyObject_Str(val),
                    2 => ffi::PyObject_Repr(val),
                    3 => ffi::PyObject_ASCII(val),
                    _ => new_ref(val),
                };
                if converted.is_null() {
                    return Err(());
                }
                if *spec == NONE && ffi::PyUnicode_CheckExact(converted) != 0 {
                    return Ok(converted);
                }
                let spec = if *spec == NONE { null_mut() } else { reg(regs, *spec) };
                let s = ffi::PyObject_Format(converted, spec);
                ffi::Py_DECREF(converted);
                checked(s)
            }
            Op::ListAppend(l, v) => status(ffi::PyList_Append(reg(regs, *l), reg(regs, *v))),
            Op::SetAdd(s, v) => status(ffi::PySet_Add(reg(regs, *s), reg(regs, *v))),
            Op::MapAdd(d, k, v) => {
                status(ffi::PyDict_SetItem(reg(regs, *d), reg(regs, *k), reg(regs, *v)))
            }
            Op::ListExtend(l, it) => {
                let r = ffi::PySequence_InPlaceConcat(reg(regs, *l), reg(regs, *it));
                if r.is_null() {
                    return Err(());
                }
                ffi::Py_DECREF(r);
                Ok(null_mut())
            }
            Op::SetUpdate(s, it) => {
                let name = ffi::PyUnicode_InternFromString(c"update".as_ptr());
                if name.is_null() {
                    return Err(());
                }
                let args = [reg(regs, *s), reg(regs, *it)];
                let r = ffi::PyObject_VectorcallMethod(name, args.as_ptr(), 2, null_mut());
                ffi::Py_DECREF(name);
                if r.is_null() {
                    return Err(());
                }
                ffi::Py_DECREF(r);
                Ok(null_mut())
            }
            Op::DictUpdate(d, o) => status(ffi::PyDict_Update(reg(regs, *d), reg(regs, *o))),
            Op::DictMerge(d, o) => {
                let (d, o) = (reg(regs, *d), reg(regs, *o));
                if ffi::PyDict_Check(o) != 0 {
                    let mut pos: ffi::Py_ssize_t = 0;
                    let mut key: Obj = null_mut();
                    let mut value: Obj = null_mut();
                    while ffi::PyDict_Next(o, &mut pos, &mut key, &mut value) != 0 {
                        let has = ffi::PyDict_Contains(d, key);
                        if has < 0 {
                            return Err(());
                        }
                        if has > 0 {
                            set_error(
                                ffi::PyExc_TypeError,
                                format!(
                                    "got multiple values for keyword argument {}",
                                    pyffi::repr_string(key)
                                ),
                            );
                            return Err(());
                        }
                    }
                }
                status(ffi::PyDict_Update(d, o))
            }
            Op::ListToTuple(l) => checked(ffi::PyList_AsTuple(reg(regs, *l))),
            Op::Unpack(seq, n) => {
                let seq = reg(regs, *seq);
                let n = *n as isize;
                let t = if ffi::PyTuple_CheckExact(seq) != 0 {
                    new_ref(seq)
                } else {
                    let t = ffi::PySequence_Tuple(seq);
                    if t.is_null() {
                        if ffi::PyErr_ExceptionMatches(ffi::PyExc_TypeError) != 0 {
                            ffi::PyErr_Clear();
                            set_error(
                                ffi::PyExc_TypeError,
                                format!("cannot unpack non-iterable {} object", pyffi::type_name(seq)),
                            );
                        }
                        return Err(());
                    }
                    t
                };
                let size = ffi::PyTuple_GET_SIZE(t);
                if size != n {
                    ffi::Py_DECREF(t);
                    set_error(
                        ffi::PyExc_ValueError,
                        if size > n {
                            format!("too many values to unpack (expected {})", n)
                        } else {
                            format!("not enough values to unpack (expected {}, got {})", n, size)
                        },
                    );
                    return Err(());
                }
                Ok(t)
            }
            Op::TupleGet(t, i) => Ok(new_ref(ffi::PyTuple_GET_ITEM(reg(regs, *t), *i as isize))),
            Op::Call { callee, args, kwnames } => {
                checked(self.vectorcall(regs, reg(regs, *callee), args, *kwnames))
            }
            Op::CallEx { callee, args, kwargs } => {
                let args = reg(regs, *args);
                let tuple = if ffi::PyTuple_CheckExact(args) != 0 {
                    new_ref(args)
                } else {
                    let t = ffi::PySequence_Tuple(args);
                    if t.is_null() {
                        return Err(());
                    }
                    t
                };
                let kwargs = if *kwargs == NONE { null_mut() } else { reg(regs, *kwargs) };
                let r = ffi::PyObject_Call(reg(regs, *callee), tuple, kwargs);
                ffi::Py_DECREF(tuple);
                checked(r)
            }
            Op::GetIter(a) => checked(ffi::PyObject_GetIter(reg(regs, *a))),
            Op::IterNext(it) => {
                let v = ffi::PyIter_Next(reg(regs, *it));
                if v.is_null() && !ffi::PyErr_Occurred().is_null() {
                    return Err(());
                }
                Ok(v)
            }
            Op::CheckBound(v, name) => {
                if reg(regs, *v).is_null() {
                    set_error(
                        ffi::PyExc_UnboundLocalError,
                        format!(
                            "cannot access local variable '{}' where it is not associated with a \
                             value",
                            pyffi::str_to_string(f.consts[*name as usize].as_ptr())
                        ),
                    );
                    return Err(());
                }
                Ok(null_mut())
            }
        }
    }

    // ------------------------------------------------------------------
    // The interpreter loop
    // ------------------------------------------------------------------

    unsafe fn exec(&self, regs: &mut [Obj]) -> Obj {
        let func = &self.func;
        let mut cur: usize = 0;
        let mut pred: usize = usize::MAX;
        let mut back_edges: u32 = 0;
        loop {
            if pred != usize::MAX {
                let block = func.blocks.get_unchecked(cur);
                let mut pi = 0;
                while *block.preds.get_unchecked(pi) as usize != pred {
                    pi += 1;
                }
                let moves = self.analysis.edge_moves.get_unchecked(cur).get_unchecked(pi);
                if !moves.is_empty() {
                    // Phis are parallel copies: read every source first.
                    let mut small: [Obj; 8] = [null_mut(); 8];
                    let mut big: Vec<Obj>;
                    let vals: &mut [Obj] = if moves.len() <= small.len() {
                        &mut small[..moves.len()]
                    } else {
                        big = vec![null_mut(); moves.len()];
                        &mut big[..]
                    };
                    for (i, (_, src)) in moves.iter().enumerate() {
                        let o = reg(regs, *src);
                        if !o.is_null() {
                            ffi::Py_INCREF(o);
                        }
                        vals[i] = o;
                    }
                    for (i, (dst, _)) in moves.iter().enumerate() {
                        let old = std::mem::replace(regs.get_unchecked_mut(*dst as usize), vals[i]);
                        xdecref(old);
                    }
                }
                for v in self.analysis.edge_kills.get_unchecked(cur).get_unchecked(pi) {
                    xdecref(take(regs, *v));
                }
                if cur <= pred {
                    back_edges = back_edges.wrapping_add(1);
                    if back_edges & 63 == 0 && ffi::PyErr_CheckSignals() != 0 {
                        return self.fail(block.bc_start);
                    }
                }
            }

            let code = self.generic.get_unchecked(cur);
            for inst in &code.insts {
                match self.exec_op(regs, &inst.op) {
                    Ok(obj) => {
                        if inst.dst != NONE {
                            let old =
                                std::mem::replace(regs.get_unchecked_mut(inst.dst as usize), obj);
                            xdecref(old);
                        }
                        for v in &inst.kills {
                            xdecref(take(regs, *v));
                        }
                    }
                    Err(()) => return self.fail(inst.bc),
                }
            }
            let next = match &code.term {
                Term::Jump(b) => *b as usize,
                Term::Br { cond, then_, else_ } => {
                    let c = reg(regs, *cond);
                    let truth = if c == ffi::Py_True() {
                        true
                    } else if c == ffi::Py_False() {
                        false
                    } else {
                        let r = ffi::PyObject_IsTrue(c);
                        if r < 0 {
                            return self.fail(func.blocks[cur].bc_start);
                        }
                        r != 0
                    };
                    (if truth { *then_ } else { *else_ }) as usize
                }
                Term::BrNull { v, null, nonnull } => {
                    (if reg(regs, *v).is_null() { *null } else { *nonnull }) as usize
                }
                Term::Return(v) => return take(regs, *v),
                Term::Raise { exc, cause } => {
                    let cause = if *cause == NONE { null_mut() } else { reg(regs, *cause) };
                    do_raise(reg(regs, *exc), cause);
                    let bc = code.insts.last().map_or(func.blocks[cur].bc_start, |i| i.bc);
                    return self.fail(bc);
                }
            };
            for v in &code.term_kills {
                xdecref(take(regs, *v));
            }
            pred = cur;
            cur = next;
        }
    }
}
