//! The virtual machine: a register interpreter over the SSA graph.
//!
//! Each basic block runs in one of three tiers:
//!   - generic: the instructions have the semantics of the bytecode they came
//!     from and go through the CPython API. Anything Python can do runs here,
//!     so what other compilers call a graph break is just a block (or an
//!     instruction) that stays in this tier;
//!   - profiling: generic, with the dispatcher capture hook installed around
//!     the operations that may be tensor operations, and with the values that
//!     flow by being recorded. Real tensors, real kernels: the first run of a
//!     block is its trace;
//!   - specialized: a version of the block produced by the pass pipeline from
//!     the profile. It calls post-dispatch kernels directly and may depend on
//!     guards; a failed guard resumes the profiling tier, in this block only.

use crate::analysis::{self, Analysis};
use crate::capi::{self, SiteSlot, TcfgApi, TcfgPyOp, TcfgTensorInfo};
use crate::ir::*;
use crate::pyffi::{self, new_ref, py_bool, xdecref, Obj, Shared};
use std::sync::atomic::AtomicU64;
use crate::spec::{self, PassCx, Pipeline, Unit, ValueObs};
use pyo3::ffi;
use std::ffi::CString;
use std::os::raw::c_char;
use std::ptr::null_mut;
use std::sync::atomic::Ordering;
use std::sync::Arc;

pub struct Options {
    /// Run the profiling tier and specialize. Off: generic tier only.
    pub specialize: bool,
    /// Let call sites enter below the Autograd kernel when no gradient is
    /// being tracked.
    pub skip_autograd: bool,
    pub max_versions: usize,
    /// How many times a block may go back to the profiling tier.
    pub max_respecializations: u32,
    pub max_site_variants: usize,
    /// Keep the tensors kernels return unboxed until Python code needs them.
    pub unboxed: bool,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            specialize: true,
            skip_autograd: true,
            max_versions: 4,
            max_respecializations: 6,
            max_site_variants: 4,
            unboxed: true,
        }
    }
}

pub struct BlockState {
    pub versions: Vec<Arc<BlockCode>>,
    pub slots: Vec<Option<Arc<SiteSlot>>>,
    /// Why an instruction that made dispatcher calls did not become a site.
    pub reasons: Vec<Option<String>>,
    pub capture_attempts: Vec<u32>,
    pub respec_left: u32,
    pub profile_runs: u32,
    pub next_version_id: u32,
    pub execs: u64,
    pub spec_execs: u64,
    pub entry_misses: u64,
    pub deopts: u64,
    pub rejected: Option<String>,
}

pub struct State {
    pub blocks: Vec<BlockState>,
    pub obs: Vec<ValueObs>,
    pub calls: u64,
    pub log: Vec<String>,
}

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
    pub generic: Vec<Arc<BlockCode>>,
    pub state: Shared<State>,
    pub options: Options,
    pub pipeline: Pipeline,
    frames: Shared<Vec<Vec<Obj>>>,
    /// Kernel results that were kept unboxed, and how many of those got a
    /// Python object after all.
    pub unboxed_results: AtomicU64,
    pub boxed_later: AtomicU64,
    name_c: CString,
    filename_c: CString,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Generic,
    Profile,
    Spec,
}

const MAX_CAPTURE_ATTEMPTS: u32 = 8;

type OpResult = Result<Obj, ()>;

unsafe fn set_error(exc: Obj, msg: String) {
    let c = CString::new(msg).unwrap_or_default();
    ffi::PyErr_SetString(exc, c.as_ptr());
}

#[inline(always)]
unsafe fn take(regs: &mut [Obj], v: V) -> Obj {
    std::mem::replace(regs.get_unchecked_mut(v as usize), null_mut())
}

/// The raw content of a register: a Python object, null, or an unboxed
/// tensor. Only for code that handles all three; see `Compiled::obj`.
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
    /// The value of a register as a Python object. A kernel result is kept
    /// unboxed until this is called on it, that is until generic code looks
    /// at it; then it gets its Python object, in place.
    #[inline(always)]
    unsafe fn obj(&self, regs: &mut [Obj], v: V) -> Obj {
        let o = reg(regs, v);
        if capi::is_unboxed(o) {
            self.box_register(regs, v)
        } else {
            o
        }
    }

    #[cold]
    unsafe fn box_register(&self, regs: &mut [Obj], v: V) -> Obj {
        let boxed = (capi::api().unwrap_unchecked().tensor_box)(reg(regs, v));
        if boxed.is_null() {
            // The reference was consumed either way.
            *regs.get_unchecked_mut(v as usize) = null_mut();
            panic!("cfg_compile: could not create the Python object of a tensor");
        }
        *regs.get_unchecked_mut(v as usize) = boxed;
        self.boxed_later.fetch_add(1, Ordering::Relaxed);
        boxed
    }

    pub fn new(func: Function, options: Options, pipeline: Pipeline) -> Compiled {
        let analysis = analysis::analyze(&func);
        let mut generic = Vec::with_capacity(func.blocks.len());
        for (b, block) in func.blocks.iter().enumerate() {
            let mut insts = block.insts.clone();
            let term_kills =
                analysis::compute_kills(&mut insts, &block.term, &analysis.live_out[b]);
            generic.push(Arc::new(BlockCode {
                insts,
                term: block.term.clone(),
                term_kills,
                entry_guards: Vec::new(),
                id: 0,
            }));
        }
        let blocks = func
            .blocks
            .iter()
            .map(|block| BlockState {
                versions: Vec::new(),
                slots: vec![None; block.insts.len()],
                reasons: vec![None; block.insts.len()],
                capture_attempts: vec![0; block.insts.len()],
                respec_left: options.max_respecializations,
                profile_runs: 0,
                next_version_id: 1,
                execs: 0,
                spec_execs: 0,
                entry_misses: 0,
                deopts: 0,
                rejected: None,
            })
            .collect();
        let state = State {
            blocks,
            obs: vec![ValueObs::Unseen; func.nvalues as usize],
            calls: 0,
            log: Vec::new(),
        };
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
            state: Shared::new(state),
            options,
            pipeline,
            frames: Shared::new(Vec::new()),
            unboxed_results: AtomicU64::new(0),
            boxed_later: AtomicU64::new(0),
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
            capi::release(std::mem::replace(r, null_mut()));
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

    unsafe fn vectorcall(&self, regs: &mut [Obj], callee: Obj, args: &[V], kwnames: u32) -> Obj {
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
            buf[i + 1] = self.obj(regs, *v);
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
                let obj = self.obj(regs, *a);
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
                self.obj(regs, *obj),
                f.consts[*name as usize].as_ptr(),
                self.obj(regs, *val),
            )),
            Op::StoreGlobal(name, v) => status(ffi::PyDict_SetItem(
                f.globals.as_ptr(),
                f.consts[*name as usize].as_ptr(),
                self.obj(regs, *v),
            )),
            Op::Binary(kind, a, b) => checked(binary(*kind, self.obj(regs, *a), self.obj(regs, *b))),
            Op::Unary(kind, a) => {
                let a = self.obj(regs, *a);
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
                checked(ffi::PyObject_RichCompare(self.obj(regs, *a), self.obj(regs, *b), *kind as i32))
            }
            Op::Is(a, b, invert) => Ok(py_bool((self.obj(regs, *a) == self.obj(regs, *b)) != *invert)),
            Op::Contains(item, container, invert) => {
                let r = ffi::PySequence_Contains(self.obj(regs, *container), self.obj(regs, *item));
                if r < 0 {
                    return Err(());
                }
                Ok(py_bool((r == 1) != *invert))
            }
            Op::ToBool(a) => {
                let a = self.obj(regs, *a);
                if a == ffi::Py_True() || a == ffi::Py_False() {
                    return Ok(new_ref(a));
                }
                let r = ffi::PyObject_IsTrue(a);
                if r < 0 {
                    return Err(());
                }
                Ok(py_bool(r != 0))
            }
            // An unboxed tensor is not None; no need to box it to find out.
            Op::IsNone(a, invert) => Ok(py_bool((reg(regs, *a) == ffi::Py_None()) != *invert)),
            Op::GetItem(a, k) => checked(ffi::PyObject_GetItem(self.obj(regs, *a), self.obj(regs, *k))),
            Op::SetItem { obj, key, val } => {
                status(ffi::PyObject_SetItem(self.obj(regs, *obj), self.obj(regs, *key), self.obj(regs, *val)))
            }
            Op::DelItem(a, k) => status(ffi::PyObject_DelItem(self.obj(regs, *a), self.obj(regs, *k))),
            Op::BuildTuple(vs) => {
                let t = ffi::PyTuple_New(vs.len() as isize);
                if t.is_null() {
                    return Err(());
                }
                for (i, v) in vs.iter().enumerate() {
                    ffi::PyTuple_SET_ITEM(t, i as isize, new_ref(self.obj(regs, *v)));
                }
                Ok(t)
            }
            Op::BuildList(vs) => {
                let l = ffi::PyList_New(vs.len() as isize);
                if l.is_null() {
                    return Err(());
                }
                for (i, v) in vs.iter().enumerate() {
                    ffi::PyList_SET_ITEM(l, i as isize, new_ref(self.obj(regs, *v)));
                }
                Ok(l)
            }
            Op::BuildSet(vs) => {
                let s = ffi::PySet_New(null_mut());
                if s.is_null() {
                    return Err(());
                }
                for v in vs {
                    if ffi::PySet_Add(s, self.obj(regs, *v)) < 0 {
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
                    if ffi::PyDict_SetItem(d, self.obj(regs, kv[0]), self.obj(regs, kv[1])) < 0 {
                        ffi::Py_DECREF(d);
                        return Err(());
                    }
                }
                Ok(d)
            }
            Op::BuildSlice(a, b, c) => {
                let step = if *c == NONE { null_mut() } else { self.obj(regs, *c) };
                checked(ffi::PySlice_New(self.obj(regs, *a), self.obj(regs, *b), step))
            }
            Op::BuildString(vs) => {
                let t = ffi::PyTuple_New(vs.len() as isize);
                if t.is_null() {
                    return Err(());
                }
                for (i, v) in vs.iter().enumerate() {
                    ffi::PyTuple_SET_ITEM(t, i as isize, new_ref(self.obj(regs, *v)));
                }
                let empty = ffi::PyUnicode_New(0, 0);
                let s = if empty.is_null() { null_mut() } else { ffi::PyUnicode_Join(empty, t) };
                xdecref(empty);
                ffi::Py_DECREF(t);
                checked(s)
            }
            Op::Format { val, conv, spec } => {
                let val = self.obj(regs, *val);
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
                let spec = if *spec == NONE { null_mut() } else { self.obj(regs, *spec) };
                let s = ffi::PyObject_Format(converted, spec);
                ffi::Py_DECREF(converted);
                checked(s)
            }
            Op::ListAppend(l, v) => status(ffi::PyList_Append(self.obj(regs, *l), self.obj(regs, *v))),
            Op::SetAdd(s, v) => status(ffi::PySet_Add(self.obj(regs, *s), self.obj(regs, *v))),
            Op::MapAdd(d, k, v) => {
                status(ffi::PyDict_SetItem(self.obj(regs, *d), self.obj(regs, *k), self.obj(regs, *v)))
            }
            Op::ListExtend(l, it) => {
                let r = ffi::PySequence_InPlaceConcat(self.obj(regs, *l), self.obj(regs, *it));
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
                let args = [self.obj(regs, *s), self.obj(regs, *it)];
                let r = ffi::PyObject_VectorcallMethod(name, args.as_ptr(), 2, null_mut());
                ffi::Py_DECREF(name);
                if r.is_null() {
                    return Err(());
                }
                ffi::Py_DECREF(r);
                Ok(null_mut())
            }
            Op::DictUpdate(d, o) => status(ffi::PyDict_Update(self.obj(regs, *d), self.obj(regs, *o))),
            Op::DictMerge(d, o) => {
                let (d, o) = (self.obj(regs, *d), self.obj(regs, *o));
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
            Op::ListToTuple(l) => checked(ffi::PyList_AsTuple(self.obj(regs, *l))),
            Op::Unpack(seq, n) => {
                let seq = self.obj(regs, *seq);
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
            Op::TupleGet(t, i) => Ok(new_ref(ffi::PyTuple_GET_ITEM(self.obj(regs, *t), *i as isize))),
            Op::Call { callee, args, kwnames } => {
                let callee = self.obj(regs, *callee);
                checked(self.vectorcall(regs, callee, args, *kwnames))
            }
            Op::CallMethod { recv, name, args, kwnames } => {
                // Looks the method up on the type and calls it with the
                // receiver, without materializing a bound method object.
                let mut small: [Obj; 9] = [null_mut(); 9];
                let mut big: Vec<Obj>;
                let buf: &mut [Obj] = if args.len() < small.len() {
                    &mut small[..]
                } else {
                    big = vec![null_mut(); args.len() + 1];
                    &mut big[..]
                };
                buf[0] = self.obj(regs, *recv);
                for (i, v) in args.iter().enumerate() {
                    buf[i + 1] = self.obj(regs, *v);
                }
                let (kw, nkw) = if *kwnames == NONE {
                    (null_mut(), 0)
                } else {
                    let kw = f.consts[*kwnames as usize].as_ptr();
                    (kw, ffi::PyTuple_GET_SIZE(kw) as usize)
                };
                checked(ffi::PyObject_VectorcallMethod(
                    f.consts[*name as usize].as_ptr(),
                    buf.as_ptr(),
                    args.len() + 1 - nkw,
                    kw,
                ))
            }
            Op::CallEx { callee, args, kwargs } => {
                let args = self.obj(regs, *args);
                let tuple = if ffi::PyTuple_CheckExact(args) != 0 {
                    new_ref(args)
                } else {
                    let t = ffi::PySequence_Tuple(args);
                    if t.is_null() {
                        return Err(());
                    }
                    t
                };
                let kwargs = if *kwargs == NONE { null_mut() } else { self.obj(regs, *kwargs) };
                let r = ffi::PyObject_Call(self.obj(regs, *callee), tuple, kwargs);
                ffi::Py_DECREF(tuple);
                checked(r)
            }
            Op::GetIter(a) => checked(ffi::PyObject_GetIter(self.obj(regs, *a))),
            Op::IterNext(it) => {
                let v = ffi::PyIter_Next(self.obj(regs, *it));
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
            Op::TensorSize { t, dim } => {
                let mut info = TcfgTensorInfo::empty();
                let size = match capi::api() {
                    Some(api) if (api.tensor_info)(reg(regs, *t), &mut info) != 0 => info.size(*dim),
                    _ => None,
                };
                match size {
                    Some(size) => checked(ffi::PyLong_FromLongLong(size)),
                    None => {
                        set_error(
                            ffi::PyExc_SystemError,
                            "cfg_compile: tensor_size ran without its guard".to_string(),
                        );
                        Err(())
                    }
                }
            }
            Op::TensorDim(t) => {
                let mut info = TcfgTensorInfo::empty();
                match capi::api() {
                    Some(api) if (api.tensor_info)(reg(regs, *t), &mut info) != 0 => {
                        checked(ffi::PyLong_FromLong(info.ndim as _))
                    }
                    _ => {
                        set_error(
                            ffi::PyExc_SystemError,
                            "cfg_compile: tensor_dim ran without its guard".to_string(),
                        );
                        Err(())
                    }
                }
            }
            Op::Dispatch(d) => match capi::api() {
                Some(api) => self.exec_dispatch(regs, &d.slot, &d.operands, &d.fallback, api),
                None => self.exec_op(regs, &d.fallback),
            },
            Op::Guard(_) => Ok(null_mut()),
        }
    }

    // ------------------------------------------------------------------
    // Dispatch sites
    // ------------------------------------------------------------------

    /// The receiver of a call whose callee is `x.name` in the generic graph.
    fn method_receiver(&self, callee: V) -> Option<V> {
        match self.analysis.def_op(&self.func, callee) {
            Some(Op::LoadAttr(recv, _)) => Some(*recv),
            _ => None,
        }
    }

    /// The operands a site of the given kind takes for `op`, in the order the
    /// C++ side expects them.
    pub fn site_operands(&self, kind: i32, op: &Op) -> Option<Vec<V>> {
        match (kind, op) {
            (capi::TCFG_BINARY, Op::Binary(_, a, b)) | (capi::TCFG_BINARY, Op::Compare(_, a, b)) => {
                Some(vec![*a, *b])
            }
            (capi::TCFG_UNARY, Op::Unary(_, a))
            | (capi::TCFG_UNARY, Op::ToBool(a))
            | (capi::TCFG_UNARY, Op::LoadAttr(a, _)) => Some(vec![*a]),
            (capi::TCFG_CALL, Op::Call { callee, args, .. }) => {
                let mut v = vec![*callee];
                v.extend(args);
                Some(v)
            }
            (capi::TCFG_METHOD, Op::Call { callee, args, .. }) => {
                let mut v = vec![self.method_receiver(*callee)?];
                v.extend(args);
                Some(v)
            }
            (capi::TCFG_METHOD, Op::CallMethod { recv, args, .. }) => {
                let mut v = vec![*recv];
                v.extend(args);
                Some(v)
            }
            _ => None,
        }
    }

    /// The objects a site takes for the generic `op`. In generic code the
    /// receiver of a method call is read from the bound method, since its own
    /// register can be dead by the time of the call. `method` is the bound
    /// method of a CallMethod. Returns None if an operand is unbound.
    unsafe fn operand_ptrs(
        &self,
        regs: &mut [Obj],
        kind: i32,
        op: &Op,
        operands: &[V],
    ) -> Option<Vec<Obj>> {
        // The capture is matched against Python objects.
        let mut ptrs: Vec<Obj> = operands.iter().map(|v| self.obj(regs, *v)).collect();
        if let (capi::TCFG_METHOD, Op::Call { callee, .. }) = (kind, op) {
            let c = self.obj(regs, *callee);
            if c.is_null() || ffi::PyCFunction_Check(c) == 0 {
                return None;
            }
            ptrs[0] = ffi::PyCFunction_GetSelf(c);
        }
        ptrs.iter().all(|p| !p.is_null()).then_some(ptrs)
    }

    /// Decides whether `op`, about to run on the current registers, could be
    /// a single tensor operation, and of which kind.
    unsafe fn site_kind(&self, regs: &[Obj], op: &Op, api: &TcfgApi) -> Option<i32> {
        let is_tensor = |v: V| -> bool {
            let o = reg(regs, v);
            let mut info = TcfgTensorInfo::empty();
            !o.is_null() && (api.tensor_info)(o, &mut info) != 0
        };
        match op {
            Op::Binary(_, a, b) | Op::Compare(_, a, b) => {
                (is_tensor(*a) || is_tensor(*b)).then_some(capi::TCFG_BINARY)
            }
            Op::Unary(UnaryKind::Neg | UnaryKind::Pos | UnaryKind::Invert, a)
            | Op::ToBool(a)
            | Op::LoadAttr(a, _) => is_tensor(*a).then_some(capi::TCFG_UNARY),
            Op::Call { callee, .. } => {
                let c = reg(regs, *callee);
                if c.is_null() || capi::is_unboxed(c) || ffi::PyCFunction_Check(c) == 0 {
                    return None;
                }
                // x.f(...) on a tensor: the callee is a builtin bound to the
                // receiver. The receiver's register may already be released
                // (the bound method keeps the object alive), so ask the
                // method.
                let this = ffi::PyCFunction_GetSelf(c);
                let mut info = TcfgTensorInfo::empty();
                match self.method_receiver(*callee) {
                    Some(recv)
                        if !this.is_null()
                            && (api.tensor_info)(this, &mut info) != 0
                            && (reg(regs, recv).is_null() || reg(regs, recv) == this) =>
                    {
                        Some(capi::TCFG_METHOD)
                    }
                    _ => Some(capi::TCFG_CALL),
                }
            }
            Op::CallMethod { recv, .. } => is_tensor(*recv).then_some(capi::TCFG_METHOD),
            _ => None,
        }
    }

    /// Runs the generic `op` under a dispatch capture and tries to turn what
    /// was recorded into a call site. Returns the result of the operation and
    /// the site (or the reason there is none).
    unsafe fn capture_exec(
        &self,
        regs: &mut [Obj],
        op: &Op,
        kind: i32,
        api: &TcfgApi,
    ) -> (OpResult, *mut capi::TcfgSite, Option<String>) {
        let Some(operands) = self.site_operands(kind, op) else {
            return (self.exec_op(regs, op), null_mut(), None);
        };
        // Operands are borrowed from the registers, which the operation does
        // not touch; read them first so a call sees its callee.
        let ptrs_before = self.operand_ptrs(regs, kind, op, &operands);
        (api.capture_begin)();
        // A method call needs the bound method it looked up for the analysis.
        let mut method: Obj = null_mut();
        let result = match op {
            Op::CallMethod { recv, name, args, kwnames } => {
                method = ffi::PyObject_GetAttr(
                    self.obj(regs, *recv),
                    self.func.consts[*name as usize].as_ptr(),
                );
                if method.is_null() {
                    Err(())
                } else {
                    checked(self.vectorcall(regs, method, args, *kwnames))
                }
            }
            _ => self.exec_op(regs, op),
        };
        let trace = (api.capture_end)();
        let mut site = null_mut();
        let mut reason = None;
        if !trace.is_null() {
            if let Ok(obj) = result {
                if let (false, Some(ptrs)) = (obj.is_null(), ptrs_before) {
                    let consts: Vec<u8> = operands
                        .iter()
                        .map(|v| matches!(self.analysis.def_op(&self.func, *v), Some(Op::Const(_))) as u8)
                        .collect();
                    let (callee, kwnames) = match op {
                        Op::Call { callee, kwnames, .. } => (self.obj(regs, *callee), *kwnames),
                        Op::CallMethod { kwnames, .. } => (method, *kwnames),
                        _ => (null_mut(), NONE),
                    };
                    let pyop = TcfgPyOp {
                        kind,
                        n_operands: ptrs.len() as i32,
                        operands: ptrs.as_ptr(),
                        is_const: consts.as_ptr(),
                        kwnames: if kwnames == NONE {
                            null_mut()
                        } else {
                            self.func.consts[kwnames as usize].as_ptr()
                        },
                        callee,
                        result: obj,
                        flags: if self.options.skip_autograd {
                            capi::TCFG_ALLOW_SKIP_AUTOGRAD
                        } else {
                            0
                        },
                    };
                    let mut buf = [0 as c_char; 160];
                    site = (api.site_from_trace)(trace, &pyop, buf.as_mut_ptr(), buf.len());
                    if site.is_null() {
                        reason = Some(
                            std::ffi::CStr::from_ptr(buf.as_ptr()).to_string_lossy().into_owned(),
                        );
                    }
                }
            }
            (api.trace_free)(trace);
        }
        xdecref(method);
        (result, site, reason)
    }

    /// Calls the kernel of the first site whose guards accept the operands.
    /// When none does, the generic operation runs, under a capture so the
    /// slot can learn the new case.
    #[inline(always)]
    unsafe fn exec_dispatch(
        &self,
        regs: &mut [Obj],
        slot: &SiteSlot,
        operands: &[V],
        fallback: &Op,
        api: &TcfgApi,
    ) -> OpResult {
        let mut small: [Obj; 12] = [null_mut(); 12];
        let big: Vec<Obj>;
        let ptr: *const Obj = if operands.len() <= small.len() {
            for (i, v) in operands.iter().enumerate() {
                small[i] = reg(regs, *v);
            }
            small.as_ptr()
        } else {
            big = operands.iter().map(|v| reg(regs, *v)).collect();
            big.as_ptr()
        };
        let flags = if self.options.unboxed { capi::TCFG_UNBOXED_RESULT } else { 0 };
        self.exec_slot(regs, slot, ptr, flags, fallback, api)
    }

    /// `ptr` points at the operands, which may be unboxed tensors. With
    /// TCFG_UNBOXED_RESULT in `flags` the result may be one as well.
    unsafe fn exec_slot(
        &self,
        regs: &mut [Obj],
        slot: &SiteSlot,
        ptr: *const Obj,
        flags: u32,
        fallback: &Op,
        api: &TcfgApi,
    ) -> OpResult {
        let n = slot.len();
        for i in 0..n {
            let mut status = 0i32;
            let r = (api.site_call)(slot.get(i), ptr, flags, &mut status);
            if !r.is_null() {
                slot.hits.fetch_add(1, Ordering::Relaxed);
                if capi::is_unboxed(r) {
                    self.unboxed_results.fetch_add(1, Ordering::Relaxed);
                }
                return Ok(r);
            }
            if status != capi::TCFG_MISS {
                return Err(());
            }
        }
        let misses = slot.misses.fetch_add(1, Ordering::Relaxed);
        if n < self.options.max_site_variants && misses < MAX_CAPTURE_ATTEMPTS as u64 * 4 {
            let (result, site, _) = self.capture_exec(regs, fallback, slot.kind, api);
            if !site.is_null() {
                slot.push(site);
            }
            return result;
        }
        self.exec_op(regs, fallback)
    }

    /// One instruction in the profiling tier.
    unsafe fn exec_profiled(
        &self,
        b: usize,
        index: usize,
        inst: &Inst,
        regs: &mut [Obj],
        api: &TcfgApi,
    ) -> OpResult {
        let (slot, attempts) = {
            let st = self.state.borrow();
            let bs = &st.blocks[b];
            (bs.slots[index].clone(), bs.capture_attempts[index])
        };
        let result = if let Some(slot) = slot {
            let ptrs = self
                .site_operands(slot.kind, &inst.op)
                .and_then(|operands| self.operand_ptrs(regs, slot.kind, &inst.op, &operands));
            match ptrs {
                Some(ptrs) => self.exec_slot(regs, &slot, ptrs.as_ptr(), 0, &inst.op, api),
                None => self.exec_op(regs, &inst.op),
            }
        } else if attempts >= MAX_CAPTURE_ATTEMPTS {
            self.exec_op(regs, &inst.op)
        } else {
            match self.site_kind(regs, &inst.op, api) {
                None => self.exec_op(regs, &inst.op),
                Some(kind) => {
                    let (result, site, reason) = self.capture_exec(regs, &inst.op, kind, api);
                    let mut st = self.state.borrow_mut();
                    let bs = &mut st.blocks[b];
                    bs.capture_attempts[index] += 1;
                    if !site.is_null() {
                        let slot = SiteSlot::new(kind);
                        slot.push(site);
                        bs.slots[index] = Some(Arc::new(slot));
                        bs.reasons[index] = None;
                    } else if reason.is_some() {
                        bs.reasons[index] = reason;
                    }
                    result
                }
            }
        };
        if let Ok(obj) = result {
            if inst.dst != NONE {
                self.state.borrow_mut().obs[inst.dst as usize].observe(obj);
            }
        }
        result
    }

    // ------------------------------------------------------------------
    // Guards, versions, specialization
    // ------------------------------------------------------------------

    #[inline(always)]
    unsafe fn guard_holds(cond: &GuardCond, regs: &[Obj], api: Option<&TcfgApi>) -> bool {
        let Some(api) = api else {
            return false;
        };
        let obj = reg(regs, cond.value());
        let mut info = TcfgTensorInfo::empty();
        if obj.is_null() || (api.tensor_info)(obj, &mut info) == 0 {
            return false;
        }
        match cond {
            GuardCond::TensorRank { min_rank, .. } => info.ndim as u32 >= *min_rank,
            GuardCond::DimRange { dim, lo, hi, .. } => {
                matches!(info.size(*dim), Some(size) if size >= *lo && size <= *hi)
            }
        }
    }

    unsafe fn select(
        &self,
        b: usize,
        regs: &[Obj],
        api: Option<&TcfgApi>,
    ) -> (Arc<BlockCode>, Mode) {
        if !self.options.specialize || api.is_none() {
            return (self.generic[b].clone(), Mode::Generic);
        }
        let mut st = self.state.borrow_mut();
        let bs = &mut st.blocks[b];
        bs.execs += 1;
        for version in &bs.versions {
            if version.entry_guards.iter().all(|g| Self::guard_holds(g, regs, api)) {
                bs.spec_execs += 1;
                return (version.clone(), Mode::Spec);
            }
        }
        if !bs.versions.is_empty() {
            bs.entry_misses += 1;
        }
        let mode = if bs.respec_left > 0 { Mode::Profile } else { Mode::Generic };
        if mode == Mode::Profile {
            // What the block is entered with is part of its profile.
            let live: Vec<V> = self.analysis.live_in[b].iter().collect();
            for v in live {
                st.obs[v as usize].observe(reg(regs, v));
            }
        }
        (self.generic[b].clone(), mode)
    }

    /// Called when a block finished a run in the profiling tier: runs the
    /// pass pipeline over a fresh copy of the block and installs the result
    /// as a new version.
    unsafe fn specialize(&self, b: usize) {
        let (slots, obs, speculate, id) = {
            let mut st = self.state.borrow_mut();
            let obs = st.obs.clone();
            let bs = &mut st.blocks[b];
            if bs.respec_left == 0 {
                return;
            }
            bs.respec_left -= 1;
            bs.profile_runs += 1;
            let id = bs.next_version_id;
            bs.next_version_id += 1;
            // The last attempt must hold for every input.
            (bs.slots.clone(), obs, bs.respec_left > 0, id)
        };
        let mut unit = Unit::from_generic(&self.func, b, speculate);
        let cx = PassCx { cf: self, slots: &slots, obs: &obs };
        let outcome = match self.pipeline.run(&mut unit, &cx) {
            Ok(true) => spec::finalize(unit, self, id).map(Some),
            Ok(false) => Ok(None),
            Err(err) => {
                // A failing pass must not take the program down with it: the
                // block simply stays generic.
                let msg = pyo3::Python::with_gil(|py| err.value_bound(py).to_string());
                Err(format!("pass failed: {}", msg))
            }
        };
        let mut st = self.state.borrow_mut();
        match outcome {
            Ok(Some(code)) => {
                let guarded = !code.entry_guards.is_empty()
                    || code.insts.iter().any(|i| matches!(i.op, Op::Guard(_)));
                let bs = &mut st.blocks[b];
                bs.versions.insert(0, Arc::new(code));
                bs.versions.truncate(self.options.max_versions);
                if !guarded {
                    // Nothing can send this version back to profiling.
                    bs.versions.truncate(1);
                }
            }
            Ok(None) => st.blocks[b].respec_left = 0,
            Err(msg) => {
                st.blocks[b].respec_left = 0;
                st.blocks[b].rejected = Some(msg.clone());
                st.log.push(format!("b{}: {}", b, msg));
            }
        }
    }

    // ------------------------------------------------------------------
    // The interpreter loop
    // ------------------------------------------------------------------

    unsafe fn exec(&self, regs: &mut [Obj]) -> Obj {
        let api = capi::api();
        let func = &self.func;
        if self.options.specialize {
            self.state.borrow_mut().calls += 1;
        }
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
                        capi::retain(o);
                        vals[i] = o;
                    }
                    for (i, (dst, _)) in moves.iter().enumerate() {
                        let old = std::mem::replace(regs.get_unchecked_mut(*dst as usize), vals[i]);
                        capi::release(old);
                    }
                }
                for v in self.analysis.edge_kills.get_unchecked(cur).get_unchecked(pi) {
                    capi::release(take(regs, *v));
                }
                if cur <= pred {
                    back_edges = back_edges.wrapping_add(1);
                    if back_edges & 63 == 0 && ffi::PyErr_CheckSignals() != 0 {
                        return self.fail(block.bc_start);
                    }
                }
            }

            let (mut code, mut mode) = self.select(cur, regs, api);
            let mut pc: usize = 0;
            let next: usize = loop {
                let mut resume: Option<usize> = None;
                {
                    let code_ref: &BlockCode = &code;
                    let insts = &code_ref.insts;
                    while pc < insts.len() {
                        let inst = insts.get_unchecked(pc);
                        let result = match &inst.op {
                            Op::Guard(g) => {
                                if Self::guard_holds(&g.cond, regs, api) {
                                    Ok(null_mut())
                                } else {
                                    resume = Some(g.resume as usize);
                                    break;
                                }
                            }
                            op => {
                                if mode == Mode::Profile {
                                    self.exec_profiled(cur, pc, inst, regs, api.unwrap())
                                } else {
                                    self.exec_op(regs, op)
                                }
                            }
                        };
                        match result {
                            Ok(obj) => {
                                if inst.dst != NONE {
                                    let old = std::mem::replace(
                                        regs.get_unchecked_mut(inst.dst as usize),
                                        obj,
                                    );
                                    capi::release(old);
                                }
                                for v in &inst.kills {
                                    capi::release(take(regs, *v));
                                }
                            }
                            Err(()) => return self.fail(inst.bc),
                        }
                        pc += 1;
                    }
                    if resume.is_none() {
                        if mode == Mode::Profile {
                            self.specialize(cur);
                        }
                        let next = match &code_ref.term {
                            Term::Jump(b) => *b as usize,
                            Term::Br { cond, then_, else_ } => {
                                let c = self.obj(regs, *cond);
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
                            Term::Return(v) => {
                                // What leaves the VM is a Python object.
                                self.obj(regs, *v);
                                return take(regs, *v);
                            }
                            Term::Raise { exc, cause } => {
                                let cause =
                                    if *cause == NONE { null_mut() } else { self.obj(regs, *cause) };
                                do_raise(self.obj(regs, *exc), cause);
                                let bc = code_ref.insts.last().map_or(func.blocks[cur].bc_start, |i| i.bc);
                                return self.fail(bc);
                            }
                        };
                        for v in &code_ref.term_kills {
                            capi::release(take(regs, *v));
                        }
                        break next;
                    }
                }
                // A guard failed: this block goes back to the generic tier,
                // from the instruction the guard names.
                let mut st = self.state.borrow_mut();
                st.blocks[cur].deopts += 1;
                mode = if st.blocks[cur].respec_left > 0 { Mode::Profile } else { Mode::Generic };
                drop(st);
                code = self.generic[cur].clone();
                pc = resume.unwrap();
            };
            pred = cur;
            cur = next;
        }
    }
}
