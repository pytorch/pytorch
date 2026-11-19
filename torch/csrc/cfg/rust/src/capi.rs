//! The C function table exported by torch/csrc/cfg/cfg.h, plus the owner of
//! the prepared call sites it hands out.

use crate::pyffi::{self, Obj, Shared};
use pyo3::ffi;
use std::os::raw::c_char;
use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};

#[repr(C)]
pub struct TcfgTrace {
    _private: [u8; 0],
}

#[repr(C)]
pub struct TcfgSite {
    _private: [u8; 0],
}

pub const TCFG_MISS: i32 = 1;

pub const TCFG_CALL: i32 = 0;
pub const TCFG_METHOD: i32 = 1;
pub const TCFG_BINARY: i32 = 2;
pub const TCFG_UNARY: i32 = 3;

pub const TCFG_ALLOW_SKIP_AUTOGRAD: u32 = 1;

pub const TCFG_UNBOXED_RESULT: u32 = 1;

pub const TCFG_API_VERSION: u32 = 2;

#[repr(C)]
pub struct TcfgPyOp {
    pub kind: i32,
    pub n_operands: i32,
    pub operands: *const Obj,
    pub is_const: *const u8,
    pub kwnames: Obj,
    pub callee: Obj,
    pub result: Obj,
    pub flags: u32,
}

#[repr(C)]
pub struct TcfgTensorInfo {
    pub ndim: i32,
    pub requires_grad: i32,
    pub sizes: *const i64,
    pub key_set: u64,
}

impl TcfgTensorInfo {
    pub fn empty() -> TcfgTensorInfo {
        TcfgTensorInfo { ndim: 0, requires_grad: 0, sizes: std::ptr::null(), key_set: 0 }
    }

    /// The size of dimension `dim` (negative counts from the end), if it
    /// exists.
    pub unsafe fn size(&self, dim: i64) -> Option<i64> {
        let d = if dim < 0 { dim + self.ndim as i64 } else { dim };
        if d < 0 || d >= self.ndim as i64 {
            return None;
        }
        Some(*self.sizes.offset(d as isize))
    }
}

#[repr(C)]
pub struct TcfgApi {
    pub version: u32,
    pub capture_begin: unsafe extern "C" fn(),
    pub capture_end: unsafe extern "C" fn() -> *mut TcfgTrace,
    pub trace_free: unsafe extern "C" fn(*mut TcfgTrace),
    pub trace_records: unsafe extern "C" fn(*const TcfgTrace) -> Obj,
    pub site_from_trace:
        unsafe extern "C" fn(*const TcfgTrace, *const TcfgPyOp, *mut c_char, usize) -> *mut TcfgSite,
    pub site_free: unsafe extern "C" fn(*mut TcfgSite),
    pub site_call: unsafe extern "C" fn(*mut TcfgSite, *const Obj, u32, *mut i32) -> Obj,
    pub site_info: unsafe extern "C" fn(*const TcfgSite) -> Obj,
    pub tensor_info: unsafe extern "C" fn(Obj, *mut TcfgTensorInfo) -> i32,
    pub tensor_box: unsafe extern "C" fn(Obj) -> Obj,
    pub tensor_retain: unsafe extern "C" fn(Obj),
    pub tensor_release: unsafe extern "C" fn(Obj),
}

static API: AtomicPtr<TcfgApi> = AtomicPtr::new(std::ptr::null_mut());

pub fn set_api(api: *mut TcfgApi) {
    API.store(api, Ordering::Release);
}

#[inline(always)]
pub fn api() -> Option<&'static TcfgApi> {
    let ptr = API.load(Ordering::Relaxed);
    if ptr.is_null() {
        None
    } else {
        Some(unsafe { &*ptr })
    }
}

/// Whether a register holds an unboxed tensor (TensorImpl* | 1, an owned
/// reference) rather than a Python object. Kernel results stay in this form
/// until something needs them as Python objects.
#[inline(always)]
pub fn is_unboxed(o: Obj) -> bool {
    (o as usize) & 1 != 0
}

/// Drops the reference a register holds, whatever its form.
#[inline(always)]
pub unsafe fn release(o: Obj) {
    if o.is_null() {
        return;
    }
    if is_unboxed(o) {
        (api().unwrap_unchecked().tensor_release)(o);
    } else {
        ffi::Py_DECREF(o);
    }
}

/// Takes another reference to what a register holds, whatever its form.
#[inline(always)]
pub unsafe fn retain(o: Obj) {
    if o.is_null() {
        return;
    }
    if is_unboxed(o) {
        (api().unwrap_unchecked().tensor_retain)(o);
    } else {
        ffi::Py_INCREF(o);
    }
}

/// The call sites of one instruction: a small polymorphic inline cache of
/// prepared kernel calls. Each variant was captured under one combination of
/// dispatch keys; a call tries them in order.
pub struct SiteSlot {
    pub kind: i32,
    variants: Shared<Vec<*mut TcfgSite>>,
    descriptions: Shared<Vec<String>>,
    pub hits: AtomicU64,
    pub misses: AtomicU64,
}

unsafe impl Send for SiteSlot {}
unsafe impl Sync for SiteSlot {}

impl SiteSlot {
    pub fn new(kind: i32) -> SiteSlot {
        SiteSlot {
            kind,
            variants: Shared::new(Vec::new()),
            descriptions: Shared::new(Vec::new()),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    pub fn len(&self) -> usize {
        self.variants.borrow().len()
    }

    #[inline(always)]
    pub fn get(&self, i: usize) -> *mut TcfgSite {
        self.variants.borrow()[i]
    }

    /// Takes ownership of the site.
    pub unsafe fn push(&self, site: *mut TcfgSite) {
        let description = describe_site(site);
        self.variants.borrow_mut().push(site);
        self.descriptions.borrow_mut().push(description);
    }

    pub fn describe(&self) -> String {
        self.descriptions.borrow().join(" | ")
    }

    /// New reference to a list with one info dict per variant.
    pub unsafe fn info(&self) -> Obj {
        let list = ffi::PyList_New(0);
        if let Some(api) = api() {
            let n = self.len();
            for i in 0..n {
                let info = (api.site_info)(self.get(i));
                if info.is_null() {
                    ffi::PyErr_Clear();
                    continue;
                }
                ffi::PyList_Append(list, info);
                ffi::Py_DECREF(info);
            }
        }
        list
    }
}

impl Drop for SiteSlot {
    fn drop(&mut self) {
        if let Some(api) = api() {
            for site in self.variants.borrow().iter() {
                unsafe { (api.site_free)(*site) };
            }
        }
    }
}

unsafe fn describe_site(site: *mut TcfgSite) -> String {
    let Some(api) = api() else {
        return "?".to_string();
    };
    let info = (api.site_info)(site);
    if info.is_null() {
        ffi::PyErr_Clear();
        return "?".to_string();
    }
    let get = |key: &str| -> Obj {
        let k = std::ffi::CString::new(key).unwrap();
        ffi::PyDict_GetItemString(info, k.as_ptr())
    };
    let op = get("op");
    let key = get("key");
    let chain = get("chain");
    let level = get("entry_level");
    let mut text = format!(
        "{}[{}]",
        if op.is_null() { "?".to_string() } else { pyffi::str_to_string(op) },
        if key.is_null() { "?".to_string() } else { pyffi::str_to_string(key) }
    );
    if !chain.is_null() && !level.is_null() {
        let level = ffi::PyLong_AsLong(level);
        let mut skipped = Vec::new();
        for i in 0..level {
            skipped.push(pyffi::str_to_string(ffi::PyList_GetItem(chain, i as isize)));
        }
        if !skipped.is_empty() {
            text.push_str(&format!(" skipping {}", skipped.join(", ")));
        } else if get("skips_autograd") == ffi::Py_True() {
            text.push_str(" with autograd excluded");
        }
    }
    ffi::Py_DECREF(info);
    text
}
