//! Passes for `torch.vm`, written in Rust.
//!
//! A pass is a type that implements [`Pass`]. A library of passes is a
//! `cdylib` crate that depends on this one and ends with
//! `export_passes!(First, Second);`. Python loads it with
//! `torch.vm.load_passes(path)`, which gives a dict of the passes by
//! name, each of them an entry for `passes=`:
//!
//! ```ignore
//! use torch_vm_pass::{export_passes, Bound, Error, Pass, Region};
//!
//! #[derive(Default)]
//! struct FuseRmsNorm;
//!
//! impl Pass for FuseRmsNorm {
//!     fn name(&self) -> &'static str {
//!         "fuse_rmsnorm"
//!     }
//!
//!     fn run(&self, region: &mut Region<'_>, kernel: Option<Bound>) -> Result<bool, Error> {
//!         let Some(kernel) = kernel else { return Ok(false) };
//!         match region.find(&["pow", "mean", "add", "rsqrt", "mul"]) {
//!             Some((start, end)) => region.swap(start, end, kernel, 8),
//!             None => Ok(false),
//!         }
//!     }
//! }
//!
//! export_passes!(FuseRmsNorm);
//! ```
//!
//! The crate has no dependencies and does not link against Python or
//! PyTorch: everything goes through a table of functions the runtime hands
//! the pass, so a library built once works with any build of PyTorch that
//! has the same [`ABI_VERSION`].

use std::ffi::{c_char, c_void, CString};

/// What `torch_vm_abi` of a library must return for it to be loaded.
pub const ABI_VERSION: u32 = 1;

/// The functions of the runtime, by which a pass looks at a basic block and
/// rewrites it. `region` is what the pass was called with.
#[repr(C)]
pub struct Api {
    pub version: u32,
    /// How many instructions the block has.
    pub len: unsafe extern "C" fn(region: *mut c_void) -> usize,
    /// The kind of instruction `index` ("call", "binary", "dispatch"), as
    /// UTF-8 into `buf`; returns its length, which may exceed `cap`.
    pub op: unsafe extern "C" fn(region: *mut c_void, index: usize, buf: *mut u8, cap: usize) -> usize,
    /// The operator the instruction was seen to be exactly one call of
    /// ("aten::mul.Tensor"); 0 if it was not.
    pub kernel: unsafe extern "C" fn(region: *mut c_void, index: usize, buf: *mut u8, cap: usize) -> usize,
    /// The value the instruction defines, or `u32::MAX`.
    pub dst: unsafe extern "C" fn(region: *mut c_void, index: usize) -> u32,
    /// The values it uses, into `buf`; returns their number.
    pub uses: unsafe extern "C" fn(region: *mut c_void, index: usize, buf: *mut u32, cap: usize) -> usize,
    /// Whether it is still the instruction the function wrote and can be run
    /// again on other values.
    pub replayable: unsafe extern "C" fn(region: *mut c_void, index: usize) -> i32,
    /// Replaces instructions `start..end` by a call of the Python callable
    /// with the values the range takes, which returns a tuple of the values
    /// it gives. 1 if replaced, 0 if not, -1 with a Python error set.
    pub fuse: unsafe extern "C" fn(region: *mut c_void, start: usize, end: usize, callable: *mut c_void) -> i32,
    /// `fuse` with a kernel that returns what the range gives (one value, or
    /// a tuple), checked against the range itself for the first `checks`
    /// calls, as `torch.vm.swap` checks it.
    pub swap: unsafe extern "C" fn(
        region: *mut c_void,
        start: usize,
        end: usize,
        kernel: *mut c_void,
        checks: u32,
    ) -> i32,
    /// Sets a Python RuntimeError with this message.
    pub error: unsafe extern "C" fn(message: *const c_char),
}

/// One pass of a library, as the runtime sees it.
#[repr(C)]
pub struct RawPass {
    pub name: *const c_char,
    pub run: unsafe extern "C" fn(state: *mut c_void, region: *mut c_void, api: *const Api, bound: *mut c_void) -> i32,
    pub state: *mut c_void,
}

unsafe impl Send for RawPass {}
unsafe impl Sync for RawPass {}

/// A Python error is set: the pass returns it to the runtime as it is.
#[derive(Debug)]
pub struct Error;

/// The Python object a pass was bound to with `handle.bind(obj)`: a kernel,
/// as a rule. It is only handed back to the runtime.
#[derive(Clone, Copy)]
pub struct Bound(*mut c_void);

/// One instruction of a basic block.
#[derive(Clone, Debug)]
pub struct Inst {
    pub index: usize,
    /// "call", "binary", "load_attr", "const", "dispatch", ...
    pub op: String,
    /// The operator it was seen to be exactly one call of.
    pub kernel: Option<String>,
    pub dst: Option<u32>,
    pub uses: Vec<u32>,
    pub replayable: bool,
}

/// The basic block a pass is rewriting. Valid while the pass runs.
pub struct Region<'a> {
    raw: *mut c_void,
    api: &'a Api,
}

/// Instructions that make a value out of nothing the function computed.
const SETUP: [&str; 4] = ["const", "load_global", "load_attr", "load_method"];

impl Region<'_> {
    pub fn len(&self) -> usize {
        unsafe { (self.api.len)(self.raw) }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn text(&self, read: unsafe extern "C" fn(*mut c_void, usize, *mut u8, usize) -> usize, index: usize) -> String {
        let mut buf = [0u8; 128];
        let n = unsafe { read(self.raw, index, buf.as_mut_ptr(), buf.len()) }.min(buf.len());
        String::from_utf8_lossy(&buf[..n]).into_owned()
    }

    pub fn inst(&self, index: usize) -> Inst {
        let kernel = self.text(self.api.kernel, index);
        let mut uses = vec![0u32; 16];
        let n = unsafe { (self.api.uses)(self.raw, index, uses.as_mut_ptr(), uses.len()) };
        if n > uses.len() {
            uses.resize(n, 0);
            unsafe { (self.api.uses)(self.raw, index, uses.as_mut_ptr(), uses.len()) };
        }
        uses.truncate(n);
        let dst = unsafe { (self.api.dst)(self.raw, index) };
        Inst {
            index,
            op: self.text(self.api.op, index),
            kernel: (!kernel.is_empty()).then_some(kernel),
            dst: (dst != u32::MAX).then_some(dst),
            uses,
            replayable: unsafe { (self.api.replayable)(self.raw, index) } != 0,
        }
    }

    pub fn insts(&self) -> Vec<Inst> {
        (0..self.len()).map(|i| self.inst(i)).collect()
    }

    /// The first run of instructions that are exactly these operators, in
    /// this order, with nothing between them that calls a kernel or cannot
    /// be run again: `start..end`, with what only sets the calls up in front
    /// (constants, globals). An operator is named as the runtime reports it
    /// ("aten::mul.Tensor"), or without namespace or overload ("mul").
    pub fn find(&self, ops: &[&str]) -> Option<(usize, usize)> {
        let insts = self.insts();
        let found: Vec<&Inst> = insts.iter().filter(|i| i.kernel.is_some()).collect();
        if ops.is_empty() || found.len() < ops.len() {
            return None;
        }
        for at in 0..=found.len() - ops.len() {
            let run = &found[at..at + ops.len()];
            if !ops.iter().zip(run).all(|(want, inst)| same(want, inst.kernel.as_deref().unwrap_or(""))) {
                continue;
            }
            let (start, end) = (run[0].index, run[ops.len() - 1].index + 1);
            if !insts[start..end].iter().all(|i| i.replayable) {
                continue;
            }
            // What only sets a call up comes along (a constant, a global, the
            // method of a tensor the run has anyway): the setup in front of
            // the run, as far back as it takes nothing new from the function
            // and as far as the run uses it.
            let setup = |i: &Inst| SETUP.contains(&i.op.as_str()) && i.replayable;
            let mut low = start;
            while low > 0 && setup(&insts[low - 1]) {
                low -= 1;
            }
            let made: Vec<u32> = insts[start..end].iter().filter_map(|i| i.dst).collect();
            let taken: Vec<u32> =
                insts[start..end].iter().flat_map(|i| i.uses.iter().copied()).filter(|u| !made.contains(u)).collect();
            let mut first = start;
            for k in low..start {
                let known = |u: &u32| taken.contains(u) || insts[k..start].iter().any(|i| i.dst == Some(*u));
                if insts[k..start].iter().all(|i| i.uses.iter().all(known)) {
                    first = k;
                    break;
                }
            }
            let used = |from: usize| {
                insts[from].dst.is_some_and(|d| insts[from..end].iter().any(|i| i.uses.contains(&d)))
            };
            while first < start && !used(first) {
                first += 1;
            }
            return Some((first, end));
        }
        None
    }

    fn done(&self, status: i32) -> Result<bool, Error> {
        match status {
            1 => Ok(true),
            0 => Ok(false),
            _ => Err(Error),
        }
    }

    /// Replaces `start..end` by a call of a Python callable that takes the
    /// values the range takes and returns a tuple of the values it gives.
    pub fn fuse(&mut self, start: usize, end: usize, callable: Bound) -> Result<bool, Error> {
        self.done(unsafe { (self.api.fuse)(self.raw, start, end, callable.0) })
    }

    /// Replaces `start..end` by a call of `kernel`, which is checked against
    /// the instructions it replaces for its first `checks` calls.
    pub fn swap(&mut self, start: usize, end: usize, kernel: Bound, checks: u32) -> Result<bool, Error> {
        self.done(unsafe { (self.api.swap)(self.raw, start, end, kernel.0, checks) })
    }

    /// Fails the pass with a message, as a Python RuntimeError.
    pub fn fail<T>(&self, message: &str) -> Result<T, Error> {
        let text = CString::new(message.replace('\0', " ")).unwrap_or_default();
        unsafe { (self.api.error)(text.as_ptr()) };
        Err(Error)
    }
}

fn same(wanted: &str, found: &str) -> bool {
    let strip = |name: &str| name.rsplit("::").next().unwrap_or(name).to_string();
    let (wanted, full) = (strip(wanted), strip(found));
    wanted == full || full.split('.').next() == Some(wanted.as_str())
}

/// A pass over one basic block of a function, run each time the block is
/// specialized. It returns whether it changed the block.
pub trait Pass: Default + Send + Sync + 'static {
    fn name(&self) -> &'static str;
    fn run(&self, region: &mut Region<'_>, bound: Option<Bound>) -> Result<bool, Error>;
}

/// What `export_passes!` puts into the table for a pass type.
pub fn raw<P: Pass>() -> RawPass {
    unsafe extern "C" fn run<P: Pass>(
        state: *mut c_void,
        region: *mut c_void,
        api: *const Api,
        bound: *mut c_void,
    ) -> i32 {
        let api = &*api;
        let pass = &*(state as *const P);
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut region = Region { raw: region, api };
            pass.run(&mut region, (!bound.is_null()).then_some(Bound(bound)))
        }));
        match outcome {
            Ok(Ok(changed)) => changed as i32,
            Ok(Err(Error)) => -1,
            Err(_) => {
                (api.error)(c"a pass panicked".as_ptr());
                -1
            }
        }
    }
    let pass = P::default();
    let name = CString::new(pass.name()).unwrap_or_default().into_raw();
    RawPass { name, run: run::<P>, state: Box::into_raw(Box::new(pass)) as *mut c_void }
}

/// Exports the passes of a library: `export_passes!(First, Second);`.
#[macro_export]
macro_rules! export_passes {
    ($($pass:ty),+ $(,)?) => {
        #[no_mangle]
        pub extern "C" fn torch_vm_abi() -> u32 {
            $crate::ABI_VERSION
        }

        #[no_mangle]
        pub unsafe extern "C" fn torch_vm_passes(out: *mut *const $crate::RawPass) -> usize {
            static TABLE: std::sync::OnceLock<Vec<$crate::RawPass>> = std::sync::OnceLock::new();
            let table = TABLE.get_or_init(|| vec![$($crate::raw::<$pass>()),+]);
            *out = table.as_ptr();
            table.len()
        }
    };
}
