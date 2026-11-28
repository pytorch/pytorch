// All C++ FFI declarations for this crate live in this single file because
// Buck's `rust_python_extension` accepts only one `cxx_bridge` source. To add
// FFI for a new type, declare another `#[cxx::bridge] pub mod <name> { ... }`
// block alongside the existing ones and put the Rust-side impls in a sibling
// file under `bindings/`.

//
// THE RULE OF THE BRIDGE: no C++ exception leaves a function declared here.
// The shim cxx makes for a function that does not return Result is noexcept:
// an exception there ends the process, whatever the program was doing. So a
// function declared in an `extern "C++"` block is one of three kinds, and
// test_vm_gate.py (TestVmBridge) holds the list, with the reason of each:
//   edge    its body is inside `edge(...)` or `edge_py(...)`
//           (csrc/tensor_ffi.cpp), which catch everything (a c10::Error from
//           a torch call, bad_alloc, a Python error turned into an
//           exception) and answer the function's refusing value: the one its
//           callers take for "not a tensor", "no key", "no word", "not
//           neutral", a miss. `edge_py` where the body may run Python code
//           or raise a Python error: an error pending before the call is put
//           back after it. Every function that has a refusing value is of
//           this kind, whether or not its body can throw today;
//   result  it returns Result, and the caller's Err branch refuses;
//   plain   it has no refusing value (both answers of a bool are answers,
//           a count is a count) and its body cannot throw: it reads a
//           thread-local or a field and calls nothing that checks,
//           allocates or runs Python code. The test records the text of
//           each such body: one that changes fails until it is read again.
// A new function is added to that list with its kind. The functions of the
// `extern "Rust"` blocks go the other way: a panic in one ends the process
// (cxx's own guard), so they do not panic. The `extern "C"` blocks elsewhere
// in the crate declare C code and CPython's API, which throw nothing.

#[cxx::bridge(namespace = "torch::rust")]
pub mod ffi {
    // See tensor_meta in tensor_ffi.h. The device and dtype are c10's enums.
    #[derive(Default)]
    struct TensorMeta {
        defined: bool,
        dtype: i8,
        requires_grad: bool,
        has_device: bool,
        device_type: i8,
        device_index: i8,
        has_shape: bool,
        sizes: Vec<i64>,
    }

    // See tensor_key in tensor_ffi.h.
    #[derive(Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
    struct TensorFacts {
        dtype: i8,
        device_type: i8,
        device_index: i8,
        bits: u8,
    }

    // See region_word in tensor_ffi.h.
    #[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
    struct Word {
        a: u64,
        b: u64,
    }

    // See tensor_span in tensor_ffi.h.
    #[derive(Default)]
    struct TensorSpan {
        storage: usize,
        // Bytes from the storage's start to the first element.
        at: usize,
        // The device type whose memory `low` and `high` are addresses of.
        space: u8,
        low: usize,
        high: usize,
    }

    unsafe extern "C++" {
        include!("torch/_rust/csrc/tensor_ffi.h");

        type Tensor;

        unsafe fn tensor_from_pyobject(py_obj: usize) -> *const Tensor;
        unsafe fn tensor_by_type(py_obj: usize) -> *const Tensor;

        fn torch_function_mode_active() -> bool;
        fn grad_mode_enabled() -> bool;
        fn region_word(out: &mut Word) -> bool;
        fn region_not_neutral() -> u8;
        fn set_grad_mode(on: bool) -> bool;
        fn tensor_version(t: &Tensor) -> i64;
        fn tensor_home(t: &Tensor, storage: &mut usize) -> usize;
        fn tensor_span(t: &Tensor, out: &mut TensorSpan) -> bool;
        fn tensor_is_cow(t: &Tensor) -> bool;
        fn forward_ad_active() -> bool;
        fn saved_tensor_hooks_active() -> bool;
        fn tensor_key(t: &Tensor, out: &mut TensorFacts) -> bool;
        fn tensor_key_is(t: &Tensor, want: &TensorFacts, sizes: &[i64], strides: &[i64]) -> bool;

        fn tensor_sizes(t: &Tensor) -> Result<&[i64]>;
        fn tensor_strides(t: &Tensor) -> Result<&[i64]>;

        fn tensor_is_contiguous(t: &Tensor) -> bool;
        fn tensor_is_cpu(t: &Tensor) -> bool;
        fn tensor_requires_grad(t: &Tensor) -> bool;
        fn tensor_is_leaf(t: &Tensor) -> bool;

        fn tensor_meta(t: &Tensor, out: &mut TensorMeta);
        fn dtype_name(dtype: i8) -> String;
        fn device_name(device_type: i8, device_index: i8) -> String;
    }
}

#[cxx::bridge(namespace = "torch::rust")]
pub mod op_hook {
    // See tensor_ident in op_hook.h.
    #[derive(Default, Clone, Copy)]
    struct TensorIdent {
        tensor: usize,
        storage: usize,
    }

    // An argument an op's schema says it writes: its position, the tensor (a
    // member of the list, for a list of tensors), and the serial of the op
    // that made its storage while a witness was open (0: none did).
    struct WitnessWrite {
        arg: u32,
        id: TensorIdent,
        made: u64,
    }

    // What on_witness_op is told of an op, as bits of its `flags`.
    #[repr(u32)]
    enum WitnessFlag {
        // No schema, or an argument or a written value it does not account for.
        Unreadable = 1,
        // Tagged nondeterministic_seeded.
        Draws = 2,
        // Tagged data_dependent_output.
        ReadsValue = 4,
        // A result that is no tensor, list of tensors or optional tensor.
        ValueOut = 8,
        // Tagged dynamic_output_shape.
        DataSized = 16,
    }

    unsafe extern "C++" {
        include!("torch/_rust/csrc/op_hook.h");

        fn op_hook_enable();
        fn op_hook_disable();
        fn op_hook_pause(paused: bool);
        fn op_hook_witness(wanted: bool, open: bool) -> bool;
        fn op_hook_seeing() -> bool;
        fn op_hook_depth() -> u32;
        fn op_hook_serial() -> u64;
        fn op_hook_innermost() -> u64;
        fn tensor_ident(py_obj: usize, out: &mut TensorIdent);
    }

    extern "Rust" {
        fn on_torch_op(op: &str, args: &str) -> Result<()>;
        fn on_witness_op(name: &str, overload: &str, depth: u32, serial: u64, flags: u32, wrote: &[WitnessWrite]);
        fn on_witness_made(storage: usize, since: u64, by: u64);
    }
}

#[cxx::bridge(namespace = "torch::rust")]
pub mod autograd {
    extern "Rust" {
        type NodeState;
        // See CfgBackward in autograd_node.cpp: `grads` holds borrowed
        // PyObject*s (a tensor, or None), `out` gets a new reference per
        // input (a tensor, or None). False with a Python exception set on
        // failure.
        fn node_apply(state: &mut NodeState, grads: &[usize], out: &mut Vec<usize>) -> bool;
        fn node_release(state: &mut NodeState);
    }

    unsafe extern "C++" {
        include!("torch/_rust/csrc/autograd_node.h");

        unsafe fn make_cfg_node(state: Box<NodeState>, inputs: &[usize], outputs: &[usize], wrapped: &mut Vec<usize>) -> bool;
    }
}

use crate::exec::autograd::{node_apply, node_release, NodeState};
use crate::exec::calls::witness::{on_witness_made, on_witness_op};
use crate::exec::torch_ops::on_torch_op;

pub use self::ffi::*;
