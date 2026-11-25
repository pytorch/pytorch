// All C++ FFI declarations for this crate live in this single file because
// Buck's `rust_python_extension` accepts only one `cxx_bridge` source. To add
// FFI for a new type, declare another `#[cxx::bridge] pub mod <name> { ... }`
// block alongside the existing ones and put the Rust-side impls in a sibling
// file under `bindings/`.

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

    // See tensor_span in tensor_ffi.h.
    #[derive(Default)]
    struct TensorSpan {
        storage: usize,
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
        fn region_thread_state() -> u64;
        fn set_grad_mode(on: bool) -> bool;
        fn tensor_version(t: &Tensor) -> i64;
        fn tensor_data_address(t: &Tensor) -> usize;
        fn tensor_span(t: &Tensor, out: &mut TensorSpan);
        fn forward_ad_active() -> bool;
        fn saved_tensor_hooks_active() -> bool;
        fn tensor_matches(
            t: &Tensor,
            dtype: i8,
            device_type: i8,
            device_index: i8,
            sizes: &[i64],
            strides: &[i64],
        ) -> bool;
        fn tensor_dim(t: &Tensor) -> i64;
        fn tensor_numel(t: &Tensor) -> i64;
        fn tensor_size_at(t: &Tensor, dim: i64) -> i64;
        fn tensor_stride_at(t: &Tensor, dim: i64) -> i64;

        fn tensor_sizes(t: &Tensor) -> &[i64];
        fn tensor_strides(t: &Tensor) -> &[i64];

        fn tensor_is_contiguous(t: &Tensor) -> bool;
        fn tensor_is_cpu(t: &Tensor) -> bool;
        fn tensor_is_cuda(t: &Tensor) -> bool;
        fn tensor_defined(t: &Tensor) -> bool;
        fn tensor_requires_grad(t: &Tensor) -> bool;

        fn tensor_meta(t: &Tensor, out: &mut TensorMeta);
        fn dtype_name(dtype: i8) -> String;
        fn device_name(device_type: i8, device_index: i8) -> String;
    }
}

#[cxx::bridge(namespace = "torch::rust")]
pub mod op_hook {
    unsafe extern "C++" {
        include!("torch/_rust/csrc/op_hook.h");

        fn op_hook_enable();
        fn op_hook_disable();
        fn op_hook_pause(paused: bool);
    }

    extern "Rust" {
        fn on_torch_op(op: &str, args: &str) -> Result<()>;
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
use crate::exec::torch_ops::on_torch_op;

pub use self::ffi::*;
