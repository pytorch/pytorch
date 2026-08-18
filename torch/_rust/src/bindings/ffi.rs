// All C++ FFI declarations for this crate live in this single file because
// Buck's `rust_python_extension` accepts only one `cxx_bridge` source. To add
// FFI for a new type, declare another `#[cxx::bridge] pub mod <name> { ... }`
// block alongside the existing ones and put the Rust-side impls in a sibling
// file under `bindings/`.

#[cxx::bridge(namespace = "torch::rust")]
pub mod ffi {
    unsafe extern "C++" {
        include!("torch/_rust/csrc/tensor_ffi.h");

        type Tensor;

        unsafe fn tensor_from_pyobject(py_obj: usize) -> *const Tensor;

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
    }
}

pub use self::ffi::*;
