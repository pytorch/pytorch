#include <torch/_rust/csrc/tensor_ffi.h>

#include <torch/csrc/autograd/python_variable.h>

// Force the linker to keep `PyInit__rust` reachable when this shim is
// statically linked alongside the Rust crate into a larger Python extension
// (`_C.so`). `-u PyInit__rust` is fragile across linker configurations
// (lld vs ld.bfd, archive scan order under ovrsource); a strong reference
// from a translation unit that's already being included is more robust.
// extern "C" external linkage forces the compiler to emit the initializer,
// which creates a relocation for `PyInit__rust` — portable across clang,
// gcc, and MSVC (no compiler-specific attributes required).
extern "C" PyObject* PyInit__rust();
extern "C" PyObject* (*torch_rust_force_pyinit_rust)() = &PyInit__rust;

namespace torch::rust {

const Tensor* TORCH_RUST_NULLABLE tensor_from_pyobject(
    std::uintptr_t py_obj) noexcept {
  // cxx cannot name PyObject directly in this bridge, so Rust passes the
  // borrowed Python object pointer through uintptr_t.
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  PyObject* obj = reinterpret_cast<PyObject*>(py_obj);
  if (!obj) {
    return nullptr;
  }
  try {
    if (!THPVariable_Check(obj)) {
      return nullptr;
    }
    return &THPVariable_Unpack(obj);
  } catch (...) {
    return nullptr;
  }
}

int64_t tensor_dim(const Tensor& t) {
  return t.dim();
}

int64_t tensor_numel(const Tensor& t) {
  return t.numel();
}

int64_t tensor_size_at(const Tensor& t, int64_t dim) {
  return t.size(dim);
}

int64_t tensor_stride_at(const Tensor& t, int64_t dim) {
  return t.stride(dim);
}

::rust::Slice<const int64_t> tensor_sizes(const Tensor& t) {
  auto s = t.sizes();
  return ::rust::Slice<const int64_t>(s.data(), s.size());
}

::rust::Slice<const int64_t> tensor_strides(const Tensor& t) {
  auto s = t.strides();
  return ::rust::Slice<const int64_t>(s.data(), s.size());
}

bool tensor_is_contiguous(const Tensor& t) {
  return t.is_contiguous();
}

bool tensor_is_cpu(const Tensor& t) {
  return t.is_cpu();
}

bool tensor_is_cuda(const Tensor& t) {
  return t.is_cuda();
}

bool tensor_defined(const Tensor& t) {
  return t.defined();
}

bool tensor_requires_grad(const Tensor& t) {
  return t.requires_grad();
}

} // namespace torch::rust
