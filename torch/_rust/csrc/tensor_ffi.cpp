#include <torch/_rust/csrc/tensor_ffi.h>

#include <ATen/PythonTorchFunctionTLS.h>
#include <ATen/Context.h>
#include <ATen/SavedTensorHooks.h>
#include <ATen/autocast_mode.h>
#include <ATen/core/grad_mode.h>
#include <c10/core/AutogradState.h>
#include <c10/core/DefaultDtype.h>
#include <torch/csrc/autograd/forward_grad.h>
#include <torch/csrc/autograd/python_variable.h>
#include "torch_rust/src/bindings/ffi.rs.h"

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

const Tensor* TORCH_RUST_NULLABLE tensor_by_type(std::uintptr_t py_obj) noexcept {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  PyObject* obj = reinterpret_cast<PyObject*>(py_obj);
  if (!obj || !THPVariableClass ||
      !PyType_IsSubtype(Py_TYPE(obj), (PyTypeObject*)THPVariableClass)) {
    return nullptr;
  }
  return &THPVariable_Unpack(obj);
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

void tensor_meta(const Tensor& t, TensorMeta& out) noexcept {
  out.defined = t.defined();
  out.has_device = false;
  out.has_shape = false;
  out.sizes.clear();
  if (!out.defined) {
    return;
  }
  out.dtype = static_cast<int8_t>(t.scalar_type());
  out.requires_grad = t.requires_grad();
  const c10::TensorImpl* impl = t.unsafeGetTensorImpl();
  if (impl->is_python_dispatch()) {
    return;
  }
  try {
    c10::Device d = t.device();
    out.device_type = static_cast<int8_t>(d.type());
    out.device_index = d.index();
    out.has_device = true;
    if (impl->has_symbolic_sizes_strides()) {
      return;
    }
    for (int64_t s : t.sizes()) {
      out.sizes.push_back(s);
    }
    out.has_shape = true;
  } catch (...) {
    // E.g. nested tensors, whose sizes() throws.
    out.sizes.clear();
  }
}

const char* dtype_short_name(at::ScalarType t) {
  switch (t) {
    case at::kFloat:
      return "f32";
    case at::kDouble:
      return "f64";
    case at::kHalf:
      return "f16";
    case at::kBFloat16:
      return "bf16";
    case at::kLong:
      return "i64";
    case at::kInt:
      return "i32";
    case at::kShort:
      return "i16";
    case at::kChar:
      return "i8";
    case at::kByte:
      return "u8";
    case at::kBool:
      return "b8";
    default:
      return c10::toString(t);
  }
}

bool torch_function_mode_active() {
  return at::impl::torch_function_mode_enabled();
}

bool grad_mode_enabled() {
  return at::GradMode::is_enabled();
}

std::uint64_t region_thread_state() {
  std::uint64_t s = 0;
  for (auto d : {at::kCPU, at::kCUDA, at::kXPU, at::kMPS}) {
    s = (s << 7) | (at::autocast::is_autocast_enabled(d) ? 0x40u : 0u) |
        (static_cast<std::uint64_t>(at::autocast::get_autocast_dtype(d)) & 0x3fu);
  }
  s = (s << 6) |
      (static_cast<std::uint64_t>(c10::get_default_dtype_as_scalartype()) & 0x3fu);
  const auto& ctx = at::globalContext();
  s = (s << 3) | (ctx.deterministicAlgorithms() ? 4u : 0u) |
      (ctx.deterministicAlgorithmsWarnOnly() ? 2u : 0u) |
      (ctx.deterministicFillUninitializedMemory() ? 1u : 0u);
  return s;
}

bool set_grad_mode(bool on) {
  bool prev = at::GradMode::is_enabled();
  at::GradMode::set_enabled(on);
  return prev;
}

int64_t tensor_version(const Tensor& t) {
  return t._version();
}

void tensor_span(const Tensor& t, TensorSpan& out) noexcept {
  out.storage = 0;
  out.low = 0;
  out.high = 0;
  const c10::TensorImpl* impl = t.unsafeGetTensorImpl();
  if (!t.defined() || impl->is_python_dispatch() || !impl->has_storage() ||
      t.layout() != c10::kStrided) {
    return;
  }
  try {
    const size_t item = impl->itemsize();
    const size_t low = static_cast<size_t>(impl->storage_offset()) * item;
    // Where the last element starts, in elements from the first.
    size_t last = 0;
    const auto sizes = impl->sizes();
    const auto strides = impl->strides();
    for (size_t d = 0; d < sizes.size(); d++) {
      if (sizes[d] > 0 && strides[d] > 0) {
        last += static_cast<size_t>(sizes[d] - 1) * static_cast<size_t>(strides[d]);
      }
    }
    out.low = low;
    out.high = impl->numel() == 0 ? low : low + (last + 1) * item;
    out.storage = reinterpret_cast<std::uintptr_t>(
        impl->unsafe_storage().unsafeGetStorageImpl());
  } catch (...) {
    out.storage = 0;
  }
}

std::uintptr_t tensor_data_address(const Tensor& t) noexcept {
  const c10::TensorImpl* impl = t.unsafeGetTensorImpl();
  if (!t.defined() || impl->is_python_dispatch() || !impl->has_storage() ||
      t.layout() != c10::kStrided) {
    return 0;
  }
  try {
    return reinterpret_cast<std::uintptr_t>(impl->data());
  } catch (...) {
    return 0;
  }
}

bool forward_ad_active() {
  return torch::autograd::ForwardADLevel::has_any_level() &&
      c10::AutogradState::get_tls_state().get_fw_grad_mode();
}

bool saved_tensor_hooks_active() {
  return at::SavedTensorDefaultHooks::get_hooks().has_value();
}

bool tensor_matches(
    const Tensor& t,
    int8_t dtype,
    int8_t device_type,
    int8_t device_index,
    ::rust::Slice<const int64_t> sizes,
    ::rust::Slice<const int64_t> strides) noexcept {
  const c10::TensorImpl* impl = t.unsafeGetTensorImpl();
  if (!t.defined() || impl->is_python_dispatch() ||
      impl->has_symbolic_sizes_strides() || t.is_nested() ||
      t.layout() != c10::kStrided ||
      static_cast<int8_t>(t.scalar_type()) != dtype) {
    return false;
  }
  c10::Device d = t.device();
  if (static_cast<int8_t>(d.type()) != device_type ||
      d.index() != device_index) {
    return false;
  }
  auto sz = t.sizes();
  auto st = t.strides();
  return std::equal(sz.begin(), sz.end(), sizes.begin(), sizes.end()) &&
      std::equal(st.begin(), st.end(), strides.begin(), strides.end());
}

::rust::String dtype_name(int8_t dtype) {
  return dtype_short_name(static_cast<at::ScalarType>(dtype));
}

::rust::String device_name(int8_t type, int8_t index) {
  return c10::Device(static_cast<c10::DeviceType>(type), index).str();
}

} // namespace torch::rust
