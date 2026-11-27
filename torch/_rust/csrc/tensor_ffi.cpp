#include <torch/_rust/csrc/tensor_ffi.h>

#include <ATen/PythonTorchFunctionTLS.h>
#include <ATen/Context.h>
#include <ATen/SavedTensorHooks.h>
#include <ATen/autocast_mode.h>
#include <ATen/core/grad_mode.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <c10/core/AutogradState.h>
#include <c10/core/DefaultDtype.h>
#include <c10/core/InferenceMode.h>
#include <c10/core/impl/COW.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/core/impl/TorchDispatchModeTLS.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/autograd/anomaly_mode.h>
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

namespace {

// The edge of the bridge (see src/bindings/ffi.rs): whatever `f` throws ends
// here, and the caller gets `refusal`, the answer it takes for a miss. A
// Python error that was turned into the exception is still set: dropped.
template <typename T, typename F>
T edge(T refusal, F&& f) noexcept {
  try {
    return f();
  } catch (...) {
    if (PyGILState_Check() && PyErr_Occurred()) {
      PyErr_Clear();
    }
    return refusal;
  }
}

} // namespace

const Tensor* TORCH_RUST_NULLABLE tensor_from_pyobject(
    std::uintptr_t py_obj) noexcept {
  // cxx cannot name PyObject directly in this bridge, so Rust passes the
  // borrowed Python object pointer through uintptr_t.
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  PyObject* obj = reinterpret_cast<PyObject*>(py_obj);
  if (!obj) {
    return nullptr;
  }
  // isinstance can run Python code, and raise.
  return edge(static_cast<const Tensor*>(nullptr), [&]() -> const Tensor* {
    return THPVariable_Check(obj) ? &THPVariable_Unpack(obj) : nullptr;
  });
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

// What a tensor answers by a policy of its own (a Python-dispatch tensor: in
// Python; a nested tensor, symbolic sizes, a dimension out of range: by
// raising) ends at the edge: -1, or false.
int64_t tensor_dim(const Tensor& t) noexcept {
  return edge(int64_t(-1), [&] { return t.dim(); });
}

int64_t tensor_numel(const Tensor& t) noexcept {
  return edge(int64_t(-1), [&] { return t.numel(); });
}

int64_t tensor_size_at(const Tensor& t, int64_t dim) noexcept {
  return edge(int64_t(-1), [&] { return t.size(dim); });
}

int64_t tensor_stride_at(const Tensor& t, int64_t dim) noexcept {
  return edge(int64_t(-1), [&] { return t.stride(dim); });
}

// No slice says "it has none": these two are fallible (Result in the
// bridge), and the caller's failure branch refuses.
::rust::Slice<const int64_t> tensor_sizes(const Tensor& t) {
  auto s = t.sizes();
  return ::rust::Slice<const int64_t>(s.data(), s.size());
}

::rust::Slice<const int64_t> tensor_strides(const Tensor& t) {
  auto s = t.strides();
  return ::rust::Slice<const int64_t>(s.data(), s.size());
}

bool tensor_is_contiguous(const Tensor& t) noexcept {
  return edge(false, [&] { return t.is_contiguous(); });
}

bool tensor_is_cpu(const Tensor& t) noexcept {
  return edge(false, [&] { return t.is_cpu(); });
}

bool tensor_is_cuda(const Tensor& t) noexcept {
  return edge(false, [&] { return t.is_cuda(); });
}

bool tensor_defined(const Tensor& t) noexcept {
  return t.defined();
}

bool tensor_requires_grad(const Tensor& t) noexcept {
  return t.requires_grad();
}

bool tensor_is_leaf(const Tensor& t) noexcept {
  return t.is_leaf();
}

void tensor_meta(const Tensor& t, TensorMeta& out) noexcept {
  out.defined = false;
  out.has_device = false;
  out.has_shape = false;
  out.sizes.clear();
  // Undefined is what a caller reads as "nothing known of it".
  out.defined = edge(false, [&] {
    if (!t.defined()) {
      return false;
    }
    out.dtype = static_cast<int8_t>(t.scalar_type());
    out.requires_grad = t.requires_grad();
    return true;
  });
  if (!out.defined) {
    return;
  }
  const c10::TensorImpl* impl = t.unsafeGetTensorImpl();
  if (impl->is_python_dispatch()) {
    return;
  }
  // E.g. nested tensors, whose sizes() throws: no shape.
  out.has_shape = edge(false, [&] {
    c10::Device d = t.device();
    out.device_type = static_cast<int8_t>(d.type());
    out.device_index = d.index();
    out.has_device = true;
    if (impl->has_symbolic_sizes_strides()) {
      return false;
    }
    for (int64_t s : t.sizes()) {
      out.sizes.push_back(s);
    }
    return true;
  });
  if (!out.has_shape) {
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

bool torch_function_mode_active() noexcept {
  return at::impl::torch_function_mode_enabled();
}

bool grad_mode_enabled() noexcept {
  return at::GradMode::is_enabled();
}

namespace {

// The global `name` of the Python module `module`, borrowed, or nullptr:
// a module that was never imported has none. No Python code runs.
PyObject* module_global(const char* module, const char* name) {
  PyObject* modules = PyImport_GetModuleDict();
  PyObject* m = modules ? PyDict_GetItemString(modules, module) : nullptr;
  PyObject* dict = m && PyModule_Check(m) ? PyModule_GetDict(m) : nullptr;
  return dict ? PyDict_GetItemString(dict, name) : nullptr;
}

// Packs pieces of state into a word. `fits` goes false where a piece has a
// value its bits have no room for (an enumerator added behind those the
// word was sized for): there is no word then.
struct Packer {
  std::uint64_t word = 0;
  bool fits = true;
  void put(std::uint64_t value, unsigned bits) {
    fits = fits && (value >> bits) == 0;
    word = (word << bits) | value;
  }
};

// The CUDA device new tensors of `device="cuda"` go to. A thread starts on
// device 0, and asking the runtime before torch.cuda is initialized would
// initialize it.
static_assert(sizeof(c10::DeviceIndex) == 1, "the word has 8 bits for it");
std::uint64_t current_cuda_device() {
  if (module_global("torch.cuda", "_initialized") != Py_True) {
    return 0;
  }
  return static_cast<std::uint8_t>(at::detail::getCUDAHooks().current_device());
}

// Whether torch.fx is tracing on this thread (a thread-local of its module,
// read as fx reads it: no `flag` yet is False): kFxOff, kFxOn, or kFxUnknown
// where that cannot be said without running code of somebody's (the module
// is loaded and the thread-local is not there or not a plain
// threading.local, the read fails otherwise, the flag is no bool).
constexpr std::uint8_t kFxOff = 0, kFxOn = 1, kFxUnknown = 2;
std::uint8_t fx_tracing() {
  PyObject* modules = PyImport_GetModuleDict();
  if (modules && !PyDict_GetItemString(modules, "torch.fx._symbolic_trace")) {
    // Never imported: nothing of fx can be tracing.
    return kFxOff;
  }
  PyObject* tls = module_global("torch.fx._symbolic_trace", "_is_fx_tracing_tls");
  PyObject* local = module_global("_thread", "_local");
  if (!tls || !local || reinterpret_cast<PyObject*>(Py_TYPE(tls)) != local) {
    return kFxUnknown;
  }
  PyObject* flag = PyObject_GetAttrString(tls, "flag");
  if (!flag) {
    const bool absent = PyErr_ExceptionMatches(PyExc_AttributeError);
    PyErr_Clear();
    return absent ? kFxOff : kFxUnknown;
  }
  Py_DECREF(flag);
  return flag == Py_True ? kFxOn : flag == Py_False ? kFxOff : kFxUnknown;
}

} // namespace

bool region_word(Word& out) noexcept {
  return edge(false, [&] {
    Packer a;
    // Six bits for a dtype.
    static_assert(static_cast<int>(at::ScalarType::NumOptions) <= 64);
    for (auto d : {at::kCPU, at::kCUDA, at::kXPU, at::kMPS}) {
      a.put(at::autocast::is_autocast_enabled(d), 1);
      a.put(static_cast<std::uint64_t>(at::autocast::get_autocast_dtype(d)), 6);
    }
    a.put(static_cast<std::uint64_t>(c10::get_default_dtype_as_scalartype()), 6);
    const auto& ctx = at::globalContext();
    a.put(ctx.deterministicAlgorithms(), 1);
    a.put(ctx.deterministicAlgorithmsWarnOnly(), 1);
    a.put(ctx.deterministicFillUninitializedMemory(), 1);
    // What a program reads through torch.backends and a kernel is picked by:
    // a region is for the flags of its trace.
    a.put(ctx.userEnabledCuDNN(), 1);
    a.put(ctx.benchmarkCuDNN(), 1);
    a.put(ctx.deterministicCuDNN(), 1);
    a.put(ctx.userEnabledMkldnn(), 1);
    // What decides whether a result is recorded for autograd and which kind
    // of tensor it is, and where a tensor of `device="cuda"` is made.
    a.put(at::GradMode::is_enabled(), 1);
    a.put(c10::InferenceMode::is_enabled(), 1);
    a.put(current_cuda_device(), 8);
    // The float32 precision of every (backend, operator) pair as a kernel
    // and torch.backends read it, inherited values resolved: three bits each
    // (six precisions today). The legacy getters (the matmul precision,
    // allow_tf32 of cuBLAS and of cuDNN) are not asked: they raise once the
    // two APIs were mixed, and no kernel asks them. Their setters write
    // these pairs.
    using B = at::Float32Backend;
    using O = at::Float32Op;
    static_assert(static_cast<int>(at::Float32Precision::BF16X9) < 8);
    Packer b;
    b.put(ctx.allowTF32OneDNN(), 1);
    for (auto [backend, op] : {std::pair{B::GENERIC, O::ALL},
                               {B::MKLDNN, O::ALL},
                               {B::MKLDNN, O::CONV},
                               {B::MKLDNN, O::RNN},
                               {B::MKLDNN, O::MATMUL},
                               {B::CUDA, O::ALL},
                               {B::CUDA, O::CONV},
                               {B::CUDA, O::RNN},
                               {B::CUDA, O::MATMUL}}) {
      b.put(static_cast<std::uint64_t>(ctx.float32Precision(backend, op)), 3);
    }
    out.a = a.word;
    out.b = b.word;
    return a.fits && b.fits;
  });
}

void region_modes(AmbientModes& out) noexcept {
  out.dispatch_modes = c10::impl::TorchDispatchModeTLS::stack_len();
  out.function_modes = at::impl::PythonTorchFunctionTLS::stack_len();
  out.fx_tracing = fx_tracing();
}

std::uint8_t region_not_neutral(const AmbientModes& own) noexcept {
  // What cannot be asked is not neutral.
  return edge(std::uint8_t(255), [&]() -> std::uint8_t {
    AmbientModes now;
    region_modes(now);
    if (now.dispatch_modes != own.dispatch_modes) {
      return 1;
    }
    if (now.function_modes != own.function_modes) {
      return 2;
    }
    // A functorch transform keeps these two keys on while one of its levels
    // is open (its own header needs more than this file can include).
    if (c10::impl::tls_is_dispatch_key_included(
            c10::DispatchKey::FuncTorchDynamicLayerFrontMode) ||
        c10::impl::tls_is_dispatch_key_included(
            c10::DispatchKey::FuncTorchDynamicLayerBackMode)) {
      return 3;
    }
    if (torch::autograd::ForwardADLevel::has_any_level()) {
      return 4;
    }
    if (saved_tensor_hooks_active()) {
      return 5;
    }
    if (torch::jit::tracer::isTracing()) {
      return 6;
    }
    if (now.fx_tracing == kFxUnknown || now.fx_tracing != own.fx_tracing) {
      return 7;
    }
    if (torch::autograd::AnomalyMode::is_enabled()) {
      return 8;
    }
    // Autocast for a device type the word does not name.
    for (auto k :
         {c10::DispatchKey::AutocastIPU, c10::DispatchKey::AutocastHPU,
          c10::DispatchKey::AutocastXLA, c10::DispatchKey::AutocastPrivateUse1,
          c10::DispatchKey::AutocastMTIA, c10::DispatchKey::AutocastMAIA}) {
      if (!c10::impl::tls_is_dispatch_key_excluded(k)) {
        return 9;
      }
    }
    return 0;
  });
}

bool set_grad_mode(bool on) noexcept {
  bool prev = at::GradMode::is_enabled();
  at::GradMode::set_enabled(on);
  return prev;
}

int64_t tensor_version(const Tensor& t) noexcept {
  // An inference tensor has no counter, and asking raises.
  return edge(int64_t(-1), [&]() -> int64_t {
    return t.defined() ? static_cast<int64_t>(t._version()) : -1;
  });
}

namespace {

bool span_of(const Tensor& t, TensorSpan& out) {
  if (!t.defined()) {
    return false;
  }
  const c10::TensorImpl* impl = t.unsafeGetTensorImpl();
  if (impl->is_python_dispatch() || !impl->has_storage() ||
      impl->has_symbolic_sizes_strides() || t.layout() != c10::kStrided) {
    return false;
  }
  // The device types whose storages hold addresses of one address space.
  const auto type = impl->device_type();
  if (type != c10::DeviceType::CPU && type != c10::DeviceType::CUDA) {
    return false;
  }
  const c10::StorageImpl* storage =
      impl->unsafe_storage().unsafeGetStorageImpl();
  if (storage->device_type() != type) {
    return false;
  }
  const size_t item = impl->itemsize();
  // Where the last element starts, in elements from the first.
  size_t last = 0;
  const auto sizes = impl->sizes();
  const auto strides = impl->strides();
  for (size_t d = 0; d < sizes.size(); d++) {
    if (strides[d] < 0) {
      return false;
    }
    if (sizes[d] > 0) {
      last += static_cast<size_t>(sizes[d] - 1) *
          static_cast<size_t>(strides[d]);
    }
  }
  const size_t bytes = impl->numel() == 0 ? 0 : (last + 1) * item;
  const auto data = reinterpret_cast<std::uintptr_t>(storage->data());
  if (bytes != 0 && data == 0) {
    return false;
  }
  out.storage = reinterpret_cast<std::uintptr_t>(storage);
  out.at = static_cast<size_t>(impl->storage_offset()) * item;
  out.space = static_cast<uint8_t>(type);
  out.low = data + out.at;
  out.high = out.low + bytes;
  return true;
}

} // namespace

bool tensor_span(const Tensor& t, TensorSpan& out) noexcept {
  return edge(false, [&] { return span_of(t, out); });
}

std::uintptr_t tensor_home(
    const Tensor& t,
    std::uintptr_t& storage) noexcept {
  std::uintptr_t home = 0;
  const bool ok = edge(false, [&] {
    const c10::TensorImpl* impl = t.unsafeGetTensorImpl();
    if (!t.defined() || impl->is_python_dispatch() || !impl->has_storage() ||
        t.layout() != c10::kStrided) {
      return false;
    }
    const c10::StorageImpl* s = impl->unsafe_storage().unsafeGetStorageImpl();
    storage = reinterpret_cast<std::uintptr_t>(s);
    home = reinterpret_cast<std::uintptr_t>(s->data()) +
        static_cast<size_t>(impl->storage_offset()) * impl->itemsize();
    return true;
  });
  if (!ok) {
    storage = 0;
    home = 0;
  }
  return home;
}

bool tensor_is_cow(const Tensor& t) noexcept {
  const c10::TensorImpl* impl = t.unsafeGetTensorImpl();
  if (!t.defined() || impl->is_python_dispatch() || !impl->has_storage()) {
    return false;
  }
  return c10::impl::cow::is_cow_data_ptr(
      impl->unsafe_storage().unsafeGetStorageImpl()->_mutable_data_ptr_no_checks());
}

bool forward_ad_active() noexcept {
  return torch::autograd::ForwardADLevel::has_any_level() &&
      c10::AutogradState::get_tls_state().get_fw_grad_mode();
}

bool saved_tensor_hooks_active() noexcept {
  return at::SavedTensorDefaultHooks::get_hooks().has_value();
}

namespace {

bool key_of(const Tensor& t, TensorFacts& out) {
  if (!t.defined()) {
    return false;
  }
  const c10::TensorImpl* impl = t.unsafeGetTensorImpl();
  // Everything a plain dense tensor carries in its key set next to its
  // backend. A key that is none of these makes a tensor of another kind
  // (sparse, quantized, nested, a zero tensor, a wrapper of functorch or of
  // functionalization, a Python-dispatch tensor): it has no key.
  static const c10::DispatchKeySet plain =
      c10::DispatchKeySet({c10::DispatchKey::Dense,
                           c10::DispatchKey::ADInplaceOrView,
                           c10::DispatchKey::AutogradFunctionality,
                           c10::DispatchKey::Conjugate,
                           c10::DispatchKey::Negative}) |
      c10::autocast_dispatch_keyset;
  const auto other = (impl->key_set() - plain).raw_repr() & ~c10::full_backend_mask;
  if (other != 0 || impl->has_symbolic_sizes_strides() ||
      t.layout() != c10::kStrided) {
    return false;
  }
  c10::Device d = t.device();
  out.dtype = static_cast<int8_t>(t.scalar_type());
  out.device_type = static_cast<int8_t>(d.type());
  out.device_index = d.index();
  out.bits = (t.requires_grad() ? 1 : 0) | (t.is_conj() ? 2 : 0) |
      (t.is_neg() ? 4 : 0) | (t.is_inference() ? 8 : 0);
  return true;
}

} // namespace

bool tensor_key(const Tensor& t, TensorFacts& out) noexcept {
  return edge(false, [&] { return key_of(t, out); });
}

bool tensor_key_is(
    const Tensor& t,
    const TensorFacts& want,
    ::rust::Slice<const int64_t> sizes,
    ::rust::Slice<const int64_t> strides) noexcept {
  return edge(false, [&] {
    TensorFacts got;
    if (!key_of(t, got) || got.dtype != want.dtype ||
        got.device_type != want.device_type ||
        got.device_index != want.device_index || got.bits != want.bits) {
      return false;
    }
    auto sz = t.sizes();
    auto st = t.strides();
    return std::equal(sz.begin(), sz.end(), sizes.begin(), sizes.end()) &&
        std::equal(st.begin(), st.end(), strides.begin(), strides.end());
  });
}

// For printing: no name is the empty string.
::rust::String dtype_name(int8_t dtype) noexcept {
  return edge(::rust::String(), [&]() -> ::rust::String {
    return dtype_short_name(static_cast<at::ScalarType>(dtype));
  });
}

::rust::String device_name(int8_t type, int8_t index) noexcept {
  return edge(::rust::String(), [&]() -> ::rust::String {
    return c10::Device(static_cast<c10::DeviceType>(type), index).str();
  });
}

} // namespace torch::rust
