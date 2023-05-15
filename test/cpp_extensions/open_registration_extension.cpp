#include <unordered_map>
#include <c10/core/impl/alloc_cpu.h>
#include <c10/core/Allocator.h>
#include <c10/core/ScalarType.h>
#include <c10/util/ArrayRef.h>

#include <torch/csrc/Device.h>
#include <torch/csrc/jit/serialization/pickler.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>
#include <torch/extension.h>

#include <ATen/native/cpu/Loops.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Resize.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/ops/abs_native.h>
#include <ATen/EmptyTensor.h>
#include <ATen/core/GeneratorForPrivateuseone.h>

static uint64_t add_counter = 0;
static uint64_t last_saved_value = 0;
static c10::DeviceIndex custom_device_index = 0;

static uint64_t abs_counter = 0;
static uint64_t last_abs_saved_value = 0;
// register guard
namespace at {
namespace detail {

C10_REGISTER_GUARD_IMPL(PrivateUse1, c10::impl::NoOpDeviceGuardImpl<DeviceType::PrivateUse1>);

}} // namespace at::detail

namespace {

void abs_kernel(::at::TensorIteratorBase& iter) {
  // Since this custom device is just for testing, not bothering to implement kernels.
  abs_counter += 1;
}

} // namespace

namespace at::native {

REGISTER_PRIVATEUSE1_DISPATCH(abs_stub, &abs_kernel);

} // namespace at::native
struct CustomBackendMetadata : public c10::BackendMeta {
  // for testing this field will mutate when clone() is called by shallow_copy_from.
  int backend_version_format_{-1};
  int format_number_{-1};
  mutable bool cloned_{false};
  // define the constructor
  CustomBackendMetadata(int backend_version_format, int format_number): backend_version_format_(backend_version_format), format_number_(format_number) {}
  c10::intrusive_ptr<c10::BackendMeta> clone(const c10::intrusive_ptr<c10::BackendMeta>& ptr) const override {
    cloned_ = true;
    return c10::BackendMeta::clone(ptr);
  }
};

// we need to register two functions for serialization
void for_serialization(const at::Tensor& t, std::unordered_map<std::string, bool>& m) {
  if (t.unsafeGetTensorImpl()->get_backend_meta_intrusive_ptr() == nullptr) {
    return;
  }
  CustomBackendMetadata* tmeta = dynamic_cast<CustomBackendMetadata*>(t.unsafeGetTensorImpl()->get_backend_meta());
  if (tmeta->backend_version_format_ == 1) {
    m["backend_version_format"] = true;
  }
  if (tmeta->format_number_ == 29) {
    m["format_number"] = true;
  }
}

void for_deserialization(const at::Tensor& t, std::unordered_map<std::string, bool>& m) {
  int backend_version_format{-1};
  int format_number{-1};
  if (m.find("backend_version_format") != m.end()) {
    backend_version_format = 1;
  }
  if (m.find("format_number") != m.end()) {
    format_number = 29;
  }
  c10::intrusive_ptr<c10::BackendMeta> new_tmeta{std::unique_ptr<c10::BackendMeta>(new CustomBackendMetadata(backend_version_format, format_number))};
  t.unsafeGetTensorImpl()->set_backend_meta(new_tmeta);
}

void custom_serialization_registry(){
torch::jit::TensorBackendMetaRegistry(c10::DeviceType::PrivateUse1, &for_serialization, &for_deserialization);
}

//check if BackendMeta serialization correctly
bool check_backend_meta(const at::Tensor& t) {
  if (t.unsafeGetTensorImpl()->get_backend_meta_intrusive_ptr()) {
    CustomBackendMetadata* tmeta = dynamic_cast<CustomBackendMetadata*>(t.unsafeGetTensorImpl()->get_backend_meta());
    if (tmeta->backend_version_format_==1 && tmeta->format_number_==29) {
      return true;
    }
  }
  return false;
}

// a fake set function is exposed to the Python side
void custom_set_backend_meta(const at::Tensor& t) {
  int backend_version_format{1};
  int format_number{29};
  c10::intrusive_ptr<c10::BackendMeta> new_tmeta{std::unique_ptr<c10::BackendMeta>(new CustomBackendMetadata(backend_version_format, format_number))};
  t.unsafeGetTensorImpl()->set_backend_meta(new_tmeta);
}

// basic dummy add function
at::Tensor custom_add_Tensor(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  add_counter += 1;
  // Since this custom device is just for testing, not bothering to implement kernels.
  return at::empty(self.sizes(), self.options());
}

// basic abs function
at::Tensor& custom_abs_out(const at::Tensor& self, at::Tensor& out) {
  return at::native::abs_out(self, out);
}

// A dummy allocator for our custom device, that secretly uses the CPU
struct DummyCustomAllocator final : at::Allocator {
  DummyCustomAllocator() = default;
  at::DataPtr allocate(size_t nbytes) const override {
    void* data = c10::alloc_cpu(nbytes);
    return {data, data, &ReportAndDelete, at::Device(at::DeviceType::PrivateUse1, custom_device_index)};
  }

  static void ReportAndDelete(void* ptr) {
    if (!ptr) {
      return;
    }
    c10::free_cpu(ptr);
  }

  at::DeleterFnPtr raw_deleter() const override {
    return &ReportAndDelete;
  }
};

// Register our dummy allocator
static DummyCustomAllocator global_custom_alloc;
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &global_custom_alloc);

// basic dummy empty function, so we can directly construct tensors on the custom device
// This dummy test device will just use the CPU allocator, and ignores pinned memory.
at::Tensor custom_empty_memory_format(at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
  constexpr c10::DispatchKeySet private_use_ks(c10::DispatchKey::PrivateUse1);
  return at::detail::empty_generic(size, &global_custom_alloc, private_use_ks, c10::dtype_or_default(dtype), memory_format);
}
at::Tensor custom_empty_symint(c10::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
  constexpr c10::DispatchKeySet private_use_ks(c10::DispatchKey::PrivateUse1);
  return at::detail::empty_generic(size, &global_custom_alloc, private_use_ks, c10::dtype_or_default(dtype), memory_format);
}

at::Tensor & custom_fill__scalar(at::Tensor & self, const at::Scalar & value) {
  // Not bothering to implement.
  return self;
}

// basic dummy copy_() function, so we can copy from the custom device to/from CPU
at::Tensor custom__copy_from(const at::Tensor& self, const at::Tensor& dst, bool non_blocking) {
  TORCH_CHECK(self.is_cpu() || self.device().type() == c10::DeviceType::PrivateUse1, "Dummy test only allows copy from cpu -> dummy device.");
  TORCH_CHECK(dst.is_cpu() || dst.device().type() == c10::DeviceType::PrivateUse1, "Dummy test only allows copy from cpu -> dummy device.");

  // Some dummy asserts for the basic use case: inputs are the same size / dtype, all contiguous.
  TORCH_CHECK(self.sizes() == dst.sizes());
  TORCH_CHECK(self.scalar_type() == dst.scalar_type());
  TORCH_CHECK(self.is_contiguous() && dst.is_contiguous());

  std::memcpy(dst.storage().data_ptr().get(), self.storage().data_ptr().get(), self.storage().nbytes());
  return dst;
}

at::Tensor custom_empty_strided(c10::IntArrayRef size, c10::IntArrayRef stride, c10::optional<at::ScalarType> dtype_opt, c10::optional<at::Layout> layout_opt, c10::optional<at::Device> device_opt, c10::optional<bool> pin_memory_opt) {
  constexpr c10::DispatchKeySet private_use_ks(c10::DispatchKey::PrivateUse1);
  auto dtype = c10::dtype_or_default(dtype_opt);
  return  at::detail::empty_strided_generic(size, stride, &global_custom_alloc, private_use_ks, dtype);
}

// Some set operations for the basic use case
at::Tensor& custom_set_source_Storage(at::Tensor& result, c10::Storage src) {
  int64_t new_size = static_cast<int64_t>(src.nbytes() / result.dtype().itemsize());
  c10::IntArrayRef stride = {};
  result.unsafeGetTensorImpl()->set_storage_offset(0);
  at::OptionalIntArrayRef stride_opt = stride.data() != nullptr ? at::OptionalIntArrayRef(stride) : c10::nullopt;
  at::native::resize_impl_cpu_(result.unsafeGetTensorImpl(), new_size, stride_opt, /*resize_storage=*/!result.is_meta());
  return result;
}

// Some set operations for the basic use case
at::Tensor& custom_set_source_Storage_storage_offset(at::Tensor& result, c10::Storage storage, int64_t storage_offset, c10::IntArrayRef size, c10::IntArrayRef stride) {
  result.unsafeGetTensorImpl()->set_storage_offset(storage_offset);
  at::OptionalIntArrayRef stride_opt = stride.data() != nullptr ? at::OptionalIntArrayRef(stride) : c10::nullopt;
  at::native::resize_impl_cpu_(result.unsafeGetTensorImpl(), size, stride_opt, /*resize_storage=*/!result.is_meta());
  return result;
}

// basic dummy functions related to pin_memory.
std::vector<void*> custom_pinned_data_ptr;

at::Tensor custom__pin_memory(const at::Tensor& self, c10::optional<at::Device> device) {
  TORCH_CHECK(self.device().is_cpu(), "cannot pin '", self.toString(), "' only dense CPU tensors can be pinned");

  // record pinned data ptr
  at::Tensor dump_pinned_tensor = self * 1.0;
  custom_pinned_data_ptr.push_back(dump_pinned_tensor.storage().data_ptr().get());

  return dump_pinned_tensor;
}

bool custom_is_pinned(const at::Tensor& self, c10::optional<at::Device> device) {
  // Only CPU tensors can be pinned
  if (!self.is_cpu()) {
    return false;
  }

  void* query_pinned_ptr = self.storage().data_ptr().get();
  for (const auto& iter_ptr : custom_pinned_data_ptr) {
    if (iter_ptr == query_pinned_ptr) {
      return true;
    }
  }
  return false;
}

const at::Tensor& custom_resize_(const at::Tensor& self, at::IntArrayRef size,
                          c10::optional<at::MemoryFormat> optional_memory_format) {
  self.unsafeGetTensorImpl()->set_sizes_contiguous(size);
  const auto itemsize = self.unsafeGetTensorImpl()->dtype().itemsize();
  const auto offset = self.unsafeGetTensorImpl()->storage_offset();
  const auto storage_size = at::detail::computeStorageNbytesContiguous(size, itemsize, offset);
  const auto &storage = self.unsafeGetTensorImpl()->unsafe_storage();
  if (storage_size > storage.nbytes()) {
    storage.unsafeGetStorageImpl()->set_nbytes(storage_size);
  }

  return self;
}

// This macro does the heavy lifting.
// With TORCH_LIBRARY_IMPL, you can register custom kernels for your backend.
// For open registration, we're registering all of our kernels to the PrivateUse1 dispatch key.
// Later in this file, we map a custom device to the PrivateUse1 device type,
// which allows user code that puts a tensor on your custom_device to eventually get plumbed
// into the kernels registered here.
//
// This macro registers your kernels to the PyTorch Dispatcher.
// More details on the dispatcher can be found at http://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/.
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("abs.out", &custom_abs_out);
  m.impl("add.Tensor", &custom_add_Tensor);
  m.impl("empty.memory_format", &custom_empty_symint);
  m.impl("fill_.Scalar", &custom_fill__scalar);
  m.impl("_copy_from", &custom__copy_from);
  m.impl("empty_strided", &custom_empty_strided);
  m.impl("set_.source_Storage", &custom_set_source_Storage);
  m.impl("set_.source_Storage_storage_offset",&custom_set_source_Storage_storage_offset);
  m.impl("_pin_memory", &custom__pin_memory);
  m.impl("is_pinned", &custom_is_pinned);
  m.impl("resize_", &custom_resize_);
}

// This basic implementation doesn't bother dealing with different device indices
// (e.g. custom_device:0 vs. custom_device:1).
// We could do that by letting the user pass in a device index in our exposed device function.
// Note that if you do that, you'll also need to register a device guard to core.
// See `c10/core/impl/DeviceGuardImplInterface.h:C10_REGISTER_GUARD_IMPL`.
c10::Device get_custom_device() {
  return c10::Device(c10::DeviceType::PrivateUse1, 0);
}

bool custom_add_called() {
  bool called = false;
  if (add_counter > last_saved_value) {
    called = true;
    last_saved_value = add_counter;
  }
  return called;
}

bool custom_abs_called() {
  bool called = false;
  if (abs_counter > last_abs_saved_value) {
    called = true;
    last_abs_saved_value = abs_counter;
  }
  return called;
}

class PrivateGeneratorImpl : public at::CPUGeneratorImpl {
public:
  // Constructors
  PrivateGeneratorImpl(c10::DeviceIndex device_index) {
    device_ = c10::Device(c10::DeviceType::PrivateUse1, device_index);
    key_set_ = c10::DispatchKeySet(c10::DispatchKey::PrivateUse1);
  }
  ~PrivateGeneratorImpl() override = default;
};

// this is used to register generator
at::Generator make_generator_privateuse1(c10::DeviceIndex device_index) {
  return at::make_generator<PrivateGeneratorImpl>(device_index);
}

void register_generator() {
  REGISTER_GENERATOR_PRIVATEUSE1(make_generator_privateuse1)
}

void set_custom_device_index(c10::DeviceIndex device_index) {
  custom_device_index = device_index;
}

// Here, we're exposing a custom device object that corresponds to our custom backend.
// We do this using pybind: exposing an "extension_name.custom_device()" function in python,
// that's implemented in C++.
// The implementation in this file maps directly to the `PrivateUse1` device type.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_device", &get_custom_device, "get custom device object");
    m.def("custom_add_called", &custom_add_called, "check if our custom add function was called");
    m.def("custom_abs_called", &custom_abs_called, "check if our custom abs function was called");
    m.def("register_generator", &register_generator, "register generator for custom device");
    m.def("set_custom_device_index", &set_custom_device_index, "set custom device index");
    m.def("custom_set_backend_meta", &custom_set_backend_meta, "a fake set tensor BackendMeta function");
    m.def("check_backend_meta", &check_backend_meta, "check if BackendMeta serialization correctly");
    m.def("custom_serialization_registry", &custom_serialization_registry, "register custom serialization function");
}
