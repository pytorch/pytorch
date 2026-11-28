#include <c10/core/Allocator.h>
#include <c10/core/impl/alloc_cpu.h>

#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/macros/Macros.h>
#include <torch/csrc/Device.h>
#include <torch/extension.h>

#include <ATen/EmptyTensor.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Resize.h>
#include <ATen/native/cpu/Loops.h>

static uint64_t op_counter = 0;
static uint64_t last_saved_value = 0;
static uint64_t int8pack_mm_calls = 0;

// register guard
namespace at {
namespace detail {

C10_REGISTER_GUARD_IMPL(PrivateUse1, c10::impl::NoOpDeviceGuardImpl<DeviceType::PrivateUse1>);

}} // namespace at::detail

// basic dummy add function
at::Tensor custom_add_Tensor(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  op_counter += 1;
  // Since this custom device is just for testing, not bothering to implement kernels.
  return at::empty(self.sizes(), self.options());
}

// basic dummy mul function
at::Tensor custom_mul_Tensor(const at::Tensor & self, const at::Tensor & other) {
  op_counter += 1;
  // Since this custom device is just for testing, not bothering to implement kernels.
  return at::empty(self.sizes(), self.options());
}

// basic dummy eq function: Only support CPU
at::Tensor custom_to_device(
    const at::Tensor & self,
    at::Device device,
    at::ScalarType dtype,
    bool non_blocking,
    bool copy,
    std::optional<at::MemoryFormat> memory_format) {
  TORCH_CHECK(self.is_cpu() || self.device().type() == c10::DeviceType::PrivateUse1, "Dummy test only allows copy from cpu -> dummy device.");
  TORCH_CHECK(device.is_cpu() || device.type() == c10::DeviceType::PrivateUse1, "Dummy test only allows copy from cpu -> dummy device.");
  // Some dummy asserts for the basic use case: inputs are the same size / dtype, all contiguous.
  TORCH_CHECK(self.scalar_type() == dtype);
  TORCH_CHECK(self.is_contiguous());

  op_counter += 1;
  if (device != at::DeviceType::CPU) {
    return at::empty(self.sizes(), self.options());
  }

  auto out = at::empty(self.sizes(), dtype, self.options().layout(), device, false, memory_format);
  memcpy(out.mutable_data_ptr(), self.mutable_data_ptr(), self.nbytes());
  // Since this custom device is just for testing, not bothering to implement kernels.
  return out;
}


// A dummy allocator for our custom device, that secretly uses the CPU
struct DummyCustomAllocator final : at::Allocator {
  DummyCustomAllocator() = default;
  at::DataPtr allocate(size_t nbytes) override {
    void* data = c10::alloc_cpu(nbytes);
    return {data, data, &ReportAndDelete, at::Device(at::DeviceType::PrivateUse1, 0)};
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

  void copy_data(void* dest, const void* src, std::size_t count) const final {
    default_copy_data(dest, src, count);
  }
};

// Register our dummy allocator
static DummyCustomAllocator global_custom_alloc;
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &global_custom_alloc);

at::Tensor & custom_fill__scalar(at::Tensor & self, const at::Scalar & value) {
  TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1, "Dummy test only allows dummy device.");
  TORCH_CHECK(self.is_contiguous());
  TORCH_CHECK(self.scalar_type() == c10::ScalarType::Float);

  op_counter += 1;
  auto _data = static_cast<float*>(self.mutable_data_ptr());
  for (size_t idx = 0; idx < self.numel(); idx++) {
    _data[idx] = value.toFloat();
  }

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

  op_counter += 1;
  std::memcpy(dst.storage().data_ptr().get(), self.storage().data_ptr().get(), self.storage().nbytes());
  return dst;
}

at::Tensor custom_empty_memory_format(at::IntArrayRef size,
                                      std::optional<at::ScalarType> dtype,
                                      std::optional<at::Layout> layout,
                                      std::optional<at::Device> device,
                                      std::optional<bool> pin_memory,
                                      std::optional<at::MemoryFormat> memory_format) {
  constexpr c10::DispatchKeySet private_use_ks(c10::DispatchKey::PrivateUse1);
  return at::detail::empty_generic(size,
                                   &global_custom_alloc,
                                   private_use_ks,
                                   c10::dtype_or_default(dtype),
                                   memory_format);
}

at::Tensor custom_empty_strided(c10::IntArrayRef size, c10::IntArrayRef stride, std::optional<at::ScalarType> dtype_opt, std::optional<at::Layout> layout_opt, std::optional<at::Device> device_opt, std::optional<bool> pin_memory_opt) {
  op_counter += 1;

  constexpr c10::DispatchKeySet private_use_ks(c10::DispatchKey::PrivateUse1);
  auto dtype = c10::dtype_or_default(dtype_opt);
  return  at::detail::empty_strided_generic(size, stride, &global_custom_alloc, private_use_ks, dtype);
}

// The dummy device's memory is CPU memory in disguise (see
// DummyCustomAllocator above), so its kernels can run the corresponding CPU
// ATen op on an alias of the tensor storage. The explicit target_device
// argument skips from_blob's pointer-to-device sniffing, which needs the
// PrivateUse1HooksInterface that this dummy backend does not register.
static at::Tensor alias_to_cpu(at::Tensor t) {
  return at::from_blob(
      t.mutable_data_ptr(),
      t.sizes(),
      t.strides(),
      [t](void*) {},
      t.options().device(at::kCPU),
      at::Device(at::kCPU));
}

static at::Tensor alias_from_cpu(at::Tensor t) {
  return at::from_blob(
      t.mutable_data_ptr(),
      t.sizes(),
      t.strides(),
      [t](void*) {},
      t.options().device(at::Device(at::kPrivateUse1, 0)),
      at::Device(at::kPrivateUse1, 0));
}

// Plain mm, so models that do not get a weight-only-quant fusion can still
// run on the dummy device.
at::Tensor custom_mm(const at::Tensor& self, const at::Tensor& mat2) {
  op_counter += 1;
  return alias_from_cpu(at::mm(alias_to_cpu(self), alias_to_cpu(mat2)));
}

at::Tensor& custom_mm_out(const at::Tensor& self, const at::Tensor& mat2, at::Tensor& out) {
  op_counter += 1;
  out.copy_(at::mm(alias_to_cpu(self), alias_to_cpu(mat2)));
  return out;
}

// Weight-only-quant int8pack mm: this is the op the Inductor WOQ fusion
// lowers to, so a backend that wants that fusion has to provide a kernel
// for it.
at::Tensor custom_weight_int8pack_mm(
    const at::Tensor& self,
    const at::Tensor& mat2,
    const at::Tensor& scales) {
  op_counter += 1;
  int8pack_mm_calls += 1;
  return alias_from_cpu(at::_weight_int8pack_mm(
      alias_to_cpu(self), alias_to_cpu(mat2), alias_to_cpu(scales)));
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
  m.impl("add.Tensor", &custom_add_Tensor);
  m.impl("mul.Tensor", &custom_mul_Tensor);
  m.impl("to.Device", &custom_to_device);
  m.impl("fill_.Scalar", &custom_fill__scalar);
  m.impl("_copy_from", &custom__copy_from);
  m.impl("empty.memory_format", &custom_empty_memory_format);
  m.impl("empty_strided", &custom_empty_strided);
  m.impl("mm", &custom_mm);
  m.impl("mm.out", &custom_mm_out);
  m.impl("_weight_int8pack_mm", &custom_weight_int8pack_mm);
}

// This basic implementation doesn't bother dealing with different device indices
// (e.g. custom_device:0 vs. custom_device:1).
// We could do that by letting the user pass in a device index in our exposed device function.
// Note that if you do that, you'll also need to register a device guard to core.
// See `c10/core/impl/DeviceGuardImplInterface.h:C10_REGISTER_GUARD_IMPL`.
c10::Device get_custom_device() {
  return c10::Device(c10::DeviceType::PrivateUse1, 0);
}

bool custom_op_called() {
  bool called = false;
  if (op_counter > last_saved_value) {
    called = true;
    last_saved_value = op_counter;
  }
  return called;
}

uint64_t int8pack_mm_call_count() {
  return int8pack_mm_calls;
}

// Here, we're exposing a custom device object that corresponds to our custom backend.
// We do this using pybind: exposing an "extension_name.custom_device()" function in python,
// that's implemented in C++.
// The implementation in this file maps directly to the `PrivateUse1` device type.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_device", &get_custom_device, "get custom device object");
    m.def("custom_op_called", &custom_op_called, "check if our custom function was called");
    m.def("int8pack_mm_call_count", &int8pack_mm_call_count, "number of aten::_weight_int8pack_mm calls");
    m.def("is_available", []() { return true; });
    m.def("device_count", []() { return 1; });
    m.def("current_device", []() { return 0; });
}
