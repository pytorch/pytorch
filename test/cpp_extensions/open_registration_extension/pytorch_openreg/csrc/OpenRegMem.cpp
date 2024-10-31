#include <OpenReg.h>

#include <torch/library.h>
#include <c10/core/Allocator.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <ATen/EmptyTensor.h>
#include <c10/util/ArrayRef.h>
#include <c10/core/TensorOptions.h>
#include <ATen/ops/as_strided_cpu_dispatch.h>
#include <ATen/ops/set_cpu_dispatch.h>

namespace openreg {

namespace {

using openreg_ptr_t = uint64_t;

// A dummy allocator for our custom device, that secretly uses the CPU
struct OpenRegAllocator final : at::Allocator {
  OpenRegAllocator() = default;

  at::DataPtr allocate(size_t nbytes) override {
    py::gil_scoped_acquire acquire;
    auto curr_device_idx = get_method("getDevice")().cast<c10::DeviceIndex>();
    auto curr_device = c10::Device(c10::DeviceType::PrivateUse1, curr_device_idx);
    void* data = nullptr;
    if (nbytes > 0) {
        data = reinterpret_cast<void*>(get_method("malloc")(nbytes).cast<openreg_ptr_t>());
        TORCH_CHECK(data, "Failed to allocator ", nbytes, " bytes on openreg device.");
    }
    return {data, data, &ReportAndDelete, curr_device};
  }

  static void ReportAndDelete(void* ptr) {
    if (!ptr) {
      return;
    }
    py::gil_scoped_acquire acquire;
    TORCH_CHECK(
        get_method("free")(reinterpret_cast<openreg_ptr_t>(ptr)).cast<bool>(),
        "Failed to free memory pointer at ", ptr
    );
  }

  at::DeleterFnPtr raw_deleter() const override {
    return &ReportAndDelete;
  }

  void copy_data(void* dest, const void* src, std::size_t count) const final {
    py::gil_scoped_acquire acquire;
    get_method("copy_data")(reinterpret_cast<openreg_ptr_t>(dest), reinterpret_cast<openreg_ptr_t>(src), count);
  }
};

// Register our dummy allocator
static OpenRegAllocator global_openreg_alloc;
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &global_openreg_alloc);


// Empty op needs C++ code and cannot be handled by python side fallback
at::Tensor empty_openreg(
    c10::IntArrayRef size,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  const auto device = c10::device_or_default(device_opt);
  const auto dtype = c10::dtype_or_default(dtype_opt);
  TORCH_CHECK(device.is_privateuseone());
  TORCH_CHECK(c10::layout_or_default(layout_opt) == c10::Layout::Strided, "Non strided layout not supported");
  TORCH_CHECK(!c10::pinned_memory_or_default(pin_memory_opt), "Pin memory can only be on CPU");
  const c10::DeviceGuard device_guard(device);
  constexpr c10::DispatchKeySet pu1_dks(c10::DispatchKey::PrivateUse1);
  return at::detail::empty_generic(
      size, &global_openreg_alloc, pu1_dks, dtype, memory_format_opt);
}

at::Tensor empty_strided_openreg(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt) {
  const auto device = c10::device_or_default(device_opt);
  const auto dtype = c10::dtype_or_default(dtype_opt);
  TORCH_CHECK(device.is_privateuseone());
  TORCH_CHECK(c10::layout_or_default(layout_opt) == c10::Layout::Strided, "Non strided layout not supported");
  TORCH_CHECK(!c10::pinned_memory_or_default(pin_memory_opt), "Pin memory can only be on CPU");
  const c10::DeviceGuard device_guard(device);
  constexpr c10::DispatchKeySet pu1_dks(c10::DispatchKey::PrivateUse1);
  return at::detail::empty_strided_generic(
      size, stride, &global_openreg_alloc, pu1_dks, dtype);
}

at::Tensor as_strided_openreg(
    const at::Tensor& self,
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    std::optional<int64_t> storage_offset_) {
    // Metadata-only change so we re-use the cpu impl
    return at::cpu::as_strided(self, size, stride, storage_offset_);
}

at::Tensor& set_openreg(
    at::Tensor& result,
    at::Storage storage,
    int64_t storage_offset,
    c10::IntArrayRef size,
    c10::IntArrayRef stride) {
    return at::cpu::set_(result, storage, storage_offset, size, stride);
}


TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("empty.memory_format", empty_openreg);
    m.impl("empty_strided", empty_strided_openreg);
    m.impl("as_strided", as_strided_openreg);
    m.impl("set_.source_Storage_storage_offset", set_openreg);
}

} // anonymous namspaces

} // openreg
