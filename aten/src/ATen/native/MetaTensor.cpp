#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Resize.h>

namespace at {
namespace native {

// The meta allocator ignores whatever allocation is requested and always
// gives you nullptr
struct MetaAllocator final : public at::Allocator {
  MetaAllocator() = default;
  ~MetaAllocator() override = default;
  static void deleter(void* const pointer) {
    TORCH_INTERNAL_ASSERT(!pointer);
  }
  DataPtr allocate(const size_t nbytes) const override {
    return {nullptr, nullptr, &deleter, at::Device(DeviceType::Meta)};
  }
  DeleterFnPtr raw_deleter() const override {
    return deleter;
  }
};

static MetaAllocator g_meta_alloc;

at::Allocator* GetMetaAllocator() {
  return &g_meta_alloc;
}

Tensor empty_meta(
  IntArrayRef size,
  c10::optional<ScalarType> dtype_opt,
  c10::optional<Layout> layout_opt,
  c10::optional<Device> device_opt,
  c10::optional<bool> pin_memory_opt,
  c10::optional<c10::MemoryFormat> memory_format_opt
) {

  auto device = device_or_default(device_opt);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(device.type() == DeviceType::Meta);
  // NB: because there is no SparseMeta (yet), non-strided layout is
  // exerciseable
  TORCH_CHECK_NOT_IMPLEMENTED(
    layout_or_default(layout_opt) == Layout::Strided,
    "strided meta tensors not supported yet"
  );

  auto* allocator = GetMetaAllocator();
  auto dtype = dtype_or_default(dtype_opt);
  constexpr c10::DispatchKeySet meta_ks(c10::DispatchKey::Meta);
  return at::detail::empty_generic(
      size, allocator, meta_ks, dtype, memory_format_opt);
}

Tensor empty_strided_meta(
  IntArrayRef size,
  IntArrayRef stride,
  c10::optional<ScalarType> dtype_opt,
  c10::optional<Layout> layout_opt,
  c10::optional<Device> device_opt,
  c10::optional<bool> pin_memory_opt
) {
  auto device = device_or_default(device_opt);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(device.type() == DeviceType::Meta);
  // NB: because there is no SparseMeta (yet), non-strided layout is
  // exerciseable
  TORCH_CHECK_NOT_IMPLEMENTED(
      layout_or_default(layout_opt) == Layout::Strided,
      "strided meta tensors not supported yet"
    );

  auto* allocator = GetMetaAllocator();
  auto dtype = dtype_or_default(dtype_opt);
  constexpr c10::DispatchKeySet meta_ks(c10::DispatchKey::Meta);
  return at::detail::empty_strided_generic(
      size, stride, allocator, meta_ks, dtype);
}

} // namespace native
} // namespace at
