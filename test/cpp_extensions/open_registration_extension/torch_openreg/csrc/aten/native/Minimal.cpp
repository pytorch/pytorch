#include "Minimal.h"

#include <unordered_set>

namespace at::native::openreg {

// LITERALINCLUDE START: EMPTY.MEMORY_FORMAT IMPL
at::Tensor empty_memory_format(
    c10::IntArrayRef size,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  const auto device = c10::device_or_default(device_opt);
  const auto dtype = c10::dtype_or_default(dtype_opt);
  TORCH_CHECK(device.is_privateuseone());
  TORCH_CHECK(
      c10::layout_or_default(layout_opt) == c10::Layout::Strided,
      "Non strided layout not supported");
  TORCH_CHECK(
      !c10::pinned_memory_or_default(pin_memory_opt),
      "Pin memory can only be on CPU");
  const c10::DeviceGuard device_guard(device);
  constexpr c10::DispatchKeySet pu1_dks(c10::DispatchKey::PrivateUse1);
  auto allocator = at::GetAllocator(at::kPrivateUse1);
  return at::detail::empty_generic(
      size, allocator, pu1_dks, dtype, memory_format_opt);
}
// LITERALINCLUDE END: EMPTY.MEMORY_FORMAT IMPL

at::Tensor empty_strided(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt) {
  const auto device = c10::device_or_default(device_opt);
  const auto dtype = c10::dtype_or_default(dtype_opt);
  TORCH_CHECK(device.is_privateuseone());
  TORCH_CHECK(
      c10::layout_or_default(layout_opt) == c10::Layout::Strided,
      "Non strided layout not supported");
  TORCH_CHECK(
      !c10::pinned_memory_or_default(pin_memory_opt),
      "Pin memory can only be on CPU");
  const c10::DeviceGuard device_guard(device);
  constexpr c10::DispatchKeySet pu1_dks(c10::DispatchKey::PrivateUse1);
  auto allocator = at::GetAllocator(at::kPrivateUse1);
  return at::detail::empty_strided_generic(
      size, stride, allocator, pu1_dks, dtype);
}

at::Tensor as_strided(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    std::optional<c10::SymInt> storage_offset) {
  MemoryGuard guard(self);

  return at::cpu::as_strided_symint(self, size, stride, storage_offset);
}

const at::Tensor& resize_(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    ::std::optional<at::MemoryFormat> memory_format) {
  return at::native::resize_(
      self, C10_AS_INTARRAYREF_SLOW(size), memory_format);
}

at::Tensor _reshape_alias(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride) {
  return at::native::_reshape_alias(
      self, C10_AS_INTARRAYREF_SLOW(size), C10_AS_INTARRAYREF_SLOW(stride));
}

at::Tensor _copy_from(
    const at::Tensor& self,
    const at::Tensor& dst,
    bool non_blocking) {
  TORCH_CHECK(self.defined(), "Source tensor (self) is not defined.");
  TORCH_CHECK(dst.defined(), "Destination tensor (dst) is not defined.");

  MemoryGuard guard(self, dst);

  if (self.device() == dst.device()) {
    at::Tensor dst_as_cpu = at::from_blob(
        dst.data_ptr(),
        dst.sizes(),
        dst.strides(),
        dst.options().device(at::kCPU));
    const at::Tensor self_as_cpu = at::from_blob(
        self.data_ptr(),
        self.sizes(),
        self.strides(),
        self.options().device(at::kCPU));

    at::native::copy_(
        const_cast<at::Tensor&>(dst_as_cpu), self_as_cpu, non_blocking);

  } else {
    if (self.is_cpu()) {
      at::Tensor dst_as_cpu = at::from_blob(
          dst.data_ptr(),
          dst.sizes(),
          dst.strides(),
          dst.options().device(at::kCPU));

      at::native::copy_(
          const_cast<at::Tensor&>(dst_as_cpu), self, non_blocking);

    } else {
      at::Tensor self_as_cpu = at::from_blob(
          self.data_ptr(),
          self.sizes(),
          self.strides(),
          self.options().device(at::kCPU));

      at::native::copy_(
          const_cast<at::Tensor&>(dst), self_as_cpu, non_blocking);
    }
  }

  return dst;
}

at::Tensor _copy_from_and_resize(
    const at::Tensor& self,
    const at::Tensor& dst) {
  at::native::resize_(dst, self.sizes(), std::nullopt);
  return at::native::copy_(const_cast<at::Tensor&>(dst), self, false);
}

at::Scalar _local_scalar_dense(const at::Tensor& self) {
  MemoryGuard guard(self);
  return at::native::_local_scalar_dense_cpu(self);
}

at::Tensor& set_source_Tensor_(at::Tensor& self, const at::Tensor& source) {
  return at::native::set_tensor_(self, source);
}

at::Tensor& set_source_Storage_(at::Tensor& self, at::Storage source) {
  return at::native::set_(self, source);
}

at::Tensor& set_source_Storage_storage_offset_(
    at::Tensor& result,
    at::Storage storage,
    int64_t storage_offset,
    c10::IntArrayRef size,
    c10::IntArrayRef stride) {
  return at::cpu::set_(result, storage, storage_offset, size, stride);
}

at::Tensor view(const at::Tensor& self, c10::SymIntArrayRef size) {
  MemoryGuard guard(self);
  return at::native::view(self, C10_AS_INTARRAYREF_SLOW(size));
}

// LITERALINCLUDE START: FALLBACK IMPL
void cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  static const std::unordered_set<c10::OperatorName> cpu_fallback_blocklist = {
      c10::OperatorName("aten::abs", ""),
      c10::OperatorName("aten::abs", "out"),
  };

  const auto& op_name = op.schema().operator_name();
  if (cpu_fallback_blocklist.count(op_name)) {
    TORCH_CHECK(
        false,
        "Operator '",
        op_name,
        "' is not implemented for device openreg.");
  } else {
    at::native::cpu_fallback(op, stack);
  }
}
// LITERALINCLUDE END: FALLBACK IMPL

} // namespace at::native::openreg
