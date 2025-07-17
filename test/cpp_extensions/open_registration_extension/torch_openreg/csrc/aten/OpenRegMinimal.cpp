#include "native/Minimal.h"

#include <ATen/native/CPUFallback.h>
#include <ATen/native/DispatchStub.h>

#include <torch/library.h>

namespace at::openreg {

at::Tensor wrapper_empty_memory_format(
    c10::IntArrayRef size,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  return at::native::empty_memory_format_openreg(
      size,
      dtype_opt,
      layout_opt,
      device_opt,
      pin_memory_opt,
      memory_format_opt);
}

at::Tensor wrapper_empty_strided(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt) {
  return at::native::empty_strided_openreg(
      size, stride, dtype_opt, layout_opt, device_opt, pin_memory_opt);
}

at::Tensor wrapper_as_strided(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    std::optional<c10::SymInt> storage_offset) {
  return at::native::as_strided_openreg(self, size, stride, storage_offset);
}

const at::Tensor& wrapper_resize_(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    ::std::optional<at::MemoryFormat> memory_format) {
  return at::native::resize_openreg_(self, size, memory_format);
}

at::Tensor wrapper__reshape_alias(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride) {
  return at::native::_reshape_alias_openreg(self, size, stride);
}

at::Tensor wrapper__copy_from(
    const at::Tensor& self,
    const at::Tensor& dst,
    bool non_blocking) {
  return at::native::_copy_from_openreg(self, dst, non_blocking);
}

at::Tensor wrapper__copy_from_and_resize(
    const at::Tensor& self,
    const at::Tensor& dst) {
  return at::native::_copy_from_and_resize_openreg(self, dst);
}

at::Scalar wrapper__local_scalar_densor(const at::Tensor& self) {
  return at::native::_local_scalar_dense_openreg(self);
}

at::Tensor& wrapper_set_source_Tensor_(
    at::Tensor& self,
    const at::Tensor& source) {
  return at::native::set_source_Tensor_openreg_(self, source);
}

at::Tensor& wrapper_set_source_Storage_(at::Tensor& self, at::Storage source) {
  return at::native::set_source_Storage_openreg_(self, source);
}

at::Tensor& wrapper_set_source_Storage_storage_offsetset_(
    at::Tensor& result,
    at::Storage storage,
    int64_t storage_offset,
    c10::IntArrayRef size,
    c10::IntArrayRef stride) {
  return at::native::set_source_Storage_storage_offset_openreg_(
      result, storage, storage_offset, size, stride);
}

at::Tensor wrapper_view(const at::Tensor& self, c10::SymIntArrayRef size) {
  return at::native::view_openreg(self, size);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("empty.memory_format", wrapper_empty_memory_format);
  m.impl("empty_strided", wrapper_empty_strided);
  m.impl("as_strided", wrapper_as_strided);
  m.impl("resize_", wrapper_resize_);
  m.impl("_reshape_alias", wrapper__reshape_alias);
  m.impl("_copy_from", wrapper__copy_from);
  m.impl("_copy_from_and_resize", wrapper__copy_from_and_resize);
  m.impl("_local_scalar_dense", wrapper__local_scalar_densor);
  m.impl("set_.source_Tensor", wrapper_set_source_Tensor_);
  m.impl("set_.source_Storage", wrapper_set_source_Storage_);
  m.impl(
      "set_.source_Storage_storage_offset",
      wrapper_set_source_Storage_storage_offsetset_);
  m.impl("view", wrapper_view);
}

void wrapper_cpu_fallback(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack) {
  at::native::cpu_fallback_openreg(op, stack);
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
  m.fallback(
      torch::CppFunction::makeFromBoxedFunction<&wrapper_cpu_fallback>());
}

} // namespace at::openreg
