#include "Common.h"

namespace at::native {

at::Tensor empty_memory_format_openreg(
    c10::IntArrayRef size,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt);

at::Tensor empty_strided_openreg(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt,
    std::optional<bool> pin_memory_opt);

at::Tensor as_strided_openreg(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    std::optional<c10::SymInt> storage_offset);

const at::Tensor& resize_openreg_(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    ::std::optional<at::MemoryFormat> memory_format);

at::Tensor _reshape_alias_openreg(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride);

at::Tensor _copy_from_openreg(
    const at::Tensor& self,
    const at::Tensor& dst,
    bool non_blocking);

at::Tensor _copy_from_and_resize_openreg(
    const at::Tensor& self,
    const at::Tensor& dst);

at::Scalar _local_scalar_dense_openreg(const at::Tensor& self);

at::Tensor& set_source_Tensor_openreg_(
    at::Tensor& self,
    const at::Tensor& source);

at::Tensor& set_source_Storage_openreg_(at::Tensor& self, at::Storage source);

at::Tensor& set_source_Storage_storage_offset_openreg_(
    at::Tensor& result,
    at::Storage storage,
    int64_t storage_offset,
    c10::IntArrayRef size,
    c10::IntArrayRef stride);

at::Tensor view_openreg(const at::Tensor& self, c10::SymIntArrayRef size);

void cpu_fallback_openreg(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack);

} // namespace at::native
