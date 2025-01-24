#pragma once

#include <c10/core/Device.h>
#include <c10/core/Layout.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <optional>

namespace at {
class Tensor;
namespace native {
bool to_will_alias(
    const Tensor& self,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    bool copy,
    std::optional<c10::MemoryFormat> optional_memory_format);

Tensor to_meta(const Tensor& tensor);
std::optional<Tensor> to_meta(const std::optional<Tensor>& tensor);
std::vector<Tensor> to_meta(at::ITensorListRef t_list);
Tensor dense_to_sparse_with_mask(
    const Tensor& self,
    const Tensor& mask,
    std::optional<c10::Layout> layout,
    OptionalIntArrayRef blocksize,
    std::optional<int64_t> dense_dim_opt);

} // namespace native
} // namespace at
