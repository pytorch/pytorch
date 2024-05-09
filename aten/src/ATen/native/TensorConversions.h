#pragma once

#include <c10/core/Device.h>
#include <c10/core/Layout.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>

namespace at {
  class Tensor;
namespace native {
bool to_will_alias(
    const Tensor& self,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    bool copy,
    c10::optional<c10::MemoryFormat> optional_memory_format);

Tensor to_meta(const Tensor& tensor);
c10::optional<Tensor> to_meta(const c10::optional<Tensor>& tensor);
std::vector<Tensor> to_meta(at::ITensorListRef t_list);
Tensor dense_to_sparse_with_mask(const Tensor& self, const Tensor& mask, c10::optional<c10::Layout> layout, OptionalIntArrayRef blocksize, c10::optional<int64_t> dense_dim_opt);

} // namespace native
} // namespace at
