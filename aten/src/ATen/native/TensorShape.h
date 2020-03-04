#pragma once
#include <ATen/Tensor.h>

namespace at {
namespace native {

TORCH_API void set_tensor_storage(const Tensor& tensor, at::Storage storage, ptrdiff_t storage_offset);
Tensor& set_cpu_(Tensor& self, Storage storage, ptrdiff_t storage_offset, IntArrayRef size, IntArrayRef stride);
Tensor& set_cpu_(Tensor& self);

}} // at::native
