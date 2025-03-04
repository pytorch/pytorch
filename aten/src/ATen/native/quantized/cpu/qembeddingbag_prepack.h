#pragma once
#include <ATen/core/Tensor.h>

namespace at::native {

Tensor& qembeddingbag_byte_prepack_out(Tensor& output, const Tensor& weight);

Tensor qembeddingbag_byte_prepack(const Tensor& weight);

Tensor qembeddingbag_byte_prepack_meta(const Tensor& weight);

} // namespace at::native
