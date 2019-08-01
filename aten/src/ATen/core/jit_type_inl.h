#pragma once

#include <ATen/core/Tensor.h>

namespace c10 {

inline ProfiledTensorType::ProfiledTensorType(const at::Tensor& tensor)
    : TensorType(),
      scalar_type_(tensor.scalar_type()),
      device_(tensor.device()),
      sizes_(tensor.sizes().vec()),
      strides_(tensor.strides().vec()),
      requires_grad_(tensor.requires_grad()) {}

inline DimensionedTensorType::DimensionedTensorType(
    const at::Tensor& tensor,
    TypeKind kind)
    : DimensionedTensorType(
          tensor.scalar_type(),
          tensor.device(),
          tensor.dim(),
          tensor.is_variable() && tensor.requires_grad(),
          kind) {}

inline CompleteTensorType::CompleteTensorType(const at::Tensor& tensor)
    : DimensionedTensorType(tensor, TypeKind::CompleteTensorType),
      sizes_(tensor.sizes().vec()),
      strides_(tensor.strides().vec()) {}

}
