#pragma once

#include <ATen/core/Tensor.h>

namespace c10 {

namespace detail {
template<typename T>
inline TypePtr getTypePtr_<T>::call() {
  if (!isCustomClassRegistered<T>()) {
    throw c10::Error("Type could not be converted to any of the known types.", "");
  }
  auto res = getCustomClassType<T>();
  return std::dynamic_pointer_cast<Type>(res.type_);
}
}

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
