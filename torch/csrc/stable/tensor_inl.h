#pragma once

// This file implements tensor.h. We separated out the Tensor struct so that
// other files can depend on the Tensor struct (like library.h) and the
// implementations of the Tensor methods can depend on APIs in library.h
// without circular dependencies.

#include <torch/csrc/stable/ScalarType.h>
#include <torch/csrc/stable/stableivalue_conversions.h>
#include <torch/csrc/stable/tensor_struct.h>
#include <torch/headeronly/util/shim_utils.h>

namespace torch::stable {

using torch::headeronly::ScalarType;

ScalarType Tensor::scalar_type() const {
  int32_t dtype;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype(ath_.get(), &dtype));
  return to<ScalarType>(from(dtype));
}

#define DEFINE_DATA_PTR_CAST(T, name)                         \
  template <>                                                 \
  T* Tensor::mutable_data_ptr() const {                       \
    STD_TORCH_CHECK(                                          \
        scalar_type() == torch::headeronly::ScalarType::name, \
        "expected scalar type " #name " but found ",          \
        toString(scalar_type()));                             \
    return static_cast<T*>(mutable_data_ptr());               \
  }                                                           \
  template <>                                                 \
  const T* Tensor::const_data_ptr() const {                   \
    STD_TORCH_CHECK(                                          \
        scalar_type() == torch::headeronly::ScalarType::name, \
        "expected scalar type " #name " but found ",          \
        toString(scalar_type()));                             \
    return static_cast<const T*>(const_data_ptr());           \
  }

STABLE_FORALL_SUPPORTED_SCALAR_TYPES(DEFINE_DATA_PTR_CAST)
#undef DEFINE_DATA_PTR_CAST

} // namespace torch::stable
