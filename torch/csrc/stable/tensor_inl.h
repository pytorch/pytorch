#pragma once

// This file implements tensor.h. We separated out the Tensor struct so that
// other files can depend on the Tensor struct (like library.h) and the
// implementations of the Tensor methods can depend on APIs in library.h
// without circular dependencies.

#include <torch/csrc/stable/stableivalue_conversions.h>
#include <torch/csrc/stable/tensor_struct.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/shim_utils.h>

namespace torch::stable {

using torch::headeronly::ScalarType;

inline ScalarType Tensor::scalar_type() const {
  int32_t dtype;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype(ath_.get(), &dtype));
  return to<ScalarType>(from(dtype));
}

} // namespace torch::stable
