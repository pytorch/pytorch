#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/core/IListRef.h>

namespace at::native {

struct ResultTypeState {
  c10::ScalarType dimResult = ScalarType::Undefined;
  c10::ScalarType wrappedResult = ScalarType::Undefined;
  c10::ScalarType zeroResult = ScalarType::Undefined;
};

TORCH_API ResultTypeState update_result_type_state(const Tensor& tensor, const ResultTypeState& in_state);
TORCH_API ResultTypeState update_result_type_state(const Scalar& scalar, const ResultTypeState& in_state);
TORCH_API ScalarType result_type(const ResultTypeState& state);

TORCH_API ScalarType result_type(ITensorListRef tensors);

} // namespace at::native
