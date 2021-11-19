#pragma once

#include <ATen/ATen.h>
#include <ATen/core/ITensorList.h>

namespace at { namespace native {

struct ResultTypeState {
  c10::ScalarType dimResult = ScalarType::Undefined;
  c10::ScalarType wrappedResult = ScalarType::Undefined;
  c10::ScalarType zeroResult = ScalarType::Undefined;
};

TORCH_API ResultTypeState update_result_type_state(const Tensor& tensor, const ResultTypeState& in_state);
TORCH_API ScalarType result_type(const ResultTypeState& state);

TORCH_API ScalarType result_type(ITensorList tensors);

}}
