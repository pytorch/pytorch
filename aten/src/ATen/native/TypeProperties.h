#pragma once

#include <ATen/ATen.h>

namespace at { namespace native {

struct ResultTypeState {
  c10::ScalarType dimResult = ScalarType::Undefined;
  c10::ScalarType wrappedResult = ScalarType::Undefined;
  c10::ScalarType zeroResult = ScalarType::Undefined;
};

CAFFE2_API ResultTypeState update_result_type_state(const Tensor& tensor, const ResultTypeState& in_state);
CAFFE2_API ScalarType result_type(const ResultTypeState& state);

CAFFE2_API ScalarType result_type(TensorList tensors);

}}
