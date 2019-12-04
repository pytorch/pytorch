#include <ATen/ATen.h>

namespace at { namespace native {

struct ResultTypeState {
  c10::ScalarType dimResult = ScalarType::Undefined;
  c10::ScalarType wrappedResult = ScalarType::Undefined;
  c10::ScalarType zeroResult = ScalarType::Undefined;
};

ResultTypeState update_result_type_state(const Tensor& tensor, const ResultTypeState& in_state);
ScalarType result_type(const ResultTypeState& state);
ScalarType result_type(TensorList tensors);

}}
