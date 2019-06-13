#pragma once
#ifdef NAMEDTENSOR_ENABLED

#include <ATen/NamedTensor.h>
#include <ATen/core/Tensor.h>

namespace at {

inline bool has_names(TensorList tensors) {
  return std::any_of(
      tensors.begin(), tensors.end(), [](const Tensor& t) { return t.is_named(); });
}

namespace namedinference {

optional<std::vector<Dimname>> reduction_op(optional<DimnameList> self_names, int64_t dim);

} // namespace namedinference

} // namespace at
#endif
