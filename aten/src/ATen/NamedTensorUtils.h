#pragma once
#ifdef NAMEDTENSOR_ENABLED

#include <ATen/NamedTensor.h>
#include <ATen/core/Tensor.h>

namespace at {

inline bool has_names(TensorList tensors) {
  return std::any_of(
      tensors.begin(), tensors.end(), [](const Tensor& t) { return t.is_named(); });
}

// Sets the names of `tensor` to be `names`.
CAFFE2_API void internal_set_names_inplace(Tensor& tensor, optional<DimnameList> names);

// Converts dim to an positional index. Errors if `dim` cannot be used to
// refer to any dimension of tensor.
CAFFE2_API int64_t dimname_to_position(const Tensor& tensor, Dimname dim);

namespace namedinference {

optional<std::vector<Dimname>> erase_name(optional<DimnameList> self_names, int64_t dim);

} // namespace namedinference

} // namespace at
#endif
