#pragma once
#ifdef NAMEDTENSOR_ENABLED

#include <ATen/core/Tensor.h>

namespace at {

inline bool has_names(TensorList tensors) {
  return std::any_of(
      tensors.begin(), tensors.end(), [](const Tensor& t) { return t.is_named(); });
}

}

#endif
