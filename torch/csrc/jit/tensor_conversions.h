#pragma once
#include "ATen/ATen.h"

template<typename T>
static inline T tensor_as(at::Tensor&& t) = delete;

template<>
inline int64_t tensor_as(at::Tensor&& t) {
  // workaround for 1-dim 1-element pytorch tensors until zero-dim
  // tensors are fully supported
  if(t.ndimension() == 1 && t.size(0) == 1) {
    t = t[0];
  }
  return at::Scalar(t).to<int64_t>();
}
