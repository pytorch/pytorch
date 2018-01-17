#pragma once

#include <ATen/ATen.h>
#include "torch/csrc/autograd/variable.h"

namespace torch {

inline size_t count_tensors_single(const at::Tensor& x) { return 1; }
inline size_t count_tensors_single(at::ArrayRef<at::Tensor> xs) { return xs.size(); }

template<typename... Args>
inline size_t count_tensors() {
  return 0;
}
template<typename T, typename... Args>
inline size_t count_tensors(T x, Args... args) {
  return count_tensors_single(x) + count_tensors(args...);
}

} // namespace torch
