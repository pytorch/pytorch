#pragma once

#include <ATen/ATen.h>
#include "torch/csrc/autograd/variable.h"

namespace torch {

template<typename... Args> inline size_t countTensors();
template<typename... Args> inline size_t countTensors(const at::Tensor& x, Args... args);
template<typename... Args> inline size_t countTensors(at::ArrayRef<at::Tensor> xs, Args... args);

template<typename... Args>
inline size_t countTensors() {
  return 0;
}
template<typename... Args>
inline size_t countTensors(const at::Tensor& x, Args... args) {
  return 1 + countTensors(args...);
}
template<typename... Args>
inline size_t countTensors(at::ArrayRef<at::Tensor> xs, Args... args) {
  return xs.size() + countTensors(args...);
}

} // namespace torch
