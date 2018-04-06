#pragma once

#include <ATen/ATen.h>

namespace torch {
// RAII structs to set CUDA device
struct AutoGPU {
  explicit AutoGPU(int device = -1);
  explicit AutoGPU(const at::Tensor& t);
  explicit AutoGPU(at::TensorList& tl);
  ~AutoGPU();

  void setDevice(int device);

  int original_device = -1;
};
} // namespace torch
