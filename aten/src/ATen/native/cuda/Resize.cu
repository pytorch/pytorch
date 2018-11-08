#include "ATen/ATen.h"
#include "ATen/cuda/CUDAContext.h"

#include "ATen/native/cuda/Resize.cuh"

namespace at { namespace native {

Tensor& resize_cuda_(Tensor& self, IntList size) {
  return resize_cuda_helper_(self, size);
}

}}
