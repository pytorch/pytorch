#pragma once
#include <ATen/TypeDefault.h>
#include <ATen/cuda/ATenCUDAGeneral.h>

namespace at {

struct AT_CUDA_API CUDATypeDefault : public TypeDefault {
  CUDATypeDefault()
      : TypeDefault() {}
};

} // namespace at
