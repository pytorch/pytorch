#pragma once

#include <ATen/Tensor.h>
#include <c10/util/Half.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace at {
template <>
inline __half* Tensor::data() const {
  return reinterpret_cast<__half*>(data<Half>());
}
} // namespace at
