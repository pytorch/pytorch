
copy: fbcode/caffe2/torch/csrc/jit/codegen/fuser/tensor_info.h
copyrev: e0a21352c98113f0c745a8538d9b7229cdf8a98b

#pragma once
#include <torch/csrc/WindowsTorchApiMacro.h>

#include <cstdint>

namespace torch {
namespace jit {
namespace fuser {

// Host-side view of TensorInfo
// Note dims[0] - we need to dynamically allocate the dims.
struct TORCH_API TensorInfo {
  uint32_t* sizes(size_t nDim) {
    return &sizes_strides[0];
  }
  uint32_t* strides(size_t nDim) {
    return &sizes_strides[nDim];
  }

  void* data;
#pragma GCC diagnostic ignored "-Wpedantic"
  uint32_t sizes_strides[0];
#pragma GCC diagnostic pop
};

} // namespace fuser
} // namespace jit
} // namespace torch
