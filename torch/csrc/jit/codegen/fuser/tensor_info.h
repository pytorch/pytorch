#pragma once
#include <torch/csrc/Export.h>

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
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  uint32_t sizes_strides[0];
#pragma GCC diagnostic pop
};

} // namespace fuser
} // namespace jit
} // namespace torch
