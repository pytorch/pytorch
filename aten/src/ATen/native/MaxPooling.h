#pragma once

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/native/DispatchStub.h>

namespace at {
namespace native {

// TODO(Heitor) Template by dimension
struct PoolingParams1D {
  int64_t NB; // Number of batches
  int64_t NC; // Number of channels
  int64_t IW; // Input width
  int64_t OW; // Output width
  int64_t KW; // Kernel width
  int64_t SJ; // Column stride
  int64_t PJ; // Column padding
  int64_t DJ; // Column dilation

  // Return index of first output within bounds for this kernel index
  inline int64_t valid_kernel_start(int64_t kj) const {
    int64_t ij = kj * DJ - PJ;
    return ij < 0 ? at::divup(-ij, SJ) : 0;
  }

  // Return index one past last output within bounds for this kernel index
  inline int64_t valid_kernel_end(int64_t kj) const {
    int64_t ij = (OW - 1) * SJ + kj * DJ - PJ;
    return ij >= IW ? OW - at::divup(ij - (IW - 1), SJ) : OW;
  }
};

using pooling_fn = void (*)(Tensor&, const Tensor&, const PoolingParams1D&);

DECLARE_DISPATCH(pooling_fn, max_pool1d_stub);

} // namespace native
} // namespace at
