#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/Parallel.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Pool.h>

namespace at::native {

inline void check_max_pool1d(
    const Tensor& self,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {

  TORCH_CHECK(
      self.dim() == 2 || self.dim() == 3,
      "max_pool1d() Expected 2D or 3D input tensor, but got ", self.sym_sizes());
  TORCH_CHECK(
      kernel_size.size() == 1,
      "max_pool1d() kernel_size must be an int, list of ints or tuple of ints of size 1 but got size ",
      kernel_size.size());
  TORCH_CHECK(
      stride.empty() || stride.size() == 1,
      "max_pool1d() stride must be None, an int, list of ints, or tuple of ints of size 1 but got size ",
      stride.size());
  TORCH_CHECK(
      padding.size() == 1,
      "max_pool1d() padding must be an int, list of ints, or tuple of ints of size 1 but got size ",
      padding.size());
  TORCH_CHECK(
      dilation.size() == 1,
      "max_pool1d() dilation must be an int, list of ints or tuple of ints of size 1 but got size ",
      dilation.size());

  // If stride=None then set it to kernel_size
  if (stride.empty()) {
    stride = kernel_size;
  }

  TORCH_CHECK(
      kernel_size[0] > 0,
      "max_pool1d() kernel_size must be greater than zero, but got ",
      kernel_size[0]);
  TORCH_CHECK(
      stride[0] > 0, "max_pool1d() stride must be greater than zero, but got ", stride[0]);
  TORCH_CHECK(
      padding[0] >= 0, "max_pool1d() padding must be non-negative, but got ", padding[0]);
  TORCH_CHECK(
      padding[0] <= kernel_size[0] / 2,
      "max_pool1d() padding should be at most half of kernel size, but got padding=",
      padding[0],
      " and kernel_size=",
      kernel_size[0]);
  TORCH_CHECK(
      dilation[0] > 0, "max_pool1d() dilation must be greater than zero, but got ", dilation[0]);

  const int64_t OW = pooling_output_shape(self.sym_size(-1).guard_int(__FILE__, __LINE__), kernel_size[0], padding[0], stride[0], dilation[0], ceil_mode);
  TORCH_CHECK(OW > 0, "max_pool1d() Invalid computed output size: ", OW);
}

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

  // Return index of input element for the given kernel and output index
  inline int64_t index(int64_t kj, int64_t oj) const {
    return oj * SJ + kj * DJ - PJ;
  }

  // Return index of first output within bounds for this kernel index
  inline int64_t valid_output_start(int64_t kj) const {
    int64_t ij = index(kj, 0);;
    return ij < 0 ? at::divup(-ij, SJ) : 0;
  }

  // Return index one past last output within bounds for this kernel index
  inline int64_t valid_output_end(int64_t kj) const {
    int64_t ij = index(kj, OW - 1);
    return ij >= IW ? OW - at::divup(ij - (IW - 1), SJ) : OW;
  }
};

using pooling_fn = void (*)(Tensor&, const Tensor&, const PoolingParams1D&);

DECLARE_DISPATCH(pooling_fn, max_pool1d_stub);

} // namespace at::native
