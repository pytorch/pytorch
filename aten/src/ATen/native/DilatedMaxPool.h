#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/NativeFunctions.h>
#include <tuple>

#pragma once

namespace at {
namespace native {

namespace {

template <typename dest_t, typename src_t>
static inline dest_t
safe_downcast(src_t v)
{
  TORCH_CHECK(std::numeric_limits<dest_t>::min() <= v && v <= std::numeric_limits<dest_t>::max(),
              "integer out of range");

  return static_cast<dest_t>(v);
}

template<typename T>
static inline T pooling_output_shape(
        T inputSize, T kernelSize, T pad, T stride, T dilation, bool ceil_mode) {
    T outputSize = ((inputSize + 2 * pad - dilation * (kernelSize - 1) - 1 + (ceil_mode ? stride - 1 : 0)) / stride + 1);
    if (pad) {
        // ensure that the last pooling starts inside the image
        // needed to avoid problems in ceil mode
        if ((outputSize - 1) * stride >= inputSize + pad)
          --outputSize;
    }
    return outputSize;
}

static inline void
max_pool2d_with_indices_shape_check(
  const Tensor& input,
  int kH, int kW, int dH, int dW, int padH, int padW, int dilationH, int dilationW,
  int64_t nInputPlane,
  int64_t inputHeight, int64_t inputWidth,
  int64_t outputHeight, int64_t outputWidth)
{
  const int64_t ndim = input.ndimension();
  const int64_t nOutputPlane = nInputPlane;

  TORCH_CHECK(kW > 0 && kH > 0,
              "kernel size should be greater than zero, but got ",
              "kH: ", kH, " kW: ", kW);
  TORCH_CHECK(dW > 0 && dH > 0,
              "stride should be greater than zero, but got "
              "dH: ", dH, " dW: ", dW);
  TORCH_CHECK(dilationH > 0 && dilationW > 0,
              "dilation should be greater than zero, but got ",
              "dilationH: ", dilationH, " dilationW: ", dilationW);

  TORCH_CHECK(input.numel() > 0 && (ndim == 3 || ndim == 4),
              "non-empty 3D or 4D input tensor expected but got ndim: ", ndim);
  TORCH_CHECK(kW/2 >= padW && kH/2 >= padH,
              "pad should be smaller than half of kernel size, but got ",
              "padW = ", padW, ", padH = ", padH, ", kW = ", kW, ", kH = ", kH);

  if (outputWidth < 1 || outputHeight < 1) {
    AT_ERROR("Given input size: (",
              nInputPlane, "x", inputHeight, "x", inputWidth, "). ",
             "Calculated output size: (",
              nOutputPlane, "x", outputHeight, "x", outputWidth, "). ",
             "Output size is too small");
  }
}

static inline void
max_pool2d_with_indices_shape_check(
  const Tensor& input,
  const Tensor& gradOutput,
  const Tensor& indices,
  int64_t nbatch,
  int kH, int kW, int dH, int dW, int padH, int padW, int dilationH, int dilationW,
  int64_t nInputPlane,
  int64_t inputHeight, int64_t inputWidth,
  int64_t outputHeight, int64_t outputWidth,
  bool cuda=false)
{
  max_pool2d_with_indices_shape_check(
    input,
    kH, kW, dH, dW, padH, padW, dilationH, dilationW,
    nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth);

  const int64_t ndim = input.ndimension();
  const int64_t nOutputPlane = nInputPlane;

  check_dim_size(gradOutput, ndim, ndim-3, nOutputPlane);
  check_dim_size(gradOutput, ndim, ndim-2, outputHeight);
  check_dim_size(gradOutput, ndim, ndim-1, outputWidth);

  if (cuda) {
    check_dim_size(indices, 4, 0, nbatch);
    check_dim_size(indices, 4, 1, nOutputPlane);
    check_dim_size(indices, 4, 2, outputHeight);
    check_dim_size(indices, 4, 3, outputWidth);
  }
  else {
    check_dim_size(indices, ndim, ndim-3, nOutputPlane);
    check_dim_size(indices, ndim, ndim-2, outputHeight);
    check_dim_size(indices, ndim, ndim-1, outputWidth);
  }
}

} // namespace

} // at::native
} // at
