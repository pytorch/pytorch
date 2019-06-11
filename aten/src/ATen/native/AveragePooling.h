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
  avg_pool2d_shape_check(
    const Tensor& input,
    int kH, int kW, int dH, int dW, int padH, int padW, bool ceil_mode, 
    int64_t nInputPlane,
    int64_t inputHeight, int64_t inputWidth,
    int64_t outputHeight, int64_t outputWidth)
  {
    TORCH_CHECK(kW > 0 && kH > 0,
             "kernel size should be greater than zero, but got kH: ", kH, " kW: ", kW);
    TORCH_CHECK(dW > 0 && dH > 0,
              "stride should be greater than zero, but got dH: ", dH, " dW: ", dW);

    TORCH_CHECK(kW/2 >= padW && kH/2 >= padH,
              "pad should be smaller than half of kernel size, but got "
              "padW =", padW, ", padH = ", padH, ", kW = ", kW, ", kH = ", kH);

    if (outputWidth < 1 || outputHeight < 1)
      AT_ERROR(
          "Given input size: (", 
          nInputPlane, 
          "x", 
          inputHeight, 
          "x", 
          inputWidth, 
          "). Calculated output size: (", 
          nInputPlane, 
          "x", 
          outputHeight, 
          "x", 
          outputWidth, 
          "). Output size is too small");
  }
} // namespace

} // at::native
} // at