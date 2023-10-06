#pragma once

#ifdef USE_XNNPACK

#include <ATen/Tensor.h>

namespace at::native::xnnpack::internal::pooling {

struct Parameters final {

  std::array<int64_t, 2> kernel;
  std::array<int64_t, 2> padding;
  std::array<int64_t, 2> stride;
  std::array<int64_t, 2> dilation;

  explicit Parameters(
      const IntArrayRef kernel_,
      const IntArrayRef padding_,
      const IntArrayRef stride_,
      const IntArrayRef dilation_)
  : kernel(normalize(kernel_)),
    padding(normalize(padding_)),
    stride(normalize(stride_)),
    dilation(normalize(dilation_)) {
  }

private:
  static std::array<int64_t, 2> normalize(const IntArrayRef parameter) {
    TORCH_INTERNAL_ASSERT(
        !parameter.empty(),
        "Invalid usage!  Reason: normalize() was called on an empty parameter.");

    return std::array<int64_t, 2>{
      parameter[0],
      (2 == parameter.size()) ? parameter[1] : parameter[0],
    };
  }
};

} // namespace at::native::xnnpack::internal::pooling

#endif /* USE_XNNPACK */
