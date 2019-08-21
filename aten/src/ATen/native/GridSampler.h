#pragma once

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

namespace at { namespace native { namespace detail {

  enum class GridSamplerInterpolation {Bilinear, Nearest};
  enum class GridSamplerPadding {Zeros, Border, Reflection};

}}}  // namespace at::native::detail
