#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"

namespace at { namespace native { namespace detail {

  enum GridSamplerInterpolationMode {GridSamplerInterpolationBilinear, GridSamplerInterpolationNearest};
  enum GridSamplerPaddingMode {GridSamplerPaddingZeros, GridSamplerPaddingBorder, GridSamplerPaddingReflection};

}}}  // namespace at::native::detail
