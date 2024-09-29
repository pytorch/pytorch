#import <ATen/native/metal/MetalConvParams.h>

#include <cmath>

namespace at {
namespace native {
namespace metal {

Conv2DParams::Conv2DParams(
    c10::IntArrayRef inputSizes,
    c10::IntArrayRef weightSizes,
    c10::IntArrayRef padding,
    c10::IntArrayRef stride,
    c10::IntArrayRef dilation,
    int64_t groups)
    : N(inputSizes[0]),
      C(inputSizes[1]),
      H(inputSizes[2]),
      W(inputSizes[3]),
      OC(weightSizes[0]),
      IC(weightSizes[1]),
      KH(weightSizes[2]),
      KW(weightSizes[3]),
      SY(stride[0]),
      SX(stride[1]),
      PY(padding[0]),
      PX(padding[1]),
      DY(dilation[0]),
      DX(dilation[1]),
      G(groups) {
  OH = std::floor((H + 2 * PY - DY * (KH - 1) - 1) / SY + 1);
  OW = std::floor((W + 2 * PX - DX * (KW - 1) - 1) / SX + 1);
};

}
}
}
