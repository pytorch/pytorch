#include <ATen/native/vulkan/VulkanCommon.h>
#include <c10/util/ArrayRef.h>

namespace at {
namespace native {
namespace vulkan {

Conv2DParams::Conv2DParams(
    c10::IntArrayRef isizes,
    int64_t OC,
    int64_t KH,
    int64_t KW,
    int64_t SY,
    int64_t SX,
    int64_t PY,
    int64_t PX,
    int64_t DY,
    int64_t DX,
    int64_t G)
    : N(isizes[0]),
      C(isizes[1]),
      H(isizes[2]),
      W(isizes[3]),
      OC(OC),
      KH(KH),
      KW(KW),
      SY(SY),
      SX(SX),
      PY(PY),
      PX(PX),
      DY(DY),
      DX(DX),
      G(G) {
  const int64_t KWE = (KW - 1) * DX + 1;
  const int64_t KHE = (KH - 1) * DY + 1;
  OW = ((W - KWE + 2 * PX) / SX) + 1;
  OH = ((H - KHE + 2 * PY) / SY) + 1;
  OC_4 = UP_DIV(OC, 4);
  C_4 = UP_DIV(C, 4);
}

Conv2DParams::Conv2DParams(
    c10::IntArrayRef isizes,
    c10::IntArrayRef wsizes,
    c10::IntArrayRef padding,
    c10::IntArrayRef stride,
    c10::IntArrayRef dilation,
    int64_t groups)
    : Conv2DParams(
          isizes,
          wsizes[0],
          wsizes[2],
          wsizes[3],
          stride[0],
          stride[1],
          padding[0],
          padding[1],
          dilation[0],
          dilation[1],
          groups) {}

} // namespace vulkan
} // namespace native
} // namespace at
