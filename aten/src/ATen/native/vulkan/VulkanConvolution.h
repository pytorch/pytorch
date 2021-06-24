#pragma once

#include <array>

#include <ATen/Tensor.h>
#include <ATen/native/vulkan/VulkanCommon.h>
#include <ATen/native/vulkan/VulkanOpContext.h>
#include <c10/util/ArrayRef.h>

namespace at {
namespace native {
namespace vulkan {

struct Conv2DParams final {
  int64_t N; // batch size
  int64_t C; // channels
  int64_t H; // input height
  int64_t W; // input width
  int64_t OC; // output channels
  int64_t KH; // kernel height
  int64_t KW; // kernel width
  int64_t SY; // stride y (height)
  int64_t SX; // stride x (width)
  int64_t PY; // padding y (height)
  int64_t PX; // padding x (width)
  int64_t DY; // dilation y (height)
  int64_t DX; // dilation x (width)
  int64_t G; // groups
  int64_t OW; // output width
  int64_t OH; // output height
  int64_t OC_4;
  int64_t C_4;

  Conv2DParams() = delete;
  Conv2DParams(
      c10::IntArrayRef inputSizes,
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
    // TODO: What if inputSizes is not of the expected dimensionality?
    // Should check prior to indexing.
      : N(inputSizes[0]),
        C(inputSizes[1]),
        H(inputSizes[2]),
        W(inputSizes[3]),
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
    OC_4 = UP_DIV(OC, 4);
    C_4 = UP_DIV(C, 4);
    const int64_t KWE = (KW - 1) * DX + 1;
    const int64_t KHE = (KH - 1) * DY + 1;
    OW = ((W - KWE + 2 * PX) / SX) + 1;
    OH = ((H - KHE + 2 * PY) / SY) + 1;
  }

  Conv2DParams(
      c10::IntArrayRef inputSizes,
      c10::IntArrayRef weightSizes,
      c10::IntArrayRef padding,
      c10::IntArrayRef stride,
      c10::IntArrayRef dilation,
      int64_t groups)
    // TODO: What if these parameters are not of the correct dimensionality?
    // Should check prior to indexing.
      : Conv2DParams(
            inputSizes,
            weightSizes[0],
            weightSizes[2],
            weightSizes[3],
            stride[0],
            stride[1],
            padding[0],
            padding[1],
            dilation[0],
            dilation[1],
            groups) {}

  std::vector<int64_t> output_sizes() const {
    return {N, OC, OH, OW};
  }
};

namespace detail {
namespace convolution2d {

c10::intrusive_ptr<at::native::vulkan::Conv2dOpContext>
createConv2dClampPrePackOpContext(
    Tensor&& weight,
    c10::optional<Tensor>&& bias,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& dilation,
    int64_t groups,
    const c10::optional<Scalar>& output_min,
    const c10::optional<Scalar>& output_max);

Tensor conv2d_clamp_run(
    const Tensor& input,
    const c10::intrusive_ptr<at::native::vulkan::Conv2dOpContext>& op_context);

ContextConv2D create(
    const Tensor& weight,
    const c10::optional<Tensor>& bias,
    const IntArrayRef padding,
    const IntArrayRef stride,
    const IntArrayRef dilation,
    const int64_t groups,
    const float output_min,
    const float output_max);

Tensor run(const ContextConv2D& context, const Tensor& input);

} // namespace convolution2d
} // namespace detail
} // namespace vulkan
} // namespace native
} // namespace at
