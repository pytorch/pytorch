#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/VulkanPackedContext.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

enum Conv2dMethod {
  Conv2dDepthwise,
  Conv2dPointwise,
  Conv2dSlidingWindow,
};

namespace conv2d {

Tensor rearrange_weights_dw(const Tensor& weight_in);
Tensor rearrange_weights_2d(const Tensor& weight_in, bool tconv);
Tensor rearrange_bias(
    const c10::optional<Tensor>& bias_in,
    const at::Tensor& weight_in,
    bool tconv);

} // namespace conv2d

namespace qconv2d_vk {

struct QParams final {
  api::utils::uvec3 outExtents;
  int32_t ic4;
  api::utils::ivec4 sizes2d;
  float outputScale;
  float inputScale;
  int32_t outputZeroPoint;
  int32_t inputZeroPoint;
  float weightScale;
  float biasScale;
  int32_t weightZeroPoint;
  int32_t biasZeroPoint;
  api::utils::ivec2 kernelSize;
  api::utils::ivec2 stride;
  api::utils::ivec2 padding;
  api::utils::ivec2 dilate;
  api::utils::vec2 clamp;
  api::utils::ivec4 srcFilter;
};

} // namespace qconv2d_vk

class Conv2dPackedContext final : virtual public VulkanPackedContext,
                                  public torch::jit::CustomClassHolder {
 private:
  c10::impl::GenericList unpacked_;
  api::ShaderInfo computeShader_{};

 public:
  Conv2dPackedContext(
      const Tensor& weight,
      const c10::optional<Tensor>& bias,
      const IntArrayRef stride_arg,
      const IntArrayRef padding_arg,
      const IntArrayRef dilation_arg,
      const bool transposed,
      const bool quantized,
      const IntArrayRef output_padding_arg,
      const int64_t groups,
      const c10::optional<Scalar>& output_min = c10::nullopt,
      const c10::optional<Scalar>& output_max = c10::nullopt);

  /*
   * Assigns a name to each index in the unpacked list.
   */
  struct Unpacked final {
    static constexpr uint32_t WEIGHT = 0u;
    static constexpr uint32_t BIAS = 1u;
    static constexpr uint32_t STRIDE = 2u;
    static constexpr uint32_t PADDING = 3u;
    static constexpr uint32_t DILATION = 4u;
    static constexpr uint32_t IS_TRANSPOSED = 5u;
    static constexpr uint32_t IS_QUANTIZED = 6u;
    static constexpr uint32_t OUTPUT_PADDING = 7u;
    static constexpr uint32_t GROUPS = 8u;
    static constexpr uint32_t OUTPUT_MIN = 9u;
    static constexpr uint32_t OUTPUT_MAX = 10u;

    static constexpr uint32_t NUM_ARGS = 11u;
  };

  /*
   * Assigns a name to each index in the packed list.
   */
  struct Packed final {
    static constexpr uint32_t WEIGHT = 0u;
    static constexpr uint32_t BIAS = 1u;
    static constexpr uint32_t OVERLAY_REGION = 2u;
    static constexpr uint32_t STRIDE = 3u;
    static constexpr uint32_t PADDING = 4u;
    static constexpr uint32_t OUTPUT_PADDING = 5u;
    static constexpr uint32_t DILATION = 6u;
    static constexpr uint32_t IS_TRANSPOSED = 7u;
    static constexpr uint32_t IS_QUANTIZED = 8u;
    static constexpr uint32_t GROUPS = 9u;
    static constexpr uint32_t OUTPUT_MIN = 10u;
    static constexpr uint32_t OUTPUT_MAX = 11u;
    static constexpr uint32_t CONV_METHOD = 12u;
    static constexpr uint32_t WEIGHT_SIZES = 13u;

    static constexpr uint32_t NUM_ARGS = 14u;
  };

  static Conv2dPackedContext pack(c10::impl::GenericList);

  const c10::impl::GenericList unpack() const override {
    TORCH_CHECK(!unpacked_.empty(), "unpacked_ does not have any elements!");

    return unpacked_;
  }

  inline api::ShaderInfo& compute_shader() {
    return computeShader_;
  }
};

c10::intrusive_ptr<Conv2dPackedContext> create_conv2d_context(
    Tensor&& weight,
    c10::optional<Tensor>&& bias,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& dilation,
    const int64_t groups,
    const c10::optional<Scalar>& output_min = c10::nullopt,
    const c10::optional<Scalar>& output_max = c10::nullopt);

Tensor run_conv2d_context(
    const Tensor& input,
    const c10::intrusive_ptr<Conv2dPackedContext>& context);

c10::intrusive_ptr<Conv2dPackedContext> create_tconv2d_context(
    Tensor&& weight,
    c10::optional<Tensor>&& bias,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& output_padding,
    std::vector<int64_t>&& dilation,
    const int64_t groups,
    const c10::optional<Scalar>& output_min = c10::nullopt,
    const c10::optional<Scalar>& output_max = c10::nullopt);

Tensor run_tconv2d_context(
    const Tensor& input,
    const c10::intrusive_ptr<Conv2dPackedContext>& context);

c10::intrusive_ptr<Conv2dPackedContext> create_qconv2d_context(
    Tensor&& weight,
    c10::optional<Tensor>&& bias,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& dilation,
    const int64_t groups,
    const c10::optional<Scalar>& output_min = c10::nullopt,
    const c10::optional<Scalar>& output_max = c10::nullopt);

Tensor run_qconv2d_context(
    const Tensor& input_arg,
    double scale,
    int64_t zero_point,
    const c10::intrusive_ptr<Conv2dPackedContext>& conv_context);

// Backwards compatibility
class Conv2dOpContext final : public torch::jit::CustomClassHolder {
 public:
  static Conv2dOpContext create(
      const Tensor& weight,
      const c10::optional<Tensor>& bias,
      IntArrayRef stride,
      IntArrayRef padding,
      IntArrayRef dilation,
      bool transposed,
      IntArrayRef output_padding,
      int64_t groups,
      const c10::optional<Scalar>& output_min = c10::nullopt,
      const c10::optional<Scalar>& output_max = c10::nullopt);

  using State = std::tuple<
      Tensor,
      c10::optional<Tensor>,
      std::vector<int64_t>,
      std::vector<int64_t>,
      std::vector<int64_t>,
      int64_t,
      c10::optional<Scalar>,
      c10::optional<Scalar>>;

  Tensor run(const Tensor& input) const;
  State unpack() const;

 private:
  explicit Conv2dOpContext(Conv2dPackedContext conv_context);
  Conv2dPackedContext convContext_;
};

Tensor conv2d_clamp_run(
    const Tensor& input,
    const c10::intrusive_ptr<Conv2dOpContext>& context);

c10::intrusive_ptr<Conv2dOpContext> conv2d_clamp_prepack(
    Tensor&& weight,
    c10::optional<Tensor>&& bias,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& dilation,
    const int64_t groups,
    const c10::optional<Scalar>& output_min,
    const c10::optional<Scalar>& output_max);

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
