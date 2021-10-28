#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>
#include <torch/custom_class.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

enum TransposeConv2dMethod {
  TransposeConv2dSlidingWindow,
};

class TransposeConv2dOpContext final : public torch::jit::CustomClassHolder {
 public:
  static TransposeConv2dOpContext create(
      const Tensor& weight,
      const c10::optional<Tensor>& bias,
      IntArrayRef stride,
      IntArrayRef padding,
      IntArrayRef output_padding,
      IntArrayRef dilation,
      int64_t groups,
      const c10::optional<Scalar>& output_min = c10::nullopt,
      const c10::optional<Scalar>& output_max = c10::nullopt);

  using State = std::tuple<
      Tensor,
      c10::optional<Tensor>,
      std::vector<int64_t>,
      std::vector<int64_t>,
      std::vector<int64_t>,
      std::vector<int64_t>,
      int64_t,
      c10::optional<Scalar>,
      c10::optional<Scalar>>;

  Tensor run(const Tensor& input) const;
  State unpack() const;

 private:
  TransposeConv2dOpContext(
      const Tensor& weight,
      const c10::optional<Tensor>& bias,
      IntArrayRef stride,
      IntArrayRef padding,
      IntArrayRef output_padding,
      IntArrayRef dilation,
      int64_t groups,
      const c10::optional<Scalar>& output_min = c10::nullopt,
      const c10::optional<Scalar>& output_max = c10::nullopt);

  void conv2d_transpose_sliding_window(
      const api::Shader::Descriptor& shader,
      vTensor& v_output,
      const vTensor& v_input) const;

 private:
  struct {
    vTensor v_weight;
    vTensor v_bias;
    std::array<int64_t, 4> filter;
    std::array<int64_t, 2> stride;
    std::array<int64_t, 2> padding;
    std::array<int64_t, 2> output_padding;
    std::array<int64_t, 2> dilation;
    int32_t groups;
    float output_min;
    float output_max;
  } packed_;

  struct {
    Tensor weight;
    c10::optional<Tensor> bias;
    std::vector<int64_t> filter;
    std::vector<int64_t> stride;
    std::vector<int64_t> padding;
    std::vector<int64_t> output_padding;
    std::vector<int64_t> dilation;
    int64_t groups;
    c10::optional<Scalar> output_min;
    c10::optional<Scalar> output_max;
  } unpacked_;

};

Tensor conv2d_transpose_clamp_run(
    const Tensor& input,
    const c10::intrusive_ptr<TransposeConv2dOpContext>& context);

c10::intrusive_ptr<TransposeConv2dOpContext> conv2d_transpose_clamp_prepack(
    Tensor&& weight,
    c10::optional<Tensor>&& bias,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& output_padding,
    std::vector<int64_t>&& dilation,
    const int64_t groups,
    const c10::optional<Scalar>& output_min,
    const c10::optional<Scalar>& output_max);

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
