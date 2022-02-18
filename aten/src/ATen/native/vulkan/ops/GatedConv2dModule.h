#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Convolution.h>
#include <torch/custom_class.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

class GatedConv2dModuleOpContext final : public torch::jit::CustomClassHolder {
 public:
  static GatedConv2dModuleOpContext create(
      const Tensor& weight_a,
      const c10::optional<Tensor>& bias_a,
      const Tensor& weight_b,
      const c10::optional<Tensor>& bias_b,
      IntArrayRef stride,
      IntArrayRef padding,
      IntArrayRef output_padding,
      IntArrayRef dilation,
      const int64_t groups,
      const bool transposed);

  using State = std::tuple<
      Tensor,
      c10::optional<Tensor>,
      Tensor,
      c10::optional<Tensor>,
      std::vector<int64_t>,
      std::vector<int64_t>,
      std::vector<int64_t>,
      std::vector<int64_t>,
      int64_t,
      bool>;

  Tensor run(const Tensor& padding, const Tensor& prev_out) const;
  Tensor run_transpose(
      const Tensor& prev_out, const Tensor& prev_enc_out, const Tensor& encoder_out, const Tensor& padding) const;
  State unpack() const;

  GatedConv2dModuleOpContext(
      const Tensor& weight_a,
      const c10::optional<Tensor>& bias_a,
      const Tensor& weight_b,
      const c10::optional<Tensor>& bias_b,
      IntArrayRef stride,
      IntArrayRef padding,
      IntArrayRef output_padding,
      IntArrayRef dilation,
      int64_t groups,
      bool transposed);

  struct {
    vTensor v_weight;
    vTensor v_bias;
    std::array<int64_t, 4> filter;
    std::array<int64_t, 2> stride;
    std::array<int64_t, 2> padding;
    std::array<int64_t, 2> output_padding;
    std::array<int64_t, 2> dilation;
  } packed_;

  struct {
    Tensor weight_a;
    c10::optional<Tensor> bias_a;
    Tensor weight_b;
    c10::optional<Tensor> bias_b;
    std::vector<int64_t> filter;
    std::vector<int64_t> stride;
    std::vector<int64_t> padding;
    std::vector<int64_t> output_padding;
    std::vector<int64_t> dilation;
    int64_t groups;
    bool transposed;
  } unpacked_;
};

Tensor gated_conv2d_module_run(
    const Tensor& padding,
    const Tensor& prev_out,
    const c10::intrusive_ptr<GatedConv2dModuleOpContext>& context);

Tensor gated_conv2d_module_run_cpu(
    const Tensor& padding,
    const Tensor& prev_out,
    const c10::intrusive_ptr<GatedConv2dModuleOpContext>& context);

Tensor gated_conv_transpose2d_module_run(
    const Tensor& padding,
    const Tensor& prev_enc_out,
    const Tensor& prev_out,
    const Tensor& encoder_out,
    const c10::intrusive_ptr<GatedConv2dModuleOpContext>& context);

Tensor gated_conv_transpose2d_module_run_cpu(
    const Tensor& padding,
    const Tensor& prev_enc_out,
    const Tensor& prev_out,
    const Tensor& encoder_out,
    const c10::intrusive_ptr<GatedConv2dModuleOpContext>& context);

c10::intrusive_ptr<GatedConv2dModuleOpContext> gated_conv2d_module_prepack(
    Tensor&& weight_a,
    c10::optional<Tensor>&& bias_a,
    Tensor&& weight_b,
    c10::optional<Tensor>&& bias_b,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& output_padding,
    std::vector<int64_t>&& dilation,
    const int64_t groups,
    const bool transposed);

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
