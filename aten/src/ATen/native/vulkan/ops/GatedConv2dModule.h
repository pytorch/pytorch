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
      const Tensor& weight_1,
      const c10::optional<Tensor>& bias_1,
      IntArrayRef stride_1,
      IntArrayRef padding_1,
      IntArrayRef output_padding_1,
      IntArrayRef dilation_1,
      const int64_t groups_1,
      const Tensor& weight_2,
      const c10::optional<Tensor>& bias_2,
      IntArrayRef stride_2,
      IntArrayRef padding_2,
      IntArrayRef output_padding_2,
      IntArrayRef dilation_2,
      const int64_t groups_2,
      const bool transposed);

  using State = std::tuple<
      Tensor,
      c10::optional<Tensor>,
      std::vector<int64_t>,
      std::vector<int64_t>,
      std::vector<int64_t>,
      std::vector<int64_t>,
      int64_t,
      Tensor,
      c10::optional<Tensor>,
      std::vector<int64_t>,
      std::vector<int64_t>,
      std::vector<int64_t>,
      std::vector<int64_t>,
      int64_t,
      bool>;

  Tensor run(const Tensor& input_arg_1, const Tensor& input_arg_2) const;
  State unpack() const;

 private:
  GatedConv2dModuleOpContext(
      const Tensor& weight_1,
      const c10::optional<Tensor>& bias_1,
      IntArrayRef stride_1,
      IntArrayRef padding_1,
      IntArrayRef output_padding_1,
      IntArrayRef dilation_1,
      int64_t groups_1,
      const Tensor& weight_2,
      const c10::optional<Tensor>& bias_2,
      IntArrayRef stride_2,
      IntArrayRef padding_2,
      IntArrayRef output_padding_2,
      IntArrayRef dilation_2,
      int64_t groups_2,
      bool transposed);

 private:
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
    Tensor weight_1;
    c10::optional<Tensor> bias_1;
    std::vector<int64_t> filter_1;
    std::vector<int64_t> stride_1;
    std::vector<int64_t> padding_1;
    std::vector<int64_t> output_padding_1;
    std::vector<int64_t> dilation_1;
    int64_t groups_1;
    Tensor weight_2;
    c10::optional<Tensor> bias_2;
    std::vector<int64_t> filter_2;
    std::vector<int64_t> stride_2;
    std::vector<int64_t> padding_2;
    std::vector<int64_t> output_padding_2;
    std::vector<int64_t> dilation_2;
    int64_t groups_2;
    bool transposed;
  } unpacked_;
};

Tensor gated_conv2d_module_run(
    const Tensor& input_1,
    const Tensor& input_2,
    const c10::intrusive_ptr<GatedConv2dModuleOpContext>& context);

c10::intrusive_ptr<GatedConv2dModuleOpContext> gated_conv2d_module_prepack(
    Tensor&& weight_1,
    c10::optional<Tensor>&& bias_1,
    std::vector<int64_t>&& stride_1,
    std::vector<int64_t>&& padding_1,
    std::vector<int64_t>&& output_padding_1,
    std::vector<int64_t>&& dilation_1,
    const int64_t groups_1,
    Tensor&& weight_2,
    c10::optional<Tensor>&& bias_2,
    std::vector<int64_t>&& stride_2,
    std::vector<int64_t>&& padding_2,
    std::vector<int64_t>&& output_padding_2,
    std::vector<int64_t>&& dilation_2,
    const int64_t groups_2,
    const bool transposed);

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
