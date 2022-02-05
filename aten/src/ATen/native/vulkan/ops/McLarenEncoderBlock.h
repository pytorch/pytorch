#pragma once

#ifdef USE_VULKAN_API

#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Convolution.h>
#include <torch/custom_class.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {

class McLarenEncoderBlockOpContext final : public torch::jit::CustomClassHolder {
 public:
  static McLarenEncoderBlockOpContext create(
      const Tensor& weight_1,
      const c10::optional<Tensor>& bias_1,
      IntArrayRef stride_1,
      IntArrayRef padding_1,
      IntArrayRef dilation_1,
      int64_t groups_1,
      const Tensor& weight_2,
      const c10::optional<Tensor>& bias_2,
      IntArrayRef stride_2,
      IntArrayRef padding_2,
      IntArrayRef dilation_2,
      int64_t groups_2);

  using State = std::tuple<
      Tensor,
      c10::optional<Tensor>,
      std::vector<int64_t>,
      std::vector<int64_t>,
      std::vector<int64_t>,
      int64_t,
      Tensor,
      c10::optional<Tensor>,
      std::vector<int64_t>,
      std::vector<int64_t>,
      std::vector<int64_t>,
      int64_t>;

  Tensor run(const Tensor& input_arg_1, const Tensor& input_arg_2) const;
  State unpack() const;

 private:
  McLarenEncoderBlockOpContext(
      const Tensor& weight_1,
      const c10::optional<Tensor>& bias_1,
      IntArrayRef stride_1,
      IntArrayRef padding_1,
      IntArrayRef dilation_1,
      int64_t groups_1,
      const Tensor& weight_2,
      const c10::optional<Tensor>& bias_2,
      IntArrayRef stride_2,
      IntArrayRef padding_2,
      IntArrayRef dilation_2,
      int64_t groups_2);

  void conv2d_sliding_window(
      vTensor& v_output,
      const vTensor& v_input,
      const vTensor& v_weight,
      const vTensor& v_bias,
      std::array<int64_t, 4> filter,
      std::vector<int64_t> orig_filter,
      std::array<int64_t, 2> stride,
      std::array<int64_t, 2> padding,
      std::array<int64_t, 2> dilation) const;

 private:
  struct {
    vTensor v_weight_1;
    vTensor v_bias_1;
    std::array<int64_t, 4> filter_1;
    std::array<int64_t, 2> stride_1;
    std::array<int64_t, 2> padding_1;
    std::array<int64_t, 2> dilation_1;
    int32_t groups_1;
    vTensor v_weight_2;
    vTensor v_bias_2;
    std::array<int64_t, 4> filter_2;
    std::array<int64_t, 2> stride_2;
    std::array<int64_t, 2> padding_2;
    std::array<int64_t, 2> dilation_2;
    int32_t groups_2;
  } packed_;

  struct {
    Tensor weight_1;
    c10::optional<Tensor> bias_1;
    std::vector<int64_t> filter_1;
    std::vector<int64_t> stride_1;
    std::vector<int64_t> padding_1;
    std::vector<int64_t> dilation_1;
    int64_t groups_1;
    Tensor weight_2;
    c10::optional<Tensor> bias_2;
    std::vector<int64_t> filter_2;
    std::vector<int64_t> stride_2;
    std::vector<int64_t> padding_2;
    std::vector<int64_t> dilation_2;
    int64_t groups_2;
  } unpacked_;
};

Tensor mclaren_encoder_block_run(
    const Tensor& input_1,
    const Tensor& input_2,
    const c10::intrusive_ptr<McLarenEncoderBlockOpContext>& context);

c10::intrusive_ptr<McLarenEncoderBlockOpContext> mclaren_encoder_block_prepack(
    Tensor&& weight_1,
    c10::optional<Tensor>&& bias_1,
    std::vector<int64_t>&& stride_1,
    std::vector<int64_t>&& padding_1,
    std::vector<int64_t>&& dilation_1,
    const int64_t groups_1,
    Tensor&& weight_2,
    c10::optional<Tensor>&& bias_2,
    std::vector<int64_t>&& stride_2,
    std::vector<int64_t>&& padding_2,
    std::vector<int64_t>&& dilation_2,
    const int64_t groups_2);

} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

#endif /* USE_VULKAN_API */
