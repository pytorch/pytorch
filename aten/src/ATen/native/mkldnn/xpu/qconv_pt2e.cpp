#include <c10/core/MemoryFormat.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/mkldnn/xpu/detail/oneDNN.h>
#include <torch/library.h>

#include <iostream>

using namespace at::native::onednn;
namespace at {
namespace native{
namespace xpu {

at::Tensor qconv_prepack_xpu(
    at::Tensor weight,
    at::Tensor weight_scales,
    double input_scale,
    int64_t input_zero_point,
    torch::List<int64_t> stride,
    torch::List<int64_t> padding,
    torch::List<int64_t> dilation,
    int64_t groups,
    std::optional<torch::List<int64_t>> input_shape
){
    // XPU has no prepack at present
    return weight;
}


class QConvoneDNNXPU final {
 public:
  static at::Tensor run_pointwise(
      at::Tensor act,
      double act_scale,
      int64_t act_zero_point,
      at::Tensor weight,
      at::Tensor weight_scales,
      at::Tensor weight_zero_points,
      c10::optional<at::Tensor> bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> dilation,
      int64_t groups,
      double inv_output_scale,
      int64_t output_zero_point,
      c10::optional<c10::ScalarType> output_dtype,
      c10::string_view attr,
      torch::List<c10::optional<at::Scalar>> scalars,
      c10::optional<c10::string_view> algorithm) {
    if (act.dim() == 3 || act.dim() == 5) {
      TORCH_CHECK(
          attr == "none",
          "quantized pointwise conv",
          act.dim() - 2,
          "d doesn't support unary_post_op fusion. Got unary_post_op:",
          attr,
          ".");
    } else {
      TORCH_CHECK(
          attr == "none" || attr == "relu" || attr == "hardtanh" ||
              attr == "hardswish" || attr=="swish",
          "none post_op or post_op relu/hardtanh/hardswish is supported for quantized pointwise conv2d. Got unary_post_op: ",
          attr,
          ".");
    }

    bool is_channels_last_suggested= use_channels_last_for_conv(act, weight);
    auto mfmt = is_channels_last_suggested ? get_cl_tag_by_ndim(act.ndimension()) : at::MemoryFormat::Contiguous;
    Tensor input_ = act.contiguous(mfmt);
    Tensor weight_ = weight.contiguous(mfmt);

    auto dst_tz = conv_dst_size(
        input_.ndimension(),
        input_.sizes(),
        weight_.sizes(),
        padding.vec(),
        padding.vec(),
        stride.vec(),
        dilation.vec());

    // TODO: handle difference of this dtype with argument dtype
    // auto dtype =
    //     (act.scalar_type() == c10::ScalarType::Byte) ? c10::kByte : c10::kChar;
    Tensor output = at::empty(
        dst_tz, device(c10::kXPU).dtype(output_dtype).memory_format(mfmt));
    
    return quantized_convolution_pt2(
        act,
        act_scale,
        act_zero_point,
        weight,
        weight_scales,
        weight_zero_points,
        bias,
        stride,
        padding,
        dilation,
        /*transposed*/ false,
        groups,
        output,
        inv_output_scale,
        output_zero_point,
        /*accum*/ c10::nullopt,
        /*accum_scale*/ 0.0,
        /*accum_zero_point*/ 0,
        /*output_dtype*/ output_dtype,
        /*binary_attr*/ c10::nullopt,
        /*binary_alpha*/ c10::nullopt,
        /*unary_attr*/ attr,
        /*unary_scalars*/ scalars,
        /*unary_algorithm*/ algorithm);
  }
};



TORCH_LIBRARY_IMPL(onednn, XPU, m){
    m.impl(TORCH_SELECTIVE_NAME("onednn::qconv_prepack"), TORCH_FN(xpu::qconv_prepack_xpu));
    m.impl(TORCH_SELECTIVE_NAME("onednn::qconv1d_pointwise"), QConvoneDNNXPU::run_pointwise);
    m.impl(TORCH_SELECTIVE_NAME("onednn::qconv2d_pointwise"), QConvoneDNNXPU::run_pointwise);
    m.impl(TORCH_SELECTIVE_NAME("onednn::qconv3d_pointwise"), QConvoneDNNXPU::run_pointwise);
}

}
}
}