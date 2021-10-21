#include <ATen/ATen.h>
#include <ATen/cpp_custom_type_hack.h>
#include <ATen/native/quantized/cpu/packed_params.h>
#include <ATen/native/mkldnn/quantization/Utils.h>
#include <ATen/native/quantized/cpu/quant_utils.h>
#include <ATen/quantized/Quantizer.h>
#include <torch/custom_class.h>
#include <torch/library.h>

#include <c10/util/irange.h>

#include <algorithm>
#include <vector>

torch::class_<LinearPackedParamsBase> register_linear_params();

#if AT_MKLDNN_ENABLED()
c10::intrusive_ptr<LinearPackedParamsBase> PackedLinearWeightsMkldnn::prepack(
    at::Tensor weight,
    c10::optional<at::Tensor> bias) {
  TORCH_CHECK(
      weight.dim() == 2,
      "The weight tensor for quantized::linear_prepack (mkldnn) should"
      " be 2-dimensional.");
  // Weight
  std::vector<int64_t> dims = weight.sizes().vec();
  auto N = weight.size(0);
  std::vector<int32_t> wgt_zero_points;
  ideep::scale_t wgt_scales;
  const auto qtype = weight.qscheme();
  if (qtype == c10::kPerTensorAffine) {
    TORCH_CHECK(
        weight.q_zero_point() == 0,
        "quantized::linear_prepack: MKLDNN only supports symmetric quantization of weight,"
        " whose zero point must be 0, but got ", weight.q_zero_point());
    wgt_zero_points = std::vector<int32_t>(1, weight.q_zero_point());
    wgt_scales = ideep::scale_t(1, 1.0/weight.q_scale()); // Scales of MKLDNN and PyTorch are reciprocal
  } else if (qtype == c10::kPerChannelAffine) {
    wgt_zero_points.resize(N);
    wgt_scales.resize(N);
    for (int i = 0; i < N; ++i) {
      wgt_zero_points[i] = weight.q_per_channel_zero_points()[i].item<int32_t>();
      TORCH_CHECK(
          wgt_zero_points[i] == 0,
          "quantized::linear_prepack: MKLDNN only supports symmetric quantization of weight,"
          " whose zero point must be 0, but got ",  wgt_zero_points[i], ", at index ", i);
      wgt_scales[i] = 1.0f / weight.q_per_channel_scales()[i].item<float>(); // Scales of MKLDNN and PyTorch are reciprocal
    }
  } else {
    TORCH_CHECK(false, "Unsupported qscheme: ", toString(qtype));
  }

  // Prepack weight
  auto w_desc = ideep::matmul_forward::expected_weights_desc(dims, dnnl::memory::data_type::s8,
                                                             dnnl::memory::data_type::u8);
  auto weight_copy = weight.clone();
  ideep::tensor wgt = ideep::tensor({dims, dnnl::memory::data_type::s8}, weight_copy.data_ptr());
  ideep::tensor exp_wgt;
  exp_wgt.init(w_desc);
  exp_wgt.feed_from(wgt);
  exp_wgt.transpose_(0, 1); // MKLDNN requires transposed weight
  ideep::tensor * packed_weight_p = new ideep::tensor(exp_wgt);
  packed_weight_p->set_scale(wgt_scales);
  packed_weight_p->set_zero_point(wgt_zero_points);
  std::unique_ptr<ideep::tensor> weight_ptr(packed_weight_p);
  // Bias
  c10::optional<ideep::tensor> mkldnn_bias{c10::nullopt};
  if (bias.has_value()) {
    auto bias_size = bias.value().sizes().vec();
    bias_size.insert(bias_size.begin(), 1);
    auto bias_desc = ideep::tensor::desc(bias_size, dnnl::memory::data_type::f32);
    ideep::tensor packed_bias;
    packed_bias.init(bias_desc, bias.value().data_ptr());
    mkldnn_bias = c10::optional<ideep::tensor>(packed_bias);
  }
  auto ret_ptr = c10::make_intrusive<PackedLinearWeightsMkldnn>(
      PackedLinearWeightsMkldnn{
        std::move(weight_ptr),
        mkldnn_bias,
        weight,
        bias});
  return ret_ptr;
}
#endif // #if AT_MKLDNN_ENABLED()

namespace at {
namespace native {

namespace {

class QLinearPackWeightInt8Mkldnn final {
 public:
  static c10::intrusive_ptr<LinearPackedParamsBase> run(
      at::Tensor weight,
      c10::optional<Tensor> bias) {
    auto& ctx = at::globalContext();

#if AT_MKLDNN_ENABLED()
    return PackedLinearWeightsMkldnn::prepack(std::move(weight), std::move(bias));
#endif // #if AT_MKLDNN_ENABLED()
    TORCH_CHECK(
        false,
        "Didn't find engine for operation quantized::linear_prepack ",
        toString(ctx.qEngine()));
  }
};

TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::linear_prepack_mkldnn"), TORCH_FN(QLinearPackWeightInt8Mkldnn::run));
}

} // namespace
} // namespace native
} // namespace at
