
#include <ATen/ATen.h>
#include <ATen/native/quantized/cpu/qnnpack_utils.h>
#include <caffe2/utils/threadpool/ThreadPoolMobile.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace {

class QDeConv2dInt8 final {
 public:
  static Tensor run(Tensor act,
                    Tensor packed_weight,
                    torch::List<int64_t> stride,
                    torch::List<int64_t> input_padding,
                    torch::List<int64_t> output_padding,
                    torch::List<int64_t> dilation,
                    int64_t groups,
                    double output_scale,
                    int64_t output_zero_point) {
    auto& ctx = at::globalContext();
#ifdef USE_PYTORCH_QNNPACK
    if (ctx.qEngine() == at::QEngine::QNNPACK) {
      return QnnpackDeconv(act,
                           packed_weight,
                           stride,
                           input_padding,
                           output_padding,
                           dilation,
                           groups,
                           output_scale,
                           output_zero_point);
    }
#else // Fallback (no engine)
    return FallBackDeconv(act,
                          packed_weight,
                          stride,
                          input_padding,
                          output_padding,
                          dilation,
                          groups,
                          output_scale,
                          output_zero_point);
#endif // USE_PYTORCH_QNNPACK

    TORCH_CHECK(false, "Didn't find engine for operation quantized::conv ",
                toString(ctx.qEngine()));
  }

 private:
#ifdef USE_PYTORCH_QNNPACK
  static at::Tensor QnnpackDeconv(Tensor act,
                       Tensor packed_weight,
                       torch::List<int64_t> stride,
                       torch::List<int64_t> input_padding,
                       torch::List<int64_t> output_padding,
                       torch::List<int64_t> dilation,
                       int64_t groups,
                       double output_scale,
                       int64_t output_zero_point) {
    TORCH_CHECK(false, "Not implemented yet!");
  }
#else // Fallback (no engine)
  static at::Tensor FallBackDeconv(Tensor act,
                        Tensor packed_weight,
                        torch::List<int64_t> stride,
                        torch::List<int64_t> input_padding,
                        torch::List<int64_t> output_padding,
                        torch::List<int64_t> dilation,
                        int64_t groups,
                        double output_scale,
                        int64_t output_zero_point) {
    const auto kUnpackFunctionName = "quantized::deconv2d_unpack";
    const std::vector<c10::IValue> unpacked_weight_list
      = callOp(kUnpackFunctionName, "", packed_weight)
    TORCH_INTERNAL_ASSERT(unpacked_weight_list.size() == 2,
      "The unpacked weight list should have exactly two elements.");
    const Tensor weight = unpacked_weight_list[0].toTensor();
    const c10::optional<Tensor>& bias = unpacked_weight_list[1].toOptional<Tensor>();
    const Tensor f_act = act.dequantize();
    const Tensor f_out = at::conv2d(f_act,
                                    weight,
                                    bias,
                                    stride,
                                    input_padding,
                                    output_padding,
                                    groups,
                                    dilation);
    if (act.qscheme() == kPerTensorAffine) {
      return at::quantize_per_tensor(f_out, output_scale, output_zero_point, at::kQInt8);
    } else {
      TORCH_CHECK(false, "Only per tensor quantization is supported.");
    }
  }
#endif // USE_PYTORCH_QNNPACK
};

TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  m.impl("conv2d_transpose", QDeConv2dInt8::run);
}

TORCH_LIBRARY_IMPL(_quantized, QuantizedCPU, m) {
  m.impl("conv2d_transpose", QDeConv2dInt8::run);
}

}  // namespace
}  // namespace native
}  // namespace at
