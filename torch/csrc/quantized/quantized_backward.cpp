#include <ATen/native/quantized/PackedParams.h>
#include <ATen/native/quantized/cpu/QuantUtils.h>
#include <torch/library.h>
#include <torch/torch.h>

namespace {
using namespace torch::autograd;
using namespace at;
// This class is a custom gradient function that enables quantized tensor to
// pass input gradient back to the previous layers This function can be used
// when the user is adapting mixed precision for traninig after quantization
// From torch layer, we have no access to linear_dynamic operator which needs to
// access via redispatching mechanism TO-DO : currently we are supporting per
// tensor quantization only, will expand to per channel later on
class PackedLinearWeightDynamicBackward
    : public Function<PackedLinearWeightDynamicBackward> {
 public:
  static torch::Tensor forward(
      AutogradContext* ctx,
      at::Tensor input,
      const c10::intrusive_ptr<LinearPackedParamsBase>& packed_weight,
      bool reduce_range) {
    static auto op =
        at::Dispatcher::singleton()
            .findSchemaOrThrow("quantized::linear_dynamic", "")
            .typed<at::Tensor(
                at::Tensor,
                c10::intrusive_ptr<
                    LinearPackedParamsBase,
                    c10::detail::intrusive_target_default_null_type<
                        LinearPackedParamsBase>> const&,
                bool)>();
    auto output = op.redispatch(
        DispatchKeySet({DispatchKey::CPU}), input, packed_weight, reduce_range);
    auto input_contig = input.contiguous();
    // Calculate statistics for quantization of input Tensor
    float x_min = 0;
    float x_max = 0;
    if (input.numel() > 0) {
      x_min = input_contig.min().item<float>();
      x_max = input_contig.max().item<float>();
    }
    auto q_params = quant_utils::ChooseQuantizationParams(
        /*min=*/x_min,
        /*max=*/x_max,
        /*qmin=*/0,
        /*qmax=*/255);
    ctx->saved_data["weight"] = packed_weight;
    // q_params.scale : shape [1] (per-tensor)
    ctx->saved_data["input_scale"] = q_params.scale;
    return output;
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    if (grad_outputs.empty()) {
      return {torch::Tensor(), torch::Tensor(), torch::Tensor()};
    }
    auto packed_weight =
        ctx->saved_data["weight"].toCustomClass<LinearPackedParamsBase>();
    auto unpacked_parameters = packed_weight->unpack();
    auto original_weight = std::get<0>(unpacked_parameters);
    auto input_scale = ctx->saved_data["input_scale"].toDouble();

    // Gradient for post-scaling
    // Let us rewrite this layer by separating the matmul from the output
    // scaling: y = (x * s1) @ w * s2 + b So you now back-propagate through four
    // operations: + b, * s2, @ W, and * s1. The steps are: start with the
    // gradient from the top, aka the adjoint, which is grad_outputs[0].
    // gradient for  + b: this is a no-op.
    // gradient for * s2: scale by s2. That's the affine/per-channel scale baked
    // into W. gradient for @ W: matmul with W.t. gradient for * s1: scale by
    // s1.
    auto grad_output0 = grad_outputs[0];
    const auto qtype = original_weight.qscheme();
    if (qtype == at::kPerTensorAffine) {
      grad_output0 *= original_weight.q_scale();
      original_weight = at::permute(original_weight, {1, 0});
    } else if (qtype == at::kPerChannelAffine) {
      // Per Channel quantizer does not support transpose.
      // Manual transpose is necessary
      original_weight = original_weight.dequantize();

// kwanghoon(TODO): This is going to be a long term solution that is applicable
// to every models One issue with quantizing a gradient, we can't get good
// enough gradient to improve model accuracy when model become complicated As of
// now, we can disable, and comeback when we figure it out better solution.
#if 0
      // Enable Kernel backend for quantized backpropagaiton matrix
      // multiplication
      original_weight = at::permute(original_weight, {1, 0});
      // Take advantage of QNNPACK for matrix multiplication
      // Per channel scales & zero point computation
      // Sources :
      // https://github.com/pytorch/pytorch/blob/master/torch/ao/quantization/observer.py#L350-L353
      auto [amin, amax] = at::aminmax(original_weight, /*dim* = */ 1);
      // QInt8 type signed quantization
      auto qmax = 127;
      auto qmin = -128;
      // Clamp with some epsilon number, so that value does not go below zero
      auto epsilon = 1e-9;
      auto new_scales = (amax - amin) / float(qmax - qmin);
      new_scales = at::clamp(new_scales, epsilon);
      auto new_zero_point =
          qmin - at::round(amin / new_scales).toType(c10::kInt);
      new_zero_point = at::clamp(new_zero_point, qmin, qmax);
      // TO-DO (BUGBUG)
      // Backend kernel is designed for inference, tightly coded for output
      // channel. For mathematical correctness, we should enable to run kernel
      // with input channel axis after transpose. As workaround, we are simply
      // either exploring per tensor quantization or per channel quantization
      // with axis = 0
      original_weight = at::quantize_per_channel(
          original_weight,
          new_scales,
          new_zero_point,
          /*axis = 1 for transpose, but we are forcing it to non-transposed case
             due to above issue*/
          0,
          c10::kQInt8);
#endif
    } else {
      TORCH_INTERNAL_ASSERT(false, "Unsupported quantization scheme.");
    }
#if 1
    // Pure FP32 computation, useful for debugging purpose
    auto dLdX1 = torch::matmul(grad_output0, original_weight);
#else
    // Take advantage of QNNPACK for matrix multiplication
    static auto op = at::Dispatcher::singleton()
                         .findSchemaOrThrow("quantized::linear_prepack", "")
                         .typed<c10::intrusive_ptr<LinearPackedParamsBase>(
                             at::Tensor, c10::optional<at::Tensor>)>();
    auto prepacked_weight = op.call(original_weight, nullopt);

    auto dLdX1 =
        prepacked_weight->apply_dynamic(grad_output0.toType(c10::kFloat));
#endif

    auto input_grad0 = dLdX1 * input_scale;
    return {input_grad0, torch::Tensor(), torch::Tensor()};
  }
};

at::Tensor packed_linear_weight_grad(
    c10::DispatchKeySet ks,
    at::Tensor input,
    const c10::intrusive_ptr<LinearPackedParamsBase>& packed_weight,
    bool reduce_range) {
  return PackedLinearWeightDynamicBackward::apply(
      input, packed_weight, reduce_range);
}
} // namespace

namespace at {
namespace native {
namespace {
TORCH_LIBRARY_IMPL(quantized, Autograd, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::linear_dynamic"),
      TORCH_FN(packed_linear_weight_grad));
}
} // namespace
} // namespace native
} // namespace at
