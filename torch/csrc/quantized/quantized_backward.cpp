#include <ATen/native/quantized/PackedParams.h>
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
    // TO-DO: passing packed_weight as saved_data requires more work in adding
    // LinearPackedParamsBase in ivalue For now, we can simply pass a weight
    // itself. Referenced :
    // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/core/ivalue.h
    auto unpacked_parameters = packed_weight->unpack();
    ctx->saved_data["weight"] = std::get<0>(unpacked_parameters);
    return output;
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto original_weight = ctx->saved_data["weight"].toTensor();
    original_weight = at::permute(original_weight, {1, 0});
    auto grad_output = grad_outputs[0];
    static auto op = at::Dispatcher::singleton()
                         .findSchemaOrThrow("quantized::linear_prepack", "")
                         .typed<c10::intrusive_ptr<LinearPackedParamsBase>(
                             at::Tensor, c10::optional<at::Tensor>)>();
    auto prepacked_weight = op.call(original_weight, nullopt);
    auto grad_input = prepacked_weight->apply_dynamic(grad_output);
    return {grad_input, torch::Tensor(), torch::Tensor()};
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
