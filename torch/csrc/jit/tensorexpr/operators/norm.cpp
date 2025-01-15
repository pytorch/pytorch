#include <torch/csrc/jit/tensorexpr/operators/misc.h>
#include <torch/csrc/jit/tensorexpr/operators/norm.h>

namespace torch::jit::tensorexpr {

Tensor computeBatchNorm(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const std::optional<ScalarType>& outputType,
    at::Device device) {
  bool hasWeight = true;
  bool hasBias = true;

  if (std::holds_alternative<ArgNone>(inputs[1])) {
    hasWeight = false;
  }

  if (std::holds_alternative<ArgNone>(inputs[2])) {
    hasBias = false;
  }

  return Compute(
      "aten_batch_norm",
      outputShape,
      outputStrides,
      [&](const std::vector<VarHandle>& axes) {
        TORCH_INTERNAL_ASSERT(axes.size() >= 2);
        // axes: N, C, H, W
        std::vector<ExprHandle> indices(axes.begin(), axes.end());
        ExprHandle c = indices[1];

        // Parameter list:
        // input, weight, bias, mean, var, training, momentum, eps,
        // cudnn_enabled
        std::vector<ExprHandle> exprInputs = {
            tensorOrConstant(inputs[0], indices), // input
            tensorOrConstant(inputs[3], {c}), // mean
            tensorOrConstant(inputs[4], {c}), // var
            constant(inputs[7]) // eps
        };

        ExprHandle weight = FloatImm::make(1);
        ExprHandle bias = FloatImm::make(0);
        if (hasWeight) {
          weight = tensorOrConstant(inputs[1], {c});
          exprInputs.push_back(weight);
        }
        if (hasBias) {
          bias = tensorOrConstant(inputs[2], {c});
          exprInputs.push_back(bias);
        }
        promoteInputs(exprInputs);

        ExprHandle input = exprInputs[0];
        ExprHandle mean = exprInputs[1];
        ExprHandle var = exprInputs[2];
        ExprHandle eps = exprInputs[3];

        auto inv_var = rsqrt(var + eps);
        auto alpha = inv_var * weight;
        auto beta = bias - mean * alpha;
        auto output = input * alpha + beta;
        return demoteOutput(output, outputType);
      });
}

} // namespace torch::jit::tensorexpr
