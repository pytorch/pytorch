#include <torch/csrc/jit/tensorexpr/operators/norm.h>

namespace torch {
namespace jit {
namespace tensorexpr {

Tensor* computeBatchNorm(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const c10::optional<ScalarType>& outputType) {
  bool hasWeight = true;
  bool hasBias = true;

  if (c10::get_if<ArgNone>(&inputs[1])) {
    hasWeight = false;
  }

  if (c10::get_if<ArgNone>(&inputs[2])) {
    hasBias = false;
  }

  return Compute(
      "aten_batch_norm",
      c10::fmap<DimArg>(outputShape),
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

        if (hasWeight) {
          exprInputs.push_back(tensorOrConstant(inputs[1], {c}));
        }
        if (hasBias) {
          exprInputs.push_back(tensorOrConstant(inputs[2], {c}));
        }
        promoteInputs(exprInputs);

        ExprHandle input = exprInputs[0];
        ExprHandle mean = exprInputs[1];
        ExprHandle var = exprInputs[2];
        ExprHandle eps = exprInputs[3];
        ExprHandle weight = FloatImm::make(1);
        ExprHandle bias = FloatImm::make(0);

        if (hasWeight) {
          weight = exprInputs[4];
        }
        // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
        if (hasBias) {
          bias = exprInputs[5];
        }

        // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
        auto inv_var = rsqrt(var + eps);
        auto alpha = inv_var * weight;
        auto beta = bias - mean * alpha;
        auto output = input * alpha + beta;
        return demoteOutput(output, outputType);
      });
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
