#include <torch/csrc/jit/tensorexpr/operators/unary.h>

namespace torch {
namespace jit {
namespace tensorexpr {

using namespace torch::jit::tensorexpr;

Tensor computeSign(
    const std::vector<ArgValue>& inputValues,
    const std::vector<ExprHandle>& outputShape) {
  return Compute(
      "aten_sign", c10::fmap<DimArg>(outputShape), [&](ParameterList& axes) {
        std::vector<ExprHandle> indices(axes.begin(), axes.end());
        std::vector<ExprHandle> inputs = {
            tensorOrConstant(inputValues[0], indices)};
        auto inp = inputs[0];
        auto zero = ExprHandle(immLike(inp, 0.0f));
        auto res = (zero < inp) - (inp < zero);
        return promoteToDtype(res, inp.dtype().scalar_type());
      });
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
