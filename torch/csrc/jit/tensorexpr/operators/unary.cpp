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
        auto inp = promoteIntegerToDefaultType(inputs[0]);
        // v1 = (inp < 0.0f) ? -1.0f : 1.0f;
        auto v1 = CompareSelect::make(
            inp, ExprHandle(0.0f), ExprHandle(-1.0f), ExprHandle(1.0f), kLT);
        // v2 = (inp == 0.0f) ? 0.0f : v1
        auto v2 = CompareSelect::make(
            inp, ExprHandle(0.0f), ExprHandle(0.0f), v1, kEQ);
        // (isnan(inp) == 1) ? 0.0f : v2
        auto res = CompareSelect::make(
            isnan(inp), ExprHandle(1), ExprHandle(0.0f), v2, kEQ);
        // Final expression:
        //    (isnan(inp) == 1) ?
        //        0.0f :
        //        (inp == 0.0f) ?
        //            0.0f :
        //            (inp < 0.0f) ? -1.0f : 1.0f
        return res;
      });
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
