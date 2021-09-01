#include <torch/csrc/jit/tensorexpr/operators/quantization.h>

using namespace torch::jit::tensorexpr;

namespace torch {
namespace jit {
namespace tensorexpr {
/*
Tensor computeQuantizePerTensor(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const c10::optional<ScalarType>& outputType) {
  auto output_sizes_expr = ExprHandleVectorToExprVector(outputShape);
  std::vector<VarPtr> vars;
  for (const auto& os : outputShape) {
    vars->push_back(alloc<Var>(
        dim_arg.name_hint(),
        expr->dtype().scalar_type() == ScalarType::Long ? kLong : kInt));
  }
  auto axes = VarVectorToVarHandleVector(vars);
  std::vector<ExprHandle> indices(axes.begin(), axes.end());

  auto scale = constants(inputs[1]);
  auto zero = constants(inputs[2]);
  ExprHandle exprHandle = (tensorOrConstant(inputs[0], indices) - zero) / scale;

  std::cout << "XXX " << __FUNCTION__ << std::endl;
  for (const auto& arg : inputs) {
    std::cout << "XXX " << getArgValueName(arg) << std::endl;
  }
  auto dtype = DType(ScalarType::Byte);
  BufPtr buf = alloc<Buf>("quantize_per_tensor", output_sizes_expr, dtype);
  return Tensor(buf, vars, exprHandle.node());
}
*/
} // namespace tensorexpr
} // namespace jit
} // namespace torch
