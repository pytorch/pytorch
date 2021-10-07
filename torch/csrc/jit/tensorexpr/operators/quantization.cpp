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
    vars.push_back(alloc<Var>("", os.node()->dtype().scalar_type() == ScalarType::Long ? kLong : kInt));
  }
  auto axes = VarVectorToVarHandleVector(vars);
  std::vector<ExprHandle> indices(axes.begin(), axes.end());

  auto qscale = constant(inputs[1]);
  auto qzero = constant(inputs[2]);
  //TODO: handle inputs[3] argument as dtype, asserts qint8, quint8
  auto dtype = Dtype(ScalarType::Byte);
  ExprHandle exprHandle = promoteToDtype((tensorOrConstant(inputs[0], indices) - qzero) / qscale, dtype.scalar_type());

  std::cout << "XXX " << __FUNCTION__ << std::endl;
  for (const auto& arg : inputs) {
    std::cout << "XXX " << getArgValueName(arg) << std::endl;
  }
  BufPtr buf = alloc<Buf>(
      "quantize_per_tensor",
      output_sizes_expr,
      dtype,
      nullptr,
      qscale.node(),
      qzero.node());
  return Tensor(buf, vars, exprHandle.node());
}

Tensor computeDequantize(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const c10::optional<ScalarType>& outputType) {
  std::cout << "XXX " << __FUNCTION__ << std::endl;
  for (const auto& arg : inputs) {
    std::cout << "XXX " << getArgValueName(arg) << std::endl;
  }
  auto qbuf = c10::get<BufHandle>(inputs[0]);
  auto qscale = qbuf.node()->qscale();
  auto qzero = qbuf.node()->qzero();
  auto dtype = Dtype(ScalarType::Float);
  std::cout << "XXX qscale:" << qscale << std::endl;
  std::cout << "XXX qzero:" << qzero << std::endl;
  std::vector<VarPtr> vars;
  for (const auto& os : outputShape) {
    vars.push_back(alloc<Var>("", os.node()->dtype().scalar_type() == ScalarType::Long ? kLong : kInt));
  }
  auto axes = VarVectorToVarHandleVector(vars);
  std::vector<ExprHandle> indices(axes.begin(), axes.end());
  ExprHandle exprHandle = promoteToDtype(tensorOrConstant(inputs[0], indices) * ExprHandle(qscale) + ExprHandle(qzero), dtype.scalar_type());
  auto output_sizes_expr = ExprHandleVectorToExprVector(outputShape);
  BufPtr buf = alloc<Buf>("dequantize", output_sizes_expr, dtype);
  return Tensor(buf, vars, exprHandle.node());
}
*/
} // namespace tensorexpr
} // namespace jit
} // namespace torch
