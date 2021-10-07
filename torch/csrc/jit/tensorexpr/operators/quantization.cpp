#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/operators/quantization.h>

using namespace torch::jit::tensorexpr;

namespace torch {
namespace jit {
namespace tensorexpr {
namespace {
std::vector<int64_t> _pair_int(ArgValue v) {
  if (auto t = c10::get_if<IntList>(&v)) {
    return {(*t)[0], (*t)[1]};
  }
  auto i = c10::get<int64_t>(v);
  return {i, i};
}
} // namespace

ExprHandle quantizePerTensorQParamFromArg(ArgValue arg) {
  if (auto b = c10::get_if<BufHandle>(&arg)) {
    return b->load({LongImm::make(0l)});
  }
  return constant(arg);
}

double immQScale(const BufHandle& qx) {
  return to<DoubleImm>(IRSimplifier::simplify(qx.node()->qscale()))->value();
}

int64_t immQZero(const BufHandle& qx) {
  return to<LongImm>(IRSimplifier::simplify(qx.node()->qzero()))->value();
}

int64_t immQDType(const BufHandle& qx) {
  return (int64_t)qx.dtype().scalar_type();
}

double isQuantized(const BufHandle& qx) {
  return qx.node()->qscale() && qx.node()->qzero();
}

BufHandle makeQBufHandle(
    const std::string& name_hint,
    const std::vector<ExprHandle>& dims,
    Dtype dtype,
    const ExprPtr qscale,
    const ExprPtr qzero) {
  BufHandle ResultBuf("quantized_conv2d", dims, dtype);
  ResultBuf.node()->set_qscale(qscale);
  ResultBuf.node()->set_qzero(qzero);
  return ResultBuf;
}

BufHandle makeQBufHandle(
    const std::string& name_hint,
    const std::vector<ExprHandle>& dims,
    Dtype dtype,
    const double qscale,
    const int64_t qzero) {
  return makeQBufHandle(
      name_hint,
      dims,
      dtype,
      DoubleImm::make(qscale).node(),
      LongImm::make(qzero).node());
}

Tensor computeQuantizePerTensor(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const c10::optional<ScalarType>& outputType) {
  std::vector<VarPtr> vars;
  for (const auto& os : outputShape) {
    vars.push_back(alloc<Var>(
        "",
        os.node()->dtype().scalar_type() == ScalarType::Long ? kLong : kInt));
  }
  auto axes = VarVectorToVarHandleVector(vars);
  std::vector<ExprHandle> indices(axes.begin(), axes.end());

  ExprHandle qscale = quantizePerTensorQParamFromArg(inputs[1]);
  ExprHandle qzero = quantizePerTensorQParamFromArg(inputs[2]);
  const auto dtype = [](auto qdtype) {
    if (static_cast<int64_t>(ScalarType::QInt8) == qdtype) {
      return Dtype(ScalarType::Char);
    } else if (static_cast<int64_t>(ScalarType::QUInt8) == qdtype) {
      return Dtype(ScalarType::Byte);
    }
    throw malformed_input("Unsupported quantized dtype");
  }(c10::get<int64_t>(inputs[3]));
  const BufHandle& x = c10::get<BufHandle>(inputs[0]);

  auto x_dtype = x.node()->dtype();
  auto promoted_qscale = promoteToDtype(qscale, x_dtype.scalar_type());
  auto promoted_qzero = promoteToDtype(qzero, x_dtype.scalar_type());
  ExprHandle exprHandle = promoteToDtype(
      tensorOrConstant(inputs[0], indices) / promoted_qscale + promoted_qzero +
          FloatImm::make(0.5f),
      dtype.scalar_type());

  BufPtr buf = alloc<Buf>(
      "quantize_per_tensor",
      ExprHandleVectorToExprVector(outputShape),
      dtype,
      nullptr,
      qscale.node(),
      qzero.node());
  return Tensor(buf, vars, exprHandle.node());
}

Tensor computeQuantizedConv2dPrepack(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const c10::optional<ScalarType>& outputType) {
  Dtype dtype = kFloat;
  if (outputType) {
    dtype = Dtype(*outputType);
  }

  BufHandle ResultBuf("quantized_conv2d_prepack", outputShape, dtype);
  const BufHandle& qw = c10::get<BufHandle>(inputs[0]);
  const BufHandle& b = c10::get<BufHandle>(inputs[1]);
  auto strides = _pair_int(inputs[2]);
  auto padding = _pair_int(inputs[3]);
  auto dilation = _pair_int(inputs[4]);
  int groups = c10::get<int64_t>(inputs[5]);
  TORCH_INTERNAL_ASSERT(
      qw.node()->qscale(),
      buildErrorMessage(
          "quantized_conv2d_prepack: Expects quantized weights, qscale is missing"));
  TORCH_INTERNAL_ASSERT(
      qw.node()->qzero(),
      buildErrorMessage(
          "quantized_conv2d_prepack: Expects quantized weights, qzero is missing"));
  StmtPtr s = ExternalCall::make(
      ResultBuf,
      "nnc_quantized_conv2d_prepack",
      {qw, b},
      {strides[0],
       strides[1],
       padding[0],
       padding[1],
       dilation[0],
       dilation[1],
       groups,
       immQScale(qw),
       immQZero(qw),
       immQDType(qw)});
  return Tensor(ResultBuf.node(), s);
}

Tensor computeQuantizedConv2d(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const c10::optional<ScalarType>& outputType) {
  Dtype dtype = kFloat;
  if (outputType) {
    dtype = Dtype(*outputType);
  }

  const BufHandle& qx = c10::get<BufHandle>(inputs[0]);
  const BufHandle& prepacked = c10::get<BufHandle>(inputs[1]);
  const auto out_qscale = c10::get<double>(inputs[2]);
  const auto out_qzero = c10::get<int64_t>(inputs[3]);
  auto ResultBuf = makeQBufHandle(
      "quantized_conv2d", outputShape, dtype, out_qscale, out_qzero);
  StmtPtr s = ExternalCall::make(
      ResultBuf,
      "nnc_quantized_conv2d",
      {qx, prepacked},
      {immQScale(qx), immQZero(qx), immQDType(qx), out_qscale, out_qzero});
  return Tensor(ResultBuf.node(), s);
}

Tensor computeQuantizedConv2dRelu(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const c10::optional<ScalarType>& outputType) {
  Dtype dtype = kFloat;
  if (outputType) {
    dtype = Dtype(*outputType);
  }

  const BufHandle& qx = c10::get<BufHandle>(inputs[0]);
  const BufHandle& prepacked = c10::get<BufHandle>(inputs[1]);
  const auto out_qscale = c10::get<double>(inputs[2]);
  const auto out_qzero = c10::get<int64_t>(inputs[3]);
  auto ResultBuf = makeQBufHandle(
      "quantized_conv2d_relu", outputShape, dtype, out_qscale, out_qzero);
  StmtPtr s = ExternalCall::make(
      ResultBuf,
      "nnc_quantized_conv2d_relu",
      {qx, prepacked},
      {immQScale(qx), immQZero(qx), immQDType(qx), out_qscale, out_qzero});
  return Tensor(ResultBuf.node(), s);
}

Tensor computeQuantizedAdd(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const c10::optional<ScalarType>& outputType) {
  Dtype dtype = kFloat;
  if (outputType) {
    dtype = Dtype(*outputType);
  }

  const BufHandle& qa = c10::get<BufHandle>(inputs[0]);
  const BufHandle& qb = c10::get<BufHandle>(inputs[1]);
  const auto out_qscale = c10::get<double>(inputs[2]);
  const auto out_qzero = c10::get<int64_t>(inputs[3]);
  auto ResultBuf = makeQBufHandle(
      "quantized_add", outputShape, dtype, out_qscale, out_qzero);
  StmtPtr s = ExternalCall::make(
      ResultBuf,
      "nnc_quantized_add",
      {qa, qb},
      {immQScale(qa),
       immQZero(qa),
       immQDType(qa),
       immQScale(qb),
       immQZero(qb),
       immQDType(qb),
       out_qscale,
       out_qzero});
  return Tensor(ResultBuf.node(), s);
}

Tensor computeDequantize(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const c10::optional<ScalarType>& outputType) {
  Dtype dtype = kFloat;
  if (outputType) {
    dtype = Dtype(*outputType);
  }
  auto qx = c10::get<BufHandle>(inputs[0]);
  auto qscale = qx.node()->qscale();
  auto qzero = qx.node()->qzero();
  auto qdtype = qx.node()->dtype();
  TORCH_INTERNAL_ASSERT(
      qscale, buildErrorMessage("Missing quantized scale for dequantize"));
  TORCH_INTERNAL_ASSERT(
      qzero, buildErrorMessage("Missing quantized zero point for dequantize"));
  std::vector<VarPtr> vars;
  for (const auto& os : outputShape) {
    vars.push_back(alloc<Var>(
        "",
        os.node()->dtype().scalar_type() == ScalarType::Long ? kLong : kInt));
  }
  auto axes = VarVectorToVarHandleVector(vars);
  std::vector<ExprHandle> indices(axes.begin(), axes.end());
  auto qx_e_promoted =
      promoteToDtype(tensorOrConstant(inputs[0], indices), dtype.scalar_type());
  auto y = promoteToDtype(
      (qx_e_promoted - ExprHandle(qzero)) * ExprHandle(qscale),
      dtype.scalar_type());

  BufPtr buf = alloc<Buf>(
      "dequantize", ExprHandleVectorToExprVector(outputShape), dtype);
  return Tensor(buf, vars, y.node());
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
