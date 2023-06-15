#include <c10/core/ScalarType.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/operators/misc.h>
#include <torch/csrc/jit/tensorexpr/operators/pointwise.h>
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

double immQScale(const BufHandle& qx) {
  TORCH_INTERNAL_ASSERT(
      qx.node()->qscale(), buildErrorMessage("Expects BufHandle with qscale"));
  return to<DoubleImm>(IRSimplifier::simplify(qx.node()->qscale()))->value();
}

int64_t immQZero(const BufHandle& qx) {
  TORCH_INTERNAL_ASSERT(
      qx.node()->qzero(), buildErrorMessage("Expects BufHandle with qzero"));
  return to<LongImm>(IRSimplifier::simplify(qx.node()->qzero()))->value();
}

ScalarType immQDType(const BufHandle& qx) {
  return qx.dtype().scalar_type();
}

bool isQuantized(const BufHandle& qx) {
  return qx.node()->qscale() && qx.node()->qzero();
}

static BufHandle makeQBufHandleChannelsLast(
    const std::string& name,
    const std::vector<ExprHandle>& dims,
    Dtype dtype,
    const ExprPtr qscale,
    const ExprPtr qzero) {
  BufHandle ResultBuf(name, dims, dtype);
  ResultBuf.node()->set_qscale(qscale);
  ResultBuf.node()->set_qzero(qzero);
  ResultBuf.node()->set_strides(make_channels_last_strides(dims));
  return ResultBuf;
}

static BufHandle makeQBufHandleChannelsLast(
    const std::string& name,
    const std::vector<ExprHandle>& dims,
    Dtype dtype,
    const double qscale,
    const int64_t qzero) {
  return makeQBufHandleChannelsLast(
      name,
      dims,
      dtype,
      DoubleImm::make(qscale).node(),
      LongImm::make(qzero).node());
}

static BufHandle makeQBufHandleContiguous(
    const std::string& name,
    const std::vector<ExprHandle>& dims,
    Dtype dtype,
    const ExprPtr qscale,
    const ExprPtr qzero) {
  BufHandle ResultBuf(name, dims, dtype);
  ResultBuf.node()->set_qscale(qscale);
  ResultBuf.node()->set_qzero(qzero);
  ResultBuf.node()->set_strides(make_contiguous_strides(dims));
  return ResultBuf;
}

static BufHandle makeQBufHandleContiguous(
    const std::string& name,
    const std::vector<ExprHandle>& dims,
    Dtype dtype,
    const double qscale,
    const int64_t qzero) {
  return makeQBufHandleContiguous(
      name,
      dims,
      dtype,
      DoubleImm::make(qscale).node(),
      LongImm::make(qzero).node());
}

static bool isChannelsLast(const BufHandle& buf) {
  const auto& strides = buf.node()->strides();
  const auto& dims = buf.node()->dims();
  const auto rank = dims.size();
  if (rank < 3) {
    return false;
  }
  auto dimsC = to<LongImm>(IRSimplifier::simplify(dims[1]))->value();
  auto stridesC = to<LongImm>(IRSimplifier::simplify(strides[1]))->value();
  auto stridesLast =
      to<LongImm>(IRSimplifier::simplify(strides[rank - 1]))->value();

  return ((stridesLast == dimsC) && (stridesC == 1));
}

static ExprHandle quant(
    ExprHandle x,
    Dtype out_dtype,
    ExprHandle qscale,
    ExprHandle qzero) {
  auto promoted_qscale = promoteToDtype(qscale, x.dtype().scalar_type());
  auto promoted_qzero = promoteToDtype(qzero, x.dtype().scalar_type());
  return promoteToDtype(
      x / promoted_qscale + promoted_qzero + FloatImm::make(0.5f),
      out_dtype.scalar_type());
}

static ExprHandle dequant(
    ExprHandle qx,
    Dtype out_dtype,
    ExprHandle qscale,
    ExprHandle qzero) {
  auto qx_promoted = promoteToDtype(qx, out_dtype.scalar_type());
  auto qscale_promoted =
      promoteToDtype(ExprHandle(qscale), out_dtype.scalar_type());
  auto qzero_promoted =
      promoteToDtype(ExprHandle(qzero), out_dtype.scalar_type());
  return promoteToDtype(
      (qx_promoted - qzero_promoted) * qscale_promoted,
      out_dtype.scalar_type());
}

Tensor computeQuantizePerTensor(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const c10::optional<ScalarType>&,
    at::Device) {
  std::vector<VarPtr> vars;
  std::vector<ExprHandle> indices;
  for (const auto& os : outputShape) {
    auto var = alloc<Var>("", os.node()->dtype());
    vars.push_back(var);
    indices.push_back(VarHandle(var));
  }

  ExprHandle qscale = constant(inputs[1]);
  ExprHandle qzero = constant(inputs[2]);

  const auto dtype = [](auto qdtype) {
    if (static_cast<int64_t>(ScalarType::QInt8) == qdtype) {
      return Dtype(ScalarType::QInt8);
    } else if (static_cast<int64_t>(ScalarType::QUInt8) == qdtype) {
      return Dtype(ScalarType::QUInt8);
    }
    throw malformed_input("Expected quantized dtype");
  }(c10::get<int64_t>(inputs[3]));

  ExprHandle e =
      quant(tensorOrConstant(inputs[0], indices), dtype, qscale, qzero);

  BufPtr buf = alloc<Buf>(
      "quantize_per_tensor",
      ExprHandleVectorToExprVector(outputShape),
      dtype,
      nullptr,
      c10::nullopt,
      qscale.node(),
      qzero.node());
  return Tensor(buf, vars, e.node());
}

Tensor computeQuantizedAdd(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const c10::optional<ScalarType>& outputType,
    at::Device) {
  const BufHandle& QA = c10::get<BufHandle>(inputs[0]);
  const BufHandle& QB = c10::get<BufHandle>(inputs[1]);
  auto qa_scale = ExprHandle(QA.node()->qscale());
  auto qa_zero = ExprHandle(QA.node()->qzero());
  auto qb_scale = ExprHandle(QB.node()->qscale());
  auto qb_zero = ExprHandle(QB.node()->qzero());
  ExprHandle out_qscale = DoubleImm::make(c10::get<double>(inputs[2]));
  ExprHandle out_qzero = LongImm::make(c10::get<int64_t>(inputs[3]));
  Dtype dequant_dtype = kFloat;
  Dtype out_dtype = outputType ? Dtype(*outputType) : QA.dtype();
  std::vector<VarPtr> vars;
  std::vector<ExprHandle> indices;
  for (const auto& os : outputShape) {
    auto var = alloc<Var>("", os.node()->dtype());
    vars.push_back(var);
    indices.push_back(VarHandle(var));
  }
  auto lhs = tensorOrConstant(inputs[0], indices);
  auto rhs = tensorOrConstant(inputs[1], indices);
  ExprHandle exprHandle = quant(
      dequant(lhs, dequant_dtype, qa_scale, qa_zero) +
          dequant(rhs, dequant_dtype, qb_scale, qb_zero),
      out_dtype,
      out_qscale,
      out_qzero);
  BufPtr buf = alloc<Buf>(
      "quantized_add",
      ExprHandleVectorToExprVector(outputShape),
      out_dtype,
      nullptr,
      isChannelsLast(QA) ? make_channels_last_strides(outputShape)
                         : make_contiguous_strides(outputShape),
      out_qscale.node(),
      out_qzero.node());
  return Tensor(buf, vars, exprHandle.node());
}

Tensor computeQuantizePerTensorExternalCall(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    // NOLINTNEXTLINE
    const c10::optional<ScalarType>& outputType,
    at::Device) {
  const BufHandle& x = c10::get<BufHandle>(inputs[0]);
  const auto qscale = c10::get<double>(inputs[1]);
  const auto qzero = c10::get<int64_t>(inputs[2]);
  const auto qdtype = c10::get<int64_t>(inputs[3]);

  const auto dtype = [](auto qdtype) {
    if (static_cast<int64_t>(ScalarType::QInt8) == qdtype) {
      return Dtype(ScalarType::QInt8);
    } else if (static_cast<int64_t>(ScalarType::QUInt8) == qdtype) {
      return Dtype(ScalarType::QUInt8);
    }
    throw malformed_input("Expected quantized dtype");
  }(qdtype);
  auto ResultBuf = [&]() {
    if (isChannelsLast(x)) {
      return makeQBufHandleChannelsLast(
          "quantize_per_tensor", outputShape, dtype, qscale, qzero);
    }
    return makeQBufHandleContiguous(
        "quantize_per_tensor", outputShape, dtype, qscale, qzero);
  }();
  StmtPtr s = ExternalCall::make(
      ResultBuf, "nnc_aten_quantize_per_tensor", {x}, {qscale, qzero, qdtype});
  return Tensor(ResultBuf.node(), s);
}

Tensor computeDequantizeExternalCall(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const c10::optional<ScalarType>& outputType,
    at::Device) {
  Dtype dtype = kFloat;
  if (outputType) {
    dtype = Dtype(*outputType);
  }

  const BufHandle& qx = c10::get<BufHandle>(inputs[0]);
  const int64_t qdtype = (int64_t)immQDType(qx);

  BufHandle ResultBuf("dequantize", outputShape, dtype);
  StmtPtr s = ExternalCall::make(
      ResultBuf,
      "nnc_aten_dequantize",
      {qx},
      {ExprHandle(IRSimplifier::simplify(qx.node()->qscale())),
       ExprHandle(IRSimplifier::simplify(qx.node()->qzero())),
       qdtype});
  return Tensor(ResultBuf.node(), s);
}

Tensor computeQuantizedConv2dPrepack(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const c10::optional<ScalarType>& outputType,
    at::Device) {
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
      "nnc_aten_quantized_conv2d_prepack",
      {qw, b},
      // NOLINTNEXTLINE(facebook-hte-LocalUncheckedArrayBounds)
      {strides[0],
       // NOLINTNEXTLINE(facebook-hte-LocalUncheckedArrayBounds)
       strides[1],
       // NOLINTNEXTLINE(facebook-hte-LocalUncheckedArrayBounds)
       padding[0],
       // NOLINTNEXTLINE(facebook-hte-LocalUncheckedArrayBounds)
       padding[1],
       // NOLINTNEXTLINE(facebook-hte-LocalUncheckedArrayBounds)
       dilation[0],
       // NOLINTNEXTLINE(facebook-hte-LocalUncheckedArrayBounds)
       dilation[1],
       groups,
       immQScale(qw),
       immQZero(qw),
       (int64_t)immQDType(qw)});
  return Tensor(ResultBuf.node(), s);
}

Tensor computeQuantizedConv1d(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    // NOLINTNEXTLINE
    const c10::optional<ScalarType>& outputType,
    // NOLINTNEXTLINE
    at::Device device) {
  const BufHandle& qx = c10::get<BufHandle>(inputs[0]);
  const BufHandle& prepacked = c10::get<BufHandle>(inputs[1]);
  const auto out_qscale = c10::get<double>(inputs[2]);
  const auto out_qzero = c10::get<int64_t>(inputs[3]);
  // Change to dtype based on outputType when dtype propagation implemented
  const auto out_qdtype = immQDType(qx);
  auto ResultBuf = makeQBufHandleChannelsLast(
      "quantized_conv1d",
      outputShape,
      Dtype(out_qdtype),
      out_qscale,
      out_qzero);
  StmtPtr s = ExternalCall::make(
      ResultBuf,
      "nnc_aten_quantized_conv1d",
      {qx, prepacked},
      {immQScale(qx),
       immQZero(qx),
       (int64_t)immQDType(qx),
       out_qscale,
       out_qzero});
  return Tensor(ResultBuf.node(), s);
}

Tensor computeQuantizedConv2d(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    // NOLINTNEXTLINE
    const c10::optional<ScalarType>& outputType,
    // NOLINTNEXTLINE
    at::Device device) {
  const BufHandle& qx = c10::get<BufHandle>(inputs[0]);
  const BufHandle& prepacked = c10::get<BufHandle>(inputs[1]);
  const auto out_qscale = c10::get<double>(inputs[2]);
  const auto out_qzero = c10::get<int64_t>(inputs[3]);
  // Change to dtype based on outputType when dtype propagation implemented
  const auto out_qdtype = immQDType(qx);
  auto ResultBuf = makeQBufHandleChannelsLast(
      "quantized_conv2d",
      outputShape,
      Dtype(out_qdtype),
      out_qscale,
      out_qzero);
  StmtPtr s = ExternalCall::make(
      ResultBuf,
      "nnc_aten_quantized_conv2d",
      {qx, prepacked},
      {immQScale(qx),
       immQZero(qx),
       (int64_t)immQDType(qx),
       out_qscale,
       out_qzero});
  return Tensor(ResultBuf.node(), s);
}

Tensor computeQuantizedConv2dRelu(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    // NOLINTNEXTLINE
    const c10::optional<ScalarType>& outputType,
    // NOLINTNEXTLINE
    at::Device device) {
  const BufHandle& qx = c10::get<BufHandle>(inputs[0]);
  const BufHandle& prepacked = c10::get<BufHandle>(inputs[1]);
  const auto out_qscale = c10::get<double>(inputs[2]);
  const auto out_qzero = c10::get<int64_t>(inputs[3]);
  // Change to dtype based on outputType when dtype propagation implemented
  const auto out_qdtype = immQDType(qx);
  auto ResultBuf = makeQBufHandleChannelsLast(
      "quantized_conv2d_relu",
      outputShape,
      Dtype(out_qdtype),
      out_qscale,
      out_qzero);
  StmtPtr s = ExternalCall::make(
      ResultBuf,
      "nnc_aten_quantized_conv2d_relu",
      {qx, prepacked},
      {immQScale(qx),
       immQZero(qx),
       (int64_t)immQDType(qx),
       out_qscale,
       out_qzero});
  return Tensor(ResultBuf.node(), s);
}

Tensor computeQuantizedLinear(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    // NOLINTNEXTLINE
    const c10::optional<ScalarType>& outputType,
    // NOLINTNEXTLINE
    at::Device device) {
  const BufHandle& qx = c10::get<BufHandle>(inputs[0]);
  const BufHandle& prepacked = c10::get<BufHandle>(inputs[1]);
  const auto out_qscale = c10::get<double>(inputs[2]);
  const auto out_qzero = c10::get<int64_t>(inputs[3]);
  // Change to dtype based on outputType when dtype propagation implemented
  const auto out_qdtype = immQDType(qx);
  auto ResultBuf = makeQBufHandleContiguous(
      "quantized_linear",
      outputShape,
      Dtype(out_qdtype),
      out_qscale,
      out_qzero);
  StmtPtr s = ExternalCall::make(
      ResultBuf,
      "nnc_aten_quantized_linear",
      {qx, prepacked},
      {immQScale(qx),
       immQZero(qx),
       (int64_t)immQDType(qx),
       out_qscale,
       out_qzero});
  return Tensor(ResultBuf.node(), s);
}

Tensor computeQuantizedLinearRelu(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    // NOLINTNEXTLINE
    const c10::optional<ScalarType>& outputType,
    // NOLINTNEXTLINE
    at::Device device) {
  const BufHandle& qx = c10::get<BufHandle>(inputs[0]);
  const BufHandle& prepacked = c10::get<BufHandle>(inputs[1]);
  const auto out_qscale = c10::get<double>(inputs[2]);
  const auto out_qzero = c10::get<int64_t>(inputs[3]);
  // Change to dtype based on outputType when dtype propagation implemented
  const auto out_qdtype = immQDType(qx);
  auto ResultBuf = makeQBufHandleContiguous(
      "quantized_linear_relu",
      outputShape,
      Dtype(out_qdtype),
      out_qscale,
      out_qzero);
  StmtPtr s = ExternalCall::make(
      ResultBuf,
      "nnc_aten_quantized_linear_relu",
      {qx, prepacked},
      {immQScale(qx),
       immQZero(qx),
       (int64_t)immQDType(qx),
       out_qscale,
       out_qzero});
  return Tensor(ResultBuf.node(), s);
}

Tensor computeQuantizedAddExternalCall(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    // NOLINTNEXTLINE
    const c10::optional<ScalarType>& outputType,
    // NOLINTNEXTLINE
    at::Device device) {
  const BufHandle& qa = c10::get<BufHandle>(inputs[0]);
  const BufHandle& qb = c10::get<BufHandle>(inputs[1]);
  const auto out_qscale = c10::get<double>(inputs[2]);
  const auto out_qzero = c10::get<int64_t>(inputs[3]);
  // Change to dtype based on outputType when dtype propagation implemented
  const auto out_qdtype = immQDType(qa);
  const bool isQAChannelsLast = isChannelsLast(qa);
  const bool isQBChannelsLast = isChannelsLast(qb);
  auto ResultBuf = (isQAChannelsLast || isQBChannelsLast)
      ? makeQBufHandleChannelsLast(
            "quantized_add",
            outputShape,
            Dtype(out_qdtype),
            out_qscale,
            out_qzero)
      : makeQBufHandleContiguous(
            "quantized_add",
            outputShape,
            Dtype(out_qdtype),
            out_qscale,
            out_qzero);
  StmtPtr s = ExternalCall::make(
      ResultBuf,
      "nnc_aten_quantized_add",
      {qa, qb},
      {immQScale(qa),
       immQZero(qa),
       (int64_t)immQDType(qa),
       immQScale(qb),
       immQZero(qb),
       (int64_t)immQDType(qb),
       out_qscale,
       out_qzero});
  return Tensor(ResultBuf.node(), s);
}

Tensor computeQuantizedMul(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    // NOLINTNEXTLINE
    const c10::optional<ScalarType>& outputType,
    // NOLINTNEXTLINE
    at::Device device) {
  const BufHandle& qa = c10::get<BufHandle>(inputs[0]);
  const BufHandle& qb = c10::get<BufHandle>(inputs[1]);
  const auto out_qscale = c10::get<double>(inputs[2]);
  const auto out_qzero = c10::get<int64_t>(inputs[3]);
  // Change to dtype based on outputType when dtype propagation implemented
  const auto out_qdtype = immQDType(qa);
  auto ResultBuf = makeQBufHandleContiguous(
      "quantized_mul", outputShape, Dtype(out_qdtype), out_qscale, out_qzero);
  StmtPtr s = ExternalCall::make(
      ResultBuf,
      "nnc_aten_quantized_mul",
      {qa, qb},
      {immQScale(qa),
       immQZero(qa),
       (int64_t)immQDType(qa),
       immQScale(qb),
       immQZero(qb),
       (int64_t)immQDType(qb),
       out_qscale,
       out_qzero});
  return Tensor(ResultBuf.node(), s);
}

Tensor computeQuantizedMulScalar(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    // NOLINTNEXTLINE
    const c10::optional<ScalarType>& outputType,
    // NOLINTNEXTLINE
    at::Device device) {
  const BufHandle& qa = c10::get<BufHandle>(inputs[0]);
  const auto scalar = c10::get<double>(inputs[1]);
  // Change to dtype based on outputType when dtype propagation implemented
  const auto out_qdtype = immQDType(qa);
  double scale1 = immQScale(qa);
  auto ResultBuf = makeQBufHandleContiguous(
      "quantized_mul_scalar",
      outputShape,
      Dtype(out_qdtype),
      scale1 * scalar,
      immQZero(qa));
  StmtPtr s = ExternalCall::make(
      ResultBuf,
      "nnc_aten_quantized_mul_scalar",
      {qa},
      {scale1, immQZero(qa), (int64_t)immQDType(qa), scalar});
  return Tensor(ResultBuf.node(), s);
}

Tensor computeQuantizedRelu(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    // NOLINTNEXTLINE
    const c10::optional<ScalarType>& outputType,
    // NOLINTNEXTLINE
    at::Device device) {
  const BufHandle& qa = c10::get<BufHandle>(inputs[0]);
  const auto out_qdtype = immQDType(qa);
  const bool isQAChannelsLast = isChannelsLast(qa);
  auto ResultBuf = isQAChannelsLast ? makeQBufHandleChannelsLast(
                                          "quantized_relu",
                                          outputShape,
                                          Dtype(out_qdtype),
                                          immQScale(qa),
                                          immQZero(qa))
                                    : makeQBufHandleContiguous(
                                          "quantized_relu",
                                          outputShape,
                                          Dtype(out_qdtype),
                                          immQScale(qa),
                                          immQZero(qa));
  StmtPtr s = ExternalCall::make(
      ResultBuf,
      "nnc_aten_quantized_relu",
      {qa},
      {immQScale(qa), immQZero(qa), (int64_t)immQDType(qa)});
  return Tensor(ResultBuf.node(), s);
}

Tensor computeQuantizedCat(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    // NOLINTNEXTLINE
    const c10::optional<ScalarType>& outputType,
    // NOLINTNEXTLINE
    at::Device device) {
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  auto inputList = c10::get<BufList>(inputs[0]);
  auto argDim = c10::get<int64_t>(inputs[1]);
  auto n = inputList.size();
  // TODO: handle optional out_qscale, out_qzero
  const auto out_qscale = c10::get<double>(inputs[2]);
  const auto out_qzero = c10::get<int64_t>(inputs[3]);

  std::vector<BufHandle> args;
  std::vector<ExprHandle> extra_args;
  for (const auto i : c10::irange(n)) {
    const BufHandle& bh = inputList[i];
    args.emplace_back(bh);
    extra_args.emplace_back(immQScale(bh));
    extra_args.emplace_back(immQZero(bh));
    extra_args.emplace_back((int64_t)immQDType(bh));
  }
  extra_args.emplace_back(argDim);
  extra_args.emplace_back(out_qscale);
  extra_args.emplace_back(out_qzero);
  auto ResultBuf = makeQBufHandleContiguous(
      "quantized_cat",
      outputShape,
      Dtype(immQDType(inputList[0])),
      out_qscale,
      out_qzero);
  StmtPtr s =
      ExternalCall::make(ResultBuf, "nnc_aten_quantized_cat", args, extra_args);
  return Tensor(ResultBuf.node(), s);
}

Tensor computeDequantize(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const c10::optional<ScalarType>& outputType,
    at::Device) {
  Dtype dtype = kFloat;
  if (outputType) {
    dtype = Dtype(*outputType);
  }
  auto qx = c10::get<BufHandle>(inputs[0]);
  TORCH_INTERNAL_ASSERT(
      qx.node()->qscale(),
      buildErrorMessage("Missing quantized scale for dequantize"));
  TORCH_INTERNAL_ASSERT(
      qx.node()->qzero(),
      buildErrorMessage("Missing quantized zero point for dequantize"));
  auto qscale = ExprHandle(qx.node()->qscale());
  auto qzero = ExprHandle(qx.node()->qzero());
  std::vector<VarPtr> vars;
  std::vector<ExprHandle> indices;
  for (const auto& os : outputShape) {
    auto var = alloc<Var>("", os.node()->dtype());
    vars.push_back(var);
    indices.push_back(VarHandle(var));
  }
  auto y = dequant(tensorOrConstant(inputs[0], indices), dtype, qscale, qzero);
  BufPtr buf = alloc<Buf>(
      "dequantize", ExprHandleVectorToExprVector(outputShape), dtype);
  return Tensor(buf, vars, y.node());
}

Tensor computeUpsampleNearest2d(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const c10::optional<ScalarType>& outputType,
    at::Device) {
  auto A = c10::get<BufHandle>(inputs[0]);
  const auto& output_height = outputShape[2];
  const auto& output_width = outputShape[3];
  auto input_height = ExprHandle(A.dim(2));
  auto input_width = ExprHandle(A.dim(3));

  std::vector<VarHandle> args = create_index_vars(outputShape);
  // Handle separately when scale is specified? as in 'scalar_t
  // compute_scales_value' in UpSample.h
  auto scale_h =
      promoteToDtype(input_height, ScalarType::Double) / output_height;
  auto scale_w = promoteToDtype(input_width, ScalarType::Double) / output_width;
  // TODO: will repetitive if in idx calculation will be taken out of the loop?
  auto compute_nearest_idx =
      [](ExprHandle scale, ExprHandle dst_index, ExprHandle input_size) {
        return Min::make(
            promoteToDtype(floor(dst_index * scale), ScalarType::Long),
            input_size - 1,
            true);
      };
  auto body_func = [&](std::vector<VarHandle> axes) {
    std::vector<ExprHandle> newAxes(axes.begin(), axes.end());
    newAxes[2] = compute_nearest_idx(scale_h, axes[2], input_height);
    newAxes[3] = compute_nearest_idx(scale_w, axes[3], input_width);
    return A.load(newAxes);
  };
  auto e = body_func(args);
  auto strides = isChannelsLast(A) ? make_channels_last_strides(outputShape)
                                   : make_contiguous_strides(outputShape);
  BufHandle buf = Buf::make(
      "upsample_nearest2d",
      outputShape,
      Dtype(*outputType),
      c10::nullopt, // initializer
      fmap(strides, [&](ExprPtr stride) { return ExprHandle(stride); }),
      ExprHandle(A.node()->qscale()),
      ExprHandle(A.node()->qzero()));
  return Tensor(buf, args, e);
}

Tensor computeUpsampleNearest2dExternalCall(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    const c10::optional<ScalarType>& outputType,
    at::Device) {
  Dtype dtype = kFloat;
  if (outputType) {
    dtype = Dtype(*outputType);
  }
  int64_t output_size_h = -1;
  int64_t output_size_w = -1;
  if (auto output_sizes = c10::get_if<IntList>(&inputs[1])) {
    output_size_h = (*output_sizes)[0];
    output_size_w = (*output_sizes)[1];
  }

  double scale_factor_h = -1.f;
  double scale_factor_w = -1.f;
  if (auto scale_factors = c10::get_if<DoubleList>(&inputs[2])) {
    scale_factor_h = (*scale_factors)[0];
    scale_factor_w = (*scale_factors)[1];
  }
  const BufHandle& x = c10::get<BufHandle>(inputs[0]);
  double qx_qscale = -1.f;
  int64_t qx_qzero = -1l;
  int64_t qx_qdtype = -1l;
  if (isQuantized(x)) {
    qx_qscale = immQScale(x);
    qx_qzero = immQZero(x);
    qx_qdtype = (int64_t)immQDType(x);
  }

  BufHandle ResultBuf = [&]() {
    if (isQuantized(x)) {
      return makeQBufHandleChannelsLast(
          "upsample_nearest2d",
          outputShape,
          Dtype(immQDType(x)),
          qx_qscale,
          qx_qzero);
    }
    return BufHandle("upsample_nearest2d", outputShape, dtype);
  }();

  StmtPtr s = ExternalCall::make(
      ResultBuf,
      "nnc_aten_upsample_nearest2d",
      {x},
      {qx_qscale,
       qx_qzero,
       qx_qdtype,
       output_size_h,
       output_size_w,
       scale_factor_h,
       scale_factor_w});
  return Tensor(ResultBuf.node(), s);
}

Tensor computeQuantizedSigmoidExternalCall(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const std::vector<ExprHandle>& outputStrides,
    // NOLINTNEXTLINE
    const c10::optional<ScalarType>& outputType,
    at::Device) {
  const BufHandle& qx = c10::get<BufHandle>(inputs[0]);

  const auto out_qdtype = immQDType(qx);
  const double out_qscale = 1.0f / 256.0f;
  const int64_t out_qzero = (out_qdtype == ScalarType::QInt8) ? -128 : 0;

  auto ResultBuf = isChannelsLast(qx) ? makeQBufHandleChannelsLast(
                                            "quantized_sigmoid",
                                            outputShape,
                                            Dtype(out_qdtype),
                                            out_qscale,
                                            out_qzero)
                                      : makeQBufHandleContiguous(
                                            "quantized_sigmoid",
                                            outputShape,
                                            Dtype(out_qdtype),
                                            out_qscale,
                                            out_qzero);
  StmtPtr s = ExternalCall::make(
      ResultBuf,
      "nnc_aten_quantized_sigmoid",
      {qx},
      {immQScale(qx),
       immQZero(qx),
       (int64_t)immQDType(qx),
       out_qscale,
       out_qzero});
  return Tensor(ResultBuf.node(), s);
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
