#include <c10/core/ScalarType.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/operators/misc.h>
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
  return to<DoubleImm>(IRSimplifier::simplify(qx.node()->qscale()))->value();
}

int64_t immQZero(const BufHandle& qx) {
  return to<LongImm>(IRSimplifier::simplify(qx.node()->qzero()))->value();
}

ScalarType immQDType(const BufHandle& qx) {
  return qx.dtype().scalar_type();
}

bool isQuantized(const BufHandle& qx) {
  return qx.node()->qscale() && qx.node()->qzero();
}

BufHandle makeQBufHandleNCHW(
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

BufHandle makeQBufHandleNHWC(
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

BufHandle makeQBufHandleNHWC(
    const std::string& name,
    const std::vector<ExprHandle>& dims,
    Dtype dtype,
    const double qscale,
    const int64_t qzero) {
  return makeQBufHandleNHWC(
      name,
      dims,
      dtype,
      DoubleImm::make(qscale).node(),
      LongImm::make(qzero).node());
}

BufHandle makeQBufHandleNCHW(
    const std::string& name,
    const std::vector<ExprHandle>& dims,
    Dtype dtype,
    const double qscale,
    const int64_t qzero) {
  return makeQBufHandleNCHW(
      name,
      dims,
      dtype,
      DoubleImm::make(qscale).node(),
      LongImm::make(qzero).node());
}

Tensor computeQuantizePerTensor(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
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
      c10::nullopt,
      qscale.node(),
      qzero.node());
  return Tensor(buf, vars, exprHandle.node());
}

Tensor computeQuantizePerTensorExternalCall(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
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
  auto ResultBuf = makeQBufHandleNCHW(
      "quantize_per_tensor", outputShape, dtype, qscale, qzero);
  StmtPtr s = ExternalCall::make(
      ResultBuf, "nnc_aten_quantize_per_tensor", {x}, {qscale, qzero, qdtype});
  return Tensor(ResultBuf.node(), s);
}

Tensor computeDequantizeExternalCall(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const c10::optional<ScalarType>& outputType,
    at::Device) {
  Dtype dtype = kFloat;
  if (outputType) {
    dtype = Dtype(*outputType);
  }

  const BufHandle& qx = c10::get<BufHandle>(inputs[0]);
  const double qscale = immQScale(qx);
  const int64_t qzero = immQZero(qx);
  const int64_t qdtype = (int64_t)immQDType(qx);

  BufHandle ResultBuf("quantize", outputShape, dtype);
  StmtPtr s = ExternalCall::make(
      ResultBuf, "nnc_aten_dequantize", {qx}, {qscale, qzero, qdtype});
  return Tensor(ResultBuf.node(), s);
}

Tensor computeQuantizedConv2dPrepack(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
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
  auto ResultBuf = makeQBufHandleNCHW(
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
  auto ResultBuf = makeQBufHandleNHWC(
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
  auto ResultBuf = makeQBufHandleNHWC(
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
  auto ResultBuf = makeQBufHandleNCHW(
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
  auto ResultBuf = makeQBufHandleNCHW(
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

bool isChannelsLast(const BufHandle& buf) {
  const auto& strides = buf.node()->strides();
  const auto& dims = buf.node()->dims();
  if (strides.size() != 4) {
    return false;
  }
  auto dims1 = to<LongImm>(IRSimplifier::simplify(dims[1]))->value();
  auto strides1 = to<LongImm>(IRSimplifier::simplify(strides[1]))->value();
  auto strides3 = to<LongImm>(IRSimplifier::simplify(strides[3]))->value();

  return ((strides3 == dims1) && (strides1 == 1));
}

Tensor computeQuantizedAdd(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
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
      ? makeQBufHandleNHWC(
            "quantized_add",
            outputShape,
            Dtype(out_qdtype),
            out_qscale,
            out_qzero)
      : makeQBufHandleNCHW(
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
  auto ResultBuf = makeQBufHandleNCHW(
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
    // NOLINTNEXTLINE
    const c10::optional<ScalarType>& outputType,
    // NOLINTNEXTLINE
    at::Device device) {
  const BufHandle& qa = c10::get<BufHandle>(inputs[0]);
  const auto scalar = c10::get<double>(inputs[1]);
  // Change to dtype based on outputType when dtype propagation implemented
  const auto out_qdtype = immQDType(qa);
  auto ResultBuf = makeQBufHandleNCHW(
      "quantized_mul_scalar",
      outputShape,
      Dtype(out_qdtype),
      immQScale(qa),
      immQZero(qa));
  StmtPtr s = ExternalCall::make(
      ResultBuf,
      "nnc_aten_quantized_mul_scalar",
      {qa},
      {immQScale(qa), immQZero(qa), (int64_t)immQDType(qa), scalar});
  return Tensor(ResultBuf.node(), s);
}

Tensor computeDequantize(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const c10::optional<ScalarType>& outputType,
    at::Device) {
  Dtype dtype = kFloat;
  if (outputType) {
    dtype = Dtype(*outputType);
  }
  auto qx = c10::get<BufHandle>(inputs[0]);
  auto qscale = qx.node()->qscale();
  auto qzero = qx.node()->qzero();
  TORCH_INTERNAL_ASSERT(
      qscale, buildErrorMessage("Missing quantized scale for dequantize"));
  TORCH_INTERNAL_ASSERT(
      qzero, buildErrorMessage("Missing quantized zero point for dequantize"));
  std::vector<VarPtr> vars;
  std::vector<ExprHandle> indices;
  for (const auto& os : outputShape) {
    auto var = alloc<Var>("", os.node()->dtype());
    vars.push_back(var);
    indices.push_back(VarHandle(var));
  }
  auto qx_e_promoted =
      promoteToDtype(tensorOrConstant(inputs[0], indices), dtype.scalar_type());
  auto qscale_promoted =
      promoteToDtype(ExprHandle(qscale), dtype.scalar_type());
  auto qzero_promoted = promoteToDtype(ExprHandle(qzero), dtype.scalar_type());
  auto y = promoteToDtype(
      (qx_e_promoted - qzero_promoted) * qscale_promoted, dtype.scalar_type());

  BufPtr buf = alloc<Buf>(
      "dequantize", ExprHandleVectorToExprVector(outputShape), dtype);
  return Tensor(buf, vars, y.node());
}

Tensor computeUpsampleNearest2d(
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
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
      return makeQBufHandleNHWC(
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
    // NOLINTNEXTLINE
    const c10::optional<ScalarType>& outputType,
    at::Device) {
  const BufHandle& qx = c10::get<BufHandle>(inputs[0]);

  const auto out_qdtype = immQDType(qx);
  const double out_qscale = 1.0f / 256.0f;
  const int64_t out_qzero = (out_qdtype == ScalarType::QInt8) ? -128 : 0;

  auto ResultBuf = makeQBufHandleNHWC(
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
