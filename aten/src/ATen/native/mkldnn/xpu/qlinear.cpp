#include <torch/library.h>

#include <ATen/BlasBackend.h>
#include <ATen/TensorUtils.h>
#include <ATen/ceil_div.h>
#include <ATen/native/ScaledBlasUtils.h>
#include <ATen/native/mkldnn/xpu/detail/oneDNN.h>
#include <ATen/native/mkldnn/xpu/qlinear.h>
#include <c10/core/ScalarType.h>

using namespace at::native::onednn;

namespace at::native::xpu {

namespace {

using at::blas::ScalingType;
using at::native::scaled::convert_int_to_enum;

Tensor& _woq_scaled_gemm(
    const Tensor& mat_a,
    const Tensor& mat_b,
    const Tensor& scale_b,
    const ScalingType scaling_choice_b,
    const std::optional<Tensor>& bias,
    Tensor& out) {
  at::native::onednn::scaled_matmul(
      mat_a,
      mat_b,
      out,
      /*scale_a*/ std::nullopt,
      scale_b,
      /*scaling_choice_a*/ std::nullopt,
      scaling_choice_b,
      bias,
      /*scale_result*/ std::nullopt,
      /*use_fast_accum*/ false);
  return out;
}

Tensor& _woq_scaled_tensorwise(
    const Tensor& mat_a,
    const Tensor& mat_b,
    const Tensor& scale_b,
    const std::optional<Tensor>& bias,
    Tensor& out) {
  TORCH_CHECK_VALUE(
      mat_b.scalar_type() == at::kFloat8_e4m3fn ||
          mat_b.scalar_type() == at::kFloat8_e5m2,
      "woq_scaled_mm: TensorWise recipe requires mat_b to be fp8 "
      "(Float8_e4m3fn or Float8_e5m2), got ",
      mat_b.scalar_type());
  TORCH_CHECK_VALUE(
      scale_b.numel() == 1 && scale_b.scalar_type() == kFloat,
      "woq_scaled_mm: TensorWise scale_b must be a float32, got "
      "dtype ",
      scale_b.scalar_type(),
      " and shape ",
      scale_b.sizes());

  return _woq_scaled_gemm(
      mat_a, mat_b, scale_b, ScalingType::TensorWise, bias, out);
}

Tensor& _woq_scaled_rowwise(
    const Tensor& mat_a,
    const Tensor& mat_b,
    const Tensor& scale_b,
    const std::optional<Tensor>& bias,
    Tensor& out) {
  const int64_t N = mat_b.size(1);
  TORCH_CHECK_VALUE(
      mat_b.scalar_type() == at::kFloat8_e4m3fn ||
          mat_b.scalar_type() == at::kFloat8_e5m2,
      "woq_scaled_mm: RowWise recipe requires mat_b to be fp8 "
      "(Float8_e4m3fn or Float8_e5m2), got ",
      mat_b.scalar_type());
  TORCH_CHECK_VALUE(
      scale_b.dim() == 2 && scale_b.size(0) == 1 && scale_b.size(1) == N,
      "woq_scaled_mm: RowWise scale_b should be (1, ",
      N,
      "), got ",
      scale_b.sizes());
  TORCH_CHECK_VALUE(
      scale_b.scalar_type() == kFloat,
      "woq_scaled_mm: RowWise scale_b must be Float, got ",
      scale_b.scalar_type());
  TORCH_CHECK_VALUE(
      scale_b.stride(1) == 1,
      "woq_scaled_mm: expected scale_b.stride(1) to be 1, but got ",
      scale_b.stride(1));

  return _woq_scaled_gemm(
      mat_a, mat_b, scale_b, ScalingType::RowWise, bias, out);
}

Tensor& _woq_scaled_block128x128(
    const Tensor& mat_a,
    const Tensor& mat_b,
    const Tensor& scale_b,
    const std::optional<Tensor>& bias,
    Tensor& out) {
  const int64_t K = mat_a.size(-1);
  const int64_t N = mat_b.size(1);
  TORCH_CHECK_VALUE(
      mat_b.scalar_type() == at::kFloat8_e4m3fn,
      "woq_scaled_mm: BlockWise128x128 recipe requires mat_b to be "
      "Float8_e4m3fn, got ",
      mat_b.scalar_type());
  TORCH_CHECK_VALUE(
      scale_b.dim() == 2 && scale_b.size(0) == ceil_div<int64_t>(N, 128) &&
          scale_b.size(1) == ceil_div<int64_t>(K, 128) &&
          (scale_b.scalar_type() == kFloat ||
           scale_b.scalar_type() == kFloat8_e8m0fnu),
      "woq_scaled_mm: BlockWise128x128 scale_b must be float32 or "
      "float8_e8m0fnu with shape ",
      ceil_div<int64_t>(N, 128),
      " x ",
      ceil_div<int64_t>(K, 128),
      ", got dtype ",
      scale_b.scalar_type(),
      " and shape ",
      scale_b.sizes());

  // scale_b_list[0] : [N//128, K//128]. However, oneDNN needs real
  // row-major 2D data, so scale_b is transposed to [K//128, N//128] below.
  Tensor sb = scale_b.t();
  sb = sb.is_contiguous() ? sb : sb.contiguous();

  return _woq_scaled_gemm(
      mat_a, mat_b, sb, ScalingType::BlockWise128x128, bias, out);
}

Tensor& _woq_scaled_mxfp8(
    const Tensor& mat_a,
    const Tensor& mat_b,
    const Tensor& scale_b,
    const std::optional<Tensor>& bias,
    Tensor& out) {
  const int64_t K = mat_a.size(-1);
  const int64_t N = mat_b.size(1);
  TORCH_CHECK_VALUE(
      mat_b.scalar_type() == at::kFloat8_e4m3fn,
      "woq_scaled_mm: BlockWise1x32 (MXFP8) recipe requires mat_b to be "
      "Float8_e4m3fn, got ",
      mat_b.scalar_type());
  TORCH_CHECK_VALUE(
      scale_b.dim() == 2 && scale_b.size(0) == N &&
          scale_b.size(1) == ceil_div<int64_t>(K, 32) &&
          scale_b.scalar_type() == kFloat8_e8m0fnu,
      "woq_scaled_mm: BlockWise1x32 scale_b must be float8_e8m0fnu with "
      "shape ",
      N,
      " x ",
      ceil_div<int64_t>(K, 32),
      ", got dtype ",
      scale_b.scalar_type(),
      " and shape ",
      scale_b.sizes());

  // scale_b_list[0] : [N, K//32]. However, oneDNN needs real row-major 2D
  // data, so scale_b is transposed to [K//32, N] below.
  Tensor sb = scale_b.t();
  sb = sb.is_contiguous() ? sb : sb.contiguous();

  return _woq_scaled_gemm(
      mat_a, mat_b, sb, ScalingType::BlockWise1x32, bias, out);
}

Tensor& _woq_scaled_mxfp4(
    const Tensor& mat_a,
    const Tensor& mat_b,
    const Tensor& scale_b,
    const std::optional<Tensor>& bias,
    Tensor& out) {
  const int64_t K = mat_a.size(-1);
  const int64_t N = mat_b.size(1);
  TORCH_CHECK_VALUE(
      mat_b.scalar_type() == c10::ScalarType::Float4_e2m1fn_x2,
      "woq_scaled_mm: BlockWise1x32 (MXFP4) recipe requires mat_b to be "
      "Float4_e2m1fn_x2, got ",
      mat_b.scalar_type());
  TORCH_CHECK_VALUE(
      scale_b.dim() == 2 && scale_b.size(0) == N &&
          scale_b.size(1) == ceil_div<int64_t>(K, 32) &&
          scale_b.scalar_type() == kFloat8_e8m0fnu,
      "woq_scaled_mm: BlockWise1x32 scale_b must be float8_e8m0fnu with "
      "shape ",
      N,
      " x ",
      ceil_div<int64_t>(K, 32),
      ", got dtype ",
      scale_b.scalar_type(),
      " and shape ",
      scale_b.sizes());

  // scale_b_list[0] : [N, K//32]. However, oneDNN needs real row-major 2D
  // data, so scale_b is transposed to [K//32, N] below.
  Tensor sb = scale_b.t();
  sb = sb.is_contiguous() ? sb : sb.contiguous();

  return _woq_scaled_gemm(
      mat_a, mat_b, sb, ScalingType::BlockWise1x32, bias, out);
}

} // namespace

inline c10::ScalarType QLinearOnednnXPU::qlinear_decide_out_dtype(
    const at::Tensor& act,
    const std::optional<c10::ScalarType> output_dtype) {
  bool fp32_output = output_dtype.has_value() && (output_dtype == c10::kFloat);
  bool bfloat16_output =
      output_dtype.has_value() && (output_dtype == c10::kBFloat16);
  auto dst_dtype = fp32_output
      ? c10::kFloat
      : (bfloat16_output ? c10::kBFloat16 : act.scalar_type());
  return dst_dtype;
}

Tensor QLinearOnednnXPU::q_linear_pointwise(
    Tensor act,
    double act_scale,
    int64_t act_zero_point,
    Tensor weight,
    Tensor weight_scales,
    Tensor weight_zero_points,
    std::optional<Tensor> bias,
    double output_scale,
    int64_t output_zero_point,
    std::optional<c10::ScalarType> output_dtype,
    std::string_view post_op_name,
    torch::List<std::optional<at::Scalar>> post_op_args,
    std::string_view post_op_algorithm) {
  Tensor b_raw = bias.has_value() ? bias.value() : at::Tensor();

  const int64_t dim = act.dim();
  TORCH_CHECK(dim == 2, "qliner XPU: input dim should be 2, but got", dim);
  TORCH_CHECK(
      act.device() == weight.device() &&
          act.device() == weight_scales.device() &&
          act.device() == weight_zero_points.device(),
      "qlinear xpu: input tensors(act, weight, weight scale, weight zero-points) should be on the same device");
  int64_t K = act.size(dim - 1);
  int64_t M = act.numel() / K;
  // [M, K] x [K, N]
  int64_t N = weight.size(1);

  std::vector<int64_t> src_dims = {M, K};
  std::vector<int64_t> dst_dims = {M, N};

  auto dst_dtype = qlinear_decide_out_dtype(act, output_dtype);
  Tensor qout = at::empty(dst_dims, act.options().dtype(dst_dtype));

  quantized_matmul(
      act.contiguous(),
      act_scale,
      act_zero_point,
      weight.contiguous(),
      weight_scales,
      weight_zero_points,
      b_raw,
      qout,
      output_scale,
      output_zero_point,
      output_dtype,
      /*other*/ std::nullopt,
      /*other scale*/ 1.0,
      /*other zp*/ 0,
      /*binary post op*/ "none",
      /*binary alpha*/ 1.0,
      post_op_name,
      post_op_args,
      post_op_algorithm,
      /*m2_trans*/ true);

  return qout;
}

Tensor QLinearOnednnXPU::q_linear_pointwise_tensor(
    Tensor act,
    Tensor act_scale,
    Tensor act_zero_point,
    Tensor weight,
    Tensor weight_scales,
    Tensor weight_zero_points,
    std::optional<Tensor> bias,
    double output_scale,
    int64_t output_zero_point,
    std::optional<c10::ScalarType> output_dtype,
    std::string_view post_op_name,
    torch::List<std::optional<at::Scalar>> post_op_args,
    std::string_view post_op_algorithm) {
  Tensor b_raw = bias.has_value() ? bias.value() : at::Tensor();

  const int64_t dim = act.dim();
  TORCH_CHECK(dim == 2, "qliner XPU: input dim should be 2, but got", dim);
  TORCH_CHECK(
      act.device() == weight.device() &&
          act.device() == weight_scales.device() &&
          act.device() == weight_zero_points.device(),
      "qlinear xpu: input tensors(act, weight, weight scale, weight zero-points) should be on the same device");
  int64_t K = act.size(dim - 1);
  int64_t M = act.numel() / K;
  // [M, K] x [K, N]
  int64_t N = weight.size(1);

  std::vector<int64_t> src_dims = {M, K};
  std::vector<int64_t> dst_dims = {M, N};

  auto dst_dtype = qlinear_decide_out_dtype(act, output_dtype);
  Tensor qout = at::empty(dst_dims, act.options().dtype(dst_dtype));

  quantized_matmul(
      act.contiguous(),
      act_scale.item().toDouble(),
      act_zero_point.item().toLong(),
      weight.contiguous(),
      weight_scales,
      weight_zero_points,
      b_raw,
      qout,
      output_scale,
      output_zero_point,
      output_dtype,
      /*other*/ std::nullopt,
      /*other scale*/ 1.0,
      /*other zp*/ 0,
      /*binary post op*/ "none",
      /*binary alpha*/ 1.0,
      post_op_name,
      post_op_args,
      post_op_algorithm,
      /*m2_trans*/ true);

  return qout;
}

Tensor QLinearOnednnXPU::q_linear_pointwise_binary(
    Tensor act,
    double act_scale,
    int64_t act_zero_point,
    Tensor weight,
    Tensor weight_scales,
    Tensor weight_zero_points,
    std::optional<at::Tensor> other,
    std::optional<Tensor> bias,
    double output_scale,
    int64_t output_zero_point,
    std::optional<c10::ScalarType> output_dtype,
    double other_scale,
    int64_t other_zero_point,
    std::string_view binary_post_op,
    double binary_alpha,
    std::string_view unary_post_op,
    torch::List<std::optional<at::Scalar>> unary_post_op_args,
    std::string_view unary_post_op_algorithm) {
  TORCH_CHECK(
      act.device() == weight.device() &&
          act.device() == weight_scales.device() &&
          act.device() == weight_zero_points.device(),
      "qlinear xpu: input tensors(act, weight, weight scale, weight zero-points) should be on the same device");
  Tensor b_raw = bias.has_value() ? bias.value() : at::Tensor();

  const int64_t dim = act.dim();
  TORCH_CHECK(
      dim == 2 || dim == 3,
      "qliner_pointwise_binary XPU: input dim should be 2 or 3, but got",
      dim);
  int64_t K = act.size(dim - 1);
  int64_t M = act.numel() / K;
  // [M, K] x [K, N]
  int64_t N = weight.size(1);
  Tensor input = dim == 3 ? act.reshape({-1, K}) : act;
  std::vector<int64_t> src_dims = {M, K};
  std::vector<int64_t> dst_dims = {M, N};
  auto dst_dtype = qlinear_decide_out_dtype(act, output_dtype);
  bool has_accum_postop_sum = (binary_post_op == "sum");
  if (dim == 3) {
    other = other.has_value() ? other.value().reshape({-1, N}) : other;
  }
  Tensor qout = has_accum_postop_sum
      ? other.value()
      : at::empty(dst_dims, act.options().dtype(dst_dtype));
  quantized_matmul(
      input.contiguous(),
      act_scale,
      act_zero_point,
      weight.contiguous(),
      weight_scales,
      weight_zero_points,
      b_raw,
      qout,
      output_scale,
      output_zero_point,
      output_dtype,
      /*other*/ other,
      /*other scale*/ other_scale,
      /*other zp*/ other_zero_point,
      /*binary post op*/ binary_post_op,
      /*binary alpha*/ binary_alpha,
      unary_post_op,
      unary_post_op_args,
      unary_post_op_algorithm,
      /*m2_trans*/ true);

  return dim == 3 ? qout.reshape({act.size(0), -1, N}) : qout;
}

Tensor QLinearOnednnXPU::q_linear_pointwise_binary_tensor(
    Tensor act,
    Tensor act_scale,
    Tensor act_zero_point,
    Tensor weight,
    Tensor weight_scales,
    Tensor weight_zero_points,
    std::optional<at::Tensor> other,
    std::optional<Tensor> bias,
    double output_scale,
    int64_t output_zero_point,
    std::optional<c10::ScalarType> output_dtype,
    double other_scale,
    int64_t other_zero_point,
    std::string_view binary_post_op,
    double binary_alpha,
    std::string_view unary_post_op,
    torch::List<std::optional<at::Scalar>> unary_post_op_args,
    std::string_view unary_post_op_algorithm) {
  return q_linear_pointwise_binary(
      act,
      act_scale.item().toDouble(),
      act_zero_point.item().toLong(),
      weight,
      weight_scales,
      weight_zero_points,
      other,
      bias,
      output_scale,
      output_zero_point,
      output_dtype,
      other_scale,
      other_zero_point,
      binary_post_op,
      binary_alpha,
      unary_post_op,
      unary_post_op_args,
      unary_post_op_algorithm);
}

Tensor QLinearOnednnXPU::q_linear_prepack_onednn(
    at::Tensor weight,
    std::optional<torch::List<int64_t>> input_shape) {
  at::Tensor weight_transposed = weight.transpose(0, 1);
  return weight_transposed;
}

// Weight-only-quantization (WOQ) _scaled_mm: computes `mat_a @ mat_b + bias`
// while applying scaling to `mat_b` only (`mat_a` stays full precision). Scales
// are applicable when `mat_b` is of Float8 or Float4 type.
//  Arguments:
//    - `mat_a`: the activation, must be 2D [M, K], can be type
//    `torch.bfloat16`, `torch.float16`, or `torch.float32`
//    - `mat_b`: the weight, must be 2D [K, N], can be type
//    `torch.float8_e4m3fn`, `torch.float8_e5m2`, or `torch.float4_e2m1fn_x2`
//    - `scale_b_list`: a single-element list with the inverse scale of
//    `mat_b`, whose shape/strides/dtype depend on the scaling scheme. For
//    the block-wise recipes (BlockWise128x128 and BlockWise1x32/MXFP8/
//    MXFP4), `scale_b_list[0]` is N-major -- shaped
//    [N//block_n, K//block_k] -- and must be N-contiguous
//    - `scale_recipe_b`: a single-element list with an integer
//    corresponding to a `ScalingType` enum describing the scaling scheme
//    used for `scale_b_list[0]`
//    - `bias`: the bias, can be type `torch.bfloat16`, `torch.float16`, or
//    `torch.float32`
Tensor QLinearOnednnXPU::woq_scaled_mm(
    const Tensor& mat_a,
    const Tensor& mat_b,
    const at::ITensorListRef& scale_b_list,
    at::IntArrayRef scale_recipe_b,
    const std::optional<Tensor>& bias) {
  TORCH_CHECK_VALUE(
      mat_a.dim() == 2,
      "woq_scaled_mm: mat_a must be 2D [M, K], got dim ",
      mat_a.dim());
  TORCH_CHECK_VALUE(
      mat_b.dim() == 2,
      "woq_scaled_mm: mat_b must be 2D [K, N], got dim ",
      mat_b.dim());
  TORCH_CHECK_VALUE(
      mat_a.scalar_type() == at::kBFloat16 ||
          mat_a.scalar_type() == at::kHalf || mat_a.scalar_type() == at::kFloat,
      "woq_scaled_mm: mat_a must be bfloat16, half or float, got ",
      mat_a.scalar_type());
  const bool is_fp4 = mat_b.scalar_type() == c10::ScalarType::Float4_e2m1fn_x2;
  TORCH_CHECK_VALUE(
      is_fp4 || mat_b.scalar_type() == at::kFloat8_e4m3fn ||
          mat_b.scalar_type() == at::kFloat8_e5m2,
      "woq_scaled_mm: mat_b must be Float8_e4m3fn, Float8_e5m2 or "
      "Float4_e2m1fn_x2 (packed), got ",
      mat_b.scalar_type());

  std::vector<Tensor> scale_b(scale_b_list.begin(), scale_b_list.end());
  TORCH_CHECK_VALUE(
      !scale_b.empty() && scale_b.size() == scale_recipe_b.size(),
      "woq_scaled_mm: scale_b_list and scale_recipe_b must be non-empty "
      "and have the same number of elements, got ",
      scale_b.size(),
      " and ",
      scale_recipe_b.size());

  const int64_t M = mat_a.size(0);
  const int64_t K = mat_a.size(1);
  const int64_t N = mat_b.size(1);
  const int64_t K_multiplier = is_fp4 ? 2 : 1;
  TORCH_CHECK_VALUE(
      mat_b.size(0) * K_multiplier == K,
      "woq_scaled_mm: mat_b's contraction dim (",
      mat_b.size(0) * K_multiplier,
      ") must match mat_a's trailing dim (",
      K,
      ")");

  if (bias.has_value()) {
    TORCH_CHECK_VALUE(
        bias->scalar_type() == kFloat ||
            bias->scalar_type() == c10::ScalarType::BFloat16 ||
            bias->scalar_type() == c10::ScalarType::Half,
        "Bias must be Float32 or BFloat16 or Half, but got ",
        bias->scalar_type());
    TORCH_CHECK_VALUE(
        bias->numel() == N,
        "Bias must be size ",
        N,
        " but got ",
        bias->numel());
  }

  Tensor out = at::empty({M, N}, mat_a.options());
  {
    auto bias_ = bias.value_or(Tensor());
    std::vector<TensorArg> targs{
        {out, "out", 0},
        {mat_a, "mat_a", 1},
        {mat_b, "mat_b", 2},
        {bias_, "bias", 3},
        {scale_b[0], "scale_b", 4}};
    checkAllSameGPU(__func__, targs);
  }

  if (M == 0 || K == 0 || N == 0) {
    if (K == 0) {
      if (bias.has_value()) {
        out.copy_(bias->expand({M, N}));
      } else {
        out.zero_();
      }
    }
    return out;
  }

  auto scale_recipe_b_enum = convert_int_to_enum<ScalingType>(scale_recipe_b);
  ScalingType scaling_choice_b = scale_recipe_b_enum[0];

  switch (scaling_choice_b) {
    case ScalingType::TensorWise:
      _woq_scaled_tensorwise(mat_a, mat_b, scale_b[0], bias, out);
      break;
    case ScalingType::RowWise:
      _woq_scaled_rowwise(mat_a, mat_b, scale_b[0], bias, out);
      break;
    case ScalingType::BlockWise128x128:
      _woq_scaled_block128x128(mat_a, mat_b, scale_b[0], bias, out);
      break;
    case ScalingType::BlockWise1x32:
      if (mat_b.scalar_type() == at::kFloat8_e4m3fn) {
        _woq_scaled_mxfp8(mat_a, mat_b, scale_b[0], bias, out);
      } else if (mat_b.scalar_type() == c10::ScalarType::Float4_e2m1fn_x2) {
        _woq_scaled_mxfp4(mat_a, mat_b, scale_b[0], bias, out);
      } else {
        TORCH_CHECK_VALUE(
            false,
            "woq_scaled_mm: BlockWise1x32 recipe requires mat_b to be "
            "Float8_e4m3fn (MXFP8) or Float4_e2m1fn_x2 (MXFP4, packed), "
            "got ",
            mat_b.scalar_type());
      }
      break;
    default:
      TORCH_CHECK_VALUE(
          false,
          "woq_scaled_mm: unsupported scale_recipe_b ",
          static_cast<int64_t>(scaling_choice_b),
          "; only TensorWise, RowWise, BlockWise128x128 and BlockWise1x32 "
          "are supported");
  }

  return out;
}

TORCH_LIBRARY_IMPL(onednn, XPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("onednn::qlinear_pointwise"),
      TORCH_FN(QLinearOnednnXPU::q_linear_pointwise));
  m.impl(
      TORCH_SELECTIVE_NAME("onednn::qlinear_pointwise.tensor"),
      TORCH_FN(QLinearOnednnXPU::q_linear_pointwise_tensor));
  m.impl(
      TORCH_SELECTIVE_NAME("onednn::qlinear_prepack"),
      TORCH_FN(QLinearOnednnXPU::q_linear_prepack_onednn));
  m.impl(
      TORCH_SELECTIVE_NAME("onednn::qlinear_pointwise.binary"),
      TORCH_FN(QLinearOnednnXPU::q_linear_pointwise_binary));
  m.impl(
      TORCH_SELECTIVE_NAME("onednn::qlinear_pointwise.binary_tensor"),
      TORCH_FN(QLinearOnednnXPU::q_linear_pointwise_binary_tensor));
  m.impl(
      TORCH_SELECTIVE_NAME("onednn::woq_scaled_mm"),
      TORCH_FN(QLinearOnednnXPU::woq_scaled_mm));
}

} // namespace at::native::xpu
