#include <torch/library.h>

#include <ATen/TensorUtils.h>
#include <ATen/native/mkldnn/xpu/detail/oneDNN.h>
#include <ATen/native/mkldnn/xpu/qlinear.h>
#include <c10/core/ScalarType.h>

using namespace at::native::onednn;

namespace at::native::xpu {

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

// Computes a wInt4aInt8 quantized linear: dequantizes `mat_a` (per-token
// quantized activation) and `mat_b` (per-block quantized int4 weight),
// matmuls them, optionally adds `bias`, and returns the result in
// `out_dtype`.
// Arguments:
//   - `mat_a`: the quantized activation, 2D, dtype `torch.int8` or
//     `torch.uint8`, shape [M, K]
//   - `mat_b`: the int4 weight packed as int32, 2D, shape [N, K/8]
//   - `mat_a_scale`: per-token float32 scale for `mat_a`, shape [M]
//   - `mat_b_scales`: per-block float32 scale for `mat_b`, shape
//     [K/group_size, N]
//   - `mat_a_zero_point`: per-token int32 zero-point for `mat_a`, shape [M]
//   - `mat_b_zero_points`: per-block int8 zero-point for `mat_b`, shape
//     [K/group_size, N]
//   - `group_size`: the quantization block size along K for `mat_b`; must
//     be a multiple of 32 and divide K
//   - `out_dtype`: the output dtype, one of float32/bfloat16/float16;
//     defaults to float32
//   - `bias`: an optional bias fused into the matmul, broadcastable to
//     [M, N]
Tensor QLinearOnednnXPU::q_linear_w4a8_int(
    Tensor mat_a,
    Tensor mat_b,
    Tensor mat_a_scale,
    Tensor mat_b_scales,
    Tensor mat_a_zero_point,
    Tensor mat_b_zero_points,
    int64_t group_size,
    std::optional<c10::ScalarType> out_dtype,
    std::optional<Tensor> bias) {
  TORCH_CHECK(
      mat_a.dtype() == c10::kChar || mat_a.dtype() == c10::kByte,
      __func__,
      " : expect mat_a to be int8 or uint8 tensor, but got ",
      mat_a.dtype());
  TORCH_CHECK(
      mat_a.is_contiguous(), __func__, " : expect mat_a to be contiguous.");
  TORCH_CHECK(mat_a.dim() == 2, __func__, " : expect mat_a to be 2D tensor.");

  TORCH_CHECK(
      mat_b.dtype() == c10::kInt,
      __func__,
      " : expect mat_b to be int32 (packed int4) tensor, but got ",
      mat_b.dtype());
  TORCH_CHECK(
      mat_b.is_contiguous(), __func__, " : expect mat_b to be contiguous.");
  TORCH_CHECK(mat_b.dim() == 2, __func__, " : expect mat_b to be 2D tensor.");

  // NOLINTNEXTLINE(*c-array*)
  TensorArg targs[]{
      {mat_a, "mat_a", 1},
      {mat_b, "mat_b", 2},
      {mat_a_scale, "mat_a_scale", 3},
      {mat_b_scales, "mat_b_scales", 4},
      {mat_a_zero_point, "mat_a_zero_point", 5},
      {mat_b_zero_points, "mat_b_zero_points", 6}};
  checkAllSameGPU(__func__, targs);

  int64_t M = mat_a.size(0);
  int64_t N = mat_b.size(0);
  int64_t K = mat_a.size(1);
  TORCH_CHECK(
      group_size > 1 && group_size % 32 == 0,
      __func__,
      " : expect group_size to be multiple of 32 and greater than 1, got ",
      group_size);
  TORCH_CHECK(
      K % group_size == 0,
      __func__,
      " : expect K (",
      K,
      ") to be a multiple of group_size (",
      group_size,
      ")");

  TORCH_CHECK(
      mat_a_scale.dtype() == c10::kFloat,
      __func__,
      " : expect mat_a_scale to be float32 tensor, but got ",
      mat_a_scale.dtype());
  TORCH_CHECK(
      mat_a_zero_point.dtype() == c10::kInt,
      __func__,
      " : expect mat_a_zero_point to be int32 tensor, but got ",
      mat_a_zero_point.dtype());
  TORCH_CHECK(
      mat_a_scale.numel() == M && mat_a_zero_point.numel() == M,
      __func__,
      " : expect mat_a_scale and mat_a_zero_point to carry one value per "
      "activation row (per-token), expected ",
      M,
      " but got ",
      mat_a_scale.numel(),
      " and ",
      mat_a_zero_point.numel());

  TORCH_CHECK(
      mat_b_scales.dtype() == c10::kFloat,
      __func__,
      " : expect mat_b_scales to be float32 tensor, but got ",
      mat_b_scales.dtype());
  TORCH_CHECK(
      mat_b_zero_points.dtype() == c10::kChar,
      __func__,
      " : expect mat_b_zero_points to be int8 tensor, but got ",
      mat_b_zero_points.dtype());
  TORCH_CHECK(
      mat_b_scales.dim() == 2 && mat_b_scales.size(0) == K / group_size &&
          mat_b_scales.size(1) == N,
      __func__,
      " : expect mat_b_scales to be 2D tensor with sizes [",
      K / group_size,
      ", ",
      N,
      "]");
  TORCH_CHECK(
      mat_b_zero_points.dim() == 2 &&
          mat_b_zero_points.size(0) == K / group_size &&
          mat_b_zero_points.size(1) == N,
      __func__,
      " : expect mat_b_zero_points to be 2D tensor with sizes [",
      K / group_size,
      ", ",
      N,
      "]");

  auto dst_dtype = out_dtype.value_or(c10::kFloat);
  TORCH_CHECK(
      dst_dtype == c10::kFloat || dst_dtype == c10::kBFloat16 ||
          dst_dtype == c10::kHalf,
      "qlinear_w4a8_int xpu: output dtype must be one of float32, bfloat16, "
      "or float16, but got ",
      dst_dtype);

  Tensor qout = at::empty({M, N}, mat_a.options().dtype(dst_dtype));

  matmul_int4_impl(
      qout,
      mat_a.contiguous(),
      mat_b.contiguous(),
      mat_a_scale,
      mat_b_scales,
      mat_a_zero_point,
      mat_b_zero_points,
      group_size,
      bias);

  return qout;
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
      TORCH_SELECTIVE_NAME("onednn::qlinear_w4a8_int"),
      TORCH_FN(QLinearOnednnXPU::q_linear_w4a8_int));
}

} // namespace at::native::xpu
