#include <torch/library.h>

#include <ATen/native/mkldnn/xpu/detail/oneDNN.h>
#include <c10/core/ScalarType.h>

using namespace at::native::onednn;

namespace at::native::xpu {

static inline c10::ScalarType qlinear_decide_out_dtype(
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

static Tensor q_linear_pointwise(
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

static Tensor q_linear_pointwise_tensor(
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

static Tensor q_linear_pointwise_binary(
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

static Tensor q_linear_pointwise_binary_tensor(
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

static at::Tensor q_linear_prepack_onednn(
    at::Tensor weight,
    std::optional<torch::List<int64_t>> input_shape) {
  at::Tensor weight_transposed = weight.transpose(0, 1);
  return weight_transposed;
}

TORCH_LIBRARY_IMPL(onednn, XPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("onednn::qlinear_pointwise"),
      TORCH_FN(q_linear_pointwise));
  m.impl(
      TORCH_SELECTIVE_NAME("onednn::qlinear_pointwise.tensor"),
      TORCH_FN(q_linear_pointwise_tensor));
  m.impl(
      TORCH_SELECTIVE_NAME("onednn::qlinear_prepack"),
      TORCH_FN(q_linear_prepack_onednn));
  m.impl(
      TORCH_SELECTIVE_NAME("onednn::qlinear_pointwise.binary"),
      TORCH_FN(q_linear_pointwise_binary));
  m.impl(
      TORCH_SELECTIVE_NAME("onednn::qlinear_pointwise.binary_tensor"),
      TORCH_FN(q_linear_pointwise_binary_tensor));
}

} // namespace at::native::xpu
