#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Config.h>
#include <ATen/Context.h>
#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/native/Resize.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_scaled_mm_native.h>
#include <ATen/ops/empty.h>
#endif

#if !AT_MKLDNN_ENABLED()
namespace at::native {

Tensor&
_scaled_mm_out_cpu(const Tensor& mat1, const Tensor& mat2,
          const Tensor& scale_a,
          const Tensor& scale_b,
          const std::optional<at::Tensor>& bias,
          const std::optional<at::Tensor>& scale_result,
          std::optional<c10::ScalarType> out_dtype,
          bool use_fast_accum,
          Tensor& out) {
  TORCH_INTERNAL_ASSERT(false, __func__, ": ATen not compiled with MKLDNN support");
}

Tensor
_scaled_mm_cpu(const Tensor& mat_a, const Tensor& mat_b,
          const Tensor& scale_a,
          const Tensor& scale_b,
          const std::optional<at::Tensor>& bias,
          const std::optional<at::Tensor>& scale_result,
          std::optional<c10::ScalarType> out_dtype,
          bool use_fast_accum) {
  TORCH_INTERNAL_ASSERT(false, __func__, ": ATen not compiled with MKLDNN support");
}
}

#else // AT_MKLDNN_ENABLED
namespace at::native {

Tensor&
_scaled_mm_out_cpu(const Tensor& mat1, const Tensor& mat2,
          const Tensor& scale_a,
          const Tensor& scale_b,
          const std::optional<at::Tensor>& bias,
          const std::optional<at::Tensor>& scale_result,
          std::optional<c10::ScalarType> out_dtype,
          bool use_fast_accum,
          Tensor& out) {
  TORCH_CHECK(mat1.dim() == 2, "mat1 must be a matrix");
  TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix");
  TORCH_CHECK(
      mat1.sizes()[1] == mat2.sizes()[0], "mat1 and mat2 shapes cannot be multiplied (",
      mat1.sizes()[0], "x", mat1.sizes()[1], " and ", mat2.sizes()[0], "x", mat2.sizes()[1], ")");

  TORCH_INTERNAL_ASSERT((scale_a.numel() == 1 && scale_b.numel() == 1), "Now _scaled_mm only supports per-tensor scaling for CPU backend.");

  // TORCH_CHECK(!scale_result || (scale_result->numel() == 1 && scale_result->scalar_type() == kFloat),
  //      "scale_result must be a float scalar");
  TORCH_CHECK(!bias || bias->numel() == mat2.sizes()[1], "Bias must be size ", mat2.sizes()[1],
       " but got ", bias->numel());

  // Check types
  TORCH_CHECK(!out_dtype || *out_dtype == out.scalar_type(), "out_dtype must match output matrix type");
  TORCH_CHECK(isFloat8Type(mat1.scalar_type()), "Expected mat1 to be Float8 matrix got ", mat1.scalar_type());
  TORCH_CHECK(isFloat8Type(mat2.scalar_type()), "Expected mat2 to be Float8 matrix got ", mat2.scalar_type());
  if (bias) {
    TORCH_CHECK(out.scalar_type() != kFloat, "Bias is not supported when out_dtype is set to Float32");
    TORCH_CHECK(bias->scalar_type() == ScalarType::BFloat16 || bias->scalar_type() == ScalarType::Half,
         "Bias must be either Half or BFloat16, but got ", bias->scalar_type());
    TORCH_CHECK((out.scalar_type() != kFloat && out.scalar_type() != ScalarType::BFloat16) ||
          bias->scalar_type() == ScalarType::BFloat16,
          "Bias must be BFloat16 to compute ", out.scalar_type(), " output, but got ", bias->scalar_type());
    TORCH_CHECK(out.scalar_type() != ScalarType::Half || bias->scalar_type() == ScalarType::Half,
          "Bias must be Float16 to compute ", out.scalar_type(), " output, but got ", bias->scalar_type());
  }

  // Validation checks have passed lets resize the output to actual size
  IntArrayRef mat1_sizes = mat1.sizes();
  IntArrayRef mat2_sizes = mat2.sizes();
  at::native::resize_output(out, {mat1_sizes[0], mat2_sizes[1]});

  float input_scale = scale_a.item<float>();
  float weight_scale = scale_b.item<float>();
  auto mat2_c = mat2.contiguous();
  auto src = at::native::itensor_view_from_dense(mat1);
  auto weight_t = at::native::itensor_view_from_dense(mat2_c);
  bool with_bias = bias.has_value();
  int64_t K = mat1_sizes[1], M = mat1_sizes[0],
          N = mat2_sizes[1];

  std::vector<int64_t> src_dims = {M, K};
  std::vector<int64_t> weight_dims = {K, N};
  std::vector<int64_t> dst_dims = {M, N};

  ideep::tensor dst = at::native::itensor_view_from_dense(out);
  auto src_desc = ideep::tensor::desc(
      src_dims,
      get_mkldnn_dtype(mat1.scalar_type()),
      ideep::format_tag::any);
  auto weights_desc = ideep::tensor::desc(
      weight_dims,
      get_mkldnn_dtype(mat2.scalar_type()),
      ideep::format_tag::any);
  auto dst_desc = ideep::tensor::desc(
      dst_dims,
      get_mkldnn_dtype(out.scalar_type()),
      ideep::format_tag::any);
  ideep::tensor onednn_bias;
  if (with_bias) {
    auto bias_value = bias.value();
    if (bias_value.dim() == 1) {
      auto b_reshape = bias_value.reshape({1, bias_value.size(0)});
      onednn_bias = at::native::itensor_view_from_dense(b_reshape);
    } else {
      onednn_bias = at::native::itensor_view_from_dense(bias_value);
    }
  }
  auto bias_desc = ideep::tensor::desc();
  if (with_bias) {
    bias_desc = ideep::tensor::desc(onednn_bias.get_dims(),
                        get_mkldnn_dtype(bias.value().scalar_type()),
                        ideep::format_tag::any);
  }
  auto op_attr = ideep::attr_t();
  if (input_scale != 1.0f) {
    op_attr.set_scales_mask(DNNL_ARG_SRC, 0);
  }
  if (weight_scale != 1.0f) {
    op_attr.set_scales_mask(DNNL_ARG_WEIGHTS, 0);
  }

  op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
  auto engine = ideep::engine::cpu_engine();
  // TODO: Remove this try/catch when oneDNN provides API to notify
  // framework whether current platform can run FP8 primitives.
  dnnl::matmul::primitive_desc primitive_desc;
  try {
    primitive_desc = with_bias
        ? dnnl::matmul::primitive_desc(
              engine, src_desc, weights_desc, bias_desc, dst_desc, op_attr)
        : dnnl::matmul::primitive_desc(
              engine, src_desc, weights_desc, dst_desc, op_attr);
  } catch (dnnl::error& e) {
    if (e.status == dnnl_unimplemented)
      throw std::runtime_error("Running FP8 on not supported platform.");
    // on any other error just re-throw
    throw;
  }
  auto primitive = dnnl::matmul(primitive_desc);

  // Prepare args and execute primitive
  ideep::tensor scratchpad(primitive_desc.scratchpad_desc());
  ideep::exec_args args;
  args.insert({DNNL_ARG_SRC, src});
  args.insert({DNNL_ARG_WEIGHTS, weight_t});
  args.insert({DNNL_ARG_DST, dst});
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad});
  if (with_bias) {
    args.insert({DNNL_ARG_BIAS, onednn_bias});
  }
  ideep::tensor src_scales_t = ideep::tensor(ideep::scale_t(1, input_scale));
  ideep::tensor wei_scales_t = ideep::tensor(ideep::scale_t(1, weight_scale));

  if (input_scale != 1.0f) {
    args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, src_scales_t});
  }
  args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, wei_scales_t});

  primitive.execute(ideep::stream::default_stream(), args);
  return out;
}

Tensor
_scaled_mm_cpu(const Tensor& mat_a, const Tensor& mat_b,
          const Tensor& scale_a,
          const Tensor& scale_b,
          const std::optional<at::Tensor>& bias,
          const std::optional<at::Tensor>& scale_result,
          std::optional<c10::ScalarType> out_dtype,
          bool use_fast_accum) {
  const auto out_dtype_ = out_dtype.value_or(mat_a.scalar_type());
  Tensor out = at::empty({0}, mat_a.options().dtype(out_dtype_));
  return at::native::_scaled_mm_out_cpu(mat_a, mat_b, scale_a, scale_b, bias, scale_result, out_dtype, use_fast_accum, out);
}

} // namespace at::native

#endif // AT_MKLDNN_ENABLED
