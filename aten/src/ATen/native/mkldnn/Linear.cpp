#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Config.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>
#include <ATen/native/mkldnn/Linear.h>
#include <ATen/native/Resize.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_to_dense_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/linear.h>
#include <ATen/ops/mkldnn_linear_backward_input.h>
#include <ATen/ops/mkldnn_linear_backward_input_native.h>
#include <ATen/ops/mkldnn_linear_backward_native.h>
#include <ATen/ops/mkldnn_linear_backward_weights.h>
#include <ATen/ops/mkldnn_linear_backward_weights_native.h>
#include <ATen/ops/mkldnn_linear_native.h>
#endif

#if !AT_MKLDNN_ENABLED()


namespace at::native {

Tensor mkldnn_linear(
    const Tensor& self,
    const Tensor& weight, const std::optional<Tensor>& bias_opt) {
  TORCH_CHECK(false, "mkldnn_linear: ATen not compiled with MKLDNN support");
}
Tensor mkldnn_linear_backward_input(
    IntArrayRef input_size, const Tensor& grad_output, const Tensor& weight) {
  TORCH_CHECK(false, "mkldnn_linear_backward_input: ATen not compiled with MKLDNN support");
}

std::tuple<Tensor, Tensor> mkldnn_linear_backward_weights(
    const Tensor& grad_output, const Tensor& input, const Tensor& weight, bool bias_defined) {
  TORCH_CHECK(false, "mkldnn_linear_backward_weights: ATen not compiled with MKLDNN support");
}

std::tuple<Tensor, Tensor, Tensor> mkldnn_linear_backward(
    const Tensor& input, const Tensor& grad_output_t,
    const Tensor& weight, std::array<bool,3> output_mask) {
  TORCH_CHECK(false, "mkldnn_linear_backward: ATen not compiled with MKLDNN support");
}

Tensor&
mkldnn_scaled_mm(const Tensor& mat1, const Tensor& mat2,
          const Tensor& scale_a,
          const Tensor& scale_b,
          const std::optional<at::Tensor>& bias,
          const std::optional<at::Tensor>& scale_result,
          std::optional<c10::ScalarType> out_dtype,
          bool use_fast_accum,
          Tensor& out) {
  TORCH_INTERNAL_ASSERT(false, "mkldnn_scaled_mm: ATen not compiled with MKLDNN support");
}

} // namespace at::native

#else // AT_MKLDNN_ENABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/native/mkldnn/Utils.h>

namespace at::native {

static bool use_mkldnn_bf32_linear() {
  return at::globalContext().float32Precision(at::Float32Backend::MKLDNN, at::Float32Op::MATMUL) == at::Float32Precision::BF16 &&
      mkldnn_bf16_device_check();
}

static bool use_mkldnn_tf32_linear() {
  return at::globalContext().float32Precision(at::Float32Backend::MKLDNN, at::Float32Op::MATMUL) == at::Float32Precision::TF32 &&
      cpuinfo_has_x86_amx_fp16();
}

Tensor mkldnn_linear(
    const Tensor& self,
    const Tensor& weight_t, const std::optional<Tensor>& bias_opt) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  const int64_t dim = self.dim();
  TORCH_CHECK(
      self.dim() != 0,
      "mkldnn_linear: input needs to has dim at least 1, input dim ",
      self.dim());
  TORCH_CHECK(self.is_mkldnn(),
      "mkldnn_linear: input needs to be mkldnn layout");
  if (self.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(mkldnn_bf16_device_check(),
        "mkldnn_linear: bf16 path needs the cpu support avx_ne_convert or avx512bw, avx512vl and avx512dq");
  } else if (self.scalar_type() == ScalarType::Half) {
    TORCH_CHECK(mkldnn_fp16_device_check(),
        "mkldnn_linear: fp16 path needs the cpu support avx_ne_convert or avx512_fp16");
  }

  // reshape first if input dim != 2 and the reshape will cost a memory copy.
  auto self_reshaped =
      dim == 2 ? self : self.reshape({-1, self.size(self.dim() - 1)});

  const ideep::tensor x = itensor_from_mkldnn(self_reshaped);
  // weight_t can be a mkldnn tensor or dense tensor.
  const Tensor weight = (weight_t.is_mkldnn() || weight_t.is_contiguous()) ? weight_t : weight_t.contiguous();
  const ideep::tensor w = itensor_from_tensor(weight);

  ideep::tensor y;
  if (bias.defined()) {
    const ideep::tensor b = itensor_from_tensor(bias);
    ideep::inner_product_forward::compute(x, w, b, y);
  } else {
    ideep::inner_product_forward::compute(x, w, y);
  }

  auto input_size = self.sizes();
  std::vector<int64_t> output_size(input_size.begin(), input_size.end() - 1);
  output_size.push_back(weight.size(0));

  if (self.dim() != 2) {
    return new_with_itensor_mkldnn(std::move(y), optTypeMetaToScalarType(self.options().dtype_opt()),
                                   self.options().device_opt()).reshape(output_size);
  }
  return new_with_itensor_mkldnn(std::move(y), optTypeMetaToScalarType(self.options().dtype_opt()),
                                 self.options().device_opt());
}


Tensor mkldnn_linear_backward_input(
    IntArrayRef input_size, const Tensor& grad_output, const Tensor& weight_t){
  TORCH_CHECK(grad_output.is_mkldnn(),
      "mkldnn_linear_backward: grad_output needs to be mkldnn layout");
  TORCH_CHECK(weight_t.device().is_cpu() && weight_t.scalar_type() == kFloat,
      "mkldnn_linear_backward: weight_t needs to be a dense tensor");
  auto grad_output_reshaped = grad_output.dim() > 2 ?
    grad_output.reshape({-1, grad_output.size(grad_output.dim() - 1)}) : grad_output;

  ideep::tensor& grady = itensor_from_mkldnn(grad_output_reshaped);
  // weight_t always dense tensor for training.
  const Tensor weight = weight_t.is_contiguous() ? weight_t : weight_t.contiguous();
  const ideep::tensor w = itensor_view_from_dense(weight);

  std::vector<int64_t> input_reshaped_size;
  input_reshaped_size.push_back(grad_output_reshaped.size(0));
  input_reshaped_size.push_back(weight.size(1));

  ideep::tensor gradx;
  ideep::inner_product_backward_data::compute(
    grady, w, {input_reshaped_size.begin(), input_reshaped_size.end()}, gradx);

  if (input_size.size() > 2) {
    return new_with_itensor_mkldnn(std::move(gradx), optTypeMetaToScalarType(grad_output.options().dtype_opt()),
                                   grad_output.options().device_opt()).reshape(input_size);
  }
  return new_with_itensor_mkldnn(std::move(gradx), optTypeMetaToScalarType(grad_output.options().dtype_opt()),
                                 grad_output.options().device_opt());
}

std::tuple<Tensor, Tensor> mkldnn_linear_backward_weights(
    const Tensor& grad_output, const Tensor& input, const Tensor& weight, bool bias_defined) {
  TORCH_CHECK(grad_output.is_mkldnn() && input.is_mkldnn(),
      "mkldnn_linear_backward: grad_output and input needs to be mkldnn layout");
  TORCH_CHECK(weight.device().is_cpu() && weight.scalar_type() == kFloat,
      "mkldnn_linear_backward: weight needs to be a dense tensor");

  auto grad_output_reshaped = grad_output.dim() > 2 ?
    grad_output.reshape({-1, grad_output.size(grad_output.dim() - 1)}) : grad_output;
  auto input_reshaped = input.dim() > 2 ? input.reshape({-1, input.size(input.dim() - 1)}) : input;

  ideep::tensor& grady = itensor_from_mkldnn(grad_output_reshaped);
  ideep::tensor& x = itensor_from_mkldnn(input_reshaped);
  ideep::tensor gradw, gradb;
  if (bias_defined) {
    ideep::inner_product_backward_weights::compute(x, grady, gradw, gradb);
  } else {
    ideep::inner_product_backward_weights::compute(x, grady, gradw);
  }

  return std::tuple<Tensor, Tensor>{
    mkldnn_to_dense(new_with_itensor_mkldnn(std::move(gradw),
                    optTypeMetaToScalarType(weight.options().dtype_opt()),
                    weight.options().device_opt())),
    mkldnn_to_dense(new_with_itensor_mkldnn(std::move(gradb),
                    optTypeMetaToScalarType(weight.options().dtype_opt()),
                    weight.options().device_opt()))};
}

std::tuple<Tensor, Tensor, Tensor> mkldnn_linear_backward(
    const Tensor& input, const Tensor& grad_output,
    const Tensor& weight, std::array<bool,3> output_mask) {
  Tensor grad_input, grad_weight, grad_bias;
  if (output_mask[0]) {
    grad_input = at::mkldnn_linear_backward_input(input.sizes(), grad_output, weight);
  }
  if (output_mask[1] || output_mask[2]) {
    std::tie(grad_weight, grad_bias) = at::mkldnn_linear_backward_weights(grad_output, input, weight, output_mask[2]);
  }
  return std::tuple<Tensor, Tensor, Tensor>{grad_input, grad_weight, grad_bias};
}

Tensor mkldnn_linear_pointwise(
    const Tensor& input_t,
    const Tensor& weight_t,
    const std::optional<Tensor>& bias_opt,
    std::string_view attr,
    c10::List<std::optional<at::Scalar>> scalars,
    std::optional<std::string_view> algorithm) {
  auto aprop_kind = ideep::prop_kind::forward;
  bool maybe_backward = GradMode::is_enabled() &&
      (input_t.requires_grad() || weight_t.requires_grad() ||
       (bias_opt.has_value() && bias_opt->defined() &&
        bias_opt->requires_grad()));
  if (!maybe_backward) {
    aprop_kind = ideep::prop_kind::forward_inference;
  }
  auto input = input_t.contiguous();
  auto input_size = input.sizes();

  // Make sure input has default contiguous strides if it's contiguous tensors for better performance.
  input = may_convert_to_default_contiguous_strides(input);

  const int64_t dim = input.dim();
  auto input_reshaped =
      dim == 2 ? input : input.reshape({-1, input.size(input.dim() - 1)});

  std::vector<int64_t> output_size(input_size.begin(), input_size.end() - 1);
  output_size.push_back(weight_t.size(0));
  auto output = at::empty(output_size, input.options());
  if (output.sym_numel() == 0) {
    return output;
  }
  if (dim != 2) {
    std::vector<int64_t> output_size_reshaped = {input_reshaped.size(0),
                                                 weight_t.size(0)};
    output = output.reshape(output_size_reshaped);
  }

  c10::impl::ExcludeDispatchKeyGuard edkg(c10::autograd_dispatch_keyset);
  ideep::tensor mkldnn_output = itensor_from_tensor(output);

  c10::MaybeOwned<Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  const ideep::tensor mkldnn_input = itensor_view_from_dense(input_reshaped);

  std::optional<ideep::tensor> mkldnn_bias{std::nullopt};
  if (bias.defined()) {
    mkldnn_bias = itensor_from_tensor(bias);
  }
  const ideep::tensor w = itensor_from_tensor(weight_t);

  ideep::attr_t op_attr = ideep::attr_t();
  if (attr != "none") {
    auto it = fusion_unary_attr_map().find(attr);
    TORCH_CHECK(
        it != fusion_unary_attr_map().end(), "Fusion behavior undefined.");
    op_attr = it->second(scalars, algorithm);
  }
  if (use_mkldnn_bf32_linear() && input_t.scalar_type() == at::kFloat){
    op_attr.set_fpmath_mode(dnnl_fpmath_mode_bf16);
  }
  if (use_mkldnn_tf32_linear() && input_t.scalar_type() == at::kFloat){
    op_attr.set_fpmath_mode(dnnl_fpmath_mode_tf32);
  }
  if (mkldnn_bias.has_value()) {
    ideep::inner_product_forward::compute</*reorder_src=*/false, /*reorder_weight=*/false>(
        mkldnn_input,
        w,
        mkldnn_bias.value(),
        mkldnn_output,
        op_attr,
        aprop_kind);
  } else {
    ideep::inner_product_forward::compute</*reorder_src=*/false, /*reorder_weight=*/false>(
        mkldnn_input,
        w,
        mkldnn_output,
        op_attr,
        aprop_kind);
  }

  if (dim != 2) {
    output = output.reshape(output_size);
  }

  return output;
}

Tensor mkldnn_linear_pointwise_binary(
    const Tensor& input_t,
    const Tensor& other_t,
    const Tensor& weight_t,
    const std::optional<Tensor>& bias_opt,
    std::string_view attr) {
  c10::MaybeOwned<Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;
  // Make sure inputs have same type(device, layout, dtype), device is cpu and
  // dtype is float or bfloat16.
  check_mkldnn_binary_fusion_inputs(input_t, other_t, weight_t, bias);

  auto input = input_t.contiguous();
  // Make sure input has default contiguous strides if it's contiguous tensors for better performance.
  input = may_convert_to_default_contiguous_strides(input);

  auto it_binary = fusion_binary_alg_map().find(attr);
  TORCH_CHECK(
      it_binary != fusion_binary_alg_map().end(), "Fusion behavior undefined.");

  auto input_size = input.sizes();

  const int64_t dim = input.dim();
  auto input_reshaped =
      dim == 2 ? input : input.reshape({-1, input.size(input.dim() - 1)});

  std::vector<int64_t> output_size(input_size.begin(), input_size.end() - 1);
  output_size.push_back(weight_t.size(0));
  auto output = at::empty(output_size, input.options());
  if (output.sym_numel() == 0) {
    return output;
  }
  auto other_reshaped = other_t.contiguous();
  other_reshaped = may_convert_to_default_contiguous_strides(other_reshaped);

  if (dim != 2) {
    std::vector<int64_t> output_size_reshaped = {
        input_reshaped.size(0), weight_t.size(0)};
    output = output.reshape(output_size_reshaped);
    other_reshaped = other_reshaped.reshape(output_size_reshaped);
    TORCH_CHECK(
        output.sizes() == other_reshaped.sizes(),
        "linear_binary_run expects the size of output and other tensor to be the same");
  } else {
    TORCH_CHECK(
        output.dim() == other_reshaped.dim(),
        "linear_binary_run expects the dimension of output and other tensor to be the same");
  }

  c10::impl::ExcludeDispatchKeyGuard edkg(c10::autograd_dispatch_keyset);
  ideep::tensor mkldnn_output = itensor_from_tensor(output);
  const ideep::tensor mkldnn_other = itensor_from_tensor(other_reshaped);
  const ideep::tensor mkldnn_input = itensor_view_from_dense(input_reshaped);

  std::optional<ideep::tensor> mkldnn_bias{std::nullopt};
  if (bias.defined()) {
    mkldnn_bias = itensor_from_tensor(bias);
  }
  const ideep::tensor w = itensor_from_tensor(weight_t);

  auto other_desc = mkldnn_other.get_desc();
  auto op_attr = ideep::attr_t::fuse_binary(it_binary->second, other_desc);
  auto aprop_kind = ideep::prop_kind::forward_inference;

  if (use_mkldnn_bf32_linear() && input_t.scalar_type() == at::kFloat){
    op_attr.set_fpmath_mode(dnnl_fpmath_mode_bf16);
  }

  if (use_mkldnn_tf32_linear() && input_t.scalar_type() == at::kFloat){
    op_attr.set_fpmath_mode(dnnl_fpmath_mode_tf32);
  }

  if (mkldnn_bias.has_value()) {
    ideep::inner_product_forward::compute_binary</*reorder_src=*/false, /*reorder_weight=*/false>(
        mkldnn_input,
        mkldnn_other,
        w,
        mkldnn_bias.value(),
        mkldnn_output,
        op_attr,
        aprop_kind);
  } else {
    ideep::inner_product_forward::compute_binary</*reorder_src=*/false, /*reorder_weight=*/false>(
        mkldnn_input, mkldnn_other, w, mkldnn_output, op_attr, aprop_kind);
  }

  if (dim != 2) {
    output = output.reshape(output_size);
  }

  return output;
}

#if AT_MKL_ENABLED()
#include <mkl.h>

Tensor mkl_linear(
    const Tensor& self,
    const Tensor& mkl_weight_t,
    const Tensor& origin_weight_t,
    const std::optional<Tensor>& bias_opt,
    const int64_t prepack_batch_size) {
  c10::MaybeOwned<Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;
  TORCH_CHECK(
      self.options().type_equal(origin_weight_t.options()),
      "Input type (",
      self.toString(),
      ") and weight type (",
      origin_weight_t.toString(),
      ") should be the same");
  TORCH_CHECK(
      !bias.defined() || (self.options().type_equal(bias.options())),
      "Input type (",
      self.toString(),
      ") and bias type (",
      bias.toString(),
      ") should be the same");
  TORCH_CHECK(
      mkl_weight_t.scalar_type() == origin_weight_t.scalar_type() &&
          origin_weight_t.scalar_type() == kFloat,
      "mkl_linear: weight dtype should be float");

  c10::impl::ExcludeDispatchKeyGuard edkg(c10::autograd_dispatch_keyset);
  auto input_size = self.sizes();
  std::vector<int64_t> output_size(input_size.begin(), input_size.end() - 1);
  output_size.push_back(origin_weight_t.size(0));
  auto output = at::empty(output_size, self.options());
  if (self.sym_numel() == 0) {
    // avoid to call self.numel() / 0 when self.size(self.dim() - 1)==0.
    return output.fill_(0);
  }
  if (output.sym_numel() == 0) {
    return output;
  }
  int64_t M = self.numel() / self.size(self.dim() - 1);
  if (M == prepack_batch_size && mkl_weight_t.is_mkldnn()) {
    auto self_ = self.is_contiguous() ? self : self.contiguous();
    auto K = origin_weight_t.size(1);
    auto N = origin_weight_t.size(0);
    const ideep::tensor& w = itensor_from_mkldnn(mkl_weight_t);
    auto in_ptr = self_.data_ptr<float>();
    auto weight_ptr = (float*)(w.get_data_handle());
    auto out_ptr = output.data_ptr<float>();
    if (bias.defined()) {
      auto bias_ = bias.is_contiguous() ? bias : bias.contiguous();
      auto bias_ptr = bias_.data_ptr<float>();
      at::parallel_for(0, M, 1, [&](int64_t begin, int64_t end) {
        for (const auto d : c10::irange(begin, end)) {
          memcpy(out_ptr + d * N, bias_ptr, sizeof(float) * N);
        }
      });
    }
    cblas_sgemm_compute(
        CblasRowMajor,
        CblasNoTrans,
        CblasPacked,
        M,
        N,
        K,
        in_ptr,
        K,
        weight_ptr,
        K,
        bias.defined() ? 1.f : 0.f,
        out_ptr,
        N);
  } else {
    output = at::linear_out(output, self, origin_weight_t, bias_opt);
  }
  return output;
}

TORCH_LIBRARY_IMPL(mkl, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("mkl::_mkl_linear"), TORCH_FN(mkl_linear));
}

TORCH_LIBRARY_IMPL(mkl, MkldnnCPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("mkl::_mkl_linear"), TORCH_FN(mkl_linear));
}

#endif// AT_MKL_ENABLED

TORCH_LIBRARY_IMPL(mkldnn, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn::_linear_pointwise"),
      TORCH_FN(mkldnn_linear_pointwise));
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn::_linear_pointwise.binary"),
      TORCH_FN(mkldnn_linear_pointwise_binary));
}

TORCH_LIBRARY_IMPL(mkldnn, MkldnnCPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn::_linear_pointwise"),
      TORCH_FN(mkldnn_linear_pointwise));
  m.impl(
      TORCH_SELECTIVE_NAME("mkldnn::_linear_pointwise.binary"),
      TORCH_FN(mkldnn_linear_pointwise_binary));
}

Tensor&
mkldnn_scaled_mm(const Tensor& mat1, const Tensor& mat2,
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
  TORCH_CHECK(
      !scale_result ||
          (scale_result->numel() == 1 && scale_result->scalar_type() == kFloat),
      "scale_result must be a float scalar");
  TORCH_CHECK(!bias || bias->numel() == mat2.sizes()[1], "Bias must be size ", mat2.sizes()[1],
       " but got ", bias->numel());

  // Check types
  TORCH_CHECK(!out_dtype || *out_dtype == out.scalar_type(), "out_dtype must match output matrix type");
  TORCH_CHECK(isFloat8Type(mat1.scalar_type()), "Expected mat1 to be Float8 matrix got ", mat1.scalar_type());
  TORCH_CHECK(isFloat8Type(mat2.scalar_type()), "Expected mat2 to be Float8 matrix got ", mat2.scalar_type());

  // Validation checks have passed lets resize the output to actual size
  auto mat1_c = mat1.contiguous();
  auto mat2_c = mat2.contiguous();
  IntArrayRef mat1_sizes = mat1_c.sizes();
  IntArrayRef mat2_sizes = mat2_c.sizes();
  at::native::resize_output(out, {mat1_sizes[0], mat2_sizes[1]});

  float input_scale = scale_a.item<float>();
  float weight_scale = scale_b.item<float>();
  float output_scale = float(1.0);
  if (scale_result.has_value() &&
      (*out_dtype == ScalarType::Float8_e4m3fn ||
       *out_dtype == ScalarType::Float8_e5m2)) {
    output_scale = scale_result.value().item<float>();
  }
  auto src = at::native::itensor_view_from_dense(mat1_c);
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
  if (output_scale != 1.0f) {
    op_attr.set_scales_mask(DNNL_ARG_DST, 0);
  }

  op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
  auto engine = ideep::engine::cpu_engine();
  dnnl::matmul::primitive_desc primitive_desc = with_bias
      ? dnnl::matmul::primitive_desc(
            engine, src_desc, weights_desc, bias_desc, dst_desc, op_attr)
      : dnnl::matmul::primitive_desc(
            engine, src_desc, weights_desc, dst_desc, op_attr);
  auto expected_weight = weight_t.reorder_if_differ_in(primitive_desc.weights_desc());
  auto primitive = dnnl::matmul(primitive_desc);

  // Prepare args and execute primitive
  ideep::tensor scratchpad(primitive_desc.scratchpad_desc());
  ideep::exec_args args;
  args.insert({DNNL_ARG_SRC, src});
  args.insert({DNNL_ARG_WEIGHTS, expected_weight});
  args.insert({DNNL_ARG_DST, dst});
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad});
  if (with_bias) {
    args.insert({DNNL_ARG_BIAS, onednn_bias});
  }
  ideep::tensor src_scales_t = ideep::tensor(ideep::scale_t(1, input_scale));
  ideep::tensor wei_scales_t = ideep::tensor(ideep::scale_t(1, weight_scale));
  ideep::tensor dst_scales_t = ideep::tensor(ideep::scale_t(1, output_scale));

  if (input_scale != 1.0f) {
    args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, src_scales_t});
  }
  args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, wei_scales_t});
  if (output_scale != 1.0f) {
    args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, dst_scales_t});
  }

  primitive.execute(ideep::stream::default_stream(), args);
  return out;
}

} // namespace at

#endif // AT_MKLDNN_ENABLED
