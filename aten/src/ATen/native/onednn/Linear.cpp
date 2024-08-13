#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Config.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

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

namespace at {
namespace native {

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

} // namespace native
} // namespace at

#else // AT_MKLDNN_ENABLED

#include <ATen/native/onednn/MKLDNNCommon.h>
#include <ATen/native/onednn/Utils.h>

namespace at {
namespace native {

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

static Tensor mkldnn_linear_pointwise(
    const Tensor& input_t,
    const Tensor& weight_t,
    const std::optional<Tensor>& bias_opt,
    c10::string_view attr,
    torch::List<std::optional<at::Scalar>> scalars,
    std::optional<c10::string_view> algorithm) {
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

  if (mkldnn_bias.has_value()) {
    ideep::inner_product_forward::compute</*reorder_src=*/false, /*reorder_weight=*/false>(
        mkldnn_input,
        w,
        mkldnn_bias.value(),
        mkldnn_output,
        op_attr);
  } else {
    ideep::inner_product_forward::compute</*reorder_src=*/false, /*reorder_weight=*/false>(
        mkldnn_input,
        w,
        mkldnn_output,
        op_attr);
  }

  if (dim != 2) {
    output = output.reshape(output_size);
  }

  return output;
}

static Tensor mkldnn_linear_pointwise_binary(
    const Tensor& input_t,
    const Tensor& other_t,
    const Tensor& weight_t,
    const std::optional<Tensor>& bias_opt,
    c10::string_view attr) {
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
  }

  TORCH_CHECK(
      output.sizes() == other_reshaped.sizes(),
      "linear_binary_run expects the size of output and other tensor to be the same");

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

  if (mkldnn_bias.has_value()) {
    ideep::inner_product_forward::compute_binary</*reorder_src=*/false, /*reorder_weight=*/false>(
        mkldnn_input,
        mkldnn_other,
        w,
        mkldnn_bias.value(),
        mkldnn_output,
        op_attr);
  } else {
    ideep::inner_product_forward::compute_binary</*reorder_src=*/false, /*reorder_weight=*/false>(
        mkldnn_input, mkldnn_other, w, mkldnn_output, op_attr);
  }

  if (dim != 2) {
    output = output.reshape(output_size);
  }

  return output;
}

#if AT_MKL_ENABLED()
#include <mkl.h>

static Tensor mkl_linear(
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

} // namespace native
} // namespace at

#endif // AT_MKLDNN_ENABLED
