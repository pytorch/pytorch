#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Config.h>
#include <tuple>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_batch_norm_with_update_native.h>
#include <ATen/ops/batch_norm_backward_native.h>
#include <ATen/ops/_native_batch_norm_legit_native.h>
#include <ATen/ops/_to_dense_native.h>
#include <ATen/ops/empty_native.h>
#include <ATen/ops/native_batch_norm_backward_native.h>
#include <ATen/ops/native_batch_norm_native.h>
#endif
#include <ATen/native/onednn/Utils.h>

#if !AT_ONEDNN_ENABLED()

namespace at {
namespace native {

std::tuple<Tensor, Tensor, Tensor> mkldnn_batch_norm(
    const Tensor& self, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt, const std::optional<Tensor>& running_mean_opt, const std::optional<Tensor>& running_var_opt,
    bool train,
    double momentum,
    double eps) {
  TORCH_CHECK(false, "mkldnn_batch_norm: ATen not compiled with MKLDNN support");
}

std::tuple<Tensor, Tensor, Tensor> mkldnn_batch_norm_backward(
    const Tensor& grad_output,
    const Tensor& input, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& running_mean_opt, const std::optional<Tensor>& running_var_opt, const std::optional<Tensor>& save_mean_opt, const std::optional<Tensor>& save_invstd_opt,
    bool train,
    double eps,
    std::array<bool,3> grad_input_mask) {
  TORCH_CHECK(false, "mkldnn_batch_norm_backward: ATen not compiled with MKLDNN support");
}

std::tuple<Tensor, Tensor, Tensor> onednn_layer_norm_last_index_weight_bias_f32(
    const Tensor& input,
    IntArrayRef normalized_shape, const Tensor& weight, const Tensor& bias,
    double eps, bool inplace) {
  TORCH_CHECK(false, "onednn_layer_norm_last_index_weight_bias_f32: ATen not compiled with MKLDNN support");
}

std::tuple<Tensor, Tensor, Tensor> _mkldnn_batch_norm_legit(
    const Tensor& input, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt, Tensor& running_mean, Tensor& running_var,
    bool train,
    double momentum,
    double eps) {
  TORCH_CHECK(false, "_mkldnn_batch_norm_legit: ATen not compiled with MKLDNN support");
}


std::tuple<Tensor, Tensor, Tensor> _mkldnn_batch_norm_legit_no_stats(
    const Tensor& input, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt,
    bool train,
    double momentum,
    double eps) {
  TORCH_CHECK(false, "_mkldnn_batch_norm_legit_no_stats: ATen not compiled with MKLDNN support");
}

std::tuple<Tensor, Tensor, Tensor, Tensor> _batch_norm_with_update_mkldnn(
    const Tensor& input, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt,
    Tensor& running_mean, Tensor& running_var, double momentum, double eps) {
  TORCH_CHECK(false, "_batch_norm_with_update_mkldnn: ATen not compiled with MKLDNN support");
}

std::tuple<Tensor, Tensor, Tensor> _new_batch_norm_backward_mkldnn(
    const Tensor& grad_output, const Tensor& input, const Tensor& weight,
    const std::optional<Tensor>& running_mean_opt, const std::optional<Tensor>& running_var_opt,
    const std::optional<Tensor>& save_mean_opt, const std::optional<Tensor>& save_var_opt,
    bool update, double eps, std::array<bool,3> grad_input_mask, const Tensor& reserve) {
  TORCH_CHECK(false, "_new_batch_norm_backward_mkldnn: ATen not compiled with MKLDNN support");
}

} // namespace native
} // namespace at

#else // AT_ONEDNN_ENABLED

#include <ATen/native/onednn/ONEDNNCommon.h>
#include <ATen/native/layer_norm.h>
#include <ideep/abstract_types.hpp>

namespace at::native {

std::tuple<Tensor, Tensor, Tensor> onednn_layer_norm_last_index_weight_bias_f32(
    const Tensor& input,
    IntArrayRef normalized_shape, const Tensor& weight, const Tensor& bias,
    double eps, bool inplace) {

  TORCH_INTERNAL_ASSERT(normalized_shape.size() == 1, "only accept shapes with the last dimension");
  TORCH_INTERNAL_ASSERT(input.scalar_type() == at::kFloat);
  auto M_N = at::native::_check_layer_norm_inputs(input, normalized_shape, weight, bias);
  auto M = M_N.first;

  auto mean = empty_mkldnn(
        {M},
        input.scalar_type(),
        input.options().layout_opt(),
        input.options().device_opt(),
        input.options().pinned_memory_opt());
  auto rstd = empty_mkldnn(
        {M},
        input.scalar_type(),
        input.options().layout_opt(),
        input.options().device_opt(),
        input.options().pinned_memory_opt());

  auto mean_it = at::native::itensor_from_onednn(mean);
  auto rstd_it = at::native::itensor_from_onednn(rstd);

  auto input_it = at::native::itensor_from_onednn(input);
  auto weight_it = at::native::itensor_from_onednn(weight);
  auto bias_it = at::native::itensor_from_onednn(bias);

  auto out_it = inplace ? input_it : ideep::tensor(input_it.get_desc());
  ideep::layer_normalization_forward::compute(input_it, weight_it, bias_it, out_it, mean_it, rstd_it, static_cast<float>(eps));

  auto dst = at::native::new_with_itensor_onednn(
      std::move(out_it),
      optTypeMetaToScalarType(input.options().dtype_opt()),
      input.options().device_opt());

  return std::make_tuple(dst, mean, rstd);
}


std::tuple<Tensor, Tensor, Tensor> mkldnn_batch_norm(
    const Tensor& input, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt, const std::optional<Tensor>& running_mean_opt, const std::optional<Tensor>& running_var_opt,
    bool train,
    double momentum,
    double eps) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  const Tensor& bias = bias_opt.value_or(Tensor());
  const Tensor& running_mean = running_mean_opt.value_or(Tensor());
  const Tensor& running_var = running_var_opt.value_or(Tensor());

  if (input.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(onednn_bf16_device_check(),
        "mkldnn_batch_norm: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq");
  }
  TORCH_CHECK(weight.defined() && bias.defined(),
             "mkldnn_batch_norm: currently mkldnn only support affine model");

  ideep::tensor& x = itensor_from_onednn(input);
  ideep::tensor w = itensor_from_tensor(weight);
  ideep::tensor b = itensor_from_tensor(bias);
  bool use_running_stat = (running_mean.defined() && running_var.defined());

  ideep::tensor y;

  if (train) {
    // TODO: enable 3d batchnorm.
    TORCH_CHECK(input.dim() == 4,
        "mkldnn_batch_norm: currently mkldnn training only support 2d batchnorm");
    ideep::tensor saved_mean;
    ideep::tensor saved_var;
    ideep::batch_normalization_forward_training::compute(
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
        x, w, b, y, saved_mean, saved_var, momentum, eps);
    if (use_running_stat) {
      auto len = x.get_nelems() / w.get_nelems(); // n*h*w
      ideep::tensor m = itensor_from_tensor(running_mean);
      ideep::tensor v = itensor_from_tensor(running_var);
      const std::vector<float> scales_mean{static_cast<float>(1 - momentum),
                                           static_cast<float>(momentum)};
      const std::vector<float> scales_var{static_cast<float>(1 - momentum),
                                          // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
                                          static_cast<float>(momentum * len / (len - 1))};
      ideep::sum::compute(scales_mean, {m, saved_mean}, m);
      ideep::sum::compute(scales_var, {v, saved_var}, v);
    }
    return std::make_tuple(
         new_with_itensor_onednn(std::move(y), optTypeMetaToScalarType(input.options().dtype_opt()),
                                 input.options().device_opt()),
         new_with_itensor_onednn(std::move(saved_mean), optTypeMetaToScalarType(weight.options().dtype_opt()),
                                 weight.options().device_opt()),
         new_with_itensor_onednn(std::move(saved_var), optTypeMetaToScalarType(weight.options().dtype_opt()),
                                 weight.options().device_opt()));
  } else {
    TORCH_CHECK(input.dim() == 4 || input.dim() == 5,
        "mkldnn_batch_norm: currently mkldnn inference only support 2d and 3d batchnorm");
    if (use_running_stat) {
      ideep::tensor m = itensor_from_tensor(running_mean);
      ideep::tensor v = itensor_from_tensor(running_var);
      ideep::batch_normalization_forward_inference::compute(
          // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
          x, m, v, w, b, y, eps);
    } else {
      // TODO: keep running estimates.
      TORCH_CHECK(false, "mkldnn_batch_norm: mkldnn inference is not keep running estimates.");
    }
    return std::make_tuple(
        new_with_itensor_onednn(std::move(y), optTypeMetaToScalarType(input.options().dtype_opt()),
                                input.options().device_opt()),
        new_with_itensor_onednn(ideep::tensor{}, optTypeMetaToScalarType(weight.options().dtype_opt()),
                                weight.options().device_opt()),
        new_with_itensor_onednn(ideep::tensor{}, optTypeMetaToScalarType(weight.options().dtype_opt()),
                                weight.options().device_opt()));
  }
}


std::tuple<Tensor, Tensor, Tensor, Tensor> _batch_norm_with_update_mkldnn(
    const Tensor& input, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt,
    Tensor& running_mean, Tensor& running_var, double momentum, double eps) {
  auto [output, save_mean, save_var] =
    mkldnn_batch_norm(input, weight_opt, bias_opt, running_mean, running_var, /*train*/true, momentum, eps);
  Tensor reserve = empty_mkldnn({0}, input.scalar_type());
  return std::tuple<Tensor, Tensor, Tensor, Tensor>(output, save_mean, save_var, reserve);
}


std::tuple<Tensor, Tensor, Tensor> _mkldnn_batch_norm_legit(
    const Tensor& input, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt, Tensor& running_mean, Tensor& running_var,
    bool train,
    double momentum,
    double eps) {
  return mkldnn_batch_norm(input, weight_opt, bias_opt, running_mean, running_var, train, momentum, eps);
}


std::tuple<Tensor, Tensor, Tensor> _mkldnn_batch_norm_legit_no_stats(
    const Tensor& input, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt,
    bool train,
    double momentum,
    double eps) {
  return mkldnn_batch_norm(input, weight_opt, bias_opt, Tensor(), Tensor(), train, momentum, eps);
}


std::tuple<Tensor, Tensor, Tensor> _new_batch_norm_backward_mkldnn(
    const Tensor& grad_output, const Tensor& input, const Tensor& weight,
    const std::optional<Tensor>& running_mean_opt, const std::optional<Tensor>& running_var_opt,
    const std::optional<Tensor>& save_mean_opt, const std::optional<Tensor>& save_var_opt,
    bool update, double eps, std::array<bool,3> grad_input_mask, const Tensor& reserve) {
  return mkldnn_batch_norm_backward(grad_output, input, weight, running_mean_opt, running_var_opt, save_mean_opt, save_var_opt, update, eps, grad_input_mask);
}


std::tuple<Tensor, Tensor, Tensor> mkldnn_batch_norm_backward(const Tensor& grad_output,
    const Tensor& input, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& running_mean_opt, const std::optional<Tensor>& running_var_opt, const std::optional<Tensor>& save_mean_opt, const std::optional<Tensor>& save_invstd_opt,
    bool train,
    double eps,
    std::array<bool,3> grad_input_mask) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  const Tensor& save_mean = save_mean_opt.value_or(Tensor());
  const Tensor& save_invstd = save_invstd_opt.value_or(Tensor());

  TORCH_CHECK(train, "mkldnn_batch_norm_backward: currently mkldnn only support train model");
  ideep::tensor& grady = itensor_from_onednn(grad_output);
  ideep::tensor& x = itensor_from_onednn(input);
  ideep::tensor w = itensor_from_tensor(weight);
  ideep::tensor& m = itensor_from_onednn(save_mean);
  ideep::tensor& v = itensor_from_onednn(save_invstd);

  ideep::tensor gradx, gradw, gradb;
  ideep::batch_normalization_backward::compute(
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      x, m, v, grady, w, gradx, gradw, gradb, eps);

  return std::make_tuple(
      new_with_itensor_onednn(std::move(gradx), optTypeMetaToScalarType(input.options().dtype_opt()),
                              input.options().device_opt()),
      mkldnn_to_dense(new_with_itensor_onednn(std::move(gradw),
                                              optTypeMetaToScalarType(weight.options().dtype_opt()),
                                              weight.options().device_opt())),
      mkldnn_to_dense(new_with_itensor_onednn(std::move(gradb),
                                              optTypeMetaToScalarType(weight.options().dtype_opt()),
                                              weight.options().device_opt())));
}

} // namespace at

#endif // AT_ONEDNN_ENABLED
