#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <tuple>

#if !AT_MKLDNN_ENABLED()

namespace at {
namespace native {

std::tuple<Tensor, Tensor, Tensor> mkldnn_batch_norm(
    const Tensor& self, const c10::optional<Tensor>& weight_opt, const c10::optional<Tensor>& bias_opt, const c10::optional<Tensor>& running_mean_opt, const c10::optional<Tensor>& running_var_opt,
    bool train,
    double momentum,
    double eps) {
  TORCH_CHECK(false, "mkldnn_batch_norm: ATen not compiled with MKLDNN support");
}

std::tuple<Tensor, Tensor, Tensor> mkldnn_batch_norm_backward(
    const Tensor& grad_output,
    const Tensor& input, const c10::optional<Tensor>& weight_opt, const c10::optional<Tensor>& running_mean_opt, const c10::optional<Tensor>& running_var_opt, const c10::optional<Tensor>& save_mean_opt, const c10::optional<Tensor>& save_invstd_opt,
    bool train,
    double eps,
    std::array<bool,3> grad_input_mask) {
  TORCH_CHECK(false, "mkldnn_batch_norm_backward: ATen not compiled with MKLDNN support");
}

} // namespace native
} // namespace at

#else // AT_MKLDNN_EBABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/native/mkldnn/Utils.h>

namespace at {
namespace native {

std::tuple<Tensor, Tensor, Tensor> mkldnn_batch_norm(
    const Tensor& input, const c10::optional<Tensor>& weight_opt, const c10::optional<Tensor>& bias_opt, const c10::optional<Tensor>& running_mean_opt, const c10::optional<Tensor>& running_var_opt,
    bool train,
    double momentum,
    double eps) {
  // See [Note: hacky wrapper removal for optional tensor]
  const Tensor& weight = c10::value_or_else(weight_opt, [] {return Tensor();});
  const Tensor& bias = c10::value_or_else(bias_opt, [] {return Tensor();});
  const Tensor& running_mean = c10::value_or_else(running_mean_opt, [] {return Tensor();});
  const Tensor& running_var = c10::value_or_else(running_var_opt, [] {return Tensor();});

  if (input.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(mkldnn_bf16_device_check(),
        "mkldnn_batch_norm: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq");
  }
  TORCH_CHECK(weight.defined() && bias.defined(),
             "mkldnn_batch_norm: currently mkldnn only support affine model");

  ideep::tensor& x = itensor_from_mkldnn(input);
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
        x, w, b, y, saved_mean, saved_var, momentum, eps);
    if (use_running_stat) {
      auto len = x.get_nelems() / w.get_nelems(); // n*h*w
      ideep::tensor m = itensor_from_tensor(running_mean);
      ideep::tensor v = itensor_from_tensor(running_var);
      const std::vector<float> scales_mean{static_cast<float>(1 - momentum),
                                           static_cast<float>(momentum)};
      const std::vector<float> scales_var{static_cast<float>(1 - momentum),
                                          static_cast<float>(momentum * len / (len - 1))};
      ideep::sum::compute(scales_mean, {m, saved_mean}, m);
      ideep::sum::compute(scales_var, {v, saved_var}, v);
    }
    return std::make_tuple(
         new_with_itensor_mkldnn(std::move(y), optTypeMetaToScalarType(input.options().dtype_opt()),
                                 input.options().device_opt()),
         new_with_itensor_mkldnn(std::move(saved_mean), optTypeMetaToScalarType(weight.options().dtype_opt()),
                                 weight.options().device_opt()),
         new_with_itensor_mkldnn(std::move(saved_var), optTypeMetaToScalarType(weight.options().dtype_opt()),
                                 weight.options().device_opt()));
  } else {
    TORCH_CHECK(input.dim() == 4 || input.dim() == 5,
        "mkldnn_batch_norm: currently mkldnn inference only support 2d and 3d batchnorm");
    if (use_running_stat) {
      ideep::tensor m = itensor_from_tensor(running_mean);
      ideep::tensor v = itensor_from_tensor(running_var);
      ideep::batch_normalization_forward_inference::compute(
          x, m, v, w, b, y, eps);
    } else {
      // TODO: keep running estimates.
      TORCH_CHECK(false, "mkldnn_batch_norm: mkldnn inference is not keep running estimates.");
    }
    return std::make_tuple(
        new_with_itensor_mkldnn(std::move(y), optTypeMetaToScalarType(input.options().dtype_opt()),
                                input.options().device_opt()),
        new_with_itensor_mkldnn(ideep::tensor{}, optTypeMetaToScalarType(weight.options().dtype_opt()),
                                weight.options().device_opt()),
        new_with_itensor_mkldnn(ideep::tensor{}, optTypeMetaToScalarType(weight.options().dtype_opt()),
                                weight.options().device_opt()));
  }
}

std::tuple<Tensor, Tensor, Tensor> mkldnn_batch_norm_backward(const Tensor& grad_output,
    const Tensor& input, const c10::optional<Tensor>& weight_opt, const c10::optional<Tensor>& running_mean_opt, const c10::optional<Tensor>& running_var_opt, const c10::optional<Tensor>& save_mean_opt, const c10::optional<Tensor>& save_invstd_opt,
    bool train,
    double eps,
    std::array<bool,3> grad_input_mask) {
  // See [Note: hacky wrapper removal for optional tensor]
  const Tensor& weight = c10::value_or_else(weight_opt, [] {return Tensor();});
  const Tensor& running_mean = c10::value_or_else(running_mean_opt, [] {return Tensor();});
  const Tensor& running_var = c10::value_or_else(running_var_opt, [] {return Tensor();});
  const Tensor& save_mean = c10::value_or_else(save_mean_opt, [] {return Tensor();});
  const Tensor& save_invstd = c10::value_or_else(save_invstd_opt, [] {return Tensor();});

  TORCH_CHECK(train, "mkldnn_batch_norm_backward: currently mkldnn only support train model");
  ideep::tensor& grady = itensor_from_mkldnn(grad_output);
  ideep::tensor& x = itensor_from_mkldnn(input);
  ideep::tensor w = itensor_from_tensor(weight);
  ideep::tensor& m = itensor_from_mkldnn(save_mean);
  ideep::tensor& v = itensor_from_mkldnn(save_invstd);

  ideep::tensor gradx, gradw, gradb;
  ideep::batch_normalization_backward::compute(
      x, m, v, grady, w, gradx, gradw, gradb, eps);

  return std::make_tuple(
      new_with_itensor_mkldnn(std::move(gradx), optTypeMetaToScalarType(input.options().dtype_opt()),
                              input.options().device_opt()),
      mkldnn_to_dense(new_with_itensor_mkldnn(std::move(gradw),
                                              optTypeMetaToScalarType(weight.options().dtype_opt()),
                                              weight.options().device_opt())),
      mkldnn_to_dense(new_with_itensor_mkldnn(std::move(gradb),
                                              optTypeMetaToScalarType(weight.options().dtype_opt()),
                                              weight.options().device_opt())));
}

} // namespace native
} // namespace at

#endif // AT_MKLDNN_EBABLED
