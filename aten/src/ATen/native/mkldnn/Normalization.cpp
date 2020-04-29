#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <tuple>

#if !AT_MKLDNN_ENABLED()

namespace at {
namespace native {

std::tuple<Tensor, Tensor, Tensor> mkldnn_batch_norm(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& running_mean,
    const Tensor& running_var,
    bool train,
    double momentum,
    double eps) {
  TORCH_CHECK(false, "mkldnn_batch_norm: ATen not compiled with MKLDNN support");
}

std::tuple<Tensor, Tensor, Tensor> mkldnn_batch_norm_backward(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& running_mean,
    const Tensor& running_var,
    const Tensor& save_mean,
    const Tensor& save_invstd,
    bool train,
    double eps,
    std::array<bool,3> grad_input_mask) {
  TORCH_CHECK(false, "mkldnn_batch_norm_backward: ATen not compiled with MKLDNN support");
}

} // namespace native
} // namespace at

#else // AT_MKLDNN_EBABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at {
namespace native {

std::tuple<Tensor, Tensor, Tensor> mkldnn_batch_norm(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& running_mean,
    const Tensor& running_var,
    bool train,
    double momentum,
    double eps) {
  TORCH_CHECK(input.dim() == 4 || input.dim() == 5,
             "mkldnn_batch_norm: currently mkldnn only support 2d and 3d batchnorm");
  TORCH_CHECK(weight.defined() && bias.defined(),
             "mkldnn_batch_norm: currently mkldnn only support affine model");

  ideep::tensor& x = itensor_from_mkldnn(input);
  const ideep::tensor w = itensor_from_tensor(weight);
  const ideep::tensor b = itensor_from_tensor(bias);

  bool use_running_stat = (running_mean.defined() && running_var.defined());
  ideep::tensor y;

  if (train) {
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
        new_with_itensor_mkldnn(std::move(y), input.options()),
        new_with_itensor_mkldnn(std::move(saved_mean), weight.options()),
        new_with_itensor_mkldnn(std::move(saved_var), weight.options()));
  } else {
    if (use_running_stat) {
      ideep::tensor m = itensor_from_tensor(running_mean);
      ideep::tensor v = itensor_from_tensor(running_var);
      ideep::batch_normalization_forward_inference::compute(
          x, m, v, w, b, y, eps);
    } else {
      ideep::batch_normalization_forward_inference::compute(
          x, w, b, y, eps);
    }
    return std::make_tuple(
        new_with_itensor_mkldnn(std::move(y), input.options()),
        new_with_itensor_mkldnn(ideep::tensor{}, weight.options()),
        new_with_itensor_mkldnn(ideep::tensor{}, weight.options()));
  }
}

std::tuple<Tensor, Tensor, Tensor> mkldnn_batch_norm_backward(const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& running_mean,
    const Tensor& running_var,
    const Tensor& save_mean,
    const Tensor& save_invstd,
    bool train,
    double eps,
    std::array<bool,3> grad_input_mask) {
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
      new_with_itensor_mkldnn(std::move(gradx), input.options()),
      mkldnn_to_dense(new_with_itensor_mkldnn(std::move(gradw), weight.options())),
      mkldnn_to_dense(new_with_itensor_mkldnn(std::move(gradb), weight.options())));
}

} // namespace native
} // namespace at

#endif // AT_MKLDNN_EBABLED
