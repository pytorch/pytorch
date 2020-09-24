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
  ideep::tensor& x = itensor_from_mkldnn(input);
  ideep::tensor& w = itensor_from_mkldnn(weight);
  ideep::tensor& b = itensor_from_mkldnn(bias);
  ideep::tensor& m = itensor_from_mkldnn(running_mean);
  ideep::tensor& v = itensor_from_mkldnn(running_var);

  ideep::tensor y;

  if (train) {
    // TODO: support training
    TORCH_CHECK(false, "mkldnn_batch_norm: mkldnn training is not supported in yet.");

    // ideep::tensor saved_mean;
    // ideep::tensor saved_var;
    // ideep::batch_normalization_forward_training::compute<AllocForMKLDNN>(
    //     x, w, b, y, saved_mean, saved_var, m, v, momentum, eps);
    // return std::make_tuple(
    //     new_with_itensor_mkldnn(std::move(y), input.options()),
    //     new_with_itensor_mkldnn(std::move(saved_mean), input.options()),
    //     new_with_itensor_mkldnn(std::move(saved_var), input.options()));
  } else {
    TORCH_CHECK(input.dim() == 4 || input.dim() == 5,
               "mkldnn_batch_norm: currently mkldnn only support 2d and 3d batchnorm");
    ideep::batch_normalization_forward_inference::compute(
        x, m, v, w, b, y, eps);
    return std::make_tuple(
        new_with_itensor_mkldnn(std::move(y), input.options()),
        new_with_itensor_mkldnn(ideep::tensor{}, input.options()),
        new_with_itensor_mkldnn(ideep::tensor{}, input.options()));
  }
}

} // namespace native
} // namespace at

#endif // AT_MKLDNN_EBABLED
