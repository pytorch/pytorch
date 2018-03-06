#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"

#include "ATen/Config.h"
#if AT_CUDNN_ENABLED()
#include "THC/THC.h"
#include "ATen/cudnn/cudnn-wrapper.h"
#endif

namespace at { namespace native {

namespace {
  void check_dims_match_num_input_features(const char* arg_name, int64_t expected, int64_t actual){
    if (actual != expected){
      std::stringstream ss;
      ss << arg_name << " should contain " << expected << " elements not " << actual ;
      throw std::runtime_error(ss.str());
    }
  }
}

Tensor batch_norm(
    const Tensor& input, const Tensor& weight /* optional */, const Tensor& bias /* optional */,
    const Tensor& running_mean /* optional */, const Tensor& running_var /* optional */,
    bool training, double momentum, double eps, bool cudnn_enabled) {

  auto num_features = input.sizes()[1];
  if (running_mean.defined()) {
    check_dims_match_num_input_features("running_mean", num_features, running_mean.numel());
  } else if (!training) {
    throw std::runtime_error("running_mean must be defined in evaluation mode");
  }
  if (running_var.defined()) {
    check_dims_match_num_input_features("running_var", num_features, running_var.numel());
  } else if (!training) {
    throw std::runtime_error("running_var must be defined in evaluation mode");
  }
  if (weight.defined()) {
    check_dims_match_num_input_features("weight", num_features, weight.numel());
  }
  if (bias.defined()) {
    check_dims_match_num_input_features("bias", num_features, bias.numel());
  }

  bool use_cudnn = false;
#if AT_CUDNN_ENABLED()
  use_cudnn = (input.type().is_cuda()
               && (input.type().scalarType() != at::kHalf
                 || weight.type().scalarType() == at::kFloat)
               && weight.defined() && bias.defined()
               && ((running_mean.defined() && running_var.defined())
                 || (!running_mean.defined() && !running_var.defined() && training))
               && input.size(0) <= 131070
               && cudnn_enabled && CUDNN_VERSION >= 5110L);
#endif

#if AT_CUDNN_ENABLED()
  if (use_cudnn && eps >= CUDNN_BN_MIN_EPSILON) {
    return std::get<0>(at::cudnn_batch_norm(
                        input, weight, bias,
                        running_mean, running_var,
                        training, momentum, eps));
  }
#endif
  return at::thnn_batch_norm(
            input, weight, bias,
            running_mean, running_var, training, momentum, eps);
}

}} // at::native
