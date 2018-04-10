#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"

#include "ATen/Config.h"
#if AT_CUDNN_ENABLED()
#include "THC/THC.h"
#include "ATen/cudnn/cudnn-wrapper.h"
#endif
#include <vector>

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
#else
  (void)cudnn_enabled; // avoid unused parameter warning
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

Tensor layer_norm(const Tensor& input, IntList normalized_shape,
    const Tensor& weight /* optional */, const Tensor& bias /* optional */,
    double eps, bool cudnn_enabled) {

    int64_t normalized_ndim = normalized_shape.size();

    if (normalized_ndim < 1) {
      std::stringstream ss;
      ss << "Expected normalized_shape to be at least 1-dimensional, i.e., "
         << "containing at least one element, but got normalized_shape="
         << normalized_shape;
      throw std::runtime_error(ss.str());
    }

    if (weight.defined() && !weight.sizes().equals(normalized_shape)) {
      std::stringstream ss;
      ss << "Expected weight to be of same shape as normalized_shape, but got "
         << "weight of shape " << weight.sizes() << " and normalized_shape="
         << normalized_shape;
      throw std::runtime_error(ss.str());
    }

    if (bias.defined() && !bias.sizes().equals(normalized_shape)) {
      std::stringstream ss;
      ss << "Expected bias to be of same shape as normalized_shape, but got "
         << "bias of shape " << bias.sizes() << " and normalized_shape="
         << normalized_shape;
      throw std::runtime_error(ss.str());
    }

    auto input_shape = input.sizes();
    auto input_ndim = input.dim();

    if (input_ndim < normalized_ndim ||
        !input_shape.slice(input_ndim - normalized_ndim).equals(normalized_shape)) {
      std::stringstream ss;
      ss << "Given normalized_shape=" << normalized_shape
         << ", expected input with shape [*";
      for (auto size : normalized_shape) {
        ss << ", " << size;
      }
      ss << "], but got input of size" << input_shape;
      throw std::runtime_error(ss.str());
    }

    int64_t n = 1;
    for (int64_t i = 0; i < input_ndim - normalized_ndim; i++) {
      n *= input_shape[i];
    }

    // Apply layer norm
    auto input_reshaped = input.contiguous().view({1, n, -1});

    auto out = at::batch_norm(input_reshaped, {}, {}, {}, {}, true, 0, eps,
                              cudnn_enabled);
    out = out.view(input_shape);

    if (weight.defined() && bias.defined()) {
      return bias.addcmul(out, weight, 1);
    } else if (weight.defined()) {
      return out.mul(weight);
    } else if (bias.defined()) {
      return out.add(bias);
    } else {
      return out;
    }
}

Tensor group_norm(const Tensor& input, int64_t num_groups,
    const Tensor& weight /* optional */, const Tensor& bias /* optional */,
    double eps, bool cudnn_enabled) {

    auto input_shape = input.sizes();
    int64_t b = input.size(0);
    int64_t c = input.size(1);

    if (c % num_groups != 0) {
      std::stringstream ss;
      ss << "Expected number of channels in input to be divisible by "
         << "num_groups, but got input of shape " << input.sizes() << " and "
         << "num_groups=" << num_groups;
      throw std::runtime_error(ss.str());
    }

    if (weight.defined() && (weight.dim() != 1 || weight.numel() != c)) {
      std::stringstream ss;
      ss << "Expected weight to be a vector of size equal to the number of "
         << "channels in input, but got weight of shape " << weight.sizes()
         << " and input of shape " <<  input.sizes();
      throw std::runtime_error(ss.str());
    }

    if (bias.defined() && (bias.dim() != 1 || bias.numel() != c)) {
      std::stringstream ss;
      ss << "Expected bias to be a vector of size equal to the number of "
         << "channels in input, but got bias of shape " << weight.sizes()
         << " and input of shape " <<  input.sizes();
      throw std::runtime_error(ss.str());
    }

    // Apply group norm
    auto input_reshaped = input.contiguous().view({1, b * num_groups, -1});

    auto out = at::batch_norm(input_reshaped, {}, {}, {}, {}, true, 0, eps,
                              cudnn_enabled);
    out = out.view(input_shape);

    if (!weight.defined() && !bias.defined()) {
      return out;
    }

    std::vector<int64_t> affine_param_shape(input.dim(), 1);
    affine_param_shape[1] = c;

    if (weight.defined() && bias.defined()) {
      return bias.view(affine_param_shape).addcmul(out, weight.view(affine_param_shape), 1);
    } else if (weight.defined()) {
      return out.mul(weight.view(affine_param_shape));
    } else {
      return out.add(bias.view(affine_param_shape));
    }
}

}} // at::native
