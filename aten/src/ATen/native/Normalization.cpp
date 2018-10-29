#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"

#include "ATen/Config.h"

#include "ATen/detail/CUDAHooksInterface.h"

#include <vector>

static const int MIOPEN_DIM_MAX = 4;

namespace at { namespace native {

namespace {
  void check_dims_match_num_input_features(const char* arg_name, int64_t expected, int64_t actual){
    AT_CHECK(actual == expected,
             arg_name, " should contain ", expected, " elements not ", actual);
  }

  static inline Tensor repeat_if_defined(const Tensor& t, int64_t repeat) {
    if (t.defined()) {
      return t.repeat(repeat);
    }
    return t;
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
    AT_ERROR("running_mean must be defined in evaluation mode");
  }
  if (running_var.defined()) {
    check_dims_match_num_input_features("running_var", num_features, running_var.numel());
  } else if (!training) {
    AT_ERROR("running_var must be defined in evaluation mode");
  }
  if (weight.defined()) {
    check_dims_match_num_input_features("weight", num_features, weight.numel());
  }
  if (bias.defined()) {
    check_dims_match_num_input_features("bias", num_features, bias.numel());
  }

  bool use_cudnn = false;
  use_cudnn = (input.type().is_cuda()
               && (input.type().scalarType() != at::kHalf
                 || weight.type().scalarType() == at::kFloat)
               && weight.defined() && bias.defined()
               && ((running_mean.defined() && running_var.defined())
                 || (!running_mean.defined() && !running_var.defined() && training))
               && input.size(0) <= 131070
               && detail::getCUDAHooks().compiledWithCuDNN()
               && cudnn_enabled && detail::getCUDAHooks().versionCuDNN() >= 5110L);

  if (use_cudnn && eps >= detail::getCUDAHooks().batchnormMinEpsilonCuDNN()) {
    return std::get<0>(at::cudnn_batch_norm(
                        input.contiguous(), weight.contiguous(),
                        bias.contiguous(),
                        running_mean.defined() ? running_mean.contiguous() : running_mean,
                        running_var.defined() ? running_var.contiguous() : running_var,
                        training, momentum, eps));
  }

  bool use_miopen = (input.type().is_cuda()
               && input.dim() <= MIOPEN_DIM_MAX
               && input.type().scalarType() != at::kDouble
               && (input.type().scalarType() == weight.type().scalarType())
               && weight.defined() && bias.defined()
               && ((running_mean.defined() && running_var.defined())
                 || (!running_mean.defined() && !running_var.defined() && training))
               && detail::getCUDAHooks().compiledWithMIOpen()
               );

  if (use_miopen) {
    return std::get<0>(at::miopen_batch_norm(
                        input.contiguous(), weight.contiguous(), bias.contiguous(),
                        running_mean.defined() ? running_mean.contiguous() : running_mean,
                        running_var.defined() ? running_var.contiguous() : running_var,
                        training, momentum, eps));
  }

  return at::thnn_batch_norm(
            input.contiguous(), weight, bias,
            running_mean, running_var, training, momentum, eps);
}

Tensor instance_norm(
    const Tensor& input, const Tensor& weight /* optional */, const Tensor& bias /* optional */,
    const Tensor& running_mean /* optional */, const Tensor& running_var /* optional */,
    bool use_input_stats, double momentum, double eps, bool cudnn_enabled) {
  AT_CHECK(use_input_stats || (running_mean.defined() && running_var.defined()),
           "Expected running_mean and running_var to be defined when use_input_stats is false");
  std::vector<int64_t> shape = input.sizes().vec();
  int64_t b = input.size(0);
  int64_t c = input.size(1);
  shape[1] = b * c;
  shape[0] = 1;

  Tensor weight_ = repeat_if_defined(weight, b);
  Tensor bias_ = repeat_if_defined(bias, b);
  Tensor running_mean_ = repeat_if_defined(running_mean, b);
  Tensor running_var_ = repeat_if_defined(running_var, b);

  auto input_reshaped = input.contiguous().view(shape);
  auto out = at::batch_norm(input_reshaped, weight_, bias_, running_mean_, running_var_,
                            use_input_stats, momentum, eps, cudnn_enabled);

  // we alias running_mean and running_var because they are const but we want to modify their data
  if (running_mean.defined()) {
    at::alias(running_mean).copy_(running_mean_.view({ b, c }).mean(0, false));
  }
  if (running_var.defined()) {
    at::alias(running_var).copy_(running_var_.view({ b, c }).mean(0, false));
  }

  return out.view(input.sizes());
}

Tensor layer_norm(const Tensor& input, IntList normalized_shape,
    const Tensor& weight /* optional */, const Tensor& bias /* optional */,
    double eps, bool cudnn_enabled) {

    int64_t normalized_ndim = normalized_shape.size();

    AT_CHECK(normalized_ndim >= 1,
             "Expected normalized_shape to be at least 1-dimensional, i.e., ",
             "containing at least one element, but got normalized_shape=",
             normalized_shape);

    AT_CHECK(!weight.defined() || weight.sizes().equals(normalized_shape),
             "Expected weight to be of same shape as normalized_shape, but got ",
             "weight of shape ", weight.sizes(), " and normalized_shape=",
             normalized_shape);
    AT_CHECK(!bias.defined() || bias.sizes().equals(normalized_shape),
             "Expected bias to be of same shape as normalized_shape, but got ",
             "bias of shape ", bias.sizes(), " and normalized_shape=",
             normalized_shape);

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
      AT_ERROR(ss.str());
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

    AT_CHECK(c % num_groups == 0,
             "Expected number of channels in input to be divisible by ",
             "num_groups, but got input of shape ", input.sizes(), " and "
             "num_groups=", num_groups);

    AT_CHECK(!weight.defined() || (weight.dim() == 1 && weight.numel() == c),
             "Expected weight to be a vector of size equal to the number of ",
             "channels in input, but got weight of shape ", weight.sizes(),
             " and input of shape ", input.sizes());
    AT_CHECK(!bias.defined() || (bias.dim() == 1 && bias.numel() == c),
             "Expected bias to be a vector of size equal to the number of ",
             "channels in input, but got bias of shape ", weight.sizes(),
             " and input of shape ", input.sizes());

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
