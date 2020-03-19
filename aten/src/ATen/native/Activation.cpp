#include <ATen/native/Activation.h>

#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/Parallel.h>
#include <ATen/core/DistributionsHelper.h>

namespace at { namespace native {

static const double SELU_ALPHA = 1.6732632423543772848170429916717;
static const double SELU_SCALE = 1.0507009873554804934193349852946;

DEFINE_DISPATCH(elu_stub);
DEFINE_DISPATCH(elu_backward_stub);
DEFINE_DISPATCH(softplus_stub);
DEFINE_DISPATCH(softplus_backward_stub);
DEFINE_DISPATCH(log_sigmoid_cpu_stub);
DEFINE_DISPATCH(log_sigmoid_backward_cpu_stub);
DEFINE_DISPATCH(threshold_stub);
DEFINE_DISPATCH(hardtanh_backward_stub);
DEFINE_DISPATCH(hardsigmoid_stub);
DEFINE_DISPATCH(hardsigmoid_backward_stub);
DEFINE_DISPATCH(hardshrink_stub);
DEFINE_DISPATCH(softshrink_stub);
DEFINE_DISPATCH(shrink_backward_stub);
DEFINE_DISPATCH(leaky_relu_stub);
DEFINE_DISPATCH(leaky_relu_backward_stub);

Tensor hardtanh(const Tensor& self, Scalar min, Scalar max) {
  return at::clamp(self, min, max);
}

Tensor& hardtanh_out(Tensor& result, const Tensor& self, Scalar min, Scalar max) {
  return at::clamp_out(result, self, min, max);
}

Tensor& hardtanh_(Tensor& self, Scalar min, Scalar max) {
  return at::clamp_(self, min, max);
}

Tensor& hardtanh_backward_out(Tensor& grad_input,
    const Tensor& grad_output, const Tensor& self, Scalar min, Scalar max) {
  auto iter = TensorIterator::binary_op(grad_input, grad_output, self);
  hardtanh_backward_stub(iter.device_type(), iter, min, max);
  return grad_input;
}

Tensor hardtanh_backward(const Tensor& grad_output, const Tensor& self, Scalar min, Scalar max) {
  Tensor result;
  auto iter = TensorIterator::binary_op(result, grad_output, self);
  hardtanh_backward_stub(iter.device_type(), iter, min, max);
  return iter.output();
}

Tensor hardsigmoid(const Tensor& self) {
  Tensor result;
  auto iter = TensorIterator::unary_op(result, self);
  hardsigmoid_stub(iter.device_type(), iter);
  return iter.output();
}

Tensor& hardsigmoid_out(Tensor& result, const Tensor& self) {
  auto iter = TensorIterator::unary_op(result, self);
  hardsigmoid_stub(iter.device_type(), iter);
  return result;
}

Tensor& hardsigmoid_(Tensor& self) {
  Tensor result;
  auto iter = TensorIterator::unary_op(self, self);
  hardsigmoid_stub(iter.device_type(), iter);
  return self;
}

Tensor hardsigmoid_backward(const Tensor& grad_output, const Tensor& self) {
  Tensor result;
  auto iter = TensorIterator::binary_op(result, grad_output, self);
  hardsigmoid_backward_stub(iter.device_type(), iter);
  return iter.output();
}

Tensor& elu_out(
    Tensor& result,
    const Tensor& self,
    Scalar alpha,
    Scalar scale,
    Scalar input_scale) {
  auto iter = TensorIterator::unary_op(result, self);
  elu_stub(iter.device_type(), iter, alpha, scale, input_scale);
  return result;
}

Tensor elu(
    const Tensor& self,
    Scalar alpha,
    Scalar scale,
    Scalar input_scale) {
  Tensor result;
  auto iter = TensorIterator::unary_op(result, self);
  elu_stub(iter.device_type(), iter, alpha, scale, input_scale);
  return iter.output();
}

Tensor & elu_(
    Tensor & self,
    Scalar alpha,
    Scalar scale,
    Scalar input_scale) {
  return at::elu_out(self, self, alpha, scale, input_scale);
}

Tensor& elu_backward_out(
    Tensor& grad_input,
    const Tensor& grad_output,
    Scalar alpha,
    Scalar scale,
    Scalar input_scale,
    const Tensor& output) {
  auto iter = TensorIterator::binary_op(grad_input, grad_output, output);
  elu_backward_stub(iter.device_type(), iter, alpha, scale, input_scale);
  return grad_input;
}

Tensor elu_backward(
    const Tensor& grad_output,
    Scalar alpha,
    Scalar scale,
    Scalar input_scale,
    const Tensor& output) {
  Tensor result;
  auto iter = TensorIterator::binary_op(result, grad_output, output);
  elu_backward_stub(iter.device_type(), iter, alpha, scale, input_scale);
  return iter.output();
}

Tensor relu(const Tensor & self) {
  return at::threshold(self, 0, 0);
}

Tensor & relu_(Tensor & self) {
  return at::threshold_(self, 0, 0);
}

Tensor selu(const Tensor & self) {
  return at::elu(self, SELU_ALPHA, SELU_SCALE);
}

Tensor & selu_(Tensor & self) {
  return at::elu_(self, SELU_ALPHA, SELU_SCALE);
}

Tensor celu(const Tensor & self, Scalar alpha) {
  double inv_alpha = 1. / alpha.to<double>();
  return at::elu(self, alpha, Scalar(1.0), Scalar(inv_alpha));
}

Tensor & celu_(Tensor & self, Scalar alpha) {
  double inv_alpha = 1. / alpha.to<double>();
  return at::elu_(self, alpha, Scalar(1.0), Scalar(inv_alpha));
}


template <typename scalar_t>
inline void _rrelu_with_noise_train(
    Tensor& output,
    const Tensor& input,
    const Tensor& noise,
    Scalar lower_,
    Scalar upper_,
    Generator* generator) {
  scalar_t lower = lower_.to<scalar_t>();
  scalar_t upper = upper_.to<scalar_t>();
  Tensor tmp_tensor = output.contiguous();
  scalar_t* output_data = tmp_tensor.data_ptr<scalar_t>();
  scalar_t* input_data = input.data_ptr<scalar_t>();
  scalar_t* noise_data = noise.data_ptr<scalar_t>();
  auto gen  = at::get_generator_or_default<CPUGenerator>(generator, detail::getDefaultCPUGenerator());
  std::lock_guard<std::mutex> lock(gen->mutex_);
  for (int64_t i = 0; i < input.numel(); i++) {
    if (input_data[i] <= 0) {
      at::uniform_real_distribution<double> uniform(lower, upper);
      const scalar_t r = (scalar_t)uniform(gen);
      output_data[i] = input_data[i] * r;
      noise_data[i] = r;
    } else {
      noise_data[i] = 1;
      output_data[i] = input_data[i];
    }
  }
  if (!output.is_contiguous()) {
    output.copy_(tmp_tensor);
  }
}

Tensor& rrelu_with_noise_out_cpu(
    Tensor& output,
    const Tensor& self,
    const Tensor& noise,
    Scalar lower,
    Scalar upper,
    bool training,
    Generator* generator) {
  if (training) {
    AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "rrelu_with_noise_out_cpu", [&] {
      _rrelu_with_noise_train<scalar_t>(output, self.contiguous(), noise, lower, upper, generator);
    });
    return output;
  } else {
    auto lower_tensor = scalar_to_tensor(lower, self.device());
    auto upper_tensor = scalar_to_tensor(upper, self.device());
    auto negative = (lower_tensor + upper_tensor) / 2;
    Scalar negative_slope = negative.item();
    return at::leaky_relu_out(output, self, negative_slope);
  }
}

Tensor rrelu_with_noise_cpu(
    const Tensor& self,
    const Tensor& noise,
    Scalar lower,
    Scalar upper,
    bool training,
    Generator* generator) {
  auto output = at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  return at::native::rrelu_with_noise_out_cpu(output, self, noise, lower, upper, training, generator);
}

Tensor& rrelu_with_noise_cpu_(
    Tensor& self,
    const Tensor& noise,
    Scalar lower,
    Scalar upper,
    bool training,
    Generator* generator) {
  return at::native::rrelu_with_noise_out_cpu(self, self, noise, lower, upper, training, generator);
}

Tensor rrelu_with_noise_backward(
    const Tensor& grad_output,
    const Tensor& self_or_result,
    const Tensor& noise,
    Scalar lower,
    Scalar upper,
    bool training,
    bool is_result) {
  auto lower_tensor = scalar_to_tensor(lower, grad_output.device());
  auto upper_tensor = scalar_to_tensor(upper, grad_output.device());
  if (training && (upper_tensor - lower_tensor).item().to<float>() > 1E-6) {
    return grad_output.mul(noise);
  } else {
    auto negative = (lower_tensor + upper_tensor) / 2;
    Scalar negative_slope = negative.item();
    return at::leaky_relu_backward(grad_output, self_or_result, negative_slope, is_result);
  }
}

Tensor rrelu(const Tensor & self, Scalar lower, Scalar upper, bool training, Generator* generator) {
  return at::rrelu_with_noise(self, at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT), lower, upper, training, generator);
}

Tensor & rrelu_(Tensor & self, Scalar lower, Scalar upper, bool training, Generator* generator) {
  return at::rrelu_with_noise_(self, at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT), lower, upper, training, generator);
}

Tensor & softplus_out(Tensor& result, const Tensor& self, Scalar beta, Scalar threshold) {
  auto iter = TensorIterator::unary_op(result, self);
  softplus_stub(iter.device_type(), iter, beta, threshold);
  return result;
}

Tensor softplus(const Tensor& self, Scalar beta, Scalar threshold) {
  Tensor result;
  auto iter = TensorIterator::unary_op(result, self);
  softplus_stub(iter.device_type(), iter, beta, threshold);
  return iter.output();
}

Tensor & softplus_backward_out(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& self,
    Scalar beta,
    Scalar threshold,
    const Tensor& output) {
  auto iter = TensorIterator::binary_op(grad_input, grad_output, output);
  softplus_backward_stub(iter.device_type(), iter, beta, threshold);
  return grad_input;
}

Tensor softplus_backward(
    const Tensor& grad_output,
    const Tensor& self,
    Scalar beta,
    Scalar threshold,
    const Tensor& output) {
  Tensor grad_input;
  auto iter = TensorIterator::binary_op(grad_input, grad_output, output);
  softplus_backward_stub(iter.device_type(), iter, beta, threshold);
  return iter.output();
}

// computes `result = self <= threshold ? value : other`
// other is `self` in threshold() and `grad` in threshold_backward()
static Tensor threshold_out(
    optional<Tensor> opt_result,
    const Tensor& self,
    Scalar threshold,
    Scalar value,
    const Tensor& other) {
  Tensor result = opt_result.value_or(Tensor());
  auto iter = TensorIterator::binary_op(result, self, other);
  threshold_stub(iter.device_type(), iter, threshold, value);
  return iter.output();
}

Tensor threshold(const Tensor& self, Scalar threshold, Scalar value) {
  return threshold_out(nullopt, self, threshold, value, self);
}

Tensor& threshold_(Tensor& self, Scalar threshold, Scalar value) {
  threshold_out(make_optional(self), self, threshold, value, self);
  return self;
}

Tensor& threshold_out(Tensor& result, const Tensor& self, Scalar threshold, Scalar value) {
  threshold_out(make_optional(result), self, threshold, value, self);
  return result;
}

Tensor threshold_backward(const Tensor& grad, const Tensor& self, Scalar threshold) {
  return threshold_out(nullopt, self, threshold, 0, grad);
}

// -----------------------------------
// prelu forward
// -----------------------------------
template <typename scalar_t>
void inline prelu_cpu_kernel_share_weights(
  Tensor& result,
  const Tensor& input,
  const Tensor& weight) {

  int64_t input_numel = input.numel();
  auto result_data = result.data_ptr<scalar_t>();
  auto input_data = input.data_ptr<scalar_t>();
  auto weight_val = weight.data_ptr<scalar_t>()[0];

  at::parallel_for(0, input_numel, 1000, [&](int64_t start, int64_t end) {
    for (auto i = start; i < end; i++) {
      scalar_t input_data_val = input_data[i];
      // to allow for compiler optimization, here splitting into two lines:
      scalar_t r = (input_data_val > 0) ? scalar_t(1) : weight_val;
      result_data[i] = r * input_data_val;
    }
  });
}

template <typename scalar_t>
void inline prelu_cpu_kernel_multi_weights(
  Tensor& result,
  const Tensor& input,
  const Tensor& weight,
  int64_t input_dim0_size,
  int64_t channel_size,
  int64_t input_stride0,
  int64_t input_stride1) {

  scalar_t* result_data = result.data_ptr<scalar_t>();
  scalar_t* input_data = input.data_ptr<scalar_t>();
  scalar_t* weight_data = weight.data_ptr<scalar_t>();

  auto loop = [&](int64_t start, int64_t end) {
    for (auto i = start; i < end; ++i) {
      int64_t offset = i * channel_size * input_stride1;
      scalar_t* n_input_data = input_data + offset;
      scalar_t* n_result_data = result_data + offset;
      for (auto j = 0; j < channel_size; ++j) {
        for (auto k = 0; k < input_stride1; ++k) {
          // to allow for compiler optimization, here splitting into two lines:
          scalar_t w = (n_input_data[k] > 0) ? scalar_t(1) : weight_data[j];
          n_result_data[k] = w * n_input_data[k];
        }
        n_input_data += input_stride1;
        n_result_data += input_stride1;
      }
    }
  };
  if (input.numel() > 1000) {
    at::parallel_for(0, input_dim0_size, 0, loop);
  } else {
    loop(0, input_dim0_size);
  }
}

Tensor prelu_cpu(const Tensor& self, const Tensor& weight_) {
  auto input = self.contiguous();
  auto weight = weight_.contiguous();

  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(weight.is_contiguous());

  int64_t weight_num = weight.numel();
  Tensor result = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto strides = input.strides();

  // case1: shared weight for all channels
  if (weight_num == 1) {
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "prelu_cpu", [&] {
      prelu_cpu_kernel_share_weights<scalar_t>(result, input, weight);
    });
  }
  else { // case2: multiple weights, one for each channel
    int64_t input_ndim = input.dim();
    TORCH_CHECK(input_ndim > 0, "Not allow zero-dim input tensor.");

    int64_t channel_size = 1; // channel_size default to 1
    int64_t input_dim0_size = 1, input_stride0 = 1, input_stride1 = 1;

    if (input_ndim > 1) {
      channel_size = input.size(1); // channel is the 2nd dim of input
      input_dim0_size = input.size(0);
      input_stride0 = strides[0];
      input_stride1 = strides[1];
    }
    TORCH_CHECK(channel_size == weight_num,
      "Mismatch of parameter numbers and input channel size. Found parameter numbers = ", weight_num,
      " and channel size = ", channel_size, ".");

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "prelu_cpu", [&] {
      prelu_cpu_kernel_multi_weights<scalar_t>(
        result,
        input,
        weight,
        input_dim0_size,
        channel_size,
        input_stride0,
        input_stride1);
    });
  }
  return result;
}

// -----------------------------------
// prelu backward
// -----------------------------------
template <typename scalar_t>
void inline prelu_cpu_backward_kernel_share_weights(
  const Tensor& input,
  const Tensor& weight,
  const Tensor& grad_out,
  Tensor& input_grad,
  Tensor& weight_grad) {

  int64_t input_numel = input.numel();
  auto input_data = input.data_ptr<scalar_t>();
  auto weight_val = weight.data_ptr<scalar_t>()[0];
  auto grad_out_data = grad_out.data_ptr<scalar_t>();
  auto input_grad_data = input_grad.data_ptr<scalar_t>();
  auto weight_grad_data = weight_grad.data_ptr<scalar_t>();

  scalar_t sum = at::parallel_reduce(0, input_numel, 1000, scalar_t(0),
      [&](int64_t start, int64_t end, scalar_t ident) -> scalar_t {
    scalar_t partial_sum = ident;
    for (auto i = start; i < end; i++) {
      scalar_t input_data_val = input_data[i];
      scalar_t grad_out_data_val = grad_out_data[i];
      // to allow for compiler optimization, here splitting into two lines:
      scalar_t w = (input_data_val > 0) ? scalar_t(1) : weight_val;
      input_grad_data[i] = w * grad_out_data_val;
      // to allow for compiler optimization, here splitting into two lines:
      scalar_t mask = (input_data_val > 0) ? scalar_t(0) : scalar_t(1);
      partial_sum += mask * input_data_val * grad_out_data_val;
    }
    return partial_sum;
  }, std::plus<scalar_t>());
  weight_grad_data[0] = sum;
}

template <typename scalar_t>
void inline prelu_cpu_backward_kernel_multi_weights(
  const Tensor& input,
  const Tensor& weight,
  const Tensor& grad_out,
  Tensor& input_grad,
  Tensor& weight_grad_collector,
  int64_t input_dim0_size,
  int64_t channel_size,
  int64_t input_stride0,
  int64_t input_stride1) {

  auto input_data = input.data_ptr<scalar_t>();
  auto weight_data = weight.data_ptr<scalar_t>();
  auto grad_out_data = grad_out.data_ptr<scalar_t>();
  auto input_grad_data = input_grad.data_ptr<scalar_t>();
  auto weight_grad_collector_data = weight_grad_collector.data_ptr<scalar_t>();

  auto loop = [&](int64_t start, int64_t end) {
    for (auto i = start; i < end; i++) {
      for (auto j = 0; j < channel_size; j++) {
        for (auto k = 0; k < input_stride1; k++) {
          int64_t pos = i * input_stride0 + j * input_stride1 + k;
          scalar_t weight_data_val = weight_data[j];
          scalar_t input_data_val = input_data[pos];
          scalar_t grad_out_data_val = grad_out_data[pos];
          // to allow for compiler optimization, here splitting into two lines:
          scalar_t w = (input_data_val > 0) ? scalar_t(1) : weight_data_val;
          input_grad_data[pos] = w * grad_out_data_val;
          // to allow for compiler optimization, here splitting into two lines:
          scalar_t mask = (input_data_val > 0) ? scalar_t(0) : scalar_t(1);
          weight_grad_collector_data[pos] = mask * input_data_val * grad_out_data_val;
        }
      }
    }
  };
  if (input.numel() > 1000) {
    at::parallel_for(0, input_dim0_size, 0, loop);
  } else {
    loop(0, input_dim0_size);
  }
}

std::tuple<Tensor, Tensor> prelu_backward_cpu(const Tensor& grad_out_, const Tensor& self, const Tensor& weight_) {
  auto input = self.contiguous();
  auto grad_out = grad_out_.contiguous();
  auto weight = weight_.contiguous();

  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(grad_out.is_contiguous());
  TORCH_CHECK(weight.is_contiguous());

  int64_t weight_num = weight.numel();
  auto strides = input.strides();
  auto dims = input.dim();

  Tensor input_grad = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  Tensor weight_grad = at::empty_like(weight, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  Tensor weight_grad_collector = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  // case1: shared parameter for all channels
  if (weight_num == 1) {
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "prelu_backward_cpu", [&] {
      prelu_cpu_backward_kernel_share_weights<scalar_t>(input, weight, grad_out, input_grad, weight_grad);
    });
  }
  else { // case2: multiple parameters, one for each channel
    int64_t input_ndim = input.dim();
    TORCH_CHECK(input_ndim > 0, "Not allow zero-dim input tensor.");

    int64_t channel_size = 1; // channel_size default to 1
    int64_t input_dim0_size = 1, input_stride0 = 1, input_stride1 = 1;

    if (input_ndim > 1) {
      channel_size = input.size(1); // channel is the 2nd dim of input
      input_dim0_size = input.size(0);
      input_stride0 = strides[0];
      input_stride1 = strides[1];
    }
    TORCH_CHECK(channel_size == weight_num,
      "Mismatch of parameter numbers and input channel size. Found parameter numbers = ", weight_num,
      " and channel size = ", channel_size, ".");

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "prelu_backward_cpu", [&] {
      prelu_cpu_backward_kernel_multi_weights<scalar_t>(
        input,
        weight,
        grad_out,
        input_grad,
        weight_grad_collector,
        input_dim0_size,
        channel_size,
        input_stride0,
        input_stride1);
    });
    // update weight_grad
    std::vector<int64_t> reduce_dims;
    reduce_dims.push_back(0);
    if (dims > 2) {
      for(int64_t i = 2; i < dims; i++) reduce_dims.push_back(i);
    }
    weight_grad = weight_grad_collector.sum(reduce_dims);
  }
  return std::tuple<Tensor, Tensor>{input_grad, weight_grad};
}

// -----------------------------------
// hardshrink
// -----------------------------------
Tensor hardshrink(const Tensor & self, Scalar lambd) {
  auto out_tensor = at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto iter = TensorIterator::unary_op(out_tensor, self);
  hardshrink_stub(iter.device_type(), iter, lambd);
  return out_tensor;
}

Tensor hardshrink_backward(const Tensor & grad, const Tensor & self, Scalar lambd) {
  auto out_tensor = at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto iter = TensorIterator::binary_op(out_tensor, grad, self);
  shrink_backward_stub(iter.device_type(), iter, lambd);
  return out_tensor;
}

static inline void softshrink_check(Scalar lambd) {
  double lamb = lambd.to<double>();
  TORCH_CHECK(lamb >= 0, "lambda must be greater or equal to 0, but found to be ", lamb, ".");
}

Tensor& softshrink_out(Tensor& result, const Tensor & self, Scalar lambd) {
  softshrink_check(lambd);
  auto iter = TensorIterator::unary_op(result, self);
  softshrink_stub(iter.device_type(), iter, lambd);
  return result;
}

Tensor softshrink(const Tensor & self, Scalar lambd) {
  softshrink_check(lambd);
  Tensor result;
  auto iter = TensorIterator::unary_op(result, self);
  softshrink_stub(iter.device_type(), iter, lambd);
  return iter.output();
}

Tensor& softshrink_backward_out(Tensor& grad_input, const Tensor & grad, const Tensor & self, Scalar lambd) {
  auto iter = TensorIterator::binary_op(grad_input, grad, self);
  shrink_backward_stub(iter.device_type(), iter, lambd);
  return grad_input;
}

Tensor softshrink_backward(const Tensor & grad, const Tensor & self, Scalar lambd) {
  Tensor result;
  auto iter = TensorIterator::binary_op(result, grad, self);
  shrink_backward_stub(iter.device_type(), iter, lambd);
  return iter.output();
}

Tensor gelu_cpu(const Tensor& self) {
  Tensor Y = at::native::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto it = TensorIterator::unary_op(Y, self);
  GeluKernel(kCPU, it);
  return Y;
}

Tensor gelu_backward_cpu(const Tensor& grad, const Tensor& self) {
  Tensor dX = at::native::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto it = TensorIterator::binary_op(dX, grad, self);
  GeluBackwardKernel(kCPU, it);
  return dX;
}

Tensor& leaky_relu_out(
    Tensor& result,
    const Tensor& self,
    Scalar negval) {
  auto iter = TensorIterator::unary_op(result, self);
  leaky_relu_stub(iter.device_type(), iter, negval);
  return result;
}

Tensor leaky_relu(
    const Tensor& self,
    Scalar negval) {
  Tensor result;
  auto iter = TensorIterator::unary_op(result, self);
  leaky_relu_stub(iter.device_type(), iter, negval);
  return iter.output();
}

Tensor & leaky_relu_(
    Tensor & self,
    Scalar neg_val) {
  return at::leaky_relu_out(self, self, neg_val);
}

// Note: leakyReLu backward calculation doesn't support in-place call with non-positive slope.
// The reason is that for in-place forward call, the forward result will be saved into autograd
// node instead of the input itself, when calculating backward gradient, there is no way to know
// whether the original input for current node is positive or not if the input slope is
// non-positive. eg. forward is 2, slope is -0.2, the original input for this node could be
// either 2, or -10, so no way to get a correct backward gradient in this case.
Tensor leaky_relu_backward(
    const Tensor& grad_output,
    const Tensor& self_or_result,
    Scalar negval,
    bool is_result) {
  TORCH_CHECK(
    !is_result || negval.to<double>() > 0.0,
    "In-place leakyReLu backward calculation is triggered with a non-positive slope which is not supported. "
    "This is caused by calling in-place forward function with a non-positive slope, "
    "please call out-of-place version instead. File an issue at https://github.com/pytorch/pytorch if you do "
    "require supporting in-place leakRelu backward calculation with non-positive slope");

  Tensor result;
  auto iter = TensorIterator::binary_op(result, self_or_result, grad_output);
  leaky_relu_backward_stub(iter.device_type(), iter, negval);
  return iter.output();
}

std::tuple<Tensor, Tensor> log_sigmoid_forward_cpu(const Tensor& input) {
  auto result = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto buffer = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  log_sigmoid_cpu_stub(kCPU, result, buffer, input.contiguous());
  return std::make_tuple(result, buffer);
}

std::tuple<Tensor&, Tensor&> log_sigmoid_forward_out_cpu(Tensor& result, Tensor& buffer, const Tensor& input) {
  log_sigmoid_cpu_stub(kCPU, result, buffer, input);
  return std::forward_as_tuple(result, buffer);
}

Tensor log_sigmoid_backward_cpu(const Tensor& grad_output, const Tensor& input, const Tensor& buffer) {
  Tensor grad_input;
  auto iter = at::TensorIterator();
  iter.set_check_mem_overlap(true);
  iter.add_output(grad_input);
  iter.add_input(input);
  iter.add_input(buffer);
  iter.add_input(grad_output);
  iter.build();
  log_sigmoid_backward_cpu_stub(kCPU, iter);
  return iter.output();
}

Tensor& log_sigmoid_backward_out_cpu(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& buffer) {
  auto iter = at::TensorIterator();
  iter.set_check_mem_overlap(true);
  iter.add_output(grad_input);
  iter.add_input(input);
  iter.add_input(buffer);
  iter.add_input(grad_output);
  iter.build();
  log_sigmoid_backward_cpu_stub(kCPU, iter);
  return grad_input;
}

DEFINE_DISPATCH(GeluKernel);
DEFINE_DISPATCH(GeluBackwardKernel);

}}  // namespace at::native
