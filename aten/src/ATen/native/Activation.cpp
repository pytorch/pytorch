#include <ATen/native/Activation.h>

#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/Parallel.h>


namespace at { namespace native {

static const double SELU_ALPHA = 1.6732632423543772848170429916717;
static const double SELU_SCALE = 1.0507009873554804934193349852946;

DEFINE_DISPATCH(threshold_stub);
DEFINE_DISPATCH(hardshrink_cpu_stub);
DEFINE_DISPATCH(hardshrink_backward_cpu_stub);

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

Tensor rrelu(const Tensor & self, Scalar lower, Scalar upper, bool training, Generator* generator) {
  return at::rrelu_with_noise(self, at::empty_like(self), lower, upper, training, generator);
}

Tensor & rrelu_(Tensor & self, Scalar lower, Scalar upper, bool training, Generator* generator) {
  return at::rrelu_with_noise_(self, at::empty_like(self), lower, upper, training, generator);
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
  threshold_stub(iter->device_type(), *iter, threshold, value);
  return iter->output();
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
  auto result_data = result.data<scalar_t>();
  auto input_data = input.data<scalar_t>();
  auto weight_val = weight.data<scalar_t>()[0];

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

  int64_t input_numel = input.numel();
  scalar_t* result_data = result.data<scalar_t>();
  scalar_t* input_data = input.data<scalar_t>();
  scalar_t* weight_data = weight.data<scalar_t>();

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
  Tensor result = at::empty_like(input);
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
  auto input_data = input.data<scalar_t>();
  auto weight_val = weight.data<scalar_t>()[0];
  auto grad_out_data = grad_out.data<scalar_t>();
  auto input_grad_data = input_grad.data<scalar_t>();
  auto weight_grad_data = weight_grad.data<scalar_t>();

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

  int64_t input_numel = input.numel();
  auto input_data = input.data<scalar_t>();
  auto weight_data = weight.data<scalar_t>();
  auto grad_out_data = grad_out.data<scalar_t>();
  auto input_grad_data = input_grad.data<scalar_t>();
  auto weight_grad_collector_data = weight_grad_collector.data<scalar_t>();

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

  Tensor input_grad = at::empty_like(input);
  Tensor weight_grad = at::empty_like(weight);
  Tensor weight_grad_collector = at::empty_like(input);

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
Tensor hardshrink_cpu(const Tensor & self, Scalar lambd) {
  auto out_tensor = at::empty_like(self);
  auto iter = TensorIterator::unary_op(out_tensor, self);
  hardshrink_cpu_stub(kCPU, *iter, lambd);
  return out_tensor;
}

Tensor hardshrink_backward_cpu(const Tensor & grad, const Tensor & self, Scalar lambd) {
  auto out_tensor = at::empty_like(self);
  auto iter = TensorIterator::binary_op(out_tensor, grad, self);
  hardshrink_backward_cpu_stub(kCPU, *iter, lambd);
  return out_tensor;
}


Tensor gelu_cpu(const Tensor& self) {
  const auto X = self.contiguous();
  Tensor Y = at::native::empty_like(X);
  GeluKernel(kCPU, X, &Y);
  return Y;
}

Tensor gelu_cuda(const Tensor& self) {
  Tensor Y = at::native::empty_like(self);
  GeluKernel(kCUDA, self, &Y);
  return Y;
}

Tensor gelu_backward_cpu(const Tensor& grad, const Tensor& self) {
  const auto X = self.contiguous();
  Tensor dX = at::native::empty_like(X);
  GeluBackwardKernel(kCPU, grad.contiguous(), X, &dX);
  return dX;
}

Tensor gelu_backward_cuda(const Tensor& grad, const Tensor& self) {
  Tensor dX = at::native::empty_like(self);
  GeluBackwardKernel(kCUDA, grad, self, &dX);
  return dX;
}

DEFINE_DISPATCH(GeluKernel);
DEFINE_DISPATCH(GeluBackwardKernel);

}}  // namespace at::native
