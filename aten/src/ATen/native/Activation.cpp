#include <ATen/native/Activation.h>

#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/TensorIterator.h>


namespace at { namespace native {

static const double SELU_ALPHA = 1.6732632423543772848170429916717;
static const double SELU_SCALE = 1.0507009873554804934193349852946;

DEFINE_DISPATCH(threshold_stub);

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
  return at::elu(self, 1.0, alpha, Scalar(inv_alpha));
}

Tensor & celu_(Tensor & self, Scalar alpha) {
  double inv_alpha = 1. / alpha.to<double>();
  return at::elu_(self, 1.0, alpha, Scalar(inv_alpha));
}

Tensor rrelu(const Tensor & self, Scalar lower, Scalar upper, bool training, Generator* generator) {
  return at::rrelu_with_noise(self, at::empty({0}, self.options()), lower, upper, training, generator);
}

Tensor & rrelu_(Tensor & self, Scalar lower, Scalar upper, bool training, Generator* generator) {
  return at::rrelu_with_noise_(self, at::empty({0}, self.options()), lower, upper, training, generator);
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

  int64_t i;
  int64_t input_numel = input.numel();
  auto result_data = result.data<scalar_t>();
  auto input_data = input.data<scalar_t>();
  auto weight_val = weight.data<scalar_t>()[0];

  #pragma omp parallel for private(i) if (input_numel > 1000)
  for (i = 0; i < input_numel; i++) {
    scalar_t input_data_val = input_data[i];
    // to allow for compiler optimization, here splitting into two lines:
    scalar_t r = (input_data_val > 0) ? scalar_t(1) : weight_val;
    result_data[i] = r * input_data_val;
  }
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

  int64_t i, j, k;
  int64_t input_numel = input.numel();
  scalar_t* result_data = result.data<scalar_t>();
  scalar_t* input_data = input.data<scalar_t>();
  scalar_t* weight_data = weight.data<scalar_t>();

  #pragma omp parallel for private(i,j,k) if (input.numel() > 1000)
  for (i = 0; i < input_dim0_size; ++i) {
    int64_t offset = i * channel_size * input_stride1;
    scalar_t* n_input_data = input_data + offset;
    scalar_t* n_result_data = result_data + offset;
    for (j = 0; j < channel_size; ++j) {
      for (k = 0; k < input_stride1; ++k) {
        // to allow for compiler optimization, here splitting into two lines:
        scalar_t w = (n_input_data[k] > 0) ? scalar_t(1) : weight_data[j];
        n_result_data[k] = w * n_input_data[k];
      }
      n_input_data += input_stride1;
      n_result_data += input_stride1;
    }
  }
}

Tensor prelu_cpu(const Tensor& self, const Tensor& weight_) {
  auto input = self.contiguous();
  auto weight = weight_.contiguous();

  AT_CHECK(input.is_contiguous());
  AT_CHECK(weight.is_contiguous());

  int64_t weight_num = weight.numel();
  Tensor result = at::empty_like(input);
  auto strides = input.strides();

  // case1: shared weight for all channels
  if (weight_num == 1) {
    AT_DISPATCH_FLOATING_TYPES(input.type(), "prelu_cpu", [&] {
      prelu_cpu_kernel_share_weights<scalar_t>(result, input, weight);
    });
  }
  else { // case2: multiple weights, one for each channel
    int64_t input_ndim = input.dim();
    AT_CHECK(input_ndim > 0, "Not allow zero-dim input tensor.");

    int64_t channel_size = 1; // channel_size default to 1
    int64_t input_dim0_size = 1, input_stride0 = 1, input_stride1 = 1;

    if (input_ndim > 1) {
      channel_size = input.size(1); // channel is the 2nd dim of input
      input_dim0_size = input.size(0);
      input_stride0 = strides[0];
      input_stride1 = strides[1];
    }
    AT_CHECK(channel_size == weight_num,
      "Mismatch of parameter numbers and input channel size. Found parameter numbers = %d, and channel size = %d.",
      weight_num, channel_size);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "prelu_cpu", [&] {
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

  int64_t i;
  int64_t input_numel = input.numel();
  scalar_t sum = 0;
  auto input_data = input.data<scalar_t>();
  auto weight_val = weight.data<scalar_t>()[0];
  auto grad_out_data = grad_out.data<scalar_t>();
  auto input_grad_data = input_grad.data<scalar_t>();
  auto weight_grad_data = weight_grad.data<scalar_t>();

  #pragma omp parallel for private(i) reduction(+:sum) if (input_numel > 1000)
  for (i = 0; i < input_numel; i++) {
    scalar_t input_data_val = input_data[i];
    scalar_t grad_out_data_val = grad_out_data[i];
    // to allow for compiler optimization, here splitting into two lines:
    scalar_t w = (input_data_val > 0) ? scalar_t(1) : weight_val;
    input_grad_data[i] = w * grad_out_data_val;
    // to allow for compiler optimization, here splitting into two lines:
    scalar_t mask = (input_data_val > 0) ? scalar_t(0) : scalar_t(1);
    sum += mask * input_data_val * grad_out_data_val;
  }
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

  int64_t i, j, k;
  int64_t input_numel = input.numel();
  auto input_data = input.data<scalar_t>();
  auto weight_data = weight.data<scalar_t>();
  auto grad_out_data = grad_out.data<scalar_t>();
  auto input_grad_data = input_grad.data<scalar_t>();
  auto weight_grad_collector_data = weight_grad_collector.data<scalar_t>();

  #pragma omp parallel for private(i, j, k) if (input.numel() > 1000)
  for (i = 0; i < input_dim0_size; i++) {
    for (j = 0; j < channel_size; j++) {
      for (k = 0; k < input_stride1; k++) {
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
}

std::tuple<Tensor, Tensor> prelu_backward_cpu(const Tensor& grad_out_, const Tensor& self, const Tensor& weight_) {
  auto input = self.contiguous();
  auto grad_out = grad_out_.contiguous();
  auto weight = weight_.contiguous();

  AT_CHECK(input.is_contiguous());
  AT_CHECK(grad_out.is_contiguous());
  AT_CHECK(weight.is_contiguous());

  int64_t weight_num = weight.numel();
  auto strides = input.strides();
  auto dims = input.dim();

  Tensor input_grad = at::empty_like(input);
  Tensor weight_grad = at::empty_like(weight);
  Tensor weight_grad_collector = at::empty_like(input);

  // case1: shared parameter for all channels
  if (weight_num == 1) {
    AT_DISPATCH_FLOATING_TYPES(input.type(), "prelu_backward_cpu", [&] {
      prelu_cpu_backward_kernel_share_weights<scalar_t>(input, weight, grad_out, input_grad, weight_grad);
    });
  }
  else { // case2: multiple parameters, one for each channel
    int64_t input_ndim = input.dim();
    AT_CHECK(input_ndim > 0, "Not allow zero-dim input tensor.");

    int64_t channel_size = 1; // channel_size default to 1
    int64_t input_dim0_size = 1, input_stride0 = 1, input_stride1 = 1;

    if (input_ndim > 1) {
      channel_size = input.size(1); // channel is the 2nd dim of input
      input_dim0_size = input.size(0);
      input_stride0 = strides[0];
      input_stride1 = strides[1];
    }
    AT_CHECK(channel_size == weight_num,
      "Mismatch of parameter numbers and input channel size. Found parameter numbers = %d, and channel size = %d.",
      weight_num, channel_size);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "prelu_backward_cpu", [&] {
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
  AT_DISPATCH_FLOATING_TYPES(self.type(), "hardshrink_cpu", [&] {
    auto lambd_val = lambd.to<scalar_t>();
    at::CPU_tensor_apply2<scalar_t, scalar_t>(
      self,
      out_tensor,
      [&](
        scalar_t& self_val,
        scalar_t& out_tensor_val) {
          out_tensor_val = (self_val >= -lambd_val && self_val <= lambd_val) ? scalar_t(0) : self_val;
    });
  });
  return out_tensor;
}

Tensor hardshrink_backward_cpu(const Tensor & grad, const Tensor & self, Scalar lambd) {
  auto out_tensor = at::empty_like(self);
  AT_DISPATCH_FLOATING_TYPES(self.type(), "hardshrink_backward_cpu", [&] {
    auto lambd_val = lambd.to<scalar_t>();
    at::CPU_tensor_apply3<scalar_t, scalar_t, scalar_t>(
      self,
      grad,
      out_tensor,
      [&](
        scalar_t& self_val,
        scalar_t& grad_val,
        scalar_t& out_tensor_val) {
          out_tensor_val = (self_val >= -lambd_val && self_val <= lambd_val) ? scalar_t(0) : grad_val;
    });
  });
  return out_tensor;
}

}}  // namespace at::native
