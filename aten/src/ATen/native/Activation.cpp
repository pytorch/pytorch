#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/Activation.h>

#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorOperators.h>
#include <ATen/OpMathType.h>
#include <ATen/Parallel.h>
#include <ATen/ScalarOps.h>
#if defined(C10_MOBILE) && defined(USE_XNNPACK)
#include <ATen/native/xnnpack/Engine.h>
#endif
#include <ATen/core/DistributionsHelper.h>

#include <c10/util/irange.h>
#include <c10/core/ScalarType.h>
#if AT_ONEDNN_ENABLED()
#include <ATen/native/onednn/ONEDNNCommon.h>
#include <ATen/native/onednn/Utils.h>
#endif

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/celu_native.h>
#include <ATen/ops/clamp.h>
#include <ATen/ops/clamp_min.h>
#include <ATen/ops/elu.h>
#include <ATen/ops/elu_backward_native.h>
#include <ATen/ops/elu_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/gelu_backward_native.h>
#include <ATen/ops/gelu_native.h>
#include <ATen/ops/hardshrink_backward_native.h>
#include <ATen/ops/hardshrink_native.h>
#include <ATen/ops/hardsigmoid_backward_native.h>
#include <ATen/ops/hardsigmoid_native.h>
#include <ATen/ops/hardswish_backward_native.h>
#include <ATen/ops/hardswish_native.h>
#include <ATen/ops/hardtanh.h>
#include <ATen/ops/hardtanh_backward_native.h>
#include <ATen/ops/hardtanh_native.h>
#include <ATen/ops/infinitely_differentiable_gelu_backward_native.h>
#include <ATen/ops/leaky_relu.h>
#include <ATen/ops/leaky_relu_backward.h>
#include <ATen/ops/leaky_relu_backward_native.h>
#include <ATen/ops/leaky_relu_native.h>
#include <ATen/ops/log_sigmoid_backward_native.h>
#include <ATen/ops/log_sigmoid_forward.h>
#include <ATen/ops/log_sigmoid_forward_native.h>
#include <ATen/ops/log_sigmoid_native.h>
#include <ATen/ops/mish_backward_native.h>
#include <ATen/ops/mish_native.h>
#include <ATen/ops/prelu_native.h>
#include <ATen/ops/_prelu_kernel.h>
#include <ATen/ops/_prelu_kernel_native.h>
#include <ATen/ops/_prelu_kernel_backward_native.h>
#include <ATen/ops/relu6_native.h>
#include <ATen/ops/relu_native.h>
#include <ATen/ops/rrelu_native.h>
#include <ATen/ops/rrelu_with_noise.h>
#include <ATen/ops/rrelu_with_noise_backward_native.h>
#include <ATen/ops/rrelu_with_noise_native.h>
#include <ATen/ops/selu_native.h>
#include <ATen/ops/sigmoid.h>
#include <ATen/ops/silu_backward_native.h>
#include <ATen/ops/silu_native.h>
#include <ATen/ops/softplus.h>
#include <ATen/ops/softplus_backward_native.h>
#include <ATen/ops/softplus_native.h>
#include <ATen/ops/softshrink_backward_native.h>
#include <ATen/ops/softshrink_native.h>
#include <ATen/ops/tanh.h>
#include <ATen/ops/threshold_backward_native.h>
#include <ATen/ops/threshold_native.h>

#include <utility>
#endif

namespace at::meta {
// computes `result = self <= threshold ? value : other`
// other is `self` in threshold() and `grad` in threshold_backward()
TORCH_META_FUNC(threshold)(const Tensor& self, const Scalar& threshold, const Scalar& value) {
  const Tensor& result = maybe_get_output();
  build(TensorIteratorConfig()
    .set_check_mem_overlap(false)  // threshold is idempotent, so overlap is okay
    .add_output(result)
    .add_const_input(self)
    .add_const_input(self) // other
    .allow_cpu_scalars(true)
    .promote_inputs_to_common_dtype(true)
    .cast_common_dtype_to_outputs(true)
    .enforce_safe_casting_to_output(true));
}
// computes `result = self <= threshold ? value : other`
// other is `self` in threshold() and `grad` in threshold_backward()
TORCH_META_FUNC(threshold_backward)(const Tensor& grad, const Tensor& self, const Scalar& threshold) {
  const Tensor& gradInput = maybe_get_output();
  build(TensorIteratorConfig()
    .set_check_mem_overlap(false)  // threshold is idempotent, so overlap is okay
    .add_output(gradInput)
    .add_const_input(self)
    .add_const_input(grad)  // other
    .allow_cpu_scalars(true)
    .promote_inputs_to_common_dtype(true)
    .cast_common_dtype_to_outputs(true)
    .enforce_safe_casting_to_output(true));
}

TORCH_META_FUNC(elu) (
  const Tensor& self, const Scalar& alpha, const Scalar& scale, const Scalar& input_scale
) {
  build_unary_op(maybe_get_output(), self);
}

TORCH_META_FUNC(elu_backward) (
  const Tensor& grad_output,
  const Scalar& alpha,
  const Scalar& scale,
  const Scalar& input_scale,
  bool is_result,
  const Tensor& self_or_result
) {
  TORCH_CHECK(
    !is_result || alpha.to<double>() >= 0.0,
    "In-place elu backward calculation is triggered with a negative slope which is not supported. "
    "This is caused by calling in-place forward function with a negative slope, "
    "please call out-of-place version instead.");

  build_borrowing_binary_op(maybe_get_output(), grad_output, self_or_result);
}

TORCH_META_FUNC(silu) (const Tensor& self) {
  build_unary_op(maybe_get_output(), self);
}

TORCH_META_FUNC(silu_backward) (
  const Tensor& grad_output, const Tensor& input
) {
  build_borrowing_binary_op(maybe_get_output(), grad_output, input);
}

TORCH_META_FUNC(mish) (const Tensor& self) {
  build_unary_op(maybe_get_output(), self);
}

TORCH_META_FUNC(softplus) (
  const Tensor& self, const Scalar& beta, const Scalar& threshold
) {
  build_unary_op(maybe_get_output(), self);
}

TORCH_META_FUNC(softplus_backward) (
  const Tensor& grad_output,
  const Tensor& self,
  const Scalar& beta,
  const Scalar& threshold
) {
  build_borrowing_binary_op(maybe_get_output(), grad_output, self);
}

TORCH_META_FUNC(leaky_relu) (
  const Tensor& self, const Scalar& negval
) {
  build_unary_op(maybe_get_output(), self);
}

// Note: leakyReLu backward calculation doesn't support in-place call with negative slope.
// The reason is that for in-place forward call, the forward result will be saved into autograd
// node instead of the input itself, when calculating backward gradient, there is no way to know
// whether the original input for current node is positive or not if the input slope is
// negative. eg. forward is 2, slope is -0.2, the original input for this node could be
// either 2, or -10, so no way to get a correct backward gradient in this case.
TORCH_META_FUNC(leaky_relu_backward) (
  const Tensor& grad_output,
  const Tensor& self_or_result,
  const Scalar& negval,
  bool is_result
) {
  TORCH_CHECK(
    !is_result || negval.to<double>() >= 0.0,
    "In-place leakyReLu backward calculation is triggered with a negative slope which is not supported. "
    "This is caused by calling in-place forward function with a negative slope, "
    "please call out-of-place version instead. File an issue at https://github.com/pytorch/pytorch if you do "
    "require supporting in-place leakRelu backward calculation with negative slope");

  build_borrowing_binary_op(maybe_get_output(), self_or_result, grad_output);
}

TORCH_META_FUNC(hardsigmoid) (const Tensor& self) {
  build_unary_op(maybe_get_output(), self);
}

TORCH_META_FUNC(hardsigmoid_backward) (const Tensor& grad_output, const Tensor& self) {
  build_borrowing_binary_op(maybe_get_output(), grad_output, self);
}

TORCH_META_FUNC(hardshrink) (const Tensor & self, const Scalar& lambd) {
  build_unary_op(maybe_get_output(), self);
}

TORCH_META_FUNC(hardshrink_backward) (
  const Tensor & grad, const Tensor & self, const Scalar& lambd
) {
  build_borrowing_binary_op(maybe_get_output(), grad, self);
}

static inline void softshrink_check(const Scalar& lambd) {
  double lamb = lambd.to<double>();
  TORCH_CHECK(lamb >= 0, "lambda must be greater or equal to 0, but found to be ", lamb, ".");
}

TORCH_META_FUNC(softshrink) (
  const Tensor & self, const Scalar& lambd
) {
  softshrink_check(lambd);
  build_unary_op(maybe_get_output(), self);
}

TORCH_META_FUNC(softshrink_backward) (
  const Tensor & grad, const Tensor & self, const Scalar& lambd
) {
  build_borrowing_binary_op(maybe_get_output(), grad, self);
}

TORCH_META_FUNC(gelu) (const Tensor & self, std::string_view approximate) {
  build_unary_op(maybe_get_output(), self);
}

TORCH_META_FUNC(gelu_backward) (
  const Tensor& grad, const Tensor& self, std::string_view approximate
) {
  build_borrowing_binary_op(maybe_get_output(), grad, self);
}

} // namespace at::meta

namespace at::native {

static const double SELU_ALPHA = 1.6732632423543772848170429916717;
static const double SELU_SCALE = 1.0507009873554804934193349852946;

DEFINE_DISPATCH(elu_stub);
DEFINE_DISPATCH(elu_backward_stub);
DEFINE_DISPATCH(softplus_stub);
DEFINE_DISPATCH(softplus_backward_stub);
DEFINE_DISPATCH(log_sigmoid_cpu_stub);
DEFINE_DISPATCH(log_sigmoid_backward_stub);
DEFINE_DISPATCH(threshold_stub);
DEFINE_DISPATCH(hardtanh_backward_stub);
DEFINE_DISPATCH(hardsigmoid_stub);
DEFINE_DISPATCH(hardsigmoid_backward_stub);
DEFINE_DISPATCH(hardswish_stub);
DEFINE_DISPATCH(hardswish_backward_stub);
DEFINE_DISPATCH(hardshrink_stub);
DEFINE_DISPATCH(softshrink_stub);
DEFINE_DISPATCH(shrink_backward_stub);
DEFINE_DISPATCH(leaky_relu_stub);
DEFINE_DISPATCH(leaky_relu_backward_stub);
DEFINE_DISPATCH(silu_stub);
DEFINE_DISPATCH(silu_backward_stub);
DEFINE_DISPATCH(mish_stub);
DEFINE_DISPATCH(mish_backward_stub);
DEFINE_DISPATCH(prelu_stub);
DEFINE_DISPATCH(prelu_backward_stub);

TORCH_IMPL_FUNC(elu_out) (
  const Tensor& self, const Scalar& alpha, const Scalar& scale, const Scalar& input_scale, const Tensor& result
) {
  elu_stub(device_type(), *this, alpha, scale, input_scale);
}

TORCH_IMPL_FUNC(elu_backward_out) (
  const Tensor& grad_output,
  const Scalar& alpha,
  const Scalar& scale,
  const Scalar& input_scale,
  bool is_result,
  const Tensor& self_or_result,
  const Tensor& grad_input
) {
  elu_backward_stub(device_type(), *this, alpha, scale, input_scale, is_result);
}

TORCH_IMPL_FUNC(silu_out) (
  const Tensor& self, const Tensor& result
) {
  silu_stub(device_type(), *this);
}

TORCH_IMPL_FUNC(silu_backward_out) (
  const Tensor& grad_output, const Tensor& input, const Tensor& grad_input
) {
  silu_backward_stub(device_type(), *this);
}

TORCH_IMPL_FUNC(mish_out) (
  const Tensor& self, const Tensor& result
) {
  mish_stub(device_type(), *this);
}

TORCH_IMPL_FUNC(softplus_out) (
  const Tensor& self, const Scalar& beta, const Scalar& threshold, const Tensor& result
) {
  softplus_stub(device_type(), *this, beta, threshold);
}

TORCH_IMPL_FUNC(softplus_backward_out) (
  const Tensor& grad_output,
  const Tensor& self,
  const Scalar& beta,
  const Scalar& threshold,
  const Tensor& grad_input
) {
  softplus_backward_stub(device_type(), *this, beta, threshold);
}

TORCH_IMPL_FUNC(leaky_relu_out) (
  const Tensor& self, const Scalar& negval, const Tensor& result
) {
  leaky_relu_stub(device_type(), *this, negval);
}

TORCH_IMPL_FUNC(leaky_relu_backward_out) (
  const Tensor& grad_output,
  const Tensor& self_or_result,
  const Scalar& negval,
  bool is_result,
  const Tensor& grad_input
) {
  leaky_relu_backward_stub(device_type(), *this, negval);
}

TORCH_IMPL_FUNC(hardsigmoid_out) (
  const Tensor& self, const Tensor& result
) {
  hardsigmoid_stub(device_type(), *this);
}

TORCH_IMPL_FUNC(hardsigmoid_backward_out) (
  const Tensor& grad_output, const Tensor& self, const Tensor& grad_input
) {
  hardsigmoid_backward_stub(device_type(), *this);
}

TORCH_IMPL_FUNC(hardshrink_out) (
  const Tensor & self, const Scalar& lambd, const Tensor& result
) {
  hardshrink_stub(device_type(), *this, lambd);
}

TORCH_IMPL_FUNC(hardshrink_backward_out) (
  const Tensor & grad, const Tensor & self, const Scalar& lambd, const Tensor& grad_input
) {
  shrink_backward_stub(device_type(), *this, lambd);
}

TORCH_IMPL_FUNC(softshrink_out) (
  const Tensor & self, const Scalar& lambd, const Tensor& result
) {
  softshrink_stub(device_type(), *this, lambd);
}

TORCH_IMPL_FUNC(softshrink_backward_out) (
  const Tensor & grad, const Tensor & self, const Scalar& lambd, const Tensor& grad_input
) {
  shrink_backward_stub(device_type(), *this, lambd);
}

#if AT_ONEDNN_ENABLED()
static bool use_onednn(const Tensor& input) {
  if (!at::globalContext().userEnabledOnednn()) {
    return false;
  }
  if (!input.is_contiguous() || input.numel() <= 1) {
    return false;
  }
  return (input.is_onednn()) || // input is onednn Tensor
    (input.device().is_cpu() &&
    (((input.scalar_type() == kBFloat16) && onednn_bf16_device_check()) ||
    (input.scalar_type() == kFloat))); // input is dense layout and bfloat16/float32
}
#endif

TORCH_IMPL_FUNC(gelu_out_cpu) (
  const Tensor& self, std::string_view approximate, const Tensor& result
) {
auto approximate_type = get_gelutype_enum(approximate);
#if AT_ONEDNN_ENABLED()
  if (use_onednn(self) && (approximate_type == GeluType::None)) {
    const ideep::tensor& x = itensor_from_tensor(self, /*from_const_data_ptr*/true);
    ideep::tensor y = itensor_from_tensor(result);
    ideep::eltwise_forward::compute(
      x, y, ideep::algorithm::eltwise_gelu_erf, ideep::prop_kind::forward_training, /*alpha*/ 0.0);
#ifdef __aarch64__
  } else if (use_onednn(self) && (approximate_type == GeluType::Tanh)) {
    const ideep::tensor& x = itensor_from_tensor(self, /*from_const_data_ptr*/true);
    ideep::tensor y = itensor_from_tensor(result);
    ideep::eltwise_forward::compute(
      x, y, ideep::algorithm::eltwise_gelu_tanh, ideep::prop_kind::forward_training, /*alpha*/ 0.0);
#endif  // ifdef __aarch64__
  } else {
    GeluKernel(kCPU, *this, approximate_type);
  }
#else
  GeluKernel(kCPU, *this, approximate_type);
#endif
}

TORCH_IMPL_FUNC(gelu_backward_out_cpu) (
  const Tensor& grad, const Tensor& self, std::string_view approximate, const Tensor& grad_input
) {
auto approximate_type = get_gelutype_enum(approximate);
#if AT_ONEDNN_ENABLED()
  if (use_onednn(self) && (approximate_type == GeluType::None)) {
    const ideep::tensor& x = itensor_from_tensor(self, /*from_const_data_ptr*/true);
    ideep::tensor grady = itensor_from_tensor(grad, /*from_const_data_ptr*/true);
    ideep::tensor gradx = itensor_from_tensor(grad_input);
    ideep::eltwise_backward::compute(x, grady, gradx,
      ideep::algorithm::eltwise_gelu_erf, /*alpha*/ 0.0);
  } else {
    GeluBackwardKernel(kCPU, *this, approximate_type);
  }
#else
  GeluBackwardKernel(kCPU, *this, approximate_type);
#endif
}

Tensor hardtanh(const Tensor& self, const Scalar& min, const Scalar& max) {
  Tensor result = at::empty_like(self);
  return at::hardtanh_out(result, self, min, max);
}

Tensor& hardtanh_out(const Tensor& self, const Scalar& min, const Scalar& max, Tensor& result) {
  TORCH_CHECK(self.scalar_type() != at::kBool,
  "Bool inputs not supported for hardtanh");
  //preserve legacy behavior of boundaries not causing type promotion
  Scalar min_, max_;
  if (at::isIntegralType(self.scalar_type(), /*include_bool*/false)) {
    int64_t minval = min.toLong();
    int64_t maxval = max.toLong();
    TORCH_CHECK(self.dtype() != at::kByte || (minval >= 0 &&
       maxval >=0), "cannot do hardtanh on an unsigned type with negative limits");
    min_ = minval;
    max_ = maxval;
  } else {
    min_ = min;
    max_ = max;
  }
  return at::clamp_out(result, self, min_, max_);
}

Tensor& hardtanh_(Tensor& self, const Scalar& min, const Scalar& max) {
  return at::hardtanh_out(self, self, min, max);
}

Tensor& hardtanh_backward_out(const Tensor& grad_output, const Tensor& self, const Scalar& min, const Scalar& max, Tensor& grad_input) {
  auto iter = TensorIterator::borrowing_binary_op(grad_input, grad_output, self);
  hardtanh_backward_stub(iter.device_type(), iter, min, max);
  return grad_input;
}

Tensor hardtanh_backward(const Tensor& grad_output, const Tensor& self, const Scalar& min, const Scalar& max) {
  Tensor result;
  auto iter = TensorIterator::borrowing_binary_op(result, grad_output, self);
  hardtanh_backward_stub(iter.device_type(), iter, min, max);
  return iter.output();
}

Tensor hardswish(const Tensor& self) {
  #if defined(C10_MOBILE) && defined(USE_XNNPACK)
  if (xnnpack::use_hardswish(self)) {
    return xnnpack::hardswish(self);
  }
  #endif
  Tensor result;
  auto iter = TensorIterator::unary_op(result, self);
  hardswish_stub(iter.device_type(), iter);
  return iter.output();
}

Tensor& hardswish_out(const Tensor& self, Tensor& result) {
  auto iter = TensorIterator::unary_op(result, self);
  hardswish_stub(iter.device_type(), iter);
  return result;
}

Tensor& hardswish_(Tensor& self) {
  #if defined(C10_MOBILE) && defined(USE_XNNPACK)
  if (xnnpack::use_hardswish(self)) {
    xnnpack::hardswish_(self);
    return self;
  }
  #endif
  auto iter = TensorIterator::unary_op(self, self);
  hardswish_stub(iter.device_type(), iter);
  return self;
}

Tensor hardswish_backward(const Tensor& grad_output, const Tensor& self) {
  Tensor grad_input;
  auto iter = TensorIterator::borrowing_binary_op(grad_input, grad_output, self);
  hardswish_backward_stub(iter.device_type(), iter);
  return iter.output();
}

Tensor relu(const Tensor & self) {
  TORCH_CHECK(self.scalar_type() != at::kBool, "Boolean inputs not supported for relu");
  return at::clamp_min(self, 0);
}

Tensor & relu_(Tensor & self) {
  TORCH_CHECK(self.scalar_type() != at::kBool, "Boolean inputs not supported for relu");
  return at::clamp_min_(self, 0);
}

Tensor selu(const Tensor & self) {
  return at::elu(self, SELU_ALPHA, SELU_SCALE);
}

Tensor relu6(const Tensor & self) {
  return at::hardtanh(self, /*min_val=*/0, /*max_val=*/6);
}

Tensor & selu_(Tensor & self) {
  return at::elu_(self, SELU_ALPHA, SELU_SCALE);
}

Tensor & relu6_(Tensor & self) {
  return at::hardtanh_(self, /*min_val=*/0, /*max_val=*/6);
}

Tensor celu(const Tensor & self, const Scalar& alpha) {
  TORCH_CHECK(alpha.to<double>() != 0,
      "ZeroDivisionError: alpha cannot be 0 for CELU");
  double inv_alpha = 1. / alpha.to<double>();
  return at::elu(self, alpha, Scalar(1.0), Scalar(inv_alpha));
}

Tensor & celu_(Tensor & self, const Scalar& alpha) {
  TORCH_CHECK(alpha.to<double>() != 0,
      "ZeroDivisionError: alpha cannot be 0 for CELU");
  double inv_alpha = 1. / alpha.to<double>();
  return at::elu_(self, alpha, Scalar(1.0), Scalar(inv_alpha));
}

Tensor math_silu_backward(
    const Tensor& grad_output,
    const Tensor& input) {
  auto input_sigmoid = at::sigmoid(input);
  return grad_output * (input_sigmoid * (1 + input * (1 - input_sigmoid)));
}

Tensor mish_backward(
    const Tensor& grad_output,
    const Tensor& input) {
  Tensor grad_input = at::empty({0}, input.options());
  auto iter = TensorIterator::binary_op(grad_input, grad_output, input);
  mish_backward_stub(iter.device_type(), iter);
  return grad_input;
}

Tensor math_mish_backward(
    const Tensor& grad_output,
    const Tensor& input) {
  auto input_tanh_softplus = at::tanh(at::softplus(input));
  auto input_sigmoid = at::sigmoid(input);
  return grad_output * (input_tanh_softplus + (input * input_sigmoid * (1 - input_tanh_softplus * input_tanh_softplus)));
}

template <typename scalar_t>
inline void _rrelu_with_noise_train(
    Tensor& output,
    const Tensor& input,
    Tensor& noise,
    const Scalar& lower_,
    const Scalar& upper_,
    std::optional<Generator> generator) {
  using opmath_t = at::opmath_type<scalar_t>;
  opmath_t lower = lower_.to<opmath_t>();
  opmath_t upper = upper_.to<opmath_t>();
  Tensor tmp_tensor = output.contiguous();
  scalar_t* output_data = tmp_tensor.data_ptr<scalar_t>();
  const scalar_t* input_data = input.const_data_ptr<scalar_t>();
  scalar_t* noise_data = noise.data_ptr<scalar_t>();
  auto gen  = at::get_generator_or_default<CPUGeneratorImpl>(generator, detail::getDefaultCPUGenerator());
  std::lock_guard<std::mutex> lock(gen->mutex_);
  for (const auto i : c10::irange(input.numel())) {
    if (input_data[i] <= 0) {
      at::uniform_real_distribution<double> uniform(lower, upper);
      const opmath_t r = (opmath_t)uniform(gen);
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

Tensor& rrelu_with_noise_out_cpu(const Tensor& self,
    Tensor& noise,
    const Scalar& lower,
    const Scalar& upper,
    bool training,
    std::optional<Generator> generator,
    Tensor& output) {
  TORCH_CHECK(self.sym_sizes() == noise.sym_sizes(), "noise tensor shape must match self tensor shape. Got self.shape = ", self.sym_sizes(), " noise.shape = ", noise.sym_sizes());
  if (training) {
    AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::BFloat16, self.scalar_type(), "rrelu_with_noise_out_cpu", [&] {
      _rrelu_with_noise_train<scalar_t>(output, self.contiguous(), noise, lower, upper, generator);
    });
    return output;
  } else {
    auto lower_tensor = scalar_to_tensor(lower);
    auto upper_tensor = scalar_to_tensor(upper);
    auto negative = (lower_tensor + upper_tensor) / 2;
    Scalar negative_slope = negative.item();
    return at::leaky_relu_out(output, self, negative_slope);
  }
}

Tensor rrelu_with_noise_cpu(
    const Tensor& self,
    Tensor& noise,
    const Scalar& lower,
    const Scalar& upper,
    bool training,
    std::optional<Generator> generator) {
  auto output = at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  return at::native::rrelu_with_noise_out_cpu(
      self, noise, lower, upper, training, std::move(generator), output);
}

Tensor& rrelu_with_noise_cpu_(
    Tensor& self,
    Tensor& noise,
    const Scalar& lower,
    const Scalar& upper,
    bool training,
    std::optional<Generator> generator) {
  return at::native::rrelu_with_noise_out_cpu(
      self, noise, lower, upper, training, std::move(generator), self);
}

Tensor rrelu_with_noise_backward(
    const Tensor& grad_output,
    const Tensor& self_or_result,
    const Tensor& noise,
    const Scalar& lower,
    const Scalar& upper,
    bool training,
    bool is_result) {
  if (training) {
    return noise * grad_output;
  } else {
    auto l = lower.toDouble();
    auto u = upper.toDouble();
    auto mid = (l + u) / 2.;
    return at::leaky_relu_backward(grad_output, self_or_result, mid, is_result);
  }
}

Tensor rrelu(const Tensor & self, const Scalar& lower, const Scalar& upper, bool training, std::optional<Generator> generator) {
  TORCH_CHECK(lower.to<double>() <= upper.to<double>(), "Lower bound should be less than or equal to the upper bound")
  auto noise = at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  return at::rrelu_with_noise(self, noise, lower, upper, training, std::move(generator));
}

Tensor & rrelu_(Tensor & self, const Scalar& lower, const Scalar& upper, bool training, std::optional<Generator> generator) {
  TORCH_CHECK(lower.to<double>() <= upper.to<double>(), "Lower bound should be less than or equal to the upper bound")
  auto noise = at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  return at::rrelu_with_noise_(self, noise, lower, upper, training, std::move(generator));
}

TORCH_IMPL_FUNC(threshold_out)(const Tensor& self, const Scalar& threshold, const Scalar& value, const Tensor& result) {
  threshold_stub(device_type(), *this, threshold, value);
}

TORCH_IMPL_FUNC(threshold_backward_out)(const Tensor& grad, const Tensor& self, const Scalar& threshold, const Tensor& gradInput) {
  threshold_stub(device_type(), *this, threshold, 0);
}

Tensor prelu(const Tensor& self, const Tensor& weight_) {
  TORCH_INTERNAL_ASSERT(weight_.defined());
  auto self_dim = self.dim();
  TORCH_CHECK(self.scalar_type() == weight_.scalar_type(),
              "prelu: Type promoting not supported. Got ",
              self.scalar_type(), " and ", weight_.scalar_type());
  if (weight_.sym_numel() != 1) {
    TORCH_CHECK(self_dim > 0, "Not allow zero-dim input tensor.");

    auto channel_size = self_dim > 1 ? self.sym_size(1) : 1; // channel_size default to 1
    TORCH_CHECK(channel_size == weight_.sym_numel(),
      "Mismatch of parameter numbers and input channel size. Found parameter numbers = ", weight_.numel(),
      " and channel size = ", channel_size, ".");
  }

  TORCH_CHECK(
    weight_.dim() <= 1,
    "prelu: Expected `weight` to be a scalar or 1D tensor, but got: ndim = ", weight_.dim());
  // Adjust weight to broadcast over self and have weight.ndim == self.ndim
  auto weight = weight_;
  if (self_dim != weight.dim()) {
    SymDimVector dim_w(self_dim, 1);
    if (self_dim > 1) {
      dim_w[1] = weight_.sym_numel();
    }
    // This will always be a view in CPU/CUDA, but some backends
    // like ONEDNN do not support views
    weight = weight.reshape_symint(dim_w);
  }
  return at::_prelu_kernel(self, weight);
}


Tensor _prelu_kernel(const Tensor& self, const Tensor& weight) {
  // Weight broadcasts over self and they have the same dtype
  auto result = at::empty_like(self);
  auto iter = TensorIteratorConfig()
    .add_output(result)
    .add_const_input(self)
    .add_const_input(weight)
    .build();
  prelu_stub(iter.device_type(), iter);
  return result;
}

std::tuple<Tensor, Tensor> _prelu_kernel_backward(const Tensor& grad_out, const Tensor& self, const Tensor& weight) {
  Tensor grad_self = at::empty({0}, self.options());
  Tensor grad_weight = at::empty({0}, weight.options());
  auto iter = TensorIteratorConfig()
    .add_output(grad_self)
    .add_output(grad_weight)
    .add_const_input(self)
    .add_const_input(weight)
    .add_const_input(grad_out)
    .build();
  prelu_backward_stub(iter.device_type(), iter);
  return {grad_self, grad_weight};
}

Tensor infinitely_differentiable_gelu_backward(
    const Tensor& grad,
    const Tensor& self) {
  constexpr double kAlpha = M_2_SQRTPI * M_SQRT1_2 * 0.5;
  Tensor cdf = (1.0 + (self * M_SQRT1_2).erf_()).mul_(0.5);
  Tensor pdf = (-0.5 * self * self).exp_();
  return cdf.addcmul_(self, pdf, kAlpha).mul_(grad);
}

std::tuple<Tensor, Tensor> log_sigmoid_forward_cpu(const Tensor& input) {
  auto result = at::empty_like(input, at::MemoryFormat::Contiguous);
  auto buffer = at::empty_like(input, at::MemoryFormat::Contiguous);
  log_sigmoid_cpu_stub(kCPU, result, buffer, input.contiguous());
  return std::make_tuple(result, buffer);
}

std::tuple<Tensor&, Tensor&> log_sigmoid_forward_out_cpu(const Tensor& input, Tensor& result, Tensor& buffer) {
  result.resize_as_(input);
  buffer.resize_as_(input, at::MemoryFormat::Contiguous);
  TORCH_CHECK(buffer.is_contiguous(), "Contiguous buffer required for log_sigmoid with out parameter");
  Tensor result_tmp = result.is_contiguous() ? result : at::empty_like(result, at::MemoryFormat::Contiguous);
  log_sigmoid_cpu_stub(kCPU, result_tmp, buffer, input.contiguous());
  if (!result.is_contiguous()) {
    result.copy_(result_tmp);
  }
  return std::forward_as_tuple(result, buffer);
}

Tensor & log_sigmoid_out(const Tensor & self, Tensor & output) {
  Tensor buffer = at::empty({0}, self.options());
  return std::get<0>(at::log_sigmoid_forward_out(output, buffer, self));
}

Tensor log_sigmoid(const Tensor & self) {
  return std::get<0>(at::log_sigmoid_forward(self));
}

Tensor log_sigmoid_backward_cuda(const Tensor& grad_output, const Tensor& input, const Tensor& buffer) {
  auto grad_input = at::empty_like(grad_output);
  // NOTE: buffer is only used by CPU dispatch, we just ignore it here
  auto iter = at::TensorIteratorConfig()
      .add_output(grad_input)
      .add_const_input(input)
      .add_const_input(grad_output)
      .build();
  log_sigmoid_backward_stub(kCUDA, iter);
  return iter.output();
}

Tensor log_sigmoid_backward_cpu(const Tensor& grad_output, const Tensor& input, const Tensor& buffer) {
  auto grad_input = at::empty_like(grad_output);
  auto iter = at::TensorIteratorConfig()
      .add_output(grad_input)
      .add_const_input(input)
      .add_const_input(buffer)
      .add_const_input(grad_output)
      .build();
  log_sigmoid_backward_stub(kCPU, iter);
  return iter.output();
}

Tensor& log_sigmoid_backward_cuda_out(const Tensor& grad_output, const Tensor& input,
                                      const Tensor& buffer, Tensor& grad_input) {
  auto iter = TensorIteratorConfig()
      .add_output(grad_input)
      .add_const_input(input)
      .add_const_input(grad_output)
      .build();
  log_sigmoid_backward_stub(kCUDA, iter);
  return grad_input;
}

Tensor& log_sigmoid_backward_cpu_out(const Tensor& grad_output,
    const Tensor& input,
    const Tensor& buffer,
    Tensor& grad_input) {
  auto iter = TensorIteratorConfig()
      .add_output(grad_input)
      .add_const_input(input)
      .add_const_input(buffer)
      .add_const_input(grad_output)
      .build();
  log_sigmoid_backward_stub(kCPU, iter);
  return grad_input;
}

DEFINE_DISPATCH(GeluKernel);
DEFINE_DISPATCH(GeluBackwardKernel);

}  // namespace at::native
