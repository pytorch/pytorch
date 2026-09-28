#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/ConvolutionLocal2d.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_conv2d_local_backward_native.h>
#include <ATen/ops/_conv2d_local_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/sum.h>
#endif

namespace at::native {
namespace {

template <typename scalar_t>
void conv2d_local_forward_cpu_kernel(
    const scalar_t* input,
    const scalar_t* weight,
    const scalar_t* bias,
    scalar_t* output,
    const Conv2dLocalParams& p) {
  using opmath_t = at::opmath_type<scalar_t>;
  const int64_t kernel_numel = p.kernel_numel;
  const int64_t numel = p.batch * p.out_channels * p.out_numel_per_channel;
  at::parallel_for(0, numel, at::internal::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
    for (const auto i : c10::irange(begin, end)) {
      const int64_t ow = i % p.out_width;
      const int64_t oh = (i / p.out_width) % p.out_height;
      const int64_t oc = (i / (p.out_width * p.out_height)) % p.out_channels;
      const int64_t n = i / (p.out_width * p.out_height * p.out_channels);
      const scalar_t* w = weight + ((oh * p.out_width + ow) * p.out_channels + oc) * kernel_numel;
      const scalar_t* x = input + n * p.in_channels * p.in_height * p.in_width;
      opmath_t acc = bias ? static_cast<opmath_t>(bias[(oc * p.out_height + oh) * p.out_width + ow]) : opmath_t(0);
      for (const auto ic : c10::irange(p.in_channels)) {
        for (const auto kh : c10::irange(p.kernel_height)) {
          const int64_t ih = oh * p.stride_height - p.pad_height + kh * p.dilation_height;
          if (ih < 0 || ih >= p.in_height) {
            continue;
          }
          for (const auto kw : c10::irange(p.kernel_width)) {
            const int64_t iw = ow * p.stride_width - p.pad_width + kw * p.dilation_width;
            if (iw < 0 || iw >= p.in_width) {
              continue;
            }
            acc += static_cast<opmath_t>(w[(ic * p.kernel_height + kh) * p.kernel_width + kw]) *
                static_cast<opmath_t>(x[(ic * p.in_height + ih) * p.in_width + iw]);
          }
        }
      }
      output[i] = static_cast<scalar_t>(acc);
    }
  });
}

template <typename scalar_t>
void conv2d_local_grad_input_cpu_kernel(
    const scalar_t* grad_output,
    const scalar_t* weight,
    scalar_t* grad_input,
    const Conv2dLocalParams& p) {
  using opmath_t = at::opmath_type<scalar_t>;
  const int64_t kernel_numel = p.kernel_numel;
  const int64_t numel = p.batch * p.in_channels * p.in_height * p.in_width;
  at::parallel_for(0, numel, at::internal::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
    for (const auto i : c10::irange(begin, end)) {
      const int64_t iw = i % p.in_width;
      const int64_t ih = (i / p.in_width) % p.in_height;
      const int64_t ic = (i / (p.in_width * p.in_height)) % p.in_channels;
      const int64_t n = i / (p.in_width * p.in_height * p.in_channels);
      const scalar_t* go = grad_output + n * p.out_channels * p.out_height * p.out_width;
      opmath_t acc(0);
      for (const auto kh : c10::irange(p.kernel_height)) {
        const int64_t oh_num = ih + p.pad_height - kh * p.dilation_height;
        if (oh_num < 0 || oh_num % p.stride_height != 0) {
          continue;
        }
        const int64_t oh = oh_num / p.stride_height;
        if (oh >= p.out_height) {
          continue;
        }
        for (const auto kw : c10::irange(p.kernel_width)) {
          const int64_t ow_num = iw + p.pad_width - kw * p.dilation_width;
          if (ow_num < 0 || ow_num % p.stride_width != 0) {
            continue;
          }
          const int64_t ow = ow_num / p.stride_width;
          if (ow >= p.out_width) {
            continue;
          }
          const scalar_t* w = weight + (oh * p.out_width + ow) * p.out_channels * kernel_numel +
              (ic * p.kernel_height + kh) * p.kernel_width + kw;
          for (const auto oc : c10::irange(p.out_channels)) {
            acc += static_cast<opmath_t>(w[oc * kernel_numel]) *
                static_cast<opmath_t>(go[(oc * p.out_height + oh) * p.out_width + ow]);
          }
        }
      }
      grad_input[i] = static_cast<scalar_t>(acc);
    }
  });
}

template <typename scalar_t>
void conv2d_local_grad_weight_cpu_kernel(
    const scalar_t* grad_output,
    const scalar_t* input,
    scalar_t* grad_weight,
    const Conv2dLocalParams& p) {
  using opmath_t = at::opmath_type<scalar_t>;
  const int64_t numel = p.out_numel_per_channel * p.out_channels * p.kernel_numel;
  const int64_t input_batch_stride = p.in_channels * p.in_height * p.in_width;
  const int64_t output_batch_stride = p.out_channels * p.out_height * p.out_width;
  at::parallel_for(0, numel, at::internal::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
    for (const auto i : c10::irange(begin, end)) {
      int64_t rest = i;
      const int64_t kw = rest % p.kernel_width;
      rest /= p.kernel_width;
      const int64_t kh = rest % p.kernel_height;
      rest /= p.kernel_height;
      const int64_t ic = rest % p.in_channels;
      rest /= p.in_channels;
      const int64_t oc = rest % p.out_channels;
      rest /= p.out_channels;
      const int64_t ow = rest % p.out_width;
      const int64_t oh = rest / p.out_width;
      const int64_t ih = oh * p.stride_height - p.pad_height + kh * p.dilation_height;
      const int64_t iw = ow * p.stride_width - p.pad_width + kw * p.dilation_width;
      opmath_t acc(0);
      if (ih >= 0 && ih < p.in_height && iw >= 0 && iw < p.in_width) {
        const scalar_t* x = input + (ic * p.in_height + ih) * p.in_width + iw;
        const scalar_t* go = grad_output + (oc * p.out_height + oh) * p.out_width + ow;
        for (const auto n : c10::irange(p.batch)) {
          acc += static_cast<opmath_t>(go[n * output_batch_stride]) * static_cast<opmath_t>(x[n * input_batch_stride]);
        }
      }
      grad_weight[i] = static_cast<scalar_t>(acc);
    }
  });
}

} // namespace

Tensor conv2d_local_cpu(
    const Tensor& self,
    const Tensor& weight,
    const std::optional<Tensor>& bias_opt,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
  c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;
  const auto p = conv2d_local_shape_check(self, weight, bias, Tensor(), stride, padding, dilation);
  auto input_c = self.contiguous();
  auto weight_c = weight.contiguous();
  auto bias_c = bias.defined() ? bias.contiguous() : bias;
  auto output = at::empty({p.batch, p.out_channels, p.out_height, p.out_width}, self.options());
  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, self.scalar_type(), "conv2d_local_cpu", [&] {
    if (output.numel() == 0) {
      return;
    }
    conv2d_local_forward_cpu_kernel<scalar_t>(
        input_c.const_data_ptr<scalar_t>(),
        weight_c.const_data_ptr<scalar_t>(),
        bias_c.defined() ? bias_c.const_data_ptr<scalar_t>() : nullptr,
        output.mutable_data_ptr<scalar_t>(),
        p);
  });
  return output;
}

std::tuple<Tensor, Tensor, Tensor> conv2d_local_backward_cpu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    std::array<bool, 3> output_mask) {
  const auto p = conv2d_local_shape_check(self, weight, Tensor(), grad_output, stride, padding, dilation);
  auto grad_output_c = grad_output.contiguous();
  Tensor grad_input, grad_weight, grad_bias;
  if (output_mask[0]) {
    auto weight_c = weight.contiguous();
    grad_input = at::empty(self.sizes(), self.options());
    if (grad_input.numel() > 0) {
      AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, self.scalar_type(), "conv2d_local_backward_cpu", [&] {
        conv2d_local_grad_input_cpu_kernel<scalar_t>(
            grad_output_c.const_data_ptr<scalar_t>(),
            weight_c.const_data_ptr<scalar_t>(),
            grad_input.mutable_data_ptr<scalar_t>(),
            p);
      });
    }
  }
  if (output_mask[1]) {
    auto input_c = self.contiguous();
    grad_weight = at::empty(weight.sizes(), weight.options());
    if (grad_weight.numel() > 0) {
      AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, self.scalar_type(), "conv2d_local_backward_cpu", [&] {
        conv2d_local_grad_weight_cpu_kernel<scalar_t>(
            grad_output_c.const_data_ptr<scalar_t>(),
            input_c.const_data_ptr<scalar_t>(),
            grad_weight.mutable_data_ptr<scalar_t>(),
            p);
      });
    }
  }
  if (output_mask[2]) {
    grad_bias = at::sum(grad_output_c, IntArrayRef{0});
  }
  return std::make_tuple(std::move(grad_input), std::move(grad_weight), std::move(grad_bias));
}

} // namespace at::native
