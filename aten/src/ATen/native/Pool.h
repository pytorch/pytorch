#include <ATen/core/Tensor.h>
#include <ATen/div_rtn.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/PoolingChecks.h>
#include <c10/util/TypeCast.h>
#include <c10/util/safe_numerics.h>
#include <c10/util/irange.h>

#include <utility>

#pragma once

namespace at::native {

using max_pool2d_fn = void(*)(const Tensor& output, const Tensor& indices, const Tensor& input,
    int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH);
using max_pool2d_backward_fn = void(*)(const Tensor& grad_input, const Tensor& grad_output, const Tensor& indices);

DECLARE_DISPATCH(max_pool2d_fn, max_pool2d_kernel)
DECLARE_DISPATCH(max_pool2d_backward_fn, max_pool2d_backward_kernel)

// average pooling has same signature for forward and backward
using avg_pool2d_fn = void(*)(const Tensor& output, const Tensor& input, int64_t kW, int64_t kH,
    int64_t dW, int64_t dH, int64_t padW, int64_t padH, bool count_include_pad, std::optional<int64_t> divisor_override);
using avg_pool2d_backward_fn = void(*)(const Tensor& output, const Tensor& input, int kW, int kH,
    int dW, int dH, int padW, int padH, bool count_include_pad, std::optional<int64_t> divisor_override);

DECLARE_DISPATCH(avg_pool2d_fn, avg_pool2d_kernel)
DECLARE_DISPATCH(avg_pool2d_backward_fn, avg_pool2d_backward_kernel)

// average pooling has same signature for forward and backward
using avg_pool3d_fn = void(*)(const Tensor& output, const Tensor& input,
    int64_t kW, int64_t kH, int64_t kD, int64_t dW, int64_t dH, int64_t dD,
    int64_t padW, int64_t padH, int64_t padD, bool count_include_pad,
    std::optional<int64_t> divisor_override);
using avg_pool3d_backward_fn = void(*)(const Tensor& output, const Tensor& input,
    int kW, int kH, int kD, int dW, int dH, int dD,
    int padW, int padH, int padD, bool count_include_pad,
    std::optional<int64_t> divisor_override);

DECLARE_DISPATCH(avg_pool3d_fn, avg_pool3d_kernel)
DECLARE_DISPATCH(avg_pool3d_backward_fn, avg_pool3d_backward_kernel)

using max_pool3d_fn = void(*)(Tensor& output, Tensor& indices, const Tensor& input,
    int kW, int kH, int kD, int dW, int dH, int dD, int pW, int pH, int pD, int dilationW, int dilationH, int dilationD);
using max_pool3d_backward_fn = void(*)(Tensor& grad_input, const Tensor& grad_output, const Tensor& indices);

DECLARE_DISPATCH(max_pool3d_fn, max_pool3d_kernel)
DECLARE_DISPATCH(max_pool3d_backward_fn, max_pool3d_backward_kernel)
namespace {

template <typename dest_t, typename src_t>
inline dest_t
safe_downcast(src_t v)
{
  return c10::checked_convert<dest_t>(v, "dest_t");
}

// dilation * (kernelSize - 1) + 1, i.e. how far the kernel reaches into the
// input. Overflow wraps this negative, which makes every shape check computed
// from it meaningless, so reject it instead.
template<typename T>
inline T effective_kernel_size(T kernelSize, T dilation) {
    T size = 0;
    bool overflow = c10::add_overflows(kernelSize, T(-1), &size);
    overflow |= c10::mul_overflows(size, dilation, &size);
    overflow |= c10::add_overflows(size, T(1), &size);
    TORCH_CHECK(!overflow,
                "effective kernel size overflows, but got kernel_size=",
                kernelSize, " and dilation=", dilation);
    return size;
}

template<typename T>
inline T pooling_output_shape_pad_lr(
        T inputSize, T kernelSize, T pad_l, T pad_r, T stride, T dilation,
        bool ceil_mode) {
    T outputSize = div_rtn<T>(
        inputSize + pad_l + pad_r - dilation * (kernelSize - 1) - 1 +
        (ceil_mode ? stride - 1 : 0), stride) + 1;
    if (ceil_mode) {
        // ensure that the last pooling starts inside the image
        // needed to avoid problems in ceil mode
        if ((outputSize - 1) * stride >= inputSize + pad_l) {
          --outputSize;
        }
    }
    return outputSize;
}

template<typename T>
inline T pooling_output_shape(
      T inputSize, T kernelSize, T pad, T stride, T dilation, bool ceil_mode) {
    TORCH_CHECK(stride != 0, "stride should not be zero");
    TORCH_CHECK(pad >= 0,
                "pad must be non-negative, but got pad: ", pad);
    TORCH_CHECK(pad <= effective_kernel_size(kernelSize, dilation) / 2,
                "pad should be at most half of effective kernel size, but got pad=",
                pad, ", kernel_size=", kernelSize, " and dilation=", dilation)
    return pooling_output_shape_pad_lr(
        inputSize, kernelSize, pad, pad, stride, dilation, ceil_mode);
}

template <typename T>
std::pair<T, T> _pooling_same_mode_padding_lr(
    T inputSize, T kernelSize, T stride, T dilation) {
  // NOTE: with strides, the output shape is ceil(inputSize/stride)
  auto total_padding = T(dilation) * (kernelSize - 1);

  // Prefer symmetric padding if possible
  if (stride > 2 && (total_padding % 2 == 1)) {
    // The floor in the output size calculation gives us a little wiggle room
    auto wiggle_room = inputSize % stride - 1;
    if (wiggle_room > 0) {
      total_padding = total_padding - 1;
    }
  }

  auto left = total_padding / 2;
  auto right = total_padding - left;
  return {std::move(left), std::move(right)};
}

inline std::pair<int64_t, int64_t> pooling_same_mode_padding_lr(
    int64_t inputSize, int64_t kernelSize, int64_t stride, int64_t dilation) {
  return _pooling_same_mode_padding_lr(inputSize, kernelSize, stride, dilation);
}

inline std::pair<c10::SymInt, c10::SymInt> pooling_same_mode_padding_lr(
    c10::SymInt inputSize, c10::SymInt kernelSize, c10::SymInt stride, c10::SymInt dilation) {
  return _pooling_same_mode_padding_lr(std::move(inputSize), std::move(kernelSize), std::move(stride), std::move(dilation));
}

} // anonymous namespace

} // namespace at::native
