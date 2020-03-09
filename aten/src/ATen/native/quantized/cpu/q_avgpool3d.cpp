#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/native/Pool.h>
#include <ATen/native/quantized/cpu/init_qnnpack.h>
#include <ATen/native/quantized/cpu/qnnpack_utils.h>
#include <ATen/native/quantized/cpu/quantized_ops.h>
#include <caffe2/utils/threadpool/ThreadPoolMobile.h>
#include <c10/util/math_compat.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace at {
namespace native {

DEFINE_DISPATCH(qavg_pool3d_nhwc_stub);

namespace {

template <typename scalar_t>
static void avg_pool3d_out_frame(
    const Tensor& input,
    Tensor& output,
    int64_t b,
    int64_t nInputPlane,
    int64_t inputWidth,
    int64_t inputHeight,
    int64_t inputDepth,
    int64_t outputWidth,
    int64_t outputHeight,
    int64_t outputDepth,
    int kW,
    int kH,
    int kD,
    int dW,
    int dH,
    int dD,
    int padW,
    int padH,
    int padD,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  at::parallel_for(0, nInputPlane, 0, [&](int64_t start, int64_t end) {
    for (auto k = start; k < end; k++) {
      int64_t od, oh, ow;
      /* For all output pixels... */
      auto input_data = input.contiguous().data_ptr<scalar_t>();
      auto output_data = output.data_ptr<scalar_t>();
      scalar_t* ptr_output = output_data +
          b * nInputPlane * outputWidth * outputHeight * outputDepth +
          k * outputWidth * outputHeight * outputDepth;
      const scalar_t* ptr_input = input_data +
          b * nInputPlane * inputWidth * inputHeight * inputDepth +
          k * inputWidth * inputHeight * inputDepth;
      auto minimum =
          std::numeric_limits<typename scalar_t::underlying>::lowest();
      auto maximum = std::numeric_limits<typename scalar_t::underlying>::max();

      for (od = 0; od < outputDepth; od++) {
        for (oh = 0; oh < outputHeight; oh++) {
          for (ow = 0; ow < outputWidth; ow++) {
            /* Compute the mean of the input image... */
            int64_t dstart = od * dD - padD;
            int64_t hstart = oh * dH - padH;
            int64_t wstart = ow * dW - padW;
            int64_t dend = std::min(dstart + kD, inputDepth + padD);
            int64_t hend = std::min(hstart + kH, inputHeight + padH);
            int64_t wend = std::min(wstart + kW, inputWidth + padW);
            int64_t pool_size = (dend - dstart) * (hend - hstart) * (wend - wstart);
            dstart = std::max(dstart, static_cast<int64_t>(0));
            hstart = std::max(hstart, static_cast<int64_t>(0));
            wstart = std::max(wstart, static_cast<int64_t>(0));
            dend = std::min(dend, inputDepth);
            hend = std::min(hend, inputHeight);
            wend = std::min(wend, inputWidth);

            int sum_int = 0;
            ptr_output->val_ = 0;

            int64_t divide_factor;
            int64_t size = (dend - dstart) * (hend - hstart) * (wend - wstart);
            if (divisor_override.has_value()) {
              divide_factor = divisor_override.value();
            } else {
              if (count_include_pad) {
                divide_factor = pool_size;
              } else {
                divide_factor = (dend - dstart) * (hend - hstart) * (wend - wstart);
              }
            }

            int64_t kx, ky, kz;
            for (kz = dstart; kz < dend; kz++) {
              for (ky = hstart; ky < hend; ky++) {
                for (kx = wstart; kx < wend; kx++)
                  sum_int += (ptr_input + kz * inputHeight * inputWidth + ky * inputWidth + kx)->val_;
              }
            }
            float multiplier = input.q_scale() / output.q_scale() / divide_factor;

            sum_int -= size * input.q_zero_point();
            float sum = sum_int * 1.0;
            /* Update output by requantizing the result */
            ptr_output->val_ =
                static_cast<typename scalar_t::underlying>(std::min<int32_t>(
                    std::max<int32_t>(
                        std::nearbyint(sum * multiplier + output.q_zero_point()),
                        minimum),
                    maximum));
            ptr_output++;
          }
        }
      }
    }
  });
}

inline std::tuple<int, int, int> get_kernel(IntArrayRef kernel_size) {
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 3,
      "avg_pool3d: kernel_size must either be a single int, or a tuple of three ints");
  const int kD = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kH = kernel_size.size() == 1
      ? kD
      : safe_downcast<int, int64_t>(kernel_size[1]);
  const int kW = kernel_size.size() == 1
      ? kD
      : safe_downcast<int, int64_t>(kernel_size[2]);
  return std::make_tuple(kW, kH, kD);
}

inline std::tuple<int, int, int> get_stride(IntArrayRef stride, int kW, int kH, int kD) {
  TORCH_CHECK(
      stride.empty() || stride.size() == 1 || stride.size() == 3,
      "avg_pool3d: stride must either be omitted, a single int, or a tuple of three ints");
  const int dD = stride.empty() ? kD : safe_downcast<int, int64_t>(stride[0]);
  const int dH = stride.empty()
      ? kH
      : stride.size() == 1 ? dD : safe_downcast<int, int64_t>(stride[1]);
  const int dW = stride.empty()
      ? kW
      : stride.size() == 1 ? dD : safe_downcast<int, int64_t>(stride[2]);
  return std::make_tuple(dW, dH, dD);
}

inline std::tuple<int, int, int> get_padding(IntArrayRef padding) {
  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 3,
      "avg_pool3d: padding must either be a single int, or a tuple of three ints");
  const int padD = safe_downcast<int, int64_t>(padding[0]);
  const int padH =
      padding.size() == 1 ? padD : safe_downcast<int, int64_t>(padding[1]);
  const int padW =
      padding.size() == 1 ? padD : safe_downcast<int, int64_t>(padding[2]);
  return std::make_tuple(padW, padH, padD);
}

std::vector<int64_t> get_output_shape(
    const Tensor& input_,
    int kW,
    int kH,
    int kD,
    int dW,
    int dH,
    int dD,
    int padW,
    int padH,
    int padD,
    bool ceil_mode) {
  const int64_t nbatch = input_.ndimension() == 5 ? input_.size(-5) : 1;
  const int64_t nInputPlane = input_.size(-4);
  const int64_t inputDepth = input_.size(-3);
  const int64_t inputHeight = input_.size(-2);
  const int64_t inputWidth = input_.size(-1);
  const int64_t outputDepth =
      pooling_output_shape<int64_t>(inputDepth, kD, padD, dD, 1, ceil_mode);
  const int64_t outputHeight =
      pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, 1, ceil_mode);
  const int64_t outputWidth =
      pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, 1, ceil_mode);
  if (input_.ndimension() == 4) {
    return {nInputPlane, outputDepth, outputHeight, outputWidth};
  }
  return {nbatch, nInputPlane, outputDepth, outputHeight, outputWidth};
}

template <typename scalar_t>
Tensor q_avg_pool3d(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  int kD, kW, kH, dD, dW, dH, padD, padW, padH;
  std::tie(kW, kH, kD) = get_kernel(kernel_size);
  std::tie(dW, dH, dD) = get_stride(stride, kW, kH, kD);
  std::tie(padW, padH, padD) = get_padding(padding);

  const int64_t nbatch = input.ndimension() == 5 ? input.size(-5) : 1;
  const int64_t nInputPlane = input.size(-4);
  const int64_t inputDepth = input.size(-3);
  const int64_t inputHeight = input.size(-2);
  const int64_t inputWidth = input.size(-1);

  TORCH_CHECK(
      !divisor_override.has_value() || divisor_override.value() != 0,
      "divisor must be not zero");

  auto output_shape =
      get_output_shape(input, kW, kH, kD, dW, dH, dD, padW, padH, padD, ceil_mode);
  const int64_t outputDepth = output_shape[output_shape.size() - 3];
  const int64_t outputHeight = output_shape[output_shape.size() - 2];
  const int64_t outputWidth = output_shape[output_shape.size() - 1];

  if (input.is_contiguous(c10::MemoryFormat::ChannelsLast3d)) {
    auto output = at::_empty_affine_quantized(
        output_shape,
        input.options(),
        input.q_scale(),
        input.q_zero_point(),
        input.suggest_memory_format());
    // fast path for channel last: qavg_pool_2d_nhwc_stub
    if (output_shape.size() == 4) {
      qavg_pool3d_nhwc_stub(
          input.device().type(),
          input,
          output,
          0,
          nInputPlane,
          inputWidth,
          inputHeight,
          inputDepth,
          outputWidth,
          outputHeight,
          outputDepth,
          kW,
          kH,
          kD,
          dW,
          dH,
          dD,
          padW,
          padH,
          padD,
          count_include_pad,
          divisor_override);
    } else {
      at::parallel_for(0, nbatch, 0, [&](int64_t start, int64_t end) {
        for (auto b = start; b < end; b++) {
          qavg_pool3d_nhwc_stub(
              input.device().type(),
              input,
              output,
              b,
              nInputPlane,
              inputWidth,
              inputHeight,
              inputDepth,
              outputWidth,
              outputHeight,
              outputDepth,
              kW,
              kH,
              kD,
              dW,
              dH,
              dD,
              padW,
              padH,
              padD,
              count_include_pad,
              divisor_override);
        }
      });
    }
    return output;
  } else {
    auto output = at::_empty_affine_quantized(
        output_shape, input.options(), input.q_scale(), input.q_zero_point());
    if (output_shape.size() == 4) {
      avg_pool3d_out_frame<scalar_t>(
          input,
          output,
          0,
          nInputPlane,
          inputWidth,
          inputHeight,
          inputDepth,
          outputWidth,
          outputHeight,
          outputDepth,
          kW,
          kH,
          kD,
          dW,
          dH,
          dD,
          padW,
          padH,
          padD,
          count_include_pad,
          divisor_override);
    } else {
      at::parallel_for(0, nbatch, 0, [&](int64_t start, int64_t end) {
        for (auto b = start; b < end; b++) {
          avg_pool3d_out_frame<scalar_t>(
              input,
              output,
              b,
              nInputPlane,
              inputWidth,
              inputHeight,
              inputDepth,
              outputWidth,
              outputHeight,
              outputDepth,
              kW,
              kH,
              kD,
              dW,
              dH,
              dD,
              padW,
              padH,
              padD,
              count_include_pad,
              divisor_override);
        }
      });
    }
    return output;
  }
}

} // namespace

Tensor quantized_avg_pool3d(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  Tensor output;
  AT_DISPATCH_QINT_TYPES(input.scalar_type(), "quantized_avg_pool3d", [&]() {
    output = q_avg_pool3d<scalar_t>(
        input,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override);
  });
  return output;
}

} // namespace native
} // namespace at
