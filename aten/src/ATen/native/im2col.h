#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at {
namespace native {

using contig_fn = void (*)(
  Tensor& im,
  const Tensor& col,
  int64_t output_channels,
  int64_t output_height, int64_t output_width,
  int64_t input_height, int64_t input_width,
  int64_t kH, int64_t kW,
  int64_t pH, int64_t pW,
  int64_t sH, int64_t sW,
  int64_t dH, int64_t dW);

using channels_last_fn = void (*)(
    Tensor& im,
    const Tensor& col,
    int64_t nbatch,
    int64_t output_channels,
    int64_t output_height, int64_t output_width,
    int64_t input_height, int64_t input_width,
    int64_t kH, int64_t kW,
    int64_t pH, int64_t pW,
    int64_t sH, int64_t sW,
    int64_t dH, int64_t dW);

DECLARE_DISPATCH(contig_fn, col2im_stub);
DECLARE_DISPATCH(contig_fn, im2col_stub);
DECLARE_DISPATCH(channels_last_fn, col2im_channels_last_stub);
DECLARE_DISPATCH(channels_last_fn, im2col_channels_last_stub);

// skip im2col or col2im on certain conditions
static inline bool skip_transforming(
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding) {
  TORCH_CHECK(
      (kernel_size.size() == stride.size()) &&
      (kernel_size.size() == padding.size()) &&
      (kernel_size.size() == output_padding.size()));
  bool res = true;
  for (int64_t k = 0; k < kernel_size.size(); k++) {
    res = res && (kernel_size[k] == 1) && (stride[k] == 1) && (padding[k] == 0) && (output_padding[k] == 0);
  }
  return res;
}

template <typename T>
static void im2col(
    const T* data_im,
    const int64_t channels,
    const int64_t height,
    const int64_t width,
    const int64_t output_height,
    const int64_t output_width,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t pad_h,
    const int64_t pad_w,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t dilation_h,
    const int64_t dilation_w,
    T* data_col) {
  const int64_t height_col = output_height;
  const int64_t width_col = output_width;
  const int64_t channels_col = channels * kernel_h * kernel_w;

  for (int64_t c_col = 0; c_col < channels_col; ++c_col) {
    int64_t w_offset = c_col % kernel_w;
    int64_t h_offset = (c_col / kernel_w) % kernel_h;
    int64_t c_im = c_col / kernel_h / kernel_w;

    for (int64_t h_col = 0; h_col < height_col; ++h_col) {
      int64_t h_im = h_col * stride_h - pad_h + h_offset * dilation_h;

      for (int64_t w_col = 0; w_col < width_col; ++w_col) {
        int64_t w_im = w_col * stride_w - pad_w + w_offset * dilation_w;
        data_col[(c_col * height_col + h_col) * width_col + w_col] =
            (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width)
            ? data_im[(c_im * height + h_im) * width + w_im]
            : static_cast<T>(0);
      }
    }
  }
}

template <typename T>
static void col2im(
    const T* data_col,
    const int64_t channels,
    const int64_t height,
    const int64_t width,
    const int64_t output_height,
    const int64_t output_width,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t pad_h,
    const int64_t pad_w,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t dilation_h,
    const int64_t dilation_w,
    T* data_im) {
  std::fill_n(data_im, height * width * channels, T(0));

  const int64_t height_col = output_height;
  const int64_t width_col = output_width;
  const int64_t channels_col = channels * kernel_h * kernel_w;

  for (int64_t c_col = 0; c_col < channels_col; ++c_col) {
    int64_t w_offset = c_col % kernel_w;
    int64_t h_offset = (c_col / kernel_w) % kernel_h;
    int64_t c_im = c_col / kernel_h / kernel_w;

    for (int64_t h_col = 0; h_col < height_col; ++h_col) {
      int64_t h_im = h_col * stride_h - pad_h + h_offset * dilation_h;

      for (int64_t w_col = 0; w_col < width_col; ++w_col) {
        int64_t w_im = w_col * stride_w - pad_w + w_offset * dilation_w;

        if (h_im >= 0 && h_im < height && w_im >= 0 && w_im < width)
          data_im[(c_im * height + h_im) * width + w_im] +=
              data_col[(c_col * height_col + h_col) * width_col + w_col];
      }
    }
  }
}

} // native
} // at
