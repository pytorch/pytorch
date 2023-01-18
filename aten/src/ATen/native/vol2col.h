#pragma once

#include <cstring>

namespace at {
namespace native {

template <typename T>
static void vol2col(
    const T* data_vol,
    const int64_t channels,
    const int64_t depth,
    const int64_t height,
    const int64_t width,
    const int64_t depth_col,
    const int64_t height_col,
    const int64_t width_col,
    const int64_t kT,
    const int64_t kernel_height,
    const int64_t kernel_width,
    const int64_t pT,
    const int64_t pH,
    const int64_t pW,
    const int64_t dT,
    const int64_t dH,
    const int64_t dW,
    const int64_t dilationT,
    const int64_t dilationH,
    const int64_t dilationW,
    T* data_col) {
  int64_t c, t, h, w;
  int64_t channels_col = channels * kT * kernel_height * kernel_width;
  for (c = 0; c < channels_col; ++c) {
    int64_t w_offset = c % kernel_width;
    int64_t h_offset = (c / kernel_width) % kernel_height;
    int64_t t_offset = (c / kernel_width / kernel_height) % kT;
    int64_t c_vol = c / kT / kernel_height / kernel_width;
    for (t = 0; t < depth_col; ++t) {
      int64_t t_pad = t * dT - pT + t_offset * dilationT;
      for (h = 0; h < height_col; ++h) {
        int64_t h_pad = h * dH - pH + h_offset * dilationH;
        for (w = 0; w < width_col; ++w) {
          int64_t w_pad = w * dW - pW + w_offset * dilationW;
          if (t_pad >= 0 && t_pad < depth && h_pad >= 0 && h_pad < height &&
              w_pad >= 0 && w_pad < width)
            data_col[((c * depth_col + t) * height_col + h) * width_col + w] =
                data_vol
                    [((c_vol * depth + t_pad) * height + h_pad) * width +
                     w_pad];
          else
            data_col[((c * depth_col + t) * height_col + h) * width_col + w] =
                0;
        }
      }
    }
  }
}

template <typename T>
static void col2vol(
    const T* data_col,
    const int64_t channels,
    const int64_t depth,
    const int64_t height,
    const int64_t width,
    const int64_t out_depth,
    const int64_t out_height,
    const int64_t out_width,
    const int64_t kT,
    const int64_t kernel_height,
    const int64_t kernel_width,
    const int64_t pT,
    const int64_t pH,
    const int64_t pW,
    const int64_t dT,
    const int64_t dH,
    const int64_t dW,
    const int64_t dilationT,
    const int64_t dilationH,
    const int64_t dilationW,
    T* data_vol) {
  int64_t c, t, h, w;
  memset(data_vol, 0, sizeof(T) * depth * height * width * channels);
  int64_t depth_col = out_depth;
  int64_t height_col = out_height;
  int64_t width_col = out_width;
  int64_t channels_col = channels * kT * kernel_height * kernel_width;
  for (c = 0; c < channels_col; ++c) {
    int64_t w_offset = c % kernel_width;
    int64_t h_offset = (c / kernel_width) % kernel_height;
    int64_t t_offset = (c / kernel_width / kernel_height) % kT;
    int64_t c_vol = c / kT / kernel_height / kernel_width;
    for (t = 0; t < depth_col; ++t) {
      int64_t t_pad = t * dT - pT + t_offset * dilationT;
      for (h = 0; h < height_col; ++h) {
        int64_t h_pad = h * dH - pH + h_offset * dilationH;
        for (w = 0; w < width_col; ++w) {
          int64_t w_pad = w * dW - pW + w_offset * dilationW;
          if (t_pad >= 0 && t_pad < depth && h_pad >= 0 && h_pad < height &&
              w_pad >= 0 && w_pad < width)
            data_vol
                [((c_vol * depth + t_pad) * height + h_pad) * width + w_pad] +=
                data_col
                    [((c * depth_col + t) * height_col + h) * width_col + w];
        }
      }
    }
  }
}

} // namespace native
} // namespace at
