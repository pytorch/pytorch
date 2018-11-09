#pragma once

#ifdef _OPENMP
#include <omp.h>
#endif

#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"
#include "caffe2/utils/math_utils.h"

namespace caffe2 {

namespace math {

template <typename T>
static void Im2ColNCHW(
    const int channels,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int dilation_h,
    const int dilation_w,
    const int pad_t,
    const int pad_l,
    const int pad_b,
    const int pad_r,
    const int stride_h,
    const int stride_w,
    const T* data_im,
    T* data_col,
    CPUContext* /*context*/,
    const T& zero_point = 0) {
  const int output_h =
      (height + pad_b + pad_t - (dilation_h * (kernel_h - 1) + 1)) / stride_h +
      1;
  const int output_w =
      (width + pad_l + pad_r - (dilation_w * (kernel_w - 1) + 1)) / stride_w +
      1;

  // Fast path for zero padding and no dilation
  // From Torch, THNN_(unfolded_copy)
  if (dilation_h == 1 && dilation_w == 1 && pad_l == 0 && pad_r == 0 &&
      pad_t == 0 && pad_b == 0) {
    for (auto k = 0; k < channels * kernel_h * kernel_w; k++) {
      const auto nip = k / (kernel_h * kernel_w);
      const auto rest = k % (kernel_h * kernel_w);
      const auto kh = rest / kernel_w;
      const auto kw = rest % kernel_w;
      auto* dst = data_col + nip * (kernel_h * kernel_w * output_h * output_w) +
          kh * (kernel_w * output_h * output_w) + kw * (output_h * output_w);
      const auto* src = data_im + nip * (height * width);
      for (auto y = 0; y < output_h; y++) {
        const auto iy = y * stride_h + kh;
        const auto ix = kw;
        if (stride_w == 1) {
          memcpy(
              dst + (y * output_w),
              src + (iy * width + ix),
              sizeof(T) * output_w);
        } else {
          for (auto x = 0; x < output_w; x++) {
            memcpy(
                dst + (y * output_w + x),
                src + (iy * width + ix + x * stride_w),
                sizeof(T));
          }
        }
      }
    }
    return;
  }

  // Fast path for equal padding
  if (pad_l == pad_r && pad_t == pad_b) {
    // From Intel, https://github.com/BVLC/caffe/pull/3536
    const int pad_h = pad_t;
    const int pad_w = pad_l;
    const int channel_size = height * width;
    for (int channel = channels; channel--; data_im += channel_size) {
      for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
        for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
          int input_row = -pad_h + kernel_row * dilation_h;
          for (int output_rows = output_h; output_rows; output_rows--) {
            if (!utils::IsAGeZeroAndALtB(input_row, height)) {
              for (int output_cols = output_w; output_cols; output_cols--) {
                *(data_col++) = zero_point;
              }
            } else {
              int input_col = -pad_w + kernel_col * dilation_w;
              for (int output_col = output_w; output_col; output_col--) {
                if (utils::IsAGeZeroAndALtB(input_col, width)) {
                  *(data_col++) = data_im[input_row * width + input_col];
                } else {
                  *(data_col++) = zero_point;
                }
                input_col += stride_w;
              }
            }
            input_row += stride_h;
          }
        }
      }
    }
    return;
  }

  // Baseline
  const int dkernel_h = dilation_h * (kernel_h - 1) + 1;
  const int dkernel_w = dilation_w * (kernel_w - 1) + 1;

  int height_col = (height + pad_t + pad_b - dkernel_h) / stride_h + 1;
  int width_col = (width + pad_l + pad_r - dkernel_w) / stride_w + 1;

  int channels_col = channels * kernel_h * kernel_w;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % kernel_w;
    int h_offset = (c / kernel_w) % kernel_h;
    int c_im = c / kernel_h / kernel_w;
    for (int h = 0; h < height_col; ++h) {
      for (int w = 0; w < width_col; ++w) {
        int h_pad = h * stride_h - pad_t + h_offset * dilation_h;
        int w_pad = w * stride_w - pad_l + w_offset * dilation_w;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
          data_col[(c * height_col + h) * width_col + w] =
              data_im[(c_im * height + h_pad) * width + w_pad];
        else
          data_col[(c * height_col + h) * width_col + w] = zero_point;
      }
    }
  }
}

template <typename T>
static void Im2ColNdNCHW(
    const int N,
    const int /* img_size*/,
    const int col_size,
    const int* img_shape,
    const int* col_shape,
    const int* kernel_shape,
    const int* stride,
    const int* dilation,
    const int* pad,
    const T* X_data,
    T* Y_data,
    CPUContext* /* context */,
    const T& zero_point = 0) {
  const int outer_size = col_shape[0];
  const int inner_size = col_size / outer_size;
  const int kernel_size = std::accumulate(
      kernel_shape, kernel_shape + N, 1, std::multiplies<int>());
  std::vector<int> d_offset(N, 0);
  std::vector<int> d_iter(N, 0);
  for (int i = 0; i < outer_size; ++i) {
    // Loop over spatial axes in reverse order to compute a per-axis offset.
    int offset = i;
    for (int d_i = N - 1; d_i >= 0; --d_i) {
      d_offset[d_i] = offset % kernel_shape[d_i];
      offset /= kernel_shape[d_i];
    }
    for (int j = 0; j < inner_size; ++j) {
      // Loop over spatial axes in forward order to compute the indices in the
      // image and column, and whether the index lies in the padding.
      const int col_index = i * inner_size + j;
      int img_index = i / kernel_size;
      bool is_padding = false;
      for (int d_i = 0; d_i < N; ++d_i) {
        const int d_img = d_iter[d_i] * stride[d_i] - pad[d_i] +
            d_offset[d_i] * dilation[d_i];
        is_padding |= d_img < 0 || d_img >= img_shape[d_i + 1];
        img_index = img_index * img_shape[d_i + 1] + d_img;
      }
      Y_data[col_index] = is_padding ? zero_point : X_data[img_index];
      utils::IncreaseIndexInDims(N, col_shape + 1, d_iter.data());
    }
  }
}

/**
 * The layout of the result is N H W G R S C/G.
 * Note that groups are pulled out to an outer dimension so that we can use
 * GEMMs efficiently.
 */
template <typename T>
static void Im2ColNHWC(
    const int channels,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int dilation_h,
    const int dilation_w,
    const int pad_t,
    const int pad_l,
    const int pad_b,
    const int pad_r,
    const int stride_h,
    const int stride_w,
    const T* data_im,
    T* data_col,
    CPUContext* /*context*/,
    const int groups,
    const T& zero_point) {
  const int dkernel_h = dilation_h * (kernel_h - 1) + 1;
  const int dkernel_w = dilation_w * (kernel_w - 1) + 1;

  int height_col = (height + pad_t + pad_b - dkernel_h) / stride_h + 1;
  int width_col = (width + pad_l + pad_r - dkernel_w) / stride_w + 1;

#ifdef _OPENMP
#pragma omp parallel for if (!omp_in_parallel())
#endif
  for (int h = 0; h < height_col; ++h) {
    int h_pad = -pad_t + h * stride_h;
    T* data_col_temp =
        data_col + h * width_col * kernel_h * kernel_w * channels;
    int w_pad = -pad_l;
    for (int w = 0; w < width_col; ++w) {
      int r = 0;
      for (int ih = h_pad; ih < h_pad + dkernel_h; ih += dilation_h, ++r) {
        int s = 0;
        for (int iw = w_pad; iw < w_pad + dkernel_w; iw += dilation_w, ++s) {
          if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
            for (int g = 0; g < groups; ++g) {
              memcpy(
                  data_col_temp +
                      ((g * kernel_h + r) * kernel_w + s) * (channels / groups),
                  data_im + (ih * width + iw) * channels +
                      g * (channels / groups),
                  sizeof(T) * (channels / groups));
            }
          } else {
            // This should be simply padded with zero.
            for (int g = 0; g < groups; ++g) {
              for (int i = 0; i < channels / groups; ++i) {
                data_col_temp
                    [(((g * kernel_h + r) * kernel_w) + s) *
                         (channels / groups) +
                     i] = zero_point;
              }
            }
          }
        } // for each iw
      } // for each ih
      data_col_temp += kernel_h * kernel_w * channels;
      w_pad += stride_w;
    } // for each output pixel
  } // for each image row
}

/**
 * The layout of the result is N T H W G Q R S C/G.
 * Note that groups are pulled out to an outer dimension so that we can use
 * GEMMs efficiently.
 */
template <typename T>
static void Im2Col3DNHWC(
    const int channels,
    const int num_frames,
    const int height,
    const int width,
    const int kernel_t,
    const int kernel_h,
    const int kernel_w,
    const int dilation_t,
    const int dilation_h,
    const int dilation_w,
    const int pad_p, // previous frame
    const int pad_t, // top
    const int pad_l, // left
    const int pad_n, // next frame
    const int pad_b, // bottom
    const int pad_r, // right
    const int stride_t,
    const int stride_h,
    const int stride_w,
    const T* data_im,
    T* data_col,
    CPUContext* /*context*/,
    const int groups,
    const T& zero_point) {
  const int dkernel_t = dilation_t * (kernel_t - 1) + 1;
  const int dkernel_h = dilation_h * (kernel_h - 1) + 1;
  const int dkernel_w = dilation_w * (kernel_w - 1) + 1;

  int frame_col = (num_frames + pad_p + pad_n - dkernel_t) / stride_t + 1;
  int height_col = (height + pad_t + pad_b - dkernel_h) / stride_h + 1;
  int width_col = (width + pad_l + pad_r - dkernel_w) / stride_w + 1;

#ifdef _OPENMP
#pragma omp parallel for if (!omp_in_parallel())
#endif
  for (int t = 0; t < frame_col; ++t) {
    int t_pad = -pad_p + t * stride_t;
    for (int h = 0; h < height_col; ++h) {
      int h_pad = -pad_t + h * stride_h;
      T* data_col_temp = data_col +
          (t * height_col + h) * width_col * kernel_t * kernel_h * kernel_w *
              channels;
      for (int w = 0; w < width_col; ++w) {
        int w_pad = -pad_l + w * stride_w;
        int q = 0;
        for (int it = t_pad; it < t_pad + dkernel_t; it += dilation_t, ++q) {
          int r = 0;
          for (int ih = h_pad; ih < h_pad + dkernel_h; ih += dilation_h, ++r) {
            int s = 0;
            for (int iw = w_pad; iw < w_pad + dkernel_w;
                 iw += dilation_w, ++s) {
              if (it >= 0 && it < num_frames && ih >= 0 && ih < height &&
                  iw >= 0 && iw < width) {
                for (int g = 0; g < groups; ++g) {
                  memcpy(
                      data_col_temp +
                          (((g * kernel_t + q) * kernel_h + r) * kernel_w + s) *
                              (channels / groups),
                      data_im + ((it * height + ih) * width + iw) * channels +
                          g * (channels / groups),
                      sizeof(T) * (channels / groups));
                }
              } else {
                // This should be simply padded with zero.
                for (int g = 0; g < groups; ++g) {
                  for (int i = 0; i < channels / groups; ++i) {
                    data_col_temp
                        [((((g * kernel_t + q) * kernel_h + r) * kernel_w) +
                          s) *
                             (channels / groups) +
                         i] = zero_point;
                  }
                }
              }
            } // for each iw
          } // for each ih
        } // for each it
        data_col_temp += kernel_t * kernel_h * kernel_w * channels;
      } // for each output pixel
    } // for each image row
  } // for each frame
}

} // namespace math

} // namespace caffe2
