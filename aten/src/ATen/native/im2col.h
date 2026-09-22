#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/TensorUtils.h>
#include <ATen/TensorIterator.h>
#include <ATen/Utils.h>
#include <ATen/Parallel.h>
#include <ATen/native/cpu/utils.h>
#include <c10/util/Half.h>
#include <c10/util/irange.h>

#include <type_traits>

#include <algorithm>

namespace at::native {

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
    T* data_col,
    bool is_channels_last = false) {
  const int64_t height_col = output_height;
  const int64_t width_col = output_width;
  const int64_t channels_col = channels * kernel_h * kernel_w;

  if (is_channels_last) {
    at::parallel_for(0, height_col * width_col, 0, [&](int64_t begin, int64_t end) {
      int64_t h_col{0}, w_col{0};
      data_index_init(begin, h_col, height_col, w_col, width_col);

      for (const auto i_col : c10::irange(begin, end)) {
        for (const auto h_offset : c10::irange(kernel_h)) {
          int64_t h_im = h_col * stride_h - pad_h + h_offset * dilation_h;
          for (const auto w_offset : c10::irange(kernel_w)) {
            int64_t w_im = w_col * stride_w - pad_w + w_offset * dilation_w;

            const T* slice_im = data_im + (h_im * width + w_im) * channels;
            T* slice_col = data_col + (i_col * kernel_h * kernel_w + h_offset * kernel_w + w_offset) * channels;

            if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) {
              std::copy_n(slice_im, channels, slice_col);
            } else {
              std::fill_n(slice_col, channels, T(0));
            }
          }
        }

        // move the next index
        data_index_step(h_col, height_col, w_col, width_col);
      }
    });
  } else {
    at::parallel_for(0, channels_col, 0, [&](int64_t begin, int64_t end) {
      int64_t c_im{0}, h_offset{0}, w_offset{0};
      data_index_init(begin, c_im, channels, h_offset, kernel_h, w_offset, kernel_w);

      for (const auto c_col : c10::irange(begin, end)) {
        for (const auto h_col : c10::irange(height_col)) {
          int64_t h_im = h_col * stride_h - pad_h + h_offset * dilation_h;
          for (const auto w_col : c10::irange(width_col)) {
            int64_t w_im = w_col * stride_w - pad_w + w_offset * dilation_w;
            data_col[(c_col * height_col + h_col) * width_col + w_col] =
                (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width)
                ? c10::load(&(data_im[(c_im * height + h_im) * width + w_im]))
                : static_cast<T>(0);
          }
        }

        // move to the next index
        data_index_step(c_im, channels, h_offset, kernel_h, w_offset, kernel_w);
      }
    });
  }
}

template <typename T>
static void col2im_slice(
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
  if (!std::is_same_v<T, c10::Half> && kernel_h == 2 && kernel_w == 2 &&
      stride_h == 2 && stride_w == 2 && dilation_h == 1 && dilation_w == 1 &&
      pad_h == 0 && pad_w == 0 && height % 2 == 0 && width % 2 == 0 &&
      output_height == height / 2 && output_width == width / 2) {
    const int64_t col_size = output_height * output_width;
    for (int64_t c = 0; c < channels; ++c) {
      for (int64_t h = 0; h < output_height; ++h) {
        const T* src = data_col + c * 4 * col_size + h * output_width;
        T* dst = data_im + c * height * width + h * 2 * width;
        for (int64_t w = 0; w < output_width; ++w) {
          // Keep the addition to zero for signed-zero behavior.
          dst[2 * w] = T(0) + src[w];
          dst[2 * w + 1] = T(0) + src[col_size + w];
          dst[width + 2 * w] = T(0) + src[2 * col_size + w];
          dst[width + 2 * w + 1] = T(0) + src[3 * col_size + w];
        }
      }
    }
    return;
  }

  if (dilation_h == 1 && dilation_w == 1 &&
      stride_h == kernel_h && stride_w == kernel_w && pad_h == 0 && pad_w == 0 &&
      height % kernel_h == 0 && width % kernel_w == 0 &&
      output_height == height / kernel_h && output_width == width / kernel_w) {
    if (!std::is_same_v<T, c10::Half> && kernel_h == 1 && kernel_w == 1) {
      for (int64_t i = 0; i < channels * height * width; ++i) {
        data_im[i] = T(0) + data_col[i];
      }
      return;
    }
    for (int64_t c_col = 0; c_col < channels * kernel_h * kernel_w; ++c_col) {
      const int64_t w_offset = c_col % kernel_w;
      const int64_t h_offset = (c_col / kernel_w) % kernel_h;
      const int64_t c_im = c_col / kernel_h / kernel_w;
      for (int64_t h_col = 0; h_col < output_height; ++h_col) {
        T* dst = data_im +
            (c_im * height + h_col * kernel_h + h_offset) * width + w_offset;
        const T* src = data_col + (c_col * output_height + h_col) * output_width;
        for (int64_t w_col = 0; w_col < output_width; ++w_col) {
          dst[w_col * kernel_w] = T(0) + src[w_col];
        }
      }
    }
    return;
  }

  std::fill_n(data_im, height * width * channels, T(0));

  const int64_t height_col = output_height;
  const int64_t width_col = output_width;
  const int64_t channels_col = channels * kernel_h * kernel_w;

  for (int64_t c_col = 0; c_col < channels_col; ++c_col) {
    int64_t w_offset = c_col % kernel_w;
    int64_t h_offset = (c_col / kernel_w) % kernel_h;
    int64_t c_im = c_col / kernel_h / kernel_w;

    const int64_t first_w = w_offset * dilation_w - pad_w;
    const int64_t first_col = first_w < 0
        ? std::min(width_col, -(first_w + 1) / stride_w + 1)
        : 0;
    if (first_col == width_col) {
      continue;
    }
    const int64_t start_w = first_w < 0
        ? stride_w - 1 - (-(first_w + 1) % stride_w)
        : first_w;
    if (start_w >= width) {
      continue;
    }
    const int64_t count = std::min(
        width_col - first_col, (width - 1 - start_w) / stride_w + 1);
    for (int64_t h_col = 0; h_col < height_col; ++h_col) {
      int64_t h_im = h_col * stride_h - pad_h + h_offset * dilation_h;
      if (h_im >= 0 && h_im < height) {
        T* dst = data_im + (c_im * height + h_im) * width + start_w;
        const T* src = data_col +
            (c_col * height_col + h_col) * width_col + first_col;
        for (int64_t w = 0; w < count; ++w) {
          dst[w * stride_w] += src[w];
        }
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
    T* data_im,
    bool is_channels_last = false) {
  if (is_channels_last) {
    std::fill_n(data_im, height * width * channels, T(0));
    const int64_t height_col = output_height;
    const int64_t width_col = output_width;
    for (const auto h_col : c10::irange(height_col)) {
      for (const auto w_col : c10::irange(width_col)) {
        for (const auto h_offset : c10::irange(kernel_h)) {
          int64_t h_im = h_col * stride_h - pad_h + h_offset * dilation_h;
          for (const auto w_offset : c10::irange(kernel_w)) {
            int64_t w_im = w_col * stride_w - pad_w + w_offset * dilation_w;

            T* slice_im = data_im + (h_im * width + w_im) * channels;
            const T* slice_col = data_col + ((h_col * width_col + w_col) * kernel_h * kernel_w
                + h_offset * kernel_w + w_offset) * channels;

            if (h_im >= 0 && h_im < height && w_im >= 0 && w_im < width) {
              std::transform(slice_col, slice_col + channels, slice_im, slice_im, std::plus<T>());
            }
          }
        }
      }
    }
    return;
  }

  const int64_t image_size = height * width;
  const int64_t col_size = kernel_h * kernel_w * output_height * output_width;
  const auto run = [&](int64_t begin, int64_t end) {
    col2im_slice(
        data_col + begin * col_size, end - begin, height, width,
        output_height, output_width, kernel_h, kernel_w, pad_h, pad_w,
        stride_h, stride_w, dilation_h, dilation_w,
        data_im + begin * image_size);
  };
  const int64_t work_size = std::max<int64_t>(1, image_size + col_size);
  at::parallel_for(
      0, channels, std::max<int64_t>(1, at::internal::GRAIN_SIZE / work_size), run);
}

} // namespace at::native
