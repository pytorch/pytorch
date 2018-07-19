#ifndef THCUNN_VOL2COL_H
#define THCUNN_VOL2COL_H

#include "common.h"
#include "THCNumerics.cuh"

// Kernel for fast unfold+copy on volumes
template <typename Dtype>
__global__ void vol2col_kernel(const int64_t n, const Dtype* data_vol,
    const int64_t depth, const int64_t height, const int64_t width,
    const int64_t ksize_t, const int64_t ksize_h, const int64_t ksize_w,
    const int64_t pad_t, const int64_t pad_h, const int64_t pad_w,
    const int64_t stride_t, const int64_t stride_h, const int64_t stride_w,
    const int64_t dilation_t, const int64_t dilation_h, const int64_t dilation_w,
    const int64_t depth_col, const int64_t height_col, const int64_t width_col,
    Dtype* data_col) {
CUDA_KERNEL_LOOP(index, n) {
    int64_t w_out = index % width_col;
    index /= width_col;
    int64_t h_out = index % height_col;
    index /= height_col;
    int64_t t_out = index % depth_col;
    int64_t channel_in = index / depth_col;
    int64_t channel_out = channel_in * ksize_t * ksize_h * ksize_w;
    int64_t t_in = t_out * stride_t - pad_t;
    int64_t h_in = h_out * stride_h - pad_h;
    int64_t w_in = w_out * stride_w - pad_w;
    data_col += ((channel_out * depth_col + t_out) * height_col + h_out) * width_col + w_out;
    data_vol += ((channel_in * depth + t_in) * height + h_in) * width + w_in;
    for (int64_t i = 0; i < ksize_t; ++i) {
      for (int64_t j = 0; j < ksize_h; ++j) {
        for (int64_t k = 0; k < ksize_w; ++k) {
          int64_t t = t_in + i * dilation_t;
          int64_t h = h_in + j * dilation_h;
          int64_t w = w_in + k * dilation_w;
          *data_col = (t >= 0 && h >= 0 && w >= 0 && t < depth && h < height && w < width) ?
            data_vol[i * dilation_t * height * width + j * dilation_h * width + k * dilation_w] : ScalarConvert<int, Dtype>::to(0);
          data_col += depth_col * height_col * width_col;
        }
      }
    }
  }
}

template <typename Dtype>
void vol2col(cudaStream_t stream, const Dtype* data_vol, const int64_t channels,
    const int64_t depth, const int64_t height, const int64_t width,
    const int64_t depth_col, const int64_t height_col, const int64_t width_col,
    const int64_t ksize_t, const int64_t ksize_h, const int64_t ksize_w,
    const int64_t pad_t, const int64_t pad_h, const int64_t pad_w,
    const int64_t stride_t, const int64_t stride_h, const int64_t stride_w,
    const int64_t dilation_t, const int64_t dilation_h, const int64_t dilation_w,
    Dtype* data_col) {
  // We are going to launch channels * depth_col * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int64_t num_kernels = channels * depth_col * height_col * width_col;
  // Launch
  vol2col_kernel <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>> (
      num_kernels, data_vol, depth, height, width, ksize_t, ksize_h, ksize_w,
      pad_t, pad_h, pad_w, stride_t, stride_h, stride_w,
      dilation_t, dilation_h, dilation_w,
      depth_col, height_col, width_col, data_col
  );
  THCudaCheck(cudaGetLastError());
}

template <typename Dtype, typename Acctype>
__global__ void vol2im_kernel(const int64_t n, const Dtype* data_col,
    const int64_t depth, const int64_t height, const int64_t width, const int64_t channels,
    const int64_t kernel_t, const int64_t kernel_h, const int64_t kernel_w,
    const int64_t pad_t, const int64_t pad_h, const int64_t pad_w,
    const int64_t stride_t, const int64_t stride_h, const int64_t stride_w,
    const int64_t dilation_t, const int64_t dilation_h, const int64_t dilation_w,
    const int64_t depth_col, const int64_t height_col, const int64_t width_col,
    Dtype* data_vol) {
  CUDA_KERNEL_LOOP(index, n) {
    Acctype val = Acctype(0);
    const int64_t w_im = index % width + pad_w;
    const int64_t h_im = (index / width) % height + pad_h;
    const int64_t t_im = (index / width / height) % depth + pad_t;
    const int64_t c_im = index / (width * height * depth);
    int64_t kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
    int64_t kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
    int64_t kernel_extent_t = (kernel_t - 1) * dilation_t + 1;
    // compute the start and end of the output
    const int64_t w_col_start =
      (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
    const int64_t w_col_end = min(w_im / stride_w + 1, width_col);
    const int64_t h_col_start =
      (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
    const int64_t h_col_end = min(h_im / stride_h + 1, height_col);
    const int64_t t_col_start =
      (t_im < kernel_extent_t) ? 0 : (t_im - kernel_extent_t) / stride_t + 1;
    const int64_t t_col_end = min(t_im / stride_t + 1, depth_col);
    // TODO: use LCM of stride and dilation to avoid unnecessary loops
    for (int64_t t_col = t_col_start; t_col < t_col_end; t_col += 1) {
      for (int64_t h_col = h_col_start; h_col < h_col_end; h_col += 1) {
        for (int64_t w_col = w_col_start; w_col < w_col_end; w_col += 1) {
          int64_t t_k = (t_im - t_col * stride_t);
          int64_t h_k = (h_im - h_col * stride_h);
          int64_t w_k = (w_im - w_col * stride_w);
          if (t_k % dilation_t == 0 && h_k % dilation_h == 0 && w_k % dilation_w == 0) {
            t_k /= dilation_t;
            h_k /= dilation_h;
            w_k /= dilation_w;
            int64_t data_col_index =
              (((((c_im * kernel_t + t_k) * kernel_h + h_k) * kernel_w + w_k)
                * depth_col + t_col) * height_col + h_col) * width_col + w_col;
            val += data_col[data_col_index];
          }
        }
      }
    }
    data_vol[index] = ScalarConvert<Acctype, Dtype>::to(val);
  }
}

template <typename Dtype, typename Acctype>
void col2vol(cudaStream_t stream, const Dtype* data_col, const int64_t channels,
    const int64_t depth, const int64_t height, const int64_t width,
    const int64_t output_depth, const int64_t output_height, const int64_t output_width,
    const int64_t patch_t, const int64_t patch_h, const int64_t patch_w,
    const int64_t pad_t, const int64_t pad_h, const int64_t pad_w,
    const int64_t stride_t, const int64_t stride_h, const int64_t stride_w,
    const int64_t dilation_t, const int64_t dilation_h, const int64_t dilation_w,
    Dtype* data_vol) {
  int64_t num_kernels = channels * depth * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  vol2im_kernel<Dtype, Acctype> <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>> (
      num_kernels, data_col, depth, height, width, channels,
      patch_t, patch_h, patch_w, pad_t, pad_h, pad_w, stride_t, stride_h, stride_w,
      dilation_t, dilation_h, dilation_w,
      output_depth, output_height, output_width, data_vol
  );
  THCudaCheck(cudaGetLastError());
}

#endif
