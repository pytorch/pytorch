#ifndef THCUNN_VOL2COL_H
#define THCUNN_VOL2COL_H

#include "common.h"

// Kernel for fast unfold+copy on volumes
template <typename Dtype>
__global__ void vol2col_kernel(const int n, const Dtype* data_vol,
    const int depth, const int height, const int width,
    const int ksize_t, const int ksize_h, const int ksize_w,
    const int pad_t, const int pad_h, const int pad_w,
    const int stride_t, const int stride_h, const int stride_w,
    const int depth_col, const int height_col, const int width_col,
    Dtype* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    int w_out = index % width_col;
    index /= width_col;
    int h_out = index % height_col;
    index /= height_col;
    int t_out = index % depth_col;
    int channel_in = index / depth_col;
    int channel_out = channel_in * ksize_t * ksize_h * ksize_w;
    int t_in = t_out * stride_t - pad_t;
    int h_in = h_out * stride_h - pad_h;
    int w_in = w_out * stride_w - pad_w;
    data_col += ((channel_out * depth_col + t_out) * height_col + h_out) * width_col + w_out;
    data_vol += ((channel_in * depth + t_in) * height + h_in) * width + w_in;
    for (int i = 0; i < ksize_t; ++i) {
      for (int j = 0; j < ksize_h; ++j) {
        for (int k = 0; k < ksize_w; ++k) {
          int t = t_in + i;
          int h = h_in + j;
          int w = w_in + k;
          *data_col = (t >= 0 && h >= 0 && w >= 0 && t < depth && h < height && w < width) ?
            data_vol[(i * height + j) * width + k] : 0;
          data_col += depth_col * height_col * width_col;
        }
      }
    }
  }
}

template <typename Dtype>
void vol2col(cudaStream_t stream, const Dtype* data_vol, const int channels,
    const int depth, const int height, const int width,
    const int ksize_t, const int ksize_h, const int ksize_w,
    const int pad_t, const int pad_h, const int pad_w,
    const int stride_t, const int stride_h, const int stride_w, Dtype* data_col) {
  // We are going to launch channels * depth_col * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int depth_col = (depth + 2 * pad_t - ksize_t) / stride_t + 1;
  int height_col = (height + 2 * pad_h - ksize_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - ksize_w) / stride_w + 1;
  int num_kernels = channels * depth_col * height_col * width_col;
  // Launch
  vol2col_kernel <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>> (
      num_kernels, data_vol, depth, height, width, ksize_t, ksize_h, ksize_w,
      pad_t, pad_h, pad_w, stride_t, stride_h, stride_w,
      depth_col, height_col, width_col, data_col
  );
}

template <typename Dtype>
__global__ void vol2im_kernel(const int n, const Dtype* data_col,
    const int depth, const int height, const int width, const int channels,
    const int patch_t, const int patch_h, const int patch_w,
    const int pad_t, const int pad_h, const int pad_w,
    const int stride_t, const int stride_h, const int stride_w,
    const int depth_col, const int height_col, const int width_col,
    Dtype* data_vol) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = 0;
    int w = index % width + pad_w;
    int h = (index / width) % height + pad_h;
    int t = (index / width / height) % depth + pad_t;
    int c = index / (width * height * depth);
    // compute the start and end of the output
    int w_col_start = (w < patch_w) ? 0 : (w - patch_w) / stride_w + 1;
    int w_col_end = min(w / stride_w + 1, width_col);
    int h_col_start = (h < patch_h) ? 0 : (h - patch_h) / stride_h + 1;
    int h_col_end = min(h / stride_h + 1, height_col);
    int t_col_start = (t < patch_t) ? 0 : (t - patch_t) / stride_t + 1;
    int t_col_end = min(t / stride_t + 1, depth_col);

    int offset = (((c * patch_t + t) * patch_h + h) * patch_w + w) * depth_col * height_col * width_col;
    int coeff_t_col = (1 - stride_t * patch_h * patch_w * depth_col) * height_col * width_col;
    int coeff_h_col = (1 - stride_h * patch_w * depth_col * height_col) * width_col;
    int coeff_w_col = (1 - stride_w * depth_col * height_col * width_col) ;

    for (int t_col = t_col_start; t_col < t_col_end; ++t_col) {
      for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
        for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
          val += data_col[offset + t_col * coeff_t_col + h_col * coeff_h_col + w_col * coeff_w_col];
        }
      }
    }
    data_vol[index] = val;
  }
}

template <typename Dtype>
void col2vol(cudaStream_t stream, const Dtype* data_col, const int channels,
    const int depth, const int height, const int width,
    const int patch_t, const int patch_h, const int patch_w,
    const int pad_t, const int pad_h, const int pad_w,
    const int stride_t, const int stride_h, const int stride_w, Dtype* data_vol) {
  int depth_col = (depth + 2 * pad_t - patch_t) / stride_t + 1;
  int height_col = (height + 2 * pad_h - patch_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - patch_w) / stride_w + 1;
  int num_kernels = channels * depth * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  vol2im_kernel <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>> (
      num_kernels, data_col, depth, height, width, channels,
      patch_t, patch_h, patch_w, pad_t, pad_h, pad_w, stride_t, stride_h, stride_w,
      depth_col, height_col, width_col, data_vol
  );
}

#endif
