#ifndef THCUNN_VOL2COL_H
#define THCUNN_VOL2COL_H

#include "common.h"
#include "THCNumerics.cuh"

// Kernel for fast unfold+copy on volumes
template <typename Dtype>
__global__ void vol2col_kernel(const int n, const Dtype* data_vol,
    const int depth, const int height, const int width,
    const int ksize_t, const int ksize_h, const int ksize_w,
    const int pad_t, const int pad_h, const int pad_w,
    const int stride_t, const int stride_h, const int stride_w,
    const int dilation_t, const int dilation_h, const int dilation_w,
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
          int t = t_in + i * dilation_t;
          int h = h_in + j * dilation_h;
          int w = w_in + k * dilation_w;
          *data_col = (t >= 0 && h >= 0 && w >= 0 && t < depth && h < height && w < width) ?
            data_vol[i * dilation_t * height * width + j * dilation_h * width + k * dilation_w] : ScalarConvert<int, Dtype>::to(0);
          data_col += depth_col * height_col * width_col;
        }
      }
    }
  }
}

template <typename Dtype>
void vol2col(cudaStream_t stream, const Dtype* data_vol, const int channels,
    const int depth, const int height, const int width,
    const int depth_col, const int height_col, const int width_col,
    const int ksize_t, const int ksize_h, const int ksize_w,
    const int pad_t, const int pad_h, const int pad_w,
    const int stride_t, const int stride_h, const int stride_w,
    const int dilation_t, const int dilation_h, const int dilation_w,
    Dtype* data_col) {
  // We are going to launch channels * depth_col * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int num_kernels = channels * depth_col * height_col * width_col;
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
__global__ void vol2im_kernel(const int n, const Dtype* data_col,
    const int depth, const int height, const int width, const int channels,
    const int kernel_t, const int kernel_h, const int kernel_w,
    const int pad_t, const int pad_h, const int pad_w,
    const int stride_t, const int stride_h, const int stride_w,
    const int dilation_t, const int dilation_h, const int dilation_w,
    const int depth_col, const int height_col, const int width_col,
    Dtype* data_vol) {
  CUDA_KERNEL_LOOP(index, n) {
    Acctype val = Acctype(0);
    const int w_im = index % width + pad_w;
    const int h_im = (index / width) % height + pad_h;
    const int t_im = (index / width / height) % depth + pad_t;
    const int c_im = index / (width * height * depth);
    int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
    int kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
    int kernel_extent_t = (kernel_t - 1) * dilation_t + 1;
    // compute the start and end of the output
    const int w_col_start =
      (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
    const int w_col_end = min(w_im / stride_w + 1, width_col);
    const int h_col_start =
      (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
    const int h_col_end = min(h_im / stride_h + 1, height_col);
    const int t_col_start =
      (t_im < kernel_extent_t) ? 0 : (t_im - kernel_extent_t) / stride_t + 1;
    const int t_col_end = min(t_im / stride_t + 1, depth_col);
    // TODO: use LCM of stride and dilation to avoid unnecessary loops
    for (int t_col = t_col_start; t_col < t_col_end; t_col += 1) {
      for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
        for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
          int t_k = (t_im - t_col * stride_t);
          int h_k = (h_im - h_col * stride_h);
          int w_k = (w_im - w_col * stride_w);
          if (t_k % dilation_t == 0 && h_k % dilation_h == 0 && w_k % dilation_w == 0) {
            t_k /= dilation_t;
            h_k /= dilation_h;
            w_k /= dilation_w;
            int data_col_index =
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
void col2vol(cudaStream_t stream, const Dtype* data_col, const int channels,
    const int depth, const int height, const int width,
    const int output_depth, const int output_height, const int output_width,
    const int patch_t, const int patch_h, const int patch_w,
    const int pad_t, const int pad_h, const int pad_w,
    const int stride_t, const int stride_h, const int stride_w,
    const int dilation_t, const int dilation_h, const int dilation_w,
    Dtype* data_vol) {
  int num_kernels = channels * depth * height * width;
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
