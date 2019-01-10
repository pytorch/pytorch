#ifndef THCUNN_IM2COL_H
#define THCUNN_IM2COL_H

#include "common.h"
#include "THCNumerics.cuh"

// Kernel for fast unfold+copy
// (borrowed from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu)
template <typename Dtype>
__global__ void im2col_kernel(const int n, const Dtype* data_im,
                              const int height, const int width,
                              const int ksize_h, const int ksize_w,
                              const int pad_h, const int pad_w,
                              const int stride_h, const int stride_w,
                              const int dilation_h, const int dilation_w,
                              const int height_col, const int width_col,
    Dtype* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    int w_out = index % width_col;
    index /= width_col;
    int h_out = index % height_col;
    int channel_in = index / height_col;
    int channel_out = channel_in * ksize_h * ksize_w;
    int h_in = h_out * stride_h - pad_h;
    int w_in = w_out * stride_w - pad_w;
    data_col += (channel_out * height_col + h_out) * width_col + w_out;
    data_im += (channel_in * height + h_in) * width + w_in;
    for (int i = 0; i < ksize_h; ++i) {
      for (int j = 0; j < ksize_w; ++j) {
        int h = h_in + i * dilation_h;
        int w = w_in + j * dilation_w;
        *data_col = (h >= 0 && w >= 0 && h < height && w < width) ?
          data_im[i * dilation_h * width + j * dilation_w] : ScalarConvert<int, Dtype>::to(0);
        data_col += height_col * width_col;
      }
    }
  }
}

template <typename Dtype>
void im2col(cudaStream_t stream, const Dtype* data_im, const int channels,
            const int height, const int width,
            const int height_col, const int width_col,
            const int ksize_h, const int ksize_w, const int pad_h,
            const int pad_w, const int stride_h, const int stride_w,
            const int dilation_h, const int dilation_w, Dtype* data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int num_kernels = channels * height_col * width_col;
  // Launch
  im2col_kernel <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>> (
      num_kernels, data_im, height, width, ksize_h, ksize_w,
      pad_h, pad_w, stride_h, stride_w,
      dilation_h, dilation_w,
      height_col, width_col, data_col
  );
  THCudaCheck(cudaGetLastError());
}

template <typename Dtype, typename Acctype>
__global__ void col2im_kernel(const int n, const Dtype* data_col,
                                  const int height, const int width, const int channels,
                                  const int kernel_h, const int kernel_w,
                                  const int pad_h, const int pad_w,
                                  const int stride_h, const int stride_w,
                                  const int dilation_h, const int dilation_w,
                                  const int height_col, const int width_col,
                                  Dtype* data_im) {
  CUDA_KERNEL_LOOP(index, n) {
    Acctype val = Acctype(0);
    const int w_im = index % width + pad_w;
    const int h_im = (index / width) % height + pad_h;
    const int c_im = index / (width * height);
    int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
    int kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
    // compute the start and end of the output
    const int w_col_start =
      (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
    const int w_col_end = min(w_im / stride_w + 1, width_col);
    const int h_col_start =
      (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
    const int h_col_end = min(h_im / stride_h + 1, height_col);
    // TODO: use LCM of stride and dilation to avoid unnecessary loops
    for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
      for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
        int h_k = (h_im - h_col * stride_h);
        int w_k = (w_im - w_col * stride_w);
        if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
          h_k /= dilation_h;
          w_k /= dilation_w;
          int data_col_index = (((c_im * kernel_h + h_k) * kernel_w + w_k) *
                                height_col + h_col) * width_col + w_col;
          val += data_col[data_col_index];
        }
      }
    }
    data_im[index] = ScalarConvert<Acctype, Dtype>::to(val);
  }
}

template <typename Dtype, typename Acctype>
void col2im(cudaStream_t stream, const Dtype* data_col, const int channels,
            const int height, const int width,
            const int output_height, const int output_width,
            const int patch_h, const int patch_w, const int pad_h,
            const int pad_w, const int stride_h, const int stride_w,
            const int dilation_h, const int dilation_w, Dtype* data_im);

template <typename Dtype, typename Acctype>
void col2im(cudaStream_t stream, const Dtype* data_col, const int channels,
            const int height, const int width,
            const int output_height, const int output_width,
            const int patch_h, const int patch_w, const int pad_h,
            const int pad_w, const int stride_h, const int stride_w,
            const int dilation_h, const int dilation_w, Dtype* data_im) {
  int num_kernels = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  col2im_kernel<Dtype, Acctype> <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>> (
      num_kernels, data_col, height, width, channels,
      patch_h, patch_w, pad_h, pad_w, stride_h, stride_w,
      dilation_h, dilation_w,
      output_height, output_width, data_im
  );
  THCudaCheck(cudaGetLastError());
}

#endif
