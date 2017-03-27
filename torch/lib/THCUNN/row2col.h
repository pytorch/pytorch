#ifndef THCUNN_ROW2COL_H
#define THCUNN_ROW2COL_H

#include "THCNumerics.cuh"
#include "common.h"

// Kernel for fast unfold+copy on rows
template <typename Dtype>
__global__ void
row2col_kernel(const int n, const Dtype *data_row, const int width,
               const int ksize_w, const int pad_w, const int stride_w,
               const int dilation_w, const int width_col, Dtype *data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    int w_out = index % width_col;
    index /= width_col;
    int channel_in = index;
    int channel_out = channel_in * ksize_w;
    int w_in = w_out * stride_w - pad_w;
    data_col += (channel_out)*width_col + w_out;
    data_row += (channel_in)*width + w_in;
    for (int j = 0; j < ksize_w; ++j) {
      int w = w_in + j * dilation_w;
      *data_col = (w >= 0 && w < width) ? data_row[j * dilation_w]
                                        : ScalarConvert<int, Dtype>::to(0);
      data_col += width_col;
    }
  }
}

template <typename Dtype>
void row2col(cudaStream_t stream, const Dtype *data_row, const int channels,
             const int width, const int ksize_w, const int pad_w,
             const int stride_w, const int dilation_w, Dtype *data_col) {
  // We are going to launch channels * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int width_col =
      (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * width_col;
  // Launch
  row2col_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
      num_kernels, data_row, width, ksize_w, pad_w, stride_w, 1, width_col,
      data_col);
  THCudaCheck(cudaGetLastError());
}

template <typename Dtype, typename Acctype>
__global__ void col2row_kernel(const int n, const Dtype *data_col,
                               const int width, const int channels,
                               const int kernel_w, const int pad_w,
                               const int stride_w, const int dilation_w,
                               const int width_col, Dtype *data_row) {
  CUDA_KERNEL_LOOP(index, n) {
    Acctype val = Acctype(0);
    const int w_row = index % width + pad_w;
    const int c_row = index / width;
    int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
    // compute the start and end of the output
    const int w_col_start = (w_row < kernel_extent_w)
                                ? 0
                                : (w_row - kernel_extent_w) / stride_w + 1;
    const int w_col_end = min(w_row / stride_w + 1, width_col);
    for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
      int w_k = (w_row - w_col * stride_w);
      if (w_k % dilation_w == 0) {
        w_k /= dilation_w;
        int data_col_index = (c_row * kernel_w + w_k) * width_col + w_col;
        val += data_col[data_col_index];
      }
    }
    data_row[index] = ScalarConvert<Acctype, Dtype>::to(val);
  }
  }

template <typename Dtype, typename Acctype>
void col2row(cudaStream_t stream, const Dtype *data_col, const int channels,
             const int width, const int patch_w, const int pad_w,
             const int stride_w, const int dilation_w, Dtype *data_row) {
  int width_col =
      (width + 2 * pad_w - (dilation_w * (patch_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  col2row_kernel<
      Dtype, Acctype><<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
      num_kernels, data_col, width, channels, patch_w, pad_w, stride_w,
      dilation_w, width_col, data_row);

  THCudaCheck(cudaGetLastError());
}
#endif
