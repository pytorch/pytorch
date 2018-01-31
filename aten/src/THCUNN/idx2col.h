#ifndef THCUNN_IDX2COL_H
#define THCUNN_IDX2COL_H

#include "common.h"
#include "THCNumerics.cuh"
#include "THCAtomics.cuh"

// Kernel for fast unfold+copy
// (borrowed from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu)
template <typename Dtype>
__global__ void idx2col_kernel(const int n, const Dtype* data_im,
                               const int width_im,
                               const int kernel_size,
                               const int width_col,
                               const int64_t* data_idx,
                               Dtype* data_col) {
  CUDA_KERNEL_LOOP(index, n) {

    int w_out = index % width_col;
    int channel_in = index / width_col;
    int channel_out = channel_in * kernel_size;

    data_col += channel_out * width_col + w_out;
    data_im += channel_in * width_im;

    for (int i = 0; i < kernel_size; ++i) {

      int w = data_idx[w_out * kernel_size + i];

      *data_col = (w >= 0 && w < width_im) ?
        data_im[w] : ScalarConvert<int, Dtype>::to(0);

      data_col += width_col;
    }
  }
}

template <typename Dtype>
void idx2col(cudaStream_t stream, const Dtype* data_im, const int channels,
             const int width,
             const int ksize,
             const int64_t* data_idx,
             Dtype* data_col) {
  // We are going to launch channels * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int width_col = width;
  int num_kernels = channels * width_col;
  // Launch
  idx2col_kernel <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>> (
      num_kernels, data_im, width, ksize, width_col,
      data_idx, data_col
  );
  THCudaCheck(cudaGetLastError());
}

template <typename Dtype, typename Acctype>
__global__ void col2idx_kernel(const int n, const Dtype* data_col,
                               const int channels, const int width_col,
                               const int kernel_size,
                               const int width_im,
                               const int64_t* data_idx,
                               Dtype* data_im) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val;

    const int c_col = index / width_col;
    const int w_col = index % width_col;

    for (int k = 0; k < kernel_size; ++k) {
      int w_im = data_idx[w_col * kernel_size + k];

      if (w_im >= 0 && w_im < width_im) {
        val = data_col[(c_col * kernel_size + k) * width_col + w_col];
        atomicAdd(data_im + c_col * width_col + w_im, val);
      }
    }
  }
}

template <typename Dtype, typename Acctype>
void col2idx(cudaStream_t stream, const Dtype* data_col, const int channels,
             const int width,
             const int kernel_size,
             const int64_t* data_idx,
             Dtype* data_im) {
  int num_kernels = channels * width;
  int width_im = width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  col2idx_kernel<Dtype, Acctype> <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>> (
      num_kernels, data_col, channels, width, kernel_size,
      width_im, data_idx, data_im
  );
  THCudaCheck(cudaGetLastError());
}

#endif
