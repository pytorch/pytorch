#pragma once

#include <THC/THCGeneral.h>
#include <THC/THCDeviceUtils.cuh>

#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>

#include <c10/macros/Macros.h>

namespace at {
namespace native {

using namespace at::cuda::detail;

// Kernel for fast unfold+copy
// (borrowed from Caffe:
// https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu)
// CUDA_NUM_THREADS = 1024

template <typename dt>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void im2col_kernel(
    const int64_t n,
    const dt* data_im,
    const int64_t height,
    const int64_t width,
    const int64_t kernel_height,
    const int64_t kernel_width,
    const int64_t pad_height,
    const int64_t pad_width,
    const int64_t stride_height,
    const int64_t stride_width,
    const int64_t dilation_height,
    const int64_t dilation_width,
    const int64_t height_col,
    const int64_t width_col,
    dt* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    int64_t w_out = index % width_col;

    int64_t idx = index / width_col;

    int64_t h_out = idx % height_col;
    int64_t channel_in = idx / height_col;
    int64_t channel_out = channel_in * kernel_height * kernel_width;
    int64_t h_in = h_out * stride_height - pad_height;
    int64_t w_in = w_out * stride_width - pad_width;

    dt* col = data_col + (channel_out * height_col + h_out) * width_col + w_out;
    const dt* im = data_im + (channel_in * height + h_in) * width + w_in;

    for (int64_t i = 0; i < kernel_height; ++i) {
      for (int64_t j = 0; j < kernel_width; ++j) {
        int64_t h = h_in + i * dilation_height;
        int64_t w = w_in + j * dilation_width;
        *col = (h >= 0 && w >= 0 && h < height && w < width)
            ? im[i * dilation_height * width + j * dilation_width]
            : ScalarConvert<int, dt>::to(0);
        col += height_col * width_col;
      }
    }
  }
}

template <typename dt>
void im2col(
    cudaStream_t stream,
    const dt* data_im,
    const int64_t channels,
    const int64_t height,
    const int64_t width,
    const int64_t height_col,
    const int64_t width_col,
    const int64_t kernel_height,
    const int64_t kernel_width,
    const int64_t pad_height,
    const int64_t pad_width,
    const int64_t stride_height,
    const int64_t stride_width,
    const int64_t dilation_height,
    const int64_t dilation_width,
    dt* data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int64_t num_kernels = channels * height_col * width_col;
  // Launch CUDA_NUM_THREADS = 1024
  im2col_kernel<<<GET_BLOCKS(num_kernels), 1024, 0, stream>>>(
      num_kernels,
      data_im,
      height,
      width,
      kernel_height,
      kernel_width,
      pad_height,
      pad_width,
      stride_height,
      stride_width,
      dilation_height,
      dilation_width,
      height_col,
      width_col,
      data_col);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename dt, typename accT>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void col2im_kernel(
    const int64_t n,
    const dt* data_col,
    const int64_t height,
    const int64_t width,
    const int64_t channels,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t pad_height,
    const int64_t pad_width,
    const int64_t stride_height,
    const int64_t stride_width,
    const int64_t dilation_height,
    const int64_t dilation_width,
    const int64_t height_col,
    const int64_t width_col,
    dt* data_im) {
  CUDA_KERNEL_LOOP(index, n) {
    accT val = static_cast<accT>(0);
    const int64_t w_im = index % width + pad_width;
    const int64_t h_im = (index / width) % height + pad_height;
    const int64_t c_im = index / (width * height);
    int64_t kernel_extent_w = (kernel_w - 1) * dilation_width + 1;
    int64_t kernel_extent_h = (kernel_h - 1) * dilation_height + 1;
    // compute the start and end of the output
    const int64_t w_col_start = (w_im < kernel_extent_w)
        ? 0
        : (w_im - kernel_extent_w) / stride_width + 1;
    const int64_t w_col_end = ::min(w_im / stride_width + 1, width_col);
    const int64_t h_col_start = (h_im < kernel_extent_h)
        ? 0
        : (h_im - kernel_extent_h) / stride_height + 1;
    const int64_t h_col_end = ::min(h_im / stride_height + 1, height_col);

    // TODO: use LCM of stride and dilation to avoid unnecessary loops
    for (int64_t h_col = h_col_start; h_col < h_col_end; h_col += 1) {
      for (int64_t w_col = w_col_start; w_col < w_col_end; w_col += 1) {
        int64_t h_k = (h_im - h_col * stride_height);
        int64_t w_k = (w_im - w_col * stride_width);
        if (h_k % dilation_height == 0 && w_k % dilation_width == 0) {
          h_k /= dilation_height;
          w_k /= dilation_width;
          int64_t data_col_index =
              (((c_im * kernel_h + h_k) * kernel_w + w_k) * height_col +
               h_col) *
                  width_col +
              w_col;
          val += data_col[data_col_index];
        }
      }
    }
    data_im[index] = static_cast<dt>(val);
  }
}

template <typename dt, typename accT>
void col2im(
    cudaStream_t stream,
    const dt* data_col,
    const int64_t channels,
    const int64_t height,
    const int64_t width,
    const int64_t output_height,
    const int64_t output_width,
    const int64_t patch_height,
    const int64_t patch_width,
    const int64_t pad_height,
    const int64_t pad_width,
    const int64_t stride_height,
    const int64_t stride_width,
    const int64_t dilation_height,
    const int64_t dilation_width,
    dt* data_im) {
  int64_t num_kernels = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // CUDA_NUM_THREADS = 1024
  col2im_kernel<dt, accT>
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
          num_kernels,
          data_col,
          height,
          width,
          channels,
          patch_height,
          patch_width,
          pad_height,
          pad_width,
          stride_height,
          stride_width,
          dilation_height,
          dilation_width,
          output_height,
          output_width,
          data_im);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace native
} // namespace at
