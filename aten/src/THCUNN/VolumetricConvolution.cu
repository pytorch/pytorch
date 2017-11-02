#include "THCUNN.h"
#include "common.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"

// Kernel for fast unfold+copy
// Borrowed from Theano
// Authors: Arjun Jain, Frédéric Bastien, Jan Schlüter, Nicolas Ballas
template <typename Dtype>
__global__ void im3d2col_kernel(const int n, const Dtype* data_im,
                                const int height, const int width, const int depth,
                                const int kernel_h, const int kernel_w, const int kernel_d,
                                const int pad_h, const int pad_w, const int pad_d,
                                const int stride_h, const int stride_w, const int stride_d,
                                const int height_col, const int width_col, const int depth_col,
                                Dtype* data_col)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    int d_out = index % depth_col;
    int w_index = index / depth_col;
    int w_out = w_index % width_col;
    int h_index = w_index / width_col;
    int h_out = h_index % height_col;

    int channel_in = h_index / height_col;
    //channel_in = 1;

    int channel_out = channel_in * kernel_h * kernel_w * kernel_d;

    int h_in = h_out * stride_h - pad_h;
    int w_in = w_out * stride_w - pad_w;
    int d_in = d_out * stride_d - pad_d;

    Dtype* data_col_ptr = data_col;
    data_col_ptr += channel_out * (height_col * width_col * depth_col) +
      h_out * (width_col * depth_col) + w_out * depth_col + d_out;

    const Dtype* data_im_ptr = data_im;
    data_im_ptr += channel_in * (height * width * depth) +
      h_in * (width * depth) + w_in * depth + d_in;

    for (int i = 0; i < kernel_h; ++i)
    {
      int h = h_in + i;
      for (int j = 0; j < kernel_w; ++j)
      {
        int w = w_in + j;
        for (int k = 0; k < kernel_d; ++k)
        {
          int d = d_in + k;
          *data_col_ptr = (h >= 0 && w >= 0 && d >= 0 &&
                           h < height && w < width && d < depth) ?
                           data_im_ptr[i * (width * depth) + j *depth + k] : ScalarConvert<int, Dtype>::to(0);
          data_col_ptr += height_col * width_col * depth_col;
        }
      }
    }
  }
}

template <typename Dtype>
void im3d2col(cudaStream_t stream, const Dtype* data_im, const int channels,
              const int height, const int width, const int depth,
              const int kernel_h, const int kernel_w, const int kernel_d,
              const int pad_h, const int pad_w, const int pad_d,
              const int stride_h, const int stride_w, const int stride_d,
              Dtype* data_col)
{
  // We are going to launch channels * height_col * width_col * depth_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  int depth_col = (depth + 2 * pad_d - kernel_d) / stride_d + 1;
  int num_kernels = channels * height_col * width_col * depth_col;
  im3d2col_kernel<<<GET_BLOCKS(num_kernels),
    CUDA_NUM_THREADS, 0, stream>>>(num_kernels, data_im,
                                   height, width, depth,
                                   kernel_h, kernel_w, kernel_d,
                                   pad_h, pad_w, pad_d,
                                   stride_h, stride_w, stride_d,
                                   height_col, width_col, depth_col,
                                   data_col);
  THCudaCheck(cudaGetLastError());
}

template <typename Dtype, typename Acctype>
__global__ void col2im3d_kernel(const int n, const Dtype* data_col,
                                const int height, const int width, const int depth,
                                const int channels,
                                const int patch_h, const int patch_w, const int patch_d,
                                const int pad_h, const int pad_w, const int pad_d,
                                const int stride_h, const int stride_w, const int stride_d,
                                const int height_col, const int width_col, const int depth_col,
                                Dtype* data_im)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    Acctype val = 0;
    int d = index % depth + pad_d;
    int w_index = index / depth;
    int w = w_index % width + pad_w;
    int h_index = w_index / width;
    int h = h_index % height + pad_h;
    int c = h_index / height;

    // compute the start and end of the output
    int d_col_start = (d < patch_d) ? 0 : (d - patch_d) / stride_d + 1;
    int d_col_end = min(d / stride_d + 1, depth_col);
    int w_col_start = (w < patch_w) ? 0 : (w - patch_w) / stride_w + 1;
    int w_col_end = min(w / stride_w + 1, width_col);
    int h_col_start = (h < patch_h) ? 0 : (h - patch_h) / stride_h + 1;
    int h_col_end = min(h / stride_h + 1, height_col);

    int offset =
      (c * patch_h * patch_w * patch_d + h * patch_w * patch_d + w * patch_d + d) * height_col * width_col * depth_col;

    int coeff_h_col = (1 - stride_h * patch_w * patch_d * height_col) * width_col * depth_col;
    int coeff_w_col = (1 - stride_w * patch_d * height_col * width_col) * depth_col;
    int coeff_d_col = (1 - stride_d * height_col * width_col * depth_col);
    for (int d_col = d_col_start; d_col < d_col_end; ++d_col)
      for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
        for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
          val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col + d_col * coeff_d_col];
      }
   }
    data_im[index] = ScalarConvert<Acctype, Dtype>::to(val);
  }
}

template <typename Dtype, typename Acctype>
void col2im3d(cudaStream_t stream, const Dtype* data_col, const int channels,
              const int height, const int width, const int depth,
              const int patch_h, const int patch_w, const int patch_d,
              const int pad_h, const int pad_w, const int pad_d,
              const int stride_h, const int stride_w, const int stride_d,
              Dtype* data_im)
{
  int height_col = (height + 2 * pad_h - patch_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - patch_w) / stride_w + 1;
  int depth_col = (depth + 2 * pad_d - patch_d) / stride_d + 1;
  int num_kernels = channels * height * width * depth;

  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  col2im3d_kernel<Dtype, Acctype><<<GET_BLOCKS(num_kernels),
    CUDA_NUM_THREADS, 0, stream>>>(num_kernels, data_col,
                                   height, width, depth, channels,
                                   patch_h, patch_w, patch_d,
                                   pad_h, pad_w, pad_d,
                                   stride_h, stride_w, stride_d,
                                   height_col, width_col, depth_col,
                                   data_im);
  THCudaCheck(cudaGetLastError());
}

#include "generic/VolumetricConvolution.cu"
#include "THCGenerateFloatTypes.h"
