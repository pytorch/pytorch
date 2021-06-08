/*!
 ******************* BEGIN Caffe Copyright Notice and Disclaimer ****************
 *
 * COPYRIGHT
 *
 * All contributions by the University of California:
 * Copyright (c) 2014-2017 The Regents of the University of California (Regents)
 * All rights reserved.
 *
 * All other contributions:
 * Copyright (c) 2014-2017, the respective contributors
 * All rights reserved.
 *
 * Caffe uses a shared copyright model: each contributor holds copyright over
 * their contributions to Caffe. The project versioning records all such
 * contribution and copyright details. If a contributor wants to further mark
 * their specific copyright on a particular contribution, they should indicate
 * their copyright solely in the commit message of the change when it is
 * committed.
 *
 * LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * CONTRIBUTION AGREEMENT
 *
 * By contributing to the BVLC/caffe repository through pull-request, comment,
 * or otherwise, the contributor releases their content to the
 * license and copyright terms herein.
 *
 ***************** END Caffe Copyright Notice and Disclaimer ********************
 *
 * Copyright (c) 2017 Microsoft
 * Licensed under The Apache-2.0 License [see LICENSE for details]
 * \file deformable_im2col.cuh
 * \brief Function definitions of converting an image to
 * column matrix based on kernel, padding, dilation, and offset.
 * These functions are mainly used in deformable convolution operators.
 * \ref: https://arxiv.org/abs/1703.06211
 * \author Yuwen Xiong, Haozhi Qi, Jifeng Dai
 */

#include <cub/block/block_reduce.cuh>
#include <vector>
#include "caffe2/core/common.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/deform_conv_op.h"
#include "caffe2/operators/deform_conv_op_impl.h"

namespace caffe2 {

typedef int64_t index_t;
typedef std::vector<int64_t> TShape;

template <typename DType>
__device__ DType deformable_im2col_bilinear(
    const DType* bottom_data,
    const int data_width,
    const int height,
    const int width,
    DType h,
    DType w) {
  int h_low = floor(h);
  int w_low = floor(w);
  int h_high;
  int w_high;
  if (h_low >= height - 1) {
    h_high = h_low = height - 1;
    h = (DType)h_low;
  } else {
    h_high = h_low + 1;
  }

  if (w_low >= width - 1) {
    w_high = w_low = width - 1;
    w = (DType)w_low;
  } else {
    w_high = w_low + 1;
  }

  DType lh = h - h_low;
  DType lw = w - w_low;
  DType hh = 1 - lh, hw = 1 - lw;

  DType v1 = bottom_data[h_low * data_width + w_low];
  DType v2 = bottom_data[h_low * data_width + w_high];
  DType v3 = bottom_data[h_high * data_width + w_low];
  DType v4 = bottom_data[h_high * data_width + w_high];
  DType w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  DType val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <typename DType>
__device__ DType get_gradient_weight(
    DType argmax_h,
    DType argmax_w,
    const int h,
    const int w,
    const int height,
    const int width) {
  if (argmax_h < 0 || argmax_h > height || argmax_w < 0 || argmax_w > width) {
    // empty
    return 0;
  }

  argmax_h = max(argmax_h, (DType)0.0f);
  argmax_w = max(argmax_w, (DType)0.0f);

  int argmax_h_low = (int)argmax_h;
  int argmax_w_low = (int)argmax_w;
  int argmax_h_high;
  int argmax_w_high;
  if (argmax_h_low >= height - 1) {
    argmax_h_high = argmax_h_low = height - 1;
    argmax_h = (DType)argmax_h_low;
  } else {
    argmax_h_high = argmax_h_low + 1;
  }
  if (argmax_w_low >= width - 1) {
    argmax_w_high = argmax_w_low = width - 1;
    argmax_w = (DType)argmax_w_low;
  } else {
    argmax_w_high = argmax_w_low + 1;
  }
  DType weight = 0;
  if (h == argmax_h_low) {
    if (w == argmax_w_low) {
      weight = (h + 1 - argmax_h) * (w + 1 - argmax_w);
    } else if (w == argmax_w_high) {
      weight = (h + 1 - argmax_h) * (argmax_w + 1 - w);
    }
  } else if (h == argmax_h_high) {
    if (w == argmax_w_low) {
      weight = (argmax_h + 1 - h) * (w + 1 - argmax_w);
    } else if (w == argmax_w_high) {
      weight = (argmax_h + 1 - h) * (argmax_w + 1 - w);
    }
  }
  return weight;
}

template <typename DType>
__device__ DType get_coordinate_weight(
    DType argmax_h,
    DType argmax_w,
    const int height,
    const int width,
    const DType* im_data,
    const int data_width,
    const int bp_dir) {
  if (argmax_h < 0 || argmax_h > height || argmax_w < 0 || argmax_w > width) {
    // empty
    return 0;
  }

  if (argmax_h < 0)
    argmax_h = 0;
  if (argmax_w < 0)
    argmax_w = 0;

  int argmax_h_low = (int)argmax_h;
  int argmax_w_low = (int)argmax_w;
  int argmax_h_high;
  int argmax_w_high;
  if (argmax_h_low >= height - 1) {
    argmax_h_high = argmax_h_low = height - 1;
    argmax_h = (DType)argmax_h_low;
  } else {
    argmax_h_high = argmax_h_low + 1;
  }
  if (argmax_w_low >= width - 1) {
    argmax_w_high = argmax_w_low = width - 1;
    argmax_w = (DType)argmax_w_low;
  } else {
    argmax_w_high = argmax_w_low + 1;
  }
  DType weight = 0;

  if (bp_dir == 0) {
    weight += -1 * (argmax_w_low + 1 - argmax_w) *
        im_data[argmax_h_low * data_width + argmax_w_low];
    weight += -1 * (argmax_w - argmax_w_low) *
        im_data[argmax_h_low * data_width + argmax_w_high];
    weight += (argmax_w_low + 1 - argmax_w) *
        im_data[argmax_h_high * data_width + argmax_w_low];
    weight += (argmax_w - argmax_w_low) *
        im_data[argmax_h_high * data_width + argmax_w_high];
  } else if (bp_dir == 1) {
    weight += -1 * (argmax_h_low + 1 - argmax_h) *
        im_data[argmax_h_low * data_width + argmax_w_low];
    weight += (argmax_h_low + 1 - argmax_h) *
        im_data[argmax_h_low * data_width + argmax_w_high];
    weight += -1 * (argmax_h - argmax_h_low) *
        im_data[argmax_h_high * data_width + argmax_w_low];
    weight += (argmax_h - argmax_h_low) *
        im_data[argmax_h_high * data_width + argmax_w_high];
  }

  return weight;
}

/*!
 * \brief deformable_im2col gpu kernel.
 * DO NOT call this directly. Use wrapper function im2col() instead;
 */
template <typename DType>
__global__ void deformable_im2col_gpu_kernel(
    const int n,
    const DType* data_im,
    const DType* data_offset,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int channel_per_deformable_group,
    const int height_col,
    const int width_col,
    DType* data_col) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    // index index of output matrix
    const int w_col = index % width_col;
    const int h_col = (index / width_col) % height_col;
    const int c_im = (index / width_col) / height_col;
    const int c_col = c_im * kernel_h * kernel_w;

    // compute deformable group index
    const int deformable_group_index = c_im / channel_per_deformable_group;

    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;
    DType* data_col_ptr =
        data_col + (c_col * height_col + h_col) * width_col + w_col;
    const DType* data_im_ptr = data_im + (c_im * height + h_in) * width + w_in;
    const DType* data_offset_ptr = data_offset +
        deformable_group_index * 2 * kernel_h * kernel_w * height_col *
            width_col;

    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        const int data_offset_h_ptr =
            ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
        const int data_offset_w_ptr =
            ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col +
            w_col;
        const DType offset_h = data_offset_ptr[data_offset_h_ptr];
        const DType offset_w = data_offset_ptr[data_offset_w_ptr];
        DType val = static_cast<DType>(0);
        const DType h_im = h_in + i * dilation_h + offset_h;
        const DType w_im = w_in + j * dilation_w + offset_w;
        if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) {
          const DType map_h = i * dilation_h + offset_h;
          const DType map_w = j * dilation_w + offset_w;
          const int cur_height = height - h_in;
          const int cur_width = width - w_in;
          val = deformable_im2col_bilinear(
              data_im_ptr, width, cur_height, cur_width, map_h, map_w);
        }
        *data_col_ptr = val;
        data_col_ptr += height_col * width_col;
      }
    }
  }
}

/*!\brief
 * cpu function of deformable_im2col algorithm
 * \param s device stream
 * \param data_im pointer of an image (C, H, W, ...) in the image batch
 * \param data_offset pointer of offset (C, H, W, ...) in the offset batch
 * \param im_shape input image shape in dimensions (N, C, H, W,)
 * \param col_shape column buffer shape (#channels, output_im_height,
 * output_im_width, ...) \param kernel_shape kernel filter shape \param pad pad
 * shape \param stride stride shape \param dilation dilation shape \param
 * deformable_group #offset group that deformable convolution use \param
 * data_col column buffer pointer
 */
template <typename DType, typename Context>
void DeformConvOpBase<DType, Context>::DeformableIm2col(
    const DType* data_im,
    const DType* data_offset,
    at::IntArrayRef im_shape,
    at::IntArrayRef col_shape,
    DType* data_col) {
  CHECK_LT(2, CAFFE_CUDA_NUM_THREADS);
  CAFFE_ENFORCE_EQ(pad_t(), pad_b());
  CAFFE_ENFORCE_EQ(pad_l(), pad_r());
  const int pad_h = pad_t();
  const int pad_w = pad_l();
  index_t channel_per_deformable_group = im_shape[1] / deformable_group_;
  index_t num_kernels = im_shape[1] * size_from_dim_(1, col_shape);
  deformable_im2col_gpu_kernel<DType>
      <<<CAFFE_GET_BLOCKS(num_kernels),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(
          num_kernels,
          data_im,
          data_offset,
          im_shape[2],
          im_shape[3],
          kernel_h(),
          kernel_w(),
          pad_h,
          pad_w,
          stride_h(),
          stride_w(),
          dilation_h(),
          dilation_w(),
          channel_per_deformable_group,
          col_shape[1],
          col_shape[2],
          data_col);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

/*!
 * \brief deformable_col2im gpu kernel.
 * \brief DO NOT call this directly. Use wrapper function deformable_col2im()
 * instead;
 */
template <typename DType>
__global__ void deformable_col2im_gpu_kernel(
    const int n,
    const DType* data_col,
    const DType* data_offset,
    const int channels,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int channel_per_deformable_group,
    const int height_col,
    const int width_col,
    DType* grad_im) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    const int j = (index / width_col / height_col) % kernel_w;
    const int i = (index / width_col / height_col / kernel_w) % kernel_h;
    const int c = index / width_col / height_col / kernel_w / kernel_h;
    // compute the start and end of the output

    const int deformable_group_index = c / channel_per_deformable_group;

    int w_out = index % width_col;
    int h_out = (index / width_col) % height_col;
    int w_in = w_out * stride_w - pad_w;
    int h_in = h_out * stride_h - pad_h;

    const DType* data_offset_ptr = data_offset +
        deformable_group_index * 2 * kernel_h * kernel_w * height_col *
            width_col;
    const int data_offset_h_ptr =
        ((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out;
    const int data_offset_w_ptr =
        ((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out;
    const DType offset_h = data_offset_ptr[data_offset_h_ptr];
    const DType offset_w = data_offset_ptr[data_offset_w_ptr];
    const DType cur_inv_h_data = h_in + i * dilation_h + offset_h;
    const DType cur_inv_w_data = w_in + j * dilation_w + offset_w;

    const DType cur_top_grad = data_col[index];
    const int cur_h = (int)cur_inv_h_data;
    const int cur_w = (int)cur_inv_w_data;
    for (int dy = -2; dy <= 2; dy++) {
      for (int dx = -2; dx <= 2; dx++) {
        if (cur_h + dy >= 0 && cur_h + dy < height && cur_w + dx >= 0 &&
            cur_w + dx < width &&
            c10::cuda::compat::abs(cur_inv_h_data - (cur_h + dy)) < 1 &&
            c10::cuda::compat::abs(cur_inv_w_data - (cur_w + dx)) < 1) {
          int cur_bottom_grad_pos =
              (c * height + cur_h + dy) * width + cur_w + dx;
          DType weight = get_gradient_weight(
              cur_inv_h_data,
              cur_inv_w_data,
              cur_h + dy,
              cur_w + dx,
              height,
              width);
          atomicAdd(grad_im + cur_bottom_grad_pos, weight * cur_top_grad);
        }
      }
    }
  }
}

/*!\brief
 * gpu function of deformable_col2im algorithm
 * \param s device stream
 * \param data_col start pointer of the column buffer to be filled
 * \param data_offset pointer of offset (C, H, W, ...) in the offset batch
 * \param im_shape input image shape in dimensions (N, C, H, W,)
 * \param col_shape column buffer shape
 * \param kernel_shape kernel filter shape
 * \param pad pad shape
 * \param stride stride shape
 * \param dilation dilation shape
 * \param deformable_group #offset group that deformable convolution use
 * \param grad_im pointer of a image (C, H, W,...) in the image batch
 */
template <typename DType, typename Context>
void DeformConvOpBase<DType, Context>::DeformableCol2im(
    const DType* data_col,
    const DType* data_offset,
    at::IntArrayRef im_shape,
    at::IntArrayRef col_shape,
    DType* grad_im) {
  CAFFE_ENFORCE_EQ(pad_t(), pad_b());
  CAFFE_ENFORCE_EQ(pad_l(), pad_r());
  const int pad_h = pad_t();
  const int pad_w = pad_l();
  index_t im_size = size_from_dim_(1, im_shape);
  index_t channel_per_deformable_group = im_shape[1] / deformable_group_;
  index_t num_kernels = size_from_dim_(0, col_shape);
  // num_axes should be smaller than block size
  CHECK_LT(2, CAFFE_CUDA_NUM_THREADS);
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)
  deformable_col2im_gpu_kernel<DType>
      <<<CAFFE_GET_BLOCKS(num_kernels),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(
          num_kernels,
          data_col,
          data_offset,
          im_shape[1],
          im_shape[2],
          im_shape[3],
          kernel_h(),
          kernel_w(),
          pad_h,
          pad_w,
          stride_h(),
          stride_w(),
          dilation_h(),
          dilation_w(),
          channel_per_deformable_group,
          col_shape[1],
          col_shape[2],
          grad_im);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

/*!
 * \brief deformable_col2im_coord gpu kernel.
 * \brief DO NOT call this directly. Use wrapper function
 * deformable_col2im_coord() instead;
 */
template <typename DType>
__global__ void deformable_col2im_coord_gpu_kernel(
    const int n,
    const DType* data_col,
    const DType* data_im,
    const DType* data_offset,
    const int channels,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int channel_per_deformable_group,
    const int height_col,
    const int width_col,
    DType* grad_offset) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    DType val = 0;
    int w = index % width_col;
    int h = (index / width_col) % height_col;
    int c = index / width_col / height_col;
    // compute the start and end of the output

    const int deformable_group_index = c / (2 * kernel_h * kernel_w);
    const int col_step = kernel_h * kernel_w;
    int cnt = 0;
    const DType* data_col_ptr = data_col +
        deformable_group_index * channel_per_deformable_group * width_col *
            height_col;
    const DType* data_im_ptr = data_im +
        deformable_group_index * channel_per_deformable_group / kernel_h /
            kernel_w * height * width;
    const DType* data_offset_ptr = data_offset +
        deformable_group_index * 2 * kernel_h * kernel_w * height_col *
            width_col;

    const int offset_c = c - deformable_group_index * 2 * kernel_h * kernel_w;

    for (int col_c = (offset_c / 2); col_c < channel_per_deformable_group;
         col_c += col_step) {
      const int col_pos = ((col_c * height_col) + h) * width_col + w;
      const int bp_dir = offset_c % 2;

      int j = (col_pos / width_col / height_col) % kernel_w;
      int i = (col_pos / width_col / height_col / kernel_w) % kernel_h;
      int w_out = col_pos % width_col;
      int h_out = (col_pos / width_col) % height_col;
      int w_in = w_out * stride_w - pad_w;
      int h_in = h_out * stride_h - pad_h;
      const int data_offset_h_ptr =
          (((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out);
      const int data_offset_w_ptr =
          (((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col +
           w_out);
      const DType offset_h = data_offset_ptr[data_offset_h_ptr];
      const DType offset_w = data_offset_ptr[data_offset_w_ptr];
      DType inv_h = h_in + i * dilation_h + offset_h;
      DType inv_w = w_in + j * dilation_w + offset_w;
      if (inv_h < 0 || inv_w < 0 || inv_h >= height || inv_w >= width) {
        inv_h = inv_w = -1;
      }
      const DType weight = get_coordinate_weight(
          inv_h,
          inv_w,
          height,
          width,
          data_im_ptr + cnt * height * width,
          width,
          bp_dir);
      val += weight * data_col_ptr[col_pos];
      cnt += 1;
    }

    grad_offset[index] = val;
  }
}

/*!\brief
 * gpu function of deformable_col2im_coord algorithm
 * \param s device stream
 * \param data_col start pointer of the column buffer to be filled
 * \param data_im pointer of an image (C, H, W, ...) in the image batch
 * \param data_offset pointer of offset (C, H, W, ...) in the offset batch
 * \param im_shape input image shape in dimensions (N, C, H, W,)
 * \param col_shape column buffer shape
 * \param kernel_shape kernel filter shape
 * \param pad pad shape
 * \param stride stride shape
 * \param dilation dilation shape
 * \param deformable_group #offset group that deformable convolution use
 * \param grad_offset pointer of the offset (C, H, W,...) in the offset batch
 */
template <typename DType, typename Context>
void DeformConvOpBase<DType, Context>::DeformableCol2imCoord(
    const DType* data_col,
    const DType* data_im,
    const DType* data_offset,
    at::IntArrayRef im_shape,
    at::IntArrayRef col_shape,
    DType* grad_offset) {
  CAFFE_ENFORCE_EQ(pad_t(), pad_b());
  CAFFE_ENFORCE_EQ(pad_l(), pad_r());
  const int pad_h = pad_t();
  const int pad_w = pad_l();
  index_t num_kernels = col_shape[1] * col_shape[2] * 2 * kernel_h() *
      kernel_w() * deformable_group_;
  index_t channel_per_deformable_group = col_shape[0] / deformable_group_;
  // num_axes should be smaller than block size
  CHECK_LT(2, CAFFE_CUDA_NUM_THREADS);
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)
  deformable_col2im_coord_gpu_kernel<DType>
      <<<CAFFE_GET_BLOCKS(num_kernels),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context_.cuda_stream()>>>(
          num_kernels,
          data_col,
          data_im,
          data_offset,
          im_shape[1],
          im_shape[2],
          im_shape[3],
          kernel_h(),
          kernel_w(),
          pad_h,
          pad_w,
          stride_h(),
          stride_w(),
          dilation_h(),
          dilation_w(),
          channel_per_deformable_group,
          col_shape[1],
          col_shape[2],
          grad_offset);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

REGISTER_CUDA_OPERATOR(DeformConv, DeformConvOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    DeformConvGradient,
    DeformConvGradientOp<float, CUDAContext>);

} // namespace caffe2
