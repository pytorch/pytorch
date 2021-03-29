/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Based on https://github.com/daijifeng001/caffe-rfcn/blob/r-fcn/src/caffe/layers/psroi_pooling_layer.cu
//
// ------------------------------------------------------------------
// R-FCN
// Copyright (c) 2016 Microsoft
// Licensed under The MIT License [see r-fcn/LICENSE for details]
// Written by Yi Li
// ------------------------------------------------------------------
//
// COPYRIGHT
//
// All contributions by the University of California:
// Copyright (c) 2014, 2015, The Regents of the University of California
// (Regents)
// All rights reserved.
//
// All other contributions:
// Copyright (c) 2014, 2015, the respective contributors
// All rights reserved.
//
// Caffe uses a shared copyright model: each contributor holds copyright over
// their contributions to Caffe. The project versioning records all such
// contribution and copyright details. If a contributor wants to further mark
// their specific copyright on a particular contribution, they should indicate
// their copyright solely in the commit message of the change when it is
// committed.
//
// LICENSE
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// CONTRIBUTION AGREEMENT
//
// By contributing to the BVLC/caffe repository through pull-request, comment,
// or otherwise, the contributor releases their content to the
// license and copyright terms herein.

#include <cfloat>

#include "caffe2/core/context_gpu.h"
#include "modules/detectron/ps_roi_pool_op.h"

namespace caffe2 {

namespace {

template <typename T>
inline __device__ T gpu_atomic_add(const T val, T* address);

template <>
inline __device__
float gpu_atomic_add(const float val, float* address) {
  return atomicAdd(address, val);
}

template <typename T>
__global__ void PSRoIPoolForward(
    const int nthreads,
    const T* bottom_data,
    const T spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const T* bottom_rois,
    const int output_dim,
    const int group_size,
    T* top_data,
    int* mapping_channel) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // The output is in order (n, ctop, ph, pw)
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int ctop = (index / pooled_width / pooled_height) % output_dim;
    int n = index / pooled_width / pooled_height / output_dim;

    // [start, end) interval for spatial sampling
    const T* offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];
    T roi_start_w = static_cast<T>(
      roundf(offset_bottom_rois[1])) * spatial_scale;
    T roi_start_h = static_cast<T>(
      roundf(offset_bottom_rois[2])) * spatial_scale;
    T roi_end_w = static_cast<T>(
      roundf(offset_bottom_rois[3]) + 1.) * spatial_scale;
    T roi_end_h = static_cast<T>(
      roundf(offset_bottom_rois[4]) + 1.) * spatial_scale;

    // Force too small ROIs to be 1x1
    T roi_width = c10::cuda::compat::max(roi_end_w - roi_start_w, static_cast<T>(0.1));  // avoid 0
    T roi_height = c10::cuda::compat::max(roi_end_h - roi_start_h, static_cast<T>(0.1));

    // Compute w and h at bottom
    T bin_size_h = roi_height / static_cast<T>(pooled_height);
    T bin_size_w = roi_width / static_cast<T>(pooled_width);

    // Add roi offsets and clip to input boundaries
    int hstart = floor(
      static_cast<T>(ph) * bin_size_h + roi_start_h);
    int wstart = floor(
      static_cast<T>(pw)* bin_size_w + roi_start_w);
    int hend = ceil(
      static_cast<T>(ph + 1) * bin_size_h + roi_start_h);
    int wend = ceil(
      static_cast<T>(pw + 1) * bin_size_w + roi_start_w);

    hstart = min(max(hstart, 0), height);
    hend = min(max(hend, 0), height);
    wstart = min(max(wstart, 0),width);
    wend = min(max(wend, 0), width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    int gw = pw;
    int gh = ph;
    int c = (ctop * group_size + gh) * group_size + gw;

    const T* offset_bottom_data =
      bottom_data + (roi_batch_ind * channels + c) * height * width;
    T out_sum = 0;
    for (int h = hstart; h < hend; ++h){
     for (int w = wstart; w < wend; ++w){
       int bottom_index = h*width + w;
       out_sum += offset_bottom_data[bottom_index];
     }
    }

    T bin_area = (hend - hstart) * (wend - wstart);
    top_data[index] = is_empty ? 0. : out_sum / bin_area;
    mapping_channel[index] = c;
  }
}

template <typename T>
__global__ void PSRoIPoolBackward(
    const int nthreads,
    const T* top_diff,
    const int* mapping_channel,
    const int num_rois,
    const T spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int output_dim,
    T* bottom_diff,
    const T* bottom_rois) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // The output is in order (n, ctop, ph, pw)
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int n = index / pooled_width / pooled_height / output_dim;

    // [start, end) interval for spatial sampling
    const T* offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];
    T roi_start_w = static_cast<T>(
      roundf(offset_bottom_rois[1])) * spatial_scale;
    T roi_start_h = static_cast<T>(
      roundf(offset_bottom_rois[2])) * spatial_scale;
    T roi_end_w = static_cast<T>(
      roundf(offset_bottom_rois[3]) + 1.) * spatial_scale;
    T roi_end_h = static_cast<T>(
      roundf(offset_bottom_rois[4]) + 1.) * spatial_scale;

    // Force too small ROIs to be 1x1
    T roi_width = c10::cuda::compat::max(roi_end_w - roi_start_w, static_cast<T>(0.1)); //avoid 0
    T roi_height = c10::cuda::compat::max(roi_end_h - roi_start_h, static_cast<T>(0.1));

    // Compute w and h at bottom
    T bin_size_h = roi_height / static_cast<T>(pooled_height);
    T bin_size_w = roi_width / static_cast<T>(pooled_width);

    int hstart = floor(
      static_cast<T>(ph)* bin_size_h + roi_start_h);
    int wstart = floor(
      static_cast<T>(pw)* bin_size_w + roi_start_w);
    int hend = ceil(
      static_cast<T>(ph + 1) * bin_size_h + roi_start_h);
    int wend = ceil(
      static_cast<T>(pw + 1) * bin_size_w + roi_start_w);
    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart, 0), height);
    hend = min(max(hend, 0), height);
    wstart = min(max(wstart, 0), width);
    wend = min(max(wend, 0), width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    // Compute c at bottom
    int c = mapping_channel[index];
    T* offset_bottom_diff =
      bottom_diff + (roi_batch_ind * channels + c) * height * width;
    T bin_area = (hend - hstart) * (wend - wstart);
    T diff_val = is_empty ? 0. : top_diff[index] / bin_area;
    for (int h = hstart; h < hend; ++h){
      for (int w = wstart; w < wend; ++w){
        int bottom_index = h * width + w;
        gpu_atomic_add(diff_val, offset_bottom_diff + bottom_index);
      }
    }
  }
}

} // namespace

template<>
bool PSRoIPoolOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);  // Input data to pool
  auto& R = Input(1);  // RoIs

  auto* Y = Output(0, {R.dim32(0), output_dim_, pooled_height_, pooled_width_}, at::dtype<float>()); // PSRoI pooled data
  auto* A = Output(1, Y->sizes(), at::dtype<int>()); // mapping_channel
  int output_size = Y->numel();
  PSRoIPoolForward<float><<<CAFFE_GET_BLOCKS(output_size),
                            CAFFE_CUDA_NUM_THREADS,
                            0, context_.cuda_stream()>>>(
      output_size, X.data<float>(), spatial_scale_, X.dim32(1), X.dim32(2),
      X.dim32(3), pooled_height_, pooled_width_, R.data<float>(), output_dim_,
      group_size_, Y->mutable_data<float>(), A->mutable_data<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return true;
}


template<>
bool PSRoIPoolGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& X  = Input(0);  // Input data to pool
  auto& R  = Input(1);  // RoIs
  auto& A  = Input(2);  // mapping channels
  auto& dY = Input(3);  // Gradient of net w.r.t. output of "forward" op
                        // (aka "gradOutput")

  auto* dX = Output(0, X.sizes(), at::dtype<float>()); // Gradient of net w.r.t. input to "forward" op
                                                       // (aka "gradInput")
  // Must zero-out dX before accumulating gradients
  math::Set<float, CUDAContext>(
      dX->size(), 0.f, dX->mutable_data<float>(), &context_);
  PSRoIPoolBackward<float><<<CAFFE_GET_BLOCKS(dY.size()),
                             CAFFE_CUDA_NUM_THREADS,
                             0, context_.cuda_stream()>>>(
      dY.size(), dY.data<float>(), A.data<int>(), R.dim32(0), spatial_scale_,
      X.dim32(1), X.dim32(2), X.dim32(3), pooled_height_, pooled_width_,
      output_dim_, dX->mutable_data<float>(), R.data<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return true;
}


REGISTER_CUDA_OPERATOR(PSRoIPool,
                       PSRoIPoolOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(PSRoIPoolGradient,
                       PSRoIPoolGradientOp<float, CUDAContext>);
} // namespace caffe2
