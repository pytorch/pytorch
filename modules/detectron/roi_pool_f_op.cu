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

#include <cfloat>

#include "caffe2/core/context_gpu.h"
#include "modules/detectron/roi_pool_f_op.h"

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
__global__ void RoIPoolFForward(const int nthreads, const T* bottom_data,
    const T spatial_scale, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const T* bottom_rois, T* top_data, int* argmax_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];
    int roi_start_w = roundf(offset_bottom_rois[1] * spatial_scale);
    int roi_start_h = roundf(offset_bottom_rois[2] * spatial_scale);
    int roi_end_w = roundf(offset_bottom_rois[3] * spatial_scale);
    int roi_end_h = roundf(offset_bottom_rois[4] * spatial_scale);

    // Force malformed ROIs to be 1x1
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    T bin_size_h = static_cast<T>(roi_height)
                       / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width)
                       / static_cast<T>(pooled_width);

    int hstart = static_cast<int>(floor(static_cast<T>(ph)
                                        * bin_size_h));
    int wstart = static_cast<int>(floor(static_cast<T>(pw)
                                        * bin_size_w));
    int hend = static_cast<int>(ceil(static_cast<T>(ph + 1)
                                     * bin_size_h));
    int wend = static_cast<int>(ceil(static_cast<T>(pw + 1)
                                     * bin_size_w));

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, 0), height);
    hend = min(max(hend + roi_start_h, 0), height);
    wstart = min(max(wstart + roi_start_w, 0), width);
    wend = min(max(wend + roi_start_w, 0), width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    // Define an empty pooling region to be zero
    T maxval = is_empty ? 0 : -FLT_MAX;
    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    int maxidx = -1;
    const T* offset_bottom_data =
        bottom_data + (roi_batch_ind * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int bottom_index = h * width + w;
        if (offset_bottom_data[bottom_index] > maxval) {
          maxval = offset_bottom_data[bottom_index];
          maxidx = bottom_index;
        }
      }
    }
    top_data[index] = maxval;
    argmax_data[index] = maxidx;
  }
}

template <typename T>
__global__ void RoIPoolFBackward(const int nthreads, const T* top_diff,
    const int* argmax_data, const int num_rois, const T spatial_scale,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, T* bottom_diff,
    const T* bottom_rois) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];
    int bottom_offset = (roi_batch_ind * channels + c) * height * width;
    int top_offset    = (n * channels + c) * pooled_height * pooled_width;
    const T* offset_top_diff = top_diff + top_offset;
    T* offset_bottom_diff = bottom_diff + bottom_offset;
    const int* offset_argmax_data = argmax_data + top_offset;

    int argmax = offset_argmax_data[ph * pooled_width + pw];
    if (argmax != -1) {
      gpu_atomic_add(
          static_cast<T>(offset_top_diff[ph * pooled_width + pw]),
          offset_bottom_diff + argmax);
    }
  }
}

} // namespace

template<>
bool RoIPoolFOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);  // Input data to pool
  auto& R = Input(1);  // RoIs

  if (R.size() == 0) {
    // Handle empty rois
    std::vector<int64_t> sizes = {0, X.dim32(1), pooled_height_, pooled_width_};
    /* auto* Y = */ Output(0, sizes, at::dtype<float>());
    /* auto* A = */ Output(1, sizes, at::dtype<int>());
    return true;
  }

  auto* Y = Output(0, {R.dim32(0), X.dim32(1), pooled_height_, pooled_width_}, at::dtype<float>()); // RoI pooled data
  auto* A = Output(1, Y->sizes(), at::dtype<int>()); // argmaxes
  int output_size = Y->size();
  RoIPoolFForward<float><<<CAFFE_GET_BLOCKS(output_size),
                          CAFFE_CUDA_NUM_THREADS,
                          0, context_.cuda_stream()>>>(
      output_size, X.data<float>(), spatial_scale_, X.dim32(1), X.dim32(2),
      X.dim32(3), pooled_height_, pooled_width_, R.data<float>(),
      Y->mutable_data<float>(), A->mutable_data<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return true;
}


template<>
bool RoIPoolFGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& X  = Input(0);  // Input data to pool
  auto& R  = Input(1);  // RoIs
  auto& A  = Input(2);  // argmaxes
  auto& dY = Input(3);  // Gradient of net w.r.t. output of "forward" op
                        // (aka "gradOutput")

  auto* dX = Output(0, X.sizes(), at::dtype<float>());    // Gradient of net w.r.t. input to "forward" op
                        // (aka "gradInput")
  // Must zero-out dX before accumulating gradients
  math::Set<float, CUDAContext>(
      dX->size(), 0.f, dX->mutable_data<float>(), &context_);
  if (dY.size() > 0) {  // Handle possibly empty gradient if there were no rois
    RoIPoolFBackward<float><<<CAFFE_GET_BLOCKS(dY.size()),
                             CAFFE_CUDA_NUM_THREADS,
                             0, context_.cuda_stream()>>>(
        dY.size(), dY.data<float>(), A.data<int>(), R.dim32(0), spatial_scale_,
        X.dim32(1), X.dim32(2), X.dim32(3), pooled_height_, pooled_width_,
        dX->mutable_data<float>(), R.data<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return true;
}


REGISTER_CUDA_OPERATOR(RoIPoolF,
                       RoIPoolFOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(RoIPoolFGradient,
                       RoIPoolFGradientOp<float, CUDAContext>);
} // namespace caffe2
