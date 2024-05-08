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

/**
 * Adapted from https://github.com/torch/cunn/blob/master/lib/THCUNN/SpatialUpSamplingNearest.cu
 *
 * Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
 * Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
 * Copyright (c) 2011-2013 NYU (Clement Farabet)
 * Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert,
 *                         Leon Bottou, Iain Melvin, Jason Weston)
 * Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
 * Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert,
 *                         Samy Bengio, Johnny Mariethoz)
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the names of NEC Laboratories American and IDIAP Research
 *    Institute nor the names of its contributors may be used to endorse or
 *    promote products derived from this software without specific prior
 *    written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */


#include "caffe2/core/context_gpu.h"
#include "modules/detectron/upsample_nearest_op.h"

namespace caffe2 {

namespace {
__device__ int translate_idx(int ii, int d1, int d2, int d3, int scale_factor) {
  int x, y, z, w;
  w = ii % d3;
  ii = ii/d3;
  z = ii % d2;
  ii = ii/d2;
  y = ii % d1;
  ii = ii/d1;
  x = ii;
  w = w/scale_factor;
  z = z/scale_factor;
  d2 /= scale_factor;
  d3 /= scale_factor;
  return (((x*d1+y)*d2)+z)*d3+w;
}

__device__ int translate_idx_inv(
    int ii, int d1, int d2, int d3, int scale_factor, int off_x, int off_y) {
  int x, y, z, w;
  w = ii % d3;
  ii = ii/d3;
  z = ii % d2;
  ii = ii/d2;
  y = ii % d1;
  ii = ii/d1;
  x = ii;
  w = w*scale_factor+off_x;
  z = z*scale_factor+off_y;
  d2 *= scale_factor;
  d3 *= scale_factor;
  return (((x*d1+y)*d2)+z)*d3+w;
}

__global__ void upscale(const float *input, float *output, long no_elements,
                        int scale_factor, int d1, int d2, int d3) {
  long ii = threadIdx.x + blockDim.x * blockIdx.x;
  ii += threadIdx.y + blockDim.y * (blockDim.x * gridDim.x) * blockIdx.y;
  if (ii >= no_elements) return;
  int ipidx = translate_idx(ii, d1, d2, d3, scale_factor);
  output[ii]=input[ipidx];
}

__global__ void downscale(float *gradInput_data, const float *gradOutput_data,
                          long no_elements, int scale_factor, int d1, int d2,
                          int d3) {
  long ii = threadIdx.x + blockDim.x * blockIdx.x;
  ii += threadIdx.y + blockDim.y * (blockDim.x * gridDim.x) * blockIdx.y;
  if (ii >= no_elements) return;
  for (int i=0; i < scale_factor; i++){
    for(int j=0; j < scale_factor; j++){
      int ipidx = translate_idx_inv(ii, d1, d2, d3, scale_factor, i, j);
      gradInput_data[ii] += gradOutput_data[ipidx];
    }
  }
}
} // namespace

template<>
bool UpsampleNearestOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);

  vector<int64_t> out_shape;
  for (int i = 0; i < X.ndim(); ++i) {
    out_shape.push_back(X.dim32(i));
  }
  out_shape[X.ndim() - 1] *= scale_;
  out_shape[X.ndim() - 2] *= scale_;
  Y->Resize(out_shape);

  int d1;
  int d2;
  int d3;
  if (X.ndim() == 3) {
    d1 = Y->dim32(0);
    d2 = Y->dim32(1);
    d3 = Y->dim32(2);
  } else {
    d1 = Y->dim32(1);
    d2 = Y->dim32(2);
    d3 = Y->dim32(3);
  }
  long no_elements = Y->size();

  const float *input_data = X.data<float>();
  float *output_data = Y->mutable_data<float>();

  // cuda blocks & threads:
  long nthreads = 256;
  // Max number of blocks: http://en.wikipedia.org/wiki/CUDA
  // 65535 for SM 2.x, 2^32 -1 for >= 3.0
  // TODO: When we move to SM 3.5 we should update this
  long n_xblocks = min(max((int)ceil((float)no_elements / nthreads), 1), 65535);
  long n_yblocks = (long)ceil(
      (float)no_elements / (float)(n_xblocks * nthreads));
  CAFFE_ENFORCE(n_yblocks <= 65535);
  dim3 blocks(n_xblocks, n_yblocks);
  dim3 threads(nthreads);

  upscale<<<blocks, threads, 0, context_.cuda_stream()>>>(
      input_data, output_data, no_elements, scale_, d1, d2, d3);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}


template<>
bool UpsampleNearestGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& X  = Input(0);   // Original input to "forward" op
  auto& dY = Input(1);   // Gradient of net w.r.t. output of "forward" op
                         // (aka "gradOutput")
  auto* dX = Output(0);  // Gradient of net w.r.t. input to "forward" op
                         // (aka "gradInput")

  dX->ResizeLike(X);
  float *gradInput_data = dX->mutable_data<float>();
  const float *gradOutput_data = dY.data<float>();

  int d1;
  int d2;
  int d3;
  if (dX->ndim() == 3) {
    d1 = dX->dim32(0);
    d2 = dX->dim32(1);
    d3 = dX->dim32(2);
  } else {
    d1 = dX->dim32(1);
    d2 = dX->dim32(2);
    d3 = dX->dim32(3);
  }
  long no_elements = dX->size();

  // cuda blocks & threads:
  long nthreads = 256;
  // Max number of blocks: http://en.wikipedia.org/wiki/CUDA
  // 65535 for SM 2.x, 2^32 -1 for >= 3.0
  // TODO: When we move to SM 3.5 we should update this
  long n_xblocks = min(max((int)ceil((float)no_elements / nthreads), 1), 65535);
  long n_yblocks = (long)ceil(
      (float)no_elements / (float)(n_xblocks * nthreads));
  CAFFE_ENFORCE(n_yblocks <= 65535);
  dim3 blocks(n_xblocks, n_yblocks);
  dim3 threads(nthreads);

  math::Set<float, CUDAContext>(no_elements, 0.f, gradInput_data, &context_);
  downscale<<<blocks, threads, 0, context_.cuda_stream()>>>(
      gradInput_data, gradOutput_data, no_elements, scale_, d1, d2, d3);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

REGISTER_CUDA_OPERATOR(UpsampleNearest,
                       UpsampleNearestOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(UpsampleNearestGradient,
                       UpsampleNearestGradientOp<float, CUDAContext>);
} // namespace caffe2
