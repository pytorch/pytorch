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
#include "modules/detectron/group_spatial_softmax_op.h"

namespace caffe2 {

namespace {

__global__ void GroupSpatialSoftmaxKernel(const int num, const int A, const int W,
    const int H, const float* Xdata, float* Pdata, const int num_classes) {
  // Loop through labels (N x A x H x W)
  CUDA_1D_KERNEL_LOOP(index, num * A * H * W) {
    int D = num_classes * A;
    int x = index % W;
    int y = (index / W) % H;
    int a = (index / (W * H)) % A;
    int i = index / W / H / A;

    // Subtract max on each cell for numerical reasons
    float max_val = -FLT_MAX;
    for(int c = a * num_classes; c < (a + 1) * num_classes; ++c) {
      int idx = i * (H * W * D) +  c * (H * W) + y * W + x;
      max_val = max(max_val, Xdata[idx]);
    }
    // Exponentiate
    float expsum = 0.0f;
    for(int c = a * num_classes; c < (a + 1) * num_classes; ++c) {
      int idx = i * (H * W * D) + c * (H * W) + y * W + x;
      float expx = exp(Xdata[idx] - max_val);
      Pdata[idx] = expx;
      expsum += expx;
    }

    // Normalize
    for(int c = a * num_classes; c < (a + 1) * num_classes; ++c) {
      int idx = i * (H * W * D) + c * (H * W) + y * W + x;
      Pdata[idx] /= expsum;
    }

  }
}

__global__ void SumProbsKernel(const int N, const int A, const int W,
    const int H, const float* Ydata, const float* dYdata,
    float* sum_probs_data, const int num_classes) {
  CUDA_1D_KERNEL_LOOP(i, N * A * W * H) {
    int D = num_classes * A;
    int x = i % W;
    int y = (i / W) % H;
    int a = (i / (W * H)) % A;
    int n = i / (W * H * A);

    sum_probs_data[i] = 0.0;
    for(int c = a * num_classes; c < (a + 1) * num_classes; ++c) {
      int idx = n * (H * W * D) + c * (H * W) + y * W + x;
      sum_probs_data[i] += (Ydata[idx] * dYdata[idx]);
    }
  }
}

__global__ void SubSumKernel(
    const int N, const int A, const int W, const int H,
    const float* sum_probs_data, float* dXdata, const int num_classes) {
  CUDA_1D_KERNEL_LOOP(i, N * (A * num_classes) * W * H) {
    int D = num_classes * A;
    int x = i % W;
    int y = (i / W) % H;
    int a = ((i / (W * H)) % D) / num_classes;
    int n = i / W / H / D;
    int idx = n * (H * W * A) + a * (H * W) + y * W + x;
    dXdata[i] = (dXdata[i] - sum_probs_data[idx]);
  }
}

} // namespace


template <>
bool GroupSpatialSoftmaxOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);  // Logits

  int N = X.dim32(0);
  int D = X.dim32(1);
  int H = X.dim32(2);
  int W = X.dim32(3);
  int A = D / num_classes_;

  auto* P = Output(0, X.sizes(), at::dtype<float>()); // Probabilities from softmax
  TORCH_DCHECK_EQ(X.ndim(), 4);

  const float* Xdata = X.data<float>();
  float* Pdata = P->mutable_data<float>();

  // Softmax for each x,y location
  GroupSpatialSoftmaxKernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS,
                         0, context_.cuda_stream()>>>(
      N, A, W, H, Xdata, Pdata, num_classes_);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return true;
}


template<>
bool GroupSpatialSoftmaxGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& Y = Input(0);  // Probabilities from softmax
  auto& dY = Input(1);


  TORCH_DCHECK_EQ(Y.ndim(), 4);

  int N = Y.dim32(0);
  int D = Y.dim32(1);
  int H = Y.dim32(2);
  int W = Y.dim32(3);
  int A = D / num_classes_;

  auto* dX = Output(0, Y.sizes(), at::dtype<float>());

  if (sum_probs_.size() != N * A * H * W) {
    ReinitializeTensor(&sum_probs_, {N * A * H * W}, at::dtype<float>().device(CUDA));
  }

  const float* Ydata = Y.data<float>();
  const float* dYdata = dY.data<float>();
  float* dXdata = dX->mutable_data<float>();

  float* sum_probs_data = sum_probs_.mutable_data<float>();
  math::Set<float, CUDAContext>(
      sum_probs_.size(), 0.0f, sum_probs_data, &context_);

  // Complete math:
  // J_ij = h_i (delta_ij - h_j)
  // d x_i = sum_j d h_ij = sum_j J_ij * dy_j
  //       = sum_j h_i (delta_ij - h_j) * dy_j
  //       = h_i dy_i - (sum_j h_i h_j dy_j)
  //       = h_i dy_i - h_i sum_j h_j dy_j

  // Step 0: dx = dy
  context_.Copy<float, CUDAContext, CUDAContext>(Y.size(), dYdata, dXdata);

  // Step 1: s = Sum(dY[j] * Y[j])
  SumProbsKernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0,
                   context_.cuda_stream()>>>(
    N, A, W, H, Ydata, dYdata, sum_probs_data, num_classes_);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // Step 2: dX[i] = dX[i] - s
  SubSumKernel<<<CAFFE_GET_BLOCKS(Y.size()), CAFFE_CUDA_NUM_THREADS, 0,
                  context_.cuda_stream()>>>(
    N, A, W, H, sum_probs_.data<float>(), dXdata, num_classes_);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // Step 3: dX[i] = Y[i] * dX[i]
  math::Mul<float, CUDAContext>(Y.size(), dXdata, Ydata, dXdata, &context_);

  return true;
}


REGISTER_CUDA_OPERATOR(GroupSpatialSoftmax,
                       GroupSpatialSoftmaxOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(GroupSpatialSoftmaxGradient,
                       GroupSpatialSoftmaxGradientOp<float, CUDAContext>);
} // namespace caffe2
