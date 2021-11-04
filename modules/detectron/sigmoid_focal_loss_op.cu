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
#include "modules/detectron/sigmoid_focal_loss_op.h"

namespace caffe2 {

namespace {

__global__ void SigmoidFocalLossKernel(
    const int N, const int D, const int H, const int W, const float* logits,
    const int* targets, const float* weight_pos,
    const float gamma, const float alpha,
    const int num_classes, float* losses) {
  CUDA_1D_KERNEL_LOOP(i, N * D * H * W) {
    int x = i % W;
    int y = (i / W) % H;
    int c = (i / (W * H)) % D;  // channel, here D is channel dim in input NxDxHxW
    int n = i / (W * H * D);    // n in NxDxHxW

    int A = D / num_classes;   // num_anchors = A
    int a = c / num_classes;   // current anchor out of A anchors in D = A * num_cls
    int d = c % num_classes;   // current class
    int t = targets[n * (H * W * A) + a * (H * W) + y * W + x];   // target

    // check whether the class is true class or not.
    // The target classes are in range 1 - 81 and the d is in range 0-80
    // because we predict A*80 dim, so for comparison purpose, compare t and (d+1)
    float c1 = (t == (d + 1));
    float c2 = (t != -1 & t != (d + 1));

    float Np = c10::cuda::compat::max(weight_pos[0], static_cast<float>(1.0));
    float zn = (1.0 - alpha) / Np;
    float zp = alpha / Np;

    // p = 1. / 1. + expf(-x)
    float p = 1. / (1. + expf(-logits[i]));

    // (1 - p)**gamma * log(p) where
    float term1 = powf((1. - p), gamma) * logf(c10::cuda::compat::max(p, FLT_MIN));
    // p**gamma * log(1 - p)
    float term2 =
        powf(p, gamma) *
        (-1. * logits[i] * (logits[i] >= 0) -
         logf(1. + expf(logits[i] - 2. * logits[i] * (logits[i] >= 0))));

    losses[i] = 0.0;
    losses[i] += -c1 * term1 * zp;
    losses[i] += -c2 * term2 * zn;
  }
}

__global__ void SigmoidFocalLossGradientKernel(
    const int N, const int D, const int H, const int W, const float* logits,
    const int* targets, float* dX_data, const float* weight_pos,
    const float gamma, const float alpha, const int num_classes,
    const float* avg_loss) {
  CUDA_1D_KERNEL_LOOP(i, N * D * H * W) {
      float a_loss = avg_loss[0];
      int x = i % W;
      int y = (i / W) % H;
      int c = (i / (W * H)) % D;
      int n = i / (W * H * D);

      int A = D / num_classes;   // num_anchors
      int a = c / num_classes;   // current anchor
      int d = c % num_classes;   // current class

      float Np = c10::cuda::compat::max(weight_pos[0], static_cast<float>(1.0));
      float zn = (1.0 - alpha) / Np;
      float zp = alpha / Np;
      int t = targets[n * (H * W * A) + a * (H * W) + y * W + x];

      float c1 = (t == (d + 1));
      float c2 = (t != -1 & t != (d + 1));
      float p = 1. / (1. + expf(-logits[i]));

      // (1-p)**g * (1 - p - g*p*log(p))
      float term1 =
          powf((1. - p), gamma) *
          (1. - p - (p * gamma * logf(c10::cuda::compat::max(p, FLT_MIN))));
      // (p**g) * (g*(1-p)*log(1-p) - p)
      float term2 =
          powf(p, gamma) *
          ((-1. * logits[i] * (logits[i] >= 0) -
           logf(1. + expf(logits[i] - 2. * logits[i] * (logits[i] >= 0)))) *
           (1. - p) * gamma - p);
      dX_data[i] = 0.0;
      dX_data[i] += -c1 * zp * term1;
      dX_data[i] += -c2 * zn * term2;
      dX_data[i] = dX_data[i] * a_loss;
  }
}
} // namespace

template<>
bool SigmoidFocalLossOp<float, CUDAContext>::RunOnDevice() {
  // Input logits, for example: N x (A * 80) x H x W in cls-agnostic
  auto& X = Input(0);
  // Target, for example: N x A x H x W
  auto& T = Input(1);
  // Number of positive examples: scalar
  auto& wp = Input(2);
  // output avg Sigmoid focal loss as mentioned in RetinaNet paper


  int N = X.dim32(0);
  int D = X.dim32(1);
  int H = X.dim32(2);
  int W = X.dim32(3);

  auto* avg_loss = Output(0, vector<int64_t>(), at::dtype<float>());
  losses_.ResizeLike(X);
  float* avg_loss_data = avg_loss->mutable_data<float>();

  SigmoidFocalLossKernel<<<CAFFE_GET_BLOCKS(X.size()),
          CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
      N, D, H, W, X.data<float>(), T.data<int>(),
      wp.data<float>(), gamma_, alpha_, num_classes_,
      losses_.mutable_data<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  math::Sum<float, CUDAContext>(
      losses_.size(), losses_.data<float>(), avg_loss_data, &context_);
  math::Scale<float, float, CUDAContext>(
      1, scale_, avg_loss_data, avg_loss_data, &context_);

  return true;
}


template<>
bool SigmoidFocalLossGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& T = Input(1);
  auto& wp = Input(2);
  auto& d_avg_loss = Input(InputSize() - 1);


  // get input shape
  int N = X.dim32(0);
  int D = X.dim32(1);
  int H = X.dim32(2);
  int W = X.dim32(3);

  auto* dX = Output(0, X.sizes(), at::dtype<float>());

  SigmoidFocalLossGradientKernel<<<CAFFE_GET_BLOCKS(X.size()),
          CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
      N, D, H, W, X.data<float>(), T.data<int>(), dX->mutable_data<float>(),
      wp.data<float>(), gamma_, alpha_, num_classes_,
      d_avg_loss.data<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  math::Scale<float, float, CUDAContext>(
      dX->size(),
      scale_,
      dX->data<float>(),
      dX->mutable_data<float>(),
      &context_);

  return true;
}


REGISTER_CUDA_OPERATOR(SigmoidFocalLoss,
                       SigmoidFocalLossOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(SigmoidFocalLossGradient,
                       SigmoidFocalLossGradientOp<float, CUDAContext>);
} // namespace caffe2
