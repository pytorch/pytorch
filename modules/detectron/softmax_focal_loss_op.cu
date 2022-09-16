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
#include "modules/detectron/softmax_focal_loss_op.h"

namespace caffe2 {

namespace {

__global__ void SpatialSoftmaxKernel(const int N, const int A,
    const int H, const int W, const float* Xdata, float* Pdata,
    const int num_classes) {
  CUDA_1D_KERNEL_LOOP(index, N * A * H * W) {
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


__global__ void SoftmaxFocalLossKernel(
    const int N, const int A, const int H, const int W,
    const float* Pdata, const int* targets, float* losses,
    const float* weight_pos, const float gamma, const float alpha,
    const int num_classes) {
  CUDA_1D_KERNEL_LOOP(i, N * A * H * W) {
    int D = A * num_classes;
    int x = i % W;
    int y = (i / W) % H;
    int a = (i / (W * H)) % A;
    int n = i / (W * H * A);
    const int label = static_cast<int>(targets[i]);

    float Np = c10::cuda::compat::max(weight_pos[0], static_cast<float>(1.0));
    float z = (label == 0) * (1 - alpha) / Np +
              (label >= 1) * alpha / Np;

    losses[i] = 0.0;
    if (label >= 0) {
      int offset = a * num_classes;
      int idx = n * (H * W * D) + (offset + label) * (H * W) + y * W + x;
      losses[i] =
          -(pow(1.0f - Pdata[idx], gamma) *
          log(c10::cuda::compat::max(Pdata[idx], FLT_MIN))) * z;
    }
  }
}


__global__ void SoftmaxFocalLossGradientWeightKernel(
    const int N, const int A, const int H, const int W,
    const float* Pdata, const int* targets, float* buff,
    const float* weight_pos, const float gamma, const float alpha,
    const int num_classes) {
  CUDA_1D_KERNEL_LOOP(i, N * A * H * W) {
    int D = A * num_classes;
    int x = i % W;
    int y = (i / W) % H;
    int a = (i / (W * H)) % A;
    int n = i / (W * H * A);
    const int label = static_cast<int>(targets[i]);
    float Np = c10::cuda::compat::max(weight_pos[0], static_cast<float>(1.0));
    float z =  (label == 0) * (1 - alpha) / Np +
               (label >= 1) * alpha / Np;

    buff[i] = 0.0;
    if (label >= 0) {
      int offset = a * num_classes;
      int idx = n * (H * W * D) + (offset + label) * (H * W) + y * W + x;
      float onemp = 1. - Pdata[idx];
      float p = Pdata[idx];
      buff[i] =
          (-pow(onemp, gamma) +
          gamma * pow(onemp, gamma - 1) * p * log(c10::cuda::compat::max(p, FLT_MIN))) * z;
    }
  }
}


__global__ void SoftmaxFocalLossGradientKernel(
    const int N, const int D, const int H, const int W,
    const float* Pdata, const int* targets, const float* buff,
    const float* d_loss_data, float* dX, const int num_classes) {
  CUDA_1D_KERNEL_LOOP(i, N * D * H * W) {
    int A = D / num_classes;
    int x = i % W;
    int y = (i / W) % H;
    int d = (i / (W * H)) % D;
    int a = d / num_classes;
    int c = d % num_classes;
    int n = i / (W * H * D);
    float d_loss = *d_loss_data;

    int ind = n * (H * W * A) + a * (H * W) + y * W + x;
    const int label = static_cast<int>(targets[ind]);

    float c1 = (label >= 0) * 1.0;
    float c2 = (label == c) * 1.0;
    dX[i] = 0.0;
    dX[i] = c1 * d_loss * buff[ind] * (c2 - Pdata[i]);
  }
}

} // namespace


template <>
bool SoftmaxFocalLossOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);         // Logits
  auto& T = Input(1);         // Labels
  auto& wp = Input(2);        // num of foreground
   // average loss as output
          // softmax probability, going to be re-used in gradient

  int N = X.dim32(0);
  int D = X.dim32(1);
  int H = X.dim32(2);
  int W = X.dim32(3);
  int A = D / num_classes_;

  ReinitializeTensor(&losses_, {N * A * H * W}, at::dtype<float>().device(CUDA));
  auto* P = Output(1, {N * D * H * W}, at::dtype<float>());
  auto* avg_loss = Output(0, vector<int64_t>(), at::dtype<float>());
  math::Set<float, CUDAContext>(
      avg_loss->size(), 0.f, avg_loss->mutable_data<float>(), &context_);
  math::Set<float, CUDAContext>(
      P->size(), 0.f, P->mutable_data<float>(), &context_);
  math::Set<float, CUDAContext>(
      losses_.size(), 0.f, losses_.mutable_data<float>(), &context_);
  TORCH_DCHECK_EQ(X.ndim(), 4);

  const float* Xdata = X.data<float>();
  const float* Wdata = wp.data<float>();


  // Spatial Softmax Kernel
  SpatialSoftmaxKernel
      <<<CAFFE_GET_BLOCKS(N * A * H * W), CAFFE_CUDA_NUM_THREADS,
         0, context_.cuda_stream()>>>(
    N, A, H, W, Xdata, P->mutable_data<float>(), num_classes_);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // Compute loss for each x,y location
  const int* Tdata = T.data<int>();
  SoftmaxFocalLossKernel
  <<<CAFFE_GET_BLOCKS(N * A * H * W), CAFFE_CUDA_NUM_THREADS,
      0, context_.cuda_stream()>>>(
    N, A, H, W, P->data<float>(), Tdata, losses_.mutable_data<float>(),
    Wdata, gamma_, alpha_, num_classes_);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // sum the losses
  float* avg_loss_data = avg_loss->mutable_data<float>();
  math::Sum<float, CUDAContext>(
      losses_.size(), losses_.data<float>(), avg_loss_data, &context_);
  math::Scale<float, float, CUDAContext>(
      1, scale_, avg_loss_data, avg_loss_data, &context_);

  return true;
}


template<>
bool SoftmaxFocalLossGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);    // Logits
  auto& T = Input(1);    // Label
  auto& wp = Input(2);   // num of foreground example
  auto& P = Input(3);    // Softmax Probability
  auto& d_avg_loss = Input(4);


  int N = X.dim32(0);
  int D = X.dim32(1);
  int H = X.dim32(2);
  int W = X.dim32(3);
  int A = D / num_classes_;

  ReinitializeTensor(&buff_, {N * A * H * W}, at::dtype<float>().device(CUDA));

  auto* dX = Output(0, X.sizes(), at::dtype<float>()); // gradient wrt logits

  const float* Xdata = X.data<float>();
  const int* Tdata = T.data<int>();
  const float* Pdata = P.data<float>();
  const float* Wdata = wp.data<float>();


  // Compute the weight for gradients
  SoftmaxFocalLossGradientWeightKernel
      <<<CAFFE_GET_BLOCKS(N * A * H * W), CAFFE_CUDA_NUM_THREADS,
         0, context_.cuda_stream()>>>(
    N, A, H, W, Pdata, Tdata, buff_.mutable_data<float>(),
    Wdata, gamma_, alpha_, num_classes_);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // Compute the gradient with the weights
  const float* Bdata = buff_.data<float>();
  SoftmaxFocalLossGradientKernel
      <<<CAFFE_GET_BLOCKS(N * D * H * W), CAFFE_CUDA_NUM_THREADS,
         0, context_.cuda_stream()>>>(
    N, D, H, W, Pdata, Tdata, Bdata, d_avg_loss.data<float>(),
    dX->mutable_data<float>(), num_classes_);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  math::Scale<float, float, CUDAContext>(
      dX->size(),
      scale_,
      dX->data<float>(),
      dX->mutable_data<float>(),
      &context_);
  return true;
}


REGISTER_CUDA_OPERATOR(SoftmaxFocalLoss,
                       SoftmaxFocalLossOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(SoftmaxFocalLossGradient,
                       SoftmaxFocalLossGradientOp<float, CUDAContext>);
} // namespace caffe2
