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

#include "caffe2/core/context_gpu.h"
#include "modules/detectron/select_smooth_l1_loss_op.h"

namespace caffe2 {

namespace {
__global__ void SelectSmoothL1Kernel(
    const int D, const int H, const int W,
    const int M, const float* Y_hat, const float* Y, const float* L, float* out,
    const float* S, const float beta) {
  // f(x) = 0.5 * x^2 / beta      if |x| < beta
  //        |x| - 0.5 * beta      otherwise
  CUDA_1D_KERNEL_LOOP(i, M) {
    int n = L[i * 4];
    int c = L[i * 4 + 1];
    int y = L[i * 4 + 2];
    int x = L[i * 4 + 3];

    for (int j = 0; j < 4; j++){
      // Y_hat: N x (A * CLS * 4) x H x W
      int ind = n * (D * H * W) + (c + j) * (H * W) + y * W + x;
      float y_hat = Y_hat[ind];
      float y = Y[i * 4 + j];
      float val = y_hat - y;
      float abs_val = c10::cuda::compat::abs(val);
      if (abs_val < beta) {
        out[ind] = (0.5 * val * val / beta) / c10::cuda::compat::max(S[0], static_cast<float>(1.0));
      } else {
        out[ind] = (abs_val - 0.5 * beta) / c10::cuda::compat::max(S[0], static_cast<float>(1.0));
      }
    }
  }
}


__global__ void SelectSmoothL1GradientKernel(
    const int D, const int H, const int W,
    const int M,
    const float* Y_hat,
    const float* Y,
    const float* L,
    float* out,
    const float* d_loss_data,
    float norm,
    const float* S,
    float beta) {
  // f'(x) = x / beta     if |x| < beta
  //       = sign(x)      otherwise
  // We also scale by norm * d_loss in this kernel for convenience
  CUDA_1D_KERNEL_LOOP(i, M) {
    int n = L[i * 4];
    int c = L[i * 4 + 1];
    int y = L[i * 4 + 2];
    int x = L[i * 4 + 3];
    float d_loss = *d_loss_data;

    for (int j = 0; j < 4; j++) {
      int ind = n * (D * H * W) + (c + j) * (H * W) + y * W + x;
      float y_hat = Y_hat[ind];
      float y = Y[i * 4 + j];
      float val = y_hat - y;
      float abs_val = c10::cuda::compat::abs(val);
      if (abs_val < beta) {
        out[ind] = norm * d_loss * val / beta / c10::cuda::compat::max(S[0], static_cast<float>(1.0));
      } else {
        out[ind] = norm * d_loss * ((float(0) < val) - (val < float(0))) / c10::cuda::compat::max(S[0], static_cast<float>(1.0));
      }
    }
  }
}
} // namespace


template<>
bool SelectSmoothL1LossOp<float, CUDAContext>::RunOnDevice() {
  // bbox targets predictions, for example: N x (A * 4) H x W in cls-agnostic case
  auto& Y_hat     = Input(0);
  // true targets: for example: M x 4 where M is the #fg boxes per fpn level
  auto& Y         = Input(1);
  // locations of fg boxes: M x 4
  auto& L         = Input(2);
  // total number of fg boxes across all FPN levels: scalar
  auto& S         = Input(3);


  auto* avg_loss = Output(0, vector<int64_t>(), at::dtype<float>());
  if (Y.size() == 0){
    math::Set<float, CUDAContext>(
      1, static_cast<float>(0), avg_loss->mutable_data<float>(), &context_);
    return true;
  }

  int N = Y_hat.dim32(0);
  int D = Y_hat.dim32(1);
  int H = Y_hat.dim32(2);
  int W = Y_hat.dim32(3);

  int M = Y.dim32(0);

  // initialization
  buff_.ResizeLike(Y_hat);
  math::Set<float, CUDAContext>(
    1, static_cast<float>(0), avg_loss->mutable_data<float>(), &context_);
  math::Set<float, CUDAContext>(
    buff_.size(), 0.0, buff_.mutable_data<float>(), &context_);

  // Element-wise smooth l1 loss
  // l := SelectSmoothL1((y_hat - y))
  SelectSmoothL1Kernel<<<CAFFE_GET_BLOCKS(buff_.size()),
                         CAFFE_CUDA_NUM_THREADS,
                         0, context_.cuda_stream()>>>(
    D, H, W,
    M, Y_hat.data<float>(), Y.data<float>(),
    L.data<float>(), buff_.mutable_data<float>(),
    S.data<float>(), beta_);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // Sum of all losses
  // al := sum_i l_i
  float* avg_loss_data = avg_loss->mutable_data<float>();
  math::Sum<float, CUDAContext>(
      buff_.size(), buff_.data<float>(), avg_loss_data, &context_);

  // Average of input batch size
  math::Scale<float, float, CUDAContext>(
      1, scale_, avg_loss_data, avg_loss_data, &context_);
  return true;
}

template<>
bool SelectSmoothL1LossGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& Y_hat      = Input(0);
  auto& Y          = Input(1);
  auto& L          = Input(2);
  auto& S          = Input(3);
  // Below is gradient of net w.r.t. avg_loss ("gradOuput"), should be all 1's
  auto& d_avg_loss = Input(4);

  auto* d_Y_hat = Output(0, Y_hat.sizes(), at::dtype<float>()); // gradient of net w.r.t. Y_hat ("gradInput")
  math::Set<float, CUDAContext>(
    d_Y_hat->size(), 0.0, d_Y_hat->mutable_data<float>(), &context_);
  if (Y.size() == 0){
    return true;
  }

  int N = Y_hat.dim32(0);
  int D = Y_hat.dim32(1);
  int H = Y_hat.dim32(2);
  int W = Y_hat.dim32(3);

  int M = Y.dim32(0);
  // Element-wise weighted difference (can be used to ignore or reweight
  // specific components)
  // d := (y_hat - y)
  // d_Y_hat := d_avg_loss * SelectSmoothL1'((y_hat - y))

  SelectSmoothL1GradientKernel<<<CAFFE_GET_BLOCKS(d_Y_hat->size()),
                                 CAFFE_CUDA_NUM_THREADS,
                                 0, context_.cuda_stream()>>>(
    D, H, W, M, Y_hat.data<float>(), Y.data<float>(),
    L.data<float>(), d_Y_hat->mutable_data<float>(),
    d_avg_loss.data<float>(), scale_, S.data<float>(), beta_);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}


REGISTER_CUDA_OPERATOR(SelectSmoothL1Loss,
                       SelectSmoothL1LossOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(SelectSmoothL1LossGradient,
                       SelectSmoothL1LossGradientOp<float, CUDAContext>);
}  // namespace caffe2
