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
#include "smooth_l1_loss_op.h"

namespace caffe2 {

namespace {
template <typename T>
__global__ void SmoothL1Kernel(
    const int n, const T* in, T* out, T beta) {
  // f(x) = 0.5 * x^2 / beta      if |x| < beta
  //        |x| - 0.5 * beta      otherwise
  CUDA_1D_KERNEL_LOOP(index, n) {
    T val = in[index];
    T abs_val = abs(val);
    if (abs_val < beta) {
      out[index] = 0.5 * val * val / beta;
    } else {
      out[index] = abs_val - 0.5 * beta;
    }
  }
}

template <typename T>
__global__ void SmoothL1GradientKernel(
    const int n,
    const T* in,
    T* out,
    const T* d_loss_data,
    T norm,
    T beta) {
  // f'(x) = x / beta     if |x| < beta
  //       = sign(x)      otherwise
  // We also scale by norm * d_loss in this kernel for convenience
  CUDA_1D_KERNEL_LOOP(index, n) {
    T val = in[index];
    T abs_val = abs(val);
    T d_loss = *d_loss_data;
    if (abs_val < beta) {
      out[index] = norm * d_loss * val / beta;
    } else {
      out[index] = norm * d_loss * ((T(0) < val) - (val < T(0)));
    }
  }
}
} // namespace

template<>
bool SmoothL1LossOp<float, CUDAContext>::RunOnDevice() {
  auto& Y_hat     = Input(0);
  auto& Y         = Input(1);
  auto& alpha_in  = Input(2);
  auto& alpha_out = Input(3);
  auto* avg_loss  = Output(0);

  int N = Y.dim32(0);
  // Require the same number of elements along axis 0 (batch size), but
  // otherwise don't care about the shape (just the number of elements)
  CAFFE_ENFORCE_EQ(Y_hat.dim32(0), Y.dim32(0),
      "Y_hat and Y must have the same number of elements along axis 0");
  CAFFE_ENFORCE_EQ(Y_hat.size(), Y.size(),
      "Y_hat and Y must have the same number of elements");
  CAFFE_ENFORCE_EQ(Y_hat.size(), alpha_in.size());
  CAFFE_ENFORCE_EQ(Y_hat.size(), alpha_out.size());

  avg_loss->Resize(vector<TIndex>());
  buff_.ResizeLike(Y);

  // Difference
  // d := y_hat - y
  math::Sub<float, CUDAContext>(
      Y.size(), Y_hat.data<float>(), Y.data<float>(),
      buff_.mutable_data<float>(), &context_);
  // Element-wise weighted difference (can be used to ignore or reweight
  // specific components)
  // d := alpha_in * (y_hat - y)
  math::Mul<float, CUDAContext>(
      buff_.size(), buff_.data<float>(), alpha_in.data<float>(),
      buff_.mutable_data<float>(), &context_);

  // Element-wise smooth l1 loss
  // l := SmoothL1(alpha_in * (y_hat - y))
  SmoothL1Kernel<float>
  <<<CAFFE_GET_BLOCKS(buff_.size()),
     CAFFE_CUDA_NUM_THREADS,
     0,
     context_.cuda_stream()>>>(
          buff_.size(), buff_.data<float>(), buff_.mutable_data<float>(),
          beta_);

  // Element-wise weighted smooth l1 loss (can be used to specify a per-element
  // loss weight)
  // l := alpha_out * SmoothL1(alpha_in * (y_hat - y))
  math::Mul<float, CUDAContext>(
      buff_.size(), buff_.data<float>(), alpha_out.data<float>(),
      buff_.mutable_data<float>(), &context_);
  // Sum of all losses
  // al := sum_i l_i
  float* avg_loss_data = avg_loss->mutable_data<float>();
  math::Sum<float, CUDAContext>(
      buff_.size(), buff_.data<float>(), avg_loss_data, &context_);
  // Average of input batch size
  // al := 1/N * al
  math::Scale<float, CUDAContext>(
      1, scale_ / N, avg_loss_data, avg_loss_data, &context_);
  return true;
}

template<>
bool SmoothL1LossGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& Y_hat      = Input(0);
  auto& Y          = Input(1);
  auto& alpha_in   = Input(2);
  auto& alpha_out  = Input(3);
  auto& d_avg_loss = Input(4);  // gradient of net w.r.t. avg_loss ("gradOuput")
  auto* d_Y_hat    = Output(0); // gradient of net w.r.t. Y_hat ("gradInput")
  // We intentially don't compute gradients for Y, alpha_{in,out} since they
  // are not needed (can change in the future if desired)

  int N = Y.dim32(0);
  // Require the same number of elements along axis 0 (batch size), but
  // otherwise don't care about the shape (just the number of elements)
  CAFFE_ENFORCE_EQ(Y_hat.dim32(0), Y.dim32(0),
      "Y_hat and Y must have the same number of elements along axis 0");
  CAFFE_ENFORCE_EQ(Y_hat.size(), Y.size(),
      "Y_hat and Y must have the same number of elements");
  CAFFE_ENFORCE_EQ(Y_hat.size(), alpha_in.size());
  CAFFE_ENFORCE_EQ(Y_hat.size(), alpha_out.size());
  CAFFE_ENFORCE_EQ(d_avg_loss.size(), 1);

  d_Y_hat->ResizeLike(Y_hat);
  buff_.ResizeLike(Y);

  // Difference
  // d := y_hat - y
  math::Sub<float, CUDAContext>(
      Y.size(), Y_hat.data<float>(), Y.data<float>(),
      buff_.mutable_data<float>(), &context_);
  // Element-wise weighted difference (can be used to ignore or reweight
  // specific components)
  // d := alpha_in * (y_hat - y)
  math::Mul<float, CUDAContext>(
      buff_.size(), buff_.data<float>(), alpha_in.data<float>(),
      buff_.mutable_data<float>(), &context_);
  // d_Y_hat := d_avg_loss / N * SmoothL1'(alpha_in * (y_hat - y))
  SmoothL1GradientKernel<float>
  <<<CAFFE_GET_BLOCKS(buff_.size()),
     CAFFE_CUDA_NUM_THREADS,
     0,
     context_.cuda_stream()>>>(
         buff_.size(), buff_.data<float>(), d_Y_hat->mutable_data<float>(),
         d_avg_loss.data<float>(), scale_ / N, beta_);
  // Element-wise scale by alpha_in and alpha_out
  math::Mul<float, CUDAContext>(
      d_Y_hat->size(), d_Y_hat->data<float>(), alpha_in.data<float>(),
      d_Y_hat->mutable_data<float>(), &context_);
  math::Mul<float, CUDAContext>(
      d_Y_hat->size(), d_Y_hat->data<float>(), alpha_out.data<float>(),
      d_Y_hat->mutable_data<float>(), &context_);
  return true;
}


REGISTER_CUDA_OPERATOR(SmoothL1Loss,
                       SmoothL1LossOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(SmoothL1LossGradient,
                       SmoothL1LossGradientOp<float, CUDAContext>);
}  // namespace caffe2
