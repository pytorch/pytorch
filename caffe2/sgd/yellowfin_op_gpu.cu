// YellowFin: An automatic tuner for momentum SGD
// (https://arxiv.org/abs/1706.03471)

#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/sgd/yellowfin_op.h"

namespace caffe2 {

__global__ void GetLrMuKernel(
    const float* g_norm2_max_deb,
    const float* g_norm2_min_deb,
    const float* distance_deb,
    const float* variance,
    float* mu,
    float* lr) {
  const float curv_ratio = sqrtf(*g_norm2_max_deb / *g_norm2_min_deb);
  const float mu_limit = (curv_ratio - 1.0f) / (curv_ratio + 1.0f);
  const float pre_p = *distance_deb * *g_norm2_min_deb;
  const float p = (pre_p * pre_p) / (2.0f * *variance);
  const float w3 = (-sqrtf(p * p + 4.0f / 27.0f * p * p * p) - p) / 2.0f;
  const float w3_sign = w3 > 0.0f ? 1.0f : -1.0f;
  const float w = w3_sign * powf(fabsf(w3), 1.0f / 3.0f);
  const float y = w - p / 3.0f / w;
  const float root = y + 1.0f;
  *mu = fmaxf(root * root, mu_limit * mu_limit);
  *lr = powf(1.0f - sqrtf(*mu), 2) / *g_norm2_min_deb;
}

template <>
void YellowFinOp<float, CUDAContext>::GetLrMu() {
  // Finding root of cubic formula for YF's Single Step
  GetLrMuKernel<<<1, 1, 0, context_.cuda_stream()>>>(
      g_norm2_max_deb_, g_norm2_min_deb_, distance_deb_, variance_, mu_, lr_);
  MovingAverage(1, mu_, mu_avg_, mu_avg_out_, mu_deb_);
  MovingAverage(1, lr_, lr_avg_, lr_avg_out_, lr_deb_);
}

__global__ void MomentumSgdKernel(
    const int N,
    const float* mu_ptr,
    const float* lr_ptr,
    const float* param,
    const float* grad,
    const float* moment,
    float* param_out,
    float* moment_out,
    bool nesterov) {
  const float mu = *mu_ptr;
  const float lr = *lr_ptr;
  if (!nesterov) {
    CUDA_1D_KERNEL_LOOP(i, N) {
      moment_out[i] = mu * moment[i] * lr * grad[i];
      param_out[i] = param[i] - moment_out[i];
    }
  } else {
    CUDA_1D_KERNEL_LOOP(i, N) {
      const float moment_i = moment[i];
      moment_out[i] = mu * moment_i + lr * grad[i];
      param_out[i] = param[i] - (1 + mu) * moment_out[i] + mu * moment_i;
    }
  }
}

template <>
void YellowFinOp<float, CUDAContext>::MomentumSgdUpdate() {
  MomentumSgdKernel<<<
      CAFFE_GET_BLOCKS(D_),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      D_,
      mu_avg_out_,
      lr_avg_out_,
      param_,
      grad_,
      moment_,
      param_out_,
      moment_out_,
      nesterov_);
}

REGISTER_CUDA_OPERATOR(YellowFin, YellowFinOp<float, CUDAContext>);

} // namespace caffe2
