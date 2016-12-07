#include "momentum_sgd_op.h"
#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"

namespace caffe2 {

__global__ void MomentumSGDKernel(
    int N,
    const float* g,
    const float* m,
    float* ng,
    float* nm,
    const float* lr,
    float momentum,
    bool nesterov,
    float* param) {
  const float LR = lr[0];
  if (!nesterov) {
    CUDA_1D_KERNEL_LOOP(i, N) {
      const float adjusted_gradient =  LR * g[i] + momentum * m[i];
      nm[i] = adjusted_gradient;
      ng[i] = adjusted_gradient;
      if (param) {
        param[i] -= adjusted_gradient;
      }
    }
  } else {
    CUDA_1D_KERNEL_LOOP(i, N) {
      const float mi = m[i];
      const float mi_new = momentum * mi + LR * g[i];
      nm[i] = mi_new;
      ng[i] = (1 + momentum) * mi_new - momentum * mi;
      if (param) {
        param[i] -= ng[i];
      }
    }
  }
}

template<>
void momentum_sgd_update<CUDAContext>(
    int N,
    const float* g,
    const float* m,
    float* ng,
    float* nm,
    const float* lr,
    float momentum,
    bool nesterov,
    float *param,
    CUDAContext* context) {
  MomentumSGDKernel<<<
      CAFFE_GET_BLOCKS(N),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context->cuda_stream()>>>(
      N, g, m, ng, nm, lr, momentum, nesterov, param);
}

namespace {
REGISTER_CUDA_OPERATOR(MomentumSGD, MomentumSGDOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(MomentumSGDUpdate, MomentumSGDUpdateOp<float, CUDAContext>);

}

}
