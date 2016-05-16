#include "adam_op.h"
#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"

namespace caffe2 {

__global__ void AdamUpdate(
    int N,
    const float* g,
    const float* m,
    const float* v,
    float* ng,
    float* nm,
    float* nv,
    float beta1,
    float beta2,
    float eps_hat,
    float corrected_local_rate) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    float gi = g[i];
    float mi = nm[i] = m[i] * beta1 + gi * (1 - beta1);
    float vi = nv[i] = v[i] * beta2 + gi * gi * (1 - beta2);
    ng[i] = corrected_local_rate * mi / (sqrt(vi) + eps_hat);
  }
}

template<>
void adam_update<CUDAContext>(
    int N,
    const float* g,
    const float* m,
    const float* v,
    float* ng,
    float* nm,
    float* nv,
    float beta1,
    float beta2,
    float eps_hat,
    float corrected_local_rate,
    CUDAContext* context) {
  AdamUpdate<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(
      N, g, m, v, ng, nm, nv, beta1, beta2, eps_hat, corrected_local_rate);
}

namespace {
REGISTER_CUDA_OPERATOR(Adam, AdamOp<float, CUDAContext>);
}

}
