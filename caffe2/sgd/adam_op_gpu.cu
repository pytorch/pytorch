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
    float correction,
    const float* lr) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    float gi = g[i];
    float mi = nm[i] = m[i] * beta1 + gi * (1 - beta1);
    float vi = nv[i] = v[i] * beta2 + gi * gi * (1 - beta2);
    ng[i] = lr[0] * correction * mi / (std::sqrt(vi) + eps_hat);
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
    float correction,
    const float* lr,
    CUDAContext* context) {
  AdamUpdate<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(
      N, g, m, v, ng, nm, nv, beta1, beta2, eps_hat, correction, lr);
}

__global__ void AdamCompute(
    int N,
    const float* w,
    const float* g,
    const float* m,
    const float* v,
    float* nw,
    float* nm,
    float* nv,
    float beta1,
    float beta2,
    float eps_hat,
    float correction,
    const float* lr) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    float gi = g[i];
    float mi = nm[i] = m[i] * beta1 + gi * (1 - beta1);
    float vi = nv[i] = v[i] * beta2 + gi * gi * (1 - beta2);
    float ng = lr[0] * correction * mi / (std::sqrt(vi) + eps_hat);
    nw[i] = w[i] + ng;
  }
}

template<>
void adam_compute<CUDAContext>(
    int N,
    const float* w,
    const float* g,
    const float* m,
    const float* v,
    float* nw,
    float* nm,
    float* nv,
    float beta1,
    float beta2,
    float eps_hat,
    float correction,
    const float* lr,
    CUDAContext* context) {
  AdamCompute<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, context->cuda_stream()>>>(
      N, w, g, m, v, nw, nm, nv, beta1, beta2, eps_hat, correction, lr);
}

template <typename SIndex>
__global__ void SparseAdamKernel(
    const size_t N,
    const size_t grad_slice_sz,
    const float beta1,
    const float beta2,
    const float epsilon,
    float *param,
    float *mom1,
    float *mom2,
    const SIndex *indices,
    const float *grad,
    const float correction,
    const float *lr,
    const float iter)
{
  CUDA_1D_KERNEL_LOOP(i, N)
  {
    const size_t gradIdx = i;
    const SIndex index = indices[i / grad_slice_sz];
    const size_t paramIdx = index * grad_slice_sz + (i % grad_slice_sz);

    float m1n = mom1[paramIdx] = mom1[paramIdx] * beta1 +
        grad[gradIdx] * (1.0f - beta1);
    float m2n = mom2[paramIdx] = mom2[paramIdx] * beta2 +
        grad[gradIdx] * grad[gradIdx] * (1.0f - beta2);
    param[paramIdx] += lr[0] * correction * m1n / (sqrt(m2n) + epsilon);
  }
}

template<>
template<typename SIndex>
bool SparseAdamOp<float, CUDAContext>::DoRunWithType()
{
  auto N = Input(GRAD).size();
  auto grad_slice_sz = Input(GRAD).size_from_dim(Input(INDICES).ndim());
  const auto iter =
      OperatorBase::Input<TensorCPU>(ITER).template data<int64_t>()[0];
  const float correction = std::sqrt(1.0f - std::pow(beta2_, iter + 1)) /
      (1.0f - std::pow(beta1_, iter + 1));

  SparseAdamKernel<SIndex><<<
    CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0,
    context_.cuda_stream()>>>(
        N, grad_slice_sz,
        beta1_, beta2_, epsilon_,
        Output(OUTPUT_PARAM)->template mutable_data<float>(),
        Output(OUTPUT_MOMENT_1)->template mutable_data<float>(),
        Output(OUTPUT_MOMENT_2)->template mutable_data<float>(),
        Input(INDICES).template data<SIndex>(),
        Input(GRAD).template data<float>(),
        correction,
        Input(LR).template data<float>(),
        iter);
  return true;
}

REGISTER_CUDA_OPERATOR(Adam, AdamOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(SparseAdam, SparseAdamOp<float, CUDAContext>);

}
