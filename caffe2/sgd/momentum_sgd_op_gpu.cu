#include "caffe2/sgd/momentum_sgd_op.h"
#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"

namespace caffe2 {

inline int CaffeGetBlocksSGD(const int N) {
  return std::max(
      (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS,
      // Use at least 1 block, since CUDA does not allow empty block
      1);
}
template <bool nesterov>
__global__ void MomentumSGDKernel(
    const int N,
    const float* g,
    const float* m,
    float* ng,
    float* nm,
    const float* lr,
    const float momentum,
    float* param);

template <>
__global__ void MomentumSGDKernel<true>(
    const int N,
    const float* g,
    const float* m,
    float* ng,
    float* nm,
    const float* lr,
    const float momentum,
    float* param) {
  const float LR = lr[0];
  CUDA_1D_KERNEL_LOOP(i, N) {
    const float mi = m[i];
    const float mi_new = momentum * mi + LR * g[i];
    nm[i] = mi_new;
    ng[i] = fmaf(momentum, mi_new - mi, mi_new);
    if (param != nullptr) {
      param[i] -= ng[i];
    }
  }
}

template <>
__global__ void MomentumSGDKernel<false>(
    const int N,
    const float* g,
    const float* m,
    float* ng,
    float* nm,
    const float* lr,
    const float momentum,
    float* param) {
  const float LR = lr[0];
  CUDA_1D_KERNEL_LOOP(i, N) {
    const float adjusted_gradient = LR * g[i] + momentum * m[i];
    nm[i] = adjusted_gradient;
    ng[i] = adjusted_gradient;
    if (param != nullptr) {
      param[i] -= adjusted_gradient;
    }
  }
}

template <>
void momentum_sgd_update<CUDAContext>(
    const int N,
    const float* g,
    const float* m,
    float* ng,
    float* nm,
    const float* lr,
    const float momentum,
    const bool nesterov,
    float* param,
    CUDAContext* context) {
  if (nesterov) {
    MomentumSGDKernel<true>
        <<<CaffeGetBlocksSGD(N),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context->cuda_stream()>>>(N, g, m, ng, nm, lr, momentum, param);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    MomentumSGDKernel<false>
        <<<CaffeGetBlocksSGD(N),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context->cuda_stream()>>>(N, g, m, ng, nm, lr, momentum, param);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}


template <typename SIndex>
__global__ void SparseMomentumSGDKernel(
    const size_t N,
    const size_t sz,
    const float momentum,
    const bool nesterov,
    float *param,
    float *param_mom,
    const SIndex *indices,
    const float *gradIn,
    float *gradOut,
    const float *lr)
{
  const float LR = lr[0];
  CUDA_1D_KERNEL_LOOP(i, N)
  {
    const size_t gradIdx = i;
    const SIndex index = indices[i / sz];
    const size_t paramIdx = index * sz + (i % sz);

    if (!nesterov)
    {
      const float adjusted_gradient = LR * gradIn[gradIdx] +
          momentum * param_mom[paramIdx];
      gradOut[gradIdx] = adjusted_gradient;
      param_mom[paramIdx] = adjusted_gradient;
      param[paramIdx] -= adjusted_gradient;
    } else {
      const float mom_old = param_mom[paramIdx];
      const float mom_new = LR * gradIn[gradIdx] + momentum * mom_old;
      param_mom[paramIdx] = mom_new;
      const float adjusted_gradient = (1 + momentum) * mom_new -
          momentum * mom_old;
      gradOut[gradIdx] = adjusted_gradient;
      param[paramIdx] -= adjusted_gradient;
    }
  }
}


// Specialization of DoRunWithType for CUDA
template <>
template <typename SIndex>
bool SparseMomentumSGDUpdateOp<float, CUDAContext>::DoRunWithType() {
  auto N = Input(GRAD).size();
  auto grad_slice_sz = Input(GRAD).size_from_dim(Input(INDICES).ndim());

  SparseMomentumSGDKernel<SIndex><<<
    CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0,
    context_.cuda_stream()>>>(
        N, grad_slice_sz,
        momentum_, nesterov_,
        Output(OUTPUT_PARAM)->template mutable_data<float>(),
        Output(OUTPUT_MOMENTUM)->template mutable_data<float>(),
        Input(INDICES).template data<SIndex>(),
        Input(GRAD).template data<float>(),
        Output(OUTPUT_GRAD)->template mutable_data<float>(),
        Input(LR).template data<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return true;
}

REGISTER_CUDA_OPERATOR(MomentumSGD, MomentumSGDOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(MomentumSGDUpdate, MomentumSGDUpdateOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(SparseMomentumSGDUpdate, SparseMomentumSGDUpdateOp<float, CUDAContext>);

}
