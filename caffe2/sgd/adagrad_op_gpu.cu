#include "adagrad_op.h"
#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"

namespace caffe2 {

__global__ void AdagradUpdate(
    int N,
    const float* w,
    const float* g,
    const float* h,
    float* nw,
    float* nh,
    float epsilon,
    const float* lr) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    float gi = g[i];
    float hi = nh[i] = h[i] + gi * gi;
    nw[i] = w[i] + lr[0] * gi / (std::sqrt(hi) + epsilon);
  }
}

template <>
void adagrad_update<CUDAContext>(
    int N,
    const float* w,
    const float* g,
    const float* h,
    float* nw,
    float* nh,
    float epsilon,
    const float* lr,
    CUDAContext* context) {
  AdagradUpdate<<<
      CAFFE_GET_BLOCKS(N),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context->cuda_stream()>>>(N, w, g, h, nw, nh, epsilon, lr);
}

template <typename SIndex>
__global__ void SparseAdagradKernel(
    const size_t N,
    const size_t grad_slice_sz,
    const float epsilon,
    float *param,
    float *param_mom,
    const SIndex *indices,
    const float *grad,
    const float *lr)
{
  const float LR = lr[0];
  CUDA_1D_KERNEL_LOOP(i, N)
  {
    const size_t gradIdx = i;
    const SIndex index = indices[i / grad_slice_sz];
    const size_t paramIdx = index * grad_slice_sz + (i % grad_slice_sz);

    const float mom_new = param_mom[paramIdx] + grad[gradIdx] * grad[gradIdx];
    param_mom[paramIdx] = mom_new;
    param[paramIdx] += LR * grad[gradIdx] / (sqrt(mom_new) + epsilon);
  }
}

template<>
template<typename SIndex>
bool SparseAdagradOp<float, CUDAContext>::DoRunWithType()
{
  auto N = Input(GRAD).size();
  auto grad_slice_sz = Input(GRAD).size_from_dim(Input(INDICES).ndim());

  if (N == 0) {
    // empty grad, nothing to do here, not even launching the kernel
    return true;
  }

  SparseAdagradKernel<SIndex><<<
    CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0,
    context_.cuda_stream()>>>(
        N, grad_slice_sz, epsilon_,
        Output(OUTPUT_PARAM)->template mutable_data<float>(),
        Output(OUTPUT_MOMENT_1)->template mutable_data<float>(),
        Input(INDICES).template data<SIndex>(),
        Input(GRAD).template data<float>(),
        Input(LR).template data<float>());
  return true;
}

REGISTER_CUDA_OPERATOR(Adagrad, AdagradOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(SparseAdagrad, SparseAdagradOp<float, CUDAContext>);
}
