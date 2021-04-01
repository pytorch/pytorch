#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/cosine_embedding_criterion_op.h"

namespace caffe2 {
namespace {


__global__ void CECKernel(
    const int N, const float* S, const int* Y, const float margin,
    float* output) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    output[i] = Y[i] == 1 ? (1. - S[i]) : fmaxf(0.f, S[i] - margin);
  }
}

__global__ void CECGradientKernel(
    const int N, const float* S, const int* Y, const float* dOutput,
    const float margin, float* dS) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    dS[i] = dOutput[i] * (Y[i] == 1 ? -1 : static_cast<float>(S[i] >= margin));
  }
}
}  // namespace

template <>
bool CosineEmbeddingCriterionOp<CUDAContext>::RunOnDevice() {
  auto& S = Input(0);
  auto& Y = Input(1);

  CAFFE_ENFORCE(S.numel() == Y.numel(),
                "The embedding and label should have the same size.");
  auto* output = Output(0, S.sizes(), at::dtype<float>());

  const float* Sdata = S.data<float>();
  const int* Ydata = Y.data<int>();
  float* output_data = output->template mutable_data<float>();

  CECKernel<<<CAFFE_GET_BLOCKS(S.numel()), CAFFE_CUDA_NUM_THREADS,
              0, context_.cuda_stream()>>>(
      S.numel(), Sdata, Ydata, margin_, output_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

template <>
bool CosineEmbeddingCriterionGradientOp<CUDAContext>::RunOnDevice() {
  auto& S = Input(0);
  auto& Y = Input(1);
  auto& dOutput = Input(2);


  auto* dS = Output(0, S.sizes(), at::dtype<float>());

  const float* Sdata = S.data<float>();
  const int* Ydata = Y.data<int>();
  const float* dOutput_data = dOutput.data<float>();
  float* dSdata = dS->template mutable_data<float>();
  CECGradientKernel<<<CAFFE_GET_BLOCKS(S.numel()), CAFFE_CUDA_NUM_THREADS,
                      0, context_.cuda_stream()>>>(
      S.numel(), Sdata, Ydata, dOutput_data, margin_, dSdata);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

REGISTER_CUDA_OPERATOR(
    CosineEmbeddingCriterion,
    CosineEmbeddingCriterionOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(
    CosineEmbeddingCriterionGradient,
    CosineEmbeddingCriterionGradientOp<CUDAContext>);
}  // namespace caffe2
