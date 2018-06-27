#include "hip/hip_runtime.h"
#include "caffe2/core/hip/context_hip.h"
#include "caffe2/operators/margin_ranking_criterion_op.h"

namespace caffe2 {
namespace {


__global__ void MRCKernel(
    const int N, const int* Y, const float* X1, const float* X2, const float margin,
    float* output) {
  HIP_1D_KERNEL_LOOP(i, N) {
    output[i] = max(0.f, -Y[i] * (X1[i] - X2[i]) + margin);
  }
}

__global__ void MRCGradientKernel(
    const int N, const int* Y, const float* X1, const float* X2, const float* dOutput,
    const float margin, float* dX1, float* dX2) {
  HIP_1D_KERNEL_LOOP(i, N) {
    float dist = -Y[i] * (X1[i] - X2[i]) + margin;
    if (dist < 0.f) {
      dX1[i] = dX2[i] = 0.f;
    } else {
      dX1[i] = -Y[i] * dOutput[i];
      dX2[i] = Y[i] * dOutput[i];
    }
  }
}
}  // namespace

template <>
bool MarginRankingCriterionOp<HIPContext>::RunOnDevice() {
  auto& X1 = Input(0);
  auto& X2 = Input(1);
  auto& Y = Input(2);
  auto* loss = Output(0);
  CAFFE_ENFORCE(
      X1.size() == X2.size(),
      "The two inputs for computing ranking loss should have the same size.");
  CAFFE_ENFORCE(
      X1.size() == Y.size(),
      "The input and label should have the same size.");
  loss->ResizeLike(X1);

  const float* X1data = X1.data<float>();
  const float* X2data = X2.data<float>();
  const int* Ydata = Y.data<int>();
  float* output_data = loss->mutable_data<float>();

  hipLaunchKernelGGL((MRCKernel), dim3(CAFFE_GET_BLOCKS(X1.size())), dim3(CAFFE_HIP_NUM_THREADS), 0, context_.hip_stream(), 
      static_cast<const int>(X1.size()), Ydata, X1data, X2data, margin_, output_data);
  return true;
}

template <>
bool MarginRankingCriterionGradientOp<HIPContext>::RunOnDevice() {
  auto& X1 = Input(0);
  auto& X2 = Input(1);
  auto& Y = Input(2);
  auto& dOutput = Input(3);
  auto* dX1 = Output(0);
  auto* dX2 = Output(1);

  dX1->ResizeLike(X1);
  dX2->ResizeLike(X2);

  const float* X1data = X1.data<float>();
  const float* X2data = X2.data<float>();
  const int* Ydata = Y.data<int>();
  const float* dOutput_data = dOutput.data<float>();

  float* dX1_data = dX1->mutable_data<float>();
  float* dX2_data = dX2->mutable_data<float>();
  hipLaunchKernelGGL((MRCGradientKernel), dim3(CAFFE_GET_BLOCKS(X1.size())), dim3(CAFFE_HIP_NUM_THREADS), 0, context_.hip_stream(), 
      static_cast<const int>(X1.size()), Ydata, X1data, X2data,
      dOutput_data, margin_, dX1_data, dX2_data);
  return true;
}

REGISTER_HIP_OPERATOR(
    MarginRankingCriterion,
    MarginRankingCriterionOp<HIPContext>);
REGISTER_HIP_OPERATOR(
    MarginRankingCriterionGradient,
    MarginRankingCriterionGradientOp<HIPContext>);
}  // namespace caffe2
