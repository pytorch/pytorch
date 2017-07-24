#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/multi_class_accuracy_op.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

namespace {
__global__ void MultiClassAccuracyKernel(const int N, const int D, const float* Xdata,
    const int* labeldata, float* accuracies, int* amounts) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    float maxval = Xdata[i * D];
    int maxid = 0;
    for (int j = 1; j < D; ++j) {
      if (Xdata[i * D + j] > maxval) {
        maxval = Xdata[i * D + j];
        maxid = j;
      }
    }
    int labelid = labeldata[i];
    if (maxid == labelid) {
      atomicAdd(accuracies + labelid, static_cast<float>(1));
    }
    atomicAdd(amounts + labelid, static_cast<int>(1));
  }
}
__global__ void MultiClassAccuracyDivideKernel(
  const int D, float* accuracies, const int* amounts) {
  CUDA_1D_KERNEL_LOOP(i, D) {
    if (amounts[i]) {
      accuracies[i] /= amounts[i];
    }
  }
}
}  // namespace

template <>
bool MultiClassAccuracyOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(PREDICTION);
  auto& label = Input(LABEL);
  auto* Y0 = Output(0);
  auto* Y1 = Output(1);
  DCHECK_EQ(X.ndim(), 2);
  // amount, number of instances
  int N = X.dim32(0);
  // dimension, number of classes
  int D = X.dim32(1);
  DCHECK_EQ(label.ndim(), 1);
  DCHECK_EQ(label.dim32(0), N);
  Y0->Resize(D);
  Y1->Resize(D);

  const float* Xdata = X.data<float>();
  const int* labeldata = label.data<int>();
  float* accuracies = Y0->mutable_data<float>();
  int* amounts = Y1->mutable_data<int>();
  math::Set<float, CUDAContext>(D, 0.0, accuracies, &context_);
  math::Set<int, CUDAContext>(D, 0, amounts, &context_);

  MultiClassAccuracyKernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS,
                              0, context_.cuda_stream()>>>(
      N, D, Xdata, labeldata, accuracies, amounts);
  MultiClassAccuracyDivideKernel<<<CAFFE_GET_BLOCKS(D), CAFFE_CUDA_NUM_THREADS,
                                  0, context_.cuda_stream()>>>(
    D, accuracies, amounts);
  return true;
}

REGISTER_CUDA_OPERATOR(
  MultiClassAccuracy, MultiClassAccuracyOp<float, CUDAContext>);
}  // namespace caffe2
