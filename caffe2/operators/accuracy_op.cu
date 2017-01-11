#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/accuracy_op.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

namespace {
__global__ void AccuracyKernel(const int N, const int D, const float* Xdata,
    const int* labeldata, float* accuracy) {
  int count = 0;
  CUDA_1D_KERNEL_LOOP(i, N) {
    float maxval = Xdata[i * D];
    int maxid = 0;
    for (int j = 1; j < D; ++j) {
      if (Xdata[i * D + j] > maxval) {
        maxval = Xdata[i * D + j];
        maxid = j;
      }
    }
    if (maxid == labeldata[i]) {
      ++count;
    }
  }
  atomicAdd(accuracy, static_cast<float>(count));
}
__global__ void AccuracyDivideKernel(const int N, float* accuracy) {
  *accuracy /= N;
}
}  // namespace

template <>
bool AccuracyOp<float, CUDAContext>::RunOnDevice() {
  CAFFE_ENFORCE_EQ(
      top_k_, 1, "Currently only top-1 accuracy supported");
  auto& X = Input(PREDICTION);
  auto& label = Input(LABEL);
  auto* Y = Output(0);
  DCHECK_EQ(X.ndim(), 2);
  int N = X.dim32(0);
  int D = X.dim32(1);
  DCHECK_EQ(label.ndim(), 1);
  DCHECK_EQ(label.dim32(0), N);
  Y->Resize(vector<TIndex>());
  float* Ydata = Y->mutable_data<float>();
  math::Set<float, CUDAContext>(1, 0, Ydata, &context_);
  AccuracyKernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS,
                   0, context_.cuda_stream()>>>(
      N, D, X.data<float>(), label.data<int>(), Ydata);
  // This is going to be executed only in one single kernel. Not very beautiful,
  // but probably we have to do this?
  AccuracyDivideKernel<<<1, 1, 0, context_.cuda_stream()>>>(
      N, Ydata);
  return true;
}

namespace {
REGISTER_CUDA_OPERATOR(Accuracy, AccuracyOp<float, CUDAContext>);
}  // namespace
}  // namespace caffe2
