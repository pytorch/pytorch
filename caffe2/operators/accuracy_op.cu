#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/accuracy_op.h"
#include "caffe2/utils/math.h"

#include <cub/block/block_reduce.cuh>

namespace caffe2 {

namespace {
__global__ void AccuracyKernel(
    const int N,
    const int D,
    const int top_k,
    const float* Xdata,
    const int* labelData,
    float* accuracy) {
  typedef cub::BlockReduce<int, CAFFE_CUDA_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int correct = 0;
  for (int row = blockIdx.x; row < N; row += gridDim.x) {
    const int label = labelData[row];
    const float label_pred = Xdata[row * D + label];
    int ngt = 0;
    for (int col = threadIdx.x; col < D; col += blockDim.x) {
      const float pred = Xdata[row * D + col];
      if (pred > label_pred || (pred == label_pred && col <= label)) {
        ++ngt;
      }
    }
    ngt = BlockReduce(temp_storage).Sum(ngt);
    if (ngt <= top_k) {
      ++correct;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicAdd(accuracy, static_cast<float>(correct));
  }
}

__global__ void AccuracyDivideKernel(const int N, float* accuracy) {
  *accuracy /= N;
}
}  // namespace

template <>
bool AccuracyOp<float, CUDAContext>::RunOnDevice() {
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
  AccuracyKernel<<<
      std::min(CAFFE_MAXIMUM_NUM_BLOCKS, N),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      N, D, top_k_, X.data<float>(), label.data<int>(), Ydata);
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
