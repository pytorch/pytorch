#include <cub/block/block_reduce.cuh>

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/distance_op.h"

namespace caffe2 {

namespace {

template <typename T>
__global__ void SquaredL2DistanceKernel(
    const int N, const int D, const T* X, const T* Y, T* distance) {
  typedef cub::BlockReduce<float, CAFFE_CUDA_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  for (int i = blockIdx.x; i < N; i += gridDim.x) {
    float dist = 0.0;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
      T diff = X[i * D + j] - Y[i * D + j];
      dist += diff * diff;
    }

    float total_dist = BlockReduce(temp_storage).Sum(dist);
    __syncthreads();
    if (threadIdx.x == 0) {
      distance[i] = total_dist / 2.0;
    }
  }
}
}  // namespace

template<>
bool SquaredL2DistanceOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& Y = Input(1);
  auto* distance = Output(0);
  DCHECK_EQ(X.ndim(), Y.ndim());
  for (int i = 0; i < X.ndim(); ++i) {
    DCHECK_EQ(X.dim32(i), Y.dim32(i));
  }
  int N = X.ndim() > 0 ? X.dim32(0) : 1;
  int D = X.size() / N;
  distance->Resize(vector<TIndex>(size_t(1), N));
  SquaredL2DistanceKernel<<<
      std::min(N, CAFFE_MAXIMUM_NUM_BLOCKS),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      N, D, X.data<float>(), Y.data<float>(), distance->mutable_data<float>());
  return true;
}

namespace {
template <typename T>
__global__ void
StripedScaleKernel(const int N, const int D, const T* alpha, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, N * D) {
    int k = i / D;
    y[i] = x[i] * alpha[k];
  }
}
}

template <>
bool SquaredL2DistanceGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& Y = Input(1);
  auto& dDistance = Input(2);
  auto* dX = Output(0);
  auto* dY = Output(1);
  int N = X.ndim() > 0 ? X.dim32(0) : 1;
  int D = N > 0 ? X.size() / N : 0;
  CAFFE_ENFORCE(X.ndim() == Y.ndim());
  for (int i = 0; i < X.ndim(); ++i) {
    CAFFE_ENFORCE(X.dim32(i) == Y.dim32(i));
  }
  CAFFE_ENFORCE(dDistance.ndim() == 1);
  CAFFE_ENFORCE(dDistance.dim32(0) == N);
  dX->ResizeLike(X);
  dY->ResizeLike(Y);
  math::Sub<float, CUDAContext>(
      X.size(),
      X.data<float>(),
      Y.data<float>(),
      dX->mutable_data<float>(),
      &context_);

  StripedScaleKernel<float><<<
      CAFFE_GET_BLOCKS(N * D),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      N,
      D,
      dDistance.data<float>(),
      dX->data<float>(),
      dX->mutable_data<float>());

  // The gradient of the other side is basically the negative.
  math::Scale<float, CUDAContext>(
      X.size(), -1, dX->data<float>(), dY->mutable_data<float>(), &context_);
  return true;
}

namespace {
REGISTER_CUDA_OPERATOR(SquaredL2Distance,
                       SquaredL2DistanceOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(SquaredL2DistanceGradient,
                       SquaredL2DistanceGradientOp<float, CUDAContext>);
}  // namespace
}  // namespace caffe2
