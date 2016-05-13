#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/l2_distance_op.h"

namespace caffe2 {

namespace {
// TODO(Yangqing): This function does very aweful memory access.
// Need improvement.
template <typename T>
__global__ void SquaredL2DistanceKernel(
    const int N, const int D, const T* X, const T* Y, T* distance) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    distance[i] = 0;
    for (int j = 0; j < D; ++j) {
      T diff = X[i * D + j] - Y[i * D + j];
      distance[i] += diff * diff;
    }
    distance[i] /= 2;
  }
}
}  // namespace

template<>
bool SquaredL2DistanceOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& Y = Input(1);
  auto* distance = Output(0);
  CAFFE_DCHECK_EQ(X.ndim(), Y.ndim());
  for (int i = 0; i < X.ndim(); ++i) {
    CAFFE_DCHECK_EQ(X.dim32(i), Y.dim32(i));
  }
  int N = X.dim32(0);
  int D = X.size() / X.dim32(0);
  distance->Reshape(vector<TIndex>(1, N));
  SquaredL2DistanceKernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS,
                            0, context_.cuda_stream()>>>(
      N, D, X.data<float>(), Y.data<float>(), distance->mutable_data<float>());
  return true;
}


namespace {
REGISTER_CUDA_OPERATOR(SquaredL2Distance,
                       SquaredL2DistanceOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(SquaredL2DistanceGradient,
                       SquaredL2DistanceGradientOp<float, CUDAContext>);
}  // namespace
}  // namespace caffe2
