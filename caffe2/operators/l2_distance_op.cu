#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/l2_distance_op.h"

namespace caffe2 {

namespace {
// TODO(Yangqing): This function does very aweful memory access.
// Need improvement.
template <typename dtype>
__global__ void SquaredL2DistanceKernel(
    const int N, const int D, const dtype* X, const dtype* Y, dtype* distance) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    distance[i] = 0;
    for (int j = 0; j < D; ++j) {
      dtype diff = X[i * D + j] - Y[i * D + j];
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
  DCHECK_EQ(X.ndim(), Y.ndim());
  for (int i = 0; i < X.ndim(); ++i) {
    DCHECK_EQ(X.dim(i), Y.dim(i));
  }
  int N = X.dim(0);
  int D = X.size() / X.dim(0);
  distance->Reshape(std::vector<int>(1, N));
  SquaredL2DistanceKernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS,
                            0, device_context_.cuda_stream()>>>(
      N, D, X.data(), Y.data(), distance->mutable_data());
  return true;
}


namespace {
REGISTER_CUDA_OPERATOR(SquaredL2Distance,
                       SquaredL2DistanceOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(SquaredL2DistanceGradient,
                       SquaredL2DistanceGradientOp<float, CUDAContext>);
}  // namespace
}  // namespace caffe2
