#include "caffe2/operators/tile_op.h"

#include <array>

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

namespace {

template <typename T>
__global__ void TileCopyCUDAKernel(
    const int total_size,
    const int inner_size,
    const int tiles,
    const T* X,
    T* Y) {
  const int x = blockIdx.x * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
  if (x < total_size) {
    const int r = x / inner_size / tiles;
    const int c = x % inner_size;
#if __CUDA_ARCH__ >= 350 || defined(__HIP_PLATFORM_HCC__)
    Y[x] = __ldg(X + r * inner_size + c);
#else
    Y[x] = X[r * inner_size + c];
#endif
  }
}

} // namespace

template <>
template <typename T>
bool TileOp<CUDAContext>::DoTile(
    const int outer_size,
    const int inner_size,
    const T* X,
    T* Y) {
  const std::int64_t total_size = static_cast<std::int64_t>(outer_size) *
      static_cast<std::int64_t>(tiles_) * static_cast<std::int64_t>(inner_size);
  const int M = math::DivUp<std::int64_t>(total_size, CAFFE_CUDA_NUM_THREADS);
  TileCopyCUDAKernel<T>
      <<<M, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
          total_size, inner_size, tiles_, X, Y);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

template <>
template <typename T>
bool TileGradientOp<CUDAContext>::DoTileGradient(
    const int outer_size,
    const int inner_size,
    const T* dY,
    T* dX) {
  const std::array<int, 3> dY_dims = {outer_size, tiles_, inner_size};
  const std::array<int, 3> dX_dims = {outer_size, 1, inner_size};
  math::ReduceSum<T, CUDAContext>(
      3, dY_dims.data(), dX_dims.data(), T(1), dY, dX, &context_);
  return true;
}

template <>
template <>
bool TileGradientOp<CUDAContext>::DoTileGradient<float>(
    const int outer_size,
    const int inner_size,
    const float* dY,
    float* dX) {
  if (inner_size == 1) {
    const std::array<int, 2> dY_dims = {outer_size, tiles_};
    const std::array<int, 2> dX_dims = {outer_size, 1};
    math::ReduceSum<float, CUDAContext>(
        2, dY_dims.data(), dX_dims.data(), 1.0f, dY, dX, &context_);
  } else {
    ReinitializeTensor(&ones_, tiles_, at::dtype<float>().device(CUDA));
    math::Set<float, CUDAContext>(
        tiles_, 1.0f, ones_.template mutable_data<float>(), &context_);
    math::GemmStridedBatched<float, CUDAContext>(
        CblasTrans,
        CblasNoTrans,
        outer_size,
        inner_size,
        1,
        tiles_,
        1.0f,
        dY,
        tiles_ * inner_size,
        ones_.template data<float>(),
        0,
        0.0f,
        dX,
        inner_size,
        &context_);
  }
  return true;
}

REGISTER_CUDA_OPERATOR(Tile, TileOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(TileGradient, TileGradientOp<CUDAContext>);

} // namespace caffe2
