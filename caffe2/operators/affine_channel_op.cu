#include "caffe2/operators/affine_channel_op.h"

#include <cub/block/block_reduce.cuh>

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

namespace {

template <typename T>
using BlockReduce = cub::BlockReduce<T, CAFFE_CUDA_NUM_THREADS>;

template <typename T, StorageOrder kOrder>
__global__ void AffineChannelScaleBiasBackwardCUDAKernel(
    const int N,
    const int C,
    const int HxW,
    const T* dY,
    const T* X,
    T* dscale,
    T* dbias) {
  const int outer_size = C;
  const int inner_size = N * HxW;
  __shared__ typename BlockReduce<T>::TempStorage ds_storage;
  __shared__ typename BlockReduce<T>::TempStorage db_storage;
  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    T ds_sum = 0;
    T db_sum = 0;
    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
      const int index = kOrder == StorageOrder::NCHW
          ? (j / HxW * C + i) * HxW + j % HxW
          : j * outer_size + i;
#if __CUDA_ARCH__ >= 350
      ds_sum += __ldg(dY + index) * __ldg(X + index);
      db_sum += __ldg(dY + index);
#else
      ds_sum += dY[index] * X[index];
      db_sum += dY[index];
#endif
    }
    ds_sum = BlockReduce<T>(ds_storage).Reduce(ds_sum, cub::Sum());
    db_sum = BlockReduce<T>(db_storage).Reduce(db_sum, cub::Sum());
    if (threadIdx.x == 0) {
      dscale[i] = ds_sum;
      dbias[i] = db_sum;
    }
    __syncthreads();
  }
}

} // namespace

template <>
bool AffineChannelGradientOp<float, CUDAContext>::RunOnDeviceWithOrderNCHW() {
  const auto& dY = Input(0);
  const auto& scale = is_learnable_ ? Input(2) : Input(1);
  
  auto* dX = Output(0, dY.sizes(), at::dtype<float>());
  const int N = dY.dim32(0);
  const int C = dY.dim32(1);
  const int HxW = dY.numel() / (N * C);
  const float* dY_data = dY.data<float>();
  const float* scale_data = scale.data<float>();
  const std::array<int, 3> X_dims = {N, C, HxW};
  const std::array<int, 3> scale_dims = {1, C, 1};
  math::Mul<float, CUDAContext>(
      3,
      X_dims.data(),
      3,
      scale_dims.data(),
      dY_data,
      scale_data,
      dX->template mutable_data<float>(),
      &context_);
  if (is_learnable_) {
    const auto& X = Input(1);
    const float* X_data = X.data<float>();
    
    
    auto* dscale = Output(1, scale.sizes(), at::dtype<float>());
    auto* dbias = Output(2, scale.sizes(), at::dtype<float>());
    const int outer_size = N * HxW;
    AffineChannelScaleBiasBackwardCUDAKernel<float, StorageOrder::NCHW>
        <<<std::min(outer_size, CAFFE_MAXIMUM_NUM_BLOCKS),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(
            N,
            C,
            HxW,
            dY_data,
            X_data,
            dscale->template mutable_data<float>(),
            dbias->template mutable_data<float>());
  }
  return true;
}

template <>
bool AffineChannelGradientOp<float, CUDAContext>::RunOnDeviceWithOrderNHWC() {
  const auto& dY = Input(0);
  const auto& scale = is_learnable_ ? Input(2) : Input(1);
  
  auto* dX = Output(0, dY.sizes(), at::dtype<float>());
  const int ndim = dY.dim();
  const int C = dY.dim32(ndim - 1);
  const int rows = dY.numel() / C;
  const int cols = C;
  const float* dY_data = dY.data<float>();
  const float* scale_data = scale.data<float>();
  math::RowwiseMul<float, CUDAContext>(
      rows,
      cols,
      dY_data,
      scale_data,
      dX->template mutable_data<float>(),
      &context_);
  if (is_learnable_) {
    const auto& X = Input(1);
    const float* X_data = X.data<float>();
    const int N = X.dim32(0);
    const int HxW = rows / N;
    
    
    auto* dscale = Output(1, scale.sizes(), at::dtype<float>());
    auto* dbias = Output(2, scale.sizes(), at::dtype<float>());
    AffineChannelScaleBiasBackwardCUDAKernel<float, StorageOrder::NHWC>
        <<<std::min(rows, CAFFE_MAXIMUM_NUM_BLOCKS),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(
            N,
            C,
            HxW,
            dY_data,
            X_data,
            dscale->template mutable_data<float>(),
            dbias->template mutable_data<float>());
  }
  return true;
}

REGISTER_CUDA_OPERATOR(AffineChannel, AffineChannelOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    AffineChannelGradient,
    AffineChannelGradientOp<float, CUDAContext>);

} // namespace caffe2
