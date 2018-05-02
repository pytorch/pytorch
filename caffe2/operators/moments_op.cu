#include "caffe2/operators/moments_op.h"

#include <array>
#include <functional>

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

namespace {

template <typename T, int D>
__global__ void ComputeMomentsGradientCUDAKernel(
    const int dX_size,
    const SimpleArray<int, D> dY_strides,
    const SimpleArray<int, D> dX_dims,
    const T scale,
    const T* dmean,
    const T* dvariance,
    const T* X,
    const T* mean,
    T* dX) {
  CUDA_1D_KERNEL_LOOP(dX_index, dX_size) {
    int dY_index = 0;
    int dX_index_val = dX_index;
#pragma unroll
    for (int i = D - 1; i >= 0; --i) {
      dY_index += dY_strides.data[i] == 0
          ? 0
          : (dX_index_val % dX_dims.data[i]) * dY_strides.data[i];
      dX_index_val /= dX_dims.data[i];
    }
#if __CUDA_ARCH__ >= 350
    dX[dX_index] =
        (__ldg(dmean + dY_index) +
         static_cast<T>(2) * (__ldg(X + dX_index) - __ldg(mean + dY_index)) *
             __ldg(dvariance + dY_index)) *
        scale;
#else
    dX[dX_index] = (dmean[dY_index] +
                    static_cast<T>(2) * (X[dX_index] - mean[dY_index]) *
                        dvariance[dY_index]) *
        scale;
#endif
  }
}

template <typename T, int D>
void ComputeMomentsGradientCUDAImpl(
    const int* dY_dims,
    const int* dX_dims,
    const T* dmean,
    const T* dvariance,
    const T* X,
    const T* mean,
    T* dX,
    CUDAContext* context) {
  SimpleArray<int, D> dY_strides_array;
  SimpleArray<int, D> dX_dims_array;
  int cur_stride = 1;
  for (int i = D - 1; i >= 0; --i) {
    dY_strides_array.data[i] = dY_dims[i] == 1 ? 0 : cur_stride;
    dX_dims_array.data[i] = dX_dims[i];
    cur_stride *= dY_dims[i];
  }
  const int dY_size =
      std::accumulate(dY_dims, dY_dims + D, 1, std::multiplies<int>());
  const int dX_size =
      std::accumulate(dX_dims, dX_dims + D, 1, std::multiplies<int>());
  const T scale = static_cast<T>(dY_size) / static_cast<T>(dX_size);
  ComputeMomentsGradientCUDAKernel<T, D>
      <<<CAFFE_GET_BLOCKS(dX_size),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(
          dX_size,
          dY_strides_array,
          dX_dims_array,
          scale,
          dmean,
          dvariance,
          X,
          mean,
          dX);
}

} // namespace

template <>
bool MomentsGradientOp<float, CUDAContext>::Compute(
    const std::vector<int>& dY_dims,
    const std::vector<int>& dX_dims,
    const float* dmean_data,
    const float* dvariance_data,
    const float* X_data,
    const float* mean_data,
    float* dX_data) {
  const int ndim = dY_dims.size();
  DISPATCH_FUNCTION_BY_VALUE_WITH_TYPE_1(
      ndim,
      ComputeMomentsGradientCUDAImpl,
      float,
      dY_dims.data(),
      dX_dims.data(),
      dmean_data,
      dvariance_data,
      X_data,
      mean_data,
      dX_data,
      &context_);
  return true;
}

REGISTER_CUDA_OPERATOR(Moments, MomentsOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(MomentsGradient, MomentsGradientOp<float, CUDAContext>);

} // namespace caffe2
