#include "caffe2/operators/moments_op.h"

#include <array>
#include <functional>

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

namespace {

constexpr int kCUDAMomentsMaxDims = 8;

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

template <typename T>
void ComputeMomentsGradientCUDA(
    const std::vector<int>& dY_dims,
    const std::vector<int>& dX_dims,
    const T* dmean,
    const T* dvariance,
    const T* X,
    const T* mean,
    T* dX,
    CUDAContext* context) {
  const int ndim = dX_dims.size();
  CAFFE_ENFORCE_LE(ndim, kCUDAMomentsMaxDims);
  switch (ndim) {
    case 1: {
      ComputeMomentsGradientCUDAImpl<T, 1>(
          dY_dims.data(),
          dX_dims.data(),
          dmean,
          dvariance,
          X,
          mean,
          dX,
          context);
      break;
    }
    case 2: {
      ComputeMomentsGradientCUDAImpl<T, 2>(
          dY_dims.data(),
          dX_dims.data(),
          dmean,
          dvariance,
          X,
          mean,
          dX,
          context);
      break;
    }
    case 3: {
      ComputeMomentsGradientCUDAImpl<T, 3>(
          dY_dims.data(),
          dX_dims.data(),
          dmean,
          dvariance,
          X,
          mean,
          dX,
          context);
      break;
    }
    case 4: {
      ComputeMomentsGradientCUDAImpl<T, 4>(
          dY_dims.data(),
          dX_dims.data(),
          dmean,
          dvariance,
          X,
          mean,
          dX,
          context);
      break;
    }
    case 5: {
      ComputeMomentsGradientCUDAImpl<T, 5>(
          dY_dims.data(),
          dX_dims.data(),
          dmean,
          dvariance,
          X,
          mean,
          dX,
          context);
      break;
    }
    case 6: {
      ComputeMomentsGradientCUDAImpl<T, 6>(
          dY_dims.data(),
          dX_dims.data(),
          dmean,
          dvariance,
          X,
          mean,
          dX,
          context);
      break;
    }
    case 7: {
      ComputeMomentsGradientCUDAImpl<T, 7>(
          dY_dims.data(),
          dX_dims.data(),
          dmean,
          dvariance,
          X,
          mean,
          dX,
          context);
      break;
    }
    case 8: {
      ComputeMomentsGradientCUDAImpl<T, 8>(
          dY_dims.data(),
          dX_dims.data(),
          dmean,
          dvariance,
          X,
          mean,
          dX,
          context);
      break;
    }
    default: { break; }
  }
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
  ComputeMomentsGradientCUDA<float>(
      dY_dims,
      dX_dims,
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
