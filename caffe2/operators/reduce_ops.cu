#include "caffe2/operators/reduce_ops.h"

#include <algorithm>
#include <functional>
#include <vector>

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

namespace {

constexpr int kCUDAReduceGradientMaxDims = 8;

template <typename T, int D>
__global__ void ComputeReduceMinMaxGradientCUDAKernel(
    const int dX_size,
    const SimpleArray<int, D> dY_strides,
    const SimpleArray<int, D> dX_dims,
    const T* dY_data,
    const T* X_data,
    const T* Y_data,
    T* dX_data) {
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
    dX_data[dX_index] = __ldg(Y_data + dY_index) == __ldg(X_data + dX_index)
        ? __ldg(dY_data + dY_index)
        : T(0);
#else
    dX_data[dX_index] =
        Y_data[dY_index] == X_data[dX_index] ? dY_data[dY_index] : T(0);
#endif
  }
}

template <typename T, int D>
void ComputeReduceMinMaxGradientCUDAImpl(
    const int* dY_dims,
    const int* dX_dims,
    const T* dY_data,
    const T* X_data,
    const T* Y_data,
    T* dX_data,
    CUDAContext* context) {
  SimpleArray<int, D> dY_strides_array;
  SimpleArray<int, D> dX_dims_array;
  int cur_stride = 1;
  for (int i = D - 1; i >= 0; --i) {
    dY_strides_array.data[i] = dY_dims[i] == 1 ? 0 : cur_stride;
    dX_dims_array.data[i] = dX_dims[i];
    cur_stride *= dY_dims[i];
  }
  const int dX_size =
      std::accumulate(dX_dims, dX_dims + D, 1, std::multiplies<int>());
  ComputeReduceMinMaxGradientCUDAKernel<T, D>
      <<<CAFFE_GET_BLOCKS(dX_size),
         CAFFE_CUDA_NUM_THREADS,
         0,
         context->cuda_stream()>>>(
          dX_size,
          dY_strides_array,
          dX_dims_array,
          dY_data,
          X_data,
          Y_data,
          dX_data);
}

template <typename T>
void ComputeReduceMinMaxGradientCUDA(
    const std::vector<int>& dY_dims,
    const std::vector<int>& dX_dims,
    const T* dY_data,
    const T* X_data,
    const T* Y_data,
    T* dX_data,
    CUDAContext* context) {
  const int ndim = dY_dims.size();
  switch (ndim) {
    case 1: {
      ComputeReduceMinMaxGradientCUDAImpl<T, 1>(
          dY_dims.data(),
          dX_dims.data(),
          dY_data,
          X_data,
          Y_data,
          dX_data,
          context);
      break;
    }
    case 2: {
      ComputeReduceMinMaxGradientCUDAImpl<T, 2>(
          dY_dims.data(),
          dX_dims.data(),
          dY_data,
          X_data,
          Y_data,
          dX_data,
          context);
      break;
    }
    case 3: {
      ComputeReduceMinMaxGradientCUDAImpl<T, 3>(
          dY_dims.data(),
          dX_dims.data(),
          dY_data,
          X_data,
          Y_data,
          dX_data,
          context);
      break;
    }
    case 4: {
      ComputeReduceMinMaxGradientCUDAImpl<T, 4>(
          dY_dims.data(),
          dX_dims.data(),
          dY_data,
          X_data,
          Y_data,
          dX_data,
          context);
      break;
    }
    case 5: {
      ComputeReduceMinMaxGradientCUDAImpl<T, 5>(
          dY_dims.data(),
          dX_dims.data(),
          dY_data,
          X_data,
          Y_data,
          dX_data,
          context);
      break;
    }
    case 6: {
      ComputeReduceMinMaxGradientCUDAImpl<T, 6>(
          dY_dims.data(),
          dX_dims.data(),
          dY_data,
          X_data,
          Y_data,
          dX_data,
          context);
      break;
    }
    case 7: {
      ComputeReduceMinMaxGradientCUDAImpl<T, 7>(
          dY_dims.data(),
          dX_dims.data(),
          dY_data,
          X_data,
          Y_data,
          dX_data,
          context);
      break;
    }
    case 8: {
      ComputeReduceMinMaxGradientCUDAImpl<T, 8>(
          dY_dims.data(),
          dX_dims.data(),
          dY_data,
          X_data,
          Y_data,
          dX_data,
          context);
      break;
    }
    default: { break; }
  }
}

} // namespace

template <typename T>
class ReduceMinMaxGradientOp<T, CUDAContext> final
    : public ReduceGradientOpBase<T, CUDAContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CUDAContext);

  ReduceMinMaxGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : ReduceGradientOpBase<T, CUDAContext>(operator_def, ws) {}

 protected:
  bool Compute(
      const std::vector<int>& dY_dims,
      const std::vector<int>& dX_dims,
      const T* dY_data,
      const T* X_data,
      const T* Y_data,
      T* dX_data) override {
    CAFFE_ENFORCE_LE(dY_dims.size(), kCUDAReduceGradientMaxDims);
    ComputeReduceMinMaxGradientCUDA(
        dY_dims, dX_dims, dY_data, X_data, Y_data, dX_data, &context_);
    return true;
  }
};

REGISTER_CUDA_OPERATOR(ReduceMin, ReduceMinOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    ReduceMinGradient,
    ReduceMinMaxGradientOp<float, CUDAContext>);

REGISTER_CUDA_OPERATOR(ReduceMax, ReduceMaxOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    ReduceMaxGradient,
    ReduceMinMaxGradientOp<float, CUDAContext>);

REGISTER_CUDA_OPERATOR(ReduceSum, ReduceSumOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    ReduceSumGradient,
    ReduceSumGradientOp<float, CUDAContext>);

REGISTER_CUDA_OPERATOR(ReduceMean, ReduceMeanOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    ReduceMeanGradient,
    ReduceMeanGradientOp<float, CUDAContext>);

} // namespace caffe2
