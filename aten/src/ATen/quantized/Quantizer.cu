#include <ATen/ATen.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/core/Allocator.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/quantized/QTensorImpl.h>
#include <ATen/core/Tensor.h>
#include <typeinfo>
#include <ATen/cuda/CUDAApplyUtils.cuh>

namespace at {

template <typename T>
CAFFE2_API Tensor quantize_tensor_cuda(Tensor rtensor, Tensor qtensor, double scale, int64_t zero_point){
  constexpr int64_t qmin = std::numeric_limits<typename T::underlying>::min();
  constexpr int64_t qmax = std::numeric_limits<typename T::underlying>::max();
  at::cuda::CUDA_tensor_apply2<float, T>(
    /*a=*/rtensor,
    /*b=*/qtensor,
    [=] __device__ (
      float& rtensor_val,
      T& qtensor_val) {
        int64_t qvalue;
        qvalue = static_cast<int64_t>(std::nearbyint(rtensor_val / scale + zero_point));
        qvalue = std::max<int64_t>(qvalue, qmin);
        qvalue = std::min<int64_t>(qvalue, qmax);
        qtensor_val = static_cast<T>(qvalue);
  },
  /*aType=*/at::cuda::TensorArgType::ReadOnly,
  /*bType=*/at::cuda::TensorArgType::ReadWrite);
  return qtensor;
}

template <typename T>
CAFFE2_API Tensor dequantize_tensor_cuda(Tensor qtensor, Tensor rtensor, double scale, int64_t zero_point){
  at::cuda::CUDA_tensor_apply2<T, float>(
    /*a=*/qtensor,
    /*b=*/rtensor,
    [=] __device__ (
      T& qtensor_val,
      float& rtensor_val) {
        rtensor_val = (static_cast<float>(qtensor_val.val_) - zero_point) * scale;
      },
  /*aType=*/at::cuda::TensorArgType::ReadOnly,
  /*bType=*/at::cuda::TensorArgType::ReadWrite);
  return rtensor;
}


template CAFFE2_API Tensor quantize_tensor_cuda<qint8>(Tensor rtensor, Tensor qtensor, double scale, int64_t zero_point);
template CAFFE2_API Tensor quantize_tensor_cuda<quint8>(Tensor rtensor, Tensor qtensor, double scale, int64_t zero_point);
template CAFFE2_API Tensor quantize_tensor_cuda<qint32>(Tensor rtensor, Tensor qtensor, double scale, int64_t zero_point);
template CAFFE2_API Tensor dequantize_tensor_cuda<qint8>(Tensor rtensor, Tensor qtensor, double scale, int64_t zero_point);
template CAFFE2_API Tensor dequantize_tensor_cuda<quint8>(Tensor rtensor, Tensor qtensor, double scale, int64_t zero_point);
template CAFFE2_API Tensor dequantize_tensor_cuda<qint32>(Tensor rtensor, Tensor qtensor, double scale, int64_t zero_point);

} // namespace at
