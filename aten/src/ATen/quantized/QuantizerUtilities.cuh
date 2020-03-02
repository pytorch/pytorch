#pragma once

#include <ATen/ATen.h>

namespace at {

template <typename T>
CAFFE2_API Tensor quantize_tensor_cuda(Tensor rtensor, Tensor qtensor, double scale, int64_t zero_point);
template <typename T>
CAFFE2_API Tensor dequantize_tensor_cuda(Tensor qtensor, Tensor rtensor, double scale, int64_t zero_point);

}
