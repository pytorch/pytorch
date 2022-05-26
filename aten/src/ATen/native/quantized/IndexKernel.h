#pragma once
#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {

void masked_fill_kernel_quantized_cpu(TensorIterator& iter, const Scalar& value, double scale, int zero_point);

// TODO: implement masked_fill_kernel_quantized_cuda in cuda/IndexKernel.cu and put CPU & CUDA kernels in a stub

} // native
} // at
