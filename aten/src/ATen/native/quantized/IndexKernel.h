#pragma once
#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {

void index_put_kernel_quantized_cpu(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride, bool accumulate, double scale, int zero_point);
void masked_fill_kernel_quantized_cpu(TensorIterator& iter, const Scalar& value, double scale, int zero_point);

// TODO: implement index_put_kernel_quantized_cuda in cuda/IndexKernel.cu and put CPU & CUDA kernels in a stub
// TODO: implement masked_fill_kernel_quantized_cuda in cuda/IndexKernel.cu and put CPU & CUDA kernels in a stub

} // native
} // at
