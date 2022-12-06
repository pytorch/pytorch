#pragma once
#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {
using masked_fill_kernel_quantized_fn = void(*)(TensorIterator& iter, const Scalar& value, double scale, int zero_point);
using index_put_kernel_quantized_fn = void(*)(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride, bool accumulate, double scale, int zero_point);

DECLARE_DISPATCH(masked_fill_kernel_quantized_fn, masked_fill_kernel_quantized_stub);
DECLARE_DISPATCH(index_put_kernel_quantized_fn, index_put_kernel_quantized_stub);


} // native
} // at
