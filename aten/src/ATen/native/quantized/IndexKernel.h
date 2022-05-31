#pragma once
#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {
using masked_fill_kernel_quantized_fn = void(*)(TensorIterator& iter, const Scalar& value, double scale, int zero_point);
DECLARE_DISPATCH(masked_fill_kernel_quantized_fn, masked_fill_kernel_quantized_stub);

} // native
} // at
