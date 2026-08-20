#pragma once
#include <c10/metal/common.h>

// Byte offsets for one strided frexp dispatch. int32 is sufficient because the
// stub splits any iterator whose per-operand offsets do not fit (see
// TensorIteratorBase::can_use_32bit_indexing).
struct FrexpParams {
  ::c10::metal::array<int32_t, ::c10::metal::max_ndim> sizes;
  ::c10::metal::array<int32_t, ::c10::metal::max_ndim> mantissa_strides;
  ::c10::metal::array<int32_t, ::c10::metal::max_ndim> exponent_strides;
  ::c10::metal::array<int32_t, ::c10::metal::max_ndim> input_strides;
  int32_t ndim;
};

// Replacement values are computed per-dtype on the host (CUDA-style: the
// posinf/neginf defaults are the input dtype's max/lowest) and ride at float,
// which represents half/bfloat extrema exactly; the struct is always
// instantiated at T = float.
template <typename T>
struct NanToNumParams {
  T nan;
  T posinf;
  T neginf;
};
