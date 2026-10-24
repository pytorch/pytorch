// DON'T include this except from Binary*.cu files. It should not leak into
// headers.
#pragma once
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <c10/util/TypeSafeSignMath.h>
#include <ATen/native/cuda/JitLoops.cuh>
#include <ATen/native/cuda/Loops.cuh>

#include <cmath>
#include <type_traits>

namespace at::native::binary_internal {

template <typename scalar_t>
struct DivFunctor {
  __device__ scalar_t operator()(scalar_t a, scalar_t b) const {
    return a / b;
  }
};

template <typename T>
struct MulFunctor {
  __device__ T operator()(T a, T b) const {
    return a * b;
  }
};

// Workaround for the error: '*' in boolean context, suggest '&&' instead
// [-Werror=int-in-bool-context]
template <>
struct MulFunctor<bool> {
  __device__ bool operator()(bool a, bool b) const {
    return a && b;
  }
};

// Whether a host-computed reciprocal 1/b of a CPU scalar divisor can replace
// division by b: it must not have overflowed or become subnormal.
template <typename T>
bool is_usable_reciprocal(T inv_b) {
  return inv_b == T(0) || std::isnormal(inv_b);
}

// Complex keeps the reciprocal, so complex divisors are not fixed here:
// c10::complex division computes 1 / (c + d * (d / c)) itself, so dividing
// would still give inf for a tiny b (e.g. complex(1e-39, 0) in complex64) and
// would overflow to 0 for a huge b, where the subnormal reciprocal only loses
// precision.
template <typename T>
bool is_usable_reciprocal(c10::complex<T>) {
  return true;
}

void div_true_kernel_cuda(TensorIteratorBase& iter);
void div_trunc_kernel_cuda(TensorIteratorBase& iter);
} // namespace at::native::binary_internal
