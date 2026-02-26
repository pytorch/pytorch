#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at::native {

template<typename scalar_t>
struct BitwiseAndFunctor {
  __device__ __forceinline__ scalar_t operator()(scalar_t a, scalar_t b) const {
    return a & b;
  }
};

template<>
struct BitwiseAndFunctor<bool> {
  __device__ __forceinline__ bool operator()(bool a, bool b) const {
    return a && b;
  }
};

void bitwise_and_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_INTEGRAL_TYPES_AND(kBool, iter.dtype(), "bitwise_and_cuda", [&]() {
    BitwiseAndFunctor<scalar_t> f;
    opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(iter, f);
  });
}

template<typename scalar_t>
struct BitwiseOrFunctor {
  __device__ __forceinline__ scalar_t operator()(scalar_t a, scalar_t b) const {
    return a | b;
  }
};

template<>
struct BitwiseOrFunctor<bool> {
  __device__ __forceinline__ bool operator()(bool a, bool b) const {
    return a || b;
  }
};

void bitwise_or_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_INTEGRAL_TYPES_AND(kBool, iter.dtype(), "bitwise_or_cuda", [&]() {
    BitwiseOrFunctor<scalar_t> f;
    opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(iter, f);
  });
}

template<typename scalar_t>
struct BitwiseXorFunctor {
  __device__ __forceinline__ scalar_t operator()(scalar_t a, scalar_t b) const {
    return a ^ b;
  }
};

template<>
struct BitwiseXorFunctor<bool> {
  __device__ __forceinline__ bool operator()(bool a, bool b) const {
    return a != b;
  }
};

void bitwise_xor_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_INTEGRAL_TYPES_AND(kBool, iter.dtype(), "bitwise_xor_cuda", [&]() {
    BitwiseXorFunctor<scalar_t> f;
    opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(iter, f);
  });
}

REGISTER_DISPATCH(bitwise_and_stub, &bitwise_and_kernel_cuda)
REGISTER_DISPATCH(bitwise_or_stub, &bitwise_or_kernel_cuda)
REGISTER_DISPATCH(bitwise_xor_stub, &bitwise_xor_kernel_cuda)


} // namespace at::native
