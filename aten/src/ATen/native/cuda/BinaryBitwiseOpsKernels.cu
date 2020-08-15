#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at { namespace native {

void bitwise_and_kernel_cuda(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    gpu_kernel_with_scalars(
        iter,
        []GPU_LAMBDA(bool a, bool b) {
          return a && b;
    });
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "bitwise_and_cuda", [&]() {
      gpu_kernel_with_scalars(
          iter,
          []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
            return a & b;
      });
    });
  }
}

void bitwise_or_kernel_cuda(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    gpu_kernel_with_scalars(
        iter,
        []GPU_LAMBDA(bool a, bool b) {
          return a || b;
    });
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "bitwise_or_cuda", [&]() {
      gpu_kernel_with_scalars(
          iter,
          []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
            return a | b;
      });
    });
  }
}

void bitwise_xor_kernel_cuda(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    // Boolean type does not work with ^ (bitwise XOR) in C++. bitwise_xor wraps this operation for both Boolean and
    // integral types.
    gpu_kernel_with_scalars(
          iter,
          []GPU_LAMBDA(bool a, bool b) {
            return a != b;
          });
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "bitwise_xor_cuda", [&]() {
      gpu_kernel_with_scalars(
          iter,
          []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
            return a ^ b;
      });
    });
  }
}

REGISTER_DISPATCH(bitwise_and_stub, &bitwise_and_kernel_cuda);
REGISTER_DISPATCH(bitwise_or_stub, &bitwise_or_kernel_cuda);
REGISTER_DISPATCH(bitwise_xor_stub, &bitwise_xor_kernel_cuda);


}} // namespace at::native
