#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>


// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at { namespace native {

void atan2_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.common_dtype(), "atan2_cuda", [&]() {
    gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return ::atan2(a, b);
    });
  });
}

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

void lshift_kernel_cuda(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Float || iter.dtype() == ScalarType::Double) {
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "lshift_cuda", [&]() {
      gpu_kernel_with_scalars(
        iter,
        []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
          return a * std::pow((scalar_t)(2), b);
      });
    });
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "lshift_cuda", [&]() {
      gpu_kernel_with_scalars(iter,
        []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
          return a << b;
      });
    });
  }
}

void rshift_kernel_cuda(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Float || iter.dtype() == ScalarType::Double) {
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "rshift_cuda", [&]() {
      gpu_kernel_with_scalars(
        iter,
        []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
          return a / std::pow((scalar_t)(2), b);
      });
    });
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "rshift_cuda", [&]() {
      gpu_kernel_with_scalars(iter,
        []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
          return a >> b;
      });
    });
  }
}

void logical_and_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBool, iter.common_dtype(), "logical_and_cuda", [&]() {
    gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> bool {
      return a && b;
    });
  });
}

void logical_or_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBool, iter.common_dtype(), "logical_or_cuda", [&]() {
    gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> bool {
      return a || b;
    });
  });
}

void logical_xor_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBool, iter.common_dtype(), "logical_xor_cuda", [&]() {
    gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> bool {
      return bool(a) != bool(b);
    });
  });
}

void smooth_l1_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "smooth_l1_cuda", [&]() {
    gpu_kernel(iter, [] GPU_LAMBDA (scalar_t a, scalar_t b) -> scalar_t {
      auto z = ::abs(a - b);
      return z < scalar_t(1.) ? scalar_t(0.5) * z * z : z - scalar_t(0.5);
    });
  });
}

void sigmoid_backward_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "sigmoid_backward_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a * (scalar_t(1.) - b) * b;
    });
  });
}

void tanh_backward_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "tanh_backward_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a * (scalar_t(1.) - b * b);
    });
  });
}

void mse_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "mse_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      auto diff = a - b;
      return diff * diff;
    });
  });
}

REGISTER_DISPATCH(atan2_stub, &atan2_kernel_cuda);
REGISTER_DISPATCH(bitwise_and_stub, &bitwise_and_kernel_cuda);
REGISTER_DISPATCH(bitwise_or_stub, &bitwise_or_kernel_cuda);
REGISTER_DISPATCH(lshift_stub, &lshift_kernel_cuda);
REGISTER_DISPATCH(bitwise_xor_stub, &bitwise_xor_kernel_cuda);
REGISTER_DISPATCH(rshift_stub, &rshift_kernel_cuda);
REGISTER_DISPATCH(logical_and_stub, &logical_and_kernel_cuda);
REGISTER_DISPATCH(logical_or_stub, &logical_or_kernel_cuda);
REGISTER_DISPATCH(logical_xor_stub, &logical_xor_kernel_cuda);
REGISTER_DISPATCH(smooth_l1_stub, &smooth_l1_kernel_cuda);
REGISTER_DISPATCH(sigmoid_backward_stub, &sigmoid_backward_kernel_cuda);
REGISTER_DISPATCH(tanh_backward_stub, &tanh_backward_kernel_cuda);
REGISTER_DISPATCH(mse_stub, &mse_kernel_cuda);

}} // namespace at::native
