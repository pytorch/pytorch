#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at { namespace native {

void div_kernel_cuda(TensorIterator& iter) {
  if (!isIntegralType(iter.common_dtype(), /*includeBool*/ false) && iter.is_cpu_scalar(2)) {
    // optimization for floating-point types: if the second operand is a CPU
    // scalar, compute a * reciprocal(b). Note that this may lose one bit of
    // precision compared to computing the division.
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kHalf, kBFloat16, iter.common_dtype(), "div_cuda", [&]() {
      auto inv_b = scalar_t(1.0) / iter.scalar_value<scalar_t>(2);
      iter.remove_operand(2);
      gpu_kernel(iter, [inv_b]GPU_LAMBDA(scalar_t a) -> scalar_t {
        return a * inv_b;
      });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kHalf, kBFloat16, iter.common_dtype(), "div_cuda", [&]() {
      gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        return a / b;
      });
    });
  }
}

void mul_kernel_cuda(TensorIterator& iter) {
  if (iter.common_dtype() == ScalarType::Bool) {
    // Workaround for the error: '*' in boolean context, suggest '&&' instead [-Werror=int-in-bool-context]
    gpu_kernel_with_scalars(iter, []GPU_LAMBDA(bool a, bool b) -> bool {
      return a && b;
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kHalf, kBFloat16, iter.common_dtype(), "mul_cuda", [&]() {
      gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        return a * b;
      });
    });
  }
}

REGISTER_DISPATCH(div_stub, &div_kernel_cuda);
REGISTER_DISPATCH(mul_stub, &mul_kernel_cuda);

}} // namespace at::native
