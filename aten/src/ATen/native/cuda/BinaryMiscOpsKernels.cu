#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/Math.cuh>
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

void smooth_l1_kernel_cuda(TensorIterator& iter, double beta) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "smooth_l1_cuda", [&]() {
    gpu_kernel(iter, [] GPU_LAMBDA (scalar_t a, scalar_t b) -> scalar_t {
      auto z = ::abs(a - b);
      return z < beta ? scalar_t(0.5) * z * z / beta : z - scalar_t(0.5) * beta;
    });
  });
}


void mse_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "mse_cuda", [&]() {
    AT_SKIP_BFLOAT16_IF_NOT_ROCM(scalar_t, "mse_cuda", [&] {
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        auto diff = a - b;
        return diff * diff;
      });
    });
  });
}

void logaddexp_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "logaddexp_cuda", [&]() {
    gpu_kernel(iter, [] GPU_LAMBDA (scalar_t a, scalar_t b) -> scalar_t {
      if (::isinf(a) && a == b) {
        return a;
      }
      else {
        scalar_t m = ::max(a, b);
        return m + ::log((scalar_t)(1.0) + ::exp(-::abs(a - b)));
      }
    });
  });
}

void logaddexp2_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "logaddexp2_cuda", [&]() {
    gpu_kernel(iter, [] GPU_LAMBDA (scalar_t a, scalar_t b) -> scalar_t {
      if (::isinf(a) && a == b) {
        return a;
      }
      else {
        scalar_t m = ::max(a, b);
        return m + ::log2((scalar_t)(1.0) + ::pow((scalar_t)(2.0), -::abs(a - b)));
      }
    });
  });
}

void gcd_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "gcd_cuda", [&]() {
    gpu_kernel(iter, [] GPU_LAMBDA (scalar_t a, scalar_t b) -> scalar_t {
      return calc_gcd(a, b);
    });
  });
}

void lcm_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "lcm_cuda", [&]() {
    gpu_kernel(iter, [] GPU_LAMBDA (scalar_t a, scalar_t b) -> scalar_t {
      scalar_t g = calc_gcd(a, b);
      return (g == 0) ? 0 : ::abs(a / g * b);
    });
  });
}

void hypot_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.common_dtype(), "hypot_cuda", [&]() {
    gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return ::hypot(a, b);
    });
  });
}

void nextafter_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "nextafter_cuda", [&]() {
    gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return ::nextafter(a, b);
    });
  });
}

void heaviside_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(kHalf, kBool, kBFloat16, iter.dtype(), "heaviside_cuda", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return a == 0 ? b : static_cast<scalar_t>(a > 0);
    });
  });
}

REGISTER_DISPATCH(atan2_stub, &atan2_kernel_cuda);
REGISTER_DISPATCH(smooth_l1_stub, &smooth_l1_kernel_cuda);
REGISTER_DISPATCH(mse_stub, &mse_kernel_cuda);
REGISTER_DISPATCH(logaddexp_stub, &logaddexp_kernel_cuda);
REGISTER_DISPATCH(logaddexp2_stub, &logaddexp2_kernel_cuda);
REGISTER_DISPATCH(gcd_stub, &gcd_kernel_cuda);
REGISTER_DISPATCH(lcm_stub, &lcm_kernel_cuda);
REGISTER_DISPATCH(hypot_stub, &hypot_kernel_cuda);
REGISTER_DISPATCH(nextafter_stub, &nextafter_kernel_cuda);
REGISTER_DISPATCH(heaviside_stub, &heaviside_kernel_cuda);

}} // namespace at::native
