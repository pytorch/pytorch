#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/zmath.cuh>
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

void smooth_l1_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(iter.dtype(), "smooth_l1_cuda", [&]() {
    gpu_kernel(iter, [] GPU_LAMBDA (scalar_t a, scalar_t b) -> scalar_t {
      auto z = ::abs(a - b);
      return z < scalar_t(1.) ? scalar_t(0.5) * z * z : z - scalar_t(0.5);
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

void fmod_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND(kHalf, iter.dtype(), "fmod_cuda", [&]() {
    gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return ::fmod(a, b);
    });
  });
}

void fmod_scalar_kernel_cuda(TensorIterator& iter, Scalar divisor) {
  AT_DISPATCH_ALL_TYPES_AND(kHalf, iter.dtype(), "fmod_scalar_cuda", [&]() {
    using thrust_t = typename ztype_cuda<scalar_t>::thrust_t;
    auto div = thrust_t(divisor.to<scalar_t>());
    gpu_kernel(iter, [div]GPU_LAMBDA(thrust_t a) -> thrust_t {
      return ::fmod(a, div);
    });
  });
}

REGISTER_DISPATCH(atan2_stub, &atan2_kernel_cuda);
REGISTER_DISPATCH(smooth_l1_stub, &smooth_l1_kernel_cuda);
REGISTER_DISPATCH(mse_stub, &mse_kernel_cuda);
REGISTER_DISPATCH(fmod_stub, &fmod_kernel_cuda);
REGISTER_DISPATCH(fmod_scalar_stub, &fmod_scalar_kernel_cuda)

}} // namespace at::native
