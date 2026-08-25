#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <c10/cuda/CUDAMathCompat.h>

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at::native {
namespace {

void complex_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND(kHalf, iter.input_dtype(0), "complex_cuda", [&]() {
    gpu_kernel(
      iter, [] GPU_LAMBDA(scalar_t a, scalar_t b) -> c10::complex<scalar_t> {
        return c10::complex<scalar_t>(a, b);
      });
  });
}

void polar_kernel_cuda(TensorIterator& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND(
      kHalf, iter.input_dtype(0), "polar_cuda", [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        gpu_kernel(
            iter,
            [] GPU_LAMBDA(scalar_t a, scalar_t b) -> c10::complex<scalar_t> {
              opmath_t sin_b;
              opmath_t cos_b;
              c10::cuda::compat::sincos(opmath_t(b), &sin_b, &cos_b);
              return c10::complex<scalar_t>(
                  static_cast<scalar_t>(opmath_t(a) * cos_b),
                  static_cast<scalar_t>(opmath_t(a) * sin_b));
            });
      });
}

} // anonymous namespace

REGISTER_DISPATCH(complex_stub, &complex_kernel_cuda)
REGISTER_DISPATCH(polar_stub, &polar_kernel_cuda)

} // namespace at::native
