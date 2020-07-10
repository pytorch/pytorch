#include <ATen/Dispatch.h>
#include <ATen/native/ComplexHelper.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at { namespace native {
namespace {

void complex_kernel_cuda(TensorIterator& iter) {
  // Necessary since comparison_op doesn't actually promote types for CUDA
  // tensors
  AT_DISPATCH_FLOATING_TYPES(
      promote_types(iter.input_dtype(0), iter.input_dtype(1)),
      "complex_cuda",
      [&]() {
        gpu_kernel_with_scalars(
            iter,
            [] GPU_LAMBDA(scalar_t a, scalar_t b) -> c10::complex<scalar_t> {
              return c10::complex<scalar_t>(a, b);
            });
      });
}

void complex_polar_kernel_cuda(TensorIterator& iter) {
  // Necessary since comparison_op doesn't actually promote types for CUDA
  // tensors
  AT_DISPATCH_FLOATING_TYPES(
      promote_types(iter.input_dtype(0), iter.input_dtype(1)),
      "complex_polar_cuda",
      [&]() {
        gpu_kernel_with_scalars(
            iter,
            [] GPU_LAMBDA(scalar_t a, scalar_t b) -> c10::complex<scalar_t> {
              return c10::complex<scalar_t>(a * std::cos(b), a * std::sin(b));
            });
      });
}

} // anonymous namespace

REGISTER_DISPATCH(complex_stub, &complex_kernel_cuda);
REGISTER_DISPATCH(complex_polar_stub, &complex_polar_kernel_cuda);

}} // namespace at::native
