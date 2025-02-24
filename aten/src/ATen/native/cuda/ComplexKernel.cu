#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>

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
  AT_DISPATCH_FLOATING_TYPES(iter.input_dtype(0), "polar_cuda", [&]() {
    gpu_kernel(
      iter, [] GPU_LAMBDA(scalar_t a, scalar_t b) -> c10::complex<scalar_t> {
        return c10::complex<scalar_t>(a * std::cos(b), a * std::sin(b));
      });
  });
}

} // anonymous namespace

REGISTER_DISPATCH(complex_stub, &complex_kernel_cuda)
REGISTER_DISPATCH(polar_stub, &polar_kernel_cuda)

} // namespace at::native
