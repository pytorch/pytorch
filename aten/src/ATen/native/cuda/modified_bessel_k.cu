#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/Dispatch.h>
#include <ATen/native/cuda/JitLoops.cuh>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/Math.h>
#include <ATen/native/cuda/Math.cuh>
#include <ATen/native/cuda/jit_utils.h>

namespace at::native {
namespace {

constexpr char modified_bessel_k_name[] = "modified_bessel_k_forward";

void modified_bessel_k_kernel_cuda(TensorIteratorBase& iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(
      iterator.common_dtype(), "modified_bessel_k_cuda", [&]() {
        opmath_jitted_gpu_kernel_with_scalars<
            modified_bessel_k_name,
            scalar_t,
            scalar_t>(iterator, modified_bessel_k_string);
      });
#else
  AT_DISPATCH_FLOATING_TYPES(
      iterator.common_dtype(), "modified_bessel_k_cuda", [&]() {
        gpu_kernel_with_scalars(
            iterator, [] GPU_LAMBDA(scalar_t x, scalar_t nu) -> scalar_t {
              return modified_bessel_k_forward<scalar_t, true>(x, nu);
            });
      });
#endif
} // modified_bessel_k_kernel_cuda

} // namespace

REGISTER_DISPATCH(modified_bessel_k_stub, &modified_bessel_k_kernel_cuda)
} // namespace at::native
