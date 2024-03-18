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
            CONSTEXPR_EXCEPT_WIN_CUDA char hermite_polynomial_he_name[] = "hermite_polynomial_he_forward";

            void hermite_polynomial_he_kernel_cuda(TensorIteratorBase& iterator) {
#if AT_USE_JITERATOR()
                AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "hermite_polynomial_he_cuda", [&]() {
                    opmath_jitted_gpu_kernel_with_scalars<hermite_polynomial_he_name, scalar_t, scalar_t>(iterator, hermite_polynomial_he_string);
                });
#else
                AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "hermite_polynomial_he_cuda", [&]() {
                    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t x, scalar_t n) -> scalar_t {
                        return hermite_polynomial_he_forward<scalar_t, true>(x, n);
                    });
                });
#endif
            } // hermite_polynomial_he_kernel_cuda
        } // namespace (anonymous)

        REGISTER_DISPATCH(hermite_polynomial_he_stub, &hermite_polynomial_he_kernel_cuda);
} // namespace at::native
