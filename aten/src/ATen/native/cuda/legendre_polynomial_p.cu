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
            const char legendre_polynomial_p_name[] = "legendre_polynomial_p_forward";

            void legendre_polynomial_p_kernel_cuda(TensorIteratorBase& iterator) {
#if AT_USE_JITERATOR()
                AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "legendre_polynomial_p_cuda", [&]() {
                    opmath_jitted_gpu_kernel_with_scalars<legendre_polynomial_p_name, scalar_t, scalar_t>(iterator, legendre_polynomial_p_string);
                });
#else
                AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "legendre_polynomial_p_cuda", [&]() {
                    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t x, scalar_t n) -> scalar_t {
                        return legendre_polynomial_p_forward<scalar_t, true>(x, n);
                    });
                });
#endif
            } // legendre_polynomial_p_kernel_cuda
        } // namespace (anonymous)

        REGISTER_DISPATCH(legendre_polynomial_p_stub, &legendre_polynomial_p_kernel_cuda)
} // namespace at::native
